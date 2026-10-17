"""Tests for summarize_results.py -- the run_results.log rollup."""

import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import summarize_results as sr

BAR = "-" * 72

FULL_BLOCK = f"""{BAR}
Run: 2026-10-15 03:00:23
Config: scenarios/code_golf_vs_maintainer.json
Log: /home/paul/dual/output/logs/20261015-030023-code_golf_vs_maintainer.log
Models: qwen3:1.7b (Maintainer) vs smollm2:1.7b (Golfer)
Turns: 10/10 (10 replies) in 6363s
Code runs: 1 ok, 8 failed, 0 timed out, 1 replies without a Python code block
Verdict:
The Maintainer wrote working code.
WINNER: Maintainer
Result: OK
=== Model performance ===
Model         Turns  Gen tok/s  Prompt tok/s  Tokens  Truncated
qwen3:1.7b        5        1.8           5.3    1875          0
smollm2:1.7b      5        1.2           4.8    1376          0
{BAR}
"""

PLAIN_BLOCK = f"""{BAR}
Run: 2026-10-14 16:24:38
Config: scenarios/ai_consciousness_debate.json
Models: qwen3:1.7b (Materialist) vs smollm2:1.7b (Skeptic)
Turns: 8/8 (8 replies) in 2118s
Result: OK
=== Model performance ===
Model         Turns  Gen tok/s  Prompt tok/s  Tokens  Truncated
qwen3:1.7b        4        2.1           6.0     540          0
smollm2:1.7b      4        1.5           5.6     544          0
{BAR}
"""

STOPPED_BLOCK = f"""{BAR}
Run: 2026-10-14 17:00:00
Config: scenarios/broken.json
Models: qwen3:1.7b (A) vs smollm2:1.7b (B)
Turns: 3/8 (3 replies) in 100s
Result: STOPPED EARLY
Ollama at http://x timed out
{BAR}
"""


class ParseBlockTests(unittest.TestCase):
    def test_full_block(self):
        d = sr.parse_block(FULL_BLOCK)
        self.assertEqual(d["scenario"], "code_golf_vs_maintainer")
        self.assertEqual(d["stamp"], "2026-10-15 03:00:23")
        self.assertEqual(d["models"],
                         [("qwen3:1.7b", "Maintainer"),
                          ("smollm2:1.7b", "Golfer")])
        self.assertEqual((d["turns_done"], d["turns_planned"]), (10, 10))
        self.assertEqual(d["code"]["ok"], 1)
        self.assertEqual(d["code"]["failed"], 8)
        self.assertEqual(d["winner"], "Maintainer")
        self.assertEqual(d["result"], "OK")
        self.assertAlmostEqual(d["perf"]["qwen3:1.7b"]["gen_tps"], 1.8)
        self.assertEqual(d["perf"]["smollm2:1.7b"]["tokens"], 1376)

    def test_block_without_verdict_or_code(self):
        d = sr.parse_block(PLAIN_BLOCK)
        self.assertEqual(d["scenario"], "ai_consciousness_debate")
        self.assertIsNone(d["code"])
        self.assertIsNone(d["winner"])
        self.assertEqual(d["result"], "OK")

    def test_stopped_early_has_no_perf_table(self):
        d = sr.parse_block(STOPPED_BLOCK)
        self.assertEqual(d["result"], "STOPPED EARLY")
        self.assertEqual(d["turns_done"], 3)
        self.assertNotIn("perf", d)

    def test_markdown_bold_winner_parses(self):
        block = FULL_BLOCK.replace("WINNER: Maintainer",
                                   "**WINNER:** Maintainer")
        self.assertEqual(sr.parse_block(block)["winner"], "Maintainer")

    def test_judge_line_parsed(self):
        block = FULL_BLOCK.replace(
            "Verdict:",
            "Judge: qwen3:8b @ http://localhost:11434, context 16384, "
            "est. budget 15636\nVerdict:")
        d = sr.parse_block(block)
        self.assertEqual(d["judge"],
                         "qwen3:8b @ http://localhost:11434, context 16384, "
                         "est. budget 15636")
        self.assertEqual(d["winner"], "Maintainer")

    def test_judge_skipped_line_parsed(self):
        block = PLAIN_BLOCK.replace(
            "Result: OK",
            "Judge: SKIPPED (j @ http://x): too big\nResult: OK")
        self.assertTrue(sr.parse_block(block)["judge"].startswith("SKIPPED"))

    def test_aggregates_count_unscored_judges(self):
        skipped = sr.parse_block(PLAIN_BLOCK.replace(
            "Result: OK", "Judge: SKIPPED (j @ http://x): big\nResult: OK"))
        failed = sr.parse_block(PLAIN_BLOCK.replace(
            "Result: OK", "Judge: FAILED (j @ http://x): down\nResult: OK"))
        out = io.StringIO()
        with mock.patch.object(sys, "stdout", out):
            sr.print_aggregates([skipped, failed])
        self.assertIn("Judge not scored: 1 skipped, 1 failed",
                      out.getvalue())

    def test_non_duel_block_returns_none(self):
        self.assertIsNone(sr.parse_block("\nSome random text\n"))


class EndToEndTests(unittest.TestCase):
    def _write(self, d, content):
        path = os.path.join(d, "run_results.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_table_and_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, FULL_BLOCK + PLAIN_BLOCK + STOPPED_BLOCK)
            out = io.StringIO()
            with mock.patch.object(sys, "stdout", out):
                self.assertEqual(sr.main(["summarize_results.py", path]), 0)
        text = out.getvalue()
        # one table row per duel
        self.assertIn("code_golf_vs_maintainer", text)
        self.assertIn("ai_consciousness_debate", text)
        self.assertIn("broken", text)
        self.assertIn("1/9 ok", text)  # code pass tally
        self.assertIn("Maintainer", text)  # winner
        # aggregates
        self.assertIn("Duels: 3 (2 OK, 1 stopped/crashed)", text)
        self.assertIn("Avg gen tok/s: qwen3:1.7b", text)
        self.assertIn("Code runs: 1 ok, 8 failed/timed out (11% pass)", text)
        self.assertIn("Judge wins: Maintainer 1", text)

    def test_missing_file_exits_1(self):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            self.assertEqual(sr.main(["summarize_results.py", "/nope/x.log"]),
                             1)

    def test_no_duels_found_exits_1(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "nothing here\n")
            err = io.StringIO()
            with mock.patch.object(sys, "stderr", err):
                self.assertEqual(sr.main(["summarize_results.py", path]), 1)


if __name__ == "__main__":
    unittest.main()
