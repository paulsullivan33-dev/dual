import argparse
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_chat


class BuildOptionsTests(unittest.TestCase):
    def test_only_max_tokens_by_default(self):
        args = argparse.Namespace(max_tokens=300, temperature=None, num_ctx=None)
        self.assertEqual(ollama_chat._build_options(args), {"num_predict": 300})

    def test_includes_temperature_and_num_ctx_when_set(self):
        args = argparse.Namespace(max_tokens=300, temperature=0.8, num_ctx=4096)
        self.assertEqual(
            ollama_chat._build_options(args),
            {"num_predict": 300, "temperature": 0.8, "num_ctx": 4096},
        )


class DuelTurnsValidationTests(unittest.TestCase):
    def _run_with_argv(self, argv):
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                ollama_chat.main()
        return ctx.exception

    def test_zero_turns_exits_before_any_request(self):
        exc = self._run_with_argv(
            ["ollama_chat.py", "duel", "--topic", "x", "--turns", "0"])
        self.assertIn("--turns must be >= 1", str(exc.code))

    def test_negative_turns_exits(self):
        exc = self._run_with_argv(
            ["ollama_chat.py", "duel", "--topic", "x", "--turns", "-3"])
        self.assertIn("--turns must be >= 1", str(exc.code))


class DuelSharesTurnLoopTests(unittest.TestCase):
    """`ollama_chat.py duel` must run the same turn loop as ollama_duel.py.
    Regression test: the two had drifted (no reply nudge, no truncation
    warning, no stats table in the chat version)."""

    def _run_duel(self, done_reason="stop"):
        import io
        import ollama_duel
        seen = []

        def fake_call_chat(host, model, messages, think, options, timeout=None):
            seen.append([m["content"] for m in messages])
            metrics = {"gen_tokens": 10, "gen_s": 1.0,
                       "prompt_tokens": 20, "prompt_s": 0.5, "gen_tps": 10.0}
            return "", "canned reply", done_reason, metrics

        argv = ["ollama_chat.py", "duel", "--topic", "the topic", "--turns", "2",
                "--name-a", "Ann", "--name-b", "Bob"]
        out = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(ollama_duel, "call_chat", fake_call_chat), \
             mock.patch.object(sys, "stdout", out):
            ollama_chat.main()
        return seen, out.getvalue()

    def test_topic_every_turn_and_nudge_after_first(self):
        seen, _ = self._run_duel()
        self.assertEqual([turn[0] for turn in seen], ["the topic", "the topic"])
        self.assertIn("Reply directly to Ann's last message", seen[1][-1])

    def test_prints_stats_and_truncation_warning(self):
        _, out = self._run_duel(done_reason="length")
        self.assertIn("=== Model performance ===", out)
        self.assertIn("hit the max_tokens ceiling", out)


if __name__ == "__main__":
    unittest.main()
