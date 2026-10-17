"""Tests for rejudge.py -- re-scoring finished duels with a new judge."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rejudge


SCENARIO = {
    "topic": "A test topic.",
    "models": [{"name": "Alice", "model": "m1"},
               {"name": "Bob", "model": "m2"}],
    "judge": {"model": "orig:8b",
              "prompt": "Judge {a} vs {b} on {topic}. "
                        "End with WINNER: <name>"},
}

LOG_TEXT = """\
Logging to output/logs/test.log
========================================================================
[m1 as Alice]  (turn 1)
========================================================================
--- REPLY ---
Alice argues first.

  (waiting for reply...)
========================================================================
[m2 as Bob]  (turn 2)
========================================================================
--- REPLY ---
Bob replies.
CPU at 81.0C (>= 80C); pausing duel to cool down to 70C.
CPU cooled to 65.0C; resuming duel after 60s.
--- CODE RUN ---
[Automatic code run] Bob's program ran successfully (exit code 0, 0.1s).
It printed: 42

--- JUDGE: orig:8b @ http://localhost:11434 is scoring the duel ---
=== Verdict ===
WINNER: Bob
"""


def fake_chat_factory(reply, record):
    def fake_chat(host, model, messages, think, options, timeout=None):
        record.update(host=host, model=model, messages=messages,
                      think=think, options=options, timeout=timeout)
        return None, reply, "stop", {}
    return fake_chat


class ParseLogTests(unittest.TestCase):
    def setUp(self):
        _topic, self.participants, _judge = self._scenario_parts()

    def _scenario_parts(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as fh:
            json.dump(SCENARIO, fh)
            path = fh.name
        self.addCleanup(os.unlink, path)
        return rejudge.load_scenario(path)

    def test_parses_turns_and_stops_before_verdict(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".log", delete=False) as fh:
            fh.write(LOG_TEXT)
            path = fh.name
        self.addCleanup(os.unlink, path)
        transcript, _reports = rejudge.parse_log(path, self.participants)
        self.assertEqual(transcript, [(0, "Alice argues first."),
                                      (1, "Bob replies.")])

    def test_recovers_code_run_report_and_drops_noise(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".log", delete=False) as fh:
            fh.write(LOG_TEXT)
            path = fh.name
        self.addCleanup(os.unlink, path)
        _transcript, reports = rejudge.parse_log(path, self.participants)
        self.assertIsNotNone(reports)
        reports = reports or {}
        self.assertIn(1, reports)
        self.assertIn("printed: 42", reports[1])
        self.assertNotIn("CPU", reports[1])

    def test_unknown_speaker_rejected(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".log", delete=False) as fh:
            fh.write(LOG_TEXT.replace("as Bob", "as Carol"))
            path = fh.name
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError):
            rejudge.parse_log(path, self.participants)


class LoadTranscriptJsonTests(unittest.TestCase):
    def test_maps_speakers_to_indexes(self):
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        entries = [{"speaker": "Bob", "model": "m2", "text": "hi"},
                   {"speaker": "Alice", "model": "m1", "text": "yo"}]
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as fh:
            json.dump(entries, fh)
            path = fh.name
        self.addCleanup(os.unlink, path)
        transcript, reports = rejudge.load_transcript_json(
            path, participants)
        self.assertEqual(transcript, [(1, "hi"), (0, "yo")])
        self.assertIsNone(reports)


class RejudgeOneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.scenario_path = os.path.join(self.tmp.name, "scenario.json")
        with open(self.scenario_path, "w", encoding="utf-8") as fh:
            json.dump(SCENARIO, fh)
        self.transcript_path = os.path.join(self.tmp.name, "duel.log")
        with open(self.transcript_path, "w", encoding="utf-8") as fh:
            fh.write(LOG_TEXT)

    def _run(self, **kwargs):
        record = {}
        chat = fake_chat_factory("Reasoning.\nWINNER: Alice", record)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rejudge.rejudge_one(
                self.scenario_path, self.transcript_path, "big:14b",
                chat=chat, **kwargs)
        return status, record, out.getvalue()

    def test_writes_sibling_verdict_with_overridden_judge(self):
        status, record, _out = self._run()
        self.assertEqual(status, "OK")
        self.assertEqual(record["model"], "big:14b")
        user = record["messages"][1]["content"]
        self.assertIn("Judge Alice vs Bob on A test topic.", user)
        self.assertIn("Alice argues first.", user)
        self.assertIn("printed: 42", user)  # run report reached the judge
        out_path = rejudge.rejudge_output_path(self.transcript_path,
                                               "big:14b")
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("original judge: orig:8b", content)
        self.assertIn("Winner: Alice", content)
        self.assertIn("WINNER: Alice", content)

    def test_original_transcript_untouched(self):
        self._run()
        with open(self.transcript_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), LOG_TEXT)

    def test_dry_run_calls_nothing_and_writes_nothing(self):
        record = {}
        chat = fake_chat_factory("WINNER: Alice", record)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rejudge.rejudge_one(
                self.scenario_path, self.transcript_path, "big:14b",
                dry_run=True, chat=chat)
        self.assertEqual(status, "OK")
        self.assertEqual(record, {})
        self.assertFalse(os.path.exists(
            rejudge.rejudge_output_path(self.transcript_path, "big:14b")))
        self.assertIn("would judge with big:14b", out.getvalue())

    def test_oversized_transcript_skips_without_calling(self):
        big = [(0, "x" * 300_000)]
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        with open(self.transcript_path, "w", encoding="utf-8") as fh:
            json.dump([{"speaker": "Alice", "model": "m1",
                        "text": big[0][1]}], fh)
        os.rename(self.transcript_path, self.transcript_path + ".json")
        record = {}
        chat = fake_chat_factory("WINNER: Alice", record)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rejudge.rejudge_one(
                self.scenario_path, self.transcript_path + ".json",
                "big:14b", chat=chat)
        self.assertEqual(status, "SKIPPED")
        self.assertEqual(record, {})
        self.assertEqual(participants[0]["name"], "Alice")

    def test_scenario_without_judge_reports_no_judge(self):
        bare = dict(SCENARIO)
        del bare["judge"]
        with open(self.scenario_path, "w", encoding="utf-8") as fh:
            json.dump(bare, fh)
        status, _record, out = self._run()
        self.assertEqual(status, "NO JUDGE")
        self.assertIn("no judge configured", out)


class ScenarioForLogTests(unittest.TestCase):
    def test_timestamped_and_plain_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenario = os.path.join(tmp, "colony_caretakers.json")
            with open(scenario, "w", encoding="utf-8") as fh:
                fh.write("{}")
            self.assertEqual(
                rejudge.scenario_for_log(
                    "output/logs/20261017-093019-colony_caretakers.log",
                    tmp),
                scenario)
            self.assertEqual(
                rejudge.scenario_for_log("colony_caretakers.log", tmp),
                scenario)
            self.assertIsNone(
                rejudge.scenario_for_log("20261017-093019-nope.log", tmp))


if __name__ == "__main__":
    unittest.main()
