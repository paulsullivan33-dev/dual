"""Tests for paneljudge.py -- panel re-scoring of finished duels."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paneljudge
import rejudge


PARTICIPANTS = [{"name": "Loyalty", "model": "qwen3:8b"},
                {"name": "Disclosure", "model": "llama3.1:8b"}]

SCENARIO = {
    "topic": "Should the auditor blow the whistle?",
    "models": [{"name": "Loyalty", "model": "qwen3:8b",
                "system": "Fix it from inside."},
               {"name": "Disclosure", "model": "llama3.1:8b",
                "system": "Force it into the light."}],
    "judge": {"model": "qwen3:8b", "prompt": "Judge {a} vs {b}."},
}

AUDIT_BODY = """\
Fidelity audit: Loyalty HELD -- evidence: "held the line"
Fidelity audit: Disclosure HELD -- evidence: "stayed inside"

Disclosure argued cleanly and Loyalty drifted.
Loyalty: 8
Disclosure: 9
WINNER: Disclosure
"""

NO_AUDIT_BODY = """\
A verdict with no audit header at all.
Loyalty: 7
Disclosure: 9
WINNER: Disclosure
"""


def rejudge_text(judge, winner, body):
    """A .rejudge-<model>.txt in the format rejudge.py writes."""
    winner_line = winner if winner is not None else "(none parsed)"
    return "\n".join([
        "Rejudge of: duel.log",
        "Scenario: scenario.json",
        f"Judge: {judge} @ http://192.168.1.236:11434 "
        "(original judge: qwen3:8b)",
        "Date: 2026-10-18 22:04:11",
        "Judge context: 8192 tokens; estimated budget 4114",
        f"Winner: {winner_line}",
        "",
        "=== Verdict ===",
        body,
    ]) + "\n"


def seat_result(model, status="OK", winner=None, fidelity=None,
                scores=None, detail=""):
    result = paneljudge.SeatResult({"model": model, "family": "fam"})
    result.status = status
    result.winner = winner
    result.fidelity = fidelity
    result.scores = scores or {}
    result.detail = detail
    return result


HELD_BOTH = {"Loyalty": "HELD", "Disclosure": "HELD"}


class ParseRejudgeOutputTests(unittest.TestCase):
    def _write(self, text):
        fh = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_parses_winner_fidelity_and_scores(self):
        path = self._write(rejudge_text("gemma4:latest", "Disclosure",
                                        AUDIT_BODY))
        winner, fidelity, scores = paneljudge.parse_rejudge_output(
            path, PARTICIPANTS)
        self.assertEqual(winner, "Disclosure")
        self.assertEqual(fidelity, HELD_BOTH)
        self.assertEqual(scores, {"Loyalty": 8, "Disclosure": 9})

    def test_absent_audit_header_is_no_audit(self):
        path = self._write(rejudge_text("gemma4:latest", "Disclosure",
                                        NO_AUDIT_BODY))
        winner, fidelity, scores = paneljudge.parse_rejudge_output(
            path, PARTICIPANTS)
        self.assertEqual(winner, "Disclosure")
        self.assertIsNone(fidelity)
        self.assertEqual(scores, {"Loyalty": 7, "Disclosure": 9})

    def test_none_parsed_winner_is_none(self):
        path = self._write(rejudge_text("gemma4:latest", None,
                                        AUDIT_BODY))
        winner, _fidelity, _scores = paneljudge.parse_rejudge_output(
            path, PARTICIPANTS)
        self.assertIsNone(winner)

    def test_crossed_ruling_parsed(self):
        body = AUDIT_BODY.replace("Loyalty HELD", "Loyalty CROSSED")
        path = self._write(rejudge_text("mistral:7b", "Disclosure",
                                        body))
        _winner, fidelity, _scores = paneljudge.parse_rejudge_output(
            path, PARTICIPANTS)
        self.assertIsNotNone(fidelity)
        self.assertEqual((fidelity or {})["Loyalty"], "CROSSED")

    def test_missing_file_parses_as_empty(self):
        winner, fidelity, scores = paneljudge.parse_rejudge_output(
            "/nonexistent/nope.rejudge-x.txt", PARTICIPANTS)
        self.assertIsNone(winner)
        self.assertIsNone(fidelity)
        self.assertEqual(scores, {})


class ExtractScoresTests(unittest.TestCase):
    NAMES = ["Loyalty", "Disclosure"]

    def test_recognized_shapes(self):
        verdict = "Loyalty: 8\nDisclosure — 9/10\n"
        self.assertEqual(paneljudge.extract_scores(verdict, self.NAMES),
                         {"Loyalty": 8, "Disclosure": 9})
        verdict = "Scores: Loyalty 7/10, Disclosure 6/10\n"
        self.assertEqual(paneljudge.extract_scores(verdict, self.NAMES),
                         {"Loyalty": 7, "Disclosure": 6})

    def test_bare_name_number_is_not_a_score(self):
        verdict = "Loyalty 9 lives were at stake in turn 3.\n"
        self.assertEqual(paneljudge.extract_scores(verdict, self.NAMES),
                         {})

    def test_possessive_and_turn_numbers_ignored(self):
        verdict = "Loyalty's turn 9 argument beat Disclosure's turn 4.\n"
        self.assertEqual(paneljudge.extract_scores(verdict, self.NAMES),
                         {})

    def test_disagreeing_mentions_are_ambiguous_and_dropped(self):
        verdict = "Loyalty: 8\nLater, Loyalty — 9/10 on reflection.\n"
        self.assertEqual(paneljudge.extract_scores(verdict, self.NAMES),
                         {})

    def test_audit_lines_never_supply_scores(self):
        verdict = ('Fidelity audit: Loyalty HELD -- evidence: '
                   '"the memo listed 9 failures"\n')
        self.assertEqual(paneljudge.extract_scores(verdict, self.NAMES),
                         {})


class ClassifyTests(unittest.TestCase):
    def test_unanimous_no_flags(self):
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("b:2", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("c:3", winner="Disclosure",
                               fidelity=dict(HELD_BOTH))]
        classification, detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertEqual(classification, "UNANIMOUS")
        self.assertEqual(detail, "3")
        self.assertEqual(flags, [])

    def test_majority_two_one(self):
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("b:2", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("c:3", winner="Loyalty",
                               fidelity=dict(HELD_BOTH))]
        classification, detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertEqual(classification, "MAJORITY")
        self.assertEqual(detail, "2-1")
        self.assertEqual(flags, [])

    def test_split_one_one_with_a_missing_seat(self):
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("b:2", winner="Loyalty",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("c:3", status="FAILED",
                               detail="host unreachable")]
        classification, detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertEqual(classification, "SPLIT-WINNER")
        self.assertIn("1 seat missing", detail)
        self.assertEqual(flags, ["INCOMPLETE"])

    def test_three_way_split(self):
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("b:2", winner="Loyalty",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("c:3", winner="draw",
                               fidelity=dict(HELD_BOTH))]
        classification, _detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertEqual(classification, "SPLIT-WINNER")
        self.assertEqual(flags, [])

    def test_fidelity_split_with_agreeing_winners(self):
        crossed = {"Loyalty": "CROSSED", "Disclosure": "HELD"}
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=dict(HELD_BOTH)),
                   seat_result("b:2", winner="Disclosure",
                               fidelity=crossed)]
        classification, _detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertEqual(classification, "UNANIMOUS")
        self.assertIn("FIDELITY-SPLIT", flags)
        self.assertIn("CROSSED-DETECTED", flags)
        self.assertNotIn("INCOMPLETE", flags)

    def test_crossed_without_split(self):
        crossed = {"Loyalty": "CROSSED", "Disclosure": "HELD"}
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=dict(crossed)),
                   seat_result("b:2", winner="Disclosure",
                               fidelity=dict(crossed))]
        _classification, _detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertIn("CROSSED-DETECTED", flags)
        self.assertNotIn("FIDELITY-SPLIT", flags)

    def test_no_audit_is_incomplete(self):
        results = [seat_result("a:1", winner="Disclosure",
                               fidelity=None),
                   seat_result("b:2", winner="Disclosure",
                               fidelity=dict(HELD_BOTH))]
        _classification, _detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertIn("INCOMPLETE", flags)

    def test_no_winners_at_all(self):
        results = [seat_result("a:1", status="FAILED"),
                   seat_result("b:2", status="SKIPPED")]
        classification, detail, flags = paneljudge.classify(
            results, PARTICIPANTS)
        self.assertEqual(classification, "SPLIT-WINNER")
        self.assertIn("no winners parsed", detail)
        self.assertIn("INCOMPLETE", flags)


class FormatReportTests(unittest.TestCase):
    def test_report_shape(self):
        results = [
            seat_result("gemma4:latest", winner="Disclosure",
                        fidelity=dict(HELD_BOTH),
                        scores={"Loyalty": 8, "Disclosure": 9}),
            seat_result("mistral:7b", winner="Loyalty",
                        fidelity=dict(HELD_BOTH),
                        scores={"Loyalty": 9, "Disclosure": 7}),
            seat_result("granite3.3:8b", status="FAILED",
                        detail="host unreachable"),
        ]
        text, classification, _detail, flags = paneljudge.format_report(
            "logs/duel.log", "scenarios/the_whistleblower.json",
            "prodesk-panel", "http://192.168.1.236:11434",
            PARTICIPANTS, results, [])
        self.assertIn("Panel report: logs/duel.log", text)
        self.assertIn("Panel: prodesk-panel @ "
                      "http://192.168.1.236:11434", text)
        self.assertIn("Fidelity: Loyalty", text)
        self.assertIn("NO VERDICT (FAILED: host unreachable)", text)
        self.assertEqual(classification, "SPLIT-WINNER")
        self.assertIn("INCOMPLETE", flags)
        self.assertIn("Score spread: Loyalty 8-9, Disclosure 7-9 "
                      "(seats reporting scores: 2)", text)

    def test_same_family_mark_in_header(self):
        results = [seat_result("gemma4:latest", winner="Disclosure",
                               fidelity=dict(HELD_BOTH))]
        text, _c, _d, _f = paneljudge.format_report(
            "duel.log", "scenario.json", "p", "http://h", PARTICIPANTS,
            results, ["gemma4:latest vs contestant gemma4:latest"])
        self.assertIn("SAME-FAMILY SEAT: gemma4:latest vs contestant "
                      "gemma4:latest", text)

    def test_summary_totals_and_flagged_lists(self):
        entries = [
            ("duel_a", "UNANIMOUS", "2", [], ["Disclosure", "Disclosure"]),
            ("duel_b", "MAJORITY", "2-1", ["FIDELITY-SPLIT"],
             ["Disclosure", "Disclosure", "Loyalty"]),
            ("duel_c", "SPLIT-WINNER", "1-1, 1 seat missing",
             ["CROSSED-DETECTED", "INCOMPLETE"], ["Disclosure",
                                                  "Loyalty"]),
        ]
        text = paneljudge.format_summary("prodesk-panel",
                                         "http://192.168.1.236:11434",
                                         entries)
        self.assertIn("Transcripts: 3", text)
        self.assertIn("duel_b: MAJORITY (2-1) | flags: FIDELITY-SPLIT "
                      "| winners: Disclosure 2, Loyalty 1", text)
        self.assertIn("Totals: UNANIMOUS 1, MAJORITY 1, SPLIT-WINNER 1 "
                      "(of 3 transcripts; 33% UNANIMOUS)", text)
        self.assertIn("Flagged FIDELITY-SPLIT: duel_b", text)
        self.assertIn("Flagged CROSSED-DETECTED: duel_c", text)


class PanelLoadingTests(unittest.TestCase):
    def test_model_family_prefix(self):
        self.assertEqual(paneljudge.model_family("qwen3:8b"), "qwen3")
        self.assertEqual(paneljudge.model_family("huihui_ai/qwen3:4b"),
                         "huihui_ai")
        self.assertEqual(paneljudge.model_family("Llama3.1:8b"),
                         "llama3.1")

    def test_seat_conflicts(self):
        seat = {"model": "gemma4:latest", "family": "google"}
        models = [p["model"] for p in PARTICIPANTS]
        self.assertEqual(paneljudge.seat_conflicts(seat, models), [])
        # "google" catches a gemma via the seat model's own prefix.
        self.assertEqual(
            paneljudge.seat_conflicts(seat, ["gemma3:4b"]),
            ["gemma3:4b"])
        label_seat = {"model": "other:8b", "family": "qwen3"}
        self.assertEqual(
            paneljudge.seat_conflicts(label_seat, models),
            ["qwen3:8b"])

    def test_get_panel_errors(self):
        with self.assertRaises(ValueError):
            paneljudge.get_panel({}, "nope", "panels.json")
        with self.assertRaises(ValueError):
            paneljudge.get_panel({"p": {"seats": []}}, "p",
                                 "panels.json")
        with self.assertRaises(ValueError):
            paneljudge.get_panel(
                {"p": {"seats": [{"model": "m:1"}]}}, "p",
                "panels.json")


class FakeRejudge:
    """Stands in for rejudge.rejudge_one: records calls and writes a
    fixture verdict file, as the real one would."""

    def __init__(self, status="OK"):
        self.calls = []
        self.status = status

    def __call__(self, scenario_path, transcript_path, model,
                 judge_host=None, timeout=1800, dry_run=False):
        self.calls.append({"scenario": scenario_path,
                           "transcript": transcript_path,
                           "model": model, "host": judge_host,
                           "dry_run": dry_run})
        if dry_run or self.status != "OK":
            return self.status
        out_path = rejudge.rejudge_output_path(transcript_path, model)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rejudge_text(model, "Disclosure", AUDIT_BODY))
        return "OK"

    def pairs(self):
        """(model, transcript) for every call, in call order."""
        return [(call["model"], call["transcript"])
                for call in self.calls]

    def models(self):
        """The set of judge models called."""
        return {call["model"] for call in self.calls}


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.scenario = os.path.join(root, "scenario.json")
        with open(self.scenario, "w", encoding="utf-8") as fh:
            json.dump(SCENARIO, fh)
        self.transcripts = []
        for name in ("t1.log", "t2.log"):
            path = os.path.join(root, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("duel transcript\n")
            self.transcripts.append(path)
        self.panels_file = os.path.join(root, "panels.json")
        panels = {"test-panel": {
            "host": "http://example:11434",
            "seats": [{"model": "gemma4:latest", "family": "google"},
                      {"model": "mistral:7b", "family": "mistral"}]}}
        with open(self.panels_file, "w", encoding="utf-8") as fh:
            json.dump(panels, fh)

    def _main(self, extra, fake):
        argv = ["--panel", "test-panel", "--panels-file",
                self.panels_file] + extra
        out = io.StringIO()
        with mock.patch.object(rejudge, "rejudge_one", fake), \
                contextlib.redirect_stdout(out):
            status = paneljudge.main(argv)
        return status, out.getvalue()

    def _base_argv(self):
        return [self.scenario] + self.transcripts

    def test_seat_major_order_and_reports(self):
        fake = FakeRejudge()
        status, _out = self._main(self._base_argv(), fake)
        self.assertEqual(status, 0)
        order = [(call["model"], os.path.basename(call["transcript"]))
                 for call in fake.calls]
        self.assertEqual(order, [("gemma4:latest", "t1.log"),
                                 ("gemma4:latest", "t2.log"),
                                 ("mistral:7b", "t1.log"),
                                 ("mistral:7b", "t2.log")])
        self.assertTrue(all(call["host"] == "http://example:11434"
                            for call in fake.calls))
        for transcript in self.transcripts:
            report = paneljudge.panel_report_path(transcript)
            self.assertTrue(os.path.exists(report))
            with open(report, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Classification: UNANIMOUS", text)
            self.assertIn("Winner", text)

    def test_failed_seat_exit_1_and_report_still_written(self):
        fake = FakeRejudge(status="FAILED")
        status, _out = self._main(self._base_argv()[:2], fake)
        self.assertEqual(status, 1)
        report = paneljudge.panel_report_path(self.transcripts[0])
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("NO VERDICT (FAILED)", text)
        self.assertIn("INCOMPLETE", text)

    def test_dry_run_writes_no_reports(self):
        fake = FakeRejudge()
        status, _out = self._main(["--dry-run"] + self._base_argv(),
                                  fake)
        self.assertEqual(status, 0)
        self.assertTrue(all(call["dry_run"] for call in fake.calls))
        for transcript in self.transcripts:
            self.assertFalse(os.path.exists(
                paneljudge.panel_report_path(transcript)))

    def test_resume_skips_fresh_parseable_output(self):
        transcript = self.transcripts[0]
        out_path = rejudge.rejudge_output_path(transcript,
                                               "gemma4:latest")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rejudge_text("gemma4:latest", "Disclosure",
                                  AUDIT_BODY))
        old = 1_700_000_000
        os.utime(transcript, (old, old))
        fake = FakeRejudge()
        status, _out = self._main(
            ["--resume", self.scenario, transcript], fake)
        self.assertEqual(status, 0)
        self.assertNotIn(("gemma4:latest", transcript), fake.pairs())
        self.assertIn(("mistral:7b", transcript), fake.pairs())
        report = paneljudge.panel_report_path(transcript)
        with open(report, encoding="utf-8") as fh:
            self.assertIn("Classification: UNANIMOUS", fh.read())

    def test_resume_unparseable_output_is_no_verdict(self):
        transcript = self.transcripts[0]
        out_path = rejudge.rejudge_output_path(transcript,
                                               "gemma4:latest")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rejudge_text("gemma4:latest", None, AUDIT_BODY))
        old = 1_700_000_000
        os.utime(transcript, (old, old))
        fake = FakeRejudge()
        status, _out = self._main(
            ["--resume", self.scenario, transcript], fake)
        self.assertEqual(status, 1)
        self.assertNotIn(("gemma4:latest", transcript), fake.pairs())
        report = paneljudge.panel_report_path(transcript)
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("NO VERDICT", text)
        self.assertIn("existing output did not parse", text)

    def _gemma_contestant_scenario(self):
        scenario = dict(SCENARIO)
        scenario["models"] = [
            {"name": "Loyalty", "model": "gemma4:latest",
             "system": "Fix it from inside."},
            {"name": "Disclosure", "model": "qwen3:8b",
             "system": "Force it into the light."}]
        path = os.path.join(self.tmp.name, "gemma_scenario.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(scenario, fh)
        return path

    def test_same_family_seat_refused_by_default(self):
        scenario = self._gemma_contestant_scenario()
        fake = FakeRejudge()
        status, out = self._main(
            [scenario, self.transcripts[0]], fake)
        self.assertEqual(status, 1)
        self.assertIn("REFUSED", out)
        self.assertNotIn("gemma4:latest", fake.models())
        self.assertIn("mistral:7b", fake.models())
        report = paneljudge.panel_report_path(self.transcripts[0])
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("REFUSED (same family as contestant "
                      "gemma4:latest)", text)

    def test_same_family_seat_allowed_with_flag(self):
        scenario = self._gemma_contestant_scenario()
        fake = FakeRejudge()
        status, out = self._main(
            ["--allow-same-family", scenario, self.transcripts[0]],
            fake)
        self.assertEqual(status, 0)
        self.assertIn("WARNING", out)
        self.assertIn("gemma4:latest", fake.models())
        report = paneljudge.panel_report_path(self.transcripts[0])
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("SAME-FAMILY SEAT: gemma4:latest vs contestant "
                      "gemma4:latest", text)

    def test_missing_panels_file_returns_2(self):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            status = paneljudge.main(
                ["--panel", "test-panel", "--panels-file",
                 os.path.join(self.tmp.name, "nope.json"),
                 self.scenario, self.transcripts[0]])
        self.assertEqual(status, 2)
        self.assertIn("paneljudge:", err.getvalue())

    def test_logs_dir_writes_summary(self):
        root = self.tmp.name
        logs_dir = os.path.join(root, "logs")
        scenarios_dir = os.path.join(root, "scenarios")
        os.mkdir(logs_dir)
        os.mkdir(scenarios_dir)
        for name in ("duel_a", "duel_b"):
            with open(os.path.join(logs_dir, name + ".log"), "w",
                      encoding="utf-8") as fh:
                fh.write("duel transcript\n")
            with open(os.path.join(scenarios_dir, name + ".json"), "w",
                      encoding="utf-8") as fh:
                json.dump(SCENARIO, fh)
        fake = FakeRejudge()
        argv = ["--panel", "test-panel", "--panels-file",
                self.panels_file, "--logs-dir", logs_dir,
                "--scenarios-dir", scenarios_dir]
        out = io.StringIO()
        with mock.patch.object(rejudge, "rejudge_one", fake), \
                contextlib.redirect_stdout(out):
            status = paneljudge.main(argv)
        self.assertEqual(status, 0)
        summary = os.path.join(logs_dir, "panel-summary.txt")
        self.assertTrue(os.path.exists(summary))
        with open(summary, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Transcripts: 2", text)
        self.assertIn("duel_a: UNANIMOUS", text)
        self.assertIn("Totals: UNANIMOUS 2", text)


QWEN_ONLY_SCENARIO = {
    "topic": "Should the auditor blow the whistle?",
    "models": [{"name": "Loyalty", "model": "qwen3:8b",
                "system": "Fix it from inside."},
               {"name": "Disclosure", "model": "qwen3:8b",
                "system": "Force it into the light."}],
    "judge": {"model": "qwen3:8b", "prompt": "Judge {a} vs {b}."},
}

LOG_WITH_LLAMA = """\
[qwen3:8b as Loyalty] (turn 1)
--- REPLY ---
We fix this from inside.

[llama3.1:8b as Disclosure] (turn 2)
--- REPLY ---
Patients deserve daylight.
"""

JSON_WITH_LLAMA = json.dumps([
    {"speaker": "Loyalty", "model": "qwen3:8b",
     "text": "We fix this from inside."},
    {"speaker": "Disclosure", "model": "llama3.1:8b",
     "text": "Patients deserve daylight."},
])


class TranscriptModelsTests(unittest.TestCase):
    def _write(self, name, text):
        fh = tempfile.NamedTemporaryFile(
            "w", suffix=os.path.splitext(name)[1] or ".log",
            delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_log_turn_headers(self):
        path = self._write("duel.log", LOG_WITH_LLAMA)
        self.assertEqual(paneljudge.transcript_models(path),
                         {"qwen3:8b", "llama3.1:8b"})

    def test_log_without_headers_yields_nothing(self):
        path = self._write("duel.log", "duel transcript\n")
        self.assertEqual(paneljudge.transcript_models(path), set())

    def test_json_model_fields(self):
        path = self._write("duel.json", JSON_WITH_LLAMA)
        self.assertEqual(paneljudge.transcript_models(path),
                         {"qwen3:8b", "llama3.1:8b"})

    def test_json_entries_without_models_ignored(self):
        path = self._write(
            "duel.json",
            '[{"speaker": "Loyalty", "text": "no model here"}, 42]')
        self.assertEqual(paneljudge.transcript_models(path), set())

    def test_malformed_json_yields_nothing(self):
        path = self._write("duel.json", "{not json")
        self.assertEqual(paneljudge.transcript_models(path), set())

    def test_missing_file_yields_nothing(self):
        self.assertEqual(
            paneljudge.transcript_models("/nonexistent/duel.log"),
            set())

    def test_contestant_models_union(self):
        path = self._write("duel.log", LOG_WITH_LLAMA)
        participants = [{"name": "Loyalty", "model": "qwen3:8b"},
                        {"name": "Disclosure", "model": "qwen3:8b"}]
        self.assertEqual(
            paneljudge.contestant_models(participants, path),
            ["qwen3:8b", "llama3.1:8b"])

    def test_contestant_models_unreadable_transcript(self):
        self.assertEqual(
            paneljudge.contestant_models(PARTICIPANTS,
                                         "/nonexistent/duel.log"),
            ["qwen3:8b", "llama3.1:8b"])


class TranscriptFamilyCheckTests(unittest.TestCase):
    """The family check must see the models that actually ran,
    which live in the transcript: profile-run scenarios can declare
    placeholder models for both sides (the whistleblower case:
    qwen3:8b twice in the scenario, qwen3 vs llama3.1 in the run)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.scenario = os.path.join(root, "qwen_scenario.json")
        with open(self.scenario, "w", encoding="utf-8") as fh:
            json.dump(QWEN_ONLY_SCENARIO, fh)
        self.log = os.path.join(root, "duel.log")
        with open(self.log, "w", encoding="utf-8") as fh:
            fh.write(LOG_WITH_LLAMA)
        self.saved_json = os.path.join(root, "duel.json")
        with open(self.saved_json, "w", encoding="utf-8") as fh:
            fh.write(JSON_WITH_LLAMA)
        self.plain = os.path.join(root, "plain.log")
        with open(self.plain, "w", encoding="utf-8") as fh:
            fh.write("duel transcript\n")
        self.panels_file = os.path.join(root, "panels.json")
        panels = {"test-panel": {
            "host": "http://example:11434",
            "seats": [{"model": "llama3.1:8b", "family": "meta"},
                      {"model": "gemma4:latest", "family": "google"}]}}
        with open(self.panels_file, "w", encoding="utf-8") as fh:
            json.dump(panels, fh)

    def _main(self, extra, fake):
        argv = ["--panel", "test-panel", "--panels-file",
                self.panels_file] + extra
        out = io.StringIO()
        with mock.patch.object(rejudge, "rejudge_one", fake), \
                contextlib.redirect_stdout(out):
            status = paneljudge.main(argv)
        return status, out.getvalue()

    def test_log_transcript_collision_refused(self):
        fake = FakeRejudge()
        status, out = self._main([self.scenario, self.log], fake)
        self.assertEqual(status, 1)
        self.assertIn("REFUSED", out)
        self.assertNotIn("llama3.1:8b", fake.models())
        self.assertIn("gemma4:latest", fake.models())
        report = paneljudge.panel_report_path(self.log)
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("REFUSED (same family as contestant "
                      "llama3.1:8b)", text)

    def test_json_transcript_collision_refused(self):
        fake = FakeRejudge()
        status, out = self._main([self.scenario, self.saved_json],
                                  fake)
        self.assertEqual(status, 1)
        self.assertIn("REFUSED", out)
        self.assertNotIn("llama3.1:8b", fake.models())
        self.assertIn("gemma4:latest", fake.models())

    def test_scenario_collision_refused_with_clean_transcript(self):
        scenario = os.path.join(self.tmp.name, "scenario.json")
        with open(scenario, "w", encoding="utf-8") as fh:
            json.dump(SCENARIO, fh)
        fake = FakeRejudge()
        status, out = self._main([scenario, self.plain], fake)
        self.assertEqual(status, 1)
        self.assertIn("REFUSED", out)
        self.assertNotIn("llama3.1:8b", fake.models())

    def test_transcript_collision_allowed_with_flag(self):
        fake = FakeRejudge()
        status, out = self._main(
            ["--allow-same-family", self.scenario, self.log], fake)
        self.assertEqual(status, 0)
        self.assertIn("WARNING", out)
        self.assertIn("llama3.1:8b", fake.models())
        report = paneljudge.panel_report_path(self.log)
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("SAME-FAMILY SEAT: llama3.1:8b vs contestant "
                      "llama3.1:8b", text)

    def test_transcript_collision_refused_in_dry_run(self):
        fake = FakeRejudge()
        status, out = self._main(
            ["--dry-run", self.scenario, self.log], fake)
        self.assertEqual(status, 1)
        self.assertIn("would REFUSE (same family as contestant "
                      "llama3.1:8b)", out)
        self.assertNotIn("llama3.1:8b", fake.models())

    def test_missing_transcript_does_not_crash_the_check(self):
        missing = os.path.join(self.tmp.name, "gone.log")
        fake = FakeRejudge()
        status, _out = self._main([self.scenario, missing], fake)
        self.assertEqual(status, 0)
        self.assertIn("llama3.1:8b", fake.models())
        self.assertIn("gemma4:latest", fake.models())


if __name__ == "__main__":
    unittest.main()
