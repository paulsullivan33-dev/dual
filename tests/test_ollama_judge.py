"""Tests for ollama_judge.py -- the impartial third-model scorer."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_judge
from ollama_common import OllamaError


class NormalizeJudgeTests(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(ollama_judge.normalize_judge(None))

    def test_string_becomes_model_dict(self):
        self.assertEqual(ollama_judge.normalize_judge("qwen3:8b"),
                         {"model": "qwen3:8b"})

    def test_dict_passes_through(self):
        raw = {"model": "qwen3:8b", "temperature": 0.2, "max_tokens": 500,
               "think": False}
        self.assertEqual(ollama_judge.normalize_judge(raw), raw)

    def test_empty_string_exits(self):
        with self.assertRaises(SystemExit):
            ollama_judge.normalize_judge("   ")

    def test_non_string_non_dict_exits(self):
        with self.assertRaises(SystemExit):
            ollama_judge.normalize_judge(42)

    def test_dict_without_model_exits(self):
        with self.assertRaises(SystemExit):
            ollama_judge.normalize_judge({"temperature": 0.2})

    def test_dict_with_blank_model_exits(self):
        with self.assertRaises(SystemExit):
            ollama_judge.normalize_judge({"model": "  "})

    def test_unknown_key_exits(self):
        with self.assertRaises(SystemExit):
            ollama_judge.normalize_judge({"model": "qwen3:8b",
                                          "temperatue": 0.2})

    def test_bad_temperature_type_exits(self):
        with self.assertRaises(SystemExit):
            ollama_judge.normalize_judge({"model": "qwen3:8b",
                                          "temperature": "low"})


class BuildJudgeMessagesTests(unittest.TestCase):
    def test_messages_carry_names_topic_and_transcript(self):
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        transcript = [(0, "cats are best"), (1, "dogs are best")]
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "pets", participants, transcript)
        self.assertEqual(msgs[0]["role"], "system")
        body = msgs[1]["content"]
        self.assertIn("Alice", body)
        self.assertIn("Bob", body)
        self.assertIn("pets", body)
        self.assertIn("cats are best", body)
        self.assertIn("dogs are best", body)
        self.assertIn("WINNER:", body)  # default prompt demands the format

    def test_custom_prompt_replaces_default(self):
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        msgs = ollama_judge.build_judge_messages(
            {"model": "j", "prompt": "Pick the funniest, {a} vs {b}."},
            "pets", participants, [(0, "ha")])
        self.assertIn("Pick the funniest, Alice vs Bob.", msgs[1]["content"])

    def test_run_reports_follow_each_turn(self):
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        transcript = [(0, "here is code"), (1, "here is more code")]
        reports = {0: "[Automatic code run] Alice's program ran successfully.",
                   1: "[Automatic code run] Bob's program FAILED."}
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "pets", participants, transcript, reports)
        body = msgs[1]["content"]
        alice_at = body.index("here is code")
        report_at = body.index("Alice's program ran successfully")
        bob_at = body.index("here is more code")
        bob_report_at = body.index("Bob's program FAILED")
        # Each report follows its own turn, before the next turn's text.
        self.assertLess(alice_at, report_at)
        self.assertLess(report_at, bob_at)
        self.assertLess(bob_at, bob_report_at)
        # The judge is told to weigh what the code actually did.
        self.assertIn("what it did, not by what its author",
                      body)

    def test_no_run_reports_no_code_guidance(self):
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "pets", participants, [(0, "hi")])
        self.assertNotIn("what it did, not by what its author",
                         msgs[1]["content"])

    def test_briefs_shown_before_transcript_with_fidelity_note(self):
        participants = [
            {"name": "Alice", "model": "m1",
             "system": "You are Alice. Defend the castle at all costs."},
            {"name": "Bob", "model": "m2",
             "system": "You are Bob. Take the castle at all costs."},
        ]
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "castles", participants,
            [(0, "I surrender the castle"), (1, "onward")])
        body = msgs[1]["content"]
        self.assertIn("Assigned briefs:", body)
        self.assertIn("Alice's assigned brief", body)
        self.assertIn("Defend the castle at all costs.", body)
        self.assertIn("Bob's assigned brief", body)
        self.assertIn("Take the castle at all costs.", body)
        # The judge is told fidelity is a gate: crossing to the other
        # brief forfeits the win, and the briefs land ahead of the
        # transcript they are meant to frame.
        self.assertIn("abandoned their brief", body)
        self.assertIn("is a gate", body)
        self.assertIn("cannot win", body)
        self.assertIn("must be scored higher", body)
        self.assertIn("When both crossed, the verdict is a draw", body)
        self.assertLess(body.index("Assigned briefs:"),
                        body.index("Transcript:"))

    def test_no_briefs_no_briefs_section(self):
        participants = [{"name": "Alice", "model": "m1"},
                        {"name": "Bob", "model": "m2"}]
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "pets", participants, [(0, "hi")])
        body = msgs[1]["content"]
        self.assertNotIn("Assigned briefs:", body)
        self.assertNotIn("abandoned their brief", body)

    def test_blank_or_missing_system_counts_as_no_brief(self):
        participants = [{"name": "Alice", "model": "m1", "system": "  "},
                        {"name": "Bob", "model": "m2", "system": None}]
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "pets", participants, [(0, "hi")])
        self.assertNotIn("Assigned briefs:", msgs[1]["content"])

    def test_fidelity_ruling_shown_as_settled_fact(self):
        participants = [
            {"name": "Alice", "model": "m1", "system": "Defend the castle."},
            {"name": "Bob", "model": "m2", "system": "Take the castle."},
        ]
        fidelity = {"Alice": {"status": "CROSSED",
                               "evidence": "\"I surrender the castle\""},
                    "Bob": {"status": "HELD",
                            "evidence": "\"onward, always onward\""}}
        msgs = ollama_judge.build_judge_messages(
            {"model": "j"}, "castles", participants,
            [(0, "I surrender the castle"), (1, "onward")],
            fidelity=fidelity)
        body = msgs[1]["content"]
        self.assertIn("Fidelity ruling (settled by audit):", body)
        self.assertIn("Alice: CROSSED", body)
        self.assertIn("I surrender the castle", body)
        self.assertIn("Bob: HELD", body)
        # The ruling is presented as closed, and the judge is NOT asked
        # to decide fidelity itself (the no-audit wording).
        self.assertIn("settled fact", body)
        self.assertIn("Do not re-open", body)
        self.assertNotIn("decided before any scoring", body)
        # Order: briefs, then the ruling, then the transcript.
        self.assertLess(body.index("Assigned briefs:"),
                        body.index("Fidelity ruling"))
        self.assertLess(body.index("Fidelity ruling"),
                        body.index("Transcript:"))


class BuildAuditMessagesTests(unittest.TestCase):
    PARTICIPANTS = [
        {"name": "Alice", "model": "m1", "system": "Defend the castle."},
        {"name": "Bob", "model": "m2", "system": "Take the castle."},
    ]

    def test_audit_messages_carry_briefs_transcript_and_charge(self):
        msgs = ollama_judge.build_audit_messages(
            "castles", self.PARTICIPANTS,
            [(0, "I surrender the castle"), (1, "onward")])
        self.assertIn("fidelity auditor", msgs[0]["content"])
        self.assertIn("do not score", msgs[0]["content"])
        body = msgs[1]["content"]
        self.assertIn("Defend the castle.", body)
        self.assertIn("Take the castle.", body)
        self.assertIn("I surrender the castle", body)
        self.assertIn("RULING:", body)
        self.assertIn("HELD", body)
        self.assertIn("CROSSED", body)
        self.assertIn("EVIDENCE:", body)


class ParseFidelityTests(unittest.TestCase):
    PARTICIPANTS = [{"name": "Alice", "model": "m1"},
                    {"name": "Bob", "model": "m2"}]

    def test_parses_both_rulings_and_evidence(self):
        text = ("RULING: Alice CROSSED\n"
                "EVIDENCE: \"I surrender the castle\"\n"
                "RULING: Bob HELD\n"
                "EVIDENCE: \"onward, always onward\"")
        ruling = ollama_judge.parse_fidelity(text, self.PARTICIPANTS)
        self.assertEqual(ruling["Alice"]["status"], "CROSSED")
        self.assertEqual(ruling["Alice"]["evidence"],
                         "\"I surrender the castle\"")
        self.assertEqual(ruling["Bob"]["status"], "HELD")

    def test_markdown_bold_and_lowercase_tolerated(self):
        text = "**RULING:** alice held\n**RULING:** Bob crossed"
        ruling = ollama_judge.parse_fidelity(text, self.PARTICIPANTS)
        self.assertEqual(ruling["Alice"]["status"], "HELD")
        self.assertEqual(ruling["Bob"]["status"], "CROSSED")

    def test_missing_ruling_returns_none(self):
        text = "RULING: Alice HELD\nEVIDENCE: \"fine\""
        self.assertIsNone(
            ollama_judge.parse_fidelity(text, self.PARTICIPANTS))

    def test_garbage_and_empty_return_none(self):
        self.assertIsNone(
            ollama_judge.parse_fidelity("no rulings here",
                                        self.PARTICIPANTS))
        self.assertIsNone(ollama_judge.parse_fidelity("", self.PARTICIPANTS))
        self.assertIsNone(
            ollama_judge.parse_fidelity(None, self.PARTICIPANTS))


class ParseWinnerTests(unittest.TestCase):
    def test_extracts_name(self):
        self.assertEqual(
            ollama_judge.parse_winner("Alice argued better.\nWINNER: Alice"),
            "Alice")

    def test_draw(self):
        self.assertEqual(ollama_judge.parse_winner("WINNER: draw"), "draw")

    def test_missing_line_returns_none(self):
        self.assertIsNone(ollama_judge.parse_winner("Alice was great."))

    def test_empty_verdict_returns_none(self):
        self.assertIsNone(ollama_judge.parse_winner(""))

    def test_markdown_bold_winner_parses(self):
        self.assertEqual(
            ollama_judge.parse_winner("Reasoning here.\n**WINNER:** Alice"),
            "Alice")

    def test_markdown_bold_name_parses(self):
        self.assertEqual(
            ollama_judge.parse_winner("WINNER: **Bob**"), "Bob")

    def test_markdown_bold_title_case_winner_parses(self):
        self.assertEqual(
            ollama_judge.parse_winner(
                "Reasoning here.\n**Winner:** Beatrice the Bee"),
            "Beatrice the Bee")

    def test_title_case_winner_parses(self):
        self.assertEqual(
            ollama_judge.parse_winner("Alice argued better.\nWinner: Alice"),
            "Alice")

    def test_lower_case_winner_parses(self):
        self.assertEqual(ollama_judge.parse_winner("winner: draw"), "draw")

    def test_mid_line_winner_mention_returns_none(self):
        self.assertIsNone(
            ollama_judge.parse_winner("The winner: nobody, really."))


class JudgeContextTests(unittest.TestCase):
    def test_automatic_window_boundaries(self):
        # For one message: 6400 input tokens + 1280 margin + 512 output.
        judge = {"max_tokens": 512}
        messages = [{"content": "x" * ((6400 - 16) * 4)}]
        self.assertEqual(ollama_judge.judge_context_window(judge, messages),
                         (8192, 8192))
        messages[0]["content"] += "x"
        self.assertEqual(ollama_judge.judge_context_window(judge, messages)[0], 16384)
        judge["max_tokens"] = 1024
        messages[0]["content"] = "x" * ((12800 - 16) * 4)
        self.assertEqual(ollama_judge.judge_context_window(judge, messages),
                         (16384, 16384))
        messages[0]["content"] += "x"
        # One char past 16384 no longer refuses: the ladder grows.
        self.assertEqual(ollama_judge.judge_context_window(judge, messages)[0],
                         32768)

    def test_automatic_window_ladder_extends_to_65536(self):
        judge = {"max_tokens": 500}
        # ~18.5K required: past the old 16384 cap, fits the 32768 rung.
        messages = [{"content": "x" * 60000}]
        self.assertEqual(ollama_judge.judge_context_window(judge, messages)[0],
                         32768)
        # ~45.5K required: fits only the top rung.
        messages = [{"content": "x" * 150000}]
        self.assertEqual(ollama_judge.judge_context_window(judge, messages)[0],
                         65536)
        # ~90.5K required: past the ladder's ceiling -- still refused.
        messages = [{"content": "x" * 300000}]
        with self.assertRaises(ollama_judge.JudgeContextExceeded):
            ollama_judge.judge_context_window(judge, messages)

    def test_explicit_window_is_fixed_and_can_override_cap(self):
        messages = [{"content": "x" * 60000}]
        with self.assertRaises(ollama_judge.JudgeContextExceeded):
            ollama_judge.judge_context_window({"num_ctx": 8192}, messages)
        self.assertEqual(ollama_judge.judge_context_window(
            {"num_ctx": 32768}, messages)[0], 32768)

    def test_long_transcript_grows_window_without_changing_reply_budget(self):
        call = mock.Mock(return_value=("", "WINNER: draw", "stop", {}))
        ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic",
            [{"name": "A"}, {"name": "B"}], [(0, "x" * 30000)], 30, call)
        self.assertEqual(call.call_args.args[4]["num_ctx"], 16384)
        self.assertEqual(call.call_args.args[4]["num_predict"], 500)

    def test_instructions_and_code_reports_count_and_skip_before_request(self):
        call = mock.Mock()
        participants = [{"name": "A"}, {"name": "B"}]
        for judge, reports in [({"model": "j", "prompt": "x" * 300000}, None),
                               ({"model": "j"}, {0: "x" * 300000})]:
            with self.subTest(judge=judge["model"], reports=reports is not None):
                with self.assertRaises(ollama_judge.JudgeContextExceeded):
                    ollama_judge.run_judge(
                        "http://x", judge, "topic", participants, [(0, "hi")],
                        30, call, run_reports=reports)
        call.assert_not_called()


class RunJudgeTests(unittest.TestCase):
    def test_judge_host_accepted(self):
        raw = {"model": "j", "host": "http://judgebox:11434"}
        self.assertEqual(ollama_judge.normalize_judge(raw), raw)

    def test_judge_call_uses_configured_model_and_defaults(self):
        seen = {}

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            seen.update(host=host, model=model, think=think,
                        options=options)
            return "", "Bob won.\nWINNER: Bob", "stop", {}

        verdict, num_ctx, budget = ollama_judge.run_judge(
            "http://x", {"model": "judge-model"}, "topic",
            [{"name": "A", "model": "m1"}, {"name": "B", "model": "m2"}],
            [(0, "hi")], 30, fake_call_chat)
        self.assertEqual(verdict, "Bob won.\nWINNER: Bob")
        self.assertEqual(num_ctx, 8192)  # the auto-sized window, reported
        self.assertGreater(budget, 500)  # input estimate + reply allowance
        self.assertEqual(seen["model"], "judge-model")
        self.assertEqual(seen["options"]["num_predict"], 500)  # the default
        self.assertEqual(seen["options"]["temperature"], 0.2)  # the default
        self.assertEqual(seen["options"]["num_ctx"], 8192)  # the default
        self.assertFalse(seen["think"])

    def test_run_reports_reach_the_judge(self):
        seen = {}

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            seen["messages"] = messages
            return "", "WINNER: draw", "stop", {}

        ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic",
            [{"name": "A", "model": "m1"}, {"name": "B", "model": "m2"}],
            [(0, "code here")], 30, fake_call_chat,
            run_reports={0: "[Automatic code run] A's program FAILED."})
        body = seen["messages"][1]["content"]
        self.assertIn("A's program FAILED", body)
        self.assertIn("what it did, not by what its author", body)

    def test_judge_options_honored(self):
        seen = {}

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            seen.update(options=options, think=think)
            return "", "WINNER: draw", "stop", {}

        ollama_judge.run_judge(
            "http://x",
            {"model": "j", "max_tokens": 100, "temperature": 0.1,
             "num_ctx": 2048, "think": True},
            "topic",
            [{"name": "A", "model": "m1"}, {"name": "B", "model": "m2"}],
            [(0, "hi")], 30, fake_call_chat)
        self.assertEqual(seen["options"]["num_predict"], 100)
        self.assertEqual(seen["options"]["temperature"], 0.1)
        self.assertEqual(seen["options"]["num_ctx"], 2048)
        self.assertTrue(seen["think"])

    BRIEFED = [{"name": "A", "model": "m1", "system": "Brief for A."},
               {"name": "B", "model": "m2", "system": "Brief for B."}]
    AUDIT_REPLY = ("RULING: A CROSSED\n"
                   "EVIDENCE: \"a argued b's side\"\n"
                   "RULING: B HELD\n"
                   "EVIDENCE: \"b held the line\"")

    def _two_call_fake(self, calls, audit_reply=None, audit_error=None):
        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append(messages)
            if "fidelity auditor" in messages[0]["content"]:
                if audit_error is not None:
                    raise audit_error
                return "", audit_reply, "stop", {}
            return "", "Scored.\nWINNER: B", "stop", {}
        return fake_call_chat

    def test_briefs_trigger_audit_then_scoring_with_ruling(self):
        calls = []
        verdict, _num_ctx, _budget = ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic", self.BRIEFED,
            [(0, "hi"), (1, "lo")], 30,
            self._two_call_fake(calls, audit_reply=self.AUDIT_REPLY))
        self.assertEqual(len(calls), 2)  # audit call, then scoring call
        scoring = calls[1][1]["content"]
        self.assertIn("Fidelity ruling (settled by audit):", scoring)
        self.assertIn("A: CROSSED", scoring)
        self.assertIn("B: HELD", scoring)
        # The verdict is headed with the audit's ruling.
        self.assertTrue(verdict.startswith("Fidelity audit: A CROSSED"))
        self.assertIn("Fidelity audit: B HELD", verdict)
        self.assertIn("WINNER: B", verdict)

    def test_no_briefs_means_single_call(self):
        calls = []
        ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic",
            [{"name": "A", "model": "m1"}, {"name": "B", "model": "m2"}],
            [(0, "hi")], 30, self._two_call_fake(calls))
        self.assertEqual(len(calls), 1)

    def test_audit_host_failure_falls_back_to_self_ruling(self):
        calls = []
        verdict, _n, _b = ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic", self.BRIEFED,
            [(0, "hi"), (1, "lo")], 30,
            self._two_call_fake(
                calls, audit_error=OllamaError("host down")))
        self.assertEqual(len(calls), 2)  # failed audit + scoring call
        scoring = calls[1][1]["content"]
        # No ruling to settle: the scoring judge rules on fidelity
        # itself (the gate wording), and the verdict carries no header.
        self.assertIn("decided before any scoring", scoring)
        self.assertNotIn("Fidelity ruling (settled by audit):", scoring)
        self.assertFalse(verdict.startswith("Fidelity audit:"))

    def test_unparseable_audit_falls_back_to_self_ruling(self):
        calls = []
        verdict, _n, _b = ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic", self.BRIEFED,
            [(0, "hi"), (1, "lo")], 30,
            self._two_call_fake(calls, audit_reply="I decline to format."))
        scoring = calls[1][1]["content"]
        self.assertIn("decided before any scoring", scoring)
        self.assertFalse(verdict.startswith("Fidelity audit:"))


class JudgeRetryTests(unittest.TestCase):
    """A judge reply cut off at the reply limit (done_reason "length")
    is retried once at double the budget; a finished reply is not."""

    PARTICIPANTS = [{"name": "A", "model": "m1"},
                    {"name": "B", "model": "m2"}]
    BRIEFED = [{"name": "A", "model": "m1", "system": "Brief for A."},
               {"name": "B", "model": "m2", "system": "Brief for B."}]
    AUDIT_REPLY = ("RULING: A CROSSED\n"
                   "EVIDENCE: \"a argued b's side\"\n"
                   "RULING: B HELD\n"
                   "EVIDENCE: \"b held the line\"")

    def test_truncated_verdict_retried_once_at_double_budget(self):
        calls = []

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append(options)
            if len(calls) == 1:
                return "", "Reasoning that never finis", "length", {}
            return "", "Full verdict.\nWINNER: B", "stop", {}

        verdict, num_ctx, _budget = ollama_judge.run_judge(
            "http://x", {"model": "j", "max_tokens": 100}, "topic",
            self.PARTICIPANTS, [(0, "hi")], 30, fake_call_chat)
        self.assertEqual(verdict, "Full verdict.\nWINNER: B")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["num_predict"], 100)
        self.assertEqual(calls[1]["num_predict"], 200)
        # The reported window is the final attempt's, not the first's.
        self.assertEqual(num_ctx, calls[1]["num_ctx"])

    def test_finished_verdict_is_not_retried(self):
        calls = []

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append(options)
            return "", "Done.\nWINNER: draw", "stop", {}

        ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic", self.PARTICIPANTS,
            [(0, "hi")], 30, fake_call_chat)
        self.assertEqual(len(calls), 1)

    def test_second_truncation_stands(self):
        calls = []

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append(options)
            return "", f"attempt {len(calls)} cut off", "length", {}

        verdict, _n, _b = ollama_judge.run_judge(
            "http://x", {"model": "j", "max_tokens": 100}, "topic",
            self.PARTICIPANTS, [(0, "hi")], 30, fake_call_chat)
        # Bounded: exactly one retry, and its reply is what stands.
        self.assertEqual(len(calls), 2)
        self.assertEqual(verdict, "attempt 2 cut off")

    def test_truncated_audit_retried_then_scoring_uses_ruling(self):
        calls = []

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append((messages, options))
            if "fidelity auditor" in messages[0]["content"]:
                if len(calls) == 1:
                    return "", "RULING: A CRO", "length", {}
                return "", self.AUDIT_REPLY, "stop", {}
            return "", "Scored.\nWINNER: B", "stop", {}

        verdict, _n, _b = ollama_judge.run_judge(
            "http://x", {"model": "j"}, "topic", self.BRIEFED,
            [(0, "hi"), (1, "lo")], 30, fake_call_chat)
        self.assertEqual(len(calls), 3)  # audit, audit retry, scoring
        self.assertEqual(calls[0][1]["num_predict"], 500)   # the default
        self.assertEqual(calls[1][1]["num_predict"], 1000)  # doubled
        self.assertTrue(verdict.startswith("Fidelity audit: A CROSSED"))
        self.assertIn("WINNER: B", verdict)


if __name__ == "__main__":
    unittest.main()
