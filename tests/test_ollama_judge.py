"""Tests for ollama_judge.py -- the impartial third-model scorer."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_judge


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
        for judge, reports in [({"model": "j", "prompt": "x" * 60000}, None),
                               ({"model": "j"}, {0: "x" * 60000})]:
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


if __name__ == "__main__":
    unittest.main()
