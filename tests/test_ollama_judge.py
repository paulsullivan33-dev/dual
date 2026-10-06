"""Tests for ollama_judge.py -- the impartial third-model scorer."""
import os
import sys
import unittest

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


class RunJudgeTests(unittest.TestCase):
    def test_judge_call_uses_configured_model_and_defaults(self):
        seen = {}

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            seen.update(host=host, model=model, think=think,
                        options=options)
            return "", "Bob won.\nWINNER: Bob", "stop", {}

        verdict = ollama_judge.run_judge(
            "http://x", {"model": "judge-model"}, "topic",
            [{"name": "A", "model": "m1"}, {"name": "B", "model": "m2"}],
            [(0, "hi")], 30, fake_call_chat)
        self.assertEqual(verdict, "Bob won.\nWINNER: Bob")
        self.assertEqual(seen["model"], "judge-model")
        self.assertEqual(seen["options"]["num_predict"], 500)  # the default
        self.assertFalse(seen["think"])

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
