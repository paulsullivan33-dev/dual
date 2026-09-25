import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_duel


def write_json(directory, name, data):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def minimal_config(**overrides):
    cfg = {
        "topic": "test topic",
        "models": [
            {"model": "m1"},
            {"model": "m2", "name": "Two"},
        ],
    }
    cfg.update(overrides)
    return cfg


class FirstNotNoneTests(unittest.TestCase):
    def test_returns_first_non_none(self):
        self.assertEqual(ollama_duel.first_not_none(None, None, 3, 4), 3)

    def test_all_none_returns_none(self):
        self.assertIsNone(ollama_duel.first_not_none(None, None))

    def test_preserves_explicit_false(self):
        # False must win over a later default -- it is not "missing".
        self.assertEqual(ollama_duel.first_not_none(None, False, True), False)

    def test_no_args_returns_none(self):
        self.assertIsNone(ollama_duel.first_not_none())


class ValidateFieldTests(unittest.TestCase):
    def test_missing_key_is_ignored(self):
        ollama_duel._validate_field({}, "turns", "int", "top level", minimum=1)

    def test_none_value_is_ignored(self):
        ollama_duel._validate_field({"turns": None}, "turns", "int", "top level")

    def test_wrong_type_exits(self):
        with self.assertRaises(SystemExit):
            ollama_duel._validate_field({"turns": "6"}, "turns", "int", "top level")

    def test_bool_rejected_for_int(self):
        # isinstance(True, int) is True in Python -- must not slip through.
        with self.assertRaises(SystemExit):
            ollama_duel._validate_field({"turns": True}, "turns", "int", "top level")

    def test_below_minimum_exits(self):
        with self.assertRaises(SystemExit):
            ollama_duel._validate_field({"turns": 0}, "turns", "int", "top level", minimum=1)

    def test_valid_value_passes(self):
        ollama_duel._validate_field({"turns": 5}, "turns", "int", "top level", minimum=1)

    def test_number_accepts_int_and_float(self):
        ollama_duel._validate_field({"temperature": 1}, "temperature", "number", "top level")
        ollama_duel._validate_field({"temperature": 0.5}, "temperature", "number", "top level")

    def test_bool_rejected_for_bool_kind_wrong_type(self):
        with self.assertRaises(SystemExit):
            ollama_duel._validate_field({"think": "yes"}, "think", "bool", "top level")


class LoadConfigTests(unittest.TestCase):
    def test_valid_minimal_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config())
            cfg = ollama_duel.load_config(path)
        self.assertEqual(cfg["models"][0]["name"], "m1")  # defaulted from "model"
        self.assertEqual(cfg["models"][1]["name"], "Two")  # explicit name kept

    def test_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            ollama_duel.load_config("no-such-file.json")

    def test_invalid_json_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_non_utf8_file_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad-encoding.json")
            with open(path, "wb") as f:
                f.write(b'{"topic": "\xff\xfe bad bytes"}')
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_non_object_top_level_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", [1, 2, 3])
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_wrong_model_count_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(models=[{"model": "m1"}]))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_model_entry_missing_model_key_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(models=[{"name": "x"}, {"model": "m2"}]))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_bad_top_level_turns_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turns="6"))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_negative_turns_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turns=-1))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_bad_per_model_temperature_exits(self):
        cfg = minimal_config(models=[{"model": "m1", "temperature": "hot"}, {"model": "m2"}])
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", cfg)
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_unknown_top_level_key_exits(self):
        cfg = minimal_config(temprature=0.7)  # typo, not "temperature"
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", cfg)
            with self.assertRaises(SystemExit) as ctx:
                ollama_duel.load_config(path)
        self.assertIn("temprature", str(ctx.exception))

    def test_unknown_model_key_exits(self):
        cfg = minimal_config(models=[{"model": "m1", "num_context": 4096}, {"model": "m2"}])
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", cfg)
            with self.assertRaises(SystemExit) as ctx:
                ollama_duel.load_config(path)
        self.assertIn("num_context", str(ctx.exception))

    def test_valid_full_config(self):
        cfg = minimal_config(
            host="http://localhost:11434", turns=4, think=False, max_tokens=300,
            temperature=0.7, num_ctx=4096, log_file="out.log", save_json="out.json",
            timeout=60,
        )
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", cfg)
            loaded = ollama_duel.load_config(path)
        self.assertEqual(loaded["turns"], 4)


class MainTurnsValidationTests(unittest.TestCase):
    def test_cli_turns_override_below_one_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turns=6))
            with mock.patch.object(sys, "argv", ["ollama_duel.py", path, "--turns", "0"]):
                with self.assertRaises(SystemExit):
                    ollama_duel.main()


class TurnNudgeTests(unittest.TestCase):
    """From turn 2 on, each turn's prompt must end with a nudge telling the
    model to reply directly to the other participant. Regression test: small
    models otherwise ignore the transcript and each emit a standalone
    continuation of the topic (parallel monologues, no back-and-forth)."""

    def _run_two_turns(self):
        seen = []

        def fake_call_chat(host, model, messages, think, options, timeout=None):
            seen.append([m["content"] for m in messages])
            metrics = {"gen_tokens": 10, "gen_s": 1.0,
                       "prompt_tokens": 20, "prompt_s": 0.5}
            return "", "canned reply", "stop", metrics

        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turns=2))
            with mock.patch.object(sys, "argv", ["ollama_duel.py", path]), \
                 mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
                ollama_duel.main()
        return seen

    def test_first_turn_has_topic_and_no_nudge(self):
        seen = self._run_two_turns()
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][-1], "test topic")

    def test_later_turns_nudge_a_direct_reply(self):
        seen = self._run_two_turns()
        nudge = seen[1][-1]
        # Turn 2 is spoken by "Two", so the nudge must name the other side.
        self.assertIn("Reply directly to m1's last message", nudge)
        self.assertIn("staying in character as Two", nudge)
        self.assertIn("Do not repeat or summarize", nudge)

    def test_topic_opens_every_turn(self):
        # Regression test: the topic used to be sent only on turn 1, so the
        # second speaker never saw it. It must open each speaker's messages.
        seen = self._run_two_turns()
        for contents in seen:
            self.assertEqual(contents[0], "test topic")
        self.assertEqual(seen[1][1], "canned reply")


def run_duel(cfg, metrics=None):
    """Run ollama_duel.main() on `cfg` with call_chat faked; return the
    message contents sent on each turn."""
    seen = []
    metrics = metrics or {"gen_tokens": 10, "gen_s": 1.0,
                          "prompt_tokens": 20, "prompt_s": 0.5}

    def fake_call_chat(host, model, messages, think, options, timeout=None):
        seen.append([m["content"] for m in messages])
        return "", "canned reply", "stop", metrics

    with tempfile.TemporaryDirectory() as d:
        path = write_json(d, "cfg.json", cfg)
        with mock.patch.object(sys, "argv", ["ollama_duel.py", path]), \
             mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
            ollama_duel.main()
    return seen


class TurnPromptTests(unittest.TestCase):
    """turn_prompt replaces the default per-turn nudge, e.g. for scenarios
    that want a whole program or a long passage rather than "a few short
    paragraphs"."""

    def test_top_level_turn_prompt_replaces_default(self):
        seen = run_duel(minimal_config(
            turns=2, turn_prompt="Improve {other}'s program, {name}."))
        self.assertEqual(seen[1][-1], "Improve m1's program, Two.")

    def test_per_model_turn_prompt_wins_over_top_level(self):
        cfg = minimal_config(turns=3, turn_prompt="top level")
        cfg["models"][0]["turn_prompt"] = "model one"
        seen = run_duel(cfg)
        self.assertEqual(seen[1][-1], "top level")   # turn 2: Two
        self.assertEqual(seen[2][-1], "model one")   # turn 3: m1

    def test_empty_turn_prompt_disables_nudge(self):
        seen = run_duel(minimal_config(turns=2, turn_prompt=""))
        self.assertEqual(seen[1][-1], "canned reply")

    def test_other_braces_are_left_alone(self):
        self.assertEqual(
            ollama_duel.render_turn_prompt("{name} vs {other}: {x} {}", "A", "B"),
            "A vs B: {x} {}")

    def test_non_string_turn_prompt_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turn_prompt=5))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)


class ContextWarningTests(unittest.TestCase):
    def test_max_tokens_above_num_ctx_warns(self):
        w = ollama_duel.context_warning("A", 8000, 4096)
        self.assertIn("larger than num_ctx (4096)", w)

    def test_max_tokens_within_num_ctx_is_fine(self):
        self.assertIsNone(ollama_duel.context_warning("A", 16384, 16384))

    def test_large_max_tokens_without_num_ctx_warns(self):
        w = ollama_duel.context_warning("A", 80000, None)
        self.assertIn("num_ctx is not", w)

    def test_small_max_tokens_without_num_ctx_is_fine(self):
        self.assertIsNone(ollama_duel.context_warning("A", 2048, None))


class MatrixSpeedTests(unittest.TestCase):
    def test_matrix_shows_measured_generation_speed(self):
        shown = []

        class FakeMatrix:
            def show_text(self, text):
                shown.append(text)

            def progress(self, done, total):
                pass

        metrics = {"gen_tokens": 10, "gen_s": 1.0, "prompt_tokens": 20,
                   "prompt_s": 0.5, "gen_tps": 12.345}
        with mock.patch("unoq_matrix.UnoQMatrix", FakeMatrix):
            run_duel(minimal_config(turns=1, display=True), metrics=metrics)
        self.assertIn("12.3T/S", shown)


class RepeatPenaltyTests(unittest.TestCase):
    """repeat_penalty must reach Ollama's options so scenarios can tame
    repetitive small models. Regression test: the key was silently dropped
    before (unknown-key validation would even have rejected it)."""

    def _run_turns(self, turns, **overrides):
        seen_options = []

        def fake_call_chat(host, model, messages, think, options, timeout=None):
            seen_options.append(options)
            metrics = {"gen_tokens": 10, "gen_s": 1.0,
                       "prompt_tokens": 20, "prompt_s": 0.5}
            return "", "canned reply", "stop", metrics

        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turns=turns, **overrides))
            with mock.patch.object(sys, "argv", ["ollama_duel.py", path]), \
                 mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
                ollama_duel.main()
        return seen_options

    def test_top_level_repeat_penalty_reaches_options(self):
        seen = self._run_turns(1, repeat_penalty=1.2)
        self.assertEqual(seen[0]["repeat_penalty"], 1.2)

    def test_per_model_repeat_penalty_wins_over_top_level(self):
        models = [
            {"model": "m1", "repeat_penalty": 1.3},
            {"model": "m2", "name": "Two"},
        ]
        seen = self._run_turns(2, models=models, repeat_penalty=1.1)
        self.assertEqual(seen[0]["repeat_penalty"], 1.3)
        self.assertEqual(seen[1]["repeat_penalty"], 1.1)

    def test_absent_repeat_penalty_not_in_options(self):
        seen = self._run_turns(1)
        self.assertNotIn("repeat_penalty", seen[0])

    def test_bad_per_model_repeat_penalty_exits(self):
        cfg = minimal_config(models=[{"model": "m1", "repeat_penalty": "high"},
                                      {"model": "m2"}])
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", cfg)
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_below_minimum_repeat_penalty_exits(self):
        cfg = minimal_config(repeat_penalty=0.5)
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", cfg)
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)


class ScenarioFilesValidateTests(unittest.TestCase):
    """Every shipped *.json scenario must load and validate cleanly, since
    these are the files new users copy and run first."""

    def _scenario_paths(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return sorted(
            os.path.join(root, name) for name in os.listdir(root) if name.endswith(".json")
        )

    def test_all_scenarios_load_and_validate(self):
        paths = self._scenario_paths()
        self.assertGreater(len(paths), 0, "expected at least one scenario JSON file")
        for path in paths:
            with self.subTest(path=os.path.basename(path)):
                cfg = ollama_duel.load_config(path)
                # Recommended, not just required: every shipped scenario should
                # set a topic, an explicit turn count, and a generation budget
                # (top-level or per-model) rather than relying on silent defaults.
                self.assertTrue(cfg.get("topic"), "missing topic")
                self.assertIsNotNone(cfg.get("turns"), "missing turns")
                has_max_tokens = cfg.get("max_tokens") is not None or all(
                    m.get("max_tokens") is not None for m in cfg["models"]
                )
                self.assertTrue(has_max_tokens, "missing max_tokens (top level or per model)")
                for m in cfg["models"]:
                    self.assertTrue(m.get("system"), f'model "{m["name"]}" has no system prompt')


class FormatDuelStatsTests(unittest.TestCase):
    def _stats(self):
        return {
            "fast": {"turns": 2, "gen_tokens": 200, "gen_s": 2.0,
                     "prompt_tokens": 80, "prompt_s": 0.2, "truncated": 0},
            "slow": {"turns": 2, "gen_tokens": 100, "gen_s": 10.0,
                     "prompt_tokens": 80, "prompt_s": 0.4, "truncated": 1},
        }

    def test_table_shows_weighted_tokens_per_second(self):
        table = ollama_duel.format_duel_stats(self._stats())
        self.assertIn("Model", table)
        self.assertIn("Gen tok/s", table)
        # 200 tokens / 2.0 s = 100.0 tok/s; 100 / 10.0 = 10.0 tok/s
        self.assertIn("100.0", table)
        self.assertIn("10.0", table)

    def test_table_shows_turns_tokens_and_truncated(self):
        table = ollama_duel.format_duel_stats(self._stats())
        self.assertIn("fast", table)
        self.assertIn("slow", table)
        lines = table.splitlines()
        slow_line = next(l for l in lines if "slow" in l)
        self.assertIn("1", slow_line.split()[-1])  # truncated count

    def test_empty_stats_returns_empty_string(self):
        self.assertEqual(ollama_duel.format_duel_stats({}), "")


class TimestampedLogPathTests(unittest.TestCase):
    def test_prepends_datetime_stamp_to_base_name(self):
        path = ollama_duel.timestamped_log_path("duel.log")
        self.assertRegex(path, r"^\d{8}-\d{6}-duel\.log$")

    def test_keeps_directory(self):
        path = ollama_duel.timestamped_log_path(os.path.join("logs", "duel.log"))
        self.assertRegex(path, r"^logs[/\\]\d{8}-\d{6}-duel\.log$")

    def test_stamp_matches_today(self):
        from datetime import datetime
        path = ollama_duel.timestamped_log_path("duel.log")
        self.assertTrue(path.startswith(datetime.now().strftime("%Y%m%d-")))


if __name__ == "__main__":
    unittest.main()
