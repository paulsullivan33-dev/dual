import io
import json
import os
import shutil
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


class MainMaxTokensValidationTests(unittest.TestCase):
    def test_cli_max_tokens_override_below_one_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config())
            with mock.patch.object(sys, "argv",
                                   ["ollama_duel.py", path, "--max-tokens", "0"]):
                with self.assertRaises(SystemExit):
                    ollama_duel.main()

    def test_cli_max_tokens_overrides_config(self):
        seen_options = []

        def fake_call_chat(host, model, messages, think, options, timeout=None):
            seen_options.append(options)
            metrics = {"gen_tokens": 10, "gen_s": 1.0,
                       "prompt_tokens": 20, "prompt_s": 0.5}
            return "", "canned reply", "stop", metrics

        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(turns=2, max_tokens=300))
            with mock.patch.object(sys, "argv",
                                   ["ollama_duel.py", path,
                                    "--max-tokens", "1234"]), \
                 mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
                ollama_duel.main()
        self.assertTrue(seen_options)
        for options in seen_options:
            self.assertEqual(options["num_predict"], 1234)


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

    def test_first_turn_ends_with_opening_nudge(self):
        # The opening speaker used to get no nudge at all, so it would
        # script-write both sides of the debate on turn 1. Now the first
        # turn closes with the first-turn nudge instead of the bare topic.
        seen = self._run_two_turns()
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0], "test topic")
        nudge = seen[0][-1]
        self.assertIn("You are m1.", nudge)
        self.assertIn("Write only m1's own words and actions", nudge)
        self.assertIn("never write", nudge)
        self.assertIn("dialogue or actions for Two", nudge)

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
        with mock.patch.object(sys, "argv",
                               ["ollama_duel.py", path, "--no-results-log"]), \
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

    def test_default_nudge_forbids_writing_the_other_side(self):
        # Regression test: small models play both parts in one reply.
        # The default nudge must tell each speaker to write only itself.
        seen = run_duel(minimal_config(turns=2))
        nudge = seen[1][-1]
        self.assertIn("Write only Two's own words and actions", nudge)
        self.assertIn("never write", nudge)
        self.assertIn("dialogue or actions for m1", nudge)

    def test_other_braces_are_left_alone(self):
        self.assertEqual(
            ollama_duel.render_turn_prompt("{name} vs {other}: {x} {}", "A", "B"),
            "A vs B: {x} {}")

    def test_non_string_turn_prompt_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turn_prompt=5))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_first_turn_prompt_replaces_default_opening_nudge(self):
        seen = run_duel(minimal_config(
            turns=2, first_turn_prompt="Open strong, {name}."))
        self.assertEqual(seen[0][-1], "Open strong, m1.")

    def test_per_model_first_turn_prompt_wins_over_top_level(self):
        cfg = minimal_config(turns=2, first_turn_prompt="top level")
        cfg["models"][0]["first_turn_prompt"] = "model one"
        seen = run_duel(cfg)
        self.assertEqual(seen[0][-1], "model one")

    def test_empty_first_turn_prompt_disables_opening_nudge(self):
        seen = run_duel(minimal_config(turns=2, first_turn_prompt=""))
        self.assertEqual(seen[0][-1], "test topic")

    def test_non_string_first_turn_prompt_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(first_turn_prompt=5))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)


class DryRunTests(unittest.TestCase):
    """--dry-run prints the exact turn-1 messages and exits without
    calling Ollama."""

    def test_dry_run_prints_turn_one_and_makes_no_calls(self):
        import contextlib
        import io
        calls = []
        metrics = {"gen_tokens": 10, "gen_s": 1.0,
                   "prompt_tokens": 20, "prompt_s": 0.5}

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append(messages)
            return "", "canned reply", "stop", metrics

        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json", minimal_config(turns=2))
            with mock.patch.object(sys, "argv",
                                   ["ollama_duel.py", path, "--dry-run",
                                    "--no-results-log"]), \
                 mock.patch.object(ollama_duel, "call_chat", fake_call_chat), \
                 contextlib.redirect_stdout(buf):
                ollama_duel.main()
        self.assertEqual(calls, [])
        out = buf.getvalue()
        self.assertIn("test topic", out)
        self.assertIn("[user]", out)
        self.assertIn("You are m1.", out)
        self.assertIn("Write only m1's own words and actions", out)


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

    def test_absent_repeat_penalty_defaults_to_1_25(self):
        seen = self._run_turns(1)
        self.assertEqual(seen[0]["repeat_penalty"], 1.25)

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
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios")
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
                # Keep run logs out of the repo root.
                if cfg.get("log_file"):
                    self.assertTrue(cfg["log_file"].startswith("logs/"),
                                    f'log_file {cfg["log_file"]!r} should be under logs/')


class LogDirectoryTests(unittest.TestCase):
    def test_missing_log_directory_is_created(self):
        with tempfile.TemporaryDirectory() as d:
            log_file = os.path.join(d, "new", "nested", "duel.log")
            run_duel(minimal_config(turns=1, log_file=log_file))
            logs = os.listdir(os.path.join(d, "new", "nested"))
            self.assertEqual(len(logs), 1)
            self.assertTrue(logs[0].endswith("-duel.log"))


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


class SessionMarkerTests(unittest.TestCase):
    def test_session_end_repeats_start_time_on_line_above(self):
        with tempfile.TemporaryDirectory() as d:
            log_file = os.path.join(d, "duel.log")
            run_duel(minimal_config(turns=1, log_file=log_file))
            logs = os.listdir(d)
            self.assertEqual(len(logs), 1)
            with open(os.path.join(d, logs[0]), encoding="utf-8") as f:
                lines = f.read().splitlines()
            end_idx = next(i for i, l in enumerate(lines)
                           if "session ended" in l)
            self.assertIn("(1 replies)", lines[end_idx])
            # The line directly above repeats the session start marker,
            # identical to the one written at the top of the log.
            first_start = next(l for l in lines if "session started" in l)
            self.assertEqual(lines[end_idx - 1], first_start)


class StopNoteTests(unittest.TestCase):
    """When a duel stops early, the reason must reach the log file.
    Regression test: a batch run on the Arduino Uno Q kept dying with
    "(0 replies)" and no reason, because the OllamaError message only
    went to stderr, which the log file never mirrors."""

    @staticmethod
    def _canned(*args, **kwargs):
        metrics = {"gen_tokens": 10, "gen_s": 1.0,
                   "prompt_tokens": 20, "prompt_s": 0.5}
        return "", "canned reply", "stop", metrics

    def _logged_run(self, d, fake_call_chat, **overrides):
        """Run main() with call_chat faked; return the log file's text.
        The config lives in its own temp dir so `d` holds only the log."""
        cfg = minimal_config(log_file=os.path.join(d, "duel.log"), **overrides)
        with tempfile.TemporaryDirectory() as cfg_dir:
            path = write_json(cfg_dir, "cfg.json", cfg)
            with mock.patch.object(sys, "argv",
                                   ["ollama_duel.py", path, "--no-results-log"]), \
                 mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
                ollama_duel.main()
        logs = os.listdir(d)
        self.assertEqual(len(logs), 1)
        with open(os.path.join(d, logs[0]), encoding="utf-8") as f:
            return f.read()

    def test_ollama_error_reason_reaches_log_file(self):
        def boom(*args, **kwargs):
            raise ollama_duel.OllamaError("Cannot reach Ollama at http://x")

        with tempfile.TemporaryDirectory() as d:
            contents = self._logged_run(d, boom, turns=3)
        self.assertIn("(0 replies)", contents)
        self.assertIn("--- stopped early ---", contents)
        self.assertIn("Cannot reach Ollama at http://x", contents)

    def test_keyboard_interrupt_reason_reaches_log_file(self):
        def intr(*args, **kwargs):
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as d:
            contents = self._logged_run(d, intr, turns=3)
        self.assertIn("(0 replies)", contents)
        self.assertIn("--- stopped early ---", contents)
        self.assertIn("interrupted by user", contents)

    def test_clean_duel_writes_no_stop_marker(self):
        with tempfile.TemporaryDirectory() as d:
            contents = self._logged_run(d, self._canned, turns=1)
        self.assertIn("(1 replies)", contents)
        self.assertNotIn("stopped early", contents)

    def test_run_duel_returns_stop_note(self):
        def boom(host, model, messages, think, options, timeout=None):
            raise ollama_duel.OllamaError("nope")

        participants = [
            {"name": "One", "model": "m1", "system": None, "think": False,
             "options": {"num_predict": 50}, "turn_prompt": ""},
            {"name": "Two", "model": "m2", "system": None, "think": False,
             "options": {"num_predict": 50}, "turn_prompt": ""},
        ]
        transcript = []
        with mock.patch.object(ollama_duel, "call_chat", boom):
            _, _, stop_note = ollama_duel.run_duel(
                "http://x", "topic", 2, participants, 60, transcript)
        self.assertEqual(stop_note, "nope")
        self.assertEqual(transcript, [])

    def test_run_duel_returns_none_when_all_turns_complete(self):
        with tempfile.TemporaryDirectory() as d:
            contents = self._logged_run(d, self._canned, turns=2)
        self.assertIn("(2 replies)", contents)
        self.assertNotIn("stopped early", contents)


class DedupGuardTests(unittest.TestCase):
    """If a speaker repeats its own previous reply verbatim, the turn is
    re-rolled once with a bumped temperature. Regression test: smollm2:1.7b
    once echoed its whole previous reply word-for-word on a later turn."""

    METRICS = {"gen_tokens": 10, "gen_s": 1.0,
               "prompt_tokens": 20, "prompt_s": 0.5}

    def _participants(self, temperature=None):
        opts = {"num_predict": 50}
        if temperature is not None:
            opts["temperature"] = temperature
        return [
            {"name": "One", "model": "m1", "system": None, "think": False,
             "options": dict(opts), "turn_prompt": ""},
            {"name": "Two", "model": "m2", "system": None, "think": False,
             "options": dict(opts), "turn_prompt": ""},
        ]

    def _run(self, turns, replies, participants=None, dedup_guard=True):
        """Drive run_duel with a canned reply script; return (transcript,
        calls, options_seen). The script is consumed in order; extra turns
        reuse the last entry."""
        script = list(replies)
        calls = []
        options_seen = []

        def fake_call_chat(host, model, messages, think, options, timeout=None):
            options_seen.append(dict(options))
            calls.append(model)
            text = script[min(len(calls) - 1, len(script) - 1)]
            return "", text, "stop", dict(self.METRICS)

        transcript = []
        with mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
            ollama_duel.run_duel(
                "http://x", "topic", turns,
                participants or self._participants(),
                60, transcript, dedup_guard=dedup_guard)
        return transcript, calls, options_seen

    def test_duplicate_triggers_one_reroll(self):
        # Turn 3 (speaker One) repeats its turn-1 reply; the re-roll wins.
        transcript, calls, _ = self._run(3, ["aaa", "bbb", "aaa", "ccc"])
        self.assertEqual(len(calls), 4)  # 3 turns + 1 retry
        self.assertEqual([t for _, t in transcript],
                         ["aaa", "bbb", "ccc"])

    def test_reroll_uses_bumped_temperature(self):
        _, _, options_seen = self._run(3, ["aaa", "bbb", "aaa", "ccc"])
        # First three calls carry no temperature (Ollama default); the
        # re-roll (4th call) bumps it by DEDUP_RETRY_TEMP_BUMP.
        self.assertNotIn("temperature", options_seen[0])
        self.assertAlmostEqual(
            options_seen[3]["temperature"],
            ollama_duel.ASSUMED_DEFAULT_TEMPERATURE
            + ollama_duel.DEDUP_RETRY_TEMP_BUMP)

    def test_reroll_bumps_explicit_temperature(self):
        _, _, options_seen = self._run(
            3, ["aaa", "bbb", "aaa", "ccc"],
            participants=self._participants(temperature=0.2))
        self.assertAlmostEqual(options_seen[0]["temperature"], 0.2)
        self.assertAlmostEqual(options_seen[3]["temperature"], 0.5)

    def test_second_duplicate_is_accepted(self):
        # The re-roll duplicates again; it is kept rather than retried
        # forever.
        transcript, calls, _ = self._run(3, ["aaa", "bbb", "aaa", "aaa"])
        self.assertEqual(len(calls), 4)
        self.assertEqual([t for _, t in transcript],
                         ["aaa", "bbb", "aaa"])

    def test_reroll_appends_no_repeat_nudge(self):
        # The retry must carry an explicit no-repeat instruction: a bare
        # temperature bump with an unchanged prompt often repeats again.
        script = ["aaa", "bbb", "aaa", "ccc"]
        calls = []
        messages_seen = []

        def fake_call_chat(host, model, messages, think, options,
                           timeout=None):
            calls.append(model)
            messages_seen.append(list(messages))
            text = script[min(len(calls) - 1, len(script) - 1)]
            return "", text, "stop", dict(self.METRICS)

        transcript = []
        with mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
            ollama_duel.run_duel("http://x", "topic", 3,
                                 self._participants(), 60, transcript)
        self.assertEqual(len(calls), 4)
        retry_messages = messages_seen[3]
        self.assertEqual(retry_messages[-1],
                         {"role": "user",
                          "content": ollama_duel.DEDUP_RETRY_NUDGE})
        # The nudge is appended to a copy; the original messages are
        # untouched.
        self.assertEqual(retry_messages[:-1], messages_seen[2])

    def test_second_duplicate_prints_note(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            transcript, calls, _ = self._run(
                3, ["aaa", "bbb", "aaa", "aaa"])
        self.assertEqual(len(calls), 4)
        self.assertIn("repeated again", buf.getvalue())
        self.assertEqual([t for _, t in transcript],
                         ["aaa", "bbb", "aaa"])

    def test_distinct_replies_never_reroll(self):
        _transcript, calls, options_seen = self._run(3, ["aaa", "bbb", "ccc"])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("temperature" not in o for o in options_seen))

    def test_whitespace_only_difference_still_counts(self):
        transcript, calls, _ = self._run(
            3, ["aaa bbb", "xyz", "aaa\nbbb", "ccc"])
        self.assertEqual(len(calls), 4)
        self.assertEqual(transcript[2][1], "ccc")

    def test_guard_can_be_disabled(self):
        transcript, calls, _ = self._run(3, ["aaa", "bbb", "aaa"],
                                         dedup_guard=False)
        self.assertEqual(len(calls), 3)
        self.assertEqual([t for _, t in transcript],
                         ["aaa", "bbb", "aaa"])

    def test_config_knob_reaches_run_duel(self):
        seen = {}

        def fake_run_duel(host, topic, turns, participants, timeout,
                          transcript, matrix=None, dedup_guard=True):
            seen["dedup_guard"] = dedup_guard
            return {}, None, None

        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(turns=1, dedup_guard=False))
            with mock.patch.object(sys, "argv", ["ollama_duel.py", path]), \
                 mock.patch.object(ollama_duel, "run_duel", fake_run_duel):
                ollama_duel.main()
        self.assertFalse(seen["dedup_guard"])

    def test_bad_config_knob_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(turns=1, dedup_guard="yes"))
            with mock.patch.object(sys, "argv", ["ollama_duel.py", path]):
                with self.assertRaises(SystemExit):
                    ollama_duel.load_config(path)

    def test_normalize_reply(self):
        self.assertEqual(ollama_duel.normalize_reply("  a\n b\tc "),
                         "a b c")


class RunSummaryTests(unittest.TestCase):
    """After every duel, a one-block summary is appended to run_results.log:
    date/time plus the stats table on success, or the error message when the
    duel stopped early -- so a batch of runs can be scanned without opening
    each transcript log."""

    @staticmethod
    def _canned(*args, **kwargs):
        metrics = {"gen_tokens": 10, "gen_s": 1.0,
                   "prompt_tokens": 20, "prompt_s": 0.5}
        return "", "canned reply", "stop", metrics

    def _run(self, d, fake_call_chat, extra_argv=(), **overrides):
        """Run main() with call_chat faked; the summary goes to d's
        run_results.log. Returns that file's text."""
        cfg = minimal_config(results_log=os.path.join(d, "run_results.log"),
                             **overrides)
        with tempfile.TemporaryDirectory() as cfg_dir:
            path = write_json(cfg_dir, "cfg.json", cfg)
            argv = ["ollama_duel.py", path] + list(extra_argv)
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(ollama_duel, "call_chat", fake_call_chat):
                ollama_duel.main()
        with open(os.path.join(d, "run_results.log"), encoding="utf-8") as f:
            return f.read()

    def test_successful_run_logs_ok_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            contents = self._run(d, self._canned, turns=2)
        self.assertIn("Result: OK", contents)
        self.assertRegex(contents, r"Run: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertIn("Models: m1 (m1) vs m2 (Two)", contents)
        self.assertIn("Turns: 2/2 (2 replies)", contents)
        self.assertIn("Model performance", contents)
        self.assertIn("Gen tok/s", contents)

    def test_failed_run_logs_error_not_stats(self):
        def boom(*args, **kwargs):
            raise ollama_duel.OllamaError("Cannot reach Ollama at http://x")

        with tempfile.TemporaryDirectory() as d:
            contents = self._run(d, boom, turns=3)
        self.assertIn("Result: STOPPED EARLY", contents)
        self.assertIn("Cannot reach Ollama at http://x", contents)
        self.assertIn("Turns: 0/3 (0 replies)", contents)
        self.assertNotIn("Model performance", contents)

    def test_entries_append_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, self._canned, turns=1)
            contents = self._run(d, self._canned, turns=1)
        self.assertEqual(contents.count("Result: OK"), 2)

    def test_default_filename_is_run_results_log(self):
        with tempfile.TemporaryDirectory() as d:
            with tempfile.TemporaryDirectory() as cfg_dir:
                path = write_json(cfg_dir, "cfg.json",
                                  minimal_config(turns=1))
                old = os.getcwd()
                os.chdir(d)
                try:
                    with mock.patch.object(
                            sys, "argv", ["ollama_duel.py", path]), \
                         mock.patch.object(ollama_duel, "call_chat",
                                           self._canned):
                        ollama_duel.main()
                finally:
                    os.chdir(old)
            self.assertTrue(
                os.path.isfile(os.path.join(d, "run_results.log")))

    def test_no_results_log_disables(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = minimal_config(
                turns=1,
                results_log=os.path.join(d, "run_results.log"))
            with tempfile.TemporaryDirectory() as cfg_dir:
                path = write_json(cfg_dir, "cfg.json", cfg)
                with mock.patch.object(
                        sys, "argv",
                        ["ollama_duel.py", path, "--no-results-log"]), \
                     mock.patch.object(ollama_duel, "call_chat",
                                       self._canned):
                    ollama_duel.main()
            self.assertFalse(
                os.path.exists(os.path.join(d, "run_results.log")))

    def test_cli_results_log_overrides_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "cfg_results.log")
            cli_path = os.path.join(d, "cli_results.log")
            cfg = minimal_config(turns=1, results_log=cfg_path)
            with tempfile.TemporaryDirectory() as cfg_dir:
                path = write_json(cfg_dir, "cfg.json", cfg)
                with mock.patch.object(
                        sys, "argv",
                        ["ollama_duel.py", path,
                         "--results-log", cli_path]), \
                     mock.patch.object(ollama_duel, "call_chat",
                                       self._canned):
                    ollama_duel.main()
            self.assertTrue(os.path.isfile(cli_path))
            self.assertFalse(os.path.exists(cfg_path))

    def test_unwritable_path_warns_but_does_not_crash(self):
        import io
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "blocker")
            with open(blocker, "w", encoding="utf-8") as f:
                f.write("x")
            cfg = minimal_config(
                turns=1,
                results_log=os.path.join(blocker, "run_results.log"))
            with tempfile.TemporaryDirectory() as cfg_dir:
                path = write_json(cfg_dir, "cfg.json", cfg)
                err = io.StringIO()
                with mock.patch.object(sys, "argv",
                                       ["ollama_duel.py", path]), \
                     mock.patch.object(ollama_duel, "call_chat",
                                       self._canned), \
                     mock.patch.object(sys, "stderr", err):
                    ollama_duel.main()  # must not raise
            self.assertIn("could not write run summary", err.getvalue())

    def test_results_log_wrong_type_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(results_log=123))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def test_crashed_run_logs_crashed(self):
        def bad(*args, **kwargs):
            raise ValueError("kaboom")

        with tempfile.TemporaryDirectory() as d:
            cfg = minimal_config(
                turns=2,
                results_log=os.path.join(d, "run_results.log"))
            with tempfile.TemporaryDirectory() as cfg_dir:
                path = write_json(cfg_dir, "cfg.json", cfg)
                with mock.patch.object(sys, "argv",
                                       ["ollama_duel.py", path]), \
                     mock.patch.object(ollama_duel, "call_chat", bad):
                    with self.assertRaises(ValueError):
                        ollama_duel.main()
            with open(os.path.join(d, "run_results.log"),
                      encoding="utf-8") as f:
                contents = f.read()
        self.assertIn("Result: CRASHED", contents)
        self.assertIn("kaboom", contents)


class NtfyTests(unittest.TestCase):
    def _home_with(self, files):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        for name, text in files.items():
            with open(os.path.join(d.name, name), "w",
                      encoding="utf-8") as f:
                f.write(text)
        return d

    def test_default_url_missing_file(self):
        d = self._home_with({})
        with mock.patch.dict(os.environ, {"HOME": d.name}):
            self.assertEqual(ollama_duel.load_default_ntfy_url(), "")

    def test_default_url_valid_file(self):
        d = self._home_with(
            {".dual.conf": json.dumps({"ntfy_url": "https://ntfy.sh/x"})})
        with mock.patch.dict(os.environ, {"HOME": d.name}):
            self.assertEqual(ollama_duel.load_default_ntfy_url(),
                             "https://ntfy.sh/x")

    def test_default_url_bad_files_treated_as_unset(self):
        for text in ('{not json', '[1, 2]', '{"ntfy_url": 42}',
                     '{"other": "x"}', '{"ntfy_url": "   "}'):
            d = self._home_with({".dual.conf": text})
            with mock.patch.dict(os.environ, {"HOME": d.name}):
                self.assertEqual(ollama_duel.load_default_ntfy_url(), "",
                                 f"for {text!r}")

    def test_config_accepts_ntfy_url(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(ntfy_url="https://ntfy.sh/x"))
            cfg = ollama_duel.load_config(path)
            self.assertEqual(cfg["ntfy_url"], "https://ntfy.sh/x")

    def test_config_rejects_non_string_ntfy_url(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, "cfg.json",
                              minimal_config(ntfy_url=123))
            with self.assertRaises(SystemExit):
                ollama_duel.load_config(path)

    def _participants(self):
        return [
            {"name": "Alice", "model": "m1"},
            {"name": "Bob", "model": "m2"},
        ]

    def test_notify_posts_title_and_body(self):
        import urllib.request
        import socket
        seen = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"ok"

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["title"] = req.get_header("Title")
            seen["data"] = req.data.decode("utf-8")
            return FakeResp()

        started = ollama_duel.datetime.now()
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            ollama_duel.notify_duel_done(
                "https://ntfy.sh/topic", "/scen/duel.json",
                self._participants(), 8, [(0, "hi"), (1, "yo")],
                started, None, False)
        self.assertEqual(seen["url"], "https://ntfy.sh/topic")
        self.assertEqual(seen["title"],
                         f"duel finished: duel.json [{socket.gethostname()}]")
        self.assertIn("m1 (Alice) vs m2 (Bob)", seen["data"])
        self.assertIn("2/8 turns", seen["data"])

    def test_notify_stopped_early_title(self):
        import urllib.request
        import socket

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"ok"

        seen = {}
        def fake_urlopen(req, timeout=None):
            seen["title"] = req.get_header("Title")
            return FakeResp()

        started = ollama_duel.datetime.now()
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            ollama_duel.notify_duel_done(
                "https://ntfy.sh/topic", "/scen/duel.json",
                self._participants(), 8, [(0, "hi")],
                started, "boom went the model", False)
        self.assertEqual(seen["title"],
                         f"duel stopped early: duel.json [{socket.gethostname()}]")

    def test_notify_includes_token_stats(self):
        import urllib.request

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"ok"

        seen = {}
        def fake_urlopen(req, timeout=None):
            seen["data"] = req.data.decode("utf-8")
            return FakeResp()

        stats = {
            "m1": {"turns": 2, "gen_tokens": 100, "gen_s": 50.0,
                   "prompt_tokens": 200, "prompt_s": 10.0, "truncated": 0},
            "m2": {"turns": 2, "gen_tokens": 80, "gen_s": 40.0,
                   "prompt_tokens": 190, "prompt_s": 9.5, "truncated": 1},
        }
        started = ollama_duel.datetime.now()
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            ollama_duel.notify_duel_done(
                "https://ntfy.sh/topic", "/scen/duel.json",
                self._participants(), 8, [(0, "hi"), (1, "yo")],
                started, None, False, model_stats=stats)
        self.assertIn("tokens: m1 100 tok @ 2.0 tok/s, "
                      "m2 80 tok @ 2.0 tok/s", seen["data"])

    def test_notify_without_stats_omits_token_line(self):
        import urllib.request

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"ok"

        seen = {}
        def fake_urlopen(req, timeout=None):
            seen["data"] = req.data.decode("utf-8")
            return FakeResp()

        started = ollama_duel.datetime.now()
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            ollama_duel.notify_duel_done(
                "https://ntfy.sh/topic", "/scen/duel.json",
                self._participants(), 8, [(0, "hi")],
                started, None, False)
        self.assertNotIn("tokens:", seen["data"])

    def test_notify_failure_warns_without_raising(self):
        import urllib.request
        started = ollama_duel.datetime.now()
        with mock.patch.object(urllib.request, "urlopen",
                               side_effect=OSError("nope")), \
             mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            ollama_duel.notify_duel_done(
                "https://ntfy.sh/topic", "/scen/duel.json",
                self._participants(), 8, [], started, None, False)
        self.assertIn("ntfy notification failed", err.getvalue())


class BatchModeTests(unittest.TestCase):
    def _make_dir(self, files):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        for f in files:
            open(os.path.join(d, f), "w").write("{}")
        return d

    def test_expand_plain_path_unchanged(self):
        self.assertEqual(ollama_duel.expand_configs("scenarios/duel.json"),
                         ["scenarios/duel.json"])

    def test_expand_directory_lists_json_sorted(self):
        d = self._make_dir(["b.json", "a.json", "notes.txt"])
        self.assertEqual(ollama_duel.expand_configs(d),
                         [os.path.join(d, "a.json"),
                          os.path.join(d, "b.json")])

    def test_expand_directory_empty_exits(self):
        d = self._make_dir(["notes.txt"])
        with self.assertRaises(SystemExit):
            ollama_duel.expand_configs(d)

    def test_expand_glob_matches_sorted(self):
        d = self._make_dir(["small_b.json", "small_a.json", "other.json"])
        pat = os.path.join(d, "small_*.json")
        self.assertEqual(ollama_duel.expand_configs(pat),
                         [os.path.join(d, "small_a.json"),
                          os.path.join(d, "small_b.json")])

    def test_expand_glob_no_match_exits(self):
        d = self._make_dir(["a.json"])
        with self.assertRaises(SystemExit):
            ollama_duel.expand_configs(os.path.join(d, "zzz_*.json"))

    def test_run_batch_reexecs_each_scenario(self):
        import subprocess
        calls = []

        class FakeCompleted:
            def __init__(self, rc): self.returncode = rc

        def fake_run(argv, **kw):
            calls.append(argv)
            return FakeCompleted(0)

        configs = ["scenarios/a.json", "scenarios/b.json"]
        with mock.patch.object(subprocess, "run", fake_run), \
             mock.patch.object(sys, "argv",
                               ["ollama_duel.py", "--turns", "8",
                                "scenarios/small_*"]), \
             mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = ollama_duel.run_batch(configs, "scenarios/small_*")
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        for argv, cfg in zip(calls, configs):
            self.assertEqual(argv[0], sys.executable)
            self.assertTrue(argv[1].endswith("ollama_duel.py"))
            self.assertEqual(argv[2], cfg)
            # CLI overrides are carried through; the glob itself is not
            self.assertEqual(argv[3:], ["--turns", "8"])

    def test_run_batch_counts_failures_and_continues(self):
        import subprocess
        calls = []

        class FakeCompleted:
            def __init__(self, rc): self.returncode = rc

        def fake_run(argv, **kw):
            calls.append(argv[2])
            return FakeCompleted(3 if "bad" in argv[2] else 0)

        configs = ["a.json", "bad.json", "c.json"]
        with mock.patch.object(subprocess, "run", fake_run), \
             mock.patch.object(sys, "argv", ["ollama_duel.py", "somedir"]), \
             mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = ollama_duel.run_batch(configs, "somedir")
        self.assertEqual(rc, 1)
        self.assertEqual(calls, configs)  # all three attempted

    def test_run_batch_ctrl_c_propagates(self):
        import subprocess

        def fake_run(argv, **kw):
            raise KeyboardInterrupt

        with mock.patch.object(subprocess, "run", fake_run), \
             mock.patch.object(sys, "argv", ["ollama_duel.py", "somedir"]), \
             mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(KeyboardInterrupt):
                ollama_duel.run_batch(["a.json", "b.json"], "somedir")

if __name__ == "__main__":
    unittest.main()
