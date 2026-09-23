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


if __name__ == "__main__":
    unittest.main()
