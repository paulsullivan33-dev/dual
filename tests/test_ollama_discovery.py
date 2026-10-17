"""Catalog and batch-preflight tests; no real network or generation."""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_discovery as discovery
import ollama_duel


class PreflightTests(unittest.TestCase):
    def _configs(self):
        return [("a.json", {"host": "http://box", "models": [{"model": "small"},
                              {"model": "small:latest"}],
                            "judge": {"model": "judge:8b", "host": "http://judge"}}),
                ("b.json", {"host": "http://box/", "models": [{"model": "small"}]})]

    def test_checks_each_server_once_and_accepts_implicit_latest(self):
        with mock.patch.object(discovery, "installed_models",
                               side_effect=[{"small:latest"}, {"judge:8b"}]) as inventory, \
             mock.patch("sys.stdout", io.StringIO()):
            discovery.preflight(self._configs())
        self.assertEqual(inventory.call_args_list,
                         [mock.call("http://box"), mock.call("http://judge")])

    def test_reports_all_missing_models_and_unreachable_servers(self):
        with mock.patch.object(discovery, "installed_models",
                               side_effect=[set(), OSError("offline")]):
            with self.assertRaises(ValueError) as error:
                discovery.preflight(self._configs())
        self.assertIn("missing model small", str(error.exception))
        self.assertIn("http://judge", str(error.exception))
        self.assertIn("offline", str(error.exception))

    def test_inventory_requires_valid_schema(self):
        for data in ([], {}, {"models": [None]}, {"models": [{}]}):
            with self.subTest(data=data), \
                 mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(data).encode())):
                with self.assertRaises(ValueError):
                    discovery.installed_models("http://box")


class DiscoveryCliTests(unittest.TestCase):
    def _write(self, directory, name, **values):
        cfg = {"topic": "Plan a workshop", "turns": 4, "tags": ["roleplay"],
               "models": [{"model": "original"}, {"model": "original"}]}
        cfg.update(values)
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(cfg, stream)
        return path

    def _main(self, argv):
        with mock.patch.object(sys, "argv", ["ollama_duel.py", *argv]):
            ollama_duel.main()

    def test_listing_applies_overrides_filters_and_never_calls_network(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "plan.json", run_code=True)
            self._write(directory, "other.json", tags=["game"])
            output = io.StringIO()
            with mock.patch("sys.stdout", output), mock.patch("urllib.request.urlopen") as network:
                self._main([directory, "--list-scenarios", "--tag", "roleplay",
                            "--turns", "2", "--model-a", "replacement", "--no-run-code"])
            network.assert_not_called()
        text = output.getvalue()
        self.assertIn("plan.json", text)
        self.assertNotIn("other.json", text)
        self.assertIn("Turns: 2", text)
        self.assertIn("Code execution: off", text)
        self.assertIn("replacement", text)

    def test_preflight_uses_profile_and_cli_judge_overrides_without_running(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "plan.json", judge={"model": "j", "host": "http://old"})
            profile = {"models": ["p1", "p2"], "settings": {"host": "http://profile"}}
            with mock.patch.object(ollama_duel, "load_profile", return_value=profile), \
                 mock.patch.object(ollama_duel, "preflight") as preflight, \
                 mock.patch.object(ollama_duel, "run_duel") as run:
                self._main([path, "--profile", "box", "--preflight", "--model-b", "cli",
                            "--host", "http://cli", "--judge-host", "http://judge"])
            cfg = preflight.call_args.args[0][0][1]
            self.assertEqual([m["model"] for m in cfg["models"]], ["p1", "cli"])
            self.assertEqual(cfg["host"], "http://cli")
            self.assertEqual(cfg["judge"]["host"], "http://judge")
            run.assert_not_called()

    def test_failed_batch_preflight_starts_no_children(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "a.json")
            self._write(directory, "b.json")
            with mock.patch.object(ollama_duel, "preflight", side_effect=ValueError("missing model")), \
                 mock.patch.object(ollama_duel, "run_batch") as batch:
                with self.assertRaisesRegex(SystemExit, "missing model"):
                    self._main([directory])
            batch.assert_not_called()

    def test_no_preflight_and_dry_run_batches_make_no_inventory_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "a.json")
            self._write(directory, "b.json")
            for flag in ("--no-preflight", "--dry-run"):
                with self.subTest(flag=flag), \
                     mock.patch.object(ollama_duel, "preflight") as preflight, \
                     mock.patch.object(ollama_duel, "run_batch", return_value=0):
                    with self.assertRaises(SystemExit) as error:
                        self._main([directory, flag])
                    self.assertEqual(error.exception.code, 0)
                    preflight.assert_not_called()

    def test_no_judge_omits_judge_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "a.json", judge="j")
            with mock.patch.object(ollama_duel, "preflight") as preflight:
                self._main([path, "--preflight", "--no-judge"])
            self.assertIsNone(preflight.call_args.args[0][0][1]["judge"])

    def test_invalid_scenarios_are_reported_together_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.json", "b.json"):
                self._write(directory, name, turns=0)
            with mock.patch.object(ollama_duel, "preflight") as preflight:
                with self.assertRaises(SystemExit) as error:
                    self._main([directory, "--preflight"])
            self.assertIn("a.json", str(error.exception))
            self.assertIn("b.json", str(error.exception))
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
