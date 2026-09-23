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


if __name__ == "__main__":
    unittest.main()
