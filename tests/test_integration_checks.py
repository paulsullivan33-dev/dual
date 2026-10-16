"""Test the live-check orchestration without contacting servers or hardware."""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import integration_checks as checks
from ollama_common import OllamaError


class IntegrationCheckTests(unittest.TestCase):
    def test_no_implicit_live_checks(self):
        with mock.patch.object(checks, "check_ollama") as ollama, \
             mock.patch.object(checks, "check_display") as display, \
             mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                checks.main([])
        ollama.assert_not_called()
        display.assert_not_called()

    def test_failure_is_nonzero_and_does_not_skip_other_selected_check(self):
        with mock.patch.object(checks, "check_ollama", side_effect=OllamaError("offline")), \
             mock.patch.object(checks, "check_display") as display, \
             mock.patch("sys.stderr", io.StringIO()), mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(checks.main(["--ollama", "--display"]), 1)
        display.assert_called_once_with()

    def test_ollama_selection_and_overrides(self):
        with mock.patch.object(checks, "check_ollama") as ollama, \
             mock.patch.object(checks, "check_display") as display, \
             mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(checks.main(["--ollama", "--host", "http://judge:11434",
                                          "--model", "llama3.1:8b", "--timeout", "60"]), 0)
        ollama.assert_called_once_with("http://judge:11434", "llama3.1:8b", 60)
        display.assert_not_called()

    def test_empty_ollama_reply_fails(self):
        with mock.patch.object(checks, "call_chat", return_value=("", "", "length", {})):
            with self.assertRaises(OllamaError):
                checks.check_ollama("http://x", "m", 30)

    def test_display_failure_still_clears_and_disconnects(self):
        import unoq_matrix
        matrix = mock.Mock()
        matrix.show_text.side_effect = RuntimeError("write failed")
        with mock.patch.object(unoq_matrix, "BridgeMatrix", return_value=matrix):
            with self.assertRaises(RuntimeError):
                checks.check_display()
        matrix.clear.assert_called_once_with()
        matrix._bridge.disconnect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
