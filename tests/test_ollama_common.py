import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_common as oc


class StripThinkTagsTests(unittest.TestCase):
    def test_no_tags(self):
        self.assertEqual(oc.strip_think_tags("hello"), "hello")

    def test_complete_block(self):
        self.assertEqual(
            oc.strip_think_tags("<think>reasoning</think>hello"), "hello")

    def test_complete_block_case_insensitive(self):
        self.assertEqual(
            oc.strip_think_tags("<THINK>reasoning</THINK>hello"), "hello")

    def test_multiple_blocks(self):
        self.assertEqual(
            oc.strip_think_tags("<think>a</think>hello<think>b</think>world"),
            "helloworld")

    def test_missing_open_tag(self):
        # Some models emit a stray closing tag with no opening tag.
        self.assertEqual(
            oc.strip_think_tags("reasoning leaked</think>hello"), "hello")

    def test_missing_close_tag(self):
        # A stray opening tag with no close: cut from it to the end.
        self.assertEqual(
            oc.strip_think_tags("hello<think>reasoning never closes"), "hello")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(oc.strip_think_tags("  hello  "), "hello")


class CallChatTests(unittest.TestCase):
    def _mock_response(self, payload):
        body = json.dumps(payload).encode()
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(body)
        cm.__exit__.return_value = False
        return cm

    def test_success(self):
        payload = {"message": {"content": "<think>hmm</think>hi", "thinking": "hmm"},
                   "done_reason": "stop"}
        with mock.patch("urllib.request.urlopen", return_value=self._mock_response(payload)):
            thinking, reply, done_reason = oc.call_chat(
                "http://x", "m", [{"role": "user", "content": "hi"}],
                True, {"num_predict": 10})
        self.assertEqual(thinking, "hmm")
        self.assertEqual(reply, "hi")
        self.assertEqual(done_reason, "stop")

    def test_missing_thinking_field_defaults_empty(self):
        payload = {"message": {"content": "hi"}}
        with mock.patch("urllib.request.urlopen", return_value=self._mock_response(payload)):
            thinking, reply, done_reason = oc.call_chat("http://x", "m", [], False, {})
        self.assertEqual(thinking, "")
        self.assertEqual(reply, "hi")
        self.assertEqual(done_reason, "")

    def test_done_reason_length_signals_truncation(self):
        payload = {"message": {"content": "hi"}, "done_reason": "length"}
        with mock.patch("urllib.request.urlopen", return_value=self._mock_response(payload)):
            _, _, done_reason = oc.call_chat("http://x", "m", [], False,
                                             {"num_predict": 10})
        self.assertEqual(done_reason, "length")

    def test_http_error_raises_ollama_error(self):
        err = urllib.error.HTTPError(
            "http://x/api/chat", 404, "not found", {}, io.BytesIO(b"model missing"))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(oc.OllamaError) as ctx:
                oc.call_chat("http://x", "missing-model", [], False, {})
        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertIn("missing-model", str(ctx.exception))

    def test_url_error_raises_ollama_error(self):
        err = urllib.error.URLError("connection refused")
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(oc.OllamaError) as ctx:
                oc.call_chat("http://x", "m", [], False, {})
        self.assertIn("Cannot reach Ollama", str(ctx.exception))


class TranscriptJsonTests(unittest.TestCase):
    def test_build_duel_json(self):
        participants = [
            {"name": "A", "model": "m1"},
            {"name": "B", "model": "m2"},
        ]
        transcript = [(0, "hi"), (1, "hello"), (0, "bye")]
        self.assertEqual(oc.build_duel_json(transcript, participants), [
            {"speaker": "A", "model": "m1", "text": "hi"},
            {"speaker": "B", "model": "m2", "text": "hello"},
            {"speaker": "A", "model": "m1", "text": "bye"},
        ])

    def test_save_and_reload(self):
        turns = [{"speaker": "A", "model": "m1", "text": "hi"}]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            oc.save_transcript_json(path, turns)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), turns)

    def test_save_safe_noop_without_path(self):
        # Should not raise even though there's nothing to write to.
        oc.save_transcript_json_safe(None, [{"speaker": "A", "model": "m", "text": "x"}])

    def test_save_safe_reports_error_without_raising(self):
        with tempfile.TemporaryDirectory() as d:
            bad_path = os.path.join(d, "missing-dir", "out.json")
            with mock.patch("sys.stderr", new_callable=io.StringIO) as fake_err:
                oc.save_transcript_json_safe(bad_path, [])
            self.assertIn("Could not save transcript", fake_err.getvalue())


if __name__ == "__main__":
    unittest.main()
