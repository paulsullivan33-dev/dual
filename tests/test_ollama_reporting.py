"""Tests for ollama_reporting.ntfy_post (generic best-effort notifier)."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_reporting import ntfy_post


class NtfyPostTest(unittest.TestCase):
    def test_skips_silently_without_url(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            ntfy_post("", "t", "m")
            ntfy_post(None, "t", "m")
            urlopen.assert_not_called()

    def test_posts_title_tags_and_body(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            resp = mock.MagicMock()
            resp.__enter__.return_value = resp
            urlopen.return_value = resp
            ntfy_post("https://ntfy.sh/x", "Hello", "body text",
                      tags="thermometer")
            (req,), kwargs = urlopen.call_args
            self.assertEqual(req.full_url, "https://ntfy.sh/x")
            self.assertEqual(req.data, b"body text")
            self.assertEqual(req.get_header("Title"), "Hello")
            self.assertEqual(req.get_header("Tags"), "thermometer")
            self.assertEqual(kwargs["timeout"], 15)

    def test_omits_tags_header_when_empty(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            resp = mock.MagicMock()
            resp.__enter__.return_value = resp
            urlopen.return_value = resp
            ntfy_post("https://ntfy.sh/x", "Hello", "body")
            (req,), _ = urlopen.call_args
            self.assertIsNone(req.get_header("Tags"))

    def test_never_raises_on_network_error(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("net down")):
            ntfy_post("https://ntfy.sh/x", "t", "m")  # must not raise


if __name__ == "__main__":
    unittest.main()
