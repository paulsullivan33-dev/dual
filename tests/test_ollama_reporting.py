"""Tests for ollama_reporting: the ntfy_post notifier and the
duration formatting used by the run summary and completion notice."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_reporting import (format_duration, format_run_summary,
                              notify_duel_done, ntfy_post)


class FormatDurationTest(unittest.TestCase):
    def test_under_a_minute_is_seconds_only(self):
        self.assertEqual(format_duration(0), "0s")
        self.assertEqual(format_duration(45.4), "45s")
        self.assertEqual(format_duration(59.6), "60s (1m 00s)")

    def test_minutes_get_a_parenthetical(self):
        self.assertEqual(format_duration(1865), "1865s (31m 05s)")
        self.assertEqual(format_duration(600), "600s (10m 00s)")

    def test_hours_get_a_parenthetical(self):
        self.assertEqual(format_duration(15403), "15403s (4h 16m 43s)")
        self.assertEqual(format_duration(3600), "3600s (1h 00m 00s)")


PARTICIPANTS = [{"model": "m1", "name": "One"}, {"model": "m2", "name": "Two"}]


class TurnsLineDurationTest(unittest.TestCase):
    """Both places a duel reports its elapsed time -- the run_results
    summary block and the ntfy notice -- show minutes next to seconds."""

    def test_run_summary_turns_line(self):
        started = datetime.now() - timedelta(seconds=1865)
        summary = format_run_summary(started, "scen.json", PARTICIPANTS, 8,
                                     [object()] * 8, {}, None)
        self.assertIn("Turns: 8/8 (8 replies) in 1865s (31m 05s)", summary)

    def test_ntfy_notice_turns_line(self):
        started = datetime.now() - timedelta(seconds=1865)
        with mock.patch("ollama_reporting.ntfy_post") as post:
            notify_duel_done("https://ntfy.sh/x", "scen.json", PARTICIPANTS,
                             8, [object()] * 8, started, None, False)
        body = post.call_args[0][2]
        self.assertIn("8/8 turns in 1865s (31m 05s)", body)


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
