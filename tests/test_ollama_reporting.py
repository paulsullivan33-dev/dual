"""Tests for ollama_reporting: the ntfy_post notifier and the
duration formatting used by the run summary and completion notice."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_reporting import (CLOUD_RATE_PRESETS, DEFAULT_CLOUD_RATE,
                              cloud_cost, format_cloud_cost_line, format_cost,
                              format_duration, format_run_summary,
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


def _speaker(name, model, index, prompt_tokens, gen_tokens):
    """One participant-stats record, as record_call accumulates it."""
    return {"name": name, "model": model, "index": index,
            "turns": 1, "calls": 1, "retries": 0, "truncated": 0,
            "gen_tokens": gen_tokens, "gen_s": 1.0,
            "prompt_tokens": prompt_tokens, "prompt_s": 0.5,
            "discarded_tokens": 0, "total_s": 1.5, "load_s": 0.0}


class CloudCostTest(unittest.TestCase):
    def _stats(self):
        return {
            "m1": {"participants": {0: _speaker("Alice", "m1", 0, 20000, 10000)}},
            "m2": {"participants": {1: _speaker("Bob", "m2", 1, 10000, 5000)}},
        }

    def test_presets_carry_the_published_rates(self):
        median = CLOUD_RATE_PRESETS["median"]
        self.assertEqual((median["input"], median["output"]), (0.435, 2.10))
        beam = CLOUD_RATE_PRESETS["beam"]
        self.assertEqual((beam["input"], beam["output"]), (0.125, 0.40))
        self.assertEqual(DEFAULT_CLOUD_RATE, "median")
        self.assertIn(DEFAULT_CLOUD_RATE, CLOUD_RATE_PRESETS)

    def test_cloud_cost_math(self):
        median = CLOUD_RATE_PRESETS["median"]
        # A million tokens each way is exactly the published rate sum.
        self.assertAlmostEqual(cloud_cost(1_000_000, 1_000_000, median), 2.535)
        self.assertEqual(cloud_cost(0, 0, median), 0.0)
        # Alice's fixture usage: 20k in, 10k out.
        self.assertAlmostEqual(cloud_cost(20000, 10000, median), 0.0297)

    def test_format_cost(self):
        self.assertEqual(format_cost(0), "$0")
        self.assertEqual(format_cost(2.5), "$2.50")
        self.assertEqual(format_cost(1500.0), "$1,500.00")
        # Below a dollar: three significant figures, never $0.03-style
        # rounding that hides a sub-cent share entirely.
        self.assertEqual(format_cost(0.0321), "$0.0321")
        self.assertEqual(format_cost(0.0000297), "$0.0000297")

    def test_line_at_median_rates(self):
        line = format_cloud_cost_line(self._stats())
        self.assertTrue(line.startswith(
            "Cloud-equivalent cost at median cloud rates "
            "($0.435 in / $2.10 out per 1M tokens): "))
        self.assertIn("Alice ≈ $0.0297", line)
        self.assertIn("Bob ≈ $0.0149", line)
        self.assertIn("total ≈ $0.0445", line)
        self.assertTrue(line.endswith("— actual cost $0 (local)."))

    def test_line_at_beam_rates(self):
        line = format_cloud_cost_line(self._stats(), "beam")
        self.assertIn("Beam cloud rates ($0.125 in / $0.40 out per 1M tokens)",
                      line)
        self.assertIn("Alice ≈ $0.00650", line)

    def test_line_without_participants_falls_back_to_models(self):
        stats = {"m1": {"turns": 1, "gen_tokens": 10000, "gen_s": 1.0,
                        "prompt_tokens": 20000, "prompt_s": 0.5,
                        "truncated": 0}}
        self.assertIn("m1 ≈ $0.0297", format_cloud_cost_line(stats))

    def test_shared_name_gets_seat_numbers(self):
        stats = {"m1": {"participants": {
            0: _speaker("Same", "m1", 0, 20000, 10000),
            1: _speaker("Same", "m1", 1, 20000, 10000)}}}
        line = format_cloud_cost_line(stats)
        self.assertIn("1: Same ≈ $0.0297", line)
        self.assertIn("2: Same ≈ $0.0297", line)

    def test_empty_stats_return_empty_string(self):
        self.assertEqual(format_cloud_cost_line({}), "")

    def test_run_summary_carries_the_line_when_given_one(self):
        line = format_cloud_cost_line(self._stats())
        summary = format_run_summary(datetime.now(), "scen.json", PARTICIPANTS,
                                     2, [object()] * 2, {}, None,
                                     cloud_cost_line=line)
        self.assertIn(line, summary)
        plain = format_run_summary(datetime.now(), "scen.json", PARTICIPANTS,
                                   2, [object()] * 2, {}, None)
        self.assertNotIn("Cloud-equivalent", plain)


if __name__ == "__main__":
    unittest.main()
