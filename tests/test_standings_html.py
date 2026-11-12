"""Tests for standings.html rendering (league-native and converter)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama_league as league
import standings_html

CFG = {"name": "Test League",
       "judge": {"mode": "panel", "panel": "test-panel"}}

STATE = {
    "league": "Test League",
    "updated": "2026-11-12 04:00:00 CST",
    "table": {
        "qwen3:4b": {"played": 2, "won": 1, "drawn": 0, "lost": 1,
                     "points": 3, "tokens_in": 1000, "tokens_out": 2000},
        "mistral:7b": {"played": 2, "won": 2, "drawn": 0, "lost": 0,
                       "points": 6, "tokens_in": 900, "tokens_out": 1500},
        "evil<model>": {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                        "points": 0, "tokens_in": 0, "tokens_out": 0},
    },
    "matches": [
        {"id": "m1", "status": "played", "date": "2026-11-12",
         "scenario": "small_coffee_vs_tea", "a": "qwen3:4b",
         "b": "mistral:7b", "winner": "mistral:7b",
         "judge": "mistral:7b: Barista Bruno"},
        {"id": "m2", "status": "failed", "note": "duel exited 1"},
    ],
}


class LeagueHtmlTests(unittest.TestCase):
    def test_table_sorted_and_escaped(self):
        page = league.render_standings_html(CFG, STATE)
        self.assertLess(page.index("mistral:7b"), page.index("qwen3:4b"))
        self.assertIn("evil&lt;model&gt;", page)
        self.assertNotIn("evil<model>", page)
        self.assertIn("<td>6</td>", page)
        self.assertIn("Matches (1 played)", page)
        self.assertIn("Unplayed (failed; retried next run)", page)

    def test_converter_parity(self):
        """The standalone converter must agree with the league's own
        table order and cost figures."""
        page = standings_html.render_html(STATE)
        self.assertLess(page.index("mistral:7b"), page.index("qwen3:4b"))
        row = STATE["table"]["mistral:7b"]
        self.assertEqual(
            standings_html.format_cost(standings_html.cloud_cost(
                row["tokens_in"], row["tokens_out"])),
            league.format_cost(league.cloud_cost(
                row["tokens_in"], row["tokens_out"],
                league.CLOUD_RATE_PRESETS["median"])))

    def test_convert_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "standings.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(STATE, fh)
            out = standings_html.convert(tmp)
            self.assertTrue(os.path.exists(out))
            with open(out, encoding="utf-8") as fh:
                self.assertIn("Test League — standings", fh.read())


if __name__ == "__main__":
    unittest.main()
