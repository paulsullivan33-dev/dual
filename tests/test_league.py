"""Tests for ollama_league: fixtures, judging math, standings.

The duel and rejudge subprocesses are faked: the fake duel runner
writes a transcript and appends a canned (but real-shaped) results
block, and rejudge_winner is patched to return a canned winner name.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama_league as league

BLOCK = """Run: 2026-11-11 22:14:19
Config: scenarios/small_coffee_vs_tea.json
Models: qwen3:4b (Barista Bruno) vs qwen3:8b (Tea Master Lin)
Turns: 4/4 (4 replies) in 30s
Participant stats: [{"turns": 2, "gen_tokens": 300, "gen_s": 1.0, "prompt_tokens": 1200, "prompt_s": 0.5, "truncated": 0, "calls": 2, "retries": 0, "discarded_tokens": 0, "total_s": 0.0, "load_s": 0.0, "name": "Barista Bruno", "model": "qwen3:4b", "index": 0}, {"turns": 2, "gen_tokens": 250, "gen_s": 1.0, "prompt_tokens": 1100, "prompt_s": 0.5, "truncated": 1, "calls": 2, "retries": 0, "discarded_tokens": 0, "total_s": 0.0, "load_s": 0.0, "name": "Tea Master Lin", "model": "qwen3:8b", "index": 1}]
Result: OK
------------------------------------------------------------------------
"""


def make_cfg(tmp, **over):
    cfg = {"name": "Test Season", "host": "http://localhost:11434",
           "contestants": ["qwen3:4b", "qwen3:8b", "mistral:7b"],
           "scenarios": ["scenarios/small_coffee_vs_tea.json"],
           "judge": {"mode": "single", "model": "qwen3:14b"},
           "output_dir": tmp}
    cfg.update(over)
    return cfg


class HelperTests(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(league.slug("qwen3:8b"), "qwen3-8b")
        self.assertEqual(league.slug("huihui_ai/qwen3-abliterated:32b"),
                         "huihui-ai-qwen3-abliterated-32b")

    def test_family_of(self):
        self.assertEqual(league.family_of("qwen3:14b"), "alibaba")
        self.assertEqual(league.family_of("huihui_ai/qwen3-abliterated:32b"),
                         "alibaba")
        self.assertEqual(league.family_of("gemma4:latest"), "google")
        self.assertEqual(league.family_of("granite3.3:8b"), "ibm")
        self.assertEqual(league.family_of("mystery:1b"), "unknown")

    def test_majority(self):
        self.assertEqual(league.majority(["a", "a", "b"]), "a")
        self.assertIsNone(league.majority(["a", "b", "c"]))
        self.assertIsNone(league.majority(["a", "b"]))
        self.assertIsNone(league.majority([None, None]))
        self.assertEqual(league.majority([None, "b", "b"]), "b")

    def test_parse_winner_name(self):
        self.assertEqual(
            league.parse_winner_name("Judge: x\nWinner: Barista Bruno\n"),
            "Barista Bruno")
        self.assertIsNone(league.parse_winner_name("Winner: (none parsed)"))
        self.assertIsNone(league.parse_winner_name("no winner here"))

    def test_winner_model(self):
        models = [("Barista Bruno", "qwen3:4b"), ("Tea Master Lin", "qwen3:8b")]
        self.assertEqual(league.winner_model("Tea Master Lin", models),
                         "qwen3:8b")
        self.assertEqual(league.winner_model("qwen3:4b", models), "qwen3:4b")
        self.assertIsNone(league.winner_model("Somebody Else", models))
        self.assertIsNone(league.winner_model(None, models))


class FixtureTests(unittest.TestCase):
    def test_round_robin_fixtures(self):
        cfg = make_cfg("unused")
        fixtures = league.build_fixtures(cfg, 2)
        # 3 contestants -> 3 pairings, x 1 scenario x 2 rounds
        self.assertEqual(len(fixtures), 6)
        self.assertEqual(fixtures[0]["a"], "qwen3:4b")
        self.assertEqual(fixtures[0]["b"], "qwen3:8b")
        self.assertTrue(fixtures[0]["id"].startswith("r1-small_coffee_vs_tea-"))
        self.assertTrue(fixtures[-1]["id"].startswith("r2-"))
        pairs = {(f["a"], f["b"]) for f in fixtures if f["round"] == 1}
        self.assertEqual(len(pairs), 3)


class StandingsTests(unittest.TestCase):
    def record(self, a="qwen3:4b", b="qwen3:8b", winner="qwen3:4b"):
        return {"id": "x", "round": 1, "scenario": "s.json", "a": a, "b": b,
                "date": "2026-11-11 22:00", "status": "played",
                "winner": winner, "judge": "j",
                "tokens": {a: {"in": 100, "out": 40},
                           b: {"in": 90, "out": 35}}, "note": ""}

    def test_apply_win(self):
        state = {"table": league.empty_table(["qwen3:4b", "qwen3:8b"]),
                 "matches": []}
        league.apply_result(state, self.record())
        self.assertEqual(state["table"]["qwen3:4b"]["points"], 3)
        self.assertEqual(state["table"]["qwen3:4b"]["won"], 1)
        self.assertEqual(state["table"]["qwen3:8b"]["lost"], 1)
        self.assertEqual(state["table"]["qwen3:4b"]["tokens_in"], 100)

    def test_apply_draw(self):
        state = {"table": league.empty_table(["qwen3:4b", "qwen3:8b"]),
                 "matches": []}
        league.apply_result(state, self.record(winner=None))
        self.assertEqual(state["table"]["qwen3:4b"]["points"], 1)
        self.assertEqual(state["table"]["qwen3:8b"]["drawn"], 1)

    def test_sorted_table_and_render(self):
        cfg = make_cfg("unused")
        state = {"league": "Test Season", "updated": "now",
                 "table": league.empty_table(cfg["contestants"]),
                 "matches": []}
        rec = self.record()
        state["matches"].append(rec)
        league.apply_result(state, rec)
        order = [m for m, _ in league.sorted_table(state)]
        self.assertEqual(order[0], "qwen3:4b")
        md = league.render_standings(cfg, state)
        self.assertIn("# Test Season — standings", md)
        self.assertIn("| 1 | qwen3:4b | 1 | 1 | 0 | 0 | 3 |", md)
        self.assertIn("qwen3:4b vs qwen3:8b → **qwen3:4b**", md)


class StateTests(unittest.TestCase):
    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            state = league.load_state(cfg)
            self.assertEqual(state["matches"], [])
            state["matches"].append({"id": "m1", "status": "played"})
            league.save_state(cfg, state)
            again = league.load_state(cfg)
            self.assertEqual(league.played_ids(again), {"m1"})
            self.assertTrue(again["updated"])


class PlayMatchTests(unittest.TestCase):
    def fake_runner(self, cmd):
        save_idx = cmd.index("--save-json") + 1
        log_idx = cmd.index("--results-log") + 1
        with open(cmd[save_idx], "w", encoding="utf-8") as fh:
            json.dump([{"speaker": "Barista Bruno", "model": "qwen3:4b",
                        "text": "Espresso is civilization."}], fh)
        with open(cmd[log_idx], "a", encoding="utf-8") as fh:
            fh.write(BLOCK)
        return mock.Mock(returncode=0, stdout="", stderr="")

    def test_play_and_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            fixture = league.build_fixtures(cfg, 1)[0]
            with mock.patch.object(league, "rejudge_winner",
                                   return_value=("Tea Master Lin", "")):
                record = league.play_match(cfg, fixture,
                                           runner=self.fake_runner)
            self.assertEqual(record["status"], "played")
            self.assertEqual(record["winner"], "qwen3:8b")
            self.assertEqual(record["tokens"]["qwen3:4b"],
                             {"in": 1200, "out": 300})
            self.assertEqual(record["tokens"]["qwen3:8b"],
                             {"in": 1100, "out": 250})

    def test_participant_pairs_come_from_stats(self):
        # Regression: the results block's "models" list is (model, role)
        # ordered, which once inverted the winner mapping live. Pairs
        # must come from the Participant stats' explicit name/model.
        from summarize_results import parse_file
        with tempfile.NamedTemporaryFile("w", suffix=".log",
                                         delete=False) as fh:
            fh.write(BLOCK)
            path = fh.name
        try:
            block = parse_file(path)[0]
        finally:
            os.unlink(path)
        pairs = league.participant_pairs(block)
        self.assertEqual(pairs, [("Barista Bruno", "qwen3:4b"),
                                 ("Tea Master Lin", "qwen3:8b")])
        self.assertEqual(league.winner_model("Barista Bruno", pairs),
                         "qwen3:4b")

    def test_failed_duel_is_unplayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            fixture = league.build_fixtures(cfg, 1)[0]
            record = league.play_match(
                cfg, fixture,
                runner=lambda cmd: mock.Mock(returncode=1, stdout="",
                                             stderr="boom"))
            self.assertEqual(record["status"], "unplayed")
            self.assertIn("exited 1", record["note"])


if __name__ == "__main__":
    unittest.main()
