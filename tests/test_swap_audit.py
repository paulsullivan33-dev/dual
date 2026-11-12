"""Tests for judge_swap_audit: the label swap and verdict math.

The rejudge subprocess is not exercised here; the swap construction,
recorded-verdict parsing, and classification are pure functions.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import judge_swap_audit as audit

ENTRIES = [
    {"speaker": "Barista Bruno", "model": "qwen3:4b",
     "text": "Coffee is civilization."},
    {"speaker": "Tea Master Lin", "model": "qwen3:8b",
     "text": "Tea is serenity."},
    {"speaker": "Barista Bruno", "model": "qwen3:4b",
     "text": "Espresso forever."},
]
PAIRS = [("Barista Bruno", "qwen3:4b"), ("Tea Master Lin", "qwen3:8b")]

SCEN = {"topic": "Bruno defends coffee, Lin defends tea.",
        "models": [
            {"name": "Barista Bruno", "model": "qwen3:1.7b",
             "system": "You are Barista Bruno, a coffee evangelist."},
            {"name": "Tea Master Lin", "model": "smollm2:1.7b",
             "system": "You are Tea Master Lin, a tea devotee."}],
        "judge": {"model": "qwen3:14b"}}


class SwapTests(unittest.TestCase):
    def test_pairs_first_appearance(self):
        self.assertEqual(audit.transcript_pairs(ENTRIES), PAIRS)

    def test_swapped_transcript_moves_labels_only(self):
        out = audit.swapped_transcript(ENTRIES, PAIRS)
        self.assertEqual(out[0]["speaker"], "Tea Master Lin")
        self.assertEqual(out[0]["model"], "qwen3:8b")
        self.assertEqual(out[0]["text"], "Coffee is civilization.")
        self.assertEqual(out[1]["speaker"], "Barista Bruno")
        self.assertEqual(out[2]["speaker"], "Tea Master Lin")
        # Original untouched.
        self.assertEqual(ENTRIES[0]["speaker"], "Barista Bruno")

    def test_swapped_scenario(self):
        out = audit.swapped_scenario(SCEN, PAIRS)
        self.assertEqual(out["topic"],
                         "Lin defends coffee, Bruno defends tea.")
        self.assertEqual(out["models"][0]["name"], "Barista Bruno")
        self.assertIn("You are Barista Bruno, a tea devotee.",
                      out["models"][0]["system"])
        self.assertIn("You are Tea Master Lin, a coffee evangelist.",
                      out["models"][1]["system"])

    def test_swap_names_substring_safe(self):
        self.assertEqual(audit.swap_names("Lin met Lin", "Lin", "Linnea"),
                         "Linnea met Linnea")


class ScoringTests(unittest.TestCase):
    def test_parse_recorded(self):
        seats = audit.parse_recorded(
            "gemma4:latest: draw; mistral:7b: Barista Bruno; "
            "granite3.3:8b: Tea Master Lin (same family as a contestant)")
        self.assertEqual(seats, {"gemma4:latest": None,
                                 "mistral:7b": "Barista Bruno",
                                 "granite3.3:8b": "Tea Master Lin"})

    def test_classify(self):
        self.assertEqual(audit.classify("Barista Bruno", "Tea Master Lin",
                                        PAIRS), "content")
        self.assertEqual(audit.classify("Barista Bruno", "Barista Bruno",
                                        PAIRS), "label")
        self.assertEqual(audit.classify("Barista Bruno", None, PAIRS),
                         "unresolved")
        self.assertEqual(audit.classify(None, "Tea Master Lin", PAIRS),
                         "unresolved")

    def test_majority_name(self):
        self.assertEqual(audit.majority_name(
            ["Barista Bruno", "Barista Bruno", None]), "Barista Bruno")
        self.assertIsNone(audit.majority_name(
            ["Barista Bruno", "Tea Master Lin", None]))
        self.assertIsNone(audit.majority_name([None, None]))


if __name__ == "__main__":
    unittest.main()
