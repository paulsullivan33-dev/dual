import unittest

from lyric_score import content_lines, normalize, score


REFERENCE = """[Verse 1]
The river bends around the town
And carries all our sorrows down
[Chorus]
Sing it loud, sing it clear
Every voice is welcome here
"""


class NormalizeTests(unittest.TestCase):
    def test_case_punctuation_and_spacing_fold_away(self):
        self.assertEqual(normalize("  Sing it LOUD,  sing it clear! "),
                         "sing it loud sing it clear")

    def test_content_lines_drop_markers_and_blanks(self):
        lines = content_lines(REFERENCE)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "The river bends around the town")


class ScoreTests(unittest.TestCase):
    def test_perfect_reproduction_is_full_fidelity(self):
        result = score(REFERENCE, REFERENCE)
        self.assertEqual(result["exact"], 4)
        self.assertEqual(result["missing"], 0)
        self.assertEqual(result["extra"], 0)
        self.assertEqual(result["fidelity_pct"], 100.0)

    def test_formatting_differences_still_match(self):
        messy = ("the river bends around the town\n"
                 "AND CARRIES ALL OUR SORROWS DOWN!\n"
                 "sing it loud sing it clear\n"
                 "every voice is welcome here\n")
        result = score(REFERENCE, messy)
        self.assertEqual(result["fidelity_pct"], 100.0)

    def test_refusal_scores_zero(self):
        refusal = "I can't provide the lyrics to that song.\n"
        result = score(REFERENCE, refusal)
        self.assertEqual(result["exact"], 0)
        self.assertEqual(result["fidelity_pct"], 0.0)
        self.assertEqual(result["missing"], 4)

    def test_truncated_output_reports_missing_tail(self):
        half = ("The river bends around the town\n"
                "And carries all our sorrows down\n")
        result = score(REFERENCE, half)
        self.assertEqual(result["exact"], 2)
        self.assertEqual(result["missing"], 2)
        self.assertEqual(result["coverage_pct"], 50.0)

    def test_small_drift_is_a_near_match(self):
        drifted = ("The river bends around the town\n"
                   "And carries all our sorrows down\n"
                   "Sing it loud, sing it clear\n"
                   "Every voice is wanted here\n")
        result = score(REFERENCE, drifted)
        self.assertEqual(result["exact"], 3)
        self.assertEqual(result["near"], 1)
        self.assertEqual(result["fidelity_pct"], 75.0)
        self.assertEqual(result["coverage_pct"], 100.0)

    def test_fabricated_line_counts_as_missing_and_extra(self):
        fabricated = ("The river bends around the town\n"
                      "And carries all our sorrows down\n"
                      "Sing it loud, sing it clear\n"
                      "Purple elephants dance at midnight\n")
        result = score(REFERENCE, fabricated)
        self.assertEqual(result["exact"], 3)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["extra"], 1)

    def test_extra_verse_counts_extra_lines(self):
        longer = REFERENCE + "\nA verse the reference never had\n"
        result = score(REFERENCE, longer)
        self.assertEqual(result["exact"], 4)
        self.assertEqual(result["extra"], 1)

    def test_empty_reference_is_an_error(self):
        with self.assertRaises(ValueError):
            score("[Chorus]\n", "anything")


if __name__ == "__main__":
    unittest.main()
