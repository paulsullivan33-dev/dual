"""Tests for ollama_coderun.py -- running the program in a duel reply.

These run real (tiny) Python programs in temporary folders.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_coderun as cr


def fenced(code, lang="python"):
    return f"```{lang}\n{code}\n```"


class ExtractProgramTests(unittest.TestCase):
    def test_takes_the_last_python_block(self):
        reply = ("A snippet:\n" + fenced("print('old')") +
                 "\nThe full program:\n" + fenced("print('new')"))
        self.assertEqual(cr.extract_program(reply).strip(), "print('new')")

    def test_accepts_py_python3_and_unlabelled_blocks(self):
        for lang in ("py", "python3", "Python", ""):
            with self.subTest(lang=lang):
                self.assertEqual(cr.extract_program(fenced("x = 1", lang)).strip(), "x = 1")

    def test_ignores_other_languages(self):
        reply = fenced("pip install requests", "bash") + "\n" + fenced('{"a": 1}', "json")
        self.assertIsNone(cr.extract_program(reply))

    def test_no_code_or_empty_block(self):
        self.assertIsNone(cr.extract_program("Just words, no code."))
        self.assertIsNone(cr.extract_program(fenced("   ")))
        self.assertIsNone(cr.extract_program(None))

    def test_windows_line_endings(self):
        self.assertEqual(cr.extract_program("```python\r\nprint(1)\r\n```").strip(), "print(1)")


class RunProgramTests(unittest.TestCase):
    def test_success_captures_output(self):
        r = cr.run_program("print('hello from the duel')")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["exit_code"], 0)
        self.assertIn("hello from the duel", r["output"])

    def test_failure_reports_exit_code_and_traceback(self):
        r = cr.run_program("def f(n):\n    return f(n + 1)\nf(0)")
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["exit_code"], 1)
        self.assertIn("RecursionError", r["output"])

    def test_assert_failures_count_as_failed(self):
        r = cr.run_program("assert 1 + 1 == 3, 'math is broken'")
        self.assertEqual(r["status"], "failed")
        self.assertIn("math is broken", r["output"])

    def test_timeout_stops_a_program_that_never_finishes(self):
        r = cr.run_program("import time\nprint('started', flush=True)\ntime.sleep(30)", timeout=1)
        self.assertEqual(r["status"], "timeout")
        self.assertIsNone(r["exit_code"])
        self.assertLess(r["seconds"], 15)

    def test_input_sees_end_of_file_instead_of_hanging(self):
        r = cr.run_program("name = input('Your name: ')", timeout=10)
        self.assertEqual(r["status"], "failed")
        self.assertIn("EOFError", r["output"])

    def test_arguments_are_passed(self):
        r = cr.run_program("import sys\nprint(sys.argv[1:])", args=["--test", "x"])
        self.assertIn("['--test', 'x']", r["output"])

    def test_runs_in_a_temporary_folder_that_is_removed(self):
        r = cr.run_program("import os\nopen('made.txt', 'w').close()\nprint(os.getcwd())")
        folder = r["output"].strip().splitlines()[-1]
        self.assertNotEqual(os.path.normcase(folder), os.path.normcase(os.getcwd()))
        self.assertFalse(os.path.exists(folder))

    def test_long_output_keeps_the_end(self):
        r = cr.run_program("for i in range(5000):\n    print('line', i)")
        self.assertLessEqual(len(r["output"]), cr.MAX_OUTPUT_CHARS + 50)
        self.assertIn("line 4999", r["output"])
        self.assertTrue(r["output"].startswith("...(earlier output cut)..."))


class ReportTests(unittest.TestCase):
    def test_reports_name_the_author_and_outcome(self):
        ok = cr.describe_run({"status": "ok", "exit_code": 0, "seconds": 0.2,
                              "output": "42", "timeout": 30}, "Builder")
        self.assertIn("Builder's program ran successfully", ok)
        self.assertIn("Output:\n42", ok)
        failed = cr.describe_run({"status": "failed", "exit_code": 1, "seconds": 0.1,
                                  "output": "", "timeout": 30}, "Breaker")
        self.assertIn("FAILED with exit code 1", failed)
        self.assertIn("It printed nothing.", failed)
        slow = cr.describe_run({"status": "timeout", "exit_code": None, "seconds": 30,
                                "output": "", "timeout": 30}, "Racer")
        self.assertIn("stopped after 30s", slow)

    def test_count_runs(self):
        line = cr.count_runs(["ok", "ok", "failed", "timeout", "no_code"])
        self.assertEqual(line, "Code runs: 2 ok, 1 failed, 1 timed out, "
                               "1 replies without a Python code block")


if __name__ == "__main__":
    unittest.main()
