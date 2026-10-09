import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_agent
from ollama_common import OllamaError


class ExtractFilesTests(unittest.TestCase):
    def test_single_file_with_lang_and_path(self):
        text = 'Here you go:\n```python:hello.py\nprint("hi")\n```\nDone.'
        self.assertEqual(ollama_agent.extract_files(text),
                         [("hello.py", 'print("hi")\n')])

    def test_multiple_files(self):
        text = "```python:a.py\n1\n```\n```js:src/b.js\n2\n```"
        self.assertEqual(ollama_agent.extract_files(text),
                         [("a.py", "1\n"), ("src/b.js", "2\n")])

    def test_plain_fence_without_path_is_ignored(self):
        text = "```python\nprint('snippet')\n```\n```text:real.txt\nx\n```"
        self.assertEqual(ollama_agent.extract_files(text), [("real.txt", "x\n")])

    def test_path_without_lang_is_accepted(self):
        text = "```:notes.txt\nhello\n```"
        self.assertEqual(ollama_agent.extract_files(text), [("notes.txt", "hello\n")])

    def test_unclosed_fence_is_ignored(self):
        text = "```python:half.py\nprint('oops')"
        self.assertEqual(ollama_agent.extract_files(text), [])

    def test_no_fences(self):
        self.assertEqual(ollama_agent.extract_files("just prose"), [])

    def test_multiline_content_preserved(self):
        text = "```python:app.py\ndef f():\n    return 1\n\n\nx = f()\n```"
        self.assertEqual(ollama_agent.extract_files(text),
                         [("app.py", "def f():\n    return 1\n\n\nx = f()\n")])


class ResolvePathTests(unittest.TestCase):
    def test_simple_relative_ok(self):
        with tempfile.TemporaryDirectory() as d:
            target = ollama_agent.resolve_path(d, "hello.py")
            self.assertEqual(target, os.path.join(os.path.abspath(d), "hello.py"))

    def test_nested_relative_ok(self):
        with tempfile.TemporaryDirectory() as d:
            target = ollama_agent.resolve_path(d, "src/main.py")
            self.assertTrue(target.endswith(os.path.join("src", "main.py")))

    def test_parent_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ollama_agent.resolve_path(d, "../evil.py"))

    def test_nested_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ollama_agent.resolve_path(d, "a/../../evil.py"))

    def test_absolute_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ollama_agent.resolve_path(d, "/tmp/evil.py"))


class WriteFilesTests(unittest.TestCase):
    def test_writes_and_creates_subdirs(self):
        with tempfile.TemporaryDirectory() as d:
            written, rejected = ollama_agent.write_files(
                d, [("src/a.py", "x=1\n"), ("b.txt", "hi\n")])
            self.assertEqual(sorted(written), ["b.txt", "src/a.py"])
            self.assertEqual(rejected, [])
            with open(os.path.join(d, "src", "a.py"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "x=1\n")

    def test_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            written, rejected = ollama_agent.write_files(
                d, [("../evil.py", "x"), ("ok.py", "y\n")])
            self.assertEqual(written, ["ok.py"])
            self.assertEqual(rejected, ["../evil.py"])
            self.assertFalse(os.path.exists(os.path.join(d, "..", "evil.py")))


class ConfirmWriteTests(unittest.TestCase):
    def test_auto_yes_skips_prompt(self):
        input_fn = mock.Mock(side_effect=AssertionError("should not prompt"))
        self.assertTrue(ollama_agent.confirm_write([("a.py", "x")], True, input_fn))

    def test_yes_writes(self):
        self.assertTrue(ollama_agent.confirm_write([("a.py", "x")], False, lambda _p: "y"))

    def test_no_skips(self):
        self.assertFalse(ollama_agent.confirm_write([("a.py", "x")], False, lambda _p: "n"))

    def test_empty_defaults_to_no(self):
        self.assertFalse(ollama_agent.confirm_write([("a.py", "x")], False, lambda _p: ""))


def _args(**overrides):
    import argparse
    base = {"task": "build a thing", "model": "m", "output_dir": ".", "host": "h",
            "max_tokens": 100, "temperature": None, "num_ctx": None,
            "timeout": 60, "yes": False}
    base.update(overrides)
    return argparse.Namespace(**base)


def _reply(text):
    metrics = {"gen_tps": 10.0, "gen_tokens": 5}
    return "", text, "stop", metrics


class RunAgentTests(unittest.TestCase):
    def test_full_loop_writes_file_then_quits(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []

            def fake_call(host, model, messages, think, options, timeout=None):
                calls.append(messages[-1]["content"])
                return _reply("```python:hello.py\nprint('hi')\n```")

            inputs = iter(["y", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d), input_fn=lambda _p: next(inputs),
                call_fn=fake_call)
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "hello.py"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "print('hi')\n")
            self.assertEqual(len(calls), 1)  # one model call before "done"

    def test_followup_sends_second_turn_with_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []

            def fake_call(host, model, messages, think, options, timeout=None):
                seen.append(messages[-1]["content"])
                return _reply("```python:hello.py\nprint('v2')\n```")

            inputs = iter(["y", "make it v2", "y", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d), input_fn=lambda _p: next(inputs),
                call_fn=fake_call)
            self.assertEqual(rc, 0)
            self.assertEqual(len(seen), 2)
            self.assertIn("make it v2", seen[1])
            self.assertIn("hello.py", seen[1])  # snapshot of current files
            with open(os.path.join(d, "hello.py"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "print('v2')\n")

    def test_declined_write_keeps_looping(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_call(host, model, messages, think, options, timeout=None):
                return _reply("```python:a.py\n1\n```")

            inputs = iter(["n", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d), input_fn=lambda _p: next(inputs),
                call_fn=fake_call)
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(os.path.join(d, "a.py")))

    def test_ollama_error_returns_1(self):
        def fake_call(host, model, messages, think, options, timeout=None):
            raise OllamaError("boom")

        with tempfile.TemporaryDirectory() as d:
            rc = ollama_agent.run_agent(
                _args(output_dir=d), input_fn=lambda _p: "done", call_fn=fake_call)
            self.assertEqual(rc, 1)


class ExtractUntaggedFencesTests(unittest.TestCase):
    def test_single_untagged_fence(self):
        text = "```python\nprint('hi')\n```"
        self.assertEqual(ollama_agent.extract_untagged_fences(text),
                         [("python", "print('hi')\n")])

    def test_bare_fence_has_empty_lang(self):
        text = "```\nhello\n```"
        self.assertEqual(ollama_agent.extract_untagged_fences(text),
                         [("", "hello\n")])

    def test_tagged_fences_are_excluded(self):
        text = "```python:a.py\n1\n```\n```js\n2\n```"
        self.assertEqual(ollama_agent.extract_untagged_fences(text),
                         [("js", "2\n")])

    def test_unclosed_untagged_ignored(self):
        self.assertEqual(
            ollama_agent.extract_untagged_fences("```python\nx=1"), [])

    def test_no_fences(self):
        self.assertEqual(ollama_agent.extract_untagged_fences("just prose"), [])

    def test_extract_files_still_ignores_untagged(self):
        text = "```python\nprint('snippet')\n```\n```text:real.txt\nx\n```"
        self.assertEqual(ollama_agent.extract_files(text),
                         [("real.txt", "x\n")])


class SuggestFilenameTests(unittest.TestCase):
    def test_python_extension_and_slug(self):
        name = ollama_agent.suggest_filename("Build a sudoku solver", "python", 0)
        self.assertEqual(name, "build_a_sudoku_solver.py")

    def test_unknown_lang_gets_txt(self):
        name = ollama_agent.suggest_filename("do things", "cobol", 0)
        self.assertTrue(name.endswith(".txt"))

    def test_index_suffix_on_later_blocks(self):
        first = ollama_agent.suggest_filename("task", "python", 0)
        second = ollama_agent.suggest_filename("task", "python", 1)
        self.assertEqual(first, "task.py")
        self.assertEqual(second, "task_2.py")

    def test_empty_task_falls_back_to_output(self):
        self.assertEqual(ollama_agent.suggest_filename("", "python", 0),
                         "output.py")

    def test_long_task_is_truncated(self):
        name = ollama_agent.suggest_filename("a" * 100, "python", 0)
        self.assertLessEqual(len(name), len("a" * 30 + ".py"))


class RunAgentUntaggedTests(unittest.TestCase):
    def test_untagged_block_prompts_for_name_then_writes(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_call(host, model, messages, think, options, timeout=None):
                return _reply("```python\nprint('hi')\n```")

            inputs = iter(["sudoku.py", "y", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d, task="build a sudoku solver"),
                input_fn=lambda _p: next(inputs), call_fn=fake_call)
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "sudoku.py"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "print('hi')\n")

    def test_untagged_block_uses_default_on_empty_answer(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_call(host, model, messages, think, options, timeout=None):
                return _reply("```python\nprint('hi')\n```")

            inputs = iter(["", "y", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d, task="build a thing"),
                input_fn=lambda _p: next(inputs), call_fn=fake_call)
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "build_a_thing.py")))

    def test_untagged_block_skip_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_call(host, model, messages, think, options, timeout=None):
                return _reply("```python\nprint('hi')\n```")

            inputs = iter(["skip", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d), input_fn=lambda _p: next(inputs),
                call_fn=fake_call)
            self.assertEqual(rc, 0)
            self.assertEqual(os.listdir(d), [])

    def test_untagged_block_auto_yes_uses_suggested_name(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_call(host, model, messages, think, options, timeout=None):
                return _reply("```python\nprint('hi')\n```")

            rc = ollama_agent.run_agent(
                _args(output_dir=d, task="build a thing", yes=True),
                input_fn=lambda _p: "done", call_fn=fake_call)
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "build_a_thing.py")))

    def test_unsafe_typed_name_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_call(host, model, messages, think, options, timeout=None):
                return _reply("```python\nprint('hi')\n```")

            inputs = iter(["../evil.py", "y", "done"])
            rc = ollama_agent.run_agent(
                _args(output_dir=d), input_fn=lambda _p: next(inputs),
                call_fn=fake_call)
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(os.path.join(d, "..", "evil.py")))
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()

class FormatSnapshotTests(unittest.TestCase):
    def test_empty_mapping_uses_empty_directory_message(self):
        self.assertEqual(
            ollama_agent.format_snapshot({}),
            "The project directory is currently empty.",
        )

    def test_sorts_paths_and_preserves_empty_file_content(self):
        self.assertEqual(
            ollama_agent.format_snapshot({"z.txt": "last", "a.txt": ""}),
            "Current project files:\n\n--- a.txt ---\n\n\n--- z.txt ---\nlast",
        )

    def test_non_mapping_input_is_rejected_by_mapping_lookup(self):
        with self.assertRaises(TypeError):
            ollama_agent.format_snapshot(["file.txt"])
