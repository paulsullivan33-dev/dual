"""Keep the docs' command lines in step with the scripts' real options.

Options are read from each script's own argparse parser (captured when its
main() calls parse_args), so a renamed, removed or newly added option makes
these tests fail instead of leaving QUICK_REFERENCE.md or the README wrong.
"""

import argparse
import os
import re
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ollama_agent
import ollama_bench
import ollama_chat
import ollama_duel
import rag_demo

SCRIPTS = {
    "ollama_agent.py": ollama_agent,
    "ollama_bench.py": ollama_bench,
    "ollama_chat.py": ollama_chat,
    "ollama_duel.py": ollama_duel,
    "rag_demo.py": rag_demo,
}
DOCS = ["QUICK_REFERENCE.md", "README.md", "RAG_DEMO.md"]
LONG_OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")


class _Captured(Exception):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


def _capture_parser(module):
    """Run module.main() just far enough to get its ArgumentParser."""
    def fake_parse_args(self, *args, **kwargs):
        raise _Captured(self)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", fake_parse_args), \
         mock.patch.object(sys, "argv", ["prog"]):
        try:
            module.main()
        except _Captured as c:
            return c.parser
    raise AssertionError(f"{module.__name__}.main() never parsed arguments")


def _long_options(parser):
    """{subcommand or None: set of --options}. Top-level options are
    included under every subcommand as well as under None."""
    top, subs = set(), {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                subs[name] = {o for a in sub._actions for o in a.option_strings
                              if o.startswith("--")}
        else:
            top |= {o for o in action.option_strings if o.startswith("--")}
    options = {None: set(top)}
    for name, opts in subs.items():
        options[name] = opts | top
    return options


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def _commands(text, script):
    """Every documented command line that runs `script`, from code blocks
    and inline code alike. Stops at the end of the line or inline span."""
    return re.findall(r"python3? " + re.escape(script) + r"([^`\n]*)", text)


def _sections(text, script):
    """The parts of QUICK_REFERENCE.md whose '## ' heading names `script`."""
    parts = re.split(r"(?m)^## ", text)
    return "".join(p for p in parts if script in p.splitlines()[0])


class DocumentedOptionsExistTests(unittest.TestCase):
    """Every --option in a documented command must be accepted by that
    script (and by the subcommand used, where there is one)."""

    @classmethod
    def setUpClass(cls):
        cls.options = {s: _long_options(_capture_parser(m)) for s, m in SCRIPTS.items()}

    def test_documented_commands_use_real_options(self):
        for doc in DOCS:
            text = _read(doc)
            for script, options in self.options.items():
                for args in _commands(text, script):
                    words = args.split()
                    sub = next((w for w in words if w in options), None)
                    allowed = options[sub] if sub else set().union(*options.values())
                    for opt in LONG_OPTION.findall(args):
                        with self.subTest(doc=doc, command=f"{script}{args}", option=opt):
                            self.assertIn(opt, allowed)

    def test_rag_global_options_come_before_the_subcommand(self):
        # argparse rejects --db/--host/--embed-model after index/ask.
        global_opts = self.options["rag_demo.py"][None]
        for doc in DOCS:
            for args in _commands(_read(doc), "rag_demo.py"):
                words = args.split()
                subs = [i for i, w in enumerate(words) if w in ("index", "ask")]
                if not subs:
                    continue
                for w in words[subs[0]:]:
                    with self.subTest(doc=doc, command=f"rag_demo.py{args}"):
                        self.assertNotIn(w, global_opts)


class AllOptionsDocumentedTests(unittest.TestCase):
    """Every option a script accepts must appear in that script's sections
    of QUICK_REFERENCE.md, so new options don't go undocumented."""

    def test_quick_reference_covers_every_option(self):
        text = _read("QUICK_REFERENCE.md")
        for script, module in SCRIPTS.items():
            section = _sections(text, script)
            self.assertTrue(section, f"no QUICK_REFERENCE.md section for {script}")
            every = set().union(*_long_options(_capture_parser(module)).values())
            for opt in sorted(every - {"--help"}):  # argparse's own; no need to document
                with self.subTest(script=script, option=opt):
                    self.assertRegex(section, r"(?<![\w-])" + re.escape(opt) + r"(?![\w-])")


if __name__ == "__main__":
    unittest.main()
