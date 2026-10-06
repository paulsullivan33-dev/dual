"""ollama_coderun.py -- optionally run the Python program in each duel reply.

Off unless a scenario sets "run_code": true or the run passes --run-code.
After each reply, the last Python code block is saved in a fresh temporary
folder and run with this same Python interpreter, a timeout, and no
keyboard input (input() sees end-of-file instead of hanging). The result is
written to the log and shown to both speakers on their next turns, so a
program that crashes gets noticed and fixed instead of drifting along.

This is NOT a sandbox: the program runs with your user's permissions and
can read and write files and use the network like any script you run.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_RUN_TIMEOUT = 30   # seconds per program run
MAX_OUTPUT_CHARS = 2000    # output kept per run (the end, where errors are)

# Opening fence: ``` plus an optional language label. Blocks labelled
# python/python3/py, or unlabelled, count as Python; others (```bash,
# ```json, ...) are skipped.
_FENCE = re.compile(r"^\s*```\s*([\w+.-]*)\s*$")
_PYTHON_LABELS = {"", "python", "python3", "py"}


def extract_program(reply):
    """The last Python code block in a reply, or None when there isn't one.

    The last block is the one that matters: replies often show a snippet
    first and the complete program after it. Fences are paired line by
    line, so a closing ``` is never mistaken for the opening of a new block.
    """
    blocks, current, label = [], None, ""
    for line in (reply or "").splitlines():
        fence = _FENCE.match(line)
        if current is None:
            if fence:
                current, label = [], fence.group(1).lower()
        elif fence and not fence.group(1):
            if label in _PYTHON_LABELS and "".join(current).strip():
                blocks.append("\n".join(current) + "\n")
            current = None
        else:
            current.append(line)
    return blocks[-1] if blocks else None


def _tail(text, limit=MAX_OUTPUT_CHARS):
    text = (text or "").rstrip()
    if len(text) <= limit:
        return text
    return "...(earlier output cut)...\n" + text[-limit:]


def _as_text(data):
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data or ""


def run_program(code, args=(), timeout=DEFAULT_RUN_TIMEOUT):
    """Run `code` as program.py in a new temporary folder and clean up.

    Returns a dict: status ("ok", "failed" or "timeout"), exit_code (None
    on timeout), seconds, and output (stdout and stderr combined, trimmed
    to the last MAX_OUTPUT_CHARS characters).
    """
    folder = tempfile.mkdtemp(prefix="duel_run_")
    try:
        path = os.path.join(folder, "program.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, path, *args], cwd=folder,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            status = "ok" if proc.returncode == 0 else "failed"
            exit_code, output = proc.returncode, _as_text(proc.stdout)
        except subprocess.TimeoutExpired as e:
            status, exit_code, output = "timeout", None, _as_text(e.output)
        seconds = time.monotonic() - start
    finally:
        # Best-effort: on Windows a lingering child can hold a file open.
        shutil.rmtree(folder, ignore_errors=True)
    return {"status": status, "exit_code": exit_code, "seconds": seconds,
            "output": _tail(output), "timeout": timeout}


def describe_run(result, author):
    """Plain-language report of a run, used both in the log and as the
    message that shows the speakers what happened."""
    secs = result["seconds"]
    if result["status"] == "ok":
        head = f"{author}'s program ran successfully (exit code 0, {secs:.1f}s)."
    elif result["status"] == "failed":
        head = (f"{author}'s program FAILED with exit code "
                f"{result['exit_code']} after {secs:.1f}s.")
    else:
        head = (f"{author}'s program was stopped after {result['timeout']:g}s "
                f"without finishing. It may be waiting for keyboard input "
                f"(none is available) or looping forever.")
    output = result["output"]
    body = f"Output:\n{output}" if output else "It printed nothing."
    return f"[Automatic code run] {head}\n{body}"


def count_runs(results):
    """One-line tally of a duel's code runs, e.g. for the end of the log."""
    counts = {"ok": 0, "failed": 0, "timeout": 0, "no_code": 0}
    for r in results:
        counts[r] += 1
    return (f"Code runs: {counts['ok']} ok, {counts['failed']} failed, "
            f"{counts['timeout']} timed out, {counts['no_code']} replies "
            f"without a Python code block")
