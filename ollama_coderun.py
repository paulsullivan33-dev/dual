"""Optionally run reply code in a disposable, restricted Docker container.

Requires Docker and a locally provisioned Python image (no implicit pulls).
The container has no network or host mounts, a read-only root, a bounded
temporary directory, and CPU/memory/PID limits. Output is streamed into a
bounded tail. No fallback ever executes generated code on the host.
"""

import os
import re
import math
import shutil
import subprocess
import threading
import uuid
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


# A provisioned image is required. Operators may select a trusted digest.
DEFAULT_RUN_IMAGE = "python:3.12-slim"
MAX_OUTPUT_BYTES = 8192
MAX_CODE_BYTES = 256 * 1024
_BOOTSTRAP = ("import sys; source=sys.stdin.read(); "
              "sys.argv[0]='program.py'; exec(compile(source,'program.py','exec'))")


def _container_command(name, args):
    docker = shutil.which("docker")
    if not docker:
        raise OSError("Docker is required for code execution; use --no-run-code to skip it")
    image = os.environ.get("DUAL_RUN_CODE_IMAGE", DEFAULT_RUN_IMAGE)
    if not image or image.startswith("-"):
        raise ValueError("DUAL_RUN_CODE_IMAGE must name a trusted, locally provisioned image")
    return [docker, "run", "--name", name, "--rm", "--pull=never",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64",
            "--memory", "128m", "--memory-swap", "128m", "--cpus", "1",
            "--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=16m",
            "--workdir", "/tmp", "--log-driver", "none", "-i", image,
            "python", "-I", "-u", "-c", _BOOTSTRAP, *args]


def _remove_container(docker, name):
    # Killing the client alone does not kill its container or descendants.
    try:
        result = subprocess.run([docker, "rm", "--force", name],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=5, check=False)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_program(code, args=(), timeout=DEFAULT_RUN_TIMEOUT):
    """Run code only in a restricted container; return a bounded result."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("code timeout must be finite and positive")
    if not isinstance(code, str) or len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("code must be text of at most 256 KiB")
    started = time.monotonic()
    name = "duel-run-" + uuid.uuid4().hex
    status, exit_code, output = "failed", 127, ""
    process, reader, command = None, None, None
    tail = bytearray()
    def drain():
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                return
            tail.extend(chunk)
            if len(tail) > MAX_OUTPUT_BYTES:
                del tail[:-MAX_OUTPUT_BYTES]
    try:
        command = _container_command(name, args)
        # A pipe/file supplies the source bytes, never a bind mount or host path.
        # The container consumes stdin to EOF before executing the program.
        with tempfile.TemporaryFile() as source:
            source.write(code.encode("utf-8"))
            source.seek(0)
            process = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            try:
                exit_code = process.wait(timeout=timeout)
                status = "ok" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                status, exit_code = "timeout", None
                process.kill()
                process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        output = "Isolated code runner unavailable: " + str(error)
    finally:
        if command is not None:
            cleaned = _remove_container(command[0], name)
            if status == "timeout" and not cleaned:
                output += "\nContainer cleanup was not acknowledged; inspect " + name
        if reader is not None:
            reader.join(timeout=2)
        if process is not None and process.stdout is not None and not (reader and reader.is_alive()):
            process.stdout.close()
    output = _tail(tail.decode("utf-8", errors="replace")) + output
    return {"status": status, "exit_code": exit_code,
            "seconds": time.monotonic() - started, "output": output,
            "timeout": timeout}


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
