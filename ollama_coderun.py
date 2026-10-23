"""Run trusted reply code natively by default, or opt into Docker isolation.

Both runners bound elapsed time and captured output. Native Linux runs also
bound per-process address space and CPU time; they still have user-account
file/network access. Docker is optional and never falls back to native.
"""

import os
import re
import math
import signal
import sys
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
DEFAULT_MEMORY_MB = 128
MAX_OUTPUT_BYTES = 8192
MAX_CODE_BYTES = 256 * 1024
_BOOTSTRAP = ("import sys; source=sys.stdin.read(); "
              "sys.argv[0]='program.py'; exec(compile(source,'program.py','exec'))")
_NATIVE_BOOTSTRAP = (
    "import runpy; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); "
    "sys.stderr.reconfigure(encoding='utf-8', errors='replace'); "
    "sys.argv[0]='program.py'; sys.path.insert(0, '.'); "
    "runpy.run_path('program.py', run_name='__main__')"
)


def _container_command(name, args, memory_mb=DEFAULT_MEMORY_MB):
    docker = shutil.which("docker")
    if not docker:
        raise OSError("Docker runner selected but Docker is unavailable; select native or use --no-run-code")
    image = os.environ.get("DUAL_RUN_CODE_IMAGE", DEFAULT_RUN_IMAGE)
    if not image or image.startswith("-"):
        raise ValueError("DUAL_RUN_CODE_IMAGE must name a trusted, locally provisioned image")
    return [docker, "run", "--name", name, "--rm", "--pull=never",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m", "--cpus", "1",
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


def _native_command(args, timeout, memory_mb):
    # Apply limits in the child, never in the LLM/duel parent and never via
    # preexec_fn (unsafe with the runner's output-reader threads).
    limits = (
        "import sys\n"
        "if sys.platform.startswith('linux'):\n"
        " import resource\n"
        f" for kind, requested in ((resource.RLIMIT_AS, {memory_mb * 1024 * 1024}), "
        f"(resource.RLIMIT_CPU, {max(1, math.ceil(timeout))})):\n"
        "  _, hard = resource.getrlimit(kind)\n"
        "  bound = requested if hard == resource.RLIM_INFINITY else min(requested, hard)\n"
        "  resource.setrlimit(kind, (bound, bound))\n"
    )
    return [sys.executable, "-I", "-u", "-c", limits + _NATIVE_BOOTSTRAP, *args]


def _stop_native(process):
    """Reap ordinary descendants on POSIX; best-effort tree cleanup on Windows."""
    if os.name == "posix":
        try:
            # POSIX-only members are absent from Windows lint environments.
            os.killpg(process.pid, signal.SIGKILL)  # pylint: disable=no-member
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def run_program(code, args=(), timeout=DEFAULT_RUN_TIMEOUT,
                backend="native", memory_mb=DEFAULT_MEMORY_MB):
    """Native is for trusted scenarios; Docker adds explicit isolation."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("code timeout must be finite and positive")
    if backend not in ("native", "docker"):
        raise ValueError("code runner must be native or docker")
    if isinstance(memory_mb, bool) or not isinstance(memory_mb, int) or memory_mb < 32:
        raise ValueError("code memory limit must be an integer of at least 32 MiB")
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
        command = (_container_command(name, args, memory_mb) if backend == "docker"
                   else _native_command(args, timeout, memory_mb))
        # Native code gets a disposable working directory, not filesystem
        # isolation. Docker receives source over stdin and no host mounts.
        with tempfile.TemporaryDirectory(prefix="duel_run_") as folder, tempfile.TemporaryFile() as source:
            if backend == "native":
                with open(os.path.join(folder, "program.py"), "w", encoding="utf-8") as program:
                    program.write(code)
                standard_input = subprocess.DEVNULL
            else:
                source.write(code.encode("utf-8"))
                source.seek(0)
                standard_input = source
            options = {"start_new_session": True} if os.name == "posix" else {}
            process = subprocess.Popen(command, stdin=standard_input, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, cwd=folder, **options)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            try:
                try:
                    exit_code = process.wait(timeout=timeout)
                    status = "ok" if exit_code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status, exit_code = "timeout", None
            finally:
                # Also clean up on KeyboardInterrupt and when the parent exits
                # while an ordinary child is still holding its output pipe.
                if backend == "native":
                    _stop_native(process)
                else:
                    cleaned = _remove_container(command[0], name)
                    if status == "timeout" and not cleaned:
                        output += "\nContainer cleanup was not acknowledged; inspect " + name
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                if reader is not None:
                    reader.join(timeout=2)
                if process.stdout is not None and not reader.is_alive():
                    process.stdout.close()
    except (OSError, subprocess.TimeoutExpired) as error:
        status, exit_code = "failed", 127
        output += f"{backend} code runner unavailable: {error}"
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
