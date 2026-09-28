"""ollama_agent.py -- a minimal coding agent on top of Ollama.

You describe a programming task, the model writes files, and this script
saves them into a project directory. Then it loops: describe a change,
the model rewrites, you approve, repeat. Think of it as the smallest
possible version of tools like Cline or Copilot Workspace.

How the model hands you files: it writes each complete file in its own
fenced code block, tagged with the language AND the file's path:

    ```python:hello.py
    print("hello")
    ```

Only fences tagged like ```lang:path become files. A plain ```python
fence with no path is treated as an illustrative snippet and ignored.

Safety rules (deliberate, not accidental):
  * Every file lands inside --output-dir. Paths with ".." or absolute
    paths are rejected, so a rogue reply can't write outside the project.
  * Nothing is ever executed. The model can write code, but this script
    will not run it. Running what it wrote is your job, in your terminal.
  * Every write is previewed and confirmed unless you pass --yes.

Example:
    python ollama_agent.py --task "a python script that renames photos in a folder by date taken"
    python ollama_agent.py --model qwen2.5-coder:14b --task "a flappy-bird clone in pygame" --output-dir ./flappy
"""

import argparse
import os
import re
import sys

import ollama_common
from ollama_common import (
    DEFAULT_HOST,
    DEFAULT_TIMEOUT,
    OllamaError,
    call_chat,
    setup_utf8_stdout,
    wrap_text,
)

# A fence that carries a file looks like ```python:src/main.py --
# language, colon, relative path. Plain ```python fences (no colon+path)
# are snippets, not files, and are ignored by extract_files().
FILE_FENCE_RE = re.compile(r"^```[\w+.-]*:([^\s`]+)\s*$")

SYSTEM_PROMPT = """\
You are a coding assistant. A scaffold program extracts files from your
replies and saves them into a project directory.

Rules:
- Write each file you create or change as a COMPLETE file in its own
  fenced code block.
- Tag the fence with the language and the file's relative path, like:
  ```python:hello.py
- Paths are relative to the project directory. Never use absolute paths
  or "..".
- Keep the prose outside the fences brief: what you built and how to run it.
- If no files are needed, just answer in prose.
"""

DONE_WORDS = {"done", "quit", "exit", "q"}
MAX_CONTEXT_CHARS = 4000  # per-file cap when sending project files back


def extract_files(text):
    """Pull (relative_path, content) pairs out of ```lang:path fences.

    Only fences tagged with a colon+path become files; plain fences are
    ignored. An unclosed fence is ignored too (better to drop a file
    than to write half of one).
    """
    files = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = FILE_FENCE_RE.match(lines[i])
        if match:
            rel_path = match.group(1)
            i += 1
            body = []
            closed = False
            while i < len(lines):
                if lines[i].strip() == "```":
                    closed = True
                    break
                body.append(lines[i])
                i += 1
            if closed:
                content = "\n".join(body)
                files.append((rel_path, content + "\n" if content else ""))
        i += 1
    return files


def resolve_path(output_dir, rel_path):
    """Return the absolute target path if rel_path stays inside output_dir.

    Returns None for absolute paths, ".." escapes, or anything that would
    land outside the project directory -- those writes are refused.
    """
    if os.path.isabs(rel_path):
        return None
    base = os.path.abspath(output_dir)
    target = os.path.abspath(os.path.join(base, rel_path))
    if target != base and target.startswith(base + os.sep):
        return target
    return None


def snapshot_files(output_dir):
    """Read current project files as {rel_path: content}.

    Binary-looking or oversized files are skipped silently; contents are
    capped so a big project doesn't blow up the model's context window.
    """
    snapshot = {}
    base = os.path.abspath(output_dir)
    for root, _dirs, names in os.walk(base):
        for name in sorted(names):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, base)
            try:
                if os.path.getsize(full) > 200_000:
                    continue
                with open(full, encoding="utf-8", errors="strict") as f:
                    content = f.read(MAX_CONTEXT_CHARS)
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable: leave it out of context
            snapshot[rel] = content
    return snapshot


def format_snapshot(snapshot):
    """Render current files as a context block for the model."""
    if not snapshot:
        return "The project directory is currently empty."
    parts = ["Current project files:"]
    for rel in sorted(snapshot):
        parts.append(f"\n--- {rel} ---\n{snapshot[rel]}")
    return "\n".join(parts)


def confirm_write(files, auto_yes, input_fn):
    """Preview extracted files and ask before writing. Returns True to write."""
    for rel_path, content in files:
        line_count = len(content.splitlines())
        print(f"\n--- {rel_path} ({line_count} lines) ---")
        preview = content if len(content) <= 2000 else content[:2000] + "\n... [truncated]"
        print(preview)
    if auto_yes:
        return True
    answer = input_fn(f"Write {len(files)} file(s)? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def write_files(output_dir, files):
    """Write files into output_dir. Returns (written, rejected) rel paths."""
    written, rejected = [], []
    for rel_path, content in files:
        target = resolve_path(output_dir, rel_path)
        if target is None:
            rejected.append(rel_path)
            continue
        os.makedirs(os.path.dirname(target) or output_dir, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(rel_path)
    return written, rejected


def run_agent(args, input_fn=input, call_fn=call_chat):
    """Main loop. input_fn/call_fn are injectable so tests can drive it."""
    setup_utf8_stdout()
    os.makedirs(args.output_dir, exist_ok=True)

    options = {}
    if args.max_tokens:
        options["num_predict"] = args.max_tokens
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.num_ctx:
        options["num_ctx"] = args.num_ctx

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": args.task + "\n\n" + format_snapshot(snapshot_files(args.output_dir))},
    ]

    while True:
        try:
            _thinking, reply, done_reason, metrics = call_fn(
                args.host, args.model, messages, False, options, timeout=args.timeout)
        except OllamaError as e:
            print(f"\nOllama error: {e}", file=sys.stderr)
            return 1
        print(f"\n{'=' * 72}\n[agent reply]\n{'=' * 72}")
        print(wrap_text(reply))
        if done_reason == "length":
            print("\n[warning: reply was cut off at the token limit; "
                  "files may be incomplete]")
        print(f"\n[{metrics['gen_tps']:.1f} tok/s, {metrics['gen_tokens']} tokens]")

        extracted = extract_files(reply)
        if extracted:
            if confirm_write(extracted, args.yes, input_fn):
                written, rejected = write_files(args.output_dir, extracted)
                for rel in written:
                    print(f"wrote {os.path.join(args.output_dir, rel)}")
                for rel in rejected:
                    print(f"refused unsafe path: {rel}", file=sys.stderr)
            else:
                print("skipped writing.")
        else:
            print("\n[no files in this reply]")

        followup = input_fn("\nDescribe a change, or 'done' to finish: ").strip()
        if followup.lower() in DONE_WORDS or not followup:
            print("Done. Project files are in", os.path.abspath(args.output_dir))
            return 0
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content":
                         followup + "\n\n" + format_snapshot(snapshot_files(args.output_dir))})


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Minimal coding agent: describe a task, approve the files it writes, iterate.")
    ap.add_argument("--task", required=True, help="what to build, in plain words")
    ap.add_argument("--model", default="qwen2.5-coder:14b",
                    help="Ollama model to use (default: qwen2.5-coder:14b)")
    ap.add_argument("--output-dir", default="./agent_out",
                    help="project directory for written files (default: ./agent_out)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="max tokens per reply (default: 4096)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="context window size override")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="seconds to wait per reply")
    ap.add_argument("--yes", action="store_true",
                    help="write files without asking (use with care)")
    args = ap.parse_args(argv)
    return run_agent(args)


if __name__ == "__main__":
    sys.exit(main())
