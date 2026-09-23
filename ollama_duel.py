#!/usr/bin/env python3
"""
ollama_duel.py -- two AI models converse with each other, configured from JSON.

Usage:
    python3 ollama_duel.py duel.json
    python3 ollama_duel.py duel.json --turns 10 --topic "A different question"

Globals in the JSON (host, topic, turns, think, max_tokens, temperature,
num_ctx, log_file) act as defaults; anything set inside a "models" entry
overrides the global for that model only. "models" must contain exactly 2 entries.
If "log_file" is set, everything printed to stdout is mirrored to that
file (appended, with session start/end markers).
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime

DEFAULT_HOST = "http://localhost:11434"


class Tee:
    """Write to multiple streams at once (used to mirror stdout to a log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def load_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Invalid JSON in {path}: {e}")
    if not isinstance(cfg, dict):
        sys.exit(f"Config {path} must be a JSON object.")
    models = cfg.get("models")
    if not isinstance(models, list) or len(models) != 2:
        sys.exit('Config must define exactly 2 models in the "models" list.')
    for m in models:
        if not isinstance(m, dict) or "model" not in m:
            sys.exit('Each model entry needs a "model" key, e.g. "qwen3:4b".')
        m.setdefault("name", m["model"])
    return cfg


def first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def strip_think_tags(text):
    """Remove <think>...</think> blocks a model may emit inside the reply text.

    Some models (notably Qwen) leak their reasoning into the content even when
    thinking is disabled via the API, sometimes with no opening <think> tag.
    Handles complete blocks, a closing tag with no opening tag (strip
    everything through it), and a stray opening tag with no close (cut from
    it to the end).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def call_chat(host, model, messages, think, options, timeout=900):
    """Single non-streaming /api/chat call. Returns (thinking, reply)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": options,
    }
    req = urllib.request.Request(
        host + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code} from Ollama: {body}\nHint: did you run `ollama pull {model}`?")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach Ollama at {host}: {e.reason}\nIs `ollama serve` running?")
    msg = data["message"]
    return msg.get("thinking", "").strip(), strip_think_tags(msg["content"])


def print_turn(label, turn_no, thinking, reply, show_thinking):
    """Print one turn with clearly separated, labeled Thinking and Reply blocks."""
    bar = "=" * 72
    print(bar)
    print(f"{label}  (turn {turn_no})")
    print(bar)
    if show_thinking:
        print("--- THINKING ---")
        if thinking:
            for line in thinking.splitlines():
                print("    " + line)
        else:
            print("    (no thinking returned)")
        print()
    print("--- REPLY ---")
    print(reply)
    print()


def main():
    ap = argparse.ArgumentParser(description="Two Ollama models converse, configured from a JSON file.")
    ap.add_argument("config", help="path to JSON config file")
    ap.add_argument("--topic", default=None, help="override the config topic")
    ap.add_argument("--turns", type=int, default=None, help="override the config turn count")
    ap.add_argument("--host", default=None, help="override the config host")
    ap.add_argument("--think", dest="think", action="store_true", default=None,
                    help="force thinking display on")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="force thinking display off")
    ap.add_argument("--log-file", default=None,
                    help="override the config log_file (mirror stdout to this file)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = first_not_none(args.host, cfg.get("host"), DEFAULT_HOST)
    topic = first_not_none(args.topic, cfg.get("topic"))
    if not topic:
        sys.exit('No topic: set "topic" in the config or pass --topic.')
    turns = first_not_none(args.turns, cfg.get("turns"), 6)

    # Optional log file: everything printed to stdout is mirrored there.
    # (stderr progress lines like "(waiting for reply...)" stay console-only.)
    log_path = first_not_none(args.log_file, cfg.get("log_file"))
    log_fh = None
    if log_path:
        try:
            log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
        except OSError as e:
            sys.exit(f"Cannot open log file {log_path}: {e}")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fh.write(f"\n--- session started {stamp} ---\n")
        sys.stdout = Tee(sys.stdout, log_fh)

    # Resolve per-model settings: model entry wins, then globals, then default.
    participants = []
    for entry in cfg["models"]:
        think = first_not_none(args.think, entry.get("think"), cfg.get("think"), False)
        max_tokens = first_not_none(
            entry.get("max_tokens"), cfg.get("max_tokens"),
            2048 if think else 300,  # thinking eats the same token budget
        )
        temperature = first_not_none(entry.get("temperature"), cfg.get("temperature"))
        num_ctx = first_not_none(entry.get("num_ctx"), cfg.get("num_ctx"))
        options = {"num_predict": max_tokens}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if temperature is not None:
            options["temperature"] = temperature
        participants.append({
            "name": entry["name"],
            "model": entry["model"],
            "system": entry.get("system"),
            "think": think,
            "options": options,
        })

    a, b = participants
    print(f"Topic: {topic}")
    print(f"[{a['model']} as {a['name']}] vs [{b['model']} as {b['name']}] -- {turns} turns\n")

    transcript = []  # list of (speaker_index, text)
    try:
        for turn in range(turns):
            i = turn % 2
            me = participants[i]
            # Rebuild the message list from this speaker's point of view:
            # their own past lines are "assistant", the other's are "user".
            messages = []
            if me["system"]:
                messages.append({"role": "system", "content": me["system"]})
            for spk, text in transcript:
                role = "assistant" if spk == i else "user"
                messages.append({"role": role, "content": text})
            if not transcript:
                messages.append({"role": "user", "content": topic})

            print("  (waiting for reply...)", file=sys.stderr, flush=True)
            thinking, reply = call_chat(host, me["model"], messages,
                                        me["think"], me["options"])
            transcript.append((i, reply))
            print_turn(f"[{me['model']} as {me['name']}]", turn + 1,
                       thinking, reply, show_thinking=me["think"])
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    print(f"Done: {len(transcript)} replies.", file=sys.stderr)
    if log_fh is not None:
        sys.stdout = sys.stdout.streams[0]  # unwrap the Tee
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fh.write(f"--- session ended {stamp} ({len(transcript)} replies) ---\n")
        log_fh.close()


if __name__ == "__main__":
    main()
