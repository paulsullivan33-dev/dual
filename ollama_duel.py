#!/usr/bin/env python3
"""
ollama_duel.py -- two AI models converse with each other, configured from JSON.

Usage:
    python3 ollama_duel.py duel.json
    python3 ollama_duel.py duel.json --turns 10 --topic "A different question"

Globals in the JSON (host, topic, turns, think, max_tokens, temperature,
num_ctx, repeat_penalty, turn_prompt, log_file, save_json, timeout, display)
act as defaults; anything set inside a "models" entry overrides the global for
that model only (think, max_tokens, temperature, num_ctx, repeat_penalty,
turn_prompt only -- the rest are duel-wide). "models" must contain exactly 2
entries.
If "log_file" is set, everything printed to stdout is mirrored to a
timestamped copy of that file: the current date/time is prepended to the
file name (e.g. "duel.log" -> "20260925-084500-duel.log") so each run gets
its own log instead of appending to a previous run's. If "save_json" is set,
the transcript (speaker/model/text per turn) is written there as JSON when
the duel ends, including after a stopped-early error or Ctrl-C.
"""

import argparse
import json
import os
import sys
from datetime import datetime

from ollama_common import (
    DEFAULT_HOST,
    DEFAULT_TIMEOUT,
    OllamaError,
    build_duel_json,
    call_chat,
    save_transcript_json_safe,
    setup_utf8_stdout,
    wrap_text,
)


TOP_LEVEL_KEYS = {
    "host", "topic", "turns", "think", "max_tokens", "temperature",
    "num_ctx", "repeat_penalty", "turn_prompt", "log_file", "save_json", "timeout",
    "display", "models",
}
MODEL_KEYS = {"model", "name", "system", "think", "max_tokens", "temperature",
              "num_ctx", "repeat_penalty", "turn_prompt"}

# Sent as the last message on every turn after the first, to nudge the model
# to answer the other participant instead of starting a fresh parallel
# monologue. Without it, small models tend to ignore the transcript and each
# emit their own standalone continuation of the topic. {name} and {other}
# are replaced with the speaker's and the other participant's names.
DEFAULT_TURN_PROMPT = (
    "Reply directly to {other}'s last message, staying in character as "
    "{name}. Keep it to a few short paragraphs. Do not repeat or summarize "
    "what has already been said; move the exchange forward."
)

# Ollama's context window when num_ctx is unset is a server-side default of a
# few thousand tokens; a larger max_tokens than that can't all be used.
ASSUMED_DEFAULT_NUM_CTX = 4096


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


def _validate_field(container, key, kind, label, minimum=None):
    """Check an optional field's type (and minimum, if given) or exit with
    a clear message. `kind` is one of "bool", "int", "number", "str"."""
    if key not in container or container[key] is None:
        return
    value = container[key]
    if kind == "bool":
        ok = isinstance(value, bool)
    elif kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:  # "str"
        ok = isinstance(value, str)
    if not ok:
        sys.exit(f'"{key}" in {label} must be a {kind}, got {value!r}.')
    if minimum is not None and value < minimum:
        sys.exit(f'"{key}" in {label} must be >= {minimum}, got {value!r}.')


def _check_unknown_keys(container, allowed, label):
    """Reject keys outside `allowed` so a typo (e.g. "temprature") fails
    loudly instead of silently falling back to a default."""
    unknown = sorted(set(container) - allowed)
    if unknown:
        sys.exit(
            f'Unknown setting(s) in {label}: {", ".join(unknown)}. '
            f'Allowed: {", ".join(sorted(allowed))}.'
        )


def load_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except UnicodeDecodeError as e:
        sys.exit(f"Config {path} is not valid UTF-8: {e}")
    except json.JSONDecodeError as e:
        sys.exit(f"Invalid JSON in {path}: {e}")
    if not isinstance(cfg, dict):
        sys.exit(f"Config {path} must be a JSON object.")
    _check_unknown_keys(cfg, TOP_LEVEL_KEYS, "top level")
    models = cfg.get("models")
    if not isinstance(models, list) or len(models) != 2:
        sys.exit('Config must define exactly 2 models in the "models" list.')
    for m in models:
        if not isinstance(m, dict) or "model" not in m:
            sys.exit('Each model entry needs a "model" key, e.g. "qwen3:4b".')
        m.setdefault("name", m["model"])
        _check_unknown_keys(m, MODEL_KEYS, f'model "{m["name"]}"')

    _validate_field(cfg, "host", "str", "top level")
    _validate_field(cfg, "topic", "str", "top level")
    _validate_field(cfg, "turns", "int", "top level", minimum=1)
    _validate_field(cfg, "think", "bool", "top level")
    _validate_field(cfg, "max_tokens", "int", "top level", minimum=1)
    _validate_field(cfg, "temperature", "number", "top level", minimum=0)
    _validate_field(cfg, "num_ctx", "int", "top level", minimum=1)
    _validate_field(cfg, "repeat_penalty", "number", "top level", minimum=1)
    _validate_field(cfg, "turn_prompt", "str", "top level")
    _validate_field(cfg, "log_file", "str", "top level")
    _validate_field(cfg, "save_json", "str", "top level")
    _validate_field(cfg, "timeout", "number", "top level", minimum=1)
    _validate_field(cfg, "display", "bool", "top level")
    for m in models:
        label = f'model "{m["name"]}"'
        _validate_field(m, "think", "bool", label)
        _validate_field(m, "max_tokens", "int", label, minimum=1)
        _validate_field(m, "temperature", "number", label, minimum=0)
        _validate_field(m, "num_ctx", "int", label, minimum=1)
        _validate_field(m, "repeat_penalty", "number", label, minimum=1)
        _validate_field(m, "turn_prompt", "str", label)
    return cfg


def first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def render_turn_prompt(template, name, other):
    """Fill {name}/{other} in a turn prompt. Plain replace rather than
    str.format, so any other braces in the text are left alone."""
    return template.replace("{name}", name).replace("{other}", other)


def context_warning(name, max_tokens, num_ctx):
    """Return a warning when max_tokens can't fit in the context window
    (generation silently stops once the window fills), else None."""
    if num_ctx is not None:
        if max_tokens > num_ctx:
            return (f"Warning: {name}: max_tokens ({max_tokens}) is larger than "
                    f"num_ctx ({num_ctx}); replies stop when the context window "
                    f"fills. Lower max_tokens or raise num_ctx.")
    elif max_tokens > ASSUMED_DEFAULT_NUM_CTX:
        return (f"Warning: {name}: max_tokens ({max_tokens}) is set but num_ctx "
                f"is not; Ollama's default context window is only a few "
                f"thousand tokens, so long replies may be cut short. Set "
                f"num_ctx to match.")
    return None


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
    print(wrap_text(reply))
    print()


def format_duel_stats(model_stats):
    """Build a per-model performance table from the stats dict accumulated
    during a duel. Each entry holds turns, token counts, phase durations in
    seconds, and truncated-turn count. Returns "" when no turns completed."""
    if not model_stats:
        return ""
    rows = []
    for model, s in model_stats.items():
        gen_tps = s["gen_tokens"] / s["gen_s"] if s["gen_s"] > 0 else 0.0
        prompt_tps = (s["prompt_tokens"] / s["prompt_s"]
                      if s["prompt_s"] > 0 else 0.0)
        rows.append((model, s["turns"], gen_tps, prompt_tps,
                     s["gen_tokens"], s["truncated"]))

    def col(values, header, align_right=True):
        """Pad one column; returns (header_cell, [data_cells])."""
        cells = [header] + [str(v) for v in values]
        width = max(len(c) for c in cells)
        pad = str.rjust if align_right else str.ljust
        padded = [pad(c, width) for c in cells]
        return padded[0], padded[1:]

    cols = [
        col([r[0] for r in rows], "Model", align_right=False),
        col([r[1] for r in rows], "Turns"),
        col([f"{r[2]:.1f}" for r in rows], "Gen tok/s"),
        col([f"{r[3]:.1f}" for r in rows], "Prompt tok/s"),
        col([r[4] for r in rows], "Tokens"),
        col([r[5] for r in rows], "Truncated"),
    ]
    lines = ["=== Model performance ===",
             "  ".join(header for header, _ in cols)]
    for i in range(len(rows)):
        lines.append("  ".join(cells[i] for _, cells in cols))
    return "\n".join(lines)


def timestamped_log_path(log_path):
    """Prepend a filesystem-safe date/time stamp to the log file's base name
    (e.g. "logs/duel.log" -> "logs/20260925-084500-duel.log") so each duel
    run writes its own log file instead of appending to a previous run's."""
    directory, base = os.path.split(log_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(directory, f"{stamp}-{base}")


def _matrix(matrix, method, *args):
    """Best-effort LED matrix update. Returns the matrix, or None if the
    display died mid-duel (the duel itself continues either way)."""
    if matrix is None:
        return None
    try:
        getattr(matrix, method)(*args)
    except Exception as e:  # DisplayUnavailable or anything unexpected
        print(f"LED matrix lost ({e}); continuing without display.",
              file=sys.stderr)
        return None
    return matrix


def build_turn_messages(participants, i, topic, transcript):
    """Build the message list for participant `i`'s next turn.

    Seen from that speaker's point of view: their own past lines are
    "assistant", the other's are "user". The topic always opens the
    conversation, so both speakers see it on every turn -- otherwise the
    second speaker never sees it at all and the first loses it after turn 1.
    After the first turn, the speaker's turn_prompt (see DEFAULT_TURN_PROMPT)
    closes the list; an empty turn_prompt disables it.
    """
    me = participants[i]
    messages = []
    if me["system"]:
        messages.append({"role": "system", "content": me["system"]})
    messages.append({"role": "user", "content": topic})
    for spk, text in transcript:
        role = "assistant" if spk == i else "user"
        messages.append({"role": role, "content": text})
    if transcript and me["turn_prompt"]:
        messages.append({
            "role": "user",
            "content": render_turn_prompt(me["turn_prompt"], me["name"],
                                          participants[1 - i]["name"]),
        })
    return messages


def print_duel_header(topic, participants, turns):
    a, b = participants
    print(wrap_text(f"Topic: {topic}", subsequent_indent=" " * len("Topic: ")))
    print(f"[{a['model']} as {a['name']}] vs [{b['model']} as {b['name']}] -- {turns} turns\n")


def run_duel(host, topic, turns, participants, timeout, transcript, matrix=None):
    """Run the duel's turn loop, printing each reply as it arrives.

    Each participant is a dict with name, model, system, think, options and
    turn_prompt. Replies are appended to the caller's `transcript` list as
    (speaker_index, text) so that whatever was generated survives an early
    stop. Ctrl-C or an OllamaError stops the loop gracefully.

    Returns (model_stats, matrix): per-model stats for format_duel_stats,
    and the LED matrix, or None if there is none or it failed mid-duel.
    """
    model_stats = {}  # model -> {turns, gen_tokens, gen_s, prompt_tokens, prompt_s, truncated}
    try:
        for turn in range(turns):
            i = turn % 2
            me = participants[i]
            messages = build_turn_messages(participants, i, topic, transcript)

            print("  (waiting for reply...)", file=sys.stderr, flush=True)
            matrix = _matrix(matrix, "progress", turn, turns)
            thinking, reply, done_reason, metrics = call_chat(host, me["model"], messages,
                                                             me["think"], me["options"],
                                                             timeout=timeout)
            # Ollama's measured generation speed (exact token count over
            # generation time, excluding model load and prompt reading).
            if matrix is not None:
                matrix = _matrix(matrix, "show_text",
                                 f"{metrics['gen_tps']:.1f}T/S")
            transcript.append((i, reply))
            stats = model_stats.setdefault(me["model"], {
                "turns": 0, "gen_tokens": 0, "gen_s": 0.0,
                "prompt_tokens": 0, "prompt_s": 0.0, "truncated": 0})
            stats["turns"] += 1
            stats["gen_tokens"] += metrics["gen_tokens"]
            stats["gen_s"] += metrics["gen_s"]
            stats["prompt_tokens"] += metrics["prompt_tokens"]
            stats["prompt_s"] += metrics["prompt_s"]
            if done_reason == "length":
                stats["truncated"] += 1
            print_turn(f"[{me['model']} as {me['name']}]", turn + 1,
                       thinking, reply, show_thinking=me["think"])
            if done_reason == "length":
                print(f"--- WARNING: reply hit the max_tokens ceiling "
                      f"({me['options']['num_predict']} tokens) and was truncated; "
                      f"consider raising max_tokens ---")
                print()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    except OllamaError as e:
        print(f"\n{e}\nStopped.", file=sys.stderr)
    return model_stats, matrix


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
    ap.add_argument("--save-json", default=None,
                    help="override the config save_json (write the transcript to this JSON file)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="override the config timeout, in seconds")
    ap.add_argument("--display", dest="display", action="store_true", default=None,
                    help="show live duel stats on the Arduino Uno Q's built-in "
                         "8x13 LED matrix (needs python3-smbus on the Uno Q)")
    ap.add_argument("--no-display", dest="display", action="store_false",
                    help="force the LED matrix display off")
    args = ap.parse_args()

    setup_utf8_stdout()

    cfg = load_config(args.config)
    host = first_not_none(args.host, cfg.get("host"), DEFAULT_HOST)
    topic = first_not_none(args.topic, cfg.get("topic"))
    if not topic:
        sys.exit('No topic: set "topic" in the config or pass --topic.')
    turns = first_not_none(args.turns, cfg.get("turns"), 6)
    if turns < 1:
        sys.exit(f'"turns" must be >= 1, got {turns}.')
    timeout = first_not_none(args.timeout, cfg.get("timeout"), DEFAULT_TIMEOUT)
    save_json_path = first_not_none(args.save_json, cfg.get("save_json"))
    want_display = first_not_none(args.display, cfg.get("display"), False)

    # Optional log file: everything printed to stdout is mirrored there.
    # (stderr progress lines like "(waiting for reply...)" stay console-only.)
    # The stamp prepended to the file name keeps each run in its own file.
    log_path = first_not_none(args.log_file, cfg.get("log_file"))
    log_fh = None
    if log_path:
        log_path = timestamped_log_path(log_path)
        try:
            # Scenarios log to logs/...; create the folder on first use.
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
        except OSError as e:
            sys.exit(f"Cannot open log file {log_path}: {e}")
        print(f"Logging to {log_path}", file=sys.stderr)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fh.write(f"\n--- session started {stamp} ---\n")
        sys.stdout = Tee(sys.stdout, log_fh)

    # Optional LED matrix display (Arduino Uno Q's built-in 8x13 matrix).
    # Strictly opt-in and best-effort: any failure warns and the duel runs
    # headless, so this never breaks the script on other machines.
    matrix = None
    if want_display:
        try:
            from unoq_matrix import DisplayUnavailable, UnoQMatrix
            matrix = UnoQMatrix()
            matrix.show_text("DUEL")
        except DisplayUnavailable as e:
            print(f"LED matrix unavailable ({e}); continuing without display.",
                  file=sys.stderr)
            matrix = None


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
        repeat_penalty = first_not_none(entry.get("repeat_penalty"),
                                        cfg.get("repeat_penalty"))
        options = {"num_predict": max_tokens}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if temperature is not None:
            options["temperature"] = temperature
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        warning = context_warning(entry["name"], max_tokens, num_ctx)
        if warning:
            print(warning, file=sys.stderr)
        participants.append({
            "name": entry["name"],
            "model": entry["model"],
            "system": entry.get("system"),
            "think": think,
            "options": options,
            "turn_prompt": first_not_none(entry.get("turn_prompt"),
                                          cfg.get("turn_prompt"),
                                          DEFAULT_TURN_PROMPT),
        })

    print_duel_header(topic, participants, turns)

    transcript = []  # list of (speaker_index, text)
    try:
        model_stats, matrix = run_duel(host, topic, turns, participants, timeout,
                                       transcript, matrix=matrix)
        print(f"Done: {len(transcript)} replies.", file=sys.stderr)
        if log_fh is not None:
            print(f"Logging to {log_path}", file=sys.stderr)
        if model_stats:
            print()
            print(format_duel_stats(model_stats))
    finally:
        # Always write the session-end marker / transcript, even if the
        # duel stopped early on an Ollama error.
        matrix = _matrix(matrix, "show_text", "DONE")
        if log_fh is not None:
            sys.stdout = sys.stdout.streams[0]  # unwrap the Tee
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_fh.write(f"--- session ended {stamp} ({len(transcript)} replies) ---\n")
            log_fh.close()
        save_transcript_json_safe(save_json_path, build_duel_json(transcript, participants))


if __name__ == "__main__":
    main()
