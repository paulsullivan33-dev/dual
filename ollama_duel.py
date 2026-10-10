#!/usr/bin/env python3
"""
ollama_duel.py -- two AI models converse with each other, configured from JSON.

Usage:
    python3 ollama_duel.py duel.json
    python3 ollama_duel.py duel.json --turns 10 --topic "A different question"
    python3 ollama_duel.py duel.json --profile arduino_q
    python3 ollama_duel.py --profile arduino_q      (runs the profile's scenario list)

A machine profile (profiles/<name>.json) swaps both speakers' models by
position and forces settings such as num_ctx and think, so one scenario
file serves every machine.

Globals in the JSON (host, topic, turns, think, max_tokens, temperature,
num_ctx, repeat_penalty, turn_prompt, first_turn_prompt, history_turns,
log_file, save_json, timeout, display, dedup_guard, ntfy_url, run_code,
run_code_args, run_code_timeout, tags, judge) act as defaults; anything set inside a
"models" entry overrides the global for that model only (think, max_tokens,
temperature, num_ctx, repeat_penalty, turn_prompt, first_turn_prompt,
history_turns only -- the rest are duel-wide). "models" must contain exactly
2 entries. run_code (or --run-code) runs each reply's Python block; see
ollama_coderun.py.
If "log_file" is set, everything printed to stdout is mirrored to a
timestamped copy of that file: the current date/time is prepended to the
file name (e.g. "logs/duel.log" -> "output/logs/20260925-084500-duel.log") so each run gets
its own log instead of appending to a previous run's. If "save_json" is set,
the transcript (speaker/model/text per turn) is written there as JSON when
the duel ends, including after a stopped-early error or Ctrl-C.

After every run the script also appends a one-block summary to
"output/run_results.log" (override with the top-level
"results_log" setting or --results-log, disable with --no-results-log):
date/time, config, models, turns completed, and the per-model stats table
on success or the error message when the duel stopped early.

Relative paths for log_file, save_json and results_log are placed under the
output/ folder (see ollama_common.output_path); absolute paths are used as-is.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from ollama_common import (
    DEFAULT_HOST,
    DEFAULT_TIMEOUT,
    OllamaError,
    build_duel_json,
    check_unknown_keys as _check_unknown_keys,
    call_chat,
    output_path,
    save_transcript_json_safe,
    setup_utf8_stdout,
    validate_field as _validate_field,
    wrap_text,
)
from ollama_reporting import (append_run_summary, format_duel_stats, format_run_summary,
                              load_default_ntfy_url, notify_duel_done)
from ollama_coderun import (DEFAULT_RUN_TIMEOUT, count_runs, describe_run,
                            extract_program, run_program)
from ollama_judge import normalize_judge, run_judge
from ollama_profiles import apply_profile, load_profile, profile_scenarios


TOP_LEVEL_KEYS = {
    "host", "topic", "turns", "think", "max_tokens", "temperature",
    "num_ctx", "repeat_penalty", "turn_prompt", "first_turn_prompt",
    "history_turns", "log_file", "save_json", "timeout",
    "display", "dedup_guard", "results_log", "ntfy_url", "models",
    "run_code", "run_code_args", "run_code_timeout", "tags", "judge",
}
MODEL_KEYS = {"model", "name", "system", "think", "max_tokens", "temperature",
              "num_ctx", "repeat_penalty", "turn_prompt", "first_turn_prompt",
              "history_turns"}

# Sent as the last message on every turn after the first, to nudge the model
# to answer the other participant instead of starting a fresh parallel
# monologue. Without it, small models tend to ignore the transcript and each
# emit their own standalone continuation of the topic. {name} and {other}
# are replaced with the speaker's and the other participant's names.
# Default repetition penalty sent when the config doesn't set one. Small
# models fall into echo loops without it; 1.25 is strong enough to break
# the loop without mangling normal prose. Overridable per model or top
# level via "repeat_penalty".
DEFAULT_REPEAT_PENALTY = 1.25

DEFAULT_TURN_PROMPT = (
    "Reply directly to {other}'s last message, staying in character as "
    "{name}. Write only {name}'s own words and actions -- never write "
    "dialogue or actions for {other}. Keep it to a few short paragraphs. "
    "Do not repeat or summarize what has already been said; move the "
    "exchange forward."
)

# Sent as the last message on the first turn only. Without it the opening
# speaker sees both sides' positions in the topic and no contrary
# instruction, so it tends to script-write the whole debate -- both voices
# -- instead of just its own opening. {name} and {other} are replaced like
# in DEFAULT_TURN_PROMPT. Overridable per model or top level via
# "first_turn_prompt"; "" disables it.
DEFAULT_FIRST_TURN_PROMPT = (
    "You are {name}. Open the exchange with your own position, staying in "
    "character. Write only {name}'s own words and actions -- never write "
    "dialogue or actions for {other}; {other} will speak for themselves. "
    "Keep it to a few short paragraphs."
)

# Dedup guard: if a speaker's reply is identical (modulo whitespace) to its
# own previous reply, the turn is re-rolled once with a bumped temperature
# and an explicit no-repeat nudge appended to the messages. Small models at
# low temperature otherwise get stuck echoing themselves verbatim --
# repeat_penalty only covers the last ~64 tokens, so a whole earlier reply
# sails through unpenalized, and a bare temperature bump often isn't enough
# to knock the model out of the rut because the prompt itself is unchanged.
# Detection stays exact-match on purpose: in code-duel scenarios a speaker's
# consecutive replies are legitimately 95%+ similar (the whole program
# re-listed with a small change), so fuzzy matching would misfire there.
# Only one retry: a second duplicate is accepted (with a note), so a stuck
# model can't spin forever. Disable with "dedup_guard": false in the
# scenario JSON.
DEDUP_RETRY_TEMP_BUMP = 0.3
DEDUP_RETRY_TEMP_CAP = 1.5
DEDUP_RETRY_NUDGE = (
    "That reply repeated what you just said. Say something new -- "
    "do not repeat your previous reply."
)
# Ollama's server-side default temperature when the config doesn't set one.
ASSUMED_DEFAULT_TEMPERATURE = 0.8

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


def load_config(path, profile=None):
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
    if profile:
        # Before names default to model ids, so an unnamed speaker is named
        # after the profile's model, not the one it replaced.
        apply_profile(cfg, profile)
    for m in models:
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
    _validate_field(cfg, "history_turns", "int", "top level", minimum=1)
    _validate_field(cfg, "turn_prompt", "str", "top level")
    _validate_field(cfg, "first_turn_prompt", "str", "top level")
    _validate_field(cfg, "log_file", "str", "top level")
    _validate_field(cfg, "save_json", "str", "top level")
    _validate_field(cfg, "timeout", "number", "top level", minimum=1)
    _validate_field(cfg, "display", "bool", "top level")
    _validate_field(cfg, "run_code", "bool", "top level")
    _validate_field(cfg, "run_code_timeout", "number", "top level", minimum=1)
    run_args = cfg.get("run_code_args")
    if run_args is not None and not (
            isinstance(run_args, list) and all(isinstance(a, str) for a in run_args)):
        sys.exit(f'"run_code_args" in top level must be a list of strings, got {run_args!r}.')
    _validate_field(cfg, "dedup_guard", "bool", "top level")
    _validate_field(cfg, "results_log", "str", "top level")
    _validate_field(cfg, "ntfy_url", "str", "top level")
    tags = cfg.get("tags")
    if tags is not None and not (
            isinstance(tags, list) and tags
            and all(isinstance(t, str) and t.strip() for t in tags)):
        sys.exit(f'"tags" in top level must be a non-empty list of strings, '
                 f'got {tags!r}.')
    cfg["tags"] = tags
    cfg["judge"] = normalize_judge(cfg.get("judge"))
    for m in models:
        label = f'model "{m["name"]}"'
        _validate_field(m, "think", "bool", label)
        _validate_field(m, "max_tokens", "int", label, minimum=1)
        _validate_field(m, "temperature", "number", label, minimum=0)
        _validate_field(m, "num_ctx", "int", label, minimum=1)
        _validate_field(m, "repeat_penalty", "number", label, minimum=1)
        _validate_field(m, "history_turns", "int", label, minimum=1)
        _validate_field(m, "turn_prompt", "str", label)
        _validate_field(m, "first_turn_prompt", "str", label)
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


def normalize_reply(text):
    """Collapse all whitespace for duplicate detection, so a re-rolled
    reply that differs only in line breaks still counts as a repeat."""
    return re.sub(r"\s+", " ", text).strip()


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


def build_turn_messages(participants, i, topic, transcript, run_reports=None):
    """Build the message list for participant `i`'s next turn.

    Seen from that speaker's point of view: their own past lines are
    "assistant", the other's are "user". The topic always opens the
    conversation, so both speakers see it on every turn -- otherwise the
    second speaker never sees it at all and the first loses it after turn 1.
    After the first turn, the speaker's turn_prompt (see DEFAULT_TURN_PROMPT)
    closes the list; an empty turn_prompt disables it. On the first turn the
    speaker's first_turn_prompt (see DEFAULT_FIRST_TURN_PROMPT) closes the
    list instead, so the opener stays in character rather than script-writing
    both sides; an empty first_turn_prompt disables it.

    When the speaker's history_turns is set, only the most recent that many
    replies are sent (the system prompt, topic and nudge always are), so a
    long duel can't outgrow a small context window.
    """
    me = participants[i]
    messages = []
    if me["system"]:
        messages.append({"role": "system", "content": me["system"]})
    messages.append({"role": "user", "content": topic})
    history = me.get("history_turns")
    recent = transcript[-history:] if history else transcript
    first_index = len(transcript) - len(recent)
    for k, (spk, text) in enumerate(recent, first_index):
        role = "assistant" if spk == i else "user"
        messages.append({"role": role, "content": text})
        # With run_code on, what happened when this reply's program ran.
        if run_reports and k in run_reports:
            messages.append({"role": "user", "content": run_reports[k]})
    first_nudge = me.get("first_turn_prompt", DEFAULT_FIRST_TURN_PROMPT)
    if transcript and me["turn_prompt"]:
        messages.append({
            "role": "user",
            "content": render_turn_prompt(me["turn_prompt"], me["name"],
                                          participants[1 - i]["name"]),
        })
    elif not transcript and first_nudge:
        messages.append({
            "role": "user",
            "content": render_turn_prompt(first_nudge, me["name"],
                                          participants[1 - i]["name"]),
        })
    return messages


def print_duel_header(topic, participants, turns):
    a, b = participants
    print(wrap_text(f"Topic: {topic}", subsequent_indent=" " * len("Topic: ")))
    print(f"[{a['model']} as {a['name']}] vs [{b['model']} as {b['name']}] -- {turns} turns\n")


CONTEXT_NEARLY_FULL = 0.8  # note when a speaker's context window passes 80%


def context_tokens_used(messages, reply, metrics):
    """How much of the context window this turn used: prompt plus reply.

    Ollama's prompt_eval_count can count only the uncached part of a prompt
    when it reuses a previous request's prefix, so take the larger of its
    figures and a rough estimate from the text (~4 characters per token).
    """
    reported = metrics.get("prompt_tokens", 0) + metrics.get("gen_tokens", 0)
    chars = sum(len(m.get("content") or "") for m in messages) + len(reply)
    return max(reported, chars // 4)


def context_usage_note(name, used, num_ctx, level):
    """Warn as a speaker's conversation fills its context window.

    Every turn resends the whole transcript, so it only grows. Returns
    (message or None, new level); each level is reported once per speaker,
    so the log isn't flooded. Without an explicit num_ctx the window is the
    server's default, which isn't known here, so nothing is reported.
    """
    if not num_ctx:
        return None, level
    pct = round(100 * used / num_ctx)
    if used >= num_ctx and level < 2:
        return (f"--- WARNING: {name}'s context window is full ({used}/{num_ctx} "
                f"tokens). Ollama has to drop older messages to fit, so {name} may "
                f"lose track of the topic and earlier turns; raise num_ctx or use "
                f"fewer turns ---"), 2
    if used >= CONTEXT_NEARLY_FULL * num_ctx and level < 1:
        return (f"--- NOTE: {name}'s context window is {pct}% full ({used}/{num_ctx} "
                f"tokens); it fills up as the conversation grows ---"), 1
    return None, level


def run_duel(host, topic, turns, participants, timeout, transcript, matrix=None,
             dedup_guard=True, run_code=None):
    """Run the duel's turn loop, printing each reply as it arrives.

    Each participant is a dict with name, model, system, think, options,
    turn_prompt and first_turn_prompt. Replies are appended to the caller's `transcript` list as
    (speaker_index, text) so that whatever was generated survives an early
    stop. Ctrl-C or an OllamaError stops the loop gracefully.

    When dedup_guard is on, a turn whose reply duplicates that speaker's
    own previous reply (ignoring whitespace differences) is re-rolled once
    with a bumped temperature and an explicit no-repeat nudge appended to
    the messages; a second duplicate is kept as-is, with a note.

    Returns (model_stats, matrix, stop_note): per-model stats for
    format_duel_stats, the LED matrix (or None if there is none or it
    failed mid-duel), and why the loop stopped early -- None when all
    turns completed, otherwise the error or interrupt message. The caller
    decides where stop_note goes; ollama_duel.py writes it into the log
    file, because the stderr stop message never reaches the log and a
    batch run is undebuggable without it.
    """
    model_stats = {}  # model -> {turns, gen_tokens, gen_s, prompt_tokens, prompt_s, truncated}
    prev_replies = {}  # speaker_index -> normalized text of their last reply
    run_reports = {}  # transcript index -> code-run report (run_code only)
    # Outcome of each code run, kept in run_code so main() can put the tally
    # in the run summary and ntfy notice.
    run_results = run_code.setdefault("results", []) if run_code is not None else []
    ctx_levels = [0, 0]  # per speaker: 0 = fine, 1 = nearly-full noted, 2 = full warned
    stop_note = None  # why the loop ended early, when it did
    try:
        for turn in range(turns):
            i = turn % 2
            me = participants[i]
            messages = build_turn_messages(participants, i, topic, transcript,
                                           run_reports)

            print("  (waiting for reply...)", file=sys.stderr, flush=True)
            matrix = _matrix(matrix, "progress", turn, turns)
            thinking, reply, done_reason, metrics = call_chat(host, me["model"], messages,
                                                             me["think"], me["options"],
                                                             timeout=timeout)
            norm = normalize_reply(reply)
            if dedup_guard and norm and norm == prev_replies.get(i):
                retry_options = dict(me["options"])
                base_temp = retry_options.get("temperature",
                                              ASSUMED_DEFAULT_TEMPERATURE)
                retry_options["temperature"] = min(base_temp + DEDUP_RETRY_TEMP_BUMP,
                                                   DEDUP_RETRY_TEMP_CAP)
                # A bare temperature bump often isn't enough: the prompt is
                # unchanged, so a stuck model happily repeats itself again.
                # The explicit nudge breaks the loop far more reliably.
                retry_messages = messages + [{"role": "user",
                                              "content": DEDUP_RETRY_NUDGE}]
                print(f"--- DEDUP GUARD: {me['name']} repeated its previous reply; "
                      f"re-rolling once with a no-repeat nudge at temperature "
                      f"{retry_options['temperature']:.2f} ---")
                thinking, reply, done_reason, metrics = call_chat(
                    host, me["model"], retry_messages, me["think"], retry_options,
                    timeout=timeout)
                norm = normalize_reply(reply)
                if norm and norm == prev_replies.get(i):
                    print(f"--- DEDUP GUARD: {me['name']} repeated again; "
                          f"keeping the retry as-is ---")
            prev_replies[i] = norm
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
            used = context_tokens_used(messages, reply, metrics)
            note, ctx_levels[i] = context_usage_note(
                me["name"], used, me["options"].get("num_ctx"), ctx_levels[i])
            if note:
                print(note)
                print()
            if run_code is not None:
                run_results.append(
                    _run_reply_code(reply, me["name"], run_code, run_reports,
                                    len(transcript) - 1))
    except KeyboardInterrupt:
        stop_note = "interrupted by user (Ctrl-C)"
        print("\nStopped.", file=sys.stderr)
    except OllamaError as e:
        stop_note = str(e)
        print(f"\n{e}\nStopped.", file=sys.stderr)
    if run_results:
        print(count_runs(run_results))
        print()
    return model_stats, matrix, stop_note


def _run_reply_code(reply, author, run_code, run_reports, index):
    """With run_code on: run the reply's last Python block, print the
    result, and remember it so later turns show it to both speakers.
    Returns the outcome ("ok", "failed", "timeout" or "no_code")."""
    program = extract_program(reply)
    if program is None:
        print("--- CODE RUN: no Python code block in this reply ---")
        print()
        return "no_code"
    result = run_program(program, run_code["args"], run_code["timeout"])
    report = describe_run(result, author)
    run_reports[index] = report
    print("--- CODE RUN ---")
    print(report)
    print()
    return result["status"]


def expand_configs(config_arg):
    """Expand the config argument into a list of scenario files.

    A directory yields every *.json inside it (sorted); a glob pattern
    yields every match (sorted) -- quote it so the shell doesn't expand
    it first. Anything else is returned as the single path, validated
    later by load_config exactly as before.
    """
    import glob as globmod
    if os.path.isdir(config_arg):
        paths = sorted(globmod.glob(os.path.join(config_arg, "*.json")))
        if not paths:
            sys.exit(f"No .json scenario files in directory: {config_arg}")
        return paths
    if globmod.has_magic(config_arg):
        paths = sorted(globmod.glob(config_arg))
        if not paths:
            sys.exit(f"Glob matched no files: {config_arg}")
        return paths
    return [config_arg]


def read_scenario_tags(path):
    """Best-effort read of a scenario file's "tags" list. Anything
    unreadable or malformed yields [] -- tag filtering must never fail a
    batch, it just skips what it can't read."""
    try:
        with open(path, encoding="utf-8") as f:
            tags = json.load(f).get("tags")
    except (OSError, ValueError):
        return []
    if isinstance(tags, list):
        return [t for t in tags if isinstance(t, str)]
    return []


def configs_matching_tags(configs, wanted):
    """Keep the scenario files carrying any of the wanted tags.

    No wanted tags -> everything passes through, so single-file runs and
    untagged batches behave exactly as before.
    """
    if not wanted:
        return list(configs)
    wanted = set(wanted)
    return [p for p in configs if wanted & set(read_scenario_tags(p))]


def list_all_tags(configs):
    """Every tag used by the given scenario files, with a count each."""
    counts = {}
    for path in configs:
        for tag in read_scenario_tags(path):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def run_batch(configs, config_arg):
    """Run each scenario as its own child process (re-exec of this script).

    Each duel then behaves exactly like a single-duel run: its own log
    file, its own ntfy notice, its own run_results.log entry, and its own
    process command line for external monitors. A failed scenario is
    reported and skipped; the rest of the batch continues. Ctrl+C aborts
    the whole batch. Returns the process exit status.
    """
    import subprocess
    script = os.path.abspath(__file__)
    rest = [a for a in sys.argv[1:] if a != config_arg]
    failures = 0
    for i, cfg_path in enumerate(configs, 1):
        print(f"=== batch {i}/{len(configs)}: {cfg_path} ===", file=sys.stderr)
        rc = subprocess.run([sys.executable, script, cfg_path] + rest,
                            check=False).returncode
        if rc != 0:
            failures += 1
            print(f"--- {cfg_path} exited with status {rc}; continuing ---",
                  file=sys.stderr)
    print(f"=== batch done: {len(configs) - failures}/{len(configs)} ok ===",
          file=sys.stderr)
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description="Two Ollama models converse, configured from a JSON file.")
    ap.add_argument("config", nargs="?", default=None,
                    help="path to a JSON config file, a directory "
                    "(runs every *.json inside it), or a glob pattern "
                    "(quote it so the shell doesn't expand it first); may be "
                    "omitted with a --profile that lists scenarios")
    ap.add_argument("--profile", default=None,
                    help="machine profile: a name in profiles/ (e.g. arduino_q) "
                         "or a path; swaps the models and forces its settings. "
                         "Without a scenario, runs the profile's scenario list")
    ap.add_argument("--topic", default=None, help="override the config topic")
    ap.add_argument("--turns", type=int, default=None, help="override the config turn count")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the config max_tokens per turn")
    ap.add_argument("--host", default=None, help="override the config host")
    ap.add_argument("--think", dest="think", action="store_true", default=None,
                    help="force thinking display on")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="force thinking display off")
    ap.add_argument("--log-file", default=None,
                    help="override the config log_file (mirror stdout to this file)")
    ap.add_argument("--save-json", default=None,
                    help="override the config save_json (write the transcript to this JSON file)")
    ap.add_argument("--results-log", default=None,
                    help="override the config results_log (append a run summary "
                         "to this file; default run_results.log)")
    ap.add_argument("--no-results-log", dest="results_log", action="store_false",
                    help="do not write the run summary file")
    ap.add_argument("--ntfy-url", default=None,
                    help="send an ntfy notification when the duel ends "
                         "(overrides the scenario config and ~/.dual.conf)")
    ap.add_argument("--no-ntfy", dest="ntfy", action="store_false",
                    help="do not send ntfy notifications even if configured")
    ap.add_argument("--timeout", type=float, default=None,
                    help="override the config timeout, in seconds")
    ap.add_argument("--display", dest="display", action="store_true", default=None,
                    help="show live duel stats on the Arduino Uno Q's built-in "
                         "8x13 LED matrix (needs python3-smbus on the Uno Q)")
    ap.add_argument("--no-display", dest="display", action="store_false",
                    help="force the LED matrix display off")
    ap.add_argument("--run-code", dest="run_code", action="store_true", default=None,
                    help="run the last Python code block of each reply (timeout, "
                         "no keyboard input) and show the result to both speakers. "
                         "Not a sandbox: the code runs with your permissions")
    ap.add_argument("--no-run-code", dest="run_code", action="store_false",
                    help="never run reply code, even if the scenario turns it on")
    ap.add_argument("--tag", dest="tags", action="append", default=None,
                    metavar="TAG",
                    help="in a batch run (directory or glob), only run "
                         "scenarios tagged with TAG; repeatable, matches any")
    ap.add_argument("--list-tags", action="store_true",
                    help="list the tags used by the given scenarios and exit")
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="skip the scenario's judge, if it has one")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact messages turn 1 would send and exit "
                         "without calling Ollama")
    args = ap.parse_args()

    setup_utf8_stdout()

    profile = load_profile(args.profile) if args.profile else None
    if args.config is None:
        if profile is None:
            ap.error("a scenario (file, directory or glob) or a --profile "
                     "that lists scenarios is required")
        configs = profile_scenarios(profile, args.profile)
    else:
        configs = expand_configs(args.config)
    if args.list_tags:
        # Discovery aid: what tags exist in these scenarios, and how many
        # carry each. Runs before any filtering or dueling.
        counts = list_all_tags(configs)
        if not counts:
            print("No tags found in the given scenarios.")
        else:
            width = max(len(t) for t in counts)
            for tag in sorted(counts):
                print(f"{tag.ljust(width)}  {counts[tag]}")
        return
    configs = configs_matching_tags(configs, args.tags)
    if not configs:
        sys.exit(f"No scenarios match tag(s): {', '.join(args.tags or [])}.")
    if len(configs) > 1:
        sys.exit(run_batch(configs, args.config))

    config_path = configs[0]
    cfg = load_config(config_path, profile)
    host = first_not_none(args.host, cfg.get("host"), DEFAULT_HOST)
    topic = first_not_none(args.topic, cfg.get("topic"))
    if not topic:
        sys.exit('No topic: set "topic" in the config or pass --topic.')
    turns = first_not_none(args.turns, cfg.get("turns"), 6)
    if turns < 1:
        sys.exit(f'"turns" must be >= 1, got {turns}.')
    if args.max_tokens is not None and args.max_tokens < 1:
        sys.exit(f'"--max-tokens" must be >= 1, got {args.max_tokens}.')
    timeout = first_not_none(args.timeout, cfg.get("timeout"), DEFAULT_TIMEOUT)
    save_json_path = output_path(first_not_none(args.save_json, cfg.get("save_json")))
    want_display = first_not_none(args.display, cfg.get("display"), False)
    dedup_guard = first_not_none(cfg.get("dedup_guard"), True)
    # Running model-written code is strictly opt-in: the scenario's run_code
    # or --run-code; --no-run-code always wins. Profiles can't turn it on.
    run_code = None
    if first_not_none(args.run_code, cfg.get("run_code"), False):
        run_code = {"args": cfg.get("run_code_args") or [],
                    "timeout": cfg.get("run_code_timeout") or DEFAULT_RUN_TIMEOUT}
    # Run summary file: appended after every duel with the date/time and
    # either the stats table (success) or the error (early stop). Defaults
    # to output/run_results.log; --no-results-log (or a false-y CLI value)
    # disables it.
    results_log_path = output_path(first_not_none(
        args.results_log, cfg.get("results_log"), "run_results.log"))

    # ntfy: explicit CLI flag wins, then the scenario config, then the
    # default ~/.dual.conf. Undefined everywhere -> ntfy_url stays None
    # and notifications are skipped silently.
    ntfy_url = first_not_none(args.ntfy_url, cfg.get("ntfy_url"),
                              load_default_ntfy_url() or None)
    if not args.ntfy:
        ntfy_url = None

    # Optional log file: everything printed to stdout is mirrored there.
    # (stderr progress lines like "(waiting for reply...)" stay console-only.)
    # The stamp prepended to the file name keeps each run in its own file.
    # A dry run never opens one -- it should not leave log artifacts behind.
    log_path = output_path(first_not_none(args.log_file, cfg.get("log_file")))
    log_fh = None
    if log_path and not args.dry_run:
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
        start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fh.write(f"\n--- session started {start_stamp} ---\n")
        sys.stdout = Tee(sys.stdout, log_fh)

    # Optional LED matrix display (Arduino Uno Q's built-in 8x13 matrix).
    # Strictly opt-in and best-effort: any failure warns and the duel runs
    # headless, so this never breaks the script on other machines.
    matrix = None
    if want_display:
        try:
            from unoq_matrix import DisplayUnavailable, create_matrix
            matrix = create_matrix()
            scenario_name = os.path.splitext(os.path.basename(config_path))[0]
            scenario_name = scenario_name.replace("_", " ").upper()
            matrix.show_text(f"DUEL: {scenario_name}")
        except DisplayUnavailable as e:
            print(f"LED matrix unavailable ({e}); continuing without display.",
                  file=sys.stderr)
            matrix = None


    # Resolve per-model settings: model entry wins, then globals, then default.
    participants = []
    for entry in cfg["models"]:
        think = first_not_none(args.think, entry.get("think"), cfg.get("think"), False)
        max_tokens = first_not_none(
            args.max_tokens, entry.get("max_tokens"), cfg.get("max_tokens"),
            2048 if think else 300,  # thinking eats the same token budget
        )
        temperature = first_not_none(entry.get("temperature"), cfg.get("temperature"))
        num_ctx = first_not_none(entry.get("num_ctx"), cfg.get("num_ctx"))
        repeat_penalty = first_not_none(entry.get("repeat_penalty"),
                                        cfg.get("repeat_penalty"),
                                        DEFAULT_REPEAT_PENALTY)
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
            "first_turn_prompt": first_not_none(entry.get("first_turn_prompt"),
                                               cfg.get("first_turn_prompt"),
                                               DEFAULT_FIRST_TURN_PROMPT),
            "history_turns": first_not_none(entry.get("history_turns"),
                                           cfg.get("history_turns")),
        })

    if profile:
        print(f"Profile: {args.profile}")
    print_duel_header(topic, participants, turns)

    if run_code is not None:
        extra = f", arguments {' '.join(run_code['args'])}" if run_code["args"] else ""
        print(f"Code running: ON -- each reply's last Python block runs with a "
              f"{run_code['timeout']:g}s timeout{extra}. This runs model-written "
              f"code on this machine with your permissions.")
        print()
    if args.dry_run:
        # Show exactly what the opening turn would send, then stop before
        # any Ollama call. Handy when writing or debugging scenarios.
        opener = participants[0]
        print(f"--- dry run: turn 1 messages for {opener['name']} "
              f"({opener['model']}) ---")
        for m in build_turn_messages(participants, 0, topic, []):
            print(f"[{m['role']}]")
            print(m["content"])
            print()
        return

    transcript = []  # list of (speaker_index, text)
    stop_note = None  # why the duel ended early, when it did
    model_stats = {}  # per-model stats; stays empty if the loop never starts
    verdict = None  # the judge's verdict text, when a judge scores the duel
    run_started = datetime.now()
    try:
        model_stats, matrix, stop_note = run_duel(host, topic, turns, participants, timeout,
                                                  transcript, matrix=matrix,
                                                  dedup_guard=dedup_guard,
                                                  run_code=run_code)
        judge = cfg.get("judge") if args.judge else None
        if judge and transcript and stop_note is None:
            # A third model scores the finished duel. Best-effort: a judge
            # failure is noted, never fatal -- the duel result stands.
            print(f"--- JUDGE: {judge['model']} is scoring the duel ---")
            print()
            try:
                verdict = run_judge(host, judge, topic, participants,
                                    transcript, timeout, call_chat)
            except OllamaError as e:
                print(f"--- JUDGE FAILED: {e} ---")
                print()
            else:
                print("=== Verdict ===")
                print(verdict)
                print()
        print(f"Done: {len(transcript)} replies.", file=sys.stderr)
        if log_fh is not None:
            print(f"Logging to {log_path}", file=sys.stderr)
        if model_stats:
            print()
            print(format_duel_stats(model_stats))
    finally:
        code_runs = (count_runs(run_code["results"])
                     if run_code is not None and run_code.get("results") else None)
        # Full path of the transcript log, for the summary and notice.
        written_log = os.path.abspath(log_path) if log_fh is not None else None
        # Always write the session-end marker / transcript, even if the
        # duel stopped early on an Ollama error.
        matrix = _matrix(matrix, "show_text", "DONE")
        if log_fh is not None:
            sys.stdout = sys.stdout.streams[0]  # unwrap the Tee
            if stop_note:
                # The stderr stop message never reaches the log file (only
                # stdout is mirrored), so record the reason here -- without
                # it, a batch run that dies early is undebuggable.
                log_fh.write(f"--- stopped early ---\n{stop_note.strip()}\n")
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_fh.write(f"--- session started {start_stamp} ---\n")
            log_fh.write(f"--- session ended {stamp} ({len(transcript)} replies) ---\n")
            log_fh.close()
        save_transcript_json_safe(save_json_path, build_duel_json(transcript, participants))
        if results_log_path:
            # One-block summary for run_results.log: timestamp plus stats on
            # success, or the error when the duel stopped early. Written
            # here so it lands even for handled early stops -- and for an
            # unexpected crash, sys.exc_info() still holds the exception
            # while the finally block runs.
            crashed = stop_note is None and sys.exc_info()[0] is not None
            crash_note = None
            if crashed:
                exc = sys.exc_info()[1]
                crash_note = f"unexpected error: {type(exc).__name__}: {exc}"
            entry = format_run_summary(run_started, config_path, participants,
                                       turns, transcript, model_stats,
                                       crash_note or stop_note,
                                       crashed=crashed, code_runs=code_runs,
                                       log_path=written_log, verdict=verdict)
            append_run_summary(results_log_path, entry)
        if ntfy_url:
            # Completion notice: OK, stopped early, or crashed. Best-effort
            # (notify_duel_done never raises); skipped silently when no
            # ntfy URL is configured anywhere. --dry-run returns before
            # the duel, so it never notifies.
            crashed_now = stop_note is None and sys.exc_info()[0] is not None
            notify_duel_done(ntfy_url, config_path, participants, turns,
                             transcript, run_started, stop_note, crashed_now,
                             model_stats=model_stats, code_runs=code_runs,
                             log_path=written_log, verdict=verdict)


if __name__ == "__main__":
    main()
