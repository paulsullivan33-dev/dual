"""ollama_reporting.py -- what ollama_duel.py reports when a duel ends.

The per-model speed table, the one-block summary appended to
run_results.log, and the optional ntfy completion notice (with the URL
looked up in ~/.dual.conf when not given). Kept separate from the duel loop
so ollama_duel.py stays a manageable size.
"""

import json
import os
import sys
from datetime import datetime


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


def format_run_summary(run_started, config_path, participants, turns,
                       transcript, model_stats, stop_note, crashed=False,
                       code_runs=None, log_path=None, verdict=None):
    """Build the one-block summary appended to run_results.log after a duel.

    Always carries the date/time, config, models, and how many of the
    requested turns completed. A finished duel gets the per-model stats
    table; a duel that stopped early gets the error message instead, so a
    batch of runs can be scanned without opening each transcript log.
    `crashed` covers an unexpected exception escaping the turn loop (the
    traceback itself goes to the console); pass the message as stop_note.
    `code_runs` is the run_code tally line, when code running was on, and
    `log_path` the full path of this run's transcript log, when one was kept.
    `verdict` is the judge's verdict text, when a judge scored the duel.
    """
    bar = "-" * 72
    stamp = run_started.strftime("%Y-%m-%d %H:%M:%S")
    duration_s = (datetime.now() - run_started).total_seconds()
    a, b = participants
    lines = [
        bar,
        f"Run: {stamp}",
        f"Config: {config_path}",
        *([f"Log: {log_path}"] if log_path else []),
        f"Models: {a['model']} ({a['name']}) vs {b['model']} ({b['name']})",
        f"Turns: {len(transcript)}/{turns} "
        f"({len(transcript)} replies) in {duration_s:.0f}s",
    ]
    if code_runs:
        lines.append(code_runs)
    if verdict:
        lines.append("Verdict:")
        lines.extend(verdict.splitlines())
    if crashed:
        lines.append("Result: CRASHED")
        lines.append(stop_note.strip())
    elif stop_note:
        lines.append("Result: STOPPED EARLY")
        lines.append(stop_note.strip())
    else:
        lines.append("Result: OK")
        stats_table = format_duel_stats(model_stats)
        if stats_table:
            lines.append(stats_table)
    lines.append(bar)
    return "\n".join(lines) + "\n"


def append_run_summary(path, entry):
    """Append one summary block to the results log. Best-effort: a bad
    path warns on stderr instead of failing the run, since the summary is
    a convenience -- the transcript log and save_json are already written."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError as e:
        print(f"Warning: could not write run summary to {path}: {e}",
              file=sys.stderr)


def load_default_ntfy_url():
    """Read the ntfy topic URL from the default config file ~/.dual.conf.

    The file is a JSON object, e.g. {"ntfy_url": "https://ntfy.sh/my-topic"}.
    Returns "" when the file is missing, unreadable, not JSON, or has no
    usable ntfy_url -- the caller then skips notifications silently.
    """
    path = os.path.join(os.path.expanduser("~"), ".dual.conf")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    if isinstance(data, dict):
        url = data.get("ntfy_url")
        return url if isinstance(url, str) and url.strip() else ""
    return ""


def format_token_stats_line(model_stats):
    """One-line per-model token summary for notifications, e.g.
    'tokens: m1 100 tok @ 2.0 tok/s, m2 80 tok @ 1.6 tok/s'.
    Returns "" when no stats are available."""
    if not model_stats:
        return ""
    parts = []
    for model, s in model_stats.items():
        gen_tps = s["gen_tokens"] / s["gen_s"] if s["gen_s"] > 0 else 0.0
        parts.append(f"{model} {s['gen_tokens']} tok @ {gen_tps:.1f} tok/s")
    return "tokens: " + ", ".join(parts)


def notify_duel_done(url, config_path, participants, turns, transcript,
                     run_started, stop_note, crashed, model_stats=None,
                     code_runs=None, log_path=None, verdict=None):
    """POST a short completion notice to ntfy. Best-effort: any failure
    warns on stderr and never fails the run."""
    import socket
    import urllib.request  # stdlib; imported here so --help stays instant
    scenario = os.path.basename(config_path)
    host = socket.gethostname()
    duration_s = (datetime.now() - run_started).total_seconds()
    a, b = participants
    if crashed:
        title, result, tags = (f"duel crashed: {scenario} [{host}]",
                               "CRASHED", "warning")
    elif stop_note:
        title, result, tags = (f"duel stopped early: {scenario} [{host}]",
                               "STOPPED EARLY", "warning")
    else:
        title, result, tags = (f"duel finished: {scenario} [{host}]",
                               "OK", "tada")
    body = "\n".join([
        f"{scenario}: {result}",
        f"{a['model']} ({a['name']}) vs {b['model']} ({b['name']})",
        f"{len(transcript)}/{turns} turns in {duration_s:.0f}s",
        *([f"Log: {log_path}"] if log_path else []),
    ])
    stats_line = format_token_stats_line(model_stats)
    if stats_line:
        body += "\n" + stats_line
    if code_runs:
        body += "\n" + code_runs
    if verdict:
        # The notice stays short: just the declared winner line.
        winner = next((ln for ln in verdict.splitlines()
                       if ln.strip().upper().startswith("WINNER:")), None)
        body += "\n" + (winner.strip() if winner else "Verdict recorded")
    if stop_note and not crashed:
        body += "\n" + stop_note.strip().splitlines()[0][:200]
    try:
        req = urllib.request.Request(url, data=body.encode("utf-8"),
                                     method="POST")
        req.add_header("Title", title)
        req.add_header("Tags", tags)
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001 -- notification must not fail the run
        print(f"Warning: ntfy notification failed: {e}", file=sys.stderr)
