#!/usr/bin/env python3
"""rejudge.py -- re-score a finished duel with a different judge model.

The duel itself is not re-run. The tool rebuilds the exact judge request
with dual's own ollama_judge.build_judge_messages() -- same scenario
prompt, same transcript, same run reports -- and changes only the judge
model (and optionally its host). That makes verdicts comparable across
judges: any difference in outcome is the judge, not the duel.

Inputs are a scenario file (for the topic, participants, and tailored
judge prompt) plus one or more saved transcripts:

  * a --save-json transcript (exact: the format the duel wrote), or
  * a duel .log file (near-exact: speaker blocks are parsed back out;
    code-run reports are recovered from the log, thermal-pause and
    waiting lines are dropped).

The original transcript and verdict are never modified. Each rejudge
writes a sibling file, <transcript stem>.rejudge-<model>.txt, carrying
the judge identity, date, full verdict, and parsed winner, so verdicts
from different judges can be diffed side by side.

Usage:
  python3 rejudge.py --model qwen3:14b SCENARIO.json TRANSCRIPT [...]
  python3 rejudge.py --model qwen3:14b --judge-host http://box:11434 \\
      scenarios/colony_caretakers.json output/logs/20261017-....log
  python3 rejudge.py --model qwen3:14b --logs-dir output/logs-2026-10-17
  python3 rejudge.py --model qwen3:14b --dry-run SCENARIO.json TRANSCRIPT

With --logs-dir, each *.log in the directory is matched to a scenario by
filename (<timestamp>-<scenario>.log) under --scenarios-dir (default
"scenarios"). Judging is best-effort per duel, as in the duel itself:
a skipped (context) or failed (host) judge is reported, never hidden,
and the exit status is 1 when any requested rejudge produced no verdict.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from ollama_common import OllamaError, call_chat
from ollama_judge import (JudgeContextExceeded, build_judge_messages,
                          judge_context_window, normalize_judge,
                          parse_winner, run_judge)

TURN_HEADER_RE = re.compile(r"^\[(?P<model>.+?) as (?P<name>.+?)\]"
                            r"\s+\(turn \d+\)\s*$")
LOG_STOP_MARKERS = ("--- JUDGE:", "--- JUDGE SKIPPED", "--- JUDGE FAILED",
                    "=== Model performance ===", "DUEL END")
NOISE_PREFIXES = ("(waiting for reply", "CPU at ", "CPU cooled to ")


def load_scenario(path):
    """Read a scenario file: (topic, participants, raw judge config).

    Participants come from the scenario's "models" entries -- the names
    the judge prompt addresses. A profile may have overridden the models
    that actually ran; the names are the scenario's either way.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    participants = [{"name": entry["name"], "model": entry["model"]}
                    for entry in data["models"]]
    return data.get("topic", ""), participants, data.get("judge")


def load_transcript_json(path, participants):
    """Load a --save-json transcript: [{"speaker", "model", "text"}].

    Returns (transcript, None): run reports are not saved in this format,
    so the rejudge sees the replies only -- noted in the output header.
    """
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)
    index = {p["name"]: i for i, p in enumerate(participants)}
    transcript = []
    for entry in entries:
        name = entry["speaker"]
        if name not in index:
            raise ValueError(f"{path}: speaker {name!r} is not one of "
                             f"{sorted(index)}")
        transcript.append((index[name], entry["text"]))
    return transcript, None


def _clean_block(lines):
    """Join a parsed log block back into text, dropping stream noise."""
    kept = [line for line in lines
            if not line.strip().startswith(NOISE_PREFIXES)]
    return "\n".join(kept).strip()


def parse_log(path, participants):
    """Parse a duel .log back into (transcript, run_reports).

    Turns are the "[model as Name] (turn N)" blocks; each reply follows
    its "--- REPLY ---" marker and an optional "--- CODE RUN ---" block
    becomes that turn's run report, as the live judge saw it. Parsing
    stops at the original judge section -- the original verdict is not
    part of the transcript.
    """
    index = {p["name"]: i for i, p in enumerate(participants)}
    transcript = []
    run_reports = {}
    current = None  # (participant_index, reply_lines, report_lines, mode)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    def flush():
        if current is None:
            return
        idx, reply_lines, report_lines, _mode = current
        transcript.append((idx, _clean_block(reply_lines)))
        if report_lines:
            run_reports[len(transcript) - 1] = _clean_block(report_lines)

    for line in lines:
        if line.startswith(LOG_STOP_MARKERS):
            break
        header = TURN_HEADER_RE.match(line)
        if header:
            flush()
            name = header.group("name")
            if name not in index:
                raise ValueError(f"{path}: speaker {name!r} is not one "
                                 f"of {sorted(index)}")
            current = (index[name], [], [], "await")
            continue
        if current is None:
            continue
        idx, reply_lines, report_lines, mode = current
        if line.strip() == "--- REPLY ---":
            current = (idx, reply_lines, report_lines, "reply")
        elif line.strip() == "--- CODE RUN ---":
            current = (idx, reply_lines, report_lines, "report")
        elif line.strip() and set(line.strip()) == {"="}:
            continue  # the next turn's opening separator, not content
        elif mode == "reply":
            reply_lines.append(line)
        elif mode == "report":
            report_lines.append(line)
    flush()
    return transcript, (run_reports or None)


def load_transcript(path, participants):
    """Dispatch on the transcript's format: .json exact, .log parsed."""
    if path.endswith(".json"):
        return load_transcript_json(path, participants)
    return parse_log(path, participants)


def model_tag(model):
    """Filesystem-safe judge tag: qwen3:14b -> qwen3_14b."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def rejudge_output_path(transcript_path, model):
    stem = os.path.splitext(transcript_path)[0]
    return f"{stem}.rejudge-{model_tag(model)}.txt"


def rejudge_one(scenario_path, transcript_path, model, judge_host=None,
                timeout=1800, dry_run=False, chat=call_chat):
    """Rejudge one transcript. Returns a status string:
    "OK", "SKIPPED", "FAILED", or "NO JUDGE" (scenario has none).
    Prints what it did; the verdict lands in a sibling .txt file.
    """
    topic, participants, raw_judge = load_scenario(scenario_path)
    if raw_judge is None:
        print(f"{transcript_path}: scenario {scenario_path} has no "
              f"judge configured -- nothing to re-run.")
        return "NO JUDGE"
    judge = normalize_judge(raw_judge)
    original_model = judge["model"]
    judge["model"] = model
    if judge_host:
        judge["host"] = judge_host
    host = judge.get("host") or "http://localhost:11434"
    transcript, run_reports = load_transcript(transcript_path, participants)
    if dry_run:
        messages = build_judge_messages(judge, topic, participants,
                                        transcript, run_reports)
        try:
            window, required = judge_context_window(judge, messages)
        except JudgeContextExceeded as exc:
            print(f"{transcript_path}: would SKIP ({exc})")
            return "SKIPPED"
        print(f"{transcript_path}: would judge with {model} @ {host} "
              f"(context {window}, est. budget {required}, "
              f"{len(transcript)} replies)")
        return "OK"
    try:
        verdict, num_ctx, required = run_judge(
            host, judge, topic, participants, transcript, timeout, chat,
            run_reports=run_reports)
    except JudgeContextExceeded as exc:
        print(f"{transcript_path}: SKIPPED ({model} @ {host}): {exc}")
        return "SKIPPED"
    except OllamaError as exc:
        print(f"{transcript_path}: FAILED ({model} @ {host}): {exc}")
        return "FAILED"
    out_path = rejudge_output_path(transcript_path, model)
    winner = parse_winner(verdict)
    header = [
        f"Rejudge of: {transcript_path}",
        f"Scenario: {scenario_path}",
        f"Judge: {model} @ {host} (original judge: {original_model})",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Judge context: {num_ctx} tokens; estimated budget {required}",
        f"Winner: {winner if winner is not None else '(none parsed)'}",
        "",
    ]
    if run_reports is None:
        header.insert(5, "Note: no code-run reports in this transcript "
                         "source; the judge saw replies only.")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n=== Verdict ===\n" + verdict + "\n")
    print(f"{transcript_path}: rejudged by {model} @ {host} -> "
          f"{out_path} (winner: {winner})")
    return "OK"


def scenario_for_log(log_path, scenarios_dir):
    """Match a batch log filename to its scenario file, or None.

    Log names are <YYYYMMDD>-<HHMMSS>-<scenario>.log; the scenario is
    scenarios/<scenario>.json. A plain <scenario>.log also matches.
    """
    stem = os.path.splitext(os.path.basename(log_path))[0]
    match = re.match(r"^\d{8}-\d{6}-(.+)$", stem)
    name = match.group(1) if match else stem
    candidate = os.path.join(scenarios_dir, name + ".json")
    return candidate if os.path.exists(candidate) else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-score finished duels with a different judge "
                    "model, without re-running the duels.")
    parser.add_argument("--model", required=True,
                        help="judge model to re-score with, e.g. qwen3:14b")
    parser.add_argument("--judge-host", default=None,
                        help="Ollama endpoint for the judge "
                             "(default: the scenario judge's host)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-judge-call timeout in seconds "
                             "(default: 1800)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the judge request and report the "
                             "context window, but call nothing")
    parser.add_argument("--logs-dir", default=None,
                        help="rejudge every *.log in this directory, "
                             "matching scenarios by filename")
    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="scenario directory for --logs-dir matching "
                             "(default: scenarios)")
    parser.add_argument("scenario", nargs="?",
                        help="scenario JSON for the transcript(s)")
    parser.add_argument("transcripts", nargs="*",
                        help="saved transcripts (.json or .log)")
    args = parser.parse_args(argv)

    jobs = []  # (scenario_path, transcript_path)
    if args.logs_dir:
        for name in sorted(os.listdir(args.logs_dir)):
            if not name.endswith(".log"):
                continue
            log_path = os.path.join(args.logs_dir, name)
            scenario = scenario_for_log(log_path, args.scenarios_dir)
            if scenario is None:
                print(f"{log_path}: no matching scenario in "
                      f"{args.scenarios_dir} -- skipped.")
                continue
            jobs.append((scenario, log_path))
    else:
        if not args.scenario or not args.transcripts:
            parser.error("give a scenario and at least one transcript, "
                         "or use --logs-dir")
        jobs = [(args.scenario, path) for path in args.transcripts]

    statuses = [rejudge_one(scenario, transcript, args.model,
                            args.judge_host, args.timeout, args.dry_run)
                for scenario, transcript in jobs]
    return 0 if all(status == "OK" for status in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
