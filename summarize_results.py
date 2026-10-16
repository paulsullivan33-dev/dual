#!/usr/bin/env python3
"""summarize_results.py -- roll up run_results.log into a per-duel table.

Reads the one-block-per-duel summaries that ollama_duel.py appends to
run_results.log and prints a Markdown table (one row per duel) plus
aggregate stats: per-model average generation speed, the overall code-run
pass rate, and judge win counts. Batches of duels can be scanned without
opening each transcript log.

Usage:
    python3 summarize_results.py [path/to/run_results.log]
    # defaults to output/run_results.log
"""

import json
import os
import re
import sys

from ollama_judge import parse_winner

BAR = "-" * 72
DEFAULT_PATH = os.path.join("output", "run_results.log")

MODELS_RE = re.compile(r"^Models: (.+?) \((.+?)\) vs (.+?) \((.+?)\)$")
TURNS_RE = re.compile(r"^Turns: (\d+)/(\d+) \(\d+ replies\) in (\d+)s$")
CODE_RE = re.compile(
    r"^Code runs: (\d+) ok, (\d+) failed, (\d+) timed out, "
    r"(\d+) replies without a Python code block$")


def parse_block(block):
    """Parse one run_results.log block into a dict. Returns None for
    blocks that carry no duel (e.g. leading blank text)."""
    duel = {"scenario": "?", "stamp": "?", "models": [], "turns_done": 0,
            "turns_planned": 0, "seconds": 0, "code": None, "verdict": None,
            "winner": None, "result": "?"}
    lines = block.splitlines()
    if not any(line.startswith("Run: ") for line in lines):
        return None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Run: "):
            duel["stamp"] = line[len("Run: "):]
        elif line.startswith("Config: "):
            base = os.path.basename(line[len("Config: "):].strip())
            duel["scenario"] = (base[:-5] if base.endswith(".json")
                                else base)
        elif line.startswith("Models: "):
            m = MODELS_RE.match(line)
            if m:
                duel["models"] = [(m.group(1), m.group(2)),
                                  (m.group(3), m.group(4))]
        elif line.startswith("Turns: "):
            m = TURNS_RE.match(line)
            if m:
                duel["turns_done"] = int(m.group(1))
                duel["turns_planned"] = int(m.group(2))
                duel["seconds"] = int(m.group(3))
        elif line.startswith("Participant stats: "):
            duel["participants"] = json.loads(line[len("Participant stats: "):])
        elif line.startswith("Code runs: "):
            m = CODE_RE.match(line)
            if m:
                duel["code"] = {"ok": int(m.group(1)),
                                "failed": int(m.group(2)),
                                "timeout": int(m.group(3)),
                                "no_code": int(m.group(4))}
        elif line == "Verdict:":
            verdict_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("Result: "):
                verdict_lines.append(lines[i])
                i += 1
            duel["verdict"] = "\n".join(verdict_lines).strip()
            duel["winner"] = parse_winner(duel["verdict"])
            continue
        elif line.startswith("Result: "):
            duel["result"] = line[len("Result: "):]
        elif line == "=== Model performance ===":
            perf = {}
            i += 2  # skip the header row
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) == 6:
                    perf[parts[0]] = {"turns": int(parts[1]),
                                      "gen_tps": float(parts[2]),
                                      "prompt_tps": float(parts[3]),
                                      "tokens": int(parts[4]),
                                      "truncated": int(parts[5])}
                i += 1
            duel["perf"] = perf
            continue
        i += 1
    return duel


def parse_file(path):
    """Split run_results.log into blocks and parse each duel."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    duels = []
    for block in content.split(BAR):
        duel = parse_block(block)
        if duel is not None:
            duels.append(duel)
    return duels


def fmt_speeds(duel):
    """'qwen3:1.7b (2.1) vs smollm2:1.7b (1.5)' from the perf table."""
    perf = duel.get("perf", {})
    bits = []
    speakers = {s["index"]: s for s in duel.get("participants", [])}
    for index, (model, role) in enumerate(duel["models"]):
        speaker = speakers.get(index)
        tps = ((speaker["gen_tokens"] / speaker["gen_s"] if speaker["gen_s"] else 0.0)
               if speaker else perf.get(model, {}).get("gen_tps"))
        if speaker:
            model = f"{index + 1}: {role} [{model}]"
        bits.append(f"{model} ({tps:.1f})" if tps is not None else model)
    return " vs ".join(bits) if bits else "?"


def fmt_code(code):
    if code is None:
        return "-"
    ran = code["ok"] + code["failed"] + code["timeout"]
    return f"{code['ok']}/{ran} ok" if ran else "-"


def print_table(duels):
    print("| Date | Scenario | Models (gen tok/s) | Turns | Code | "
          "Winner | Result |")
    print("|---|---|---|---|---|---|---|")
    for d in duels:
        date = d["stamp"][:10]
        turns = f"{d['turns_done']}/{d['turns_planned']}"
        print(f"| {date} | {d['scenario']} | {fmt_speeds(d)} | {turns} | "
              f"{fmt_code(d['code'])} | {d['winner'] or '-'} | "
              f"{d['result']} |")
    print()


def print_aggregates(duels):
    ok = sum(1 for d in duels if d["result"] == "OK")
    print(f"Duels: {len(duels)} "
          f"({ok} OK, {len(duels) - ok} stopped/crashed)")
    speeds = {}
    for d in duels:
        for model, tps in ((m, d.get("perf", {}).get(m, {}).get("gen_tps"))
                           for m in dict.fromkeys(m for m, _ in d["models"])):
            if tps is not None:
                speeds.setdefault(model, []).append(tps)
    for model in sorted(speeds):
        vals = speeds[model]
        print(f"Avg gen tok/s: {model} "
              f"{sum(vals) / len(vals):.1f} ({len(vals)} duels)")
    code_ok = sum(d["code"]["ok"] for d in duels if d["code"])
    code_bad = sum(d["code"]["failed"] + d["code"]["timeout"]
                   for d in duels if d["code"])
    if code_ok + code_bad:
        print(f"Code runs: {code_ok} ok, {code_bad} failed/timed out "
              f"({100 * code_ok / (code_ok + code_bad):.0f}% pass)")
    wins = {}
    for d in duels:
        if d["winner"]:
            wins[d["winner"]] = wins.get(d["winner"], 0) + 1
    if wins:
        print("Judge wins: " + ", ".join(
            f"{name} {n}" for name, n in sorted(wins.items())))
    else:
        print("Judge wins: no judged duels in this log")


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        print(f"No results log at {path}", file=sys.stderr)
        return 1
    duels = parse_file(path)
    if not duels:
        print(f"No duels found in {path}", file=sys.stderr)
        return 1
    print_table(duels)
    print_aggregates(duels)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
