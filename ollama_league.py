#!/usr/bin/env python3
"""ollama_league.py -- a season layer over ollama_duel.py.

Registered models play a shared battery of scenarios round-robin. Each
match is an ordinary duel run with --no-judge; a uniform league judge
then scores the saved transcript with rejudge.py, so every match in the
season is decided by the same standard. Standings persist in
standings.json (points 3/1/0, tokens, cloud-equivalent cost at the
median reference rate) and are rendered to standings.md after every
match. Results can be synced to a folder on another host after each
match (the intended home is a Vostro share Paul browses with fileList).

A league is described by one JSON config:

    {
      "name": "Season 1",
      "host": "http://192.168.1.236:11434",
      "contestants": ["qwen3:4b", "qwen3:8b", "mistral:7b"],
      "scenarios": ["scenarios/small_coffee_vs_tea.json"],
      "judge": {"mode": "single", "model": "qwen3:14b"},
      "output_dir": "league/season1",
      "sync": {"host": "192.168.1.119",
               "dir": "/myData/share1/sabnzbd/duel-league"}
    }

Judge modes: "single" uses one judge model; "panel" rejudges with
every seat of a named panel from panels.json and takes the majority
(a split panel is a draw). A seat whose family matches a contestant's
is allowed but marked, per the standing panel rule: warn, never refuse.

State is resumable: fixtures already recorded in standings.json are
skipped, so an interrupted season continues where it stopped. A match
whose duel or judging fails is recorded as unplayed with the reason,
scores nothing, and is retried on the next run.

Usage:
    python3 ollama_league.py --config league.json [--rounds N]
        [--max-matches N] [--dry-run] [--no-sync]
"""

import argparse
import glob
import html
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama_reporting import CLOUD_RATE_PRESETS, cloud_cost, format_cost
from summarize_results import parse_file

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
POINTS_WIN, POINTS_DRAW = 3, 1

# Model family by name prefix, for the panel same-family warning.
FAMILIES = {
    "qwen": "alibaba", "gemma": "google", "mistral": "mistral",
    "granite": "ibm", "llama": "meta", "smollm": "huggingface",
    "dolphin": "cognitive", "phi": "microsoft", "deepseek": "deepseek",
    "kimi": "moonshot", "nomic": "nomic",
}


def family_of(model):
    """Best-effort family for a model name ('huihui_ai/qwen3...' ->
    'alibaba', 'gemma4:latest' -> 'google'); 'unknown' when unmapped."""
    base = model.rsplit("/", 1)[-1].split(":")[0].lower()
    for prefix, family in FAMILIES.items():
        if base.startswith(prefix):
            return family
    return "unknown"


def slug(model):
    """Filesystem-safe short tag for a model name."""
    out = "".join(ch if ch.isalnum() else "-" for ch in model.lower())
    return "-".join(part for part in out.split("-") if part)


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    contestants = cfg.get("contestants") or []
    if len(contestants) < 2:
        sys.exit("league config needs at least two contestants")
    if len(set(contestants)) != len(contestants):
        sys.exit("league config has duplicate contestants")
    scenarios = cfg.get("scenarios") or []
    if not scenarios:
        sys.exit("league config needs at least one scenario")
    for scen in scenarios:
        if not os.path.exists(os.path.join(REPO_DIR, scen)):
            sys.exit(f"scenario not found: {scen}")
    judge = cfg.get("judge") or {}
    mode = judge.get("mode", "single")
    if mode == "single" and not judge.get("model"):
        sys.exit('judge mode "single" needs judge.model')
    if mode == "panel" and not judge.get("panel"):
        sys.exit('judge mode "panel" needs judge.panel (a name in panels.json)')
    if mode not in ("single", "panel"):
        sys.exit(f"unknown judge mode: {mode}")
    cfg.setdefault("name", "Duel League")
    cfg.setdefault("host", "http://localhost:11434")
    cfg.setdefault("output_dir", "league-output")
    return cfg


def build_fixtures(cfg, rounds):
    """Ordered fixture list: round-major, then scenario, then pairing.
    Each fixture: {id, round, scenario, a, b} with model names a/b."""
    fixtures = []
    for rnd in range(1, rounds + 1):
        for scen in cfg["scenarios"]:
            stem = os.path.splitext(os.path.basename(scen))[0]
            for a, b in itertools.combinations(cfg["contestants"], 2):
                fixtures.append({
                    "id": f"r{rnd}-{stem}-{slug(a)}-vs-{slug(b)}",
                    "round": rnd, "scenario": scen, "a": a, "b": b,
                })
    return fixtures


def empty_table(models):
    return {m: {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                "points": 0, "tokens_in": 0, "tokens_out": 0}
            for m in models}


def load_state(cfg):
    path = os.path.join(cfg["output_dir"], "standings.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"league": cfg["name"], "updated": None,
            "table": empty_table(cfg["contestants"]), "matches": []}


def save_state(cfg, state):
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    path = os.path.join(cfg["output_dir"], "standings.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


def played_ids(state):
    return {m["id"] for m in state["matches"] if m.get("status") == "played"}


def apply_result(state, record):
    """Fold one played match record into the standings table."""
    table = state["table"]
    for model in (record["a"], record["b"]):
        row = table.setdefault(model, {"played": 0, "won": 0, "drawn": 0,
                                       "lost": 0, "points": 0,
                                       "tokens_in": 0, "tokens_out": 0})
        row["played"] += 1
    table[record["a"]]["tokens_in"] += record["tokens"][record["a"]]["in"]
    table[record["a"]]["tokens_out"] += record["tokens"][record["a"]]["out"]
    table[record["b"]]["tokens_in"] += record["tokens"][record["b"]]["in"]
    table[record["b"]]["tokens_out"] += record["tokens"][record["b"]]["out"]
    winner = record.get("winner")
    if winner in (record["a"], record["b"]):
        loser = record["b"] if winner == record["a"] else record["a"]
        table[winner]["won"] += 1
        table[winner]["points"] += POINTS_WIN
        table[loser]["lost"] += 1
    else:
        for model in (record["a"], record["b"]):
            table[model]["drawn"] += 1
            table[model]["points"] += POINTS_DRAW


def majority(winners):
    """Majority of a list of winner model names (None = no verdict).
    Returns the winning model, or None for a tie / no verdicts."""
    votes = [w for w in winners if w]
    if not votes:
        return None
    for cand in set(votes):
        if votes.count(cand) > len(votes) / 2:
            return cand
    return None


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO_DIR, capture_output=True,
                          text=True, **kw)


def judging_scenario(cfg, fixture, match_dir):
    """Scenario copy for rejudge: the original file, plus an injected
    judge block when the scenario has none (rejudge needs one to know
    the judging setup; the league's --model overrides who judges).
    Returns the path rejudge should read."""
    src = os.path.join(REPO_DIR, fixture["scenario"])
    with open(src, encoding="utf-8") as fh:
        scen = json.load(fh)
    if scen.get("judge"):
        return src
    judge = cfg["judge"]
    scen["judge"] = {"model": judge.get("model") or "qwen3:14b",
                     "host": cfg["host"]}
    path = os.path.join(match_dir, "scenario.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scen, fh, indent=2, ensure_ascii=False)
    return path


def parse_winner_name(text):
    for line in text.splitlines():
        if line.startswith("Winner:"):
            name = line[len("Winner:"):].strip()
            return None if name.startswith("(none") else name
    return None


def winner_model(winner_name, models):
    """Map a judge's winner (a participant name) to its model, using
    the results block's [(name, model), ...] pairing. A winner given
    as a model name is accepted too. None when unmappable."""
    if not winner_name:
        return None
    for name, model in models:
        if winner_name == name or winner_name == model:
            return model
    return None


def rejudge_winner(cfg, judge_model, scenario_path, transcript_path):
    """Run rejudge.py with one judge model; return (winner_name, note).
    winner_name is None when judging failed or parsed no winner."""
    proc = run([sys.executable, "rejudge.py", "--model", judge_model,
                "--judge-host", cfg["host"], scenario_path, transcript_path])
    stem = os.path.splitext(transcript_path)[0]
    candidates = sorted(glob.glob(stem + ".rejudge-*.txt"),
                        key=os.path.getmtime)
    if not candidates:
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        return None, f"rejudge produced no verdict ({tail[-1] if tail else 'no output'})"
    with open(candidates[-1], encoding="utf-8") as fh:
        return parse_winner_name(fh.read()), ""


def judge_match(cfg, fixture, scenario_path, transcript_path, models):
    """Decide a match. Returns (winner_model_or_None, judge_summary).
    Panel mode: every seat rejudges; majority wins; a split is a draw.
    Seats sharing a family with a contestant are marked, not refused."""
    judge = cfg["judge"]
    if judge.get("mode") == "panel":
        with open(os.path.join(REPO_DIR, "panels.json"),
                  encoding="utf-8") as fh:
            panels = json.load(fh)
        panel = panels[judge["panel"]]
        seat_results, notes = [], []
        fams = {family_of(fixture["a"]), family_of(fixture["b"])}
        for seat in panel["seats"]:
            name, note = rejudge_winner(cfg, seat["model"], scenario_path,
                                        transcript_path)
            seat_results.append(winner_model(name, models))
            mark = " (same family as a contestant)" \
                if seat.get("family") in fams else ""
            notes.append(f"{seat['model']}: {name or 'no verdict'}{mark}")
            if note:
                notes.append(f"{seat['model']} error: {note}")
        return majority(seat_results), "; ".join(notes)
    name, note = rejudge_winner(cfg, judge["model"], scenario_path,
                                transcript_path)
    if note:
        return None, note
    return winner_model(name, models), f"{judge['model']}: {name or 'no verdict'}"


def participant_pairs(block):
    """[(name, model), ...] for a parsed results block, taken from the
    Participant stats entries, which carry explicit name/model keys.
    (The block's "models" list is (model, role) ordered — do not use
    it for name mapping.)"""
    return [(p["name"], p["model"]) for p in block["participants"]]


def play_match(cfg, fixture, runner=run):
    """Play and judge one fixture; return its match record."""
    out_dir = cfg["output_dir"]
    match_dir = os.path.join(out_dir, "matches", fixture["id"])
    os.makedirs(match_dir, exist_ok=True)
    transcript = os.path.join(match_dir, "transcript.json")
    results_log = os.path.join(out_dir, "run_results.log")
    record = {"id": fixture["id"], "round": fixture["round"],
              "scenario": os.path.basename(fixture["scenario"]),
              "a": fixture["a"], "b": fixture["b"],
              "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "status": "unplayed", "winner": None, "judge": "",
              "tokens": {}, "note": ""}
    cmd = [sys.executable, "ollama_duel.py", fixture["scenario"],
           "--model-a", fixture["a"], "--model-b", fixture["b"],
           "--host", cfg["host"], "--no-judge", "--no-ntfy", "--no-display",
           "--save-json", transcript,
           "--log-file", os.path.join(match_dir, "duel.log"),
           "--results-log", results_log]
    for opt in ("turns", "max_tokens", "timeout"):
        if cfg.get(opt) is not None:
            cmd += [f"--{opt.replace('_', '-')}", str(cfg[opt])]
    proc = runner(cmd)
    if proc.returncode != 0 or not os.path.exists(transcript):
        record["note"] = f"duel exited {proc.returncode}"
        return record
    duels = parse_file(results_log)
    block = duels[-1] if duels else None
    if not block or not block.get("participants"):
        record["note"] = "no parseable result block for the duel"
        return record
    models = participant_pairs(block)
    tokens = {}
    for speaker in block["participants"]:
        tokens[speaker["model"]] = {"in": speaker.get("prompt_tokens", 0),
                                    "out": speaker.get("gen_tokens", 0)}
    if fixture["a"] not in tokens or fixture["b"] not in tokens:
        record["note"] = "result block models do not match the fixture"
        return record
    record["tokens"] = tokens
    scenario_path = judging_scenario(cfg, fixture, match_dir)
    winner, summary = judge_match(cfg, fixture, scenario_path,
                                  transcript, models)
    record["judge"] = summary
    record["winner"] = winner
    record["status"] = "played"
    record["transcript"] = os.path.join("matches", fixture["id"],
                                        "transcript.json")
    return record


def sorted_table(state):
    return sorted(state["table"].items(),
                  key=lambda kv: (-kv[1]["points"], -kv[1]["won"],
                                  kv[1]["tokens_in"] + kv[1]["tokens_out"],
                                  kv[0]))


TABLE_LEGEND = ("Pos = position · P = played · W = won · D = drawn · "
                "L = lost · Pts = points (3 for a win, 1 for a draw, "
                "0 for a loss) · Tokens in = prompt tokens read · "
                "Tokens out = tokens generated · Cloud-equiv = what "
                "those tokens would cost at median cloud rates")


def render_standings(cfg, state):
    preset = CLOUD_RATE_PRESETS["median"]
    lines = [f"# {cfg['name']} — standings", "",
             f"Updated: {state.get('updated') or '-'}",
             f"Judge: {cfg['judge'].get('mode', 'single')} "
             f"{cfg['judge'].get('model') or cfg['judge'].get('panel', '')}",
             f"Cloud-equivalent cost at median rates "
             f"(${preset['input']} in / ${preset['output']} out per 1M tokens).",
             "",
             "| Pos | Model | P | W | D | L | Pts | Tokens in | Tokens out | Cloud-equiv |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for pos, (model, row) in enumerate(sorted_table(state), 1):
        cost = format_cost(cloud_cost(row["tokens_in"], row["tokens_out"],
                                      preset))
        lines.append(f"| {pos} | {model} | {row['played']} | {row['won']} "
                     f"| {row['drawn']} | {row['lost']} | {row['points']} "
                     f"| {row['tokens_in']:,} | {row['tokens_out']:,} | {cost} |")
    lines += ["", f"Legend: {TABLE_LEGEND}"]
    played = [m for m in state["matches"] if m.get("status") == "played"]
    lines += ["", f"## Matches ({len(played)} played)", ""]
    for m in reversed(played[-25:]):
        outcome = m["winner"] if m.get("winner") else "draw"
        lines.append(f"- {m['date']} · {m['scenario']} · "
                     f"{m['a']} vs {m['b']} → **{outcome}** "
                     f"(judge: {m['judge']})")
    unplayed = [m for m in state["matches"] if m.get("status") != "played"]
    if unplayed:
        lines += ["", "## Unplayed (failed; retried next run)", ""]
        for m in unplayed[-10:]:
            lines.append(f"- {m['id']}: {m.get('note') or 'unknown failure'}")
    return "\n".join(lines) + "\n"


_PAGE_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
  sans-serif; margin: 2em auto; max-width: 920px; padding: 0 1em;
  line-height: 1.45; color: #1a1a1a; background: #fff; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.35em 0.6em;
  text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {
  text-align: left; }
thead th { background: #f0f0f0; }
tbody tr:nth-child(even) { background: #fafafa; }
.meta { color: #555; }
.winner { font-weight: bold; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #16181d; }
  th, td { border-color: #444; }
  thead th { background: #24262c; }
  tbody tr:nth-child(even) { background: #1c1e24; }
  .meta { color: #9a9a9a; } }
"""


def render_standings_html(cfg, state):
    """The standings as a standalone HTML page: same data as
    render_standings, inline CSS and no external assets, so it renders
    in any browser served straight off the file share (no markdown
    extension needed)."""
    esc = html.escape
    preset = CLOUD_RATE_PRESETS["median"]
    judge = cfg["judge"].get("model") or cfg["judge"].get("panel", "")
    parts = ["<!DOCTYPE html>", '<html lang="en">', "<head>",
             '<meta charset="utf-8">',
             f"<title>{esc(cfg['name'])} — standings</title>",
             f"<style>{_PAGE_CSS}</style>", "</head>", "<body>",
             f"<h1>{esc(cfg['name'])} — standings</h1>",
             f'<p class="meta">Updated: {esc(state.get("updated") or "-")}'
             f"<br>Judge: {esc(cfg['judge'].get('mode', 'single'))} "
             f"{esc(judge)}<br>Cloud-equivalent cost at median rates "
             f"(${preset['input']} in / ${preset['output']} out per 1M "
             "tokens).</p>",
             "<table>", "<thead><tr><th>Pos</th><th>Model</th><th>P</th>"
             "<th>W</th><th>D</th><th>L</th><th>Pts</th><th>Tokens in</th>"
             "<th>Tokens out</th><th>Cloud-equiv</th></tr></thead>",
             "<tbody>"]
    for pos, (model, row) in enumerate(sorted_table(state), 1):
        cost = format_cost(cloud_cost(row["tokens_in"], row["tokens_out"],
                                      preset))
        cells = [str(pos), esc(model), str(row["played"]), str(row["won"]),
                 str(row["drawn"]), str(row["lost"]), str(row["points"]),
                 f"{row['tokens_in']:,}", f"{row['tokens_out']:,}", cost]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells)
                     + "</tr>")
    parts += ["</tbody>", "</table>",
              f'<p class="meta">Legend: {esc(TABLE_LEGEND)}</p>']
    played = [m for m in state["matches"] if m.get("status") == "played"]
    parts.append(f"<h2>Matches ({len(played)} played)</h2>")
    parts.append("<ul>")
    for m in reversed(played[-25:]):
        outcome = m["winner"] if m.get("winner") else "draw"
        parts.append(f"<li>{esc(m['date'])} · {esc(m['scenario'])} · "
                     f"{esc(m['a'])} vs {esc(m['b'])} → "
                     f'<span class="winner">{esc(outcome)}</span> '
                     f"(judge: {esc(m['judge'])})</li>")
    parts.append("</ul>")
    unplayed = [m for m in state["matches"] if m.get("status") != "played"]
    if unplayed:
        parts.append("<h2>Unplayed (failed; retried next run)</h2>")
        parts.append("<ul>")
        for m in unplayed[-10:]:
            parts.append(f"<li>{esc(m['id'])}: "
                         f"{esc(m.get('note') or 'unknown failure')}</li>")
        parts.append("</ul>")
    parts += ["</body>", "</html>"]
    return "\n".join(parts) + "\n"


def sync_results(cfg):
    sync = cfg.get("sync")
    if not sync:
        return
    dest = f"paul@{sync['host']}:{sync['dir']}/"
    proc = run(["rsync", "-a", "--timeout=60", "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=15",
                cfg["output_dir"] + "/", dest])
    if proc.returncode != 0:
        print(f"warning: sync to {dest} failed "
              f"({proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'rsync error'})")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run a duel league season over ollama_duel.py.")
    ap.add_argument("--config", required=True, help="league config JSON")
    ap.add_argument("--rounds", type=int, default=None,
                    help="season rounds (default: config rounds or 1)")
    ap.add_argument("--max-matches", type=int, default=None,
                    help="play at most this many new matches this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the fixture list and judge plan; play nothing")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not sync results to the configured host")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["output_dir"] = os.path.join(REPO_DIR, cfg["output_dir"]) \
        if not os.path.isabs(cfg["output_dir"]) else cfg["output_dir"]
    rounds = args.rounds or cfg.get("rounds") or 1
    fixtures = build_fixtures(cfg, rounds)
    state = load_state(cfg)
    done = played_ids(state)
    todo = [f for f in fixtures if f["id"] not in done]
    judge = cfg["judge"]
    print(f"{cfg['name']}: {len(fixtures)} fixtures over {rounds} round(s); "
          f"{len(done)} already played, {len(todo)} to play. "
          f"Judge: {judge.get('mode', 'single')} "
          f"{judge.get('model') or judge.get('panel', '')}.")
    if args.dry_run:
        for f in todo:
            print(f"  {f['id']}: {f['a']} vs {f['b']} on {f['scenario']}")
        return 0
    played_now = 0
    for fixture in todo:
        if args.max_matches is not None and played_now >= args.max_matches:
            break
        print(f"[{fixture['id']}] {fixture['a']} vs {fixture['b']} "
              f"({fixture['scenario']}) ...", flush=True)
        record = play_match(cfg, fixture)
        state["matches"] = [m for m in state["matches"]
                            if m["id"] != record["id"]]
        state["matches"].append(record)
        if record["status"] == "played":
            apply_result(state, record)
            played_now += 1
            print(f"    -> {record['winner'] or 'draw'} "
                  f"(judge: {record['judge']})")
        else:
            print(f"    -> unplayed: {record['note']}")
        save_state(cfg, state)
        md = render_standings(cfg, state)
        with open(os.path.join(cfg["output_dir"], "standings.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(md)
        try:
            page = render_standings_html(cfg, state)
            with open(os.path.join(cfg["output_dir"], "standings.html"),
                      "w", encoding="utf-8") as fh:
                fh.write(page)
        except Exception as exc:  # the .md stays the primary artifact
            print(f"    warning: standings.html render failed: {exc}")
        if not args.no_sync:
            sync_results(cfg)
    print(f"Done: {played_now} match(es) played this run. "
          f"Standings: {os.path.join(cfg['output_dir'], 'standings.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
