#!/usr/bin/env python3
"""Attribution-swap audit for league judging (a duel-lab tool).

Question: when a judge panel crowns a persona in a league match, is it
following the ARGUMENTS or the NAME TAGS? For every played match in a
league season, each panel seat rejudges the match twice:

  repeat arm -- the transcript exactly as the league judged it, and
  swap arm   -- the same transcript with the two contestants' names
                (and model tags) exchanged everywhere the judge can
                see them: transcript speakers, scenario persona names,
                briefs, and the topic text. Content and speaking order
                are untouched; only the labels move.

A seat whose winner NAME follows the lines it was attached to is
content-driven; a seat that crowns the same name even when that name
now sits on the other side's lines is label-driven. The repeat arm is
the noise control: if a seat disagrees with its own recorded league
verdict on an unchanged transcript, judge instability is measured
before any swap effect is claimed.

Outputs audit.json (incremental) and report.md in --out. With --sync
HOST:DIR the out dir is rsynced there at the end (league convention:
results land on the Vostro for fileList).
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollama_league as league

PLACEHOLDER = "\u0000SWAP\u0000"


def swap_names(text, name_a, name_b):
    """Exchange two names in a string (placeholder-safe)."""
    if not text:
        return text
    return (text.replace(name_a, PLACEHOLDER)
                .replace(name_b, name_a)
                .replace(PLACEHOLDER, name_b))


def swap_labels(text, name_a, name_b):
    """Exchange persona references in prose: full names and, when the
    two last tokens differ, the short forms too (scenario topics say
    'Bruno defends coffee', not 'Barista Bruno defends coffee'). Full
    names are extracted first so their tokens are not double-swapped.
    """
    if not text:
        return text
    tok_a, tok_b = name_a.split()[-1], name_b.split()[-1]
    out = (text.replace(name_a, "\u0001A\u0001")
               .replace(name_b, "\u0001B\u0001"))
    if tok_a != tok_b:
        out = re.sub(rf"\b{re.escape(tok_a)}\b", "\u0001a\u0001", out)
        out = re.sub(rf"\b{re.escape(tok_b)}\b", "\u0001b\u0001", out)
        out = (out.replace("\u0001a\u0001", tok_b)
                  .replace("\u0001b\u0001", tok_a))
    return (out.replace("\u0001A\u0001", name_b)
               .replace("\u0001B\u0001", name_a))


def transcript_pairs(entries):
    """[(name, model), (name, model)] in first-appearance order."""
    pairs = []
    for entry in entries:
        pair = (entry["speaker"], entry["model"])
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def swapped_transcript(entries, pairs):
    """Copy of the transcript with the two contestants' speaker names
    and model tags exchanged. Texts and order are untouched."""
    (name_a, model_a), (name_b, model_b) = pairs
    swap = {name_a: name_b, name_b: name_a,
            model_a: model_b, model_b: model_a}
    out = []
    for entry in entries:
        new = dict(entry)
        new["speaker"] = swap.get(entry["speaker"], entry["speaker"])
        new["model"] = swap.get(entry["model"], entry["model"])
        out.append(new)
    return out


def swapped_scenario(scen, pairs):
    """Copy of the judging scenario with the persona name tags
    exchanged (topic, persona names, and briefs) and the persona list
    order reversed, so the judge's presentation mirrors the swapped
    transcript. The sides' briefs stay with their sides."""
    (name_a, _), (name_b, _) = pairs
    scen = json.loads(json.dumps(scen))
    if scen.get("topic"):
        scen["topic"] = swap_labels(scen["topic"], name_a, name_b)
    models = scen.get("models") or []
    for persona in models:
        if persona.get("name"):
            persona["name"] = swap_names(persona["name"], name_a, name_b)
        if persona.get("system"):
            persona["system"] = swap_labels(persona["system"],
                                            name_a, name_b)
    scen["models"] = list(reversed(models))
    return scen


def parse_recorded(judge_summary):
    """Per-seat winner names from a league match record's judge
    summary ('seat: Name; seat: draw; ...'). 'draw' and 'no verdict'
    become None; the same-family marking is stripped."""
    seats = {}
    for segment in (judge_summary or "").split("; "):
        if ": " not in segment:
            continue
        seat, name = segment.split(": ", 1)
        name = name.replace(" (same family as a contestant)", "").strip()
        seats[seat] = None if name in ("draw", "no verdict") else name
    return seats


def classify(repeat_name, swap_name, pairs):
    """One seat's swap behaviour for a match."""
    if not repeat_name or not swap_name:
        return "unresolved"
    if swap_name == repeat_name:
        return "label"
    other = pairs[1][0] if repeat_name == pairs[0][0] else pairs[0][0]
    if swap_name == other:
        return "content"
    return "other"


def majority_name(names):
    """Majority of winner names (None = no verdict); split = None."""
    return league.majority(names)


def audit_match(cfg_host, league_dir, match, seats, out_dir):
    """Run both arms for one match; return its audit row."""
    match_dir = os.path.join(league_dir, "matches", match["id"])
    with open(os.path.join(league_dir, match["transcript"]),
              encoding="utf-8") as fh:
        entries = json.load(fh)
    with open(os.path.join(match_dir, "scenario.json"),
              encoding="utf-8") as fh:
        scen = json.load(fh)
    pairs = transcript_pairs(entries)
    row = {"id": match["id"], "a": match["a"], "b": match["b"],
           "pairs": pairs, "recorded_winner": match.get("winner"),
           "seats": {}}
    recorded = parse_recorded(match.get("judge"))
    arms = {}
    for arm, ents, scn in (
            ("repeat", entries, scen),
            ("swap", swapped_transcript(entries, pairs),
             swapped_scenario(scen, pairs))):
        arm_dir = os.path.join(out_dir, match["id"], arm)
        os.makedirs(arm_dir, exist_ok=True)
        tpath = os.path.join(arm_dir, "transcript.json")
        spath = os.path.join(arm_dir, "scenario.json")
        with open(tpath, "w", encoding="utf-8") as fh:
            json.dump(ents, fh, indent=2, ensure_ascii=False)
        with open(spath, "w", encoding="utf-8") as fh:
            json.dump(scn, fh, indent=2, ensure_ascii=False)
        arms[arm] = (tpath, spath)
    cfg = {"host": cfg_host}
    for seat in seats:
        model = seat["model"]
        rep, _ = league.rejudge_winner(cfg, model, arms["repeat"][1],
                                       arms["repeat"][0])
        swp, _ = league.rejudge_winner(cfg, model, arms["swap"][1],
                                       arms["swap"][0])
        row["seats"][model] = {
            "recorded": recorded.get(model),
            "repeat": rep, "swap": swp,
            "repeat_agrees_recorded": rep == recorded.get(model),
            "class": classify(rep, swp, pairs)}
        print(f"  {model}: recorded={recorded.get(model)} "
              f"repeat={rep} swap={swp} -> {row['seats'][model]['class']}",
              flush=True)
    for arm in ("recorded", "repeat", "swap"):
        names = [row["seats"][s["model"]][
            "recorded" if arm == "recorded" else arm] for s in seats]
        row[f"panel_{arm}"] = majority_name(names)
    return row


def render_report(cfg, rows):
    lines = [f"# Judge attribution-swap audit — {cfg['name']}", "",
             "Each panel seat rejudged every match twice: unchanged "
             "(repeat) and with the contestants' name tags exchanged "
             "(swap). A seat follows **content** when its winner's name "
             "moves with the lines; it follows the **label** when it "
             "crowns the same name on the other side's lines.", ""]
    tally = {"content": 0, "label": 0, "unresolved": 0, "other": 0}
    unstable = 0
    total = 0
    panel_flips = 0
    for row in rows:
        lines.append(f"## {row['id']}")
        lines.append(f"Recorded league winner: **{row['recorded_winner']}** "
                     f"(panel by name: {row['panel_recorded']})")
        lines.append("")
        lines.append("| Seat | Recorded | Repeat | Swap | Class |")
        lines.append("| --- | --- | --- | --- | --- |")
        for seat, res in row["seats"].items():
            tally[res["class"]] += 1
            total += 1
            if not res["repeat_agrees_recorded"]:
                unstable += 1
            lines.append(f"| {seat} | {res['recorded'] or '—'} | "
                         f"{res['repeat'] or '—'} | {res['swap'] or '—'} | "
                         f"{res['class']} |")
        flip = (row["panel_repeat"] != row["panel_swap"]
                and row["panel_repeat"] and row["panel_swap"])
        panel_flips += 1 if flip else 0
        lines.append("")
        lines.append(f"Panel by name: repeat {row['panel_repeat'] or '—'} → "
                     f"swap {row['panel_swap'] or '—'}"
                     f" ({'FLIPS' if flip else 'holds'})")
        lines.append("")
    lines.append("## Totals")
    lines.append(f"- Seat verdicts audited: {total}")
    lines.append(f"- Followed content: {tally['content']}")
    lines.append(f"- Followed the label: {tally['label']}")
    lines.append(f"- Unresolved (a side returned no verdict): "
                 f"{tally['unresolved']}")
    lines.append(f"- Repeat disagreed with the recorded league verdict: "
                 f"{unstable} (judge instability baseline)")
    lines.append(f"- Panel majority (by name) flipped under swap: "
                 f"{panel_flips} of {len(rows)} matches")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league-dir", required=True,
                    help="season output dir holding standings.json")
    ap.add_argument("--panel", required=True,
                    help="panel name in panels.json")
    ap.add_argument("--panels-file", default="panels.json")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", required=True, help="audit output dir")
    ap.add_argument("--sync", help="HOST:DIR to rsync --out to at the end")
    args = ap.parse_args(argv)

    with open(os.path.join(args.league_dir, "standings.json"),
              encoding="utf-8") as fh:
        state = json.load(fh)
    with open(args.panels_file, encoding="utf-8") as fh:
        panels = json.load(fh)
    seats = panels[args.panel]["seats"]
    played = [m for m in state["matches"] if m.get("status") == "played"]
    os.makedirs(args.out, exist_ok=True)
    print(f"Auditing {len(played)} played matches x {len(seats)} seats "
          f"x 2 arms from {args.league_dir}", flush=True)
    rows = []
    for match in played:
        print(f"[{match['id']}]", flush=True)
        row = audit_match(args.host, args.league_dir, match, seats,
                          args.out)
        rows.append(row)
        with open(os.path.join(args.out, "audit.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"league": state.get("league"), "rows": rows},
                      fh, indent=2, ensure_ascii=False)
    report = render_report({"name": state.get("league")}, rows)
    with open(os.path.join(args.out, "report.md"), "w",
              encoding="utf-8") as fh:
        fh.write(report)
    print(f"Report: {os.path.join(args.out, 'report.md')}", flush=True)
    if args.sync:
        host, _, remote = args.sync.partition(":")
        proc = subprocess.run(
            ["rsync", "-a", "-e", "ssh -o BatchMode=yes",
             args.out.rstrip("/") + "/", f"paul@{host}:{remote}/"],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"warning: sync failed ({proc.stderr.strip()[:200]})",
                  flush=True)
        else:
            print(f"Synced to {args.sync}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
