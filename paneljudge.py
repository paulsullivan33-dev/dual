#!/usr/bin/env python3
"""paneljudge.py -- re-score a finished duel with a panel of judge models.

One judge gives a verdict; a panel gives a verdict and its own
reliability signal. paneljudge runs the same transcript past several
judge models from different model families (the seats of a named
panel), then writes a report that lines their verdicts up side by side:
each seat's fidelity-audit rulings, its winner, and whether the panel
agrees. Disagreement is the point -- where independent families split
is where a single judge's verdict should not be trusted on its own.

Judging itself is entirely rejudge.py's: each seat is one
rejudge.rejudge_one() call with the seat's model on the panel's host,
so briefs, the two-call fidelity audit, context sizing and the
<stem>.rejudge-<model>.txt output files are exactly what a manual
rejudge would produce. The panel adds, per transcript, a sibling
<stem>.panel.txt, and for --logs-dir runs a panel-summary.txt in that
directory. Originals are never modified, and panel results are
reporting only: they never feed summarize_results.py aggregates.

Panels are named in a JSON file (default panels.json):

  {"prodesk-panel": {"host": "http://192.168.1.236:11434",
                     "seats": [{"model": "gemma4:latest",
                                "family": "google"}, ...]}}

A seat's family is declared, never inferred from its model name. A
seat whose family matches a contestant's is REFUSED for that
transcript unless --allow-same-family is given -- cross-family
judging is the standing rule, and a loud logged refusal is not a
silent fallback. The contestant models checked are the scenario's,
unioned with the models recorded in the transcript itself: a duel
log's turn headers (or a saved JSON transcript's model fields) say
what actually ran, which matters because profile-run scenarios can
declare placeholder models. Families are compared by model-string
prefix (the part before ':' or '/') with generation digits stripped,
so a gemma4 seat also matches a gemma3 contestant; a seat matches
through its declared family or through its own model's prefix. A
seat that skips or fails is recorded in the report as NO VERDICT
with its status; nothing is dropped quietly.

Execution is seat-major (seat 1 judges every transcript, then seat 2,
...): the judge host loads one model at a time, so this pays that load
once per seat. Panel runs are slow -- the settled ProDesk panel
measures about 20 minutes per transcript -- so --resume skips a
(seat, transcript) pair whose rejudge output already exists, is newer
than the transcript, and still parses.

Usage:
  python3 paneljudge.py --panel prodesk-panel SCENARIO.json TRANSCRIPT [...]
  python3 paneljudge.py --panel prodesk-panel --logs-dir output/logs-2026-10-18
  python3 paneljudge.py --panel prodesk-panel --dry-run SCENARIO.json TRANSCRIPT

Exit status is 0 only when every seat produced (or resumed) a verdict
for every transcript; the reports are written in full either way.
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import rejudge

DEFAULT_HOST = "http://localhost:11434"
VERDICT_STATUSES = ("OK", "RESUMED")  # statuses that produced a verdict

WINNER_HEADER_RE = re.compile(r"^Winner:\s*(?P<winner>.+?)\s*$",
                              re.MULTILINE)
FIDELITY_LINE_RE = re.compile(
    r"^Fidelity audit:\s*(?P<name>.+?)\s+(?P<status>HELD|CROSSED)"
    r"(?:\s+--\s+evidence:.*)?$")
FAIL_DETAIL_RE = re.compile(r":\s*(?:FAILED|SKIPPED)\s*\([^)]*\):\s*(.+)$",
                            re.MULTILINE)
SCORE_SKIP_PREFIXES = ("Fidelity audit:", "WINNER:")


def model_family(model):
    """A model string's family prefix: the part before ':' or '/'."""
    return re.split(r"[:/]", model, maxsplit=1)[0].strip().lower()


def _family_keys(text):
    """Comparison keys for a family label or model prefix: the text
    itself plus its generation stem (trailing digits/dots stripped),
    so gemma3 and gemma4 share the key "gemma"."""
    text = text.strip().lower()
    return {text, text.rstrip("0123456789.")}


def seat_conflicts(seat, models):
    """The contestant model strings (from `models`) whose family
    collides with the seat's.

    A model collides when its family keys (from the prefix of its
    model string) intersect the seat's keys, which come from both
    the seat's declared family and its own model's prefix -- so
    family "google" still catches a gemma contestant, through the
    seat's gemma4 model.
    """
    seat_keys = _family_keys(seat["family"]) | \
        _family_keys(model_family(seat["model"]))
    return [model for model in models
            if _family_keys(model_family(model)) & seat_keys]


def transcript_models(path):
    """The models recorded in a transcript itself, as a set.

    A --save-json (.json) transcript carries a "model" field per
    entry; a duel log names the model in each turn header (rejudge's
    TURN_HEADER_RE). Fail-soft: an unreadable transcript, or one
    that yields no models, contributes an empty set -- the scan
    never crashes a run; the judging step reports its own failures.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return set()
    if path.endswith(".json"):
        try:
            entries = json.loads(text)
        except ValueError:
            return set()
        if not isinstance(entries, list):
            return set()
        return {entry["model"] for entry in entries
                if isinstance(entry, dict) and entry.get("model")}
    models = set()
    for line in text.splitlines():
        match = rejudge.TURN_HEADER_RE.match(line)
        if match:
            models.add(match.group("model"))
    return models


def contestant_models(participants, transcript_path):
    """The model strings the family check runs against: the
    scenario's participant models, unioned with the models recorded
    in the transcript (see transcript_models). Scenario models come
    first, in participant order; transcript-only models follow,
    sorted, so refusal messages are deterministic. Duplicates
    collapse: both participants can declare the same model."""
    models = []
    known = set()
    ordered = [p["model"] for p in participants] + \
        sorted(transcript_models(transcript_path))
    for model in ordered:
        if model not in known:
            known.add(model)
            models.append(model)
    return models


class SeatResult:
    """One seat's outcome on one transcript, as the report needs it."""

    def __init__(self, seat):
        self.model = seat["model"]
        self.family = seat["family"]
        self.status = "PENDING"
        self.detail = ""      # failure reason or refusal note
        self.winner = None    # parsed winner, or None
        self.fidelity = None  # {participant: HELD|CROSSED}, or None
        self.scores = {}      # {participant: int}, best-effort

    @property
    def has_verdict(self):
        return self.status in VERDICT_STATUSES

    def fidelity_for(self, name):
        if self.fidelity is None:
            return None
        return self.fidelity.get(name)


def load_panels(path):
    """Read a panels file: {name: {"host": ..., "seats": [...]}}."""
    with open(path, encoding="utf-8") as fh:
        panels = json.load(fh)
    if not isinstance(panels, dict):
        raise ValueError(f"{path}: top level must be an object "
                         "mapping panel names to panels")
    return panels


def get_panel(panels, name, path):
    """The named panel, normalized to {"host", "seats"}; ValueError
    with a usable message when the panel or a seat is malformed."""
    if name not in panels:
        known = ", ".join(sorted(panels)) or "none"
        raise ValueError(f"panel {name!r} not in {path} (it defines: "
                         f"{known})")
    panel = panels[name]
    seats = panel.get("seats") or []
    if not seats:
        raise ValueError(f"panel {name!r} in {path} has no seats")
    for seat in seats:
        if not seat.get("model") or not seat.get("family"):
            raise ValueError(f"panel {name!r} in {path}: every seat "
                             "needs a \"model\" and a \"family\"")
    return {"host": panel.get("host") or DEFAULT_HOST, "seats": seats}


def warn_panel_shape(seats):
    """Warn (stderr) about panel definitions that break the rules:
    two seats sharing a family, or the same model seated twice (their
    rejudge output files would collide). Warnings only -- the run
    itself decides nothing silently either way."""
    families = Counter(s["family"].strip().lower() for s in seats)
    for family, count in families.items():
        if count > 1:
            print(f"paneljudge: WARNING: {count} seats declare family "
                  f"{family!r}; panel seats should be distinct "
                  "families", file=sys.stderr)
    models = Counter(s["model"] for s in seats)
    for model, count in models.items():
        if count > 1:
            print(f"paneljudge: WARNING: model {model!r} is seated "
                  f"{count} times; their rejudge outputs would "
                  "collide", file=sys.stderr)


def _score_values(line, pattern):
    """The score values one line assigns to a participant (0, 1 or
    more regex hits, validity decided by the caller)."""
    values = []
    for match in pattern.finditer(line):
        if match.group("sep") or match.group("frac"):
            values.append(int(match.group("num")))
    return values


def _score_pattern(name):
    return re.compile(r"(?<!\w)" + re.escape(name) + r"(?![\w'])\s*"
                      r"(?P<sep>:|—|–|-)?\s*"
                      r"(?P<num>10|[0-9])(?!\d)"
                      r"(?P<frac>\s*/\s*10)?")


def extract_scores(verdict, names):
    """Best-effort per-participant scores from a verdict body.

    Recognizes "Name: 9", "Name — 9/10" and "Name 9/10". A bare
    "Name 9" (no colon/dash and no /10) is too easily a turn number
    or a count, so it does not count. When the mentions disagree, the
    score is ambiguous and dropped: the spread is a statistic for
    the report, never an input to classification.
    """
    patterns = {name: _score_pattern(name) for name in names}
    found = {name: set() for name in names}
    for line in verdict.splitlines():
        stripped = line.strip()
        if stripped.startswith(SCORE_SKIP_PREFIXES):
            continue
        for name in names:
            for value in _score_values(stripped, patterns[name]):
                found[name].add(value)
    return {name: values.pop() for name, values in found.items()
            if len(values) == 1}


def parse_rejudge_output(path, participants):
    """Parse a .rejudge-<model>.txt file into
    (winner, fidelity, scores).

    winner is None when the header says "(none parsed)" (or the file
    cannot be read -- the panel trusts the header, never re-derives a
    winner from scores). fidelity is {name: "HELD"|"CROSSED"} from the
    verdict's Fidelity audit lines, or None when there are none.
    scores is extract_scores() over the verdict body.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None, None, {}
    head, _sep, verdict = text.partition("=== Verdict ===")
    winner = None
    match = WINNER_HEADER_RE.search(head)
    if match and match.group("winner") != "(none parsed)":
        winner = match.group("winner")
    fidelity = {}
    for line in verdict.splitlines():
        match = FIDELITY_LINE_RE.match(line.strip())
        if match:
            fidelity[match.group("name").strip()] = match.group("status")
    names = [p["name"] for p in participants]
    return winner, fidelity or None, extract_scores(verdict, names)


def _seat_incomplete(result, names):
    """A seat leaves the panel incomplete when it produced no verdict,
    its verdict yielded no winner, or its audit is missing a ruling
    for any participant."""
    if not result.has_verdict or result.winner is None:
        return True
    return any(result.fidelity_for(name) is None for name in names)


def classify(results, participants):
    """(classification, detail, flags) for one transcript.

    Classification is over the seats that produced a verdict with a
    parsed winner: UNANIMOUS when they all name the same winner,
    MAJORITY when one winner holds more than half of them,
    SPLIT-WINNER otherwise. Flags are independent: FIDELITY-SPLIT
    (seats disagree HELD vs CROSSED on a participant),
    CROSSED-DETECTED (any CROSSED ruling), INCOMPLETE (any seat
    NO VERDICT / REFUSED, missing winner, or audit gap).
    """
    names = [p["name"] for p in participants]
    winners = [r.winner for r in results
               if r.has_verdict and r.winner is not None]
    counts = Counter(winners)
    if not winners:
        classification = "SPLIT-WINNER"
    elif len(counts) == 1:
        classification = "UNANIMOUS"
    elif counts.most_common(1)[0][1] > len(winners) / 2:
        classification = "MAJORITY"
    else:
        classification = "SPLIT-WINNER"
    detail_bits = []
    if counts:
        detail_bits.append("-".join(str(c) for _w, c in
                                    counts.most_common()))
    missing = len(results) - len(winners)
    if missing:
        detail_bits.append(f"{missing} seat{'s' if missing != 1 else ''}"
                           " missing")
    if not winners:
        detail_bits.append("no winners parsed")
    flags = []
    for name in names:
        rulings = {r.fidelity_for(name) for r in results}
        rulings.discard(None)
        if len(rulings) > 1:
            flags.append("FIDELITY-SPLIT")
            break
    if any(r.fidelity and "CROSSED" in r.fidelity.values()
           for r in results):
        flags.append("CROSSED-DETECTED")
    if any(_seat_incomplete(r, names) for r in results):
        flags.append("INCOMPLETE")
    return classification, ", ".join(detail_bits), flags


def _no_verdict_cell(result):
    if result.status == "REFUSED":
        return f"REFUSED (same family as contestant {result.detail})"
    if result.status == "NO VERDICT":
        return f"NO VERDICT ({result.detail})" if result.detail \
            else "NO VERDICT"
    text = f"NO VERDICT ({result.status}"
    if result.detail:
        text += f": {result.detail}"
    return text + ")"


def _score_spread_line(results, names):
    per_name = {}
    reporters = 0
    for result in results:
        if result.scores:
            reporters += 1
        for name in names:
            if name in result.scores:
                per_name.setdefault(name, []).append(result.scores[name])
    if not per_name:
        return "Score spread: none reported"
    parts = []
    for name in names:
        if name in per_name:
            low, high = min(per_name[name]), max(per_name[name])
            parts.append(f"{name} {low}" if low == high
                         else f"{name} {low}-{high}")
    return (f"Score spread: {', '.join(parts)} "
            f"(seats reporting scores: {reporters})")


def format_report(transcript_path, scenario_path, panel_name, host,
                  participants, results, same_family_marks, when=None):
    """The per-transcript .panel.txt content (spec section 3.7)."""
    names = [p["name"] for p in participants]
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Panel report: {transcript_path}",
             f"Scenario: {scenario_path}",
             f"Panel: {panel_name} @ {host}",
             f"Date: {stamp}"]
    for mark in same_family_marks:
        lines.append(f"SAME-FAMILY SEAT: {mark}")
    lines.append("")
    header = ["Seat"] + [f"Fidelity: {name}" for name in names] + ["Winner"]
    rows = []
    for result in results:
        if result.has_verdict:
            row = [result.model]
            row += [result.fidelity_for(name) or "NO AUDIT"
                    for name in names]
            row.append(result.winner or "(none parsed)")
        else:
            row = [result.model, _no_verdict_cell(result)]
        rows.append(row)
    widths = [len(cell) for cell in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells):
        return "  ".join(cell.ljust(widths[i])
                         for i, cell in enumerate(cells)).rstrip()

    lines.append(render(header))
    for row in rows:
        lines.append(render(row))
    classification, detail, flags = classify(results, participants)
    lines.append("")
    suffix = f" ({detail})" if detail else ""
    lines.append(f"Classification: {classification}{suffix}")
    lines.append(f"Flags: {', '.join(flags) if flags else 'none'}")
    lines.append(_score_spread_line(results, names))
    return "\n".join(lines) + "\n", classification, detail, flags


def format_summary(panel_name, host, entries, when=None):
    """The panel-summary.txt content for a --logs-dir run.

    entries: [(stem, classification, detail, flags, winners)].
    """
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Panel summary: {panel_name} @ {host}",
             f"Date: {stamp}",
             f"Transcripts: {len(entries)}",
             ""]
    for stem, classification, detail, flags, winners in entries:
        tally = ", ".join(f"{winner} {count}" for winner, count
                          in Counter(winners).most_common()) or "none"
        suffix = f" ({detail})" if detail else ""
        flag_text = ", ".join(flags) if flags else "none"
        lines.append(f"{stem}: {classification}{suffix} | "
                     f"flags: {flag_text} | winners: {tally}")
    counts = Counter(entry[1] for entry in entries)
    total = len(entries)
    percent = round(100 * counts.get("UNANIMOUS", 0) / total) if total else 0
    lines.append("")
    lines.append(f"Totals: UNANIMOUS {counts.get('UNANIMOUS', 0)}, "
                 f"MAJORITY {counts.get('MAJORITY', 0)}, "
                 f"SPLIT-WINNER {counts.get('SPLIT-WINNER', 0)} "
                 f"(of {total} transcripts; {percent}% UNANIMOUS)")
    for flag in ("FIDELITY-SPLIT", "CROSSED-DETECTED", "INCOMPLETE"):
        flagged = [entry[0] for entry in entries if flag in entry[3]]
        if flagged:
            lines.append(f"Flagged {flag}: {', '.join(flagged)}")
    return "\n".join(lines) + "\n"


def panel_report_path(transcript_path):
    stem = os.path.splitext(transcript_path)[0]
    return f"{stem}.panel.txt"


def _failure_detail(captured, status):
    """A one-line reason for a NO VERDICT seat, mined from what
    rejudge printed about the failure (its status string is all it
    returns)."""
    if status == "NO JUDGE":
        return "scenario has no judge configured"
    match = FAIL_DETAIL_RE.search(captured)
    if not match:
        return ""
    detail = " ".join(match.group(1).split())
    return detail[:200]


def _call_rejudge(scenario_path, transcript_path, model, host, timeout,
                  dry_run=False):
    """One rejudge_one call with its console output preserved: the
    output is captured (so a failure's reason can be mined for the
    report) and re-emitted unchanged. Returns (status, captured)."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        status = rejudge.rejudge_one(scenario_path, transcript_path,
                                     model, judge_host=host,
                                     timeout=timeout, dry_run=dry_run)
    sys.stdout.write(captured.getvalue())
    return status, captured.getvalue()


def run_pair(scenario_path, transcript_path, participants, seat, host,
             timeout, resume):
    """Run one seat on one transcript (or resume it). Returns a
    SeatResult with status/detail filled; winner/fidelity/scores are
    parsed from the output file afterwards by the caller."""
    result = SeatResult(seat)
    out_path = rejudge.rejudge_output_path(transcript_path, seat["model"])
    if resume and os.path.exists(out_path) and \
            os.path.getmtime(out_path) > os.path.getmtime(transcript_path):
        winner, _fidelity, _scores = parse_rejudge_output(
            out_path, participants)
        if winner is not None:
            print(f"{transcript_path}: {seat['model']}: resumed from "
                  f"{out_path} (newer than the transcript)")
            result.status = "RESUMED"
            return result
        print(f"{transcript_path}: {seat['model']}: existing output "
              f"{out_path} has no parsed winner -- NO VERDICT")
        result.status = "NO VERDICT"
        result.detail = "existing output did not parse"
        return result
    status, captured = _call_rejudge(scenario_path, transcript_path,
                                     seat["model"], host, timeout)
    result.status = status
    if status not in VERDICT_STATUSES:
        result.detail = _failure_detail(captured, status)
    return result


def refused_result(seat, conflict, transcript_path):
    """The recorded outcome for a same-family seat that may not run."""
    result = SeatResult(seat)
    result.status = "REFUSED"
    result.detail = conflict
    print(f"{transcript_path}: seat {seat['model']} REFUSED -- same "
          f"family as contestant {conflict} (pass --allow-same-family "
          "to run it)")
    return result


def build_jobs(parser, args):
    """[(scenario_path, transcript_path)], mirroring rejudge's two
    input shapes."""
    if args.logs_dir:
        jobs = []
        for name in sorted(os.listdir(args.logs_dir)):
            if not name.endswith(".log"):
                continue
            log_path = os.path.join(args.logs_dir, name)
            scenario = rejudge.scenario_for_log(log_path,
                                                args.scenarios_dir)
            if scenario is None:
                print(f"{log_path}: no matching scenario in "
                      f"{args.scenarios_dir} -- skipped.")
                continue
            jobs.append((scenario, log_path))
        return jobs
    if not args.scenario or not args.transcripts:
        parser.error("give a scenario and at least one transcript, "
                     "or use --logs-dir")
    return [(args.scenario, path) for path in args.transcripts]


def _dry_run(args, seats, jobs, host, participants_for):
    """Each seat's dry-run sizing, plus the family check. Exit 0 only
    when every (seat, transcript) pair would judge."""
    print(f"Dry run: panel {args.panel} @ {host} "
          f"({len(seats)} seats x {len(jobs)} transcripts)")
    ok = True
    for seat in seats:
        print(f"Seat: {seat['model']} (family {seat['family']})")
        for scenario, transcript in jobs:
            models = contestant_models(participants_for(scenario),
                                       transcript)
            conflicts = seat_conflicts(seat, models)
            if conflicts and not args.allow_same_family:
                print(f"{transcript}: {seat['model']}: would REFUSE "
                      f"(same family as contestant {conflicts[0]})")
                ok = False
                continue
            status, _captured = _call_rejudge(
                scenario, transcript, seat["model"], host,
                args.timeout, dry_run=True)
            if status != "OK":
                ok = False
    return 0 if ok else 1


def _run_panel(args, seats, jobs, host, participants_for):
    """Seat-major judging, then per-transcript reports. Returns
    (exit_status, summary_entries)."""
    results = [{} for _job in jobs]           # per job: model -> result
    marks = [[] for _job in jobs]             # per job: same-family marks
    warned = set()
    complete = True
    for seat in seats:
        for index, (scenario, transcript) in enumerate(jobs):
            participants = participants_for(scenario)
            models = contestant_models(participants, transcript)
            conflicts = seat_conflicts(seat, models)
            if conflicts and not args.allow_same_family:
                results[index][seat["model"]] = refused_result(
                    seat, conflicts[0], transcript)
                complete = False
                continue
            if conflicts:
                for conflict in conflicts:
                    mark = f"{seat['model']} vs contestant {conflict}"
                    if mark not in marks[index]:
                        marks[index].append(mark)
                    key = (seat["model"], scenario, conflict)
                    if key not in warned:
                        warned.add(key)
                        print(f"WARNING: seat {seat['model']} (family "
                              f"{seat['family']}) matches contestant "
                              f"{conflict} in {scenario} -- running "
                              "anyway (--allow-same-family)")
            result = run_pair(scenario, transcript, participants, seat,
                              host, args.timeout, args.resume)
            results[index][seat["model"]] = result
            if not result.has_verdict:
                complete = False
    entries = []
    for index, (scenario, transcript) in enumerate(jobs):
        participants = participants_for(scenario)
        seat_results = [results[index][seat["model"]] for seat in seats]
        for result in seat_results:
            if result.has_verdict:
                out_path = rejudge.rejudge_output_path(
                    transcript, result.model)
                parsed = parse_rejudge_output(out_path, participants)
                result.winner, result.fidelity, result.scores = parsed
        report, classification, detail, flags = format_report(
            transcript, scenario, args.panel, host, participants,
            seat_results, marks[index])
        out_path = panel_report_path(transcript)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"{transcript}: panel report -> {out_path} "
              f"({classification})")
        winners = [r.winner for r in seat_results
                   if r.has_verdict and r.winner is not None]
        stem = os.path.splitext(os.path.basename(transcript))[0]
        entries.append((stem, classification, detail, flags, winners))
    return (0 if complete else 1), entries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Re-score finished duels with a panel of judge "
                    "models from different families, and report "
                    "where the panel agrees and where it splits.")
    parser.add_argument("--panel", required=True,
                        help="panel name in the panels file")
    parser.add_argument("--panels-file", default="panels.json",
                        help="JSON file defining the panels "
                             "(default: panels.json)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-judge-call timeout in seconds "
                             "(default: 1800)")
    parser.add_argument("--dry-run", action="store_true",
                        help="size every seat's judge request and "
                             "print the panel plan, but call nothing")
    parser.add_argument("--resume", action="store_true",
                        help="skip (seat, transcript) pairs whose "
                             "rejudge output exists, is newer than "
                             "the transcript, and still parses")
    parser.add_argument("--allow-same-family", action="store_true",
                        help="run seats whose family matches a "
                             "contestant's (warned, and marked in "
                             "the report) instead of refusing them")
    parser.add_argument("--logs-dir", default=None,
                        help="panel-judge every *.log in this "
                             "directory, matching scenarios by filename")
    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="scenario directory for --logs-dir "
                             "matching (default: scenarios)")
    parser.add_argument("scenario", nargs="?",
                        help="scenario JSON for the transcript(s)")
    parser.add_argument("transcripts", nargs="*",
                        help="saved transcripts (.json or .log)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        panel = get_panel(load_panels(args.panels_file), args.panel,
                          args.panels_file)
    except (OSError, ValueError) as exc:
        print(f"paneljudge: {exc}", file=sys.stderr)
        return 2
    host, seats = panel["host"], panel["seats"]
    warn_panel_shape(seats)
    jobs = build_jobs(parser, args)
    scenario_cache = {}

    def participants_for(scenario):
        if scenario not in scenario_cache:
            _topic, participants, _judge = rejudge.load_scenario(scenario)
            scenario_cache[scenario] = participants
        return scenario_cache[scenario]

    if args.dry_run:
        return _dry_run(args, seats, jobs, host, participants_for)
    print(f"Panel {args.panel} @ {host}: {len(seats)} seat(s) x "
          f"{len(jobs)} transcript(s)")
    status, entries = _run_panel(args, seats, jobs, host,
                                  participants_for)
    if args.logs_dir:
        summary_path = os.path.join(args.logs_dir, "panel-summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(format_summary(args.panel, host, entries))
        print(f"Panel summary -> {summary_path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
