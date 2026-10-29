#!/usr/bin/env python3
"""judge_context_audit.py -- read-only audit of dual's judge context sizing.

For each saved duel transcript (the --save-json format: a list of
{"speaker", "model", "text"} dicts), this rebuilds the exact judge
messages with dual's own build_judge_messages() and asks dual's own
judge_context_window() which window the new auto-sizer would choose
for a default judge (no explicit num_ctx, max_tokens default 500):

  8192   fits the small window
  16384  needs the large window (the OLD code would have run this at a
         fixed 8192 and silently truncated the transcript)
  32768+ the auto-sizer's ladder grows (32768, then 65536) for long
         transcripts instead of skipping
  SKIP   exceeds even 65536: the judge is skipped loudly

It also flags, per transcript, whether the estimated judge input
alone exceeded the old fixed window's usable input budget
(8192 - 500 reply tokens = 7692), i.e. the old judge was probably
grading a cut-off transcript.

Modes:
  python3 judge_context_audit.py T1.json [T2.json ...]
  python3 judge_context_audit.py --results run_results.log
      No transcript text survives in run_results.log, so this mode
      ESTIMATES: chars ~= total generated tokens x chars-per-token
      (run at 4.0 and again at 3.2, a code-heavy density, and the
      range is reported). Blocks with "Code runs:" are floors --
      run reports add judge input that run_results.log doesn't keep.
  python3 judge_context_audit.py --demo
      Synthetic sizes, to see the tool work anywhere.

Needs dual's ollama_judge.py importable: run it from a dual checkout,
or it looks in ~/dual and ~/workspace/dual. Nothing is written, no
network is touched, no duel is run.
"""

import argparse
import json
import os
import re
import sys

OLD_NUM_CTX = 8192
DEFAULT_MAX_TOKENS = 500


def load_dual():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.getcwd(), here, os.path.expanduser("~/dual"),
                 os.path.expanduser("~/workspace/dual")):
        if os.path.isfile(os.path.join(cand, "ollama_judge.py")):
            sys.path.insert(0, cand)
            import ollama_judge
            return ollama_judge, cand
    sys.exit("Cannot find dual's ollama_judge.py -- run from a dual checkout.")


def audit_messages(judge_mod, judge_cfg, messages):
    """Return (verdict, required, est_input_tokens)."""
    chars = sum(len(m["content"]) for m in messages)
    est_input = (chars + 3) // 4 + 16 * len(messages)
    try:
        window, required = judge_mod.judge_context_window(judge_cfg, messages)
        verdict = str(window)
    except judge_mod.JudgeContextExceeded:
        required = est_input + (est_input + 4) // 5 + (
            judge_cfg.get("max_tokens") or DEFAULT_MAX_TOKENS)
        verdict = "SKIP"
    return verdict, required, est_input


def audit_transcript(judge_mod, path):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    speakers = []
    for row in rows:
        if row["speaker"] not in speakers:
            speakers.append(row["speaker"])
    if len(speakers) != 2:
        return {"file": os.path.basename(path), "error":
                f"expected 2 speakers, found {len(speakers)}"}
    models = {}
    for row in rows:
        models.setdefault(row["speaker"], row.get("model", "?"))
    participants = [{"name": name, "model": models[name]} for name in speakers]
    index = {name: i for i, name in enumerate(speakers)}
    transcript = [(index[row["speaker"]], row["text"]) for row in rows]
    messages = judge_mod.build_judge_messages(
        {}, "(audit placeholder topic)", participants, transcript)
    verdict, required, est_input = audit_messages(judge_mod, {}, messages)
    return {"file": os.path.basename(path), "turns": len(transcript),
            "chars": sum(len(t) for _, t in transcript),
            "verdict": verdict, "required": required,
            "old_overflow_tokens": max(0, est_input - (OLD_NUM_CTX - DEFAULT_MAX_TOKENS))}


BLOCK_SEP = re.compile(r"^-{10,}\s*$", re.M)
PERF_ROW = re.compile(r"^(\S+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s*$", re.M)


def audit_results(judge_mod, path, densities):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out = []
    for block in BLOCK_SEP.split(text):
        rows = PERF_ROW.findall(block)
        if not rows:
            continue
        cfg = re.search(r"^Config: (.+)$", block, re.M)
        code = "Code runs:" in block
        verdicts = {}
        for k in densities:
            transcript = []
            for i, (_model, turns, tokens, _trunc) in enumerate(rows[:2]):
                per = int(int(tokens) * k) // max(1, int(turns))
                transcript += [(i, "x" * per)] * int(turns)
            participants = [{"name": m, "model": m} for m, *_ in rows[:2]]
            if len(participants) < 2:
                participants.append({"name": "?", "model": "?"})
            messages = judge_mod.build_judge_messages(
                {}, "(from run_results.log)", participants, transcript)
            verdict, _req, _est = audit_messages(judge_mod, {}, messages)
            verdicts[k] = verdict
        out.append({"file": os.path.basename(cfg.group(1)) if cfg else "?",
                    "verdicts": verdicts, "code": code})
    return out


def main():
    ap = argparse.ArgumentParser(description="Audit dual judge context sizing.")
    ap.add_argument("transcripts", nargs="*", help="saved transcript JSON files")
    ap.add_argument("--results", metavar="LOG", help="audit a run_results.log (estimated)")
    ap.add_argument("--demo", action="store_true", help="synthetic transcripts")
    args = ap.parse_args()
    judge_mod, where = load_dual()
    print(f"(using dual code from {where})\n")

    if args.demo:
        demo = []
        base = ("The debaters exchanged careful arguments about the topic, "
                "citing examples, conceding small points, and pressing "
                "their strongest claims with evident preparation. ")
        for label, reps in (("short duel", 6), ("long duel", 60),
                            ("marathon duel", 220)):
            text = base * reps
            demo.append({"speaker": "A", "model": "m", "text": text})
            demo.append({"speaker": "B", "model": "m", "text": text})
            path = f"/tmp/audit-demo-{label.replace(' ', '-')}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(demo, fh)
            demo = []
            args.transcripts.append(path)

    if args.results:
        rows = audit_results(judge_mod, args.results, (4.0, 3.2))
        counts = {}
        for r in rows:
            key = (r["verdicts"][4.0], r["verdicts"][3.2])
            counts[key] = counts.get(key, 0) + 1
        print(f"{len(rows)} duels estimated from {args.results}:")
        for (v40, v32), n in sorted(counts.items()):
            spread = v40 if v40 == v32 else f"{v40} at 4.0 chars/tok, {v32} at 3.2"
            print(f"  {n:4d}  window {spread}")
        coded = sum(1 for r in rows if r["code"])
        if coded:
            print(f"  ({coded} blocks ran code: their estimates are floors, "
                  "run reports add judge input.)")
        return

    if not args.transcripts:
        ap.error("give transcript JSON files, --results LOG, or --demo")
    counts = {"8192": 0, "16384": 0, "SKIP": 0}
    overflowed = 0
    print(f"{'transcript':44s} {'turns':>5} {'chars':>8} {'budget':>7}  window")
    for path in args.transcripts:
        r = audit_transcript(judge_mod, path)
        if "error" in r:
            print(f"{r['file']:44s}  ERROR: {r['error']}")
            continue
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        note = ""
        if r["old_overflow_tokens"]:
            overflowed += 1
            note = f"  (old 8192 judge input over by ~{r['old_overflow_tokens']} tok)"
        print(f"{r['file']:44s} {r['turns']:5d} {r['chars']:8d} "
              f"{r['required']:7d}  {r['verdict']}{note}")
    print(f"\n{sum(counts.values())} transcripts: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"{overflowed} would have overflowed the old fixed-8192 judge "
          "(input alone, before the reply allowance).")


if __name__ == "__main__":
    main()
