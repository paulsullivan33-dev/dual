#!/usr/bin/env python3
"""
ollama_bench.py -- benchmark Ollama models and report tokens/second.

Usage:
    python3 ollama_bench.py qwen3:8b
    python3 ollama_bench.py qwen3:8b qwen2.5-coder:14b --iterations 5
    python3 ollama_bench.py tinyllama --host http://192.168.1.10:11434

For each model: one warmup run (loads the model; not counted), then N timed
runs of a fixed prompt. Reports prompt-processing speed, generation speed,
and total time per run, averaged across runs. With several models it prints
a comparison table sorted by generation speed.

Ollama reports exact token counts and nanosecond timings in every API
response, so tokens/second is measured, not estimated. Thinking is forced
off so results are comparable between thinking and non-thinking models.
"""

import argparse
import json
import statistics
import sys
import urllib.request
import urllib.error

from ollama_common import (
    DEFAULT_HOST,
    DEFAULT_TIMEOUT,
    compute_metrics,
    setup_utf8_stdout,
)

DEFAULT_PROMPT = (
    "Write a short paragraph about the history of the transistor, "
    "then list three key dates."
)


class BenchError(Exception):
    """A benchmark run failed (bad response, unreachable host)."""


def call_generate(host, model, prompt, options, timeout):
    """Single non-streaming /api/generate call with thinking disabled.

    Returns the raw response dict, which includes token counts and
    nanosecond timings. Raises BenchError on failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": options,
    }
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise BenchError(
            f"HTTP {e.code} from Ollama: {body}\n"
            f"Hint: did you run `ollama pull {model}`?"
        ) from e
    except urllib.error.URLError as e:
        raise BenchError(
            f"Cannot reach Ollama at {host}: {e.reason}\nIs `ollama serve` running?"
        ) from e


def summarize(runs):
    """Average a list of per-run metric dicts; includes stdev when n > 1."""
    summary = {}
    for key in ("prompt_tps", "gen_tps", "total_s"):
        vals = [r[key] for r in runs]
        summary[key] = sum(vals) / len(vals) if vals else 0.0
        summary[key + "_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    summary["prompt_tokens"] = runs[0]["prompt_tokens"]
    summary["gen_tokens"] = runs[0]["gen_tokens"]
    return summary


def bench_model(host, model, prompt, options, timeout, iterations, warmup=True):
    """Benchmark one model. Returns (warmup_metrics_or_None, summary).

    Raises BenchError if any run fails.
    """
    warm = None
    if warmup:
        warm = compute_metrics(call_generate(host, model, prompt, options, timeout))
    runs = [
        compute_metrics(call_generate(host, model, prompt, options, timeout))
        for _ in range(iterations)
    ]
    return warm, summarize(runs)


def _col(values, header, fmt, align_right=True):
    """Format one table column: header + values, padded to the widest.
    Returns (header_cell, [data_cells])."""
    cells = [header] + [fmt(v) for v in values]
    width = max(len(c) for c in cells)
    pad = str.rjust if align_right else str.ljust
    padded = [pad(c, width) for c in cells]
    return padded[0], padded[1:]


def format_comparison(results):
    """Build a comparison table string from [(model, warm, summary), ...],
    sorted by generation speed, fastest first."""
    ordered = sorted(results, key=lambda r: r[2]["gen_tps"], reverse=True)
    models = [m for m, _, _ in ordered]
    sums = [s for _, _, s in ordered]

    cols = [
        _col(models, "Model", lambda v: v, align_right=False),
        _col(sums, "Prompt tok/s",
             lambda s: f"{s['prompt_tps']:.1f} +/- {s['prompt_tps_stdev']:.1f}"),
        _col(sums, "Gen tok/s",
             lambda s: f"{s['gen_tps']:.1f} +/- {s['gen_tps_stdev']:.1f}"),
        _col(sums, "Total s/run",
             lambda s: f"{s['total_s']:.1f} +/- {s['total_s_stdev']:.1f}"),
        _col([w["load_s"] if w else 0.0 for _, w, _ in ordered],
             "Load s", lambda v: f"{v:.1f}"),
    ]
    lines = ["  ".join(header for header, _ in cols)]
    for i in range(len(ordered)):
        lines.append("  ".join(cells[i] for _, cells in cols))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark Ollama models and report tokens/second.")
    ap.add_argument("models", nargs="+", help="one or more Ollama model names")
    ap.add_argument("--host", default=None, help="Ollama server base URL")
    ap.add_argument("--iterations", "-n", type=int, default=3,
                    help="timed runs per model (default: 3)")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false",
                    help="skip the uncounted warmup run")
    ap.add_argument("--prompt", default=None,
                    help="override the benchmark prompt")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="generation budget per run (default: 256)")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="context window size passed to Ollama")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-request timeout, in seconds")
    args = ap.parse_args()

    setup_utf8_stdout()

    host = args.host or DEFAULT_HOST
    timeout = args.timeout or DEFAULT_TIMEOUT
    if args.iterations < 1:
        sys.exit(f"--iterations must be >= 1, got {args.iterations}.")
    if args.max_tokens < 1:
        sys.exit(f"--max-tokens must be >= 1, got {args.max_tokens}.")
    options = {"num_predict": args.max_tokens}
    if args.num_ctx is not None:
        if args.num_ctx < 1:
            sys.exit(f"--num-ctx must be >= 1, got {args.num_ctx}.")
        options["num_ctx"] = args.num_ctx
    prompt = args.prompt or DEFAULT_PROMPT

    results = []
    failed = []
    for model in args.models:
        print(f"Benchmarking {model} ...", file=sys.stderr, flush=True)
        try:
            warm, summary = bench_model(host, model, prompt, options,
                                        timeout, args.iterations,
                                        warmup=args.warmup)
        except BenchError as e:
            print(f"{model}: FAILED\n{e}", file=sys.stderr)
            failed.append(model)
            continue
        results.append((model, warm, summary))
        load_note = f", load {warm['load_s']:.1f}s" if warm else ""
        print(f"{model}: prompt {summary['prompt_tps']:.1f} tok/s, "
              f"gen {summary['gen_tps']:.1f} tok/s "
              f"({summary['gen_tokens']} tokens in {summary['total_s']:.1f}s{load_note})")

    if results:
        print()
        print(format_comparison(results))
    if failed:
        sys.exit(f"\n{len(failed)} model(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
