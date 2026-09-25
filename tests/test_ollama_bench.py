"""Tests for ollama_bench.py -- tokens/second benchmarking.

urlopen is mocked, so no Ollama server is needed.
"""

import io
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_bench
from ollama_bench import (
    BenchError,
    bench_model,
    call_generate,
    compute_metrics,
    format_comparison,
    summarize,
)

TIMING = {
    "prompt_eval_count": 40,
    "prompt_eval_duration": 200_000_000,   # 0.2 s -> 200 tok/s
    "eval_count": 100,
    "eval_duration": 5_000_000_000,        # 5 s   -> 20 tok/s
    "total_duration": 5_400_000_000,       # 5.4 s
    "load_duration": 800_000_000,          # 0.8 s
    "response": "ok",
}


class FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _patch_urlopen(monkeypatch, payloads):
    """Serve canned responses in order for each urlopen call."""
    bodies = [FakeResp(p) for p in payloads]

    def fake(req, timeout=None):
        return bodies.pop(0)

    monkeypatch.setattr(ollama_bench.urllib.request, "urlopen", fake)


def test_compute_metrics():
    m = compute_metrics(TIMING)
    assert m["prompt_tps"] == 200.0
    assert m["gen_tps"] == 20.0
    assert m["total_s"] == 5.4
    assert m["load_s"] == 0.8
    assert m["prompt_tokens"] == 40
    assert m["gen_tokens"] == 100


def test_compute_metrics_zero_duration_is_safe():
    m = compute_metrics({"eval_count": 10, "eval_duration": 0})
    assert m["gen_tps"] == 0.0


def test_summarize_averages_and_stdev():
    runs = [
        {"prompt_tps": 200.0, "gen_tps": 20.0, "total_s": 5.0,
         "prompt_tokens": 40, "gen_tokens": 100},
        {"prompt_tps": 220.0, "gen_tps": 22.0, "total_s": 5.5,
         "prompt_tokens": 40, "gen_tokens": 100},
    ]
    s = summarize(runs)
    assert s["prompt_tps"] == 210.0
    assert s["gen_tps"] == 21.0
    assert s["total_s"] == 5.25
    assert s["gen_tps_stdev"] > 0


def test_summarize_single_run_has_zero_stdev():
    s = summarize([{"prompt_tps": 1.0, "gen_tps": 2.0, "total_s": 3.0,
                    "prompt_tokens": 1, "gen_tokens": 2}])
    assert s["gen_tps_stdev"] == 0.0


def test_call_generate_returns_parsed_json(monkeypatch):
    _patch_urlopen(monkeypatch, [TIMING])
    data = call_generate("http://x", "m", "hi", {}, 30)
    assert data["eval_count"] == 100


def test_call_generate_http_error_becomes_bench_error(monkeypatch):
    def fake(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 404, "not found", {}, io.BytesIO(b"no such model"))

    monkeypatch.setattr(ollama_bench.urllib.request, "urlopen", fake)
    try:
        call_generate("http://x", "nope", "hi", {}, 30)
    except BenchError as e:
        assert "404" in str(e)
        return
    raise AssertionError("expected BenchError")


def test_call_generate_unreachable_host_becomes_bench_error(monkeypatch):
    def fake(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(ollama_bench.urllib.request, "urlopen", fake)
    try:
        call_generate("http://x", "m", "hi", {}, 30)
    except BenchError as e:
        assert "Cannot reach Ollama" in str(e)
        return
    raise AssertionError("expected BenchError")


def test_bench_model_does_warmup_then_iterations(monkeypatch):
    calls = []

    def fake(host, model, prompt, options, timeout):
        calls.append(model)
        return TIMING

    monkeypatch.setattr(ollama_bench, "call_generate", fake)
    warm, summary = bench_model("http://x", "m", "hi", {}, 30,
                                iterations=3, warmup=True)
    assert len(calls) == 4  # 1 warmup + 3 timed
    assert warm["gen_tps"] == 20.0
    assert summary["gen_tps"] == 20.0


def test_bench_model_no_warmup(monkeypatch):
    calls = []

    def fake(host, model, prompt, options, timeout):
        calls.append(model)
        return TIMING

    monkeypatch.setattr(ollama_bench, "call_generate", fake)
    warm, _ = bench_model("http://x", "m", "hi", {}, 30,
                          iterations=2, warmup=False)
    assert warm is None
    assert len(calls) == 2


def test_format_comparison_sorts_fastest_first():
    results = [
        ("slow", {"load_s": 1.0},
         {"prompt_tps": 100.0, "prompt_tps_stdev": 0.0,
          "gen_tps": 10.0, "gen_tps_stdev": 0.0,
          "total_s": 20.0, "total_s_stdev": 0.0}),
        ("fast", {"load_s": 0.5},
         {"prompt_tps": 400.0, "prompt_tps_stdev": 0.0,
          "gen_tps": 40.0, "gen_tps_stdev": 0.0,
          "total_s": 5.0, "total_s_stdev": 0.0}),
    ]
    table = format_comparison(results)
    assert table.index("fast") < table.index("slow")
    assert "Model" in table and "Gen tok/s" in table
