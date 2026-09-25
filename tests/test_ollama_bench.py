"""Tests for ollama_bench.py -- tokens/second benchmarking.

urlopen is mocked, so no Ollama server is needed.
"""

import io
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama_bench
from ollama_common import compute_metrics
from ollama_bench import (
    BenchError,
    bench_model,
    call_generate,
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


def _summary(gen_tps):
    return {"prompt_tps": 100.0, "prompt_tps_stdev": 0.0,
            "gen_tps": gen_tps, "gen_tps_stdev": 0.0,
            "total_s": 5.0, "total_s_stdev": 0.0}


class ComputeMetricsTests(unittest.TestCase):
    def test_compute_metrics(self):
        m = compute_metrics(TIMING)
        self.assertEqual(m["prompt_tps"], 200.0)
        self.assertEqual(m["gen_tps"], 20.0)
        self.assertEqual(m["total_s"], 5.4)
        self.assertEqual(m["load_s"], 0.8)
        self.assertEqual(m["prompt_tokens"], 40)
        self.assertEqual(m["gen_tokens"], 100)

    def test_zero_duration_is_safe(self):
        m = compute_metrics({"eval_count": 10, "eval_duration": 0})
        self.assertEqual(m["gen_tps"], 0.0)


class SummarizeTests(unittest.TestCase):
    def test_averages_and_stdev(self):
        runs = [
            {"prompt_tps": 200.0, "gen_tps": 20.0, "total_s": 5.0,
             "prompt_tokens": 40, "gen_tokens": 100},
            {"prompt_tps": 220.0, "gen_tps": 22.0, "total_s": 5.5,
             "prompt_tokens": 40, "gen_tokens": 100},
        ]
        s = summarize(runs)
        self.assertEqual(s["prompt_tps"], 210.0)
        self.assertEqual(s["gen_tps"], 21.0)
        self.assertEqual(s["total_s"], 5.25)
        self.assertGreater(s["gen_tps_stdev"], 0)

    def test_single_run_has_zero_stdev(self):
        s = summarize([{"prompt_tps": 1.0, "gen_tps": 2.0, "total_s": 3.0,
                        "prompt_tokens": 1, "gen_tokens": 2}])
        self.assertEqual(s["gen_tps_stdev"], 0.0)


class CallGenerateTests(unittest.TestCase):
    def _call_with_urlopen(self, side_effect, timeout=30):
        with mock.patch.object(ollama_bench.urllib.request, "urlopen",
                               side_effect=side_effect):
            return call_generate("http://x", "m", "hi", {}, timeout)

    def test_returns_parsed_json(self):
        data = self._call_with_urlopen([FakeResp(TIMING)])
        self.assertEqual(data["eval_count"], 100)

    def test_http_error_becomes_bench_error(self):
        err = urllib.error.HTTPError(
            "http://x/api/generate", 404, "not found", {},
            io.BytesIO(b"no such model"))
        with self.assertRaises(BenchError) as ctx:
            self._call_with_urlopen(err)
        self.assertIn("404", str(ctx.exception))

    def test_unreachable_host_becomes_bench_error(self):
        with self.assertRaises(BenchError) as ctx:
            self._call_with_urlopen(urllib.error.URLError("refused"))
        self.assertIn("Cannot reach Ollama", str(ctx.exception))

    def test_socket_timeout_becomes_bench_error(self):
        # A server that accepts the request but never answers must surface
        # as BenchError, not a raw traceback.
        with self.assertRaises(BenchError) as ctx:
            self._call_with_urlopen(TimeoutError("timed out"), timeout=5)
        self.assertIn("timed out after 5s", str(ctx.exception))


class BenchModelTests(unittest.TestCase):
    def _bench(self, iterations, warmup):
        calls = []

        def fake(host, model, prompt, options, timeout):
            calls.append(model)
            return TIMING

        with mock.patch.object(ollama_bench, "call_generate", fake):
            warm, summary = bench_model("http://x", "m", "hi", {}, 30,
                                        iterations=iterations, warmup=warmup)
        return calls, warm, summary

    def test_does_warmup_then_iterations(self):
        calls, warm, summary = self._bench(iterations=3, warmup=True)
        self.assertEqual(len(calls), 4)  # 1 warmup + 3 timed
        self.assertEqual(warm["gen_tps"], 20.0)
        self.assertEqual(summary["gen_tps"], 20.0)

    def test_no_warmup(self):
        calls, warm, _ = self._bench(iterations=2, warmup=False)
        self.assertIsNone(warm)
        self.assertEqual(len(calls), 2)


class FormatComparisonTests(unittest.TestCase):
    def test_sorts_fastest_first(self):
        results = [("slow", {"load_s": 1.0}, _summary(10.0)),
                   ("fast", {"load_s": 0.5}, _summary(40.0))]
        table = format_comparison(results)
        self.assertLess(table.index("fast"), table.index("slow"))
        self.assertIn("Model", table)
        self.assertIn("Gen tok/s", table)

    def test_lists_every_model_once(self):
        results = [("slow", {"load_s": 1.0}, _summary(10.0)),
                   ("fast", {"load_s": 0.5}, _summary(40.0))]
        table = format_comparison(results)
        data_lines = [line for line in table.splitlines()
                      if "slow" in line or "fast" in line]
        self.assertEqual(len(data_lines), 2)
        self.assertEqual(sum("slow" in line for line in data_lines), 1)
        self.assertEqual(sum("fast" in line for line in data_lines), 1)


if __name__ == "__main__":
    unittest.main()
