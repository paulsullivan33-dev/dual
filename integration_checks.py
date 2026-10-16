#!/usr/bin/env python3
"""Explicit live checks; never imported or run by normal test discovery."""

import argparse
import sys
import time

from ollama_common import DEFAULT_HOST, OllamaError, call_chat, setup_utf8_stdout


def check_ollama(host, model, timeout):
    """Exercise the production chat client against a real model."""
    _, reply, _, metrics = call_chat(
        host, model, [{"role": "user", "content": "Reply with a short greeting."}],
        False, {"num_predict": 64, "num_ctx": 2048, "temperature": 0},
        timeout=timeout)
    if not reply.strip():
        raise OllamaError("Ollama returned no visible reply; the live chat check failed.")
    print(f"Ollama replied using {model}: {reply[:200]}")
    print(f"Generation: {metrics['gen_tokens']} tokens, {metrics['gen_tps']:.1f} tok/s")


def check_display():
    """Exercise the working bridge and rendering code, without I2C fallback."""
    from unoq_matrix import BridgeMatrix
    matrix = BridgeMatrix()
    try:
        matrix.show_text("OK", scroll=False)
        time.sleep(1)
        matrix.progress(1, 2)
        time.sleep(1)
    finally:
        try:
            matrix.clear()
        finally:
            matrix._bridge.disconnect()  # pylint: disable=protected-access
    print("Bridge accepted text, progress, and clear frames. "
          "Confirm visually that OK and a half-width bar appeared, then cleared.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run explicitly selected live Ollama/Uno Q integration checks.")
    parser.add_argument("--ollama", action="store_true", help="make a real model request")
    parser.add_argument("--display", action="store_true", help="exercise the Uno Q Router Bridge display")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL")
    parser.add_argument("--model", default="qwen3:4b", help="already installed Ollama model")
    parser.add_argument("--timeout", type=float, default=120, help="Ollama request timeout in seconds")
    args = parser.parse_args(argv)
    if not (args.ollama or args.display):
        parser.error("select --ollama, --display, or both; no live checks run implicitly")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be a positive finite number")
    setup_utf8_stdout()
    failed = False
    checks = []
    if args.ollama:
        checks.append(("Ollama", lambda: check_ollama(args.host, args.model, args.timeout)))
    if args.display:
        checks.append(("Display transport", check_display))
    for label, check in checks:
        print(f"Checking {label}...", flush=True)
        try:
            check()
        except Exception as exc:  # checks are independent; report all selected failures
            failed = True
            print(f"FAIL: {label}: {exc}", file=sys.stderr)
        else:
            print(f"PASS: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
