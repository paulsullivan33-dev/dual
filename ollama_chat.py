#!/usr/bin/env python3
"""
ollama_chat.py -- two Ollama utilities in one script (stdlib only).

Chat mode: converse with a model; history is kept for you.
    python3 ollama_chat.py chat
    python3 ollama_chat.py chat --model qwen3:8b --system "You are a pirate."

Duel mode: two models/personas converse with each other.
    python3 ollama_chat.py duel --topic "Tabs or spaces?"
    python3 ollama_chat.py duel \\
        --model-a qwen3:4b --name-a "Optimist" --system-a "You are relentlessly optimistic." \\
        --model-b qwen3:8b --name-b "Skeptic"  --system-b "You are a grumpy skeptic." \\
        --topic "Should we rewrite it in Rust?" --turns 8

Ollama must be running (default http://localhost:11434).
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error


def strip_think_tags(text):
    """Remove <think>...</think> blocks a model may emit inside the reply text.

    Some models (notably Qwen) leak their reasoning into the content even when
    thinking is disabled via the API, sometimes with no opening <think> tag.
    Handles complete blocks, a closing tag with no opening tag (strip
    everything through it), and a stray opening tag with no close (cut from
    it to the end).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def call_chat(host, model, messages, think=False, num_predict=300, timeout=900):
    """Single non-streaming /api/chat call. Returns (thinking, reply)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"num_predict": num_predict},
    }
    req = urllib.request.Request(
        host + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code} from Ollama: {body}\nHint: did you run `ollama pull {model}`?")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach Ollama at {host}: {e.reason}\nIs `ollama serve` running?")
    msg = data["message"]
    return msg.get("thinking", "").strip(), strip_think_tags(msg["content"])


def print_turn(thinking, reply, show_thinking):
    """Print thinking (indented, labeled) and reply as clearly separate blocks."""
    if show_thinking and thinking:
        print("  Thinking:")
        for line in thinking.splitlines():
            print("    " + line)
        print("  Reply:")
    print(reply)


def cmd_chat(args):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    print(f"Chatting with {args.model} -- blank line, 'quit', or Ctrl-C to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if text.lower() in ("quit", "exit", ":q"):
            print("Bye.")
            return
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        print(f"\n{args.model}:")
        print("  (waiting for reply...)", file=sys.stderr, flush=True)
        thinking, reply = call_chat(args.host, args.model, messages,
                                    think=args.think, num_predict=args.max_tokens)
        messages.append({"role": "assistant", "content": reply})
        print_turn(thinking, reply, show_thinking=args.think)
        print()


def cmd_duel(args):
    personas = [
        {"model": args.model_a, "name": args.name_a, "system": args.system_a},
        {"model": args.model_b, "name": args.name_b, "system": args.system_b},
    ]
    transcript = []  # list of (speaker_index, text)
    print(f"Topic: {args.topic}\n")
    try:
        for turn in range(args.turns):
            i = turn % 2
            me = personas[i]
            # Build the message list from this speaker's point of view:
            # their own past lines are "assistant", the other's are "user".
            messages = []
            if me["system"]:
                messages.append({"role": "system", "content": me["system"]})
            for spk, text in transcript:
                role = "assistant" if spk == i else "user"
                messages.append({"role": role, "content": text})
            if not transcript:
                messages.append({"role": "user", "content": args.topic})
            print(f"[{me['model']} as {me['name']}]")
            print("  (waiting for reply...)", file=sys.stderr, flush=True)
            thinking, reply = call_chat(args.host, me["model"], messages,
                                        think=args.think, num_predict=args.max_tokens)
            transcript.append((i, reply))
            # Label, then thinking and reply as distinct blocks, then a blank
            # line so speakers stay visually distinct.
            print_turn(thinking, reply, show_thinking=args.think)
            print()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    print(f"Done: {len(transcript)} replies.", file=sys.stderr)


def main():
    # Common options live on a parent parser so they work both before and
    # after the subcommand: `ollama_chat.py --think duel ...` and
    # `ollama_chat.py duel --think ...` are equivalent.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="http://localhost:11434",
                        help="Ollama base URL")
    common.add_argument("--think", action="store_true",
                        help="show the model's reasoning (thinking) as well as its reply")
    common.add_argument("--max-tokens", type=int, default=None,
                        help="max tokens per response (default 300, or 2048 with --think, "
                             "since thinking tokens count against the same budget)")

    p = argparse.ArgumentParser(
        description="Chat with Ollama, or make two models talk to each other.",
        parents=[common])
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("chat", help="converse with a single model", parents=[common])
    c.add_argument("--model", default="qwen3:4b")
    c.add_argument("--system", default=None, help="system prompt / persona")

    d = sub.add_parser("duel", help="two models converse with each other", parents=[common])
    d.add_argument("--topic", required=True, help="opening topic or question")
    d.add_argument("--turns", type=int, default=6, help="total replies (default 6)")
    d.add_argument("--model-a", default="qwen3:4b")
    d.add_argument("--model-b", default="qwen3:4b")
    d.add_argument("--name-a", default="AI-A")
    d.add_argument("--name-b", default="AI-B")
    d.add_argument("--system-a", default=None, help="persona for A, e.g. 'You are an optimist.'")
    d.add_argument("--system-b", default=None, help="persona for B, e.g. 'You are a skeptic.'")

    args = p.parse_args()

    # Model output often contains emoji/CJK/smart quotes, which crash
    # Windows consoles on a legacy code page (e.g. cp1252 PowerShell).
    # Force UTF-8 output; fall back silently on non-standard streams.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Thinking tokens come out of the same num_predict budget as the reply,
    # so --think needs a much larger default or the reply gets starved.
    if args.max_tokens is None:
        args.max_tokens = 2048 if args.think else 300
    if args.mode == "chat":
        cmd_chat(args)
    else:
        cmd_duel(args)


if __name__ == "__main__":
    main()
