"""ollama_judge.py -- an impartial third model scores a finished duel.

A scenario opts in with a top-level "judge" key, either a model name::

    "judge": "qwen3:8b"

or a dict for more control::

    "judge": {
        "model": "qwen3:8b",
        "prompt": "Declare the winner based on code correctness only.",
        "temperature": 0.2,
        "max_tokens": 500,
        "think": false
    }

After the duel's turns complete, the judge reads the full transcript and
writes a short verdict ending in a "WINNER: <name>" line (or "WINNER:
draw"). The verdict is printed into the transcript log, appended to the
run summary, and its winner line goes into the ntfy notice.

The judge is best-effort: if its call fails, the duel's result stands and
the failure is noted, never fatal. Disable a configured judge for one run
with --no-judge.
"""

import re
import sys

from ollama_common import check_unknown_keys as _check_unknown_keys
from ollama_common import validate_field as _validate_field

JUDGE_KEYS = {"model", "prompt", "temperature", "max_tokens", "think",
              "num_ctx"}

DEFAULT_JUDGE_PROMPT = (
    "You are an impartial judge. Below is the transcript of an exchange "
    "between {a} and {b}.\n\n"
    "Topic: {topic}\n\n"
    "Score the participants on how well they argued or played: quality of "
    "reasoning, use of evidence, and staying in character. Do not favor "
    "verbosity or politeness alone. Explain your reasoning in 3-5 sentences, "
    "then end your reply with exactly one line:\n"
    "WINNER: <name>\n"
    "where <name> is one of: {a}, {b}, or draw."
)

WINNER_RE = re.compile(r"^WINNER:\s*(.+?)\s*$", re.MULTILINE)


def normalize_judge(raw):
    """Turn the "judge" config value into a dict, or None when unset.

    Accepts a model-name string (defaults for everything else) or a dict.
    Anything else exits with a clear message -- a typo'd judge block should
    fail loudly, not silently skip judging.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            sys.exit('"judge" must be a model name or a settings dict, '
                     'got an empty string.')
        return {"model": raw}
    if isinstance(raw, dict):
        _check_unknown_keys(raw, JUDGE_KEYS, '"judge"')
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            sys.exit('"judge" needs a "model" name, e.g. "qwen3:8b", '
                     f'got {model!r}.')
        _validate_field(raw, "prompt", "str", '"judge"')
        _validate_field(raw, "temperature", "number", '"judge"', minimum=0)
        _validate_field(raw, "max_tokens", "int", '"judge"', minimum=1)
        _validate_field(raw, "num_ctx", "int", '"judge"', minimum=1)
        _validate_field(raw, "think", "bool", '"judge"')
        return dict(raw)
    sys.exit('"judge" must be a model name or a settings dict, '
             f'got {raw!r}.')


def build_judge_messages(judge, topic, participants, transcript):
    """The messages the judge model sees: its instructions plus the full
    transcript, each turn labeled with the speaker's name."""
    a, b = participants
    prompt = judge.get("prompt") or DEFAULT_JUDGE_PROMPT
    instructions = prompt.replace("{a}", a["name"]).replace("{b}", b["name"])
    # Plain replace, not str.format, so braces elsewhere survive.
    instructions = instructions.replace("{topic}", topic)
    lines = [f"--- {participants[i]['name']} ---" for i, _ in transcript]
    body = "\n\n".join(
        f"{label}\n{text}" for label, (_, text) in zip(lines, transcript))
    return [
        {"role": "system",
         "content": "You are an impartial judge of debates and games."},
        {"role": "user", "content": instructions + "\n\nTranscript:\n" + body},
    ]


def parse_winner(verdict):
    """Pull the declared winner out of a verdict ("WINNER: <name>"), or
    None when the judge didn't follow the format."""
    if not verdict:
        return None
    match = WINNER_RE.search(verdict)
    return match.group(1).strip() if match else None


def run_judge(host, judge, topic, participants, transcript, timeout,
              call_chat):
    """Ask the judge model to score the transcript. Returns the verdict
    text. `call_chat` is passed in (ollama_duel's) so tests can fake it.
    Raises OllamaError when the judge call fails -- the caller decides
    whether that is fatal (it shouldn't be)."""
    messages = build_judge_messages(judge, topic, participants, transcript)
    options = {"num_predict": judge.get("max_tokens") or 500}
    if judge.get("num_ctx") is not None:
        options["num_ctx"] = judge["num_ctx"]
    if judge.get("temperature") is not None:
        options["temperature"] = judge["temperature"]
    _thinking, reply, _done_reason, _metrics = call_chat(
        host, judge["model"], messages, judge.get("think", False), options,
        timeout=timeout)
    return reply.strip()
