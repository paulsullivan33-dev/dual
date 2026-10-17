"""ollama_judge.py -- an impartial third model scores a finished duel.

A scenario opts in with a top-level "judge" key, either a model name::

    "judge": "qwen3:8b"

or a dict for more control::

    "judge": {
        "model": "llama3.1:8b",
        "host": "http://192.168.1.236:11434",
        "prompt": "Declare the winner based on code correctness only.",
        "temperature": 0.2,
        "max_tokens": 500,
        "think": false
    }

The judge should run on **another machine** than the duel -- a bigger box
that can hold a judge model larger than the contestants, so judging never
steals RAM or CPU from the duel itself. "host" is that machine's Ollama
endpoint; it defaults to the duel host when unset.

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

JUDGE_KEYS = {"model", "host", "prompt", "temperature", "max_tokens",
              "think", "num_ctx"}

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


class JudgeContextExceeded(Exception):
    """The complete judging request exceeds the permitted context window."""


def judge_context_window(judge, messages):
    """Budget ~4 characters/token, message overhead, 20% input margin,
    and the full configured output allowance. Explicit num_ctx stays fixed.
    This estimate is a safeguard, not an exact model tokenizer.
    """
    chars = sum(len(message["content"]) for message in messages)
    estimated_input = (chars + 3) // 4 + 16 * len(messages)
    required = estimated_input + (estimated_input + 4) // 5
    required += judge.get("max_tokens") or 500
    explicit = judge.get("num_ctx")
    windows = (explicit,) if explicit else (8192, 16384)
    for window in windows:
        if required <= window:
            return window, required
    raise JudgeContextExceeded(
        f"estimated budget {required} tokens (including reply allowance and "
        f"20% input margin) exceeds the {windows[-1]}-token "
        f"{'configured window' if explicit else 'automatic cap'}. "
        "Use a shorter conversation or explicitly set judge.num_ctx to a "
        "larger window supported by the judge model and host.")


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
        _validate_field(raw, "host", "str", '"judge"')
        _validate_field(raw, "temperature", "number", '"judge"', minimum=0)
        _validate_field(raw, "max_tokens", "int", '"judge"', minimum=1)
        _validate_field(raw, "num_ctx", "int", '"judge"', minimum=1)
        _validate_field(raw, "think", "bool", '"judge"')
        return dict(raw)
    sys.exit('"judge" must be a model name or a settings dict, '
             f'got {raw!r}.')


def build_judge_messages(judge, topic, participants, transcript,
                         run_reports=None):
    """The messages the judge model sees: its instructions plus the full
    transcript, each turn labeled with the speaker's name. When the
    participants carry assigned briefs (their scenario system prompts),
    the briefs are shown ahead of the transcript with a fidelity
    instruction -- without them the judge has no way to know what each
    side was supposed to argue. When the duel
    ran code (run_reports maps transcript indexes to code-run reports),
    each reply is followed by what its program actually did -- a program
    that crashed or printed wrong output should count against its author,
    so the instructions say so."""
    a, b = participants
    prompt = judge.get("prompt") or DEFAULT_JUDGE_PROMPT
    instructions = prompt.replace("{a}", a["name"]).replace("{b}", b["name"])
    # Plain replace, not str.format, so braces elsewhere survive.
    instructions = instructions.replace("{topic}", topic)
    if run_reports:
        instructions += (
            "\n\nEach reply above is followed by the result of running its "
            "code. Judge the code by what it did, not by what its author "
            "claimed: a program that crashed, timed out, or printed wrong "
            "output counts heavily against its author.")
    # The assigned briefs (the participants' system prompts), when the
    # scenario gave any. Without them the judge cannot tell a held
    # position from an abandoned one -- a duelist arguing the other
    # side's brief reads as merely "consistent". Judges scored exactly
    # that as a win before this section existed.
    briefs = []
    for participant in participants:
        system = participant.get("system")
        if isinstance(system, str) and system.strip():
            briefs.append(f"--- {participant['name']}'s assigned brief "
                          f"---\n{system.strip()}")
    if briefs:
        instructions += (
            "\n\nEach participant was assigned the brief shown below, "
            "before the transcript. Fidelity to that brief is a gate, "
            "decided before any scoring: for each participant, judge "
            "whether they ended the duel still arguing their own "
            "brief's position, or crossed over to the other side's "
            "assigned position. Escalating inside a brief -- pressing "
            "harder, naming higher costs -- is fidelity, however "
            "intense it gets; adopting the opposing brief's conclusion "
            "is abandonment, and nothing in the transcript justifies "
            "it. A participant who abandoned their brief cannot win: "
            "when one side crossed and the other held, the one who "
            "held wins regardless of the remaining scores and must be "
            "scored higher. When both crossed, the verdict is a draw.")
    turns = []
    for idx, ((i, text)) in enumerate(transcript):
        turns.append(f"--- {participants[i]['name']} ---\n{text}")
        if run_reports and idx in run_reports:
            turns.append(run_reports[idx])
    body = "\n\n".join(turns)
    content = instructions
    if briefs:
        content += "\n\nAssigned briefs:\n" + "\n\n".join(briefs)
    content += "\n\nTranscript:\n" + body
    return [
        {"role": "system",
         "content": "You are an impartial judge of debates and games."},
        {"role": "user", "content": content},
    ]


def parse_winner(verdict):
    """Pull the declared winner out of a verdict ("WINNER: <name>"), or
    None when the judge didn't follow the format. Tolerates markdown bold
    ("**WINNER:** Alice"), which judges emit constantly."""
    if not verdict:
        return None
    match = WINNER_RE.search(verdict.replace("**", ""))
    return match.group(1).strip() if match else None


def run_judge(host, judge, topic, participants, transcript, timeout,
              call_chat, run_reports=None):
    """Ask the judge model to score the transcript. Returns
    (verdict text, num_ctx used, estimated budget) -- the sizing pair
    lets the caller record which window judged the duel (or, on
    JudgeContextExceeded, why none did) in the run summary.
    `host` is the judge's Ollama endpoint -- normally another
    machine, bigger than the duel box. `call_chat` is passed in
    (ollama_duel's) so tests can fake it. `run_reports` maps transcript
    indexes to code-run reports; the judge sees them.
    Raises OllamaError when the judge call fails -- the caller decides
    whether that is fatal (it shouldn't be).

    Defaults: temperature 0.2; automatically choose 8192 or 16384 context
    tokens. An explicit num_ctx is honored as a fixed window. Raises
    JudgeContextExceeded before any request if the estimated budget cannot
    fit. The caller reports the skipped judge without failing the duel.
    """
    messages = build_judge_messages(judge, topic, participants, transcript,
                                    run_reports)
    num_ctx, required = judge_context_window(judge, messages)
    print(f"Judge context: {num_ctx} tokens; estimated budget {required} "
          "including reply allowance and safety margin.")
    options = {"num_predict": judge.get("max_tokens") or 500,
               "temperature": judge.get("temperature", 0.2),
               "num_ctx": num_ctx}
    _thinking, reply, _done_reason, _metrics = call_chat(
        host, judge["model"], messages, judge.get("think", False), options,
        timeout=timeout)
    return reply.strip(), num_ctx, required
