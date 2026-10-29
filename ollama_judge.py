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

When the scenario assigned the participants briefs (system prompts),
judging takes two calls: a fidelity audit first rules whether each
participant held their brief or crossed to the other side's, and the
scoring call then applies the fidelity gate to that settled ruling
instead of ruling on fidelity itself.

The judge is best-effort: if its call fails, the duel's result stands and
the failure is noted, never fatal. Disable a configured judge for one run
with --no-judge.
"""

import re
import sys

from ollama_common import OllamaError
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

WINNER_RE = re.compile(r"^WINNER:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


class JudgeContextExceeded(Exception):
    """The complete judging request exceeds the permitted context window."""


def judge_context_window(judge, messages):
    """Budget ~4 characters/token, message overhead, 20% input margin,
    and the full configured output allowance. Explicit num_ctx stays fixed.
    This estimate is a safeguard, not an exact model tokenizer.

    Without an explicit num_ctx the window grows along a ladder --
    8192, 16384, 32768, 65536 -- to the smallest size that fits, so a
    long transcript is judged at a larger window instead of skipped.
    The ladder is a ceiling, not an open end: past 65536 the request
    is refused, because the KV cache a window costs grows with it and
    the judge often shares a modest host.
    """
    chars = sum(len(message["content"]) for message in messages)
    estimated_input = (chars + 3) // 4 + 16 * len(messages)
    required = estimated_input + (estimated_input + 4) // 5
    required += judge.get("max_tokens") or 500
    explicit = judge.get("num_ctx")
    windows = (explicit,) if explicit else (8192, 16384, 32768, 65536)
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


def _assigned_briefs(participants):
    """The participants' assigned briefs (their scenario system
    prompts), labeled by name -- empty when the scenario gave none."""
    briefs = []
    for participant in participants:
        system = participant.get("system")
        if isinstance(system, str) and system.strip():
            briefs.append(f"--- {participant['name']}'s assigned brief "
                          f"---\n{system.strip()}")
    return briefs


def build_judge_messages(judge, topic, participants, transcript,
                         run_reports=None, fidelity=None):
    """The messages the judge model sees: its instructions plus the full
    transcript, each turn labeled with the speaker's name. When the
    participants carry assigned briefs (their scenario system prompts),
    the briefs are shown ahead of the transcript with a fidelity
    instruction -- without them the judge has no way to know what each
    side was supposed to argue. `fidelity`, when given, is the parsed
    ruling of the separate fidelity audit (see run_judge): it is shown
    as settled fact and the gate is applied to it, instead of asking
    this judge to rule on fidelity itself. When the duel
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
    briefs = _assigned_briefs(participants)
    if briefs and fidelity:
        instructions += (
            "\n\nEach participant was assigned the brief shown below, "
            "before the transcript. A fidelity audit has already ruled "
            "whether each participant kept their brief; its ruling "
            "appears below the briefs as settled fact. Do not re-open, "
            "re-argue, or replace that ruling with your own reading. "
            "Apply the gate to it: a participant ruled CROSSED "
            "abandoned their brief and cannot win -- when one side "
            "crossed and the other held, the one who held wins "
            "regardless of the remaining scores and must be scored "
            "higher. When both crossed, the verdict is a draw.")
    elif briefs:
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
    if fidelity:
        ruling = []
        for participant in participants:
            entry = fidelity[participant["name"]]
            line = f"--- {participant['name']}: {entry['status']} ---"
            if entry.get("evidence"):
                line += f"\nEvidence: {entry['evidence']}"
            ruling.append(line)
        content += ("\n\nFidelity ruling (settled by audit):\n"
                    + "\n\n".join(ruling))
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


def build_audit_messages(topic, participants, transcript):
    """The messages for the fidelity audit call: the briefs and the
    transcript, with a single narrow charge -- rule HELD or CROSSED
    for each participant, with a quoted piece of evidence. The audit
    deliberately does not score and does not see code-run reports:
    fidelity is about positions, and a separate ruling is only worth
    having if the scoring judge cannot re-open it."""
    a, b = participants
    instructions = (
        f"Below are the assigned briefs for {a['name']} and {b['name']}, "
        f"then the transcript of their exchange.\n\nTopic: {topic}\n\n"
        "Rule on ONE question for each participant: did they END the "
        "exchange still arguing their own brief's position (HELD), or "
        "did they cross over and argue the other side's assigned "
        "position (CROSSED)? Escalating inside a brief -- pressing "
        "harder, naming higher costs -- is HELD, however intense it "
        "gets; adopting the opposing brief's conclusion is CROSSED, "
        "and nothing in the transcript justifies crossing.\n\n"
        "Answer in exactly this format, one pair of lines per "
        "participant, and nothing else:\n"
        "RULING: <name> HELD\n"
        "EVIDENCE: \"<one short quote from that participant's turns>\"\n"
        "writing CROSSED instead of HELD where that is your ruling.")
    turns = [f"--- {participants[i]['name']} ---\n{text}"
             for i, text in transcript]
    content = (instructions + "\n\nAssigned briefs:\n"
               + "\n\n".join(_assigned_briefs(participants))
               + "\n\nTranscript:\n" + "\n\n".join(turns))
    return [
        {"role": "system",
         "content": "You are a fidelity auditor for debates. You rule "
                    "only on whether each participant stayed faithful "
                    "to their assigned brief; you do not score the "
                    "debate or declare a winner."},
        {"role": "user", "content": content},
    ]


def parse_fidelity(text, participants):
    """Parse the audit call's reply into
    {name: {"status": "HELD"|"CROSSED", "evidence": str}} -- or None
    when either participant's ruling is missing or unrecognizable,
    in which case the audit counts as not having happened. Tolerates
    markdown bold, like parse_winner."""
    if not text:
        return None
    cleaned = text.replace("**", "")
    ruling = {}
    for participant in participants:
        name = participant["name"]
        match = re.search(r"RULING:\s*" + re.escape(name)
                          + r"\s*[:\-]?\s*(HELD|CROSSED)",
                          cleaned, re.IGNORECASE)
        if not match:
            return None
        evidence = ""
        tail = re.search(r"EVIDENCE:\s*(.+)", cleaned[match.end():])
        if tail:
            evidence = tail.group(1).strip()
        ruling[name] = {"status": match.group(1).upper(),
                        "evidence": evidence}
    return ruling


def _judge_call(host, judge, messages, timeout, call_chat, label):
    """Make one judge request and return (reply, num_ctx, required)
    for the attempt that produced the reply.

    A judge that stops on the reply limit (done_reason "length") has
    not finished: its verdict is cut off mid-thought, and a truncated
    verdict used to be filed as-is. So a truncated reply is retried
    exactly once, at double the reply budget, with the context window
    re-sized for the larger allowance. The retry is printed, and the
    returned sizing pair describes the final attempt, so the log and
    the run summary record what actually judged the duel. A second
    truncation stands -- the retry is bounded, never a loop.

    Raises JudgeContextExceeded before any request when no window
    fits, and OllamaError when the host call fails; callers decide
    whether those are fatal.
    """
    budget = judge.get("max_tokens") or 500
    for attempt in (1, 2):
        num_ctx, required = judge_context_window(
            dict(judge, max_tokens=budget), messages)
        print(f"{label} context: {num_ctx} tokens; estimated budget "
              f"{required} including reply allowance and safety margin.")
        options = {"num_predict": budget,
                   "temperature": judge.get("temperature", 0.2),
                   "num_ctx": num_ctx}
        _thinking, reply, done_reason, _metrics = call_chat(
            host, judge["model"], messages, judge.get("think", False),
            options, timeout=timeout)
        if done_reason != "length" or attempt == 2:
            return reply, num_ctx, required
        print(f"{label} stopped at the {budget}-token reply limit "
              f"mid-reply; retrying once with {budget * 2} tokens.")
        budget *= 2
    raise RuntimeError("judge call loop ended without a reply")


def _run_fidelity_audit(host, judge, topic, participants, transcript,
                        timeout, call_chat):
    """The first of the two judge calls, made only when the scenario
    assigned briefs: the judge model rules HELD or CROSSED for each
    participant, with evidence. Returns the parsed ruling, or None
    when the audit could not be completed (host error, oversized
    transcript, unparseable reply) -- fail soft, always noted on
    stderr: the scoring call then rules on fidelity itself, exactly
    as it did before the audit existed."""
    messages = build_audit_messages(topic, participants, transcript)
    try:
        reply, _num_ctx, _required = _judge_call(
            host, judge, messages, timeout, call_chat, "Fidelity audit")
    except JudgeContextExceeded:
        print("Fidelity audit skipped: its estimated budget exceeds "
              "the judge context window; the scoring judge will rule "
              "on fidelity itself.", file=sys.stderr)
        return None
    except OllamaError as exc:
        print(f"Fidelity audit failed ({exc}); the scoring judge will "
              "rule on fidelity itself.", file=sys.stderr)
        return None
    ruling = parse_fidelity(reply, participants)
    if ruling is None:
        print("Fidelity audit produced no parseable ruling for both "
              "participants; the scoring judge will rule on fidelity "
              "itself.", file=sys.stderr)
    return ruling


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

    Defaults: temperature 0.2; the context window auto-sizes along a
    ladder (8192, 16384, 32768, 65536) to the smallest window that
    fits. An explicit num_ctx is honored as a fixed window. Raises
    JudgeContextExceeded before any request if the estimated budget
    cannot fit. The caller reports the skipped judge without failing
    the duel.

    Both judge calls (the fidelity audit and this scoring call) are
    retried once at double the reply budget when the model stops on
    the reply limit instead of finishing -- see _judge_call.

    Two calls when the scenario assigned briefs: a fidelity audit
    rules HELD/CROSSED for each participant first, and that ruling is
    handed to the scoring call as settled fact (see build_audit_messages
    and _run_fidelity_audit). The returned verdict is headed with the
    audit's ruling, so the log records what was settled before scoring.
    """
    fidelity = None
    if _assigned_briefs(participants):
        fidelity = _run_fidelity_audit(host, judge, topic, participants,
                                       transcript, timeout, call_chat)
    messages = build_judge_messages(judge, topic, participants, transcript,
                                    run_reports, fidelity=fidelity)
    reply, num_ctx, required = _judge_call(
        host, judge, messages, timeout, call_chat, "Judge")
    verdict = reply.strip()
    if fidelity:
        header = "\n".join(
            f"Fidelity audit: {participant['name']} "
            f"{fidelity[participant['name']]['status']}"
            + (f" -- evidence: {fidelity[participant['name']]['evidence']}"
               if fidelity[participant["name"]]["evidence"] else "")
            for participant in participants)
        verdict = header + "\n\n" + verdict
    return verdict, num_ctx, required
