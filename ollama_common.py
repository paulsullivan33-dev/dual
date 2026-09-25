"""ollama_common.py -- shared helpers for ollama_chat.py and ollama_duel.py.

Not a public API; just the bits that would otherwise be copy-pasted between
the two scripts.
"""

import json
import re
import sys
import textwrap
import urllib.request
import urllib.error

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 1200


class OllamaError(Exception):
    """Raised when a /api/chat request fails (bad response, unreachable host).

    Callers decide how to react: a duel loop can stop gracefully and keep
    the transcript generated so far, rather than the whole process dying
    via sys.exit mid-run.
    """


def setup_utf8_stdout():
    """Force UTF-8 stdout so emoji/CJK/smart quotes in model output don't
    crash Windows consoles on a legacy code page (e.g. cp1252 PowerShell).
    Falls back silently on non-standard streams."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


WRAP_WIDTH = 75


def wrap_text(text, width=WRAP_WIDTH):
    """Word-wrap text for console display: break at spaces so words are never
    split across lines (a single over-long word keeps its own line rather
    than being cut). Blank lines and existing paragraph breaks are kept."""
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(
                para, width=width,
                break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines)


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


def compute_metrics(data):
    """Extract tokens/sec figures from one /api/generate or /api/chat
    response dict. prompt_s/gen_s are the raw phase durations in seconds,
    useful for aggregating across runs."""
    def tps(count, duration_ns):
        seconds = duration_ns / 1e9
        return count / seconds if seconds > 0 else 0.0

    prompt_s = data.get("prompt_eval_duration", 0) / 1e9
    gen_s = data.get("eval_duration", 0) / 1e9
    return {
        "prompt_tps": tps(data.get("prompt_eval_count", 0),
                          data.get("prompt_eval_duration", 0)),
        "gen_tps": tps(data.get("eval_count", 0),
                       data.get("eval_duration", 0)),
        "prompt_s": prompt_s,
        "gen_s": gen_s,
        "total_s": data.get("total_duration", 0) / 1e9,
        "load_s": data.get("load_duration", 0) / 1e9,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "gen_tokens": data.get("eval_count", 0),
    }


def call_chat(host, model, messages, think, options, timeout=DEFAULT_TIMEOUT):
    """Single non-streaming /api/chat call.

    Returns (thinking, reply, done_reason, metrics).

    done_reason is "stop" when the model finished on its own, "length" when
    generation hit the max_tokens (num_predict) ceiling and the reply was
    truncated. metrics is the compute_metrics() dict for this turn
    (token counts and tokens/sec figures).

    Raises OllamaError on an unreachable host or an error response; never
    calls sys.exit so a caller mid-conversation can decide how to stop.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": options,
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
        raise OllamaError(
            f"HTTP {e.code} from Ollama: {body}\nHint: did you run `ollama pull {model}`?"
        ) from e
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {host}: {e.reason}\nIs `ollama serve` running?"
        ) from e
    msg = data["message"]
    return (msg.get("thinking", "").strip(),
            strip_think_tags(msg["content"]),
            data.get("done_reason", ""),
            compute_metrics(data))


def save_transcript_json(path, turns):
    """Write a duel/chat transcript to `path` as JSON.

    `turns` is a list of plain dicts (e.g. {"speaker": ..., "model": ...,
    "text": ...} for a duel, or {"role": ..., "content": ...} for a chat).
    Raises OSError on a write failure.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_transcript_json_safe(path, turns):
    """Like save_transcript_json, but a no-op when `path` is falsy and
    reports a write failure to stderr instead of raising -- "couldn't save
    the transcript" shouldn't look like "the conversation failed"."""
    if not path:
        return
    try:
        save_transcript_json(path, turns)
    except OSError as e:
        print(f"Could not save transcript to {path}: {e}", file=sys.stderr)


def build_duel_json(transcript, participants):
    """Convert a duel's (speaker_index, text) transcript into the
    {"speaker", "model", "text"} dicts save_transcript_json expects,
    using each participant's "name" and "model"."""
    return [
        {"speaker": participants[i]["name"], "model": participants[i]["model"], "text": text}
        for i, text in transcript
    ]
