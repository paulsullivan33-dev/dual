# Dual

Command-line Python utilities for chatting with an Ollama model or letting two AI personas take turns in a conversation. Use them for debates, collaborative stories, programming discussions, and role-play experiments.

The scripts use only the Python standard library. They send requests to an Ollama server; they do not run models themselves.

## Requirements and setup

- Python 3.7 or later.
- Ollama running at `http://localhost:11434`, or another reachable Ollama host.
- The models named in your command or JSON configuration available on that server.

Clone this repository and enter its directory:

```shell
git clone https://github.com/paulsullivan33-dev/dual.git
cd dual
```

For the examples below, prepare the default model:

```shell
ollama pull qwen3:4b
```

If Ollama is not already running, start it in another terminal with `ollama serve`. No `pip install` step is needed. Examples use `python`; substitute `python3` or `py` if that is how Python is installed on your system.

## Interactive chat

```shell
python ollama_chat.py chat
python ollama_chat.py chat --model qwen3:4b --system "You are a patient Python tutor." --max-tokens 600
```

Type a message at the `You:` prompt. Conversation history is retained in memory for the current session. Enter `quit`, `exit`, or `:q` to leave; blank lines are ignored. You can also press Ctrl+C at the input prompt.

## Two-persona conversation from the command line

```shell
python ollama_chat.py duel --topic "Should we rewrite it in Rust?" --turns 8 --model-a qwen3:4b --name-a Optimist --system-a "You are relentlessly optimistic." --model-b qwen3:4b --name-b Skeptic --system-b "You question assumptions and practical costs."
```

Both participants can use the same model with different system prompts, or different models. The first participant speaks first, then the scripts alternate between them. `--turns 8` means eight total replies, four from each participant.

Common options for `ollama_chat.py` go after the `chat` or `duel` subcommand:

| Option | Purpose | Default |
| --- | --- | --- |
| `--host` | Ollama server base URL | `http://localhost:11434` |
| `--think` | Request thinking and display the separate thinking field when returned | Off |
| `--max-tokens` | Per-response generation budget | 300, or 2048 with `--think` |

Chat accepts `--model` (default `qwen3:4b`) and `--system`. Duel requires `--topic` and accepts `--turns` (default 6), `--model-a`, `--model-b` (both default `qwen3:4b`), `--name-a`, `--name-b`, `--system-a`, and `--system-b`.

## Run a JSON scenario

`ollama_duel.py` adds reusable configuration, per-participant generation settings, and optional transcript logging.

```shell
python ollama_duel.py duel-example.json
python ollama_duel.py ai_will_kill_us.json --turns 4 --no-think
python ollama_duel.py salesperson_vs_customer.json --log-file negotiation.log
python ollama_duel.py duel-example.json --topic "Should a small team adopt AI coding tools?" --turns 6
```

Before running a supplied scenario, inspect its `models` entries and pull those exact models, or replace them with models available on your server. For example, `duel-example.json` uses both `qwen3:4b` and `qwen3:8b`.

### Create your own configuration

Save the following as `my-duel.json`, then run `python ollama_duel.py my-duel.json`:

```json
{
  "host": "http://localhost:11434",
  "topic": "Should our team adopt a four-day workweek?",
  "turns": 6,
  "think": false,
  "max_tokens": 600,
  "temperature": 0.7,
  "log_file": "my-duel.log",
  "models": [
    {
      "model": "qwen3:4b",
      "name": "Advocate",
      "system": "Argue for a four-day workweek with concrete examples."
    },
    {
      "model": "qwen3:4b",
      "name": "Skeptic",
      "system": "Question the costs and operational risks of a four-day workweek.",
      "temperature": 0.4
    }
  ]
}
```

The JSON must be an object with exactly two entries in `models`. Each entry requires `model`; `name` defaults to the model identifier, and `system` is optional. Provide a nonempty topic in the file or through `--topic`.

| Setting | Where it belongs | Behavior |
| --- | --- | --- |
| `host` | Top level | Server URL; defaults to `http://localhost:11434` |
| `topic` | Top level | Opening prompt for the first participant |
| `turns` | Top level | Total replies; defaults to 6 |
| `log_file` | Top level | Optional path for an appended transcript |
| `think` | Top level or model entry | Request and display thinking; defaults to false |
| `max_tokens` | Top level or model entry | Passed as Ollama's `num_predict`; defaults to 300, or 2048 when thinking is enabled |
| `temperature` | Top level or model entry | Passed to Ollama if specified |
| `num_ctx` | Top level or model entry | Context-window setting passed to Ollama if specified |

Model-level `think`, `max_tokens`, `temperature`, and `num_ctx` override their top-level values. The CLI supports `--topic`, `--turns`, `--host`, `--log-file`, `--think`, and `--no-think`; these override the corresponding configuration values. The two thinking flags apply to both participants. Change model identifiers and generation budgets in JSON; `ollama_duel.py` has no `--model` or `--max-tokens` option.

### Included scenarios

| File | Scenario |
| --- | --- |
| `duel-example.json` | Optimist and skeptic discuss rewriting software in Rust |
| `ai_will_kill_us.json` | Debate the proposition "AI will kill us" |
| `ai_will_save_us_all.json` | Debate the proposition "AI will save us all" |
| `duel-coder-vs-gemma.json` | Pragmatic and enthusiastic developers debate AI coding assistants |
| `duel-coder-vs-gemma2.json` | Alternate configuration for the same developer debate |
| `program_writing.json` | Two programmers take turns proposing and improving a single Python program |
| `adventure_novel.json` | Despite the filename, its current prompt requests a murder/crime novel |
| `ai_driven_crime_novel.json` | An optimistic investigator and cynical detective build a murder mystery |
| `salesperson_vs_customer.json` | A salesperson and customer negotiate a car purchase |

## Conversation behavior and logs

Each request includes the participant's system prompt and all previous generated replies. The participant's own replies are represented as assistant messages; the other participant's replies are represented as user messages. The initial topic is sent only on the first request, so later turns rely on the generated conversation to retain it.

Requests are sequential and non-streaming: a complete reply appears after the server finishes generating it. Each request has a 900-second timeout. Press Ctrl+C to stop a duel early.

The scripts remove inline `<think>` blocks from reply text. When thinking is enabled, they separately display the server's `thinking` field when available. Thinking behavior depends on the model and server, and its token use can reduce the budget available for the visible reply.

When `log_file` is set, `ollama_duel.py` appends its standard output to that file while also printing it to the console. Logs include session start markers and, on normal completion or a handled Ctrl+C, session end markers. Progress messages and the final reply count go to standard error and are not mirrored. Relative log paths are resolved from the directory where you run the command, and parent directories must already exist. Omit `log_file` to disable logging. Saved logs are not automatically loaded into a later session.

The programming scenario produces code as conversation text. Neither script executes, tests, or automatically saves generated code as a Python file.

## Troubleshooting

- **Cannot reach Ollama:** check that the server is running and `--host` points to it.
- **HTTP error or missing model:** check the error text and ensure the model is available on the selected server; use `ollama pull MODEL` for an available model identifier.
- **Empty or truncated replies:** increase the generation budget, especially when thinking is enabled. For JSON scenarios, check for per-model `max_tokens` overrides.
- **Long waits:** the novel and debate scenarios allow up to 16,384 tokens per reply. Lower the applicable `max_tokens` values for shorter experiments.
- **Truncated replies:** generation silently stops when the model's context window fills, so keep `max_tokens` at or below the effective window — the `num_ctx` setting, or the server default when `num_ctx` is unset. Raising `max_tokens` without raising `num_ctx` does not produce longer replies.
- **Long conversations lose details:** the scripts resend the transcript without summarizing it, but the model's context capacity still limits what it can use.

For the full command-line help:

```shell
python ollama_chat.py chat --help
python ollama_chat.py duel --help
python ollama_duel.py --help
```
