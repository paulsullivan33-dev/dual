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
| `--timeout` | Per-request timeout, in seconds | 1200 |
| `--display` / `--no-display` | Show live duel stats (progress bar, tokens/sec) on the Arduino Uno Q's built-in 8x13 LED matrix | Off |
| `--save-json` | Write the finished transcript to this JSON file | Off |
| `--temperature` | Sampling temperature passed to Ollama | Server default |
| `--num-ctx` | Context window size passed to Ollama | Server default |

Chat accepts `--model` (default `qwen3:4b`) and `--system`. Duel requires `--topic` and accepts `--turns` (default 6), `--model-a`, `--model-b` (both default `qwen3:4b`), `--name-a`, `--name-b`, `--system-a`, and `--system-b`. In duel mode, `--temperature` and `--num-ctx` apply to both participants; for per-participant values, use `ollama_duel.py` with a JSON config instead.

For `chat`, `--save-json` writes the message list (`role`/`content` pairs) when you quit. For `duel`, it writes one entry per reply (`speaker`, `model`, `text`) — including whatever was generated before a Ctrl-C or an Ollama error stopped the duel early.

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
| `turns` | Top level | Total replies; defaults to 6; must be a positive integer |
| `log_file` | Top level | Optional path for the transcript; a date/time stamp is prepended to the file name so each run gets its own log |
| `save_json` | Top level | Optional path to write the structured transcript (`speaker`/`model`/`text` per reply) as JSON when the duel ends, including after an early stop |
| `timeout` | Top level | Per-request timeout in seconds; defaults to 1200 |
| `display` | Top level | Show live duel stats on the Arduino Uno Q's built-in 8x13 LED matrix; defaults to false. Needs `python3-smbus` on the Uno Q. The script runs headless with a warning anywhere the matrix is unreachable, so this is safe to leave on in shared configs |
| `think` | Top level or model entry | Request and display thinking; defaults to false |
| `max_tokens` | Top level or model entry | Passed as Ollama's `num_predict`; defaults to 300, or 2048 when thinking is enabled |
| `temperature` | Top level or model entry | Passed to Ollama if specified |
| `num_ctx` | Top level or model entry | Context-window setting passed to Ollama if specified |
| `turn_prompt` | Top level or model entry | Instruction added as the last message on every turn after the first, to keep each reply answering the other participant. `{name}` and `{other}` are replaced with the speaker's and the other participant's names. Defaults to asking for a direct reply of a few short paragraphs that moves the exchange forward; set to `""` to turn it off. Useful for scenarios that want a full program or a long passage each turn |
| `repeat_penalty` | Top level or model entry | Passed to Ollama if specified; must be at least 1. Values above 1 (e.g. `1.1`–`1.3`) discourage the model from repeating itself; `1` means no penalty |

Model-level `think`, `max_tokens`, `temperature`, `num_ctx`, `repeat_penalty`, and `turn_prompt` override their top-level values. The CLI supports `--topic`, `--turns`, `--host`, `--log-file`, `--save-json`, `--timeout`, `--think`, and `--no-think`; these override the corresponding configuration values. The two thinking flags apply to both participants. Change model identifiers and generation budgets in JSON; `ollama_duel.py` has no `--model` or `--max-tokens` option.

Invalid settings (an unrecognized key, wrong type, a `turns`/`max_tokens`/`num_ctx` less than 1, a negative `temperature`, or a `repeat_penalty` below 1) are rejected with an error naming the offending key before any request is sent — a typo like `"temprature"` fails loudly instead of silently falling back to a default.

If Ollama becomes unreachable or returns an error mid-duel, the duel stops the way Ctrl-C does: it prints the error, keeps whatever replies were already generated, and still writes the log file's end marker and `save_json` transcript.

### Included scenarios

| File | Scenario |
| --- | --- |
| `duel-example.json` | Optimist and skeptic discuss rewriting software in Rust |
| `ai_will_kill_us.json` | Debate the proposition "AI will kill us" |
| `ai_will_save_us_all.json` | Debate the proposition "AI will save us all" |
| `duel-coder-vs-gemma.json` | Pragmatic and enthusiastic developers debate AI coding assistants |
| `duel-coder-vs-gemma2.json` | Alternate configuration for the same developer debate |
| `program_writing.json` | Two programmers take turns proposing and improving a single Python program |
| `program_writing_v2.json` | The same program-writing duel on `qwen2.5-coder:14b`, with thinking off |
| `factorial.json` | Two programmers take turns improving a single-file Python factorial program |
| `murder_crime_novel_uncensored.json` | An abusive optimist and skeptic build a murder/crime novel, using an uncensored model |
| `ai_driven_crime_novel.json` | An optimistic investigator and cynical detective build a murder mystery |
| `slow_crime.json` | A long (20-turn) sci-fi murder mystery written paragraph by paragraph on `tinyllama`, for small machines |
| `slow_crime_v3.json` | The same 20-turn mystery on `qwen3:8b`, with `repeat_penalty` set to curb repetitive prose |
| `salesperson_vs_customer.json` | A salesperson and customer negotiate a car purchase |
| `devops_interview.json` | A hiring manager interviews a DevOps candidate, one question at a time |
| `time_traveler_1988.json` | A visitor from December 1988 meets modern tech; a patient explainer answers |
| `code_review_duel.json` | A security-paranoid reviewer vs. a ship-it pragmatist on the same snippet |
| `socratic_debugging.json` | Two programmers take turns proposing and stress-testing bug hypotheses |
| `first_contact.json` | A human diplomat and an alien envoy negotiate first contact |
| `scope_negotiation.json` | A product manager and engineer negotiate scope against a tight deadline |
| `incident_response.json` | An on-call engineer and SRE lead triage a live production outage |
| `architecture_review.json` | Two engineers argue microservices vs. a monolith for a new system |
| `trolley_problem_ethics.json` | A utilitarian and a deontologist debate a self-driving-car dilemma |
| `ai_consciousness_debate.json` | A materialist and a skeptic debate whether an LLM could be conscious |
| `text_adventure_dungeon.json` | A Dungeon Master and an adventurer build a fantasy dungeon crawl together |
| `roast_battle.json` | Two comedians trade escalating, good-natured roasts |
| `roast_battle_v2.json` | The same roast battle with one comedian on an uncensored 35B model |
| `sports_commentary_duel.json` | Two rival commentators call a fictional, escalating championship finish |
| `angry_customer_support.json` | A frustrated customer and a support rep work toward a resolution |
| `tube_vs_modeler.json` | A tube-amp purist and a digital-modeler fan debate gigging tone |
| `salary_negotiation.json` | A DevOps candidate negotiates an offer with a hiring manager |
| `alien_food_critics.json` | Two rival alien critics review human cuisine |
| `vim_vs_emacs.json` | Lifelong believers argue the eternal editor war |
| `baseball_mvp_debate.json` | A stat-head and an old-school analyst debate the MVP |

## Conversation behavior and logs

Each request includes the participant's system prompt and all previous generated replies. The participant's own replies are represented as assistant messages; the other participant's replies are represented as user messages. The topic opens every request, so both participants see it on every turn.

Requests are sequential and non-streaming: a complete reply appears after the server finishes generating it. Each request has a 1200-second timeout. Press Ctrl+C to stop a duel early.

The scripts remove inline `<think>` blocks from reply text. When thinking is enabled, they separately display the server's `thinking` field when available. Thinking behavior depends on the model and server, and its token use can reduce the budget available for the visible reply.

When `log_file` is set, `ollama_duel.py` mirrors its standard output to a timestamped copy of that file (e.g. `"my-duel.log"` becomes `"20260925-084500-my-duel.log"`) while also printing it to the console, so each run gets its own log. Logs include session start markers and, on normal completion or a handled Ctrl+C, session end markers. Progress messages and the final reply count go to standard error and are not mirrored. Relative log paths are resolved from the directory where you run the command, and parent directories must already exist. Omit `log_file` to disable logging. Saved logs are not automatically loaded into a later session.

The programming scenario produces code as conversation text. Neither script executes, tests, or automatically saves generated code as a Python file.

## Choosing models

Match the model size to the machine running Ollama. Any model the server
has pulled works — put its exact name in the scenario's `"model"` field.

| Machine class | Example models | Notes |
|---|---|---|
| Memory-constrained single-board computers (a few GB of RAM, weak CPU) | `qwen3:0.6b`, `qwen3:1.7b`, `smollm2:1.7b`, `tinyllama:1.1b` | Expect a few tokens per second. Keep `turns` low and thinking budgets small; turn thinking off if replies get too slow. |
| Older CPU-only desktops | `qwen3:4b`, `qwen3:8b`, `llama3.1:8b` | 8B models are the sweet spot for CPU-only machines with 12GB+ of RAM. |
| Modern machines with ample RAM or a GPU | `qwen2.5-coder:14b`, `qwen3:14b`, larger 20B–30B models | Best duel quality. Note: `qwen2.5-coder:14b` rejects thinking-enabled requests, so use `"think": false` with it; the Qwen3 family supports thinking. |

On a slow machine, prefer shorter `max_tokens` values and fewer turns — a
duel that takes minutes on a fast machine can take an hour or more on a
tiny one. The 1200-second default timeout is there to cover slow
generations. For an always-on low-power box, small models chugging away
unattended beat fast models competing for cycles on your main machine.

## Benchmarking model speed

`ollama_bench.py` measures tokens/second for one or more models so you can
compare them objectively:

```shell
python ollama_bench.py qwen3:8b
python ollama_bench.py qwen3:8b qwen2.5-coder:14b --iterations 5
```

Each model gets one warmup run (loads the model; not counted), then the
requested number of timed runs of a fixed prompt. It reports
prompt-processing speed, generation speed, and total time per run —
averaged across runs — then prints a comparison table sorted by generation
speed. Ollama reports exact token counts and timings, so the figures are
measured, not estimated. Thinking is forced off so thinking and
non-thinking models compare fairly. Useful flags: `--iterations`/`-n`,
`--max-tokens`, `--num-ctx`, `--prompt`, `--host`, `--timeout`,
`--no-warmup`.

## Troubleshooting

- **Cannot reach Ollama:** check that the server is running and `--host` points to it.
- **HTTP error or missing model:** check the error text and ensure the model is available on the selected server; use `ollama pull MODEL` for an available model identifier.
- **Empty or truncated replies:** increase the generation budget, especially when thinking is enabled. For JSON scenarios, check for per-model `max_tokens` overrides.
- **Long waits:** the novel and debate scenarios allow up to 16,384 tokens per reply. Lower the applicable `max_tokens` values for shorter experiments.
- **Truncated replies:** generation silently stops when the model's context window fills, so keep `max_tokens` at or below the effective window — the `num_ctx` setting, or the server default when `num_ctx` is unset. Raising `max_tokens` without raising `num_ctx` does not produce longer replies. `ollama_duel.py` prints a warning at startup when a participant's `max_tokens` is larger than its `num_ctx`, or above 4096 with no `num_ctx` set.
- **Replies loop or repeat phrases:** set `repeat_penalty` to something like `1.1`–`1.3`, at the top level or for just the participant that repeats. Very high values can make wording erratic.
- **Long conversations lose details:** the scripts resend the transcript without summarizing it, but the model's context capacity still limits what it can use.

For the full command-line help:

```shell
python ollama_chat.py chat --help
python ollama_chat.py duel --help
python ollama_duel.py --help
```

## Running the tests

The `tests/` directory has stdlib-only `unittest` coverage for the shared helpers (`ollama_common.py`), config loading and validation (`ollama_duel.py`), CLI argument handling (`ollama_chat.py`), benchmarking (`ollama_bench.py`), and the LED matrix driver (`unoq_matrix.py`) — no live Ollama server required; network calls are mocked. It also checks that every included scenario JSON file loads and validates.

```shell
python -m unittest discover -s tests
```
