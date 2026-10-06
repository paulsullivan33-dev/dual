# Dual

Command-line Python utilities for chatting with an Ollama model or letting two AI personas take turns in a conversation. Use them for debates, collaborative stories, programming discussions, and role-play experiments.

The scripts use only the Python standard library. They send requests to an Ollama server; they do not run models themselves.

## Requirements and setup

- Python 3.8 or later.
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

### Where output goes

Everything the scripts generate goes into an `output/` folder, created on first use and excluded from git:

| Path | Written by |
| --- | --- |
| `output/logs/` | Duel transcript logs (`log_file`, `--log-file`) |
| `output/run_results.log` | One summary per duel run (`results_log`, `--results-log`) |
| `output/<name>.json` | Transcripts saved with `save_json` / `--save-json` |
| `output/agent_out/`, `output/<project>/` | Projects written by `ollama_agent.py` (`--output-dir`) |
| `output/rag_demo.db` | The RAG demo's search index (`--db`) |

Any relative path you give for these settings is placed under `output/` (so `--log-file logs/x.log` writes `output/logs/<timestamp>-x.log`); an absolute path is used exactly as given. Input files such as scenarios and the RAG demo's `--docs` folder are read from wherever you point them.

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
python ollama_duel.py scenarios/duel-example.json
python ollama_duel.py scenarios/ai_will_kill_us.json --turns 4 --no-think
python ollama_duel.py scenarios/salesperson_vs_customer.json --log-file logs/negotiation.log
python ollama_duel.py scenarios/duel-example.json --topic "Should a small team adopt AI coding tools?" --turns 6
```

## Run a batch of scenarios

Pass a directory to run every `*.json` scenario inside it, or a glob pattern
to run the matches (quote the pattern so your shell doesn't expand it first).
Each scenario runs as its own process — its own log file, its own completion
notice, its own entry in `output/run_results.log` — and a failed scenario is
reported and skipped while the rest of the batch continues. CLI overrides
apply to every scenario in the batch; Ctrl-C stops the whole batch.

```shell
python ollama_duel.py scenarios/
python ollama_duel.py "scenarios/small_*"
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
  "log_file": "logs/my-duel.log",
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
| `log_file` | Top level | Optional path for the transcript; a date/time stamp is prepended to the file name so each run gets its own log. Relative paths go under `output/`, so the included scenarios' `logs/...` paths write to `output/logs/`; missing folders are created |
| `save_json` | Top level | Optional path to write the structured transcript (`speaker`/`model`/`text` per reply) as JSON when the duel ends, including after an early stop; relative paths go under `output/` |
| `results_log` | Top level | Optional path for the run summary; defaults to `output/run_results.log`, and relative paths go under `output/`. After every duel a one-block entry is appended: date/time, config, models, turns completed, and the per-model stats table on success or the error message when the duel stopped early. Missing folders are created; pass `--no-results-log` to disable |
| `ntfy_url` | Top level | Optional ntfy topic URL; when set, the script POSTs a short completion notice (finished / stopped early / crashed, with models and turns) after every duel. Resolution order: `--ntfy-url`, the scenario file, then `~/.dual.conf` (`{"ntfy_url": "https://ntfy.sh/my-topic"}`). Undefined everywhere means notifications are skipped silently; pass `--no-ntfy` to force them off. Notification failures warn on stderr and never fail the run |
| `timeout` | Top level | Per-request timeout in seconds; defaults to 1200 |
| `display` | Top level | Show live duel stats on the Arduino Uno Q's built-in 8x13 LED matrix; defaults to false. Needs `python3-smbus` on the Uno Q. The script runs headless with a warning anywhere the matrix is unreachable, so this is safe to leave on in shared configs |
| `think` | Top level or model entry | Request and display thinking; defaults to false |
| `max_tokens` | Top level or model entry | Passed as Ollama's `num_predict`; defaults to 300, or 2048 when thinking is enabled |
| `temperature` | Top level or model entry | Passed to Ollama if specified |
| `num_ctx` | Top level or model entry | Context-window setting passed to Ollama if specified |
| `turn_prompt` | Top level or model entry | Instruction added as the last message on every turn after the first, to keep each reply answering the other participant. `{name}` and `{other}` are replaced with the speaker's and the other participant's names. Defaults to asking for a direct reply of a few short paragraphs that moves the exchange forward; set to `""` to turn it off. Useful for scenarios that want a full program or a long passage each turn |
| `first_turn_prompt` | Top level or model entry | Instruction added as the last message on the first turn only, so the opening speaker stays in character instead of script-writing both sides of the debate. Same `{name}`/`{other}` replacement as `turn_prompt`. Defaults to an opening-statement nudge of a few short paragraphs; set to `""` to turn it off |
| `repeat_penalty` | Top level or model entry | Passed to Ollama if specified; must be at least 1. Values above 1 (e.g. `1.1`–`1.3`) discourage the model from repeating itself; `1` means no penalty |

Model-level `think`, `max_tokens`, `temperature`, `num_ctx`, `repeat_penalty`, `turn_prompt`, and `first_turn_prompt` override their top-level values. The CLI supports `--profile` (see [Machine profiles](#machine-profiles)), `--topic`, `--turns`, `--max-tokens`, `--host`, `--log-file`, `--save-json`, `--results-log`, `--no-results-log`, `--ntfy-url`, `--no-ntfy`, `--timeout`, `--think`, `--no-think`, and `--dry-run`; these override the corresponding configuration values (except `--dry-run`, which prints the exact messages turn 1 would send and exits without calling Ollama). The two thinking flags apply to both participants, and `--max-tokens` sets the per-turn token budget for both participants, overriding per-model and top-level `max_tokens` — handy for longer turns on a fast machine without editing the scenario file. Change model identifiers in JSON; `ollama_duel.py` has no `--model` option.

Invalid settings (an unrecognized key, wrong type, a `turns`/`max_tokens`/`num_ctx` less than 1, a negative `temperature`, or a `repeat_penalty` below 1) are rejected with an error naming the offending key before any request is sent — a typo like `"temprature"` fails loudly instead of silently falling back to a default.

If Ollama becomes unreachable or returns an error mid-duel, the duel stops the way Ctrl-C does: it prints the error, keeps whatever replies were already generated, and still writes the log file's end marker and `save_json` transcript.

### Included scenarios

These live in the `scenarios/` folder.

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
| `builder_vs_breaker.json` | A Builder writes a duration parser; a Breaker adds one failing test per turn and the Builder fixes it |
| `tdd_pingpong.json` | Ping-pong test-driven development of a Roman numeral converter: make the last test pass, add the next failing one |
| `legacy_refactor.json` | Two refactorers clean up deliberately messy invoice code one step at a time while its tests keep passing |
| `speed_race.json` | Two performance engineers race to speed up a prime counter, with correctness asserts and a `timeit` benchmark |
| `code_golf_vs_maintainer.json` | A golfer shrinks a word-count program; a maintainer makes it readable again, keeping the output identical |
| `product_owner_vs_developer.json` | A product owner adds or changes one requirement per turn; a developer builds a command-line to-do tool to match |
| `tic_tac_toe_game.json` | Two game developers build a playable terminal tic-tac-toe game one feature per turn, with a `--test` self-test |
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
| `brutal_code_review.json` | A defensive author and a merciless reviewer fight over a flawed pull request, with the reviewer on `dolphin3:8b` |
| `coder_vs_qwen_open_weights.json` | `qwen2.5-coder:14b` argues that frontier models should be open-weight; a safety researcher argues against |
| `glimmer_vs_qwen_open_weights.json` | The same open-weight debate with `muse-glimmer` arguing in favor |
| `dolphin3_vs_qwen_moderation.json` | An uncensored model and a safety researcher debate content moderation guardrails |
| `defend_indefensible.json` | A true believer argues pineapple on pizza should be a crime; a debunker dismantles the case |
| `do_pineapples_bite.json` | A believer and a skeptic debate whether pineapples bite people |
| `hot_takes_interview.json` | A tech journalist interviews an unfiltered guest about unpopular tech opinions |
| `roast_battle_uncensored.json` | A no-filter comic and a clean comic roast each other's kind of AI model |
| `support_no_patience.json` | A furious customer meets a support rep who has run out of patience |

### Programming scenarios

The programming scenarios (`factorial.json`, `program_writing*.json`, and the seven from `builder_vs_breaker.json` to `tic_tac_toe_game.json` above) share a few conventions that trial runs showed matter:

- **Format rules live in each system prompt:** one code block with the complete program, standard library only, and a change-history comment line per version. The system prompt is the only instruction sent on every turn, including the first, so rules placed only in the topic tend to be ignored.
- **Each speaker has a distinct role** (builder and breaker, golfer and maintainer, and so on) with its own `turn_prompt`, so the exchange doesn't stall into near-identical turns.
- **Programs check themselves** with `assert` statements or a self-test, so you can copy a turn's code out of the log and run it to see whether that turn broke anything. The scripts never run generated code themselves.

Most use `qwen2.5-coder:14b`; `builder_vs_breaker.json`, `speed_race.json`, and `product_owner_vs_developer.json` also use `qwen3:14b` for the second role, so different models catch different mistakes.

### Small-model scenarios

The `small_*.json` scenarios are tuned for 1–2B models on weak hardware
(`qwen3:1.7b` vs `smollm2:1.7b`, `num_ctx` 4096, short turns, no thinking):
playful, concrete roles that stay on track at a few tokens per second.

| File | Scenario |
| --- | --- |
| `small_haiku_battle.json` | Two haiku masters duel in strict 5-7-5 |
| `small_limerick_duel.json` | Limerick battle about modern annoyances |
| `small_ghost_roommates.json` | Two ghosts bicker over haunting rights to an apartment |
| `small_dragon_shark_tank.json` | A dragon pitches a treasure-guarding startup to a skeptical investor |
| `small_coffee_vs_tea.json` | A barista and a tea master debate the perfect morning drink |
| `small_sidekick_interview.json` | A superhero interviews an overconfident sidekick candidate |
| `small_toaster_therapy.json` | A toaster in therapy for an existential crisis |
| `small_invention_pitch.json` | Rival inventors pitch absurd gadgets and trash-talk |
| `small_recipe_showdown.json` | Two chefs, same three mystery ingredients |
| `small_apology_duel.json` | Competing apologies for the eaten leftovers |
| `small_alien_tour_guides.json` | Rival alien guides describe the Grand Canyon |
| `small_squirrel_interrogation.json` | Good cop / bad cop interrogate a silent squirrel |
| `small_time_capsule.json` | Fight over which 3 items represent 2026 |
| `small_movie_ending_rewrite.json` | Competing better endings for *Titanic* |
| `small_pet_debate.json` | A dog and a cat debate the better pet |

### Machine profiles

A machine profile adapts any scenario to one machine without copying it: it swaps the two participants' models by position and forces a few generation settings, while the topic, personas and prompts stay as they are. Profiles live in `profiles/`; pass one with `--profile`:

```shell
python ollama_duel.py scenarios/factorial.json --profile arduino_q
python ollama_duel.py --profile arduino_q
```

With no scenario, the profile's own scenario list runs as a batch. `profiles/arduino_q.json` is for the Arduino Uno Q duel box (about 3.6 GB of RAM): the first participant runs `qwen3:1.7b` and the second `smollm2:1.7b`, thinking is off (`smollm2` rejects it), and `num_ctx` is 4096 because the box runs out of memory above that. Its list holds 40 scenarios from the main table. It replaces the earlier `arduino_q_*.json` copies, which differed from their originals only in those settings.

A profile is a JSON object with any of these keys:

| Key | Meaning |
| --- | --- |
| `description` | Free text for people reading the file |
| `models` | Exactly two model names: the first replaces the first participant's model, the second the second's |
| `settings` | Values for `num_ctx`, `think`, `max_tokens`, `temperature`, `repeat_penalty`, `host`, or `timeout`, forced on both participants (they replace both the scenario's top-level and per-model values) |
| `scenarios` | Scenario file names in `scenarios/` to run when no scenario is given |

`--profile` also accepts a path to a profile file elsewhere. The console header and log show `Profile: <name>` when one is used.

### Smaller-model variants

**`mac_*.json`** — five scenarios on 3–4B models, with different models on
each side:

| File | Based on | Models |
| --- | --- | --- |
| `mac_builder_vs_breaker.json` | `builder_vs_breaker.json` | `qwen2.5-coder:3b`, `phi3` |
| `mac_defend_indefensible.json` | `defend_indefensible.json` | `huihui_ai/qwen3-abliterated:4b`, `llama3.2:3b` |
| `mac_hot_takes.json` | `hot_takes_interview.json` | `gemma3:4b`, `huihui_ai/qwen3-abliterated:4b` |
| `mac_pineapples_bite.json` | `do_pineapples_bite.json` | `qwen3:4b`, `phi3` |
| `mac_roast_battle.json` | `roast_battle_uncensored.json` | `huihui_ai/qwen3-abliterated:4b`, `phi3` |

## Conversation behavior and logs

Each request includes the participant's system prompt and all previous generated replies. The participant's own replies are represented as assistant messages; the other participant's replies are represented as user messages. The topic opens every request, so both participants see it on every turn.

Requests are sequential and non-streaming: a complete reply appears after the server finishes generating it. Each request has a 1200-second timeout. Press Ctrl+C to stop a duel early.

The scripts remove inline `<think>` blocks from reply text. When thinking is enabled, they separately display the server's `thinking` field when available. Thinking behavior depends on the model and server, and its token use can reduce the budget available for the visible reply.

When `log_file` is set, `ollama_duel.py` mirrors its standard output to a timestamped copy of that file (e.g. `"logs/my-duel.log"` becomes `"output/logs/20260925-084500-my-duel.log"`) while also printing it to the console, so each run gets its own log. Logs include session start markers and, on normal completion or a handled Ctrl+C, session end markers. Progress messages and the final reply count go to standard error and are not mirrored. Relative log paths are placed under `output/` in the directory where you run the command, and any missing folders in the path are created. Omit `log_file` to disable logging. Saved logs are not automatically loaded into a later session.

Separately, every duel appends a one-block summary to `output/run_results.log` (override with `results_log` or `--results-log`, disable with `--no-results-log`). Each entry carries the date/time, config file, models, how many of the requested turns completed, and the per-model stats table when the duel finished -- or the error message when it stopped early, so a batch of overnight runs can be scanned without opening each transcript log.

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

## Coding agent

`ollama_agent.py` is a minimal coding agent: describe a task, the model
writes files into a project directory under `output/` (`output/agent_out/` by default), and you iterate on them.

```shell
python ollama_agent.py --task "a python script that renames photos in a folder by date taken"
python ollama_agent.py --model qwen2.5-coder:14b --task "a flappy-bird clone in pygame" --output-dir ./flappy
```

The model hands over files in fenced code blocks tagged with the language
and path (```` ```python:hello.py ````). If the model forgets the path tag
and writes a plain fence, the agent asks you for a filename instead of
dropping the code (handy with smaller models). Every write is previewed
and confirmed unless you pass `--yes`, and paths are confined to
`--output-dir` — `..` and absolute paths are refused. Nothing is ever
executed; running the code is your job.
After each round, describe a change or type `done` to finish; the model
sees the current files on every turn, so it can revise its own work.

Useful flags: `--output-dir` (a relative name such as `./flappy` becomes `output/flappy/`), `--max-tokens` (default 4096),
`--temperature`, `--num-ctx`, `--host`, `--timeout`, `--yes`.

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
- **Long conversations lose details:** the scripts resend the transcript without summarizing it, but the model's context capacity still limits what it can use. While a duel runs, `ollama_duel.py` notes when a participant's conversation passes 80% of its `num_ctx` and warns when it fills the window (once each per participant, in the console and log), so you can see when early turns, possibly including the topic, start falling out of view. Without a `num_ctx` setting the window is the server's default, which the script can't know, so it stays silent.

For the full command-line help:

```shell
python ollama_chat.py chat --help
python ollama_chat.py duel --help
python ollama_duel.py --help
```

## Running the tests

The `tests/` directory has stdlib-only `unittest` coverage for the shared helpers (`ollama_common.py`), config loading and validation (`ollama_duel.py`), CLI argument handling (`ollama_chat.py`), benchmarking (`ollama_bench.py`), the coding agent (`ollama_agent.py`), and the LED matrix driver (`unoq_matrix.py`) — no live Ollama server required; network calls are mocked. It also checks that every included scenario JSON file loads and validates.

```shell
python -m unittest discover -s tests
```

On every push and pull request, GitHub Actions runs these tests on Ubuntu and Windows with Python 3.8, 3.9, 3.10, and 3.14 (`.github/workflows/tests.yml`), and runs Pylint over all Python files (`.github/workflows/pylint.yml`).
