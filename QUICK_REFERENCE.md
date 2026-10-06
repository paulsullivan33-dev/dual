# Quick reference

Copy-paste commands for every script in this repo, grouped by task. Find the
section you need, take the command, change the model or file names. For
background and design notes, see [README.md](README.md); for a guided tour
of the RAG demo, see [RAG_DEMO.md](RAG_DEMO.md).

**Contents**

- [Find a command by task](#find-a-command-by-task)
- [Before you start](#before-you-start)
- [Chat with one model — `ollama_chat.py chat`](#chat-with-one-model--ollama_chatpy-chat)
- [Quick two-model duel — `ollama_chat.py duel`](#quick-two-model-duel--ollama_chatpy-duel)
- [Scenario duels — `ollama_duel.py`](#scenario-duels--ollama_duelpy)
- [Coding agent — `ollama_agent.py`](#coding-agent--ollama_agentpy)
- [Ask your own documents — `rag_demo.py`](#ask-your-own-documents--rag_demopy)
- [Benchmark model speed — `ollama_bench.py`](#benchmark-model-speed--ollama_benchpy)
- [Support modules (not run directly)](#support-modules-not-run-directly)
- [Run the tests](#run-the-tests)
- [Troubleshooting](#troubleshooting)

---

## Find a command by task

| I want to… | Command |
| --- | --- |
| Chat with a model | `python ollama_chat.py chat --model qwen3:8b` |
| Chat with a persona | `python ollama_chat.py chat --system "You are a patient Python tutor."` |
| See a model's reasoning while chatting | `python ollama_chat.py chat --model qwen3:8b --think` |
| Make two models argue about something, no setup | `python ollama_chat.py duel --topic "Tabs or spaces?"` |
| Run a ready-made scenario | `python ollama_duel.py scenarios/vim_vs_emacs.json` |
| Run a scenario with a different topic | `python ollama_duel.py scenarios/duel-example.json --topic "Should we adopt Kubernetes?"` |
| Run a shorter test of a scenario | `python ollama_duel.py scenarios/factorial.json --turns 2` |
| See what a scenario sends, without running it | `python ollama_duel.py scenarios/factorial.json --dry-run` |
| Run every scenario in a folder | `python ollama_duel.py scenarios/` |
| Run a group of scenarios | `python ollama_duel.py "scenarios/small_*.json"` |
| Run a scenario on the Arduino box's models | `python ollama_duel.py scenarios/factorial.json --profile arduino_q` |
| Run the whole Arduino set | `python ollama_duel.py --profile arduino_q` |
| Run each reply's program to catch broken code | On already in the programming scenarios; elsewhere add `--run-code` |
| Skip running code for one run | `python ollama_duel.py scenarios/factorial.json --no-run-code` |
| Save a duel's transcript as JSON | `python ollama_duel.py scenarios/roast_battle.json --save-json roast.json` |
| Get a phone notification when a duel ends | `python ollama_duel.py scenarios/roast_battle.json --ntfy-url https://ntfy.sh/my-topic` |
| Have a model write a small project | `python ollama_agent.py --task "a CLI that renames photos by date taken"` |
| Choose where the agent writes files | `python ollama_agent.py --task "a sudoku solver in python" --output-dir sudoku` |
| Index a folder of notes for Q&A | `python rag_demo.py index --docs ./news` |
| Ask questions about indexed notes | `python rag_demo.py ask` |
| Compare model speeds | `python ollama_bench.py qwen3:4b qwen3:8b` |
| Run the tests | `python -m unittest discover -s tests` |

---

## Before you start

```shell
ollama serve
ollama list
ollama pull qwen3:4b
```

- **Ollama must be running.** Every script talks to an Ollama server at
  `http://localhost:11434` unless you pass `--host` (or set `host` in a
  scenario). `ollama serve` starts it if the desktop app isn't already
  running it.
- **Models must be pulled first.** Use the exact name from `ollama list`.
  An "HTTP 404 … model not found" error means it isn't pulled on that server.
- **No install step.** Everything is standard-library Python 3.8+. Use
  `python`, `python3` or `py`, whichever your system has.
- **Generated files go in `output/`.** Logs, run summaries, saved
  transcripts, agent projects and the RAG index all land under `output/`
  (git-ignored). A *relative* path you give for any of these is placed
  under `output/`; an *absolute* path is used exactly as given.

  | Path | What's there |
  | --- | --- |
  | `output/logs/` | Duel transcript logs, one timestamped file per run |
  | `output/run_results.log` | One summary block per duel run |
  | `output/agent_out/`, `output/<name>/` | Projects written by the agent |
  | `output/rag_demo.db` | The RAG demo's search index |

- **Shells.** Commands are written to work in PowerShell, Command Prompt
  and bash alike. Quote glob patterns (`"scenarios/small_*.json"`) so the
  script expands them, not the shell. Forward slashes work in paths on
  Windows too.

---

## Chat with one model — `ollama_chat.py chat`

An interactive conversation in the terminal. History is kept for the
session. Type `quit`, `exit` or `:q` (or press Ctrl+C) to leave; blank
lines are ignored.

### Common

```shell
python ollama_chat.py chat
python ollama_chat.py chat --model qwen3:8b
python ollama_chat.py chat --model qwen3:8b --system "You are a patient Python tutor. Explain with short examples."
```

### Less common

```shell
# Show the model's reasoning (Qwen3 and other thinking models)
python ollama_chat.py chat --model qwen3:8b --think

# Longer answers, more creative
python ollama_chat.py chat --model qwen3:8b --max-tokens 1500 --temperature 1.0

# Large context for long conversations
python ollama_chat.py chat --model qwen3:14b --num-ctx 16384 --max-tokens 2000

# Save the conversation when you quit (written to output/tutor-session.json)
python ollama_chat.py chat --system "You are a patient Python tutor." --save-json tutor-session.json

# Use Ollama on another machine, and give a slow box more time per reply
python ollama_chat.py chat --host http://192.168.1.10:11434 --model qwen3:1.7b --timeout 3600
```

### Options

These options also work for `duel`, and can go before or after the
subcommand (`ollama_chat.py --think chat` is the same as
`ollama_chat.py chat --think`).

| Option | Default | What it does |
| --- | --- | --- |
| `--model` | `qwen3:4b` | Model to chat with (`chat` only) |
| `--system` | none | System prompt: the persona or ground rules for the model (`chat` only) |
| `--think` | off | Ask for and print the model's reasoning separately from its reply. Thinking uses the same token budget as the reply. Some models reject it (`qwen2.5-coder`, `smollm2`) |
| `--max-tokens` | 300, or 2048 with `--think` | Most tokens one reply may use; replies stop when they hit it |
| `--temperature` | server default | Randomness: lower is more focused, higher is more varied (around 0.2–1.2 is typical) |
| `--num-ctx` | server default | Context window in tokens: how much conversation the model can see at once |
| `--timeout` | 1200 | Seconds to wait for one reply before giving up |
| `--host` | `http://localhost:11434` | Ollama server address |
| `--save-json` | off | Write the conversation to this JSON file when you quit (relative paths go under `output/`) |

---

## Quick two-model duel — `ollama_chat.py duel`

Two personas take turns, configured entirely on the command line. Good for
quick experiments; for repeatable runs with logs and per-speaker settings,
use [`ollama_duel.py`](#scenario-duels--ollama_duelpy).

### Common

```shell
python ollama_chat.py duel --topic "Tabs or spaces?"
python ollama_chat.py duel --topic "Should we rewrite it in Rust?" --turns 8 --name-a Optimist --system-a "You are relentlessly optimistic." --name-b Skeptic --system-b "You question every assumption and its cost."
```

### Less common

```shell
# Two different models
python ollama_chat.py duel --topic "Is remote work here to stay?" --model-a qwen3:8b --model-b qwen3:4b

# Show both speakers' reasoning, with room for it
python ollama_chat.py duel --topic "Is P equal to NP?" --model-a qwen3:8b --model-b qwen3:8b --think --max-tokens 3000

# Keep the transcript (written to output/rust-debate.json)
python ollama_chat.py duel --topic "Should we rewrite it in Rust?" --save-json rust-debate.json
```

### Options

All the [chat options](#options) except `--model` and `--system`, plus:

| Option | Default | What it does |
| --- | --- | --- |
| `--topic` | required | The opening prompt; both speakers see it on every turn |
| `--turns` | 6 | Total replies across both speakers (8 = four each) |
| `--model-a`, `--model-b` | `qwen3:4b` | Model for each speaker; A speaks first |
| `--name-a`, `--name-b` | `AI-A`, `AI-B` | Display names, also used in the per-turn "reply to …" nudge |
| `--system-a`, `--system-b` | none | Persona for each speaker |

`--temperature`, `--num-ctx` and `--max-tokens` apply to both speakers. For
different settings per speaker, use a scenario file with `ollama_duel.py`.

---

## Scenario duels — `ollama_duel.py`

Runs a two-speaker conversation described in a JSON scenario file (69 are
included in `scenarios/`). Adds per-speaker settings, transcript logs, run
summaries, batch runs, machine profiles, notifications and a dry-run mode.

### Run a scenario

```shell
python ollama_duel.py scenarios/duel-example.json
python ollama_duel.py scenarios/vim_vs_emacs.json
python ollama_duel.py scenarios/factorial.json
```

Before running one, check its `"model"` entries and pull those models (or
use a [profile](#machine-profiles) to swap them).

### Override settings for one run

```shell
# Different topic, same personas
python ollama_duel.py scenarios/duel-example.json --topic "Should a small team adopt AI coding tools?"

# Quick 2-turn smoke test
python ollama_duel.py scenarios/tic_tac_toe_game.json --turns 2

# Longer replies for both speakers (overrides every max_tokens in the file)
python ollama_duel.py scenarios/legacy_refactor.json --max-tokens 6000

# Thinking on or off for both speakers
python ollama_duel.py scenarios/trolley_problem_ethics.json --think
python ollama_duel.py scenarios/program_writing.json --no-think

# Another Ollama server, with more patience
python ollama_duel.py scenarios/roast_battle.json --host http://192.168.1.10:11434 --timeout 3600
```

### Check a scenario without running it

```shell
python ollama_duel.py scenarios/factorial.json --dry-run
python ollama_duel.py scenarios/factorial.json --profile arduino_q --dry-run
```

`--dry-run` prints the header and the exact messages turn 1 would send, then
exits: no Ollama call, no log, no summary, no notification. It's also how you
check a new scenario file for mistakes; invalid settings are reported before
anything else happens.

### Run many scenarios (batch mode)

```shell
# Every scenario in a folder
python ollama_duel.py scenarios/

# A group by name pattern (keep the quotes)
python ollama_duel.py "scenarios/small_*.json"
python ollama_duel.py "scenarios/mac_*.json"

# Any other options apply to every scenario in the batch
python ollama_duel.py "scenarios/small_*.json" --turns 4 --no-ntfy
```

Each scenario runs as its own process with its own log, summary entry and
notification. A failing scenario is reported and skipped; Ctrl+C stops the
whole batch.

### Machine profiles

A profile adapts any scenario to one machine: it swaps both speakers'
models (first speaker, second speaker) and forces settings, keeping the
topic and personas.

```shell
# One scenario on the Arduino Uno Q's small models
python ollama_duel.py scenarios/factorial.json --profile arduino_q

# The profile's own list of 40 scenarios, as a batch
python ollama_duel.py --profile arduino_q

# A whole folder or group through a profile
python ollama_duel.py "scenarios/small_*.json" --profile arduino_q

# A profile file stored somewhere else
python ollama_duel.py scenarios/factorial.json --profile C:/profiles/laptop.json
```

`profiles/arduino_q.json` uses `qwen3:1.7b` and `smollm2:1.7b`, a 4096-token
context, thinking off, and `history_turns` 4 (only the last four replies are
sent each turn). To make your own, add `profiles/<name>.json`:

```json
{
  "description": "Work laptop: 8B models, 8k context",
  "models": ["qwen3:8b", "llama3.1:8b"],
  "settings": {"num_ctx": 8192, "think": false},
  "scenarios": ["duel-example.json", "vim_vs_emacs.json"]
}
```

Every key is optional. `settings` may set `num_ctx`, `think`, `max_tokens`,
`temperature`, `repeat_penalty`, `history_turns`, `host` or `timeout`; these override the
scenario's values for both speakers. `scenarios` lists files in
`scenarios/` to run when you pass `--profile` without a scenario.

### Run each reply's code

The programming scenarios (`factorial`, `program_writing*`,
`builder_vs_breaker`, `mac_builder_vs_breaker`, `tdd_pingpong`,
`legacy_refactor`, `speed_race`, `code_golf_vs_maintainer`,
`product_owner_vs_developer`, `tic_tac_toe_game`) already have it on.

```shell
# Programming scenarios run each reply's code automatically
python ollama_duel.py scenarios/factorial.json

# Skip it for one run, e.g. on a slow machine
python ollama_duel.py scenarios/speed_race.json --no-run-code

# Turn it on for any other scenario
python ollama_duel.py scenarios/socratic_debugging.json --run-code

# Turn it on in your own scenario file instead, with arguments for each run
# (e.g. a game's self-test mode) and a time limit:
#   "run_code": true, "run_code_args": ["--test"], "run_code_timeout": 20
python ollama_duel.py my_game_duel.json
```

Interactive programs wait for keyboard input and time out, so give them a
test mode and pass it with `run_code_args` (the tic-tac-toe scenario runs
`--test` this way).

After each reply, its last Python code block is run in a fresh temporary
folder with a time limit (default 30s) and no keyboard input. The result
(exit code and the end of the output) goes in the log, and both speakers
see it on their next turns, so a broken program gets fixed. The log ends
with a tally, `Code runs: 6 ok, 2 failed, 0 timed out, …`, which also goes
into the `run_results.log` entry and the ntfy notice.

Off unless the scenario sets `"run_code": true` or you pass `--run-code`;
profiles can't turn it on. **Not a sandbox:** the code runs with your
permissions and can touch files and the network, so only use it for code
you're comfortable running.

### Logs, summaries and transcripts

```shell
# Log to a file of your choice (becomes output/logs/<timestamp>-negotiation.log)
python ollama_duel.py scenarios/salesperson_vs_customer.json --log-file logs/negotiation.log

# Save the structured transcript (speaker, model, text per reply)
python ollama_duel.py scenarios/roast_battle.json --save-json roast.json

# Keep run summaries in a separate file, or don't write one
python ollama_duel.py scenarios/roast_battle.json --results-log experiments.log
python ollama_duel.py scenarios/roast_battle.json --no-results-log
```

- **Transcript log**: everything printed during the duel, mirrored to a
  timestamped file. The scenario's `log_file` sets it; included scenarios
  log to `output/logs/`.
- **Run summary**: after every run, one block is appended to
  `output/run_results.log` with date, scenario, the full path of that run's
  transcript log, models, turns completed,
  and either the speed table or the error that stopped it. Handy for
  scanning an overnight batch.
- **Transcript JSON**: written when the duel ends, even after an early stop.

### Notifications (ntfy)

```shell
python ollama_duel.py scenarios/roast_battle.json --ntfy-url https://ntfy.sh/my-duel-topic
python ollama_duel.py scenarios/roast_battle.json --no-ntfy
```

To get a notice after every duel without typing the URL, create
`~/.dual.conf` (on Windows, `C:\Users\<you>\.dual.conf`):

```json
{"ntfy_url": "https://ntfy.sh/my-duel-topic"}
```

The notice says whether the duel finished, stopped early or crashed, with
models, turns, token speeds and the transcript log's path. `--ntfy-url` overrides the scenario's
`ntfy_url`, which overrides `~/.dual.conf`; `--no-ntfy` turns it off.

### Arduino Uno Q LED matrix

```shell
python ollama_duel.py --profile arduino_q --display
python ollama_duel.py scenarios/roast_battle.json --no-display
```

`--display` shows progress and tokens per second on the Uno Q's built-in
8×13 LED matrix (needs `python3-smbus` on the board). On any other machine
it prints a warning and carries on without the display.

### Options

| Option | What it does |
| --- | --- |
| `config` | A scenario file, a folder (runs every `.json` in it), or a quoted glob pattern. May be left out when `--profile` lists scenarios |
| `--profile` | Machine profile: a name in `profiles/` or a path to a profile file |
| `--topic` | Replace the scenario's topic |
| `--turns` | Replace the number of replies |
| `--max-tokens` | Replace every `max_tokens` in the scenario, for both speakers |
| `--think` / `--no-think` | Force thinking on or off for both speakers |
| `--host` | Ollama server address |
| `--timeout` | Seconds to wait for one reply (default 1200) |
| `--log-file` | Transcript log path (timestamp added; relative paths under `output/`) |
| `--save-json` | Write the transcript as JSON when the duel ends |
| `--results-log` | Append the run summary to this file instead of `output/run_results.log` |
| `--no-results-log` | Don't write a run summary |
| `--ntfy-url` | Send a completion notice to this ntfy topic URL |
| `--no-ntfy` | Never send a notice, even if one is configured |
| `--display` / `--no-display` | LED matrix on the Arduino Uno Q |
| `--run-code` / `--no-run-code` | Run (or never run) each reply's last Python block, showing the result to both speakers |
| `--dry-run` | Print turn 1's messages and exit without calling Ollama |

### Scenario file settings

A scenario is a JSON object with exactly two entries in `models`. Settings
marked *both* can go at the top level (for both speakers) or inside a model
entry (that speaker only, overriding the top level).

```json
{
  "topic": "Should our team adopt a four-day workweek?",
  "turns": 6,
  "max_tokens": 600,
  "num_ctx": 8192,
  "temperature": 0.7,
  "log_file": "logs/four-day-week.log",
  "models": [
    {"model": "qwen3:8b", "name": "Advocate",
     "system": "Argue for a four-day workweek with concrete examples."},
    {"model": "qwen3:4b", "name": "Skeptic",
     "system": "Question the costs and operational risks.", "temperature": 0.4}
  ]
}
```

| Setting | Where | Default | What it does |
| --- | --- | --- | --- |
| `topic` | top | required | Opening prompt, sent to both speakers on every turn |
| `turns` | top | 6 | Total replies |
| `model` | model entry | required | Ollama model name |
| `name` | model entry | the model name | Display name, used in the nudges |
| `system` | model entry | none | The speaker's persona and rules |
| `think` | both | false | Request and show reasoning |
| `max_tokens` | both | 300, or 2048 with thinking | Most tokens per reply |
| `temperature` | both | server default | Randomness |
| `num_ctx` | both | server default | Context window in tokens |
| `repeat_penalty` | both | server default | Above 1 (try 1.1–1.3) discourages repetition |
| `history_turns` | both | all | Send only the last N replies (plus topic and nudge), so long duels fit a small `num_ctx`. Older turns are forgotten |
| `first_turn_prompt` | both | an opening-statement nudge | Extra instruction on the first turn only. `{name}` and `{other}` are filled in; `""` turns it off |
| `turn_prompt` | both | a "reply directly to {other}" nudge | Extra instruction on every later turn. Same placeholders; `""` turns it off |
| `dedup_guard` | top | true | If a speaker repeats its own previous reply, re-roll that turn once with a no-repeat nudge and slightly higher temperature |
| `host` | top | `http://localhost:11434` | Ollama server |
| `timeout` | top | 1200 | Seconds per reply |
| `log_file` | top | none | Transcript log path |
| `save_json` | top | none | Transcript JSON path |
| `results_log` | top | `run_results.log` | Run summary path |
| `ntfy_url` | top | none | Completion notice URL |
| `display` | top | false | LED matrix on the Uno Q |
| `run_code` | top | false | Run each reply's last Python block and show the result to both speakers (not a sandbox) |
| `run_code_args` | top | none | Arguments for each run, e.g. `["--test"]` |
| `run_code_timeout` | top | 30 | Seconds before a run is stopped |

Unknown keys, wrong types and out-of-range values are rejected before any
model is called, so a typo like `"temprature"` fails loudly.

### Warnings you may see during a duel

- **"reply hit the max_tokens ceiling … truncated"**: raise `max_tokens`.
- **"context window is N% full" / "is full"**: the resent conversation is
  outgrowing `num_ctx`; raise `num_ctx`, use fewer turns, or set
  `history_turns` to send only recent replies. (Shown only when `num_ctx`
  is set.)
- **"max_tokens … is larger than num_ctx"** (at startup): a reply can't use
  more tokens than the window holds; lower one or raise the other.
- **"CODE RUN … FAILED" / "was stopped after Ns"** (with `run_code`): the
  reply's program crashed or didn't finish; the speakers see the same report.
  A program waiting for keyboard input times out, so use `run_code_args`
  for a test mode instead.
- **"DEDUP GUARD: … repeated its previous reply"**: the guard re-rolled a
  repeated turn (see `dedup_guard`).

---

## Coding agent — `ollama_agent.py`

Describe a program; the model writes the files, you approve each write,
then describe changes and it revises. It never runs the code; that's up to
you.

### Common

```shell
python ollama_agent.py --task "a python script that renames photos in a folder by date taken"
python ollama_agent.py --task "build a sudoku solver in python" --output-dir sudoku
python ollama_agent.py --model qwen2.5-coder:14b --task "a flask todo app with sqlite" --output-dir todo_app
```

Files go to `output/agent_out/` by default, or `output/<name>/` with
`--output-dir <name>`. Running it again with the same `--output-dir`
continues that project: the model sees the existing files.

### Less common

```shell
# Small model on a modest machine
python ollama_agent.py --model phi3 --task "build a sudoku solver in python" --output-dir sudoku3

# Write files without asking each time (the project folder is still enforced)
python ollama_agent.py --task "a markdown-to-html converter" --output-dir md2html --yes

# Bigger replies and context for multi-file projects
python ollama_agent.py --task "a pygame breakout clone with levels" --output-dir breakout --max-tokens 8000 --num-ctx 16384

# More deterministic code
python ollama_agent.py --task "a CSV deduplication tool" --temperature 0.2

# Write the project somewhere outside the repo (absolute paths are used as-is)
python ollama_agent.py --task "a password generator CLI" --output-dir C:/Projects/passgen
```

### During a session

- Before writing, it lists the files and asks **Write N file(s)? [y/N]**.
- If the model forgets to name a code block, it suggests a filename; press
  Enter to accept, type another name, or type `skip`.
- After each round, describe a change ("add a --dry-run flag") or type
  `done` (also `quit`, `exit`, `q`) to finish.

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `--task` | required | What to build, in plain words |
| `--model` | `qwen2.5-coder:14b` | Model that writes the code |
| `--output-dir` | `agent_out` | Project folder; relative names go under `output/` |
| `--yes` | off | Write files without asking for confirmation |
| `--max-tokens` | 4096 | Most tokens per reply; raise for multi-file projects |
| `--temperature` | server default | Lower for more predictable code |
| `--num-ctx` | server default | Context window; raise when the project grows |
| `--timeout` | 1200 | Seconds to wait for one reply |
| `--host` | `http://localhost:11434` | Ollama server |

The model can only write inside the project folder: file names with `..`
or absolute paths in its replies are refused.

---

## Ask your own documents — `rag_demo.py`

Indexes a folder of `.txt`/`.md` files, then answers questions using the
most relevant passages. Needs an embedding model and a chat model:

```shell
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

### Common

```shell
# 1. Index a folder (searches subfolders too; replaces any previous index)
python rag_demo.py index --docs ./news

# 2. Ask interactively (blank line quits)
python rag_demo.py ask

# Or ask one question and exit
python rag_demo.py ask --question "What happened with the flight into Israel?"
```

### Less common

```shell
# Answer with a different chat model, using more passages
python rag_demo.py ask --chat-model qwen3:14b --top-k 6 --question "Summarize this week's news"

# Answer only, without printing the retrieved passages
python rag_demo.py ask --hide-chunks --question "Who is mentioned most?"

# Smaller chunks for short notes, larger for long documents
python rag_demo.py index --docs ./notes --chunk-words 120 --overlap-words 30
python rag_demo.py index --docs ./manuals --chunk-words 400 --overlap-words 80

# Keep separate indexes for separate collections (note: --db goes BEFORE index/ask)
python rag_demo.py --db news.db index --docs ./news
python rag_demo.py --db news.db ask

# A different embedding model, or another Ollama server
python rag_demo.py --embed-model mxbai-embed-large index --docs ./news
python rag_demo.py --host http://192.168.1.10:11434 ask
```

**`--host`, `--db` and `--embed-model` must come before `index` or `ask`.**
Placed after, they're rejected as unrecognized. If you index with one
embedding model, ask with the same one; the script warns if they differ.

### Options

| Option | Goes | Default | What it does |
| --- | --- | --- | --- |
| `--host` | before the subcommand | `http://localhost:11434` | Ollama server |
| `--db` | before the subcommand | `rag_demo.db` | Index file; relative paths go under `output/` |
| `--embed-model` | before the subcommand | `nomic-embed-text` | Model that turns text into vectors for searching |
| `--docs` | `index` | required | Folder of `.txt`/`.md`/`.markdown` files |
| `--chunk-words` | `index` | 250 | Words per passage |
| `--overlap-words` | `index` | 50 | Words shared between neighbouring passages, so sentences aren't cut off |
| `--question` | `ask` | interactive | Ask once and exit |
| `--chat-model` | `ask` | `qwen3:8b` | Model that writes the answer |
| `--top-k` | `ask` | 3 | How many passages to give the model |
| `--hide-chunks` | `ask` | show | Don't print the retrieved passages |

---

## Benchmark model speed — `ollama_bench.py`

Measures prompt-reading and generation speed (tokens per second) using
Ollama's own timings. Thinking is forced off so models compare fairly.

### Common

```shell
python ollama_bench.py qwen3:8b
python ollama_bench.py qwen3:4b qwen3:8b qwen2.5-coder:14b
```

With several models it ends with a comparison table, fastest first.

### Less common

```shell
# More runs for steadier numbers
python ollama_bench.py qwen3:8b qwen3:14b -n 10

# Test a long-answer workload with a realistic context size
python ollama_bench.py qwen3:8b --max-tokens 1024 --num-ctx 8192

# Your own prompt
python ollama_bench.py qwen3:8b --prompt "Write a Python function that parses ISO 8601 dates, with tests."

# Benchmark the Arduino box remotely; skip warmup if the model is already loaded
python ollama_bench.py qwen3:1.7b smollm2:1.7b --host http://192.168.1.50:11434 --no-warmup --timeout 3600
```

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `models` | required | One or more model names |
| `--iterations`, `-n` | 3 | Timed runs per model, averaged |
| `--no-warmup` | warmup on | Skip the uncounted first run that loads the model into memory |
| `--prompt` | a short history-of-the-transistor request | Text to generate from |
| `--max-tokens` | 256 | Tokens to generate per run |
| `--num-ctx` | server default | Context window |
| `--timeout` | 1200 | Seconds per request |
| `--host` | `http://localhost:11434` | Ollama server |

---

## Support modules (not run directly)

| File | Role |
| --- | --- |
| `ollama_common.py` | Shared code: the Ollama API call, text wrapping, transcript saving, the `output/` path rule, setting validation |
| `ollama_profiles.py` | Loading and applying machine profiles for `ollama_duel.py --profile` |
| `ollama_coderun.py` | Running each reply's Python code for `ollama_duel.py --run-code` |
| `ollama_reporting.py` | End-of-duel reporting: the speed table, `run_results.log` entries and ntfy notices |
| `unoq_matrix.py` | Driver for the Arduino Uno Q's LED matrix, used by `ollama_duel.py --display` |

---

## Run the tests

```shell
python -m unittest discover -s tests
python -m unittest tests.test_ollama_duel
python -m unittest tests.test_ollama_duel.ProfileTests
python -m unittest discover -s tests -v
```

No Ollama server is needed; network calls are faked. GitHub Actions runs the
same suite on every push.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Cannot reach Ollama" | Start it with `ollama serve`, or check `--host` |
| "HTTP 404 … model not found" | `ollama pull <model>`, or use a profile/scenario with models you have |
| "HTTP 400" with thinking on | The model rejects thinking; use `--no-think` (duel) or drop `--think` |
| Replies cut off mid-sentence | Raise `--max-tokens` (or `max_tokens`); if `num_ctx` is small, raise it too |
| Speakers forget the topic in long duels | Raise `num_ctx`, lower `turns`, set `history_turns` (e.g. 4), or watch for the "context window" warnings |
| A speaker keeps repeating itself | Leave `dedup_guard` on and try `repeat_penalty` 1.1–1.3 |
| Very slow replies | Smaller models, fewer turns, lower `max_tokens`; raise `--timeout` on slow machines |
| `rag_demo.py`: "unrecognized arguments: --db" | Put `--db`/`--host`/`--embed-model` before `index`/`ask` |
| Glob runs only one file or errors in bash | Quote the pattern: `"scenarios/small_*.json"` |
