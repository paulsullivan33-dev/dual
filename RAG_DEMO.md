# RAG Demo Guide

`rag_demo.py` is a small, self-contained playground for
**retrieval-augmented generation (RAG)**: instead of hoping the model
memorized the facts, you look up the relevant pieces of your own documents
at question time and paste them into the prompt. The model then answers
from *your* documents instead of its training memory.

Everything is stdlib-only Python. Documents and their embeddings live in
a SQLite file, so you can open it and inspect what's actually stored.

## Prerequisites

- Ollama running locally (`ollama serve`, or the desktop app).
- An embedding model (turns text into numbers capturing its meaning):

  ```bash
  ollama pull nomic-embed-text
  ```

- A chat model for answers (default is `qwen3:8b`; anything Ollama
  serves works via `--chat-model`):

  ```bash
  ollama pull qwen3:8b
  ```

## Steps

**1. Gather some documents.** Put any `.txt` or `.md` files in a folder.
Your own notes are ideal — the more the model *doesn't* already know,
the clearer the demo.

**2. Index them.** This splits each file into overlapping chunks, embeds
every chunk with `nomic-embed-text`, and stores it all in `output/rag_demo.db`:

```bash
python rag_demo.py index --docs ./my-docs
```

Re-running `index` rebuilds the database from scratch. To add more
files later without re-embedding everything, use `--append`:

```bash
python rag_demo.py index --docs ./more-docs --append
```

It keeps the existing chunks and only (re-)embeds the files in
`--docs` — handy for dropping a second novel into the same index, or
picking up new notes. It refuses to mix embedding models, since their
vectors aren't comparable.

**3. Ask questions.**

```bash
python rag_demo.py ask
```

This is interactive — blank line quits. Or ask once and exit:

```bash
python rag_demo.py ask --question "what does my note say about ...?"
```

Before the answer, you'll see the chunks it retrieved, each with its
source file and a similarity score. That's the whole trick made visible:
those chunks are what the model actually gets to work with.

## Experiments worth trying

1. **Ask something that's not in your documents.** Watch it say the
   answer isn't in the context instead of guessing. Then ask the same
   thing *without* RAG (plain `ollama_chat.py`) and compare — that's the
   hallucination gap RAG closes.
2. **Change the chunk size.** Re-index with `--chunk-words 100` versus
   the default 250 and ask the same questions. Small chunks retrieve
   precisely but can lose surrounding meaning; large chunks keep context
   but dilute the match. There is no universally right value.
3. **Retrieve more chunks.** Try `--top-k 6` on a question whose answer
   spans multiple files, and watch the answer get more complete — then
   try it on a small-context setup and watch the window fill up.

## How it works

- **Chunking:** each file is split into ~250-word windows overlapping by
  50 words, so an idea straddling a boundary isn't cut in half.
- **Embedding:** each chunk becomes a list of ~768 numbers (its position
  in "meaning space"). Similar meanings land near each other.
- **Retrieval:** your question is embedded the same way, and the chunks
  pointing in nearly the same direction (cosine similarity) win.
- **Prompt:** the top chunks are pasted into the prompt under a system
  instruction to answer *only* from that context, then sent to the chat
  model.

## Knobs

| Flag | Where | Default | What it does |
| --- | --- | --- | --- |
| `--chunk-words` | `index` | 250 | target words per chunk |
| `--overlap-words` | `index` | 50 | words shared between adjacent chunks |
| `--top-k` | `ask` | 3 | how many chunks to retrieve |
| `--chat-model` | `ask` | qwen3:8b | model that writes the answer |
| `--embed-model` | both | nomic-embed-text | model that embeds text |
| `--hide-chunks` | `ask` | off | skip printing retrieved chunks |

## Gotchas

- Index with one embedding model and query with another and the numbers
  won't line up — the script warns you if the database was built with a
  different model than you're asking with.
- Embeddings capture *meaning*, not *truth*. A well-written wrong
  document retrieves just as confidently as a right one.
- The "answer only from the context" instruction helps a lot, but it's
  an instruction, not a guarantee — the model can still wander.
- On small-context setups (the 4096-token window used on the Pi/Arduino
  duel boxes), retrieved chunks eat the window fast: keep chunks small
  and `top-k` low.
