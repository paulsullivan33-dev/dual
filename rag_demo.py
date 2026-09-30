#!/usr/bin/env python3
"""rag_demo.py -- a tiny retrieval-augmented generation (RAG) playground.

Point it at a folder of your own .txt/.md files, ask questions, and watch
a local model answer from YOUR documents instead of its training memory.

Prerequisites: Ollama running locally, plus two models --
    ollama pull nomic-embed-text      (small model that turns text into numbers)
    ollama pull qwen3:8b              (or any chat model you prefer)

Two steps:
    python rag_demo.py index --docs ./my-docs
    python rag_demo.py ask                        (interactive; blank line quits)
    python rag_demo.py ask --question "what is ..."

Everything is stdlib-only: documents + embeddings live in a SQLite file
(rag_demo.db by default), so you can open it and poke at it yourself.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import urllib.request
import urllib.error

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_DB = "rag_demo.db"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context below. The context is a set "
    "of excerpts from the user's own documents, each labeled with its source "
    "file. If the answer is not in the context, say so plainly instead of "
    "guessing or using outside knowledge."
)


# ---------------------------------------------------------------- HTTP helpers

def ollama_post(host, path, payload, timeout=600):
    """POST JSON to Ollama; exit with a plain-English error if it's down."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        host + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        sys.exit(f"Can't reach Ollama at {host} -- is it running? ({exc})")


def embed_text(host, model, text):
    body = ollama_post(host, "/api/embeddings",
                       {"model": model, "prompt": text})
    return body["embedding"]


def chat(host, model, system, user_text):
    body = ollama_post(host, "/api/chat", {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    })
    return body["message"]["content"]


# ---------------------------------------------------------------- chunking

def chunk_text(text, chunk_words=250, overlap_words=50):
    """Split text into word windows with overlap.

    Overlap matters: without it, an idea that straddles a chunk boundary
    gets cut in half and neither chunk makes sense alone. Try different
    --chunk-words / --overlap-words values and re-index to feel the effect.
    """
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------- database

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    source    TEXT,     -- file the chunk came from
    idx       INTEGER,  -- chunk number within that file
    text      TEXT,     -- the chunk itself
    embedding TEXT      -- JSON list of floats from the embedding model
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def open_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def cosine(a, b):
    """Cosine similarity: 1.0 = same direction (similar meaning)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------- index

def cmd_index(args):
    docs = os.path.abspath(args.docs)
    if not os.path.isdir(docs):
        sys.exit(f"No such folder: {docs}")

    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(docs)
        for f in names
        if f.lower().endswith((".txt", ".md", ".markdown"))
    )
    if not files:
        sys.exit(f"No .txt/.md files found under {docs}")
    print(f"Found {len(files)} files. Embedding with {args.embed_model} ...")

    conn = open_db(args.db)
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM meta")
    total_chunks = 0
    for n, path in enumerate(files, 1):
        text = read_text_file(path)
        chunks = chunk_text(text, args.chunk_words, args.overlap_words)
        rel = os.path.relpath(path, docs)
        rows = []
        for i, c in enumerate(chunks):
            emb = embed_text(args.host, args.embed_model, c)
            rows.append((rel, i, c, json.dumps(emb)))
            total_chunks += 1
        conn.executemany(
            "INSERT INTO chunks (source, idx, text, embedding) VALUES (?,?,?,?)",
            rows)
        conn.commit()
        if n % 5 == 0 or n == len(files):
            print(f"  {n}/{len(files)} files, {total_chunks} chunks so far")
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
        ("embed_model", args.embed_model),
        ("chunk_words", str(args.chunk_words)),
        ("overlap_words", str(args.overlap_words)),
    ])
    conn.commit()
    conn.close()
    print(f"Done: {total_chunks} chunks from {len(files)} files -> {args.db}")


# ---------------------------------------------------------------- ask

def load_chunks(conn):
    cur = conn.execute("SELECT source, idx, text, embedding FROM chunks")
    return [(src, i, txt, json.loads(emb)) for src, i, txt, emb in cur]


def retrieve(question_emb, chunks, top_k):
    ranked = sorted(chunks,
                    key=lambda c: cosine(question_emb, c[3]),
                    reverse=True)
    return ranked[:top_k]


def build_prompt(question, hits):
    parts = []
    for n, (src, i, txt, _emb) in enumerate(hits, 1):
        parts.append(f"[{n}] (source: {src}, chunk {i})\n{txt}")
    return (f"Context:\n{chr(10).join(parts)}\n\n"
            f"Question: {question}\n\nAnswer:")


def answer_one(args, chunks):
    q_emb = embed_text(args.host, args.embed_model, args.question)
    hits = retrieve(q_emb, chunks, args.top_k)
    if args.show_chunks:
        print("\n--- retrieved chunks ---")
        for n, (src, i, _txt, emb) in enumerate(hits, 1):
            print(f"[{n}] {src} chunk {i}  (similarity {cosine(q_emb, emb):.3f})")
        print("------------------------\n")
    prompt = build_prompt(args.question, hits)
    print(chat(args.host, args.chat_model, SYSTEM_PROMPT, prompt))


def cmd_ask(args):
    if not os.path.exists(args.db):
        sys.exit(f"No database at {args.db} -- run the index step first.")
    conn = open_db(args.db)
    chunks = load_chunks(conn)
    conn.close()
    if not chunks:
        sys.exit("Database is empty -- run the index step first.")
    row = sqlite3.connect(args.db).execute(
        "SELECT value FROM meta WHERE key='embed_model'").fetchone()
    if row and row[0] != args.embed_model:
        print(f"Note: index was built with '{row[0]}' but you're querying "
              f"with '{args.embed_model}'. Re-index if answers look off.")

    if args.question:
        answer_one(args, chunks)
        return
    print("Ask about your documents (blank line quits).")
    while True:
        try:
            q = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        args.question = q
        answer_one(args, chunks)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tiny RAG playground: index your docs, then ask about them.")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama address")
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite index file")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="Ollama embedding model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="embed a folder of .txt/.md files")
    p.add_argument("--docs", required=True, help="folder of documents")
    p.add_argument("--chunk-words", type=int, default=250,
                   help="target words per chunk")
    p.add_argument("--overlap-words", type=int, default=50,
                   help="words overlapping between chunks")

    p = sub.add_parser("ask", help="ask a question about the indexed docs")
    p.add_argument("--question", default=None,
                   help="ask once and exit; omit for interactive mode")
    p.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL,
                   help="Ollama chat model for answers")
    p.add_argument("--top-k", type=int, default=3,
                   help="how many chunks to retrieve")
    p.add_argument("--hide-chunks", dest="show_chunks", action="store_false",
                   help="don't print the retrieved chunks")
    p.set_defaults(show_chunks=True)

    args = ap.parse_args(argv)
    if args.cmd == "index":
        cmd_index(args)
    else:
        cmd_ask(args)


if __name__ == "__main__":
    main()
