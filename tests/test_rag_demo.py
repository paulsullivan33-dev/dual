"""Tests for rag_demo.py. Everything here runs without Ollama: the HTTP
calls (embed_text/chat) are covered only by mocking, since no model is
available in the test environment."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_demo


class OutputPathTests(unittest.TestCase):
    def test_default_db_lives_under_output(self):
        self.assertEqual(rag_demo.DEFAULT_DB, os.path.join("output", "rag_demo.db"))

    def test_relative_db_goes_under_output_absolute_unchanged(self):
        self.assertEqual(rag_demo.output_path("my.db"),
                         os.path.join("output", "my.db"))
        absolute = os.path.abspath("my.db")
        self.assertEqual(rag_demo.output_path(absolute), absolute)

    def test_open_db_creates_missing_folders(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "new", "index.db")
            rag_demo.open_db(path).close()
            self.assertTrue(os.path.isfile(path))


class ChunkingTests(unittest.TestCase):
    def test_overlap_keeps_straddling_words_in_both_chunks(self):
        words = [f"w{i}" for i in range(600)]
        chunks = rag_demo.chunk_text(" ".join(words),
                                     chunk_words=250, overlap_words=50)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0].split()), 250)
        # chunk 2 starts 50 words before chunk 1 ends
        self.assertEqual(chunks[0].split()[-1], "w249")
        self.assertEqual(chunks[1].split()[0], "w200")
        self.assertEqual(chunks[1].split()[-1], "w449")
        self.assertEqual(chunks[2].split()[0], "w400")

    def test_short_and_empty_text(self):
        self.assertEqual(rag_demo.chunk_text("hi"), ["hi"])
        self.assertEqual(rag_demo.chunk_text(""), [])
        self.assertEqual(rag_demo.chunk_text("   "), [])

    def test_line_breaks_survive_chunking(self):
        text = "line one\nline two\n\nsecond para here"
        chunks = rag_demo.chunk_text(text, chunk_words=10, overlap_words=0)
        self.assertEqual(chunks, [text])
        self.assertIn("\n", chunks[0])
        self.assertIn("\n\n", chunks[0])


class CosineTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertEqual(rag_demo.cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(rag_demo.cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_scores_zero(self):
        self.assertEqual(rag_demo.cosine([0.0, 0.0], [1.0, 0.0]), 0.0)


class RetrievalTests(unittest.TestCase):
    def test_most_similar_chunk_ranks_first(self):
        chunks = [("f1", 0, "cats are great", [1.0, 0.1]),
                  ("f2", 0, "dogs are great", [0.1, 1.0])]
        hits = rag_demo.retrieve([1.0, 0.0], chunks, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], "f1")

    def test_top_k_limits_results(self):
        chunks = [("f", i, "t", [float(i + 1), 0.0]) for i in range(5)]
        hits = rag_demo.retrieve([1.0, 0.0], chunks, top_k=2)
        self.assertEqual(len(hits), 2)


class PromptTests(unittest.TestCase):
    def test_prompt_labels_sources_and_carries_question(self):
        prompt = rag_demo.build_prompt("q?", [("s.txt", 2, "chunk text", [0.1])])
        self.assertIn("[1] (source: s.txt, chunk 2)", prompt)
        self.assertIn("chunk text", prompt)
        self.assertIn("Question: q?", prompt)


class ThinkStripTests(unittest.TestCase):
    def _chat(self, content):
        with mock.patch.object(
                rag_demo, "ollama_post",
                return_value={"message": {"content": content}}):
            return rag_demo.chat("http://x", "m", "sys", "q?")

    def test_think_block_stripped(self):
        out = self._chat("<think>reasoning here</think>The answer.")
        self.assertEqual(out, "The answer.")

    def test_multiline_think_block_stripped(self):
        out = self._chat("<think>line one\nline two</think>\nThe answer.")
        self.assertEqual(out, "The answer.")

    def test_lone_closing_tag_strips_preceding_thinking(self):
        out = self._chat("Let me think about this.\n</think>\nThe answer.")
        self.assertEqual(out, "The answer.")


class WrapTextTests(unittest.TestCase):
    def test_long_line_wraps_at_word_boundaries(self):
        text = "word " * 30
        out = rag_demo.wrap_text(text.strip(), width=20)
        self.assertGreater(len(out.split("\n")), 1)
        for line in out.split("\n"):
            self.assertLessEqual(len(line), 20)

    def test_existing_breaks_preserved(self):
        text = "para one\n\npara two"
        self.assertEqual(rag_demo.wrap_text(text), text)


class ThinkStripNoTagTests(unittest.TestCase):
    def _chat(self, content):
        with mock.patch.object(
                rag_demo, "ollama_post",
                return_value={"message": {"content": content}}):
            return rag_demo.chat("http://x", "m", "sys", "q?")

    def test_no_think_block_unchanged(self):
        out = self._chat("Just the answer.")
        self.assertEqual(out, "Just the answer.")

    def test_think_false_sent_to_ollama(self):
        with mock.patch.object(
                rag_demo, "ollama_post",
                return_value={"message": {"content": "x"}}) as m:
            rag_demo.chat("http://x", "m", "sys", "q?")
        self.assertFalse(m.call_args[0][2]["think"])


class DatabaseTests(unittest.TestCase):
    def test_chunk_and_embedding_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db")
            conn = rag_demo.open_db(db)
            conn.execute(
                "INSERT INTO chunks (source, idx, text, embedding)"
                " VALUES (?,?,?,?)",
                ("s.txt", 0, "hello", json.dumps([0.5, 0.5])))
            conn.commit()
            loaded = rag_demo.load_chunks(conn)
            conn.close()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][:3], ("s.txt", 0, "hello"))
        self.assertEqual(loaded[0][3], [0.5, 0.5])


class AskPathTests(unittest.TestCase):
    """End-to-end ask flow with the Ollama HTTP calls faked."""

    def _seed_db(self, path):
        conn = rag_demo.open_db(path)
        conn.execute(
            "INSERT INTO chunks (source, idx, text, embedding)"
            " VALUES (?,?,?,?)",
            ("notes.txt", 0, "The Arduino Q runs at 2.8 tokens per second.",
             json.dumps([1.0, 0.0])))
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)",
                     ("embed_model", "nomic-embed-text"))
        conn.commit()
        conn.close()

    def test_ask_shows_chunks_builds_prompt_and_prints_answer(self):
        seen = {}

        def fake_embed(host, model, text):
            seen["embed"] = text
            return [1.0, 0.0]

        def fake_chat(host, model, system, user_text):
            seen["system"], seen["user"] = system, user_text
            return "2.8 tokens per second."

        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db")
            self._seed_db(db)
            buf = io.StringIO()
            with mock.patch.object(rag_demo, "embed_text", fake_embed), \
                 mock.patch.object(rag_demo, "chat", fake_chat), \
                 redirect_stdout(buf):
                rag_demo.main(["--db", db, "ask",
                               "--question", "how fast is the Q?"])
            out = buf.getvalue()

        self.assertIn("notes.txt", out)          # chunk listing shown
        self.assertIn("similarity 1.000", out)   # with its score
        self.assertIn("ONLY the context", seen["system"])
        self.assertIn("The Arduino Q runs at 2.8", seen["user"])
        self.assertIn("how fast is the Q?", seen["user"])
        self.assertIn("2.8 tokens per second.", out)

    def test_ask_without_database_exits(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                rag_demo.main(["--db", os.path.join(d, "nope.db"),
                               "ask", "--question", "q"])


class IndexAppendTests(unittest.TestCase):
    """`index --append` grows an existing index instead of wiping it."""

    def _write(self, docs, name, text):
        path = os.path.join(docs, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _sources(self, db):
        conn = rag_demo.open_db(db)
        try:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT source FROM chunks ORDER BY source")]
        finally:
            conn.close()

    def _chunks_for(self, db, source):
        conn = rag_demo.open_db(db)
        try:
            return [r[0] for r in conn.execute(
                "SELECT text FROM chunks WHERE source = ? ORDER BY idx",
                (source,))]
        finally:
            conn.close()

    def _run_index(self, db, docs, *extra):
        with mock.patch.object(rag_demo, "embed_text",
                               lambda _h, _m, _t: [1.0, 0.0]):
            rag_demo.main(["--db", db, "index", "--docs", docs, *extra])

    def _setup(self, d):
        docs = os.path.join(d, "docs")
        os.mkdir(docs)
        return docs, os.path.join(d, "t.db")

    def test_append_keeps_old_chunks_and_adds_new_files(self):
        with tempfile.TemporaryDirectory() as d:
            docs, db = self._setup(d)
            self._write(docs, "old.txt", "the old note")
            self._run_index(db, docs)
            self._write(docs, "new.txt", "the new note")
            with redirect_stdout(io.StringIO()):
                self._run_index(db, docs, "--append")
            self.assertEqual(self._sources(db), ["new.txt", "old.txt"])
            self.assertEqual(self._chunks_for(db, "old.txt"), ["the old note"])
            self.assertEqual(self._chunks_for(db, "new.txt"), ["the new note"])

    def test_append_replaces_changed_file_without_duplicating(self):
        with tempfile.TemporaryDirectory() as d:
            docs, db = self._setup(d)
            self._write(docs, "novel.txt", "chapter one")
            self._run_index(db, docs)
            self._write(docs, "novel.txt", "chapter one, revised")
            with redirect_stdout(io.StringIO()):
                self._run_index(db, docs, "--append")
            self.assertEqual(self._chunks_for(db, "novel.txt"),
                             ["chapter one, revised"])

    def test_append_refuses_a_different_embedding_model(self):
        with tempfile.TemporaryDirectory() as d:
            docs, db = self._setup(d)
            self._write(docs, "a.txt", "hello")
            self._run_index(db, docs)
            conn = rag_demo.open_db(db)
            conn.execute("UPDATE meta SET value='other-model' "
                         "WHERE key='embed_model'")
            conn.commit()
            conn.close()
            with redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self._run_index(db, docs, "--append")

    def test_plain_index_still_rebuilds_from_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            docs, db = self._setup(d)
            self._write(docs, "gone.txt", "will be wiped")
            self._run_index(db, docs)
            os.remove(os.path.join(docs, "gone.txt"))
            self._write(docs, "kept.txt", "fresh")
            self._run_index(db, docs)
            self.assertEqual(self._sources(db), ["kept.txt"])


if __name__ == "__main__":
    unittest.main()
