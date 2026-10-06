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


if __name__ == "__main__":
    unittest.main()
