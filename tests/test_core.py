import tempfile
import unittest
from pathlib import Path

from movie_rag.answer import extractive_answer
from movie_rag.app import diverse_hits
from movie_rag.chunking import chunks
from movie_rag.db import Database


class ChunkingTests(unittest.TestCase):
    def test_overlap_and_complete_coverage(self):
        words = [f"word{i}" for i in range(1000)]
        pieces = chunks(" ".join(words), target_words=400, overlap_words=60)
        self.assertGreaterEqual(len(pieces), 3)
        self.assertEqual(pieces[0].split()[:5], words[:5])
        self.assertEqual(pieces[-1].split()[-5:], words[-5:])
        self.assertEqual(pieces[0].split()[-60:], pieces[1].split()[:60])


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp.name) / "test.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_dedup_update_and_bm25(self):
        args = dict(source="TMDB", source_id="1", entity_type="movie", entity_id="1",
                    title="Inception", url="https://example.org/movie/1",
                    text="Christopher Nolan directed the film Inception about dreams.")
        ident, changed = self.db.put_document(**args)
        self.assertTrue(changed)
        self.assertEqual(self.db.chunk_count(), 1)
        ident_again, changed = self.db.put_document(**args)
        self.assertFalse(changed)
        self.assertEqual(ident, ident_again)
        self.assertEqual(self.db.chunk_count(), 1)
        self.assertEqual(len(self.db.bm25("Nolan")), 1)
        args["text"] = "The film explores nested dreams."
        _, changed = self.db.put_document(**args)
        self.assertTrue(changed)
        self.assertEqual(self.db.bm25("Nolan"), [])
        self.assertEqual(len(self.db.bm25("nested")), 1)

    def test_entity_filter(self):
        for number, title in [(1, "Inception"), (2, "Interstellar")]:
            self.db.put_document(
                source="TMDB", source_id=str(number), entity_type="movie", entity_id=str(number),
                title=title, url=f"https://example.org/{number}", text="Christopher Nolan film",
            )
        self.assertEqual(len(self.db.bm25("Nolan")), 2)
        selected = self.db.bm25("Nolan", entity=("movie", "2"))
        self.assertEqual(len(selected), 1)
        self.assertEqual(self.db.chunk_details([selected[0][0]])[selected[0][0]]["entity_id"], "2")


class AnswerTests(unittest.TestCase):
    def test_extractive_answer_has_source_marker(self):
        hits = [{"text": "Режиссёр: Кристофер Нолан. Фильм рассказывает об архитектуре снов и памяти.",
                 "score": 0.03, "source": "TMDB"}]
        answer = extractive_answer("Кто снял Inception?", hits)
        self.assertIn("[1]", answer)
        self.assertIn("Кристофер Нолан", answer)

    def test_context_includes_distinct_sources(self):
        hits = [{"url": "wiki", "text": str(i)} for i in range(5)] + [
            {"url": "tmdb", "text": "credits"}, {"url": "wiki-ru", "text": "история"}
        ]
        selected = diverse_hits(hits, limit=4)
        self.assertEqual({hit["url"] for hit in selected}, {"wiki", "tmdb", "wiki-ru"})
        self.assertEqual(len(selected), 4)


if __name__ == "__main__":
    unittest.main()
