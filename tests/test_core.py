import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from movie_rag.answer import PLOT_QUESTION, extractive_answer, grounded_fallback, identity_evidence, plot_evidence
from movie_rag.app import diverse_hits
from movie_rag.chunking import chunks
from movie_rag.db import Database
from movie_rag.search import SearchEngine


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

    def test_warm_vector_index_skips_full_database_scan(self):
        engine = SearchEngine(self.db, SimpleNamespace(vector_path=Path(self.temp.name) / "vectors.npz"))
        engine._ids = np.array([4, 5], dtype=np.int64)
        engine._db_signature = self.db.chunk_signature()
        with patch.object(self.db, "all_chunks", side_effect=AssertionError("unexpected full scan")):
            self.assertEqual(engine.rebuild(), 2)


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

    def test_plot_evidence_keeps_atomic_facts_across_chunks(self):
        query = "кто такая Айгуль и почему она спрыгнула"
        hits = [
            (1, {"text": "Между Айгуль и Маратом начинаются романтические отношения. "
                 "Члены соседней банды похищают Айгуль."}),
            (3, {"text": "На следующий день Айгуль, не найдя у родителей жалости, выбрасывается из окна."}),
        ]
        evidence = plot_evidence(query, hits)
        self.assertIn("[1] Между Айгуль и Маратом", evidence)
        self.assertIn("[3] На следующий день Айгуль", evidence)

    def test_identity_evidence_skips_incidental_role_words(self):
        query = "кто такая Айгуль и почему она спрыгнула"
        evidence = "\n".join([
            "[1] Айгуль обращается за помощью, когда к ней пристаёт незнакомый парень из другой группировки.",
            "[1] Между Айгуль и Маратом начинаются романтические отношения.",
        ])
        self.assertEqual(identity_evidence(query, evidence), "[1] Между Айгуль и Маратом начинаются романтические отношения.")

    def test_plot_context_prioritizes_the_requested_event(self):
        query = "кто такая Айгуль и почему она спрыгнула"
        hits = [(1, {"text": "Он узнаёт, что у Ирины день рождения, и поёт для неё песню. "
                     "Между Айгуль и Маратом начинаются романтические отношения. "
                     "На дискотеке все узнают, что Айгуль была изнасилована, и подруги отворачиваются. "
                     "На следующий день Айгуль, не найдя у родителей жалости и сочувствия, выбрасывается из окна."})]
        evidence = plot_evidence(query, hits)
        self.assertIn("выбрасывается из окна", evidence)
        self.assertIn("романтические отношения", evidence)
        self.assertNotIn("день рождения", evidence)

    def test_plot_evidence_drops_pronoun_sentences_without_their_subject(self):
        query = "почему Айгуль спрыгнула"
        hit = {"text": "Айгуль остаётся наедине с нападавшим. Он насилует девушку, а вернувшийся Жёлтый его прогоняет. "
                       "На следующий день Айгуль выбрасывается из окна."}
        evidence = plot_evidence(query, [(1, hit)])
        self.assertNotIn("Он насилует", evidence)
        self.assertIn("выбрасывается из окна", evidence)

    def test_plot_fallback_uses_only_episode_evidence(self):
        hits = [{"source": "Wikipedia RU", "url": "wiki", "title": "Сериал", "score": 1.0,
                 "text": "На следующий день Айгуль, не найдя у родителей жалости и сочувствия, выбрасывается из окна."},
                {"source": "TMDB", "url": "tmdb", "title": "Сериал", "score": 0.5,
                 "text": "Описание: Молодёжная криминальная драма."}]
        answer = grounded_fallback("почему Айгуль спрыгнула", hits)
        self.assertIn("жалости и сочувствия", answer)
        self.assertNotIn("криминальная драма", answer)

    def test_character_questions_use_plot_evidence_path(self):
        self.assertIsNotNone(PLOT_QUESTION.search("Кто такая Айгуль и почему она спрыгнула?"))


if __name__ == "__main__":
    unittest.main()
