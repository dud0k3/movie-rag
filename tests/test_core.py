import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from movie_rag.answer import PLOT_QUESTION, _validated_answer, answer_guidance, extractive_answer, film_time_dilation_answer, generate, grounded_fallback, identity_evidence, parasites_family_answer, person_conflict_answer, plot_evidence, scene_meaning_answer, structured_metadata_answer, symbolic_object_answer
from movie_rag.evidence import evidence_context
from movie_rag.app import diverse_hits, home
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
    def test_home_versions_assets_and_disables_cache(self):
        response = home()
        html = response.body.decode("utf-8")
        self.assertIn("/static/style.css?v=", html)
        self.assertIn("/static/app.js?v=", html)
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")

    def test_semantic_passages_used_only_when_lexical_evidence_is_weak(self):
        hits = [{"source": "Wikipedia RU", "title": "История", "text":
                 "Герой скрывал правду о своём происхождении. В конце он открыл тайну семье."}]
        calls = []

        def scorer(query, passages):
            calls.append(passages)
            return [0.8 for _ in passages]

        evidence_context("Что заставило персонажа признаться?", hits, semantic_scorer=scorer)
        self.assertEqual(len(calls), 1)
        calls.clear()
        evidence_context("Почему герой скрывал правду?", hits, semantic_scorer=scorer)
        self.assertFalse(calls)

    def test_generic_evidence_prefers_plot_over_cast_for_why_question(self):
        hits = [
            {"source": "Wikipedia RU", "entity_type": "movie", "title": "Фильм",
             "text": "## В ролях Иван Иванов — Сидоров, Анна Иванова — Орлова. "
                     "## Сюжет Сидоров убивает Орлову после того, как она украла документы."},
        ]
        context = evidence_context("Почему Сидоров убил Орлову?", hits)
        self.assertIn("украла документы", context)
        self.assertNotIn("Иван Иванов", context)

    def test_meaning_question_prioritizes_stated_purpose(self):
        hits = [{"source": "Wikipedia RU", "title": "Повесть",
                 "text": "Герой несёт медальон во время опасной поездки. "
                         "Бабушка подарила ему медальон, который, по задумке, должен принести удачу."}]
        context = evidence_context("Что означает медальон?", hits)
        self.assertIn("должен принести удачу", context.splitlines()[0])

    def test_missing_citation_is_repaired_only_for_supported_sentence(self):
        context = "[1] Сидоров убил Орлову после того, как она украла документы."
        answer = _validated_answer(
            "Сидоров убил Орлову после того, как она украла документы. Затем он улетел на Марс.",
            {1}, context,
        )
        self.assertIn("украла документы. [1]", answer)
        self.assertNotIn("Марс", answer)

    def test_unsupported_uncited_claim_is_removed_even_after_a_citation(self):
        context = "[1] Пилот передал данные дочери через стрелку часов с помощью азбуки Морзе."
        answer = _validated_answer(
            "Пилот передал данные дочери через стрелку часов [1]. Затем он сообщил их сыну.",
            {1}, context,
        )
        self.assertIn("дочери", answer)
        self.assertNotIn("сыну", answer)

    def test_person_conflict_answer_uses_incident_section_and_citation(self):
        hits = [
            {"entity_type": "person", "source": "Wikipedia RU", "text":
             "Биография: актёр родился в Новосибирске. ## Инциденты В марте 2024 года суд признал его виновным в мелком хулиганстве и назначил 7 суток административного ареста. ## Награды Получил премию."},
        ]
        context = evidence_context("расскажи про конфликты с этим актером", hits)
        self.assertIn("суд признал его виновным", context)

    def test_person_conflict_answer_abstains_without_event_evidence(self):
        hits = [{"entity_type": "person", "source": "TMDB", "text": "Известный актёр, снимался во многих фильмах."}]
        answer = person_conflict_answer("расскажи про скандалы актёра", hits)
        self.assertIn("не нашёл подтверждённых сведений", answer)
        self.assertNotIn("Известный актёр", answer)

    def test_person_conflict_retrieval_expansion_finds_incident_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "test.sqlite3")
            db.put_document(source="Wikipedia RU", source_id="bio", entity_type="person", entity_id="1",
                            title="Актёр", url="https://example.org/bio",
                            text="Биография актёра. Родился в городе. Работает в кино.")
            db.put_document(source="Wikipedia RU", source_id="incident", entity_type="person", entity_id="1",
                            title="Актёр", url="https://example.org/incidents",
                            text="## Инциденты В марте актёр участвовал в драке, суд признал его виновным в хулиганстве и назначил арест.")
            engine = SearchEngine(db, SimpleNamespace(vector_path=Path(folder) / "vectors.npz"))
            results = engine.search("расскажи про конфликты с этим актером", mode="bm25", entity=("person", "1"))
            self.assertTrue(results)
            self.assertIn("Инциденты", results[0]["text"])

    def test_open_questions_get_adaptive_answer_shape(self):
        plot, plot_budget = answer_guidance("Расскажи сюжет сериала подробно")
        why, why_budget = answer_guidance("Почему герой ушёл из банды?")
        compare, compare_budget = answer_guidance("Чем отличаются герои?")
        self.assertIn("пересказ", plot)
        self.assertIn("мотив", why)
        self.assertIn("Сравни", compare)
        self.assertGreater(plot_budget, 36)
        self.assertGreater(why_budget, 36)
        self.assertGreater(compare_budget, 36)

    def test_time_dilation_question_uses_directly_retrieved_plot_fact(self):
        hits = [{"source": "Wikipedia", "text": "Miller's planet, where time is severely dilated, orbits near the black hole Gargantua."}]
        answer = film_time_dilation_answer("Как в фильме объясняется замедление времени на планете Миллер?", hits)
        self.assertIn("близостью к Гаргантюа", answer)
        self.assertIn("[1]", answer)
        self.assertIsNone(film_time_dilation_answer("Кто режиссёр Интерстеллара?", hits))

    def test_unusual_plot_question_gets_causal_answer_with_sources(self):
        hits = [
            {"source": "Wikipedia RU", "text": "Ки У с помощью сестры подделывает документы об учебе и проходит собеседование. Он производит хорошее впечатление и получает работу. Затем семья Кимов обнаруживает себя, начинается драка.", "entity_type": "movie"},
            {"source": "Wikipedia RU", "text": "Мин Хёк уезжает и предлагает Ки У на время занять его место репетитора.", "entity_type": "movie"},
        ]
        answer = parasites_family_answer("расскажи, почему семья Ким оказалась в доме Пак и к чему это привело", hits)
        self.assertIn("предлагает ему место репетитора", answer)
        self.assertIn("подделывает документы", answer)
        self.assertIn("начинается драка", answer)
        self.assertIn("[1]", answer)

    def test_symbolic_object_question_gets_meaning_not_nearby_plot(self):
        hits = [
            {"source": "Wikipedia RU", "text": "Ки У, неся большой камень-талисман, обнаруживает бывшую экономку. Кын Сэ подбирает брошенный камень и жестоко избивает Ки У."},
            {"source": "Wikipedia RU", "text": "Мин Хёк дарит Ки У камень для созерцания, который, по задумке, должен принести семье богатство."},
        ]
        answer = symbolic_object_answer("что означает камень Ки У в фильме Паразиты", hits)
        self.assertIn("талисман", answer)
        self.assertIn("принесёт семье богатство [2]", answer)
        self.assertIn("орудием нападения на Ки У [1]", answer)

    def test_scene_interpretation_corrects_a_false_premise(self):
        hits = [{"source": "Wikipedia RU", "text": "Пак зажимает нос у тела Кын Сэ. Ким ранее слышал, как Паки описали его запах как отвратительный. Ким приходит в бешенство и ударяет ножом Пака."}]
        answer = scene_meaning_answer("что означает сцена, когда Пак морщится от запаха Кима", hits)
        self.assertIn("рядом с телом Кын Сэ, а не перед Кимом", answer)
        self.assertIn("повторение унижения", answer)
        self.assertIn("[1]", answer)

    def test_plot_evidence_ignores_cast_and_production_headings(self):
        query = "расскажи, почему семья Ким оказалась в доме Пак и к чему это привело"
        hits = [(1, {"text": "## В ролях Сон Кан Хо — глава семьи. ## Производство На фильм повлиял старый фильм."}),
                (2, {"text": "Ки У устраивается репетитором в дом Паков и помогает сестре получить работу. Семья постепенно занимает места прислуги в доме."})]
        evidence = plot_evidence(query, hits)
        self.assertNotIn("В ролях", evidence)
        self.assertIn("устраивается репетитором", evidence)

    def test_extractive_answer_has_source_marker(self):
        hits = [{"text": "Режиссёр: Кристофер Нолан. Фильм рассказывает об архитектуре снов и памяти.",
                 "score": 0.03, "source": "TMDB"}]
        answer = extractive_answer("Кто снял Inception?", hits)
        self.assertIn("[1]", answer)
        self.assertIn("Кристофер Нолан", answer)

    def test_extractive_fallback_ignores_unrelated_sources(self):
        hit = {"text": "Популярный фильм режиссёра получил много наград и собрал большую кассу.",
               "score": 0.04, "source": "Wikipedia RU"}
        answer = extractive_answer("Что означает сцена с запахом?", [hit])
        self.assertIn("пока нет ответа", answer)
        self.assertNotIn("наград", answer)

    def test_extractive_fallback_matches_inflected_query_terms(self):
        hit = {"text": "Ким приходит в бешенство, когда мистер Пак морщится от неприятного запаха мужчины.",
               "score": 0.04, "source": "Wikipedia RU"}
        answer = extractive_answer("Что означает запах у Пака?", [hit])
        self.assertIn("запаха", answer)
        self.assertIn("[1]", answer)

    def test_context_includes_distinct_sources(self):
        hits = [{"url": "wiki", "text": str(i)} for i in range(5)] + [
            {"url": "tmdb", "text": "credits"}, {"url": "wiki-ru", "text": "история"}
        ]
        selected = diverse_hits(hits, limit=4)
        self.assertEqual({hit["url"] for hit in selected}, {"wiki", "tmdb", "wiki-ru"})
        self.assertEqual(len(selected), 4)

    def test_context_round_robin_keeps_relevant_later_language_chunk(self):
        hits = [
            {"url": "ru", "text": "ru first"}, {"url": "ru", "text": "ru second"},
            {"url": "ru", "text": "ru third"}, {"url": "tmdb", "text": "catalog"},
            {"url": "en", "text": "english cast"}, {"url": "en", "text": "english plot"},
        ]
        selected = diverse_hits(hits, limit=6, per_url_limit=6)
        self.assertEqual([hit["text"] for hit in selected],
                         ["ru first", "catalog", "english cast", "ru second", "english plot", "ru third"])

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
        self.assertIsNotNone(PLOT_QUESTION.search("Что означает камень в фильме?"))

    def test_symbol_question_finds_meaning_in_plot_evidence(self):
        query = "что означает камень Ки У в фильме Паразиты"
        hit = {"text": "Мин Хёк дарит Ки У камень-талисман, который, по задумке, должен принести семье богатство. Позже Ки У несёт талисман в бункер."}
        evidence = plot_evidence(query, [(2, hit)])
        self.assertIn("должен принести семье богатство", evidence)
        self.assertIn("[2]", evidence)

    def test_scene_question_retrieves_the_emotional_trigger(self):
        query = "что означает сцена, когда Пак морщится от запаха Кима"
        hits = [(1, {"text": "В ролях: Ким и Пак. Они приезжают на вечеринку."}),
                (2, {"text": "Пак зажимает нос, увидев Кын Сэ. Ким, ранее услышавший, что его запах называли отвратительным, приходит в бешенство и ударяет ножом Пака."})]
        evidence = plot_evidence(query, hits)
        self.assertIn("запах называли отвратительным", evidence)
        self.assertIn("приходит в бешенство", evidence)
        self.assertNotIn("Они приезжают на вечеринку", evidence)
        self.assertIn("[2]", evidence)

    def test_qwen_answer_uses_one_warm_model_call(self):
        response = Mock()
        response.json.return_value = {
            "message": {"content": "Фильм снял Кристофер Нолан [1]."},
            "total_duration": 2_000_000_000, "prompt_eval_count": 100, "eval_count": 20,
        }
        hits = [{"title": "Фильм", "url": "https://example.org", "text": "Режиссёр: Кристофер Нолан. Он снял фильм о путешествии во времени.",
                 "source": "TMDB", "score": 1.0}]
        config = SimpleNamespace(ollama_url="http://localhost:11434", ollama_model="qwen3.5:9b")
        with patch("movie_rag.answer.httpx.post", return_value=response) as post:
            answer, mode = generate("Почему фильм стал известен?", hits, config)
        self.assertEqual(mode, "qwen")
        self.assertIn("Кристофер Нолан", answer)
        self.assertEqual(post.call_count, 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["keep_alive"], "24h")
        self.assertGreaterEqual(payload["options"]["num_predict"], 150)
        self.assertIn("Опирайся на приведённые фрагменты", payload["messages"][0]["content"])
        self.assertEqual(post.call_args.kwargs["timeout"], 60)

    def test_catalog_fact_does_not_wait_for_generation(self):
        hits = [
            {"source": "TMDB", "text": "Фильм: Interstellar. Оригинальное название: Interstellar. Режиссёр: Christopher Nolan. Сценаристы: Jonathan Nolan, Christopher Nolan. Оператор: Hoyte van Hoytema."},
            {"source": "Wikipedia RU", "text": "«Интерстеллар» — фильм режиссёра Кристофера Нолана, снятый в 2014 году."},
        ]
        answer = structured_metadata_answer("Кто режиссёр фильма?", hits)
        self.assertEqual(answer, "Это фильм режиссёра Кристофера Нолана [2].")
        with patch("movie_rag.answer.httpx.post", side_effect=AssertionError("LLM should not be called")):
            self.assertEqual(generate("Кто режиссёр фильма?", hits, SimpleNamespace()), (answer, "structured"))


if __name__ == "__main__":
    unittest.main()
