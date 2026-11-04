import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from movie_rag.answer import _direct_answer, _validated_answer, generate, stream_generate
from movie_rag.db import Database
from movie_rag.evidence import evidence_context, planned_context, question_parts
from movie_rag.sources import Wikipedia, mentioned_title


class SubmissionTests(unittest.TestCase):
    def test_remake_does_not_override_requested_release(self):
        wiki = Wikipedia("", "ru")
        response = Mock()
        response.json.return_value = {"pages": [
            {"title": "Дюна (фильм, 1984)", "description": "фильм 1984 года"},
            {"title": "Дюна (фильм, 2021)", "description": "фильм 2021 года"},
        ]}
        with patch.object(wiki, "get", return_value=response):
            self.assertEqual(wiki.search_title("Дюна", "movie", "2021"), "Дюна (фильм, 2021)")
            self.assertIsNone(wiki.search_title("Дюна", "movie"))

    def test_local_titles_resolve_without_crossing_remakes(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "cinema.sqlite3")
            db.put_entity("movie", "42", "Solaris", {"title_ru": "Солярис"})
            self.assertEqual(db.match_entity("Кто снял Солярис?"), ("movie", "42"))
            db.put_entity("movie", "43", "Solaris", {"title_ru": "Солярис"})
            self.assertIsNone(db.match_entity("Кто снял Солярис?"))
        self.assertEqual(mentioned_title("Кто такой герой из сериала Слово пацана?"), "Слово пацана")

    def test_plot_reveal_keeps_answer_after_the_setup(self):
        hits = [{"title": "Фильм", "source": "Wikipedia RU",
                 "text": "## Сюжет Следователь Марк опрашивает свидетелей. "
                         "В финале доктор объясняет Марку происходящее. "
                         "Марк — пациент клиники, а расследование было частью лечения."}]
        self.assertIn("пациент клиники", evidence_context("Кем на самом деле был Марк?", hits))

    def test_abstention_does_not_require_a_citation(self):
        self.assertEqual(
            _validated_answer("В предоставленных фрагментах нет ответа на этот вопрос.", {1}, "[1] Текст."),
            "В найденных источниках ответа нет.",
        )

    def test_catalog_answers_every_question_or_yields_to_generation(self):
        hits = [{"source": "TMDB", "text": "Режиссёр: Мария Волкова. Сценаристы: Иван Белов. Композитор: Елена Соколова."}]
        answer = _direct_answer("Кто снял фильм и кто написал музыку?", hits)
        self.assertIn("Мария Волкова [1]", answer)
        self.assertIn("Елена Соколова [1]", answer)
        self.assertNotIn("Иван Белов", answer)
        self.assertIsNone(_direct_answer("Кто снял фильм и почему он выбрал такой финал?", hits))

    def test_context_keeps_two_independent_parts_of_a_question(self):
        question = "Почему Нина боялась моря и как она выбралась из города?"
        hits = [
            {"text": "Нина боялась моря после того, как её отец утонул в шторме."},
            {"text": "Нина выбралась из города через закрытый туннель в скале."},
        ]
        self.assertEqual(len(question_parts(question)), 2)
        context = planned_context(question, hits, 500)
        self.assertIn("её отец утонул", context)
        self.assertIn("туннель", context)

    def test_validation_discards_year_missing_from_cited_evidence(self):
        self.assertEqual(
            _validated_answer(
                "Фильм выпущен в 2008 году [1]. Главную роль сыграла Анна Иванова [2].",
                {1, 2}, "[1] Фильм выпущен в 2009 году.\n[2] Главную роль сыграла Анна Иванова.",
            ),
            "Главную роль сыграла Анна Иванова [2].",
        )

    def test_cache_reuses_completed_answer_and_invalidates_changed_evidence(self):
        config = SimpleNamespace(ollama_url="http://localhost:11434", ollama_model="test-cache")
        hits = [{"text": "Капитан открыл запертый вход, нажав кнопку рядом с дверью."}]
        query = "Как капитан открыл вход?"
        first = Mock()
        first.json.return_value = {"message": {"content": "Капитан нажал кнопку [1]."}}
        second = Mock()
        second.json.return_value = {"message": {"content": "Капитан использовал ключ [1]."}}
        with patch("movie_rag.answer.httpx.post", side_effect=[first, second]) as post:
            self.assertIn("кнопку", generate(query, hits, config)[0])
            event, = stream_generate(query, hits, config)
            self.assertTrue(event["cached"])
            self.assertEqual(event["usage"], {})
            hits[0]["text"] = "Капитан открыл запертый вход ключом, который нашёл на полу."
            self.assertIn("ключ", generate(query, hits, config)[0])
            self.assertEqual(post.call_count, 2)

    def test_cast_lookup_keeps_latin_role_name_for_russian_question(self):
        hits = [{
            "source": "TMDB",
            "text": "Персонажи и исполнители: Персонаж Vova — актёр Иван Янковский; "
                    "Персонаж Aygul' Akhmerova — актёр Анна Пересильд; "
                    "Персонаж Natasha — актёр Елизавета Базыкина.",
        }]
        context = evidence_context("Кто такая Айгуль?", hits)
        self.assertIn("Aygul' Akhmerova — актёр Анна Пересильд", context.splitlines()[0])

    def test_causal_sentence_keeps_the_event_it_refers_to(self):
        hits = [{
            "source": "Wikipedia RU",
            "text": "## Сюжет Отец Трумана якобы утонул на глазах у сына. "
                    "С тех пор Труман панически боится воды.",
        }]
        context = evidence_context("Почему Труман боялся воды?", hits)
        self.assertIn("отец трумана якобы утонул", context.casefold().splitlines()[0])

    def test_validation_preserves_a_title_containing_a_full_stop(self):
        answer = "Айгуль — героиня сериала «Слово пацана. Кровь на асфальте» [2]."
        self.assertEqual(_validated_answer(answer, {2}, "[2] Айгуль Ахмерова."), answer)


if __name__ == "__main__":
    unittest.main()
