import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from movie_rag.answer import _validated_answer
from movie_rag.db import Database
from movie_rag.evidence import evidence_context
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


if __name__ == "__main__":
    unittest.main()
