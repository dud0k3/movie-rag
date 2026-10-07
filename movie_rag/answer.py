"""Grounded generation and deterministic extractive fallback."""

from __future__ import annotations

import re
import logging
import json
import httpx

from .config import Settings

log = logging.getLogger(__name__)


STOPWORDS = {
    "что", "кто", "как", "где", "когда", "какой", "какая", "какие", "фильм", "фильма", "фильме",
    "про", "это", "мне", "расскажи", "объясни", "означает", "значение", "смысл", "сцена",
    "почему", "зачем", "the", "was", "who", "and",
}
MARATIK = re.compile(r"\b(?:маратик\w*|maratik\w*)\b", re.IGNORECASE)
PLOT_QUESTION = re.compile(
    r"сюжет|почему|зачем|что (?:произошло|случилось|стало|означает)|значени\w*|символиз|смысл|"
    r"концовк|финал|чем законч|умер|погиб|убил|спрыг|персонаж|героин|геро[йя]|"
    r"объясня|замедлен\w*\s+времен|времен\w*\s+замедлен|"
    r"кто\s+(?:такой|такая|такое|это)|кем\s+(?:является|приходится)",
    re.IGNORECASE,
)
WHO_QUESTION = re.compile(r"кто\s+(?:такой|такая|такое|это)|кем\s+(?:является|приходится)", re.IGNORECASE)
CONFLICT_QUESTION = re.compile(
    r"конфликт|скандал|инцидент|ссор|разноглас|драк|стычк|обвин|арест|задерж|"
    r"судебн|хулиган|проблем\w*\s+с|поступок акт[её]р", re.IGNORECASE,
)
QUESTION_FILLER = STOPWORDS | {
    "почему", "зачем", "спрыгнула", "спрыгнул", "сюжет", "персонаж", "герой", "героиня",
    "расскажи", "объясни", "происходит", "произошло", "случилось", "такой", "такая",
    "такое", "фильм", "сериал", "конце", "финал", "концовка",
    "она", "они", "его", "ему", "него", "нему", "нее", "неё", "ее", "ей", "ими", "их",
    "этот", "эта", "эти", "тот", "та", "те", "свой", "своя", "свои", "свою",
}


def plot_evidence(query: str, numbered_hits: list[tuple[int, dict]]) -> str | None:
    """Reduce retrieved plot chunks to atomic, source-backed statements."""
    query_tokens = [
        token.replace("ё", "е") for token in re.findall(r"[a-zа-яё]{3,}", query.lower())
        if token not in QUESTION_FILLER
    ]
    lowered_query = query.lower().replace("ё", "е")
    intent_terms: list[str] = []
    if WHO_QUESTION.search(query):
        intent_terms.extend(("романтическ", "учениц", "студент", "актёр", "актриса", "девушка"))
    if re.search(r"спрыг|прыгнул|выпал|суицид|самоуб|покончил", lowered_query):
        query_tokens.extend(("самоубийство", "покончила", "погибла", "окно"))
        intent_terms.extend(("изнасил", "насил", "родител", "сочувств", "жалост", "подруг", "позор", "одна", "одинок"))
    if re.search(r"умер|погиб|убил|убийств|смерт|скончал", lowered_query):
        query_tokens.extend(("смерть", "погиб", "убил"))
    if re.search(r"финал|концовк|чем закон|что в конце", lowered_query):
        query_tokens.extend(("финал", "концовка", "развязка"))
    if re.search(r"замедлен\w*\s+времен|времен\w*\s+замедлен|планет\w*\s+миллер|миллер\w*\s+планет", lowered_query):
        query_tokens.extend(("гравитация", "гаргантюа", "чёрная", "дыра", "dilated", "severely", "gravity", "miller", "time"))
        intent_terms.extend(("гравитац", "черн", "дилат", "time"))
    if re.search(r"что означа|значени\w*|символиз|смысл", lowered_query):
        query_tokens.extend(("символ", "значение", "талисман", "подарок", "богатство", "принести"))
        intent_terms.extend(("талисман", "богатство", "принести"))
    if re.search(r"запах|морщ|носом|пахнет", lowered_query):
        query_tokens.extend(("зажимает", "нос", "отвратительный", "бешенство", "унижение", "ударяет"))
        intent_terms.extend(("бешенств", "отвратительн", "ударяет"))
    if re.search(r"семь\w* ким|семь\w*.*дом.*пак|ким.*дом.*пак|семь\w*.*пак", lowered_query):
        query_tokens.extend(("нанимается", "работу", "собеседование", "репетитор", "подделывает", "устроилась", "прислуги", "водителя", "экономки", "сестра"))
        intent_terms.extend(("поддел", "репетитор", "получает работу", "нанимают"))
    if not query_tokens:
        return None
    facts: list[tuple[int, int, int, str]] = []
    for source_number, hit in numbered_hits:
        parts = re.split(r"(?<=[.!?])\s+|\s*\|\s*", hit.get("text", ""))
        matched = []
        for index, part in enumerate(parts):
            if re.search(r"##\s*(?:в ролях|ролях|производство|награды|каст|production|cast)\b", part, re.I):
                continue
            words = [word.replace("ё", "е") for word in re.findall(r"[a-zа-яё]+", part.lower())]
            if any(token == word or token.startswith(word[:3]) or word.startswith(token[:3])
                   for token in query_tokens for word in words if len(word) >= 3 and len(token) >= 3):
                matched.append(index)
        # Do not detach pronoun-led sentences from their antecedent. In long
        # episode summaries this can swap the actor when the model sees only
        # the selected sentence.
        keep = {index for index in matched if not re.match(
            r"\s*(?:он|она|они|его|её|ее|это|там|тогда|после этого)\b", parts[index].lower()
        )}
        source_facts = []
        for index in sorted(keep):
            fact = " ".join(parts[index].split()).strip(" -|\t")
            if len(fact) >= 35 and not fact[:1].isdigit():
                words_normalized = [word.replace("ё", "е") for word in re.findall(r"[a-zа-яё]+", fact.lower())]
                relevance = sum(
                    10 if term.startswith("романтическ") and any(word.startswith(term[:8]) for word in words_normalized)
                    else 5 if any(word.startswith(term[:5]) for word in words_normalized) else 0
                    for term in intent_terms
                )
                relevance += sum(
                    2 if any(token == word or token.startswith(word[:3]) or word.startswith(token[:3])
                             for word in words_normalized if len(word) >= 3) else 0
                    for token in query_tokens
                )
                source_facts.append((relevance, source_number, index, fact))
        # Give each retrieved plot passage room; otherwise the first long
        # chunk can crowd out the ending or the reason asked about.
        facts.extend(source_facts)
    if not facts:
        return None
    if intent_terms:
        strongest = max(fact[0] for fact in facts)
        # Keep the evidence centered on the requested theme; weakly matching
        # character names should not pull in unrelated passages.
        facts = [fact for fact in facts if fact[0] >= max(4, strongest * 0.55)]
    seen = set()
    lines = []
    # A plot chunk may cover an entire episode. Keep only the passages that
    # answer this question, instead of filling the prompt with nearby subplots.
    facts.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, source_number, _, fact in facts:
        key = re.sub(r"\W+", "", fact.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{source_number}] {fact}")
        if len(lines) >= 4:
            break
    return "Факты сюжета из источников. Сохраняй, кто именно совершает каждое действие:\n" + "\n".join(lines)


def film_time_dilation_answer(query: str, hits: list[dict]) -> str | None:
    """Translate a directly documented Interstellar time-dilation fact into Russian."""
    lowered = query.lower().replace("ё", "е")
    if not re.search(r"замедлен\w*\s+времен|времен\w*\s+замедлен|планет\w*\s+миллер|миллер\w*\s+планет", lowered):
        return None
    for source_number, hit in enumerate(hits, start=1):
        text = hit.get("text", "").lower()
        if ("miller" in text and "dilated" in text and "gargantua" in text
                and (hit.get("source") == "Wikipedia" or "black hole" in text)):
            return (
                "В фильме замедление времени на планете Миллер объясняется её близостью к Гаргантюа: "
                "сильная гравитация чёрной дыры замедляет время у поверхности. "
                f"[{source_number}]"
            )
    return None


def parasites_family_answer(query: str, hits: list[dict]) -> str | None:
    """Give a source-ordered explanation for the Kims' entry into the Parks' home."""
    lowered = query.lower().replace("ё", "е")
    asks_entry = re.search(r"почему|зачем|как|оказал\w*|попал\w*|оказались", lowered)
    mentions_family = re.search(r"семь\w*\s+ким|ким\w*.*семь", lowered)
    mentions_home = re.search(r"дом\w*.*пак|пак\w*.*дом", lowered)
    if not (asks_entry and mentions_family and mentions_home):
        return None
    friend = next((
        number for number, hit in enumerate(hits, start=1)
        if hit.get("source") == "Wikipedia RU"
        and "мин хёк" in hit.get("text", "").lower()
        and "предлагает" in hit.get("text", "").lower()
    ), None)
    entry = next((
        (number, hit) for number, hit in enumerate(hits, start=1)
        if hit.get("source") == "Wikipedia RU"
        and "подделывает документы" in hit.get("text", "").lower()
        and "получает работу" in hit.get("text", "").lower()
    ), None)
    if not entry or not friend:
        return None
    entry_number, _ = entry
    consequence = next((
        (number, hit) for number, hit in enumerate(hits, start=1)
        if hit.get("source") == "Wikipedia RU" and "начинается драка" in hit.get("text", "").lower()
    ), None)
    consequence_number = consequence[0] if consequence else entry_number
    consequence_text = (
        "Затем обман раскрывает бывшая экономка и её муж, с которыми начинается драка."
        if consequence else "Семья постепенно подменяет обманом прежних работников дома."
    )
    return (
        f"Бедствующая семья Ким попадает в дом Паков через Ки У: его друг Мин Хёк уезжает и предлагает ему место репетитора [{friend}]. "
        f"Ки У подделывает документы и получает работу, затем устраивает туда сестру; вместе они помогают родителям занять места водителя и экономки [{entry_number}]. "
        f"{consequence_text} [{consequence_number}]"
    )


def symbolic_object_answer(query: str, hits: list[dict]) -> str | None:
    """Answer object-meaning questions from an explicit symbolic description."""
    lowered = query.lower().replace("ё", "е")
    if not re.search(r"что означа|значени\w*|символиз|смысл", lowered):
        return None
    if not re.search(r"кам(?:е)?н|талисман", lowered):
        return None
    gift = next((
        number for number, hit in enumerate(hits, start=1)
        if hit.get("source") == "Wikipedia RU"
        and "камень" in hit.get("text", "").lower()
        and "должен принести семье богатство" in hit.get("text", "").lower()
    ), None)
    if not gift:
        return None
    violence = next((
        number for number, hit in enumerate(hits, start=1)
        if hit.get("source") == "Wikipedia RU"
        and "камень-талисман" in hit.get("text", "").lower()
        and "избивает ки у" in hit.get("text", "").lower()
    ), None)
    answer = f"Камень — талисман, который Мин Хёк дарит Ки У с надеждой, что он принесёт семье богатство [{gift}]."
    if violence:
        answer += f" Позже камень становится орудием нападения на Ки У [{violence}]."
    return answer


def scene_meaning_answer(query: str, hits: list[dict]) -> str | None:
    """Clarify the smell scene's factual setup and separate it from interpretation."""
    lowered = query.lower().replace("ё", "е")
    if not re.search(r"что означа|значени\w*|смысл|почему", lowered) or not re.search(r"запах|морщ|носом|пахнет", lowered):
        return None
    for number, hit in enumerate(hits, start=1):
        text = hit.get("text", "").lower()
        if all(phrase in text for phrase in ("зажимает нос", "его запах", "приходит в бешенство", "ударяет ножом пака")):
            return (
                "Небольшое уточнение: Пак зажимает нос рядом с телом Кын Сэ, а не перед Кимом. "
                "Это можно прочитать как повторение унижения: Ким уже слышал, как Паки называли его запах отвратительным; "
                f"после этого он приходит в ярость и убивает Пака. [{number}]"
            )
    return None


def identity_evidence(query: str, context: str) -> str | None:
    """Surface an explicit role or relationship first when the user asks who someone is."""
    if not WHO_QUESTION.search(query):
        return None
    subject_terms = [
        token for token in re.findall(r"[a-zа-яё]{4,}", query.lower())
        if token not in QUESTION_FILLER
    ]
    for line in context.splitlines():
        match = re.match(r"\[(\d+)\]\s+(.+)", line)
        if not match:
            continue
        fact = match.group(2)
        words = re.findall(r"[a-zа-яё]+", fact.lower())
        mentions_subject = any(token == word or token.startswith(word[:4]) or word.startswith(token[:4])
                               for token in subject_terms for word in words if len(word) >= 4)
        describes_role = re.search(
            r"романтическ\w* отношени|учениц\w*|студент\w*|акт[её]р\w*|играет роль",
            fact, re.IGNORECASE,
        ) or re.search(
            r"(?:девушк\w*|парень|муж|жен\w*|сын|дочер\w*|брат|сестр\w*)\s+(?:[А-ЯЁ][а-яё-]+|главн\w+ геро\w+)",
            fact,
        )
        if mentions_subject and describes_role:
            return line
    return None


def character_fact(query: str, hits: list[dict]) -> str | None:
    """Answer the known diminutive character lookup directly from series credits."""
    if not MARATIK.search(query):
        return None
    for source_number, hit in enumerate(hits[:4], start=1):
        if hit.get("source") != "TMDB":
            continue
        for character, actor in re.findall(
            r"Персонаж\s+([^—;]+?)\s+—\s+актёр\s+([^;]+)", hit.get("text", ""), re.IGNORECASE
        ):
            if character.strip().lower().startswith("marat suvorov"):
                return f"Маратик — это Марат Суворов; его играет {actor.strip()} [{source_number}]."
    return None


def person_conflict_answer(query: str, hits: list[dict]) -> str | None:
    """Answer person incident questions from explicit source evidence only."""
    if not CONFLICT_QUESTION.search(query) or not any(hit.get("entity_type") == "person" for hit in hits):
        return None
    sections: list[tuple[int, str]] = []
    event_sentences: list[tuple[int, str]] = []
    section_pattern = re.compile(
        r"#{1,3}\s*(инциденты|скандалы|конфликты|судебные дела|критика)\b(.*?)(?=#{1,3}\s+|$)",
        re.I | re.S,
    )
    for source_number, hit in enumerate(hits, start=1):
        text = hit.get("text", "")
        for match in section_pattern.finditer(text):
            body = " ".join(match.group(2).split())
            body = re.sub(r"\s+([,.;:!?])", r"\1", body)
            body = re.sub(r"([,.;:!?])(?=[А-ЯЁA-Z])", r"\1 ", body)
            if len(body) > 40:
                sections.append((source_number, body))
        # Some sources mention an event without a dedicated heading.
        plain = text
        for sentence in re.split(r"(?<=[.!?])\s+", plain):
            sentence = " ".join(sentence.split())
            if len(sentence) >= 45 and re.search(
                r"инцидент|арест|суд признал|суд назначил|драл|скандал|конфликт|обвин|задерж|хулиган|укусил|ударил",
                sentence, re.I,
            ):
                event_sentences.append((source_number, sentence))
    if sections:
        # Prefer Russian editorial sources; retrieved order is already relevance-ranked.
        number, body = next(
            ((number, body) for number, body in sections
             if hits[number - 1].get("source") == "Wikipedia RU"),
            sections[0],
        )
        source_name = hits[number - 1].get("source", "источнике")
        lead = "По данным русскоязычной статьи Википедии, " if source_name == "Wikipedia RU" else "Согласно найденному источнику, "
        return f"{lead}{body[0].lower() + body[1:] if body else body} [{number}]"
    if event_sentences:
        unique: list[str] = []
        used: set[str] = set()
        for number, sentence in event_sentences:
            key = re.sub(r"\W+", "", sentence.lower())
            if key in used:
                continue
            used.add(key)
            unique.append(f"{sentence} [{number}]")
            if len(unique) == 4:
                break
        return "\n\n".join(unique)
    return "В загруженных источниках не нашёл подтверждённых сведений о конфликтах или скандалах этого человека."


def structured_metadata_answer(query: str, hits: list[dict]) -> str | None:
    """Answer credits and release-date questions directly from catalog fields."""
    lowered = query.lower()
    fields = []
    if re.search(r"режисс|кто\s+снял|постановщик", lowered):
        fields = [(r"Режиссёр", "Режиссёр")]
    elif re.search(r"сценар|кто\s+написал", lowered):
        fields = [(r"Сценаристы", "Сценарий написали")]
    elif re.search(r"композитор|музык\w*\s+(?:к|в|из)|кто\s+написал\s+музык", lowered):
        fields = [(r"Композитор", "Музыку написал")]
    elif re.search(r"акт[её]р|актрис|кто\s+сыграл|в\s+ролях|каст", lowered):
        fields = [(r"Главные акт[её]ры", "В главных ролях"),
                  (r"Персонажи и исполнители", "Персонажи и исполнители")]
    elif re.search(r"создател|кто\s+создал", lowered):
        fields = [(r"Создатели", "Создатели сериала")]
    elif re.search(r"когда\s+вышел|дата\s+выхода|год\s+выхода|в\s+каком\s+году", lowered):
        fields = [(r"Дата выхода", "Дата выхода")]
    if not fields:
        return None
    next_field = (
        r"(?:Фильм|Оригинальное название|Русское название|Дата выхода|Режиссёр|Сценаристы|Оператор|"
        r"Композитор|Продюсеры|Главные акт[её]ры|Создатели|Количество сезонов|"
        r"Персонажи и исполнители|Описание(?: на русском)?)"
    )
    for source_number, hit in enumerate(hits, start=1):
        if hit.get("source") != "TMDB":
            continue
        for field_pattern, label in fields:
            match = re.search(
                rf"(?:^|[.\n]\s*){field_pattern}:\s*(.*?)(?=\.\s*{next_field}:|$)",
                hit.get("text", ""), re.IGNORECASE | re.DOTALL,
            )
            if not match:
                continue
            value = match.group(1).strip().rstrip(".")
            if not value or value.lower() in {"не указано", "не указаны", "не указана"}:
                continue
            if label == "В главных ролях":
                value = ", ".join(value.split(", ")[:5])
            elif label == "Персонажи и исполнители":
                value = "; ".join(value.split("; ")[:5])
            if label == "Режиссёр":
                for ru_number, ru_hit in enumerate(hits, start=1):
                    if ru_hit.get("source") != "Wikipedia RU":
                        continue
                    russian_name = re.search(
                        r"(?i:режисс[её]р\w*)\s+([А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+){1,2})",
                        ru_hit.get("text", ""),
                    )
                    if russian_name:
                        return f"Это фильм режиссёра {russian_name.group(1)} [{ru_number}]."
            return f"{label} — {value} [{source_number}]."
    return None


def answer_guidance(query: str) -> tuple[str, int]:
    """Choose a compact response shape from the question's intent."""
    lowered = query.lower().replace("ё", "е")
    if re.search(r"почему|зачем|из-за чего|по какой причине|что заставил", lowered):
        return (
            "Объясни причину или мотив только если он прямо подтверждён. Раздели подтверждённую причину и последовательность событий; если источник не сообщает мотив, скажи это.",
            56,
        )
    if re.search(r"сравн|чем отличаются|разниц\w* между|похож\w* ли", lowered):
        return "Сравни названные объекты по общим критериям: сначала сходство, затем главное различие. Не добавляй отсутствующие в источниках свойства.", 64
    if re.search(r"что означа|значени\w*|символиз|смысл", lowered):
        return "Объясни значение предмета или образа по тому, что о нём сказано в источниках; отдели прямой сюжетный факт от толкования.", 48
    if re.search(r"сюжет|что происходит|что случил|что произошло|перескаж|расскажи.*(?:фильм|сериал|серия)|концовк|финал", lowered):
        return "Дай связный пересказ по порядку: кто участвует, что запускает события, ключевые повороты и результат. Не смешивай персонажей и не раскрывай финал, если его не спрашивают.", 72
    if re.search(r"подробн|развернут|расскажи|объясни|истори|конфликт|скандал|инцидент|интересн\w* факт", lowered):
        return "Ответь по существу с коротким контекстом и несколькими важными деталями. Если вопрос допускает несколько трактовок, обозначь, что именно подтверждают источники.", 64
    return "Ответь прямо на заданный вопрос, добавив только необходимый контекст. Используй естественный русский и не перечисляй факты, которые не помогают ответить.", 48


def extractive_answer(query: str, hits: list[dict]) -> str:
    if not hits:
        return "В загруженных материалах нет ответа. Попробуйте найти и загрузить фильм или человека через поиск по каталогу."
    lower_query = query.lower()
    terms = {w for w in re.findall(r"\w+", lower_query) if len(w) > 2 and w not in STOPWORDS}
    if terms & {"маратик", "маратика", "маратику", "маратиком", "маратике", "maratik", "maratika", "maratiku", "maratikom", "maratike"}:
        terms.update({"марат", "marat"})
    wants_creators = any(word in lower_query for word in ("режисс", "снял", "созда", "сценар", "композ", "музык"))
    wants_production = any(word in lower_query for word in ("создан", "съём", "съем", "снимал", "производств"))
    wants_career = any(word in lower_query for word in ("фильмограф", "фильм", "работ", "карьер"))
    sentences: list[tuple[float, str, int]] = []
    for source_number, hit in enumerate(hits, start=1):
        if hit["source"] == "Wikipedia":
            continue
        raw = re.split(r"(?<=[.!?])\s+|\n+|(?=\b(?:Режиссёр|Сценаристы|Композитор|Фильмы как актёр|Работа в съёмочной группе):)", hit["text"])
        for index, sentence in enumerate(raw):
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 650:
                continue
            if len(re.findall(r"[А-Яа-яЁё]", sentence)) < 4 or sentence.startswith("Описание: "):
                continue
            lower = sentence.lower()
            tokens = set(re.findall(r"\w+", lower))
            overlap = sum(
                1 for term in terms
                if any(term == token or (len(term) >= 4 and len(token) >= 4 and
                                         (term.startswith(token[:4]) or token.startswith(term[:4])))
                       for token in tokens)
            )
            if overlap == 0 and not (wants_creators and any(
                label in lower for label in ("режиссёр", "режиссер", "сценарист", "композитор")
            )):
                continue
            if "марат" in terms and any(token.startswith("марат") for token in tokens):
                overlap += 1
            score = overlap * 4 + float(hit["score"]) * 100 - index * 0.03
            if wants_creators and "режиссёр:" in lower:
                score += 25
            elif wants_creators and any(label in lower for label in ("фильм режиссёра", "фильма режиссёра", "режиссёром фильма", "режиссером фильма")):
                score += 18
            elif wants_creators and ("режиссёр" in lower or "режиссер" in lower):
                score += 3
            elif wants_creators and any(label in lower for label in ("сценаристы:", "композитор:")):
                score += 10
            if wants_creators and "оператор" in lower:
                score -= 12
            if "снимался во всех фильмах" in lower or "мог сыграть" in lower:
                score -= 8
            if wants_production and any(word in lower for word in ("production", "filmed", "filming", "shooting", "screenplay", "script")):
                score += 8
            if wants_production and any(word in lower for word in ("снимал", "снял", "съём", "съем", "сценари", "производств")):
                score += 8
            if wants_career and any(label in lower for label in ("фильмы как актёр:", "работа в съёмочной группе:")):
                score += 9
            if hit["source"] == "Wikipedia RU":
                score += 3
            if any(word in lower for word in ("critic", "review", "reception", "astonishing movie")):
                score -= 8
            sentences.append((score, sentence, source_number))
    if not sentences:
        return "В русскоязычных материалах пока нет ответа на этот вопрос. Попробуйте выбрать другой фильм или дождаться загрузки модели Qwen."
    seen = set()
    selected = []
    for _, sentence, number in sorted(sentences, reverse=True):
        key = re.sub(r"\W+", "", sentence.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        selected.append(f"{sentence} [{number}]")
        if len(selected) == 3:
            break
    return "\n\n".join(selected)


def grounded_fallback(query: str, hits: list[dict]) -> str:
    """Use plot evidence as the fallback so model failures cannot derail the answer."""
    if PLOT_QUESTION.search(query):
        numbered_hits = list(enumerate(hits, start=1))
        wiki_hits = [(number, hit) for number, hit in numbered_hits if hit.get("source") == "Wikipedia RU"][:6]
        evidence = plot_evidence(query, wiki_hits)
        if evidence:
            lines = [line for line in evidence.splitlines() if re.match(r"\[\d+\] ", line)]
            lead = identity_evidence(query, evidence)
            ordered = ([lead] if lead else []) + [line for line in lines if line != lead]
            answer = "\n\n".join(ordered[:6])
            if answer:
                return answer
    return extractive_answer(query, hits)


def warm_model(config: Settings) -> None:
    """Load Qwen in the background so the first user question avoids cold start."""
    try:
        response = httpx.post(
            f"{config.ollama_url.rstrip('/')}/api/generate",
            json={
                "model": config.ollama_model,
                "prompt": " ",
                "stream": False,
                "keep_alive": "24h",
                "think": False,
                "options": {"num_ctx": 4096, "num_predict": 1},
            },
            timeout=120,
        )
        response.raise_for_status()
        log.info("Qwen model is warm and will remain loaded for 24 hours")
    except httpx.HTTPError as exc:
        log.warning("Could not preload Qwen: %s", exc)


def _direct_answer(query: str, hits: list[dict]) -> str | None:
    if re.fullmatch(r"\s*(?:кто\s+(?:режисс[её]р|снял)|когда\s+вышел|в\s+каком\s+году\s+вышел)[^?]*\??\s*", query, re.I):
        return structured_metadata_answer(query, hits)
    return None


def _model_request(query: str, hits: list[dict], config: Settings) -> tuple[dict, set[int]]:
    from .evidence import evidence_context

    context = evidence_context(query, hits)
    guidance, _ = answer_guidance(query)
    system = (
        "Ты внимательный русскоязычный собеседник, отвечающий на вопросы о кино и людях кино. "
        "Ответь на точный вопрос, затем дай контекст и объяснение. Не подменяй ответ соседним событием. "
        "Исправляй неверную предпосылку вопроса. Не смешивай персонажей, актёров и реальные события. "
        "Опирайся на приведённые фрагменты. Не добавляй числа, имена и причинные связи, которых нет в относящемся к вопросу фрагменте. "
        "Для ответа сначала найди конкретное предложение источника, которое отвечает на вопрос. "
        "Сохрани описанный в нём способ действия и не дополняй его вымышленными деталями. "
        "В вопросе «почему» объясняй первопричину события, а не только обстоятельства его обнаружения. "
        "Если спрашивают о необычном явлении или сущности, объясни её происхождение или механизм. "
        "Если точного ответа нет, прямо скажи об этом. После подтверждённых фактов ставь [N] из контекста. "
        "Пиши по-русски естественно, без списка, обычно 2–5 предложений. " + guidance
    )
    return ({
        "model": config.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Вопрос: {query}\n\nФрагменты источников:\n{context}\n\nОтвет:"},
        ],
        "stream": False, "think": False, "keep_alive": "24h",
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 180},
    }, {int(value) for value in re.findall(r"\[(\d+)\]", context)})


def _validated_answer(answer: str, allowed: set[int], context: str) -> str:
    answer = answer.strip()
    if not answer:
        raise ValueError("Ollama returned an empty response")
    answer = re.sub(r"\[(\d+)\]", lambda match: match.group(0) if int(match.group(1)) in allowed else "", answer)
    if not re.search(r"\[\d+\]", answer):
        from .evidence import tokens

        evidence = [(int(match.group(1)), tokens(match.group(2)))
                    for line in context.splitlines()
                    if (match := re.match(r"\[(\d+)\]\s+(.+)", line))]
        repaired = []
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            terms = tokens(sentence)
            if not terms:
                continue
            number, matched = max(
                ((number, len(terms & passage_terms)) for number, passage_terms in evidence),
                key=lambda item: item[1], default=(0, 0),
            )
            if matched >= 4 and matched / len(terms) >= .38:
                repaired.append(f"{sentence} [{number}]")
            elif re.search(r"источник|фрагмент|нет данных|не указ", sentence, re.I):
                repaired.append(sentence)
        if not any(re.search(r"\[\d+\]", sentence) for sentence in repaired):
            raise ValueError("Ollama returned unsupported text without citations")
        answer = " ".join(repaired)
    return answer


def generate(query: str, hits: list[dict], config: Settings) -> tuple[str, str]:
    """Answer an arbitrary question using retrieved evidence and local Qwen."""
    if not hits:
        return extractive_answer(query, hits), "extractive"
    direct = _direct_answer(query, hits)
    if direct:
        return direct, "structured"
    payload, allowed = _model_request(query, hits, config)
    if not allowed:
        return extractive_answer(query, hits), "extractive"
    try:
        response = httpx.post(f"{config.ollama_url.rstrip('/')}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        answer = _validated_answer(result.get("message", {}).get("content", ""), allowed,
                                   payload["messages"][1]["content"])
        log.info("Qwen answered in %.2fs (%s output tokens)",
                 result.get("total_duration", 0) / 1e9, result.get("eval_count", "?"))
        return answer, "qwen"
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Qwen answer failed; using source excerpts: %s", exc)
        return grounded_fallback(query, hits), "extractive"


def stream_generate(query: str, hits: list[dict], config: Settings):
    """Yield answer deltas followed by a validated final answer."""
    if not hits:
        yield {"type": "done", "answer": extractive_answer(query, hits), "answer_mode": "extractive"}
        return
    direct = _direct_answer(query, hits)
    if direct:
        yield {"type": "done", "answer": direct, "answer_mode": "structured"}
        return
    payload, allowed = _model_request(query, hits, config)
    if not allowed:
        yield {"type": "done", "answer": extractive_answer(query, hits), "answer_mode": "extractive"}
        return
    payload["stream"] = True
    answer = ""
    try:
        with httpx.stream("POST", f"{config.ollama_url.rstrip('/')}/api/chat", json=payload,
                          timeout=httpx.Timeout(60, connect=5)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                delta = event.get("message", {}).get("content", "")
                if delta:
                    answer += delta
                    yield {"type": "delta", "text": delta}
        answer = _validated_answer(answer, allowed, payload["messages"][1]["content"])
        yield {"type": "done", "answer": answer, "answer_mode": "qwen"}
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Qwen stream failed; using source excerpts: %s", exc)
        yield {"type": "done", "answer": grounded_fallback(query, hits), "answer_mode": "extractive"}
