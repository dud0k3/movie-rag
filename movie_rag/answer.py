"""Grounded generation and deterministic extractive fallback."""

from __future__ import annotations

import re
import logging
import httpx

from .config import Settings

log = logging.getLogger(__name__)


STOPWORDS = {"что", "кто", "как", "где", "когда", "какой", "какая", "какие", "фильм", "фильма", "про", "это", "the", "was", "who", "and"}
MARATIK = re.compile(r"\b(?:маратик\w*|maratik\w*)\b", re.IGNORECASE)
PLOT_QUESTION = re.compile(
    r"сюжет|почему|зачем|что (?:произошло|случилось|стало)|"
    r"концовк|финал|чем законч|умер|погиб|убил|спрыг|персонаж|героин|геро[йя]|"
    r"кто\s+(?:такой|такая|такое|это)|кем\s+(?:является|приходится)",
    re.IGNORECASE,
)
WHO_QUESTION = re.compile(r"кто\s+(?:такой|такая|такое|это)|кем\s+(?:является|приходится)", re.IGNORECASE)
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
    if not query_tokens:
        return None
    facts: list[tuple[int, int, int, str]] = []
    for source_number, hit in numbered_hits:
        parts = re.split(r"(?<=[.!?])\s+|\s*\|\s*", hit.get("text", ""))
        matched = []
        for index, part in enumerate(parts):
            words = [word.replace("ё", "е") for word in re.findall(r"[a-zа-яё]+", part.lower())]
            if any(token == word or token.startswith(word[:4]) or word.startswith(token[:4])
                   for token in query_tokens for word in words if len(word) >= 4):
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
                    2 if any(token == word or token.startswith(word[:4]) or word.startswith(token[:4])
                             for word in words_normalized if len(word) >= 4) else 0
                    for token in query_tokens
                )
                source_facts.append((relevance, source_number, index, fact))
        # Give each retrieved plot passage room; otherwise the first long
        # chunk can crowd out the ending or the reason asked about.
        facts.extend(source_facts)
    if not facts:
        return None
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
        if len(lines) >= 8:
            break
    return "Факты сюжета из источников. Сохраняй, кто именно совершает каждое действие:\n" + "\n".join(lines)


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
            overlap = len(terms & tokens)
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


def generate(query: str, hits: list[dict], config: Settings) -> tuple[str, str]:
    direct_answer = character_fact(query, hits)
    if direct_answer:
        return direct_answer, "structured"
    if not hits:
        return extractive_answer(query, hits), "extractive"
    numbered_hits = list(enumerate(hits, start=1))
    # Plot articles usually contain far more detail than TMDb's one-paragraph
    # synopsis. Give those focused questions several plot chunks when available.
    if PLOT_QUESTION.search(query) and any(hit.get("source") == "Wikipedia RU" for hit in hits):
        preferred = [(number, hit) for number, hit in numbered_hits if hit.get("source") == "Wikipedia RU"]
        numbered_context = preferred[:4]
    else:
        numbered_context = numbered_hits[:4]
    evidence = plot_evidence(query, numbered_context) if PLOT_QUESTION.search(query) else None
    context = evidence or "\n\n".join(
        f"[{index}] Источник: {hit['title']} ({hit['url']})\n{hit['text'][:4200]}"
        for index, hit in numbered_context
    )
    system = (
        "Ты помощник по истории кино. Отвечай только на русском языке, даже если источник на английском. "
        "Для ответа о сюжете сначала внимательно сверь подробные пересказы серий; они важнее кратких карточек каталога. "
        "Если источники не раскрывают нужную деталь, можешь осторожно дополнить ответ своими знаниями, но обозначь неуверенность. "
        "В контексте группировки слово «отшить» означает исключить человека из группировки, а не избить или расправиться с ним. "
        "Не додумывай мотивы, угрозы, принадлежность героя к группировке и другие факты, которых нет в тексте источника. "
        "Если принадлежность или причина поступка не названа прямо, не приписывай её персонажу. Точно сохраняй родство, отношения и последовательность событий. "
        "Сохраняй точного действующего персонажа: не меняй местами, кто напал, спас, убил или сказал. "
        "Не превращай угрозу исключения из группировки в угрозу убийством. Не цитируй английские предложения без перевода. "
        "Ставь ссылку [номер] рядом с фактами, подтверждёнными источником; не приписывай источнику то, чего в нём нет. "
        "Сначала ответь прямо; если спрашивают, кто персонаж, в первом предложении назови его роль или связь с героями по источнику. "
        "На вопрос о сюжете дай содержательное объяснение причин и последовательности событий, обычно 5–8 предложений, "
        "сохраняя важные детали и предупреждая о спойлерах, если раскрываешь развязку. На простой вопрос отвечай короче. "
        "Не смешивай персонажей и сюжетные линии. Не добавляй сведения о создании фильма, если об этом не спросили. "
        "Перед отправкой проверь, что каждое предложение подтверждается источниками и не противоречит им. "
        "Пиши естественно и ясно, как собеседник, а не как справочник."
    )
    try:
        response = httpx.post(
            f"{config.ollama_url.rstrip('/')}/api/chat",
            json={
                "model": config.ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Вопрос: {query}\n\nИсточники:\n{context}\n\nОтветь по-русски; укажи [номер] после каждого факта."},
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.25,
                    "num_ctx": 4096 if evidence else 8192,
                    "num_predict": 520 if evidence else 650,
                },
            }, timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise ValueError("Ollama returned an empty response")
        cited = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
        allowed_citations = {index for index, _ in numbered_context}
        if not cited or any(value not in allowed_citations for value in cited):
            raise ValueError("Ollama returned an answer without valid source markers")
        if evidence:
            # A second, short grounding pass catches role swaps and invented
            # relationships that a small local model can add while narrating.
            review = httpx.post(
                f"{config.ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": config.ollama_model,
                    "messages": [
                        {"role": "system", "content": (
                            "Ты редактор-верификатор ответов о сюжете. Сверь черновик с фактами. "
                            "Исправь неверные роли, причины и неподтверждённые утверждения. "
                            "Не выводи принадлежность героя к группировке из того, что он общается с её членами "
                            "или стал жертвой её нападения. Не меняй персонажа, совершившего действие, узнавшего новость или испытавшего событие. "
                            "Слово «отшить» в контексте группировки означает исключение из неё, не физическую расправу. "
                            "Не приписывай персонажам мысли, чувства, вину или мотивы, если это прямо не сказано. "
                            "Не повторяй один факт разными словами. Если факт не подтверждается списком, удали его. "
                            "Пиши по-русски, естественно и содержательно. "
                            "Ставь ссылку [номер] после каждого предложения с фактами. "
                            "Сохрани ссылки [номер] у подтверждённых фактов. Выведи только исправленный ответ."
                        )},
                        {"role": "user", "content": f"{context}\n\nЧерновик:\n{answer}"},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.05, "num_ctx": 4096, "num_predict": 220},
                }, timeout=90,
            )
            review.raise_for_status()
            reviewed = review.json().get("message", {}).get("content", "").strip()
            review_citations = [int(value) for value in re.findall(r"\[(\d+)\]", reviewed)]
            if reviewed and review_citations and all(value in allowed_citations for value in review_citations):
                answer = reviewed
        lead = identity_evidence(query, context) if evidence else None
        if lead and not re.search(r"романтическ|возлюблен|учениц|студент|девушк|сын|дочер|брат|сестр", answer, re.IGNORECASE):
            fact = re.sub(r"^\[\d+\]\s*", "", lead)
            number = re.match(r"^\[(\d+)\]", lead).group(1)
            answer = f"{fact} [{number}]\n\n{answer}"
        return answer, "qwen"
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Qwen answer failed; using grounded fallback: %s", exc)
        return grounded_fallback(query, hits), "extractive"
