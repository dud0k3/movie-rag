"""Grounded generation and deterministic extractive fallback."""

from __future__ import annotations

import re
import httpx

from .config import Settings


STOPWORDS = {"что", "кто", "как", "где", "когда", "какой", "какая", "какие", "фильм", "фильма", "про", "это", "the", "was", "who", "and"}
MARATIK = re.compile(r"\b(?:маратик\w*|maratik\w*)\b", re.IGNORECASE)
PLOT_QUESTION = re.compile(
    r"сюжет|почему|зачем|что (?:произошло|случилось|стало)|"
    r"концовк|финал|чем законч|умер|погиб|убил|спрыг|персонаж|героин|геро[йя]",
    re.IGNORECASE,
)
QUESTION_FILLER = STOPWORDS | {
    "почему", "зачем", "спрыгнула", "спрыгнул", "сюжет", "персонаж", "герой", "героиня",
    "расскажи", "объясни", "происходит", "произошло", "случилось", "такой", "такая",
    "такое", "фильм", "сериал", "конце", "финал", "концовка",
}


def plot_evidence(query: str, numbered_hits: list[tuple[int, dict]]) -> str | None:
    """Reduce retrieved plot chunks to atomic, source-backed statements."""
    query_tokens = [
        token for token in re.findall(r"[a-zа-яё]{3,}", query.lower())
        if token not in QUESTION_FILLER
    ]
    if not query_tokens:
        return None
    facts: list[tuple[int, int, str]] = []
    for source_number, hit in numbered_hits:
        parts = re.split(r"(?<=[.!?])\s+|\s*\|\s*", hit.get("text", ""))
        matched = []
        for index, part in enumerate(parts):
            words = re.findall(r"[a-zа-яё]+", part.lower())
            if any(token == word or token.startswith(word[:4]) or word.startswith(token[:4])
                   for token in query_tokens for word in words if len(word) >= 4):
                matched.append(index)
        keep = {neighbor for index in matched for neighbor in (index - 1, index, index + 1)
                if 0 <= neighbor < len(parts)}
        source_facts = []
        for index in sorted(keep):
            fact = " ".join(parts[index].split()).strip(" -|\t")
            if len(fact) >= 35 and not fact[:1].isdigit():
                source_facts.append((source_number, index, fact))
        # Give each retrieved plot passage room; otherwise the first long
        # chunk can crowd out the ending or the reason asked about.
        facts.extend(source_facts[:10])
    if not facts:
        return None
    seen = set()
    lines = []
    for source_number, _, fact in facts:
        key = re.sub(r"\W+", "", fact.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{source_number}] {fact}")
        if len(lines) >= 24:
            break
    return "Факты сюжета из источников. Сохраняй, кто именно совершает каждое действие:\n" + "\n".join(lines)


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
                "options": {"temperature": 0.25, "num_ctx": 8192, "num_predict": 700},
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
                            "или стал жертвой её нападения. Не меняй персонажа, совершившего действие. "
                            "Не приписывай персонажам мысли, чувства, вину или мотивы, если это прямо не сказано. "
                            "Если факт не подтверждается списком, удали его. Пиши по-русски, естественно и содержательно. "
                            "Ставь ссылку [номер] после каждого предложения с фактами. "
                            "Сохрани ссылки [номер] у подтверждённых фактов. Выведи только исправленный ответ."
                        )},
                        {"role": "user", "content": f"{context}\n\nЧерновик:\n{answer}"},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.05, "num_ctx": 8192, "num_predict": 450},
                }, timeout=90,
            )
            review.raise_for_status()
            reviewed = review.json().get("message", {}).get("content", "").strip()
            review_citations = [int(value) for value in re.findall(r"\[(\d+)\]", reviewed)]
            if reviewed and review_citations and all(value in allowed_citations for value in review_citations):
                answer = reviewed
        return answer, "qwen"
    except (httpx.HTTPError, ValueError):
        return extractive_answer(query, hits), "extractive"
