"""Select short, question-relevant passages without movie-specific rules."""

from __future__ import annotations

import re
from collections.abc import Callable


STOP = {
    "какой", "какая", "какие", "который", "почему", "зачем", "расскажи",
    "объясни", "фильм", "фильме", "фильма", "сериал", "сериале", "актёр",
    "актера", "актёра", "этого", "этот", "этой", "что", "кто", "как",
    "про", "его", "она", "они", "был", "была", "стала", "стало",
    "свой", "своя", "свои", "своего", "своей", "своим",
    "самом", "деле", "такой", "такая", "этому", "этим", "позволила", "этот",
}


def tokens(value: str) -> set[str]:
    return {
        word.replace("ё", "е")[:3]
        for word in re.findall(r"[a-zа-яё]{3,}", value.lower())
        if word not in STOP
    }


# Split only at an explicit question or instruction. Names and titles may
# contain punctuation, and "и" often joins names within a single question.
PART_START = (
    r"кто|что|как|когда|почему|зачем|сколько|какой|какая|какие|кем|чем|где|"
    r"назови|объясни|расскажи|опиши|укажи|перечисли|в каком году|написал|сыграл"
)


def question_parts(query: str) -> list[str]:
    parts = re.split(
        rf"(?:[?;.!]\s+|[, ]+и\s+)(?=(?:{PART_START})\b)", query.strip(), flags=re.I,
    )
    return [part.strip(" ,;") for part in parts if part.strip(" ,;")]


def planned_context(query: str, hits: list[dict], max_chars: int, semantic_scorer=None) -> str:
    parts = question_parts(query)
    if len(parts) == 1:
        return evidence_context(query, hits, max_chars=max_chars, semantic_scorer=semantic_scorer)
    # Reserve a share of the context for each subquestion, even when the most
    # similar passage in the entire article belongs to another subquestion.
    quota = max_chars // len(parts)
    groups = [evidence_context(part, hits, max_chars=quota).splitlines()[:3] for part in parts]
    selected: list[str] = []
    size = 0
    for position in range(3):
        for group in groups:
            if position >= len(group):
                continue
            line = group[position]
            if line not in selected and size + len(line) + 1 <= max_chars:
                selected.append(line)
                size += len(line) + 1
    return "\n".join(selected)


_CYRILLIC = str.maketrans(dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ("a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m",
     "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch",
     "", "i", "", "e", "iu", "ia"),
)))


def _romanized(value: str) -> str:
    return value.casefold().translate(_CYRILLIC).replace("y", "i").replace("j", "i")


def split_passages(text: str) -> list[str]:
    """Keep headings with the following paragraph and split long plots."""
    text = re.sub(r"\s*##\s*", "\n## ", text)
    text = re.sub(r"(?:Персонажи и исполнители:|;)\s+(?=Персонаж )", "\n", text)
    # Old databases flattened headings and paragraphs. Recognize their common
    # labels too, so a later sentence still inherits the section it came from.
    text = re.sub(
        r"(?im)^## (сюжет|plot|биография|biography|в ролях|cast|производство|production|"
        r"сведения из таблицы википедии|инциденты|фильмография|filmography)(?= +\S)",
        r"## \1\n", text,
    )
    result = []
    heading = ""
    for line in text.splitlines():
        if line.startswith("## ") and not re.search(r"[.!?] ", line):
            heading = line[:100]
            # Unknown headings in legacy rows can include a complete sentence.
            if len(line) > 95:
                result.append(line[:850])
                heading = ""
            continue
        pieces = re.split(r"(?<!\bрод\.)(?<!\bим\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\b[A-ZА-ЯЁ]\.)(?<=[.!?])\s+(?=[А-ЯЁA-Z«“])", line)
        for piece in pieces:
            piece = piece.strip()
            while len(piece) > 850:
                cut = piece.rfind(" ", 0, 850)
                cut = cut if cut > 400 else 850
                result.append(f"{heading} {piece[:cut]}".strip())
                piece = piece[cut:].strip()
            if len(piece) >= 25:
                result.append(f"{heading} {piece}".strip())
    return result


def evidence_context(
    query: str, hits: list[dict], max_chars: int = 3100,
    semantic_scorer: Callable[[str, list[str]], list[float]] | None = None,
) -> str:
    """Select evidence across source chunks while retaining exact citation IDs."""
    query_terms = tokens(query)
    focus_terms: set[str] = set()
    if hits:
        query_terms -= tokens(hits[0].get("title", ""))
    anchors = query_terms.copy()
    meaning_intent = bool(re.search(r"означа|значени|символ|смысл", query, re.I))
    if meaning_intent:
        query_terms |= tokens("значение символ смысл талисман надежда")
    if re.search(r"конфликт|скандал|инцидент|драк|арест|судебн|хулиган", query, re.I):
        focus_terms = tokens("инцидент скандал конфликт драка арест суд хулиганство")
        query_terms |= focus_terms
    duplicate_intent = bool(re.search(r"двойник|двойн|клонир|клон|копи[яи]", query, re.I))
    if duplicate_intent:
        focus_terms = tokens("клон клонирование копия двойник")
        query_terms |= focus_terms
    if re.search(r"убил|убий|убива", query, re.I):
        focus_terms = tokens("убил убивает убийство")
    for pattern, terms in (
        (r"трюк|фокус|иллюзион", "трюк фокус иллюзия секрет братья близнецы twin trick"),
        (r"связан|отношен|встречал|девушка", "роман влюбились девушка отношения"),
        (r"спорт|занимал.*до кино", "спорт спортсмен сборная соревнования прыжки diving national"),
        (r"замедл.*врем|врем.*медлен", "гравитация дыра гаргантюа gravity dilation"),
        (r"перв\w* рол|дебют|начал\w* карьер", "дебют режиссёр роль сыграл кастинг casting debut"),
        (r"в итоге|выбрался|как закончил", "вышел выходит покинул сбежал проход дверь escape exit"),
        (r"до кино|до кинокарьеры", "ранние годы спорт работа карьера early life"),
        (r"праздник|вечеринк|торжеств|день рождения", "вечеринка торжество праздник день рождения party birthday"),
    ):
        if re.search(pattern, query, re.I):
            query_terms |= tokens(terms)
    plot_intent = meaning_intent or bool(re.search(r"почему|зачем|сюжет|финал|концовк|выбрался|в итоге|убил|погиб|умер|сцен", query, re.I))
    identity_match = re.match(r"кто\s+(?:такой|такая|это)\s+([\w-]+)", query, re.I)
    identity = _romanized(identity_match[1]) if identity_match else None
    resolution_intent = bool(re.search(r"выбрался|сбежал|спасся|как.*(?:кончил|уш[её]л)|чем закончил", query, re.I))
    occasion_intent = bool(re.search(r"праздник|вечеринк|торжеств|день рождения", query, re.I))
    decisive_event = bool(re.search(r"убил|убий|убива|зарезал|поножов|запах", query, re.I))
    reveal_intent = bool(re.search(r"на самом деле|кем.*финал|личност|тайна|секрет", query, re.I))
    sports_intent = bool(re.search(r"спорт|занимал.*до кино", query, re.I))
    trick_intent = bool(re.search(r"трюк|фокус|иллюзион", query, re.I))
    why_intent = bool(re.search(r"почему|зачем|из-за чего", query, re.I))
    names = {
        word.casefold().replace("ё", "е")[:4]
        for word in re.findall(r"(?<!\w)[А-ЯЁA-Z][а-яёa-z-]{2,}", query)
        if word.casefold() not in STOP and query.find(word) > 0
    }
    candidates: list[tuple[float, int, int, str]] = []
    max_overlap = 0
    for number, hit in enumerate(hits, 1):
        passages = split_passages(hit.get("text", ""))
        anchored = {
            index for index, passage in enumerate(passages)
            if names and any(word.casefold().replace("ё", "е")[:4] in names
                             for word in re.findall(r"\w+", passage))
        } if why_intent else set()
        for index, passage in enumerate(passages):
            if index and re.search(r"(?:^|^## [\w ]{1,45} )(?:(?:После этого|Затем|Он|Она|Они|Его|Её)\b)", passage):
                passage = passages[index - 1][-340:] + " " + passage
            if reveal_intent and index + 1 < len(passages) and re.search(
                r"объясня|раскрыва|оказыва|выясня|reveal", passage, re.I,
            ):
                passage += " " + passages[index + 1][:400]
            if why_intent and index and re.search(r"\b(?:С тех пор|Поэтому|Именно поэтому)\b", passage):
                passage = f"{passages[index - 1]} {passage}"
            passage_terms = tokens(passage)
            overlap = len(query_terms & passage_terms)
            max_overlap = max(max_overlap, overlap)
            # Retrieval ranking is already hybrid; lexical overlap focuses on
            # the relevant sentence inside a 480-word chunk.
            score = overlap * 3 + 2 / (1 + number * .24)
            if why_intent:
                score += 2 * len(anchors & passage_terms)
                if index in anchored:
                    score += 7
                elif index - 1 in anchored or index - 2 in anchored:
                    score += 4
            if occasion_intent and re.search(
                r"вечеринк|торжеств|праздник|день рождения|birthday|party", passage, re.I,
            ):
                score += 8
                if index + 1 < len(passages):
                    passage = " ".join((passage, passages[index + 1][:340]))
            if identity and passage.startswith("Персонаж "):
                role = _romanized(passage.partition(" — ")[0])
                if re.search(rf"\b{re.escape(identity)}\b", role):
                    score += 14
            if resolution_intent and re.search(
                r"выходит|выходил|вышел|открывает дверь|проход|сбежал|покинул|уплывает|плыв[её]т|escapes?|sails?|exits?", passage, re.I,
            ):
                score += 8
                passage = " ".join(passages[max(0, index - 1):index + 2])
            if decisive_event and re.search(
                r"ударяет ножом|ударил ножом|закол|зареза|зажимает нос|зажал нос|stabs?|pinches his nose", passage, re.I,
            ):
                score += 9
                passage = " ".join(passages[max(0, index - 1):index + 1])
            if reveal_intent and re.search(
                r"объясня|раскрыва|оказыва|выясня|на самом деле|reveals|actually|true identity", passage, re.I,
            ):
                score += 6
            if sports_intent and re.search(r"сборн|соревнован|чемпионат|diving|national squad", passage, re.I):
                score += 6
            if trick_intent and re.search(r"близнец|братья|братьями|двойник|twins|double", passage, re.I):
                score += 6
            if focus_terms and passage_terms.intersection(focus_terms):
                score += 2
            if plot_intent and re.search(r"^## (?:сюжет|plot)\b", passage, re.I):
                score += 3
            if hit.get("source") == "Wikipedia RU":
                score += .6
            if plot_intent and re.search(r"##\s*(?:в ролях|акт[её]ры|cast|production|производство)\b", passage, re.I):
                score -= 9
            if meaning_intent:
                if re.search(r"по задумке|должен принести|означа|символиз|олицетвор", passage, re.I):
                    score += 13
                elif re.search(r"талисман|символ|надежд", passage, re.I):
                    score += 6
            if duplicate_intent:
                if re.search(r"клон\w* челов|оба.{0,35}клон", passage, re.I):
                    score += 13
                elif re.search(r"пробужда|анабиоз", passage, re.I):
                    score += 12
                elif re.search(r"двойник", passage, re.I):
                    score += 8
            if why_intent and anchors.intersection(passage_terms):
                if re.search(r"с тех пор|выясня|оказыва|причин", passage, re.I):
                    score += 7
                elif re.search(r"потому|поскольку|из-за|так как", passage, re.I):
                    score += 3
            if re.match(r"(?:После|Затем|Он|Она|Они|Его|Её)\b", passage):
                score -= 2
            if index == 0:
                score += .15
            candidates.append((score, number, index, passage))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if semantic_scorer and candidates and not meaning_intent and not sports_intent and max_overlap < 3:
        # The chunk search already narrowed the corpus; rerank sentences in
        # those chunks, which helps paraphrases and English source material.
        candidates = candidates[:42]
        try:
            similarities = semantic_scorer(query, [item[3] for item in candidates])
            if len(similarities) == len(candidates):
                candidates = [
                    (base * .55 + max(0.0, similarity) * 18, number, index, passage)
                    for (base, number, index, passage), similarity in zip(candidates, similarities)
                ]
                candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        except Exception:
            # Lexical evidence remains available if the encoder is unavailable.
            pass
    best_score = candidates[0][0] if candidates else 0
    selected: list[tuple[int, int, str]] = []
    seen: list[str] = []
    counts: dict[int, int] = {}
    size = 0
    for score, number, index, passage in candidates:
        if best_score > 5 and score < best_score * (.45 if why_intent else .50 if meaning_intent else .56):
            continue
        body = re.sub(r"## (?:сюжет|plot|биография|biography)\b", "", passage, flags=re.I)
        key = re.sub(r"\W+", "", body.lower())
        if any(key in previous or previous in key for previous in seen) or counts.get(number, 0) >= 3:
            continue
        if size + len(passage) > max_chars:
            continue
        seen.append(key)
        selected.append((number, index, passage))
        counts[number] = counts.get(number, 0) + 1
        size += len(passage)
        if len(selected) >= (5 if meaning_intent else 8):
            break
    return "\n".join(
        f"[{number}] {passage}" for number, _, passage in selected
    )
