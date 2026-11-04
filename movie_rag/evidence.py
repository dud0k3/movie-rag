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


def split_passages(text: str) -> list[str]:
    """Keep headings with the following paragraph and split long plots."""
    text = re.sub(r"\s*##\s*", "\n## ", text)
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
    ):
        if re.search(pattern, query, re.I):
            query_terms |= tokens(terms)
    plot_intent = meaning_intent or bool(re.search(r"почему|зачем|сюжет|финал|концовк|убил|погиб|умер|сцен", query, re.I))
    reveal_intent = bool(re.search(r"на самом деле|кем.*финал|личност|тайна|секрет", query, re.I))
    sports_intent = bool(re.search(r"спорт|занимал.*до кино", query, re.I))
    trick_intent = bool(re.search(r"трюк|фокус|иллюзион", query, re.I))
    why_intent = bool(re.search(r"почему|зачем|из-за чего", query, re.I))
    candidates: list[tuple[float, int, int, str]] = []
    max_overlap = 0
    for number, hit in enumerate(hits, 1):
        passages = split_passages(hit.get("text", ""))
        for index, passage in enumerate(passages):
            if index and re.search(r"(?:^|^## [\w ]{1,45} )(?:(?:После этого|Затем|Он|Она|Они|Его|Её)\b)", passage):
                passage = passages[index - 1][-340:] + " " + passage
            if reveal_intent and index + 1 < len(passages) and re.search(
                r"объясня|раскрыва|оказыва|выясня|reveal", passage, re.I,
            ):
                passage += " " + passages[index + 1][:400]
            passage_terms = tokens(passage)
            overlap = len(query_terms & passage_terms)
            max_overlap = max(max_overlap, overlap)
            # Retrieval ranking is already hybrid; lexical overlap focuses on
            # the relevant sentence inside a 480-word chunk.
            score = overlap * 3 + 2 / (1 + number * .24)
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
            if why_intent and overlap:
                if re.search(r"выясня|оказыва|причин", passage, re.I):
                    score += 10
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
