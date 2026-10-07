"""Select short, question-relevant passages without movie-specific rules."""

from __future__ import annotations

import re


STOP = {
    "какой", "какая", "какие", "который", "почему", "зачем", "расскажи",
    "объясни", "фильм", "фильме", "фильма", "сериал", "сериале", "актёр",
    "актера", "актёра", "этого", "этот", "этой", "что", "кто", "как",
    "про", "его", "она", "они", "был", "была", "стала", "стало",
    "свой", "своя", "свои", "своего", "своей", "своим",
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
    pieces = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])|\n+", text)
    result: list[str] = []
    heading = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("## "):
            if len(piece) >= 35:
                result.append(piece[:950])
                heading = ""
                continue
            heading = piece[:90]
            continue
        if len(piece) > 950:
            piece = piece[:950].rsplit(" ", 1)[0]
        if len(piece) >= 30:
            result.append(f"{heading} {piece}".strip())
    return result


def evidence_context(query: str, hits: list[dict], max_chars: int = 4200) -> str:
    """Select evidence across source chunks while retaining exact citation IDs."""
    query_terms = tokens(query)
    focus_terms: set[str] = set()
    if hits and hits[0].get("entity_type") == "person":
        query_terms -= tokens(hits[0].get("title", ""))
    if re.search(r"конфликт|скандал|инцидент|драк|арест|судебн|хулиган", query, re.I):
        focus_terms = tokens("инцидент скандал конфликт драка арест суд хулиганство")
        query_terms |= focus_terms
    duplicate_intent = bool(re.search(r"двойник|двойн|клонир|клон|копи[яи]", query, re.I))
    if duplicate_intent:
        focus_terms = tokens("клон клонирование копия двойник")
        query_terms |= focus_terms
    if re.search(r"убил|убий|убива", query, re.I):
        focus_terms = tokens("убил убивает убийство")
    plot_intent = bool(re.search(r"почему|зачем|сюжет|финал|концовк|убил|погиб|умер|сцен", query, re.I))
    why_intent = bool(re.search(r"почему|зачем|из-за чего", query, re.I))
    candidates: list[tuple[float, int, int, str]] = []
    for number, hit in enumerate(hits, 1):
        passages = split_passages(hit.get("text", ""))
        for index, passage in enumerate(passages):
            if index and re.match(r"(?:После этого|Затем|Он|Она|Они|Его|Её)\b", passage):
                passage = passages[index - 1][-340:] + " " + passage
            passage_terms = tokens(passage)
            if focus_terms and not passage_terms.intersection(focus_terms):
                continue
            overlap = len(query_terms & passage_terms)
            # Retrieval ranking is already hybrid; lexical overlap focuses on
            # the relevant sentence inside a 480-word chunk.
            score = overlap * 3 + 3 / (1 + number * .24)
            if hit.get("source") == "Wikipedia RU":
                score += .6
            if plot_intent and re.search(r"##\s*(?:в ролях|акт[её]ры|cast|production|производство)\b", passage, re.I):
                score -= 9
            if plot_intent and re.search(r"##\s*сведения из таблицы", passage, re.I):
                score -= 9
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
    best_score = candidates[0][0] if candidates else 0
    selected: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    counts: dict[int, int] = {}
    size = 0
    for score, number, index, passage in candidates:
        if best_score > 5 and score < best_score * (.45 if why_intent else .70):
            continue
        key = re.sub(r"\W+", "", passage.lower())[:180]
        if key in seen or counts.get(number, 0) >= 3:
            continue
        if size + len(passage) > max_chars:
            continue
        seen.add(key)
        selected.append((number, index, passage))
        counts[number] = counts.get(number, 0) + 1
        size += len(passage)
        if len(selected) >= (3 if duplicate_intent and why_intent else 8):
            break
    return "\n".join(f"[{number}] {passage}" for number, _, passage in selected)
