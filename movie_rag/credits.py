"""Resolve a named role from the selected title's TMDB credits."""

import re
from difflib import SequenceMatcher

from .evidence import _romanized


def role_query(question: str) -> str | None:
    match = re.fullmatch(
        r'\s*кто\s+(?:сыграл[аи]?|играет|исполняет|исполнил[аи]?)\s+'
        r'(?:роль\s+|персонажа\s+)?[«"]?([\w][\w .\-’\x27]{0,75}?)[»"]?[?.!]?\s*',
        question, re.I,
    )
    if not match:
        return None
    role = match[1].strip()
    if re.search(r"\b(?:и|что|где|почему|в фильме|в сериале|главн\w* роль)\b", role, re.I):
        return None
    return role


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _romanized(value))


def _match_word(requested: str, credited: str, *, fuzzy: bool = False) -> bool:
    if requested == credited:
        return True
    if fuzzy and min(len(requested), len(credited)) >= 5 and requested[:3] == credited[:3]:
        if SequenceMatcher(None, requested, credited).ratio() >= .83:
            return True
    # Russian case endings and common diminutives can remain after
    # transliteration. Only accept an intact credited name as the stem.
    return len(credited) >= (3 if fuzzy else 4) and requested.startswith(credited) and requested[len(credited):] in {
        "a", "u", "e", "oi", "om", "em", "ia", "iu", "ik", "ika", "iku", "ikom",
    }


def cast_credit(item: dict, kind: str, role: str, *, fuzzy: bool = False) -> tuple[str, str] | None:
    requested = _words(role)
    if not requested:
        return None
    if kind == "tv":
        cast = item.get("aggregate_credits", {}).get("cast", [])
    elif kind == "movie":
        cast = item.get("credits", {}).get("cast", [])
    else:
        return None
    matches = set()
    short_matches = set()
    for person in cast:
        actor = person.get("name")
        if not actor:
            continue
        roles = person.get("roles", []) if kind == "tv" else [person]
        for credit in roles:
            character = credit.get("character") or ""
            # Aygul and "Aygul's mother" are different roles.
            if re.search(r"[’']+s\s", character, re.I):
                continue
            for alias in re.split(r"[/|;()«»]", character):
                names = _words(alias)
                if len(names) >= len(requested) and all(
                    _match_word(word, names[i], fuzzy=fuzzy) for i, word in enumerate(requested)
                ):
                    matches.add((actor, character))
                elif fuzzy and len(requested) > 1 and len(names) == 1 and _match_word(
                    requested[-1], names[0], fuzzy=True,
                ):
                    short_matches.add((actor, character))
    best = matches or short_matches
    return next(iter(best)) if len(best) == 1 else None
