"""Keyless discovery and ingestion from public Wikipedia articles."""

from __future__ import annotations

from difflib import SequenceMatcher
from threading import Lock
from urllib.parse import quote
import re


from .db import Database
from .sources import SourceError, Wikipedia, title_key


_KIND_HINTS = {
    "person": r"\b(?:акт[её]р\w*|актрис\w*|режисс[её]р\w*|кинематографист\w*|actor|actress|filmmaker|film director)\b",
    "tv": r"\b(?:телесериал\w*|сериал\w*|ситком\w*|телешоу|television series|tv series|miniseries|sitcom)\b",
    "movie": r"\b(?:фильм\w*|кинофильм\w*|мультфильм\w*|кинокомеди\w*|film|animated feature)\b",
}
_QUALIFIER = re.compile(r"\s*\((?:кино)?фильм[^)]*\)$|\s*\((?:теле)?сериал[^)]*\)$|\s*\((?:[^)]*\bfilm|TV series)[^)]*\)$", re.I)
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_OFFSETS = {"ru": 1_000_000_000, "en": 2_000_000_000}


def _kind(page: dict) -> str | None:
    title = page.get("title") or ""
    description = page.get("description") or ""
    if re.search(r"\(значения\)|страница значений|disambiguation|список|^list of|media franchise|медиафраншиза|\(франшиза\)|\(franchise\)|сезон телесериала|эпизод телесериала|серия телесериала|television episode", f"{title} {description}", re.I):
        return None
    # An explicit title qualifier outranks names of directors in descriptions.
    if qualifier := _QUALIFIER.search(title):
        return "tv" if re.search(r"сериал|TV series", qualifier[0], re.I) else "movie"
    ranked = [(match.start(), kind) for kind, pattern in _KIND_HINTS.items()
              if (match := re.search(pattern, description, re.I))]
    if ranked:
        return min(ranked)[1]
    if _YEAR.search(description) and re.search(r"боевик|триллер|комеди|драм|хоррор|вестерн|мелодрам|action thriller", description, re.I):
        return "movie"
    return None



def _url(language: str, title: str) -> str:
    return f"https://{language}.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


class WikiCatalog:
    def __init__(self, db: Database, ru: Wikipedia, en: Wikipedia):
        self.db = db
        self.wikis = {"ru": ru, "en": en}
        self._locks = [Lock() for _ in range(32)]

    def discover(self, query: str) -> list[dict]:
        language = "ru" if re.search(r"[а-яё]", query, re.I) else "en"
        lookup = title_key(query)
        rows = []
        failure = None
        for candidate_language in (language, "en" if language == "ru" else "ru"):
            try:
                pages = self.wikis[candidate_language].search_pages(query)
            except SourceError as exc:
                failure = exc
                continue
            for rank, page in enumerate(pages):
                kind, page_id = _kind(page), page.get("id")
                if not kind or not isinstance(page_id, int) or not 0 < page_id < 1_000_000_000:
                    continue
                title = page["title"]
                display = _QUALIFIER.sub("", title)
                if kind == "person" and ", " in display:
                    last, first = display.split(", ", 1)
                    display = f"{first} {last}"
                display_key = title_key(display)
                similarity = SequenceMatcher(None, lookup, display_key).ratio()
                redirect = title_key(page.get("matched_title") or "")
                exact = lookup in {display_key, title_key(title), redirect}
                prefix = display_key.startswith(lookup)
                if not (exact or prefix or similarity >= .57):
                    continue
                year_match = _YEAR.search(title + " " + (page.get("description") or "")) if kind != "person" else None
                year = year_match[0] if year_match else ""
                url = _url(candidate_language, title)
                owner = self.db.entity_for_url(url)
                if owner and owner[0] == kind:
                    ident = int(owner[1])
                else:
                    ident = _OFFSETS[candidate_language] + page_id
                summary = page.get("description") or ""
                item = {"title": display, "name": display, "original_title": title, "year": year,
                        "overview_ru": summary, "wiki_page": title, "wiki_language": candidate_language,
                        "wiki_loaded": False}
                score = 4 * exact + 2 * prefix + similarity - rank * .04
                rows.append((score, kind, ident, item))
            if rows:
                break
        if not rows and failure:
            raise failure
        output = []
        for _, kind, ident, item in sorted(rows, key=lambda row: -row[0])[:8]:
            self.db.put_entity(kind, str(ident), item["title"], item, replace=False)
            output.append({"type": kind, "id": ident, "title": item["title"],
                           "original_title": item["original_title"], "year": item["year"],
                           "summary": item["overview_ru"], "poster_path": None})
        return output

    def ingest(self, kind: str, entity_id: int, *, refresh: bool = False) -> dict:
        with self._locks[entity_id % len(self._locks)]:
            return self._ingest(kind, entity_id, refresh)

    def _ingest(self, kind: str, entity_id: int, refresh: bool) -> dict:
        ident = str(entity_id)
        item = self.db.get_entity(kind, ident)
        if not item:
            raise KeyError("Сначала найдите фильм, сериал или актёра через поиск.")
        title = item.get("title_ru") or item.get("title") or item.get("name")
        article_title = item.get("wiki_page") or item.get("demo_article")
        language = item.get("wiki_language", "ru")
        result = {"type": kind, "id": entity_id, "title": title, "cached": True}
        has_text = bool(self.db.entity_chunk_ids(kind, ident))
        if has_text and not refresh and (item.get("wiki_loaded") or not article_title):
            return result
        if not article_title:
            if has_text and not refresh:
                return result
            raise SourceError("Для этого объекта нет сохранённой ссылки на статью Wikipedia.")
        wiki = self.wikis[language]
        article = None
        if language == "en" and (translated := wiki.translated_title(article_title, "ru")):
            localized_url = _url("ru", translated)
            if self.db.entity_for_url(localized_url) in {None, (kind, ident)}:
                article = self.wikis["ru"].article_details(translated)
                if article:
                    article_title, language = translated, "ru"
        if not article:
            article = wiki.article_details(article_title)
        if not article:
            if has_text and not refresh:
                return result
            raise SourceError("Статья Wikipedia сейчас недоступна. Попробуйте обновить позже.")
        text = article.text
        url = item.get("demo_url") or _url(language, article_title)
        owner = self.db.entity_for_url(url)
        if owner and owner != (kind, ident):
            raise SourceError("Эта статья уже связана с другой карточкой. Повторите поиск.")
        _, changed = self.db.put_document(
            source="Wikipedia RU" if language == "ru" else "Wikipedia",
            source_id=article_title, entity_type=kind, entity_id=ident, title=title, url=url, text=text,
            published_at=item.get("release_date") or item.get("first_air_date"),
        )
        self.db.put_entity(kind, ident, title, {**item, "wiki_loaded": True, "wiki_facts": article.facts,
                                              "wiki_source_url": url, "wiki_actual_page": article_title,
                                              "wiki_actual_language": language})
        return {**result, "cached": False, "changed": changed}

    def extra_cast(self, kind: str, entity_id: str) -> tuple[list[dict], str | None]:
        """Fetch the linked English cast only when the primary article cannot answer a role question."""
        with self._locks[int(entity_id) % len(self._locks)]:
            item = self.db.get_entity(kind, entity_id) or {}
            if item.get("wiki_extra_cast") is not None:
                return item["wiki_extra_cast"], item.get("wiki_extra_url")
            title = item.get("wiki_actual_page")
            if not title or item.get("wiki_actual_language") != "ru":
                return [], None
            english = self.wikis["ru"].translated_title(title, "en")
            if not english:
                return [], None
            article = self.wikis["en"].article_details(english)
            if not article:
                return [], None
            url = _url("en", english)
            if self.db.entity_for_url(url) not in {None, (kind, entity_id)}:
                return [], None
            name = item["title"]
            cast = article.facts.get("cast", [])
            if cast:
                self.db.put_document(source="Wikipedia", source_id=english, entity_type=kind,
                                     entity_id=entity_id, title=name, url=url, text=article.text)
            self.db.put_entity(kind, entity_id, name, {**item, "wiki_extra_cast": cast, "wiki_extra_url": url})
            return cast, url
