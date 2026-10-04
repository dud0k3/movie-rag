"""TMDB metadata and Wikipedia article ingestion."""

from __future__ import annotations

import re
import time
from datetime import date
from difflib import SequenceMatcher
from urllib.parse import quote

from bs4 import BeautifulSoup
import httpx
import requests

from .config import Settings
from .db import Database


USER_AGENT = "MovieRAG-Educational/0.1 (local student project; contact via GitHub repository)"


class SourceError(RuntimeError):
    pass


class TMDb:
    def __init__(self, config: Settings):
        self.config = config
        self.client = httpx.Client(
            base_url="https://api.themoviedb.org/3/", timeout=25,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )

    def get(self, path: str, **params) -> dict:
        if not self.config.has_tmdb_auth:
            raise SourceError("Добавьте ключ TMDb в локальный файл .env")
        headers = {}
        if self.config.tmdb_token:
            headers["Authorization"] = f"Bearer {self.config.tmdb_token}"
        else:
            params["api_key"] = self.config.tmdb_api_key
        try:
            response = self.client.get(path.lstrip("/"), params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise SourceError("TMDb отклонил ключ. Проверьте значение в .env") from exc
            raise SourceError(f"TMDb вернул ошибку {status}") from exc
        except httpx.HTTPError as exc:
            raise SourceError(f"Не удалось связаться с TMDb: {exc}") from exc

    def search(self, query: str) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        data = self.get("search/multi", query=query, language="ru-RU", include_adult="false", page=1)
        output = []
        for row in data.get("results", []):
            kind = row.get("media_type")
            if kind not in {"movie", "tv", "person"}:
                continue
            title = row.get("title") or row.get("name") or ""
            output.append({
                "type": kind,
                "id": row.get("id"),
                "title": title,
                "original_title": row.get("original_title") or row.get("original_name") or title,
                "year": (row.get("release_date") or row.get("first_air_date") or "")[:4],
                "summary": row.get("overview") or row.get("known_for_department") or "",
                "poster_path": row.get("poster_path") or row.get("profile_path"),
            })
        if not output:
            # TMDB cannot resolve character names. If a natural query contains
            # a series title, try its content words from the end (e.g. “... из
            # сериала Слово пацана” -> “пацана”) as a lightweight title lookup.
            ignored = {"кто", "что", "такой", "такая", "из", "о", "об", "сериала", "сериал", "персонаж", "герой", "героя", "героиня", "маратик", "маратика", "марат", "марата", "maratik", "marat"}
            title_terms = [term for term in re.findall(r"[\w]+", query.lower()) if len(term) >= 4 and term not in ignored]
            for term in reversed(re.findall(r"[\w]+", query.lower())):
                if len(term) < 4 or term in ignored:
                    continue
                try:
                    tv_data = self.get("search/tv", query=term, language="ru-RU", include_adult="false", page=1)
                except SourceError:
                    break
                matches = []
                for row in tv_data.get("results", []):
                    title = row.get("name") or ""
                    title_words = re.findall(r"[\w]+", title.lower())
                    matches_title = lambda word: word in title_words or any(SequenceMatcher(None, word, title_word).ratio() >= 0.8 for title_word in title_words)
                    if matches_title(term) and all(matches_title(word) for word in title_terms):
                        matches.append(row)
                if matches:
                    for row in matches[:3]:
                        output.append({
                            "type": "tv", "id": row.get("id"),
                            "title": row.get("name") or row.get("original_name") or "",
                            "original_title": row.get("original_name") or row.get("name") or "",
                            "year": (row.get("first_air_date") or "")[:4],
                            "summary": row.get("overview") or "",
                            "poster_path": row.get("poster_path"),
                        })
                    break
        return output

    def seed_candidates(self, pages: int = 12) -> list[int]:
        ids: list[int] = []
        for page in range(1, pages + 1):
            data = self.get(
                "discover/movie", language="en-US", page=page,
                sort_by="vote_average.desc", **{"vote_count.gte": 5000},
                include_adult="false", include_video="false",
            )
            for item in data.get("results", []):
                released = item.get("release_date") or ""
                if (item.get("id") and released and released <= date.today().isoformat()
                        and item.get("original_language") != "ru" and item.get("overview")):
                    ids.append(item["id"])
        return list(dict.fromkeys(ids))

    def movie(self, movie_id: int) -> dict:
        english = self.get(f"movie/{movie_id}", language="en-US", append_to_response="credits,external_ids")
        russian = self.get(f"movie/{movie_id}", language="ru-RU")
        english["title_ru"] = russian.get("title") or english.get("title")
        english["overview_ru"] = russian.get("overview") or ""
        return english

    def person(self, person_id: int) -> dict:
        person = self.get(f"person/{person_id}", language="en-US", append_to_response="combined_credits,external_ids")
        russian = self.get(f"person/{person_id}", language="ru-RU")
        person["name_ru"] = russian.get("name") or person.get("name")
        return person

    def tv(self, tv_id: int) -> dict:
        series = self.get(
            f"tv/{tv_id}", language="ru-RU",
            append_to_response="aggregate_credits,external_ids",
        )
        original = self.get(f"tv/{tv_id}", language="en-US")
        series["original_name"] = original.get("original_name") or series.get("original_name")
        series["overview_original"] = original.get("overview") or ""
        return series


class Wikipedia:
    def __init__(self, contact_url: str, language: str = "en"):
        if language not in {"en", "ru"}:
            raise ValueError("Unsupported Wikipedia language")
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/rest.php/v1/"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"MovieRAGStudent/0.1 ({contact_url})" if contact_url else USER_AGENT,
            "Accept": "application/json",
        })
        self.last_request = 0.0

    def get(self, path: str, **params):
        elapsed = time.monotonic() - self.last_request
        if elapsed < 1.2:
            time.sleep(1.2 - elapsed)
        for attempt in range(3):
            response = self.session.get(self.base_url + path, params=params, timeout=25)
            self.last_request = time.monotonic()
            if response.status_code in {429, 503}:
                retry_after = response.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else min(3 * (attempt + 1), 10)
                time.sleep(delay)
                continue
            return response
        return response

    def search_title(self, name: str, kind: str, year: str = "") -> str | None:
        if self.language == "ru":
            query = (f"{name} {year} фильм" if kind == "movie" else
                     f"{name} сериал" if kind == "tv" else f"{name} актёр режиссёр")
        else:
            query = (f"{name} {year} film" if kind == "movie" else
                     f"{name} television series" if kind == "tv" else f"{name} actor director")
        try:
            response = self.get("search/page", q=query, limit=5)
            response.raise_for_status()
            pages = response.json().get("pages", [])
        except (requests.RequestException, ValueError):
            return None
        def normalized(value: str) -> str:
            value = re.sub(r"\([^)]*\)", "", value.lower().replace("ё", "е"))
            return " ".join(re.findall(r"[\w]+", value))

        requested = normalized(name)
        for page in pages:
            title = page.get("title", "")
            if normalized(title) == requested and "значения" not in title.lower():
                return title
        for page in pages:
            title = page.get("title", "")
            description = (page.get("description") or "").lower()
            if kind in {"movie", "tv"}:
                film_word = ("фильм" if kind == "movie" else "сериал") if self.language == "ru" else ("film" if kind == "movie" else "series")
                similarity = SequenceMatcher(None, requested, normalized(title)).ratio()
                if similarity >= 0.60 and (film_word in description or film_word in title.lower()):
                    return title
            elif any(word in description for word in (
                ("актёр", "актер", "актриса", "режиссёр", "режиссер")
                if self.language == "ru" else ("actor", "actress", "director", "filmmaker")
            )):
                return title
        return None

    def article(self, title: str) -> tuple[str, str] | None:
        try:
            response = self.get(f"page/{quote(title, safe='')}/with_html")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return None
        soup = BeautifulSoup(data.get("html", ""), "html.parser")
        body = soup.select_one(".mw-parser-output") or soup
        # Wikipedia often stores episode synopses and cast lists in wikitable
        # markup. Preserve useful tables before removing layout/navigation tables.
        table_sections = []
        for table in body.select("table.wikitable"):
            rows = []
            for row in table.find_all("tr"):
                cells = [" ".join(cell.get_text(" ", strip=True).split())
                         for cell in row.find_all(["th", "td"], recursive=False)]
                cells = list(dict.fromkeys(re.sub(r"\[\d+\]", "", cell).strip() for cell in cells))
                if cells and any(cells):
                    rows.append(" | ".join(cell for cell in cells if cell))
            if not rows:
                continue
            content = "\n".join(rows)
            signals = ("описание серии", "содержание", "актёр", "актер", "роль", "episode", "cast", "plot")
            if any(signal in content.lower() for signal in signals) or any(len(row) > 250 for row in rows):
                table_sections.append((0 if "описание серии" in content.lower() or "episode" in content.lower() else 1,
                                       f"\n## Сведения из таблицы Википедии\n{content}"))
            table.decompose()
        table_sections.sort(key=lambda item: item[0])
        for tag in body.select(
            "table, nav, aside, style, script, sup.reference, .mw-editsection, "
            ".reflist, .navbox, .hatnote, .metadata, .mw-empty-elt"
        ):
            tag.decompose()
        sections = []
        for node in body.find_all(["h2", "h3", "p", "li"]):
            text = " ".join(node.get_text(" ", strip=True).split())
            if len(text) < 35 and node.name not in {"h2", "h3"}:
                continue
            if node.name in {"h2", "h3"}:
                if text.lower() in {"references", "external links", "notes", "see also", "further reading",
                                    "примечания", "ссылки", "литература", "см. также"}:
                    break
                sections.append(f"\n## {text}\n")
            else:
                sections.append(text)
        text = re.sub(r"\[\d+\]", "", "\n".join(sections + [item[1] for item in table_sections])).strip()
        if len(text) < 200:
            return None
        url = data.get("html_url") or f"https://{self.language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        return text[:45000], url


def movie_document(movie: dict) -> str:
    crew = movie.get("credits", {}).get("crew", [])
    cast = movie.get("credits", {}).get("cast", [])
    def names_for(jobs: set[str]) -> str:
        return ", ".join(dict.fromkeys(row["name"] for row in crew if row.get("job") in jobs and row.get("name"))) or "не указано"
    title = movie.get("title") or movie.get("original_title") or "Фильм"
    lines = [
        f"Фильм: {title}.",
        f"Оригинальное название: {movie.get('original_title') or title}.",
        f"Русское название: {movie.get('title_ru') or title}.",
        f"Дата выхода: {movie.get('release_date') or 'не указана'}.",
        f"Режиссёр: {names_for({'Director'})}.",
        f"Сценаристы: {names_for({'Screenplay', 'Writer', 'Story'})}.",
        f"Оператор: {names_for({'Director of Photography', 'Cinematography'})}.",
        f"Композитор: {names_for({'Original Music Composer', 'Music'})}.",
        f"Продюсеры: {names_for({'Producer'})}.",
        "Главные актёры: " + (", ".join(
            f"{row.get('name')} ({row.get('character')})" if row.get("character") else row.get("name", "")
            for row in cast[:15] if row.get("name")
        ) or "не указаны") + ".",
    ]
    if movie.get("overview"):
        lines.append("Описание: " + movie["overview"])
    if movie.get("overview_ru"):
        lines.append("Описание на русском: " + movie["overview_ru"])
    return "\n".join(lines)


def person_document(person: dict) -> str:
    movie_credits = person.get("combined_credits", {})
    cast = [row for row in movie_credits.get("cast", []) if row.get("media_type") == "movie"]
    crew = [row for row in movie_credits.get("crew", []) if row.get("media_type") == "movie"]
    lines = [
        f"Имя: {person.get('name', 'неизвестно')}.",
        f"Русское имя: {person.get('name_ru') or person.get('name', 'неизвестно')}.",
        f"Известен в области: {person.get('known_for_department') or 'не указано'}.",
        f"Дата рождения: {person.get('birthday') or 'не указана'}.",
        f"Место рождения: {person.get('place_of_birth') or 'не указано'}.",
        f"Биография: {person.get('biography') or 'не указана'}",
    ]
    if cast:
        lines.append("Фильмы как актёр: " + "; ".join(
            f"{row.get('title')} ({(row.get('release_date') or '')[:4]}) — {row.get('character') or 'роль не указана'}"
            for row in cast[:35] if row.get("title")
        ))
    if crew:
        lines.append("Работа в съёмочной группе: " + "; ".join(
            f"{row.get('title')} ({(row.get('release_date') or '')[:4]}) — {row.get('job') or 'участник'}"
            for row in crew[:35] if row.get("title")
        ))
    return "\n".join(lines)


def tv_document(series: dict) -> str:
    title = series.get("name") or series.get("original_name") or "Сериал"
    credits = series.get("aggregate_credits", {})
    lines = [
        f"Сериал: {title}.",
        f"Оригинальное название: {series.get('original_name') or title}.",
        f"Дата выхода: {series.get('first_air_date') or 'не указана'}.",
        f"Создатели: {', '.join(person.get('name', '') for person in series.get('created_by', []) if person.get('name')) or 'не указаны'}.",
        f"Количество сезонов: {series.get('number_of_seasons') or 'не указано'}.",
    ]
    cast = credits.get("cast", [])
    if cast:
        roles = []
        for person in cast:
            name = person.get("name")
            if not name:
                continue
            for role in person.get("roles", []):
                character = role.get("character")
                if character:
                    roles.append(f"Персонаж {character} — актёр {name}")
        if roles:
            lines.append("Персонажи и исполнители: " + "; ".join(roles[:70]) + ".")
    if series.get("overview"):
        lines.append("Описание на русском: " + series["overview"])
    if series.get("overview_original"):
        lines.append("Описание: " + series["overview_original"])
    return "\n".join(lines)


class Ingestor:
    def __init__(self, db: Database, config: Settings):
        self.db = db
        self.tmdb = TMDb(config)
        self.wiki = Wikipedia(config.wikimedia_contact_url)
        self.wiki_ru = Wikipedia(config.wikimedia_contact_url, "ru")

    def enrich_russian(self, kind: str, entity_id: int) -> bool:
        item = self.db.get_entity(kind, str(entity_id))
        if not item:
            return False
        title = (item.get("title_ru") or item.get("title") or item.get("original_title")
                 if kind == "movie" else item.get("name_ru") or item.get("name"))
        year = (item.get("release_date") or item.get("first_air_date") or "")[:4] if kind in {"movie", "tv"} else ""
        wiki_title = self.wiki_ru.search_title(title, kind, year) if title else None
        if not wiki_title:
            return False
        article = self.wiki_ru.article(wiki_title)
        if not article:
            return False
        article_text, url = article
        self.db.remove_other_source_documents(kind, str(entity_id), "Wikipedia RU", url)
        self.db.put_document(
            source="Wikipedia RU", source_id=wiki_title, entity_type=kind,
            entity_id=str(entity_id), title=wiki_title, url=url, text=article_text,
        )
        return True

    def ingest(self, kind: str, entity_id: int, *, refresh: bool = False) -> dict:
        if kind not in {"movie", "tv", "person"}:
            raise ValueError("kind must be movie, tv or person")
        if not refresh:
            stored = self.db.get_entity(kind, str(entity_id))
            if stored:
                return {"type": kind, "id": entity_id, "title": stored.get("title") or stored.get("name"), "cached": True}
        if kind == "movie":
            item = self.tmdb.movie(entity_id)
            title = item.get("title") or str(entity_id)
            text = movie_document(item)
            year = (item.get("release_date") or "")[:4]
            wiki_title = self.wiki.search_title(item.get("original_title") or title, "movie", year)
            published = item.get("release_date")
        elif kind == "tv":
            item = self.tmdb.tv(entity_id)
            title = item.get("name") or str(entity_id)
            text = tv_document(item)
            year = (item.get("first_air_date") or "")[:4]
            wiki_title = self.wiki.search_title(title, "tv", year)
            published = item.get("first_air_date")
        else:
            item = self.tmdb.person(entity_id)
            title = item.get("name") or str(entity_id)
            text = person_document(item)
            wiki_title = self.wiki.search_title(title, "person")
            published = item.get("birthday")
        self.db.put_entity(kind, str(entity_id), title, item)
        self.db.put_document(
            source="TMDB", source_id=str(entity_id), entity_type=kind, entity_id=str(entity_id),
            title=(f"{title} / {item['title_ru']} — TMDB"
                   if kind == "movie" and item.get("title_ru") and item["title_ru"] != title
                   else f"{title} — TMDB"),
            url=f"https://www.themoviedb.org/{kind}/{entity_id}",
            text=text, published_at=published,
        )
        wiki_added = False
        if wiki_title:
            article = self.wiki.article(wiki_title)
            if article:
                article_text, url = article
                self.db.put_document(
                    source="Wikipedia", source_id=wiki_title, entity_type=kind,
                    entity_id=str(entity_id), title=wiki_title, url=url,
                    text=article_text,
                )
                wiki_added = True
        ru_added = self.enrich_russian(kind, entity_id)
        return {"type": kind, "id": entity_id, "title": title, "cached": False,
                "wikipedia": wiki_added, "wikipedia_ru": ru_added}

    def seed_popular(self, count: int = 100, progress=None) -> dict:
        ids = self.tmdb.seed_candidates(pages=max(8, (count + 19) // 20 + 4))
        results = []
        for movie_id in ids:
            if sum("error" not in row for row in results) >= count:
                break
            try:
                results.append(self.ingest("movie", movie_id))
            except (SourceError, httpx.HTTPError, ValueError) as exc:
                results.append({"type": "movie", "id": movie_id, "error": str(exc)})
            if progress:
                progress(len(results), count, results[-1])
            time.sleep(0.12)
        return {"requested": count, "loaded": sum("error" not in row for row in results), "results": results}
