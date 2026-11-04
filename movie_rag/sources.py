"""TMDB metadata and Wikipedia article ingestion."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import date
from difflib import SequenceMatcher
from urllib.parse import quote

from bs4 import BeautifulSoup
import httpx
import requests

from .config import Settings
from .db import Database


USER_AGENT = "MovieRAG-Educational/0.1 (local student project; contact via GitHub repository)"


def mentioned_title(query: str) -> str | None:
    quoted = re.search(r'«([^»]+)»|"([^"]+)"', query)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip() or None
    qualified = re.search(
        r"\b(?:из|в|про|о)\s+(?:фильм[аеу]?|сериал[аеу]?)\s+(.{2,120}?)\s*[?!.]*$",
        query, re.I,
    )
    return qualified.group(1).strip() if qualified else None


def title_key(title: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", title.casefold().replace("ё", "е")))


class SourceError(RuntimeError):
    pass


class TMDb:
    def __init__(self, config: Settings):
        self.config = config
        self.client = httpx.Client(
            base_url="https://api.themoviedb.org/3/", timeout=httpx.Timeout(8, connect=4),
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
            raise SourceError("Не удалось связаться с TMDb") from exc

    def _optional_translation(self, path: str, language: str) -> dict:
        try:
            return self.get(path, language=language)
        except SourceError:
            return {}

    def search(self, query: str) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        query = mentioned_title(query) or query
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
        russian = self._optional_translation(f"movie/{movie_id}", "ru-RU")
        english["title_ru"] = russian.get("title") or english.get("title")
        english["overview_ru"] = russian.get("overview") or ""
        return english

    def person(self, person_id: int) -> dict:
        person = self.get(f"person/{person_id}", language="en-US", append_to_response="combined_credits,external_ids")
        russian = self._optional_translation(f"person/{person_id}", "ru-RU")
        person["name_ru"] = russian.get("name") or person.get("name")
        person["biography_ru"] = russian.get("biography") or ""
        return person

    def tv(self, tv_id: int) -> dict:
        series = self.get(
            f"tv/{tv_id}", language="ru-RU",
            append_to_response="aggregate_credits,external_ids",
        )
        original = self._optional_translation(f"tv/{tv_id}", "en-US")
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
        self._lock = Lock()

    def get(self, path: str, **params):
        with self._lock:
            delay = 1.2 - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            try:
                return self.session.get(self.base_url + path, params=params, timeout=(3, 6))
            finally:
                self.last_request = time.monotonic()

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
            value = value.lower().replace("ё", "е")
            if kind == "person":
                value = re.sub(r"\bмл\.?(?=\W|$)", "младший", value)
            else:
                value = re.sub(r"\([^)]*\)", "", value)
            return " ".join(re.findall(r"[\w]+", value))

        requested = normalized(name)
        if not requested:
            return None
        kind_pattern = {
            "movie": r"\b(?:фильм\w*|кинофильм\w*|мультфильм\w*|film)\b",
            "tv": r"\b(?:телесериал\w*|сериал\w*|series|miniseries|sitcom)\b",
            "person": r"\b(?:акт[её]р\w*|актрис\w*|режисс[её]р\w*|кинематографист\w*|actor|actress|director|filmmaker)\b",
        }[kind]
        ranked = []
        for page in pages:
            title = page.get("title") or ""
            description = page.get("description") or ""
            hint = f"{title} {description}".lower()
            if re.search(r"значения|disambiguation", hint):
                continue
            found = normalized(title)
            similarity = SequenceMatcher(None, requested, found).ratio()
            if kind == "person" and sorted(requested.split()) == sorted(found.split()):
                similarity = 1.0
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", description + " " + title))
            if similarity < .78 or (year and years and year not in years and kind == "movie"):
                continue
            typed = bool(re.search(kind_pattern, hint))
            if not typed and not (similarity == 1 and not description):
                continue
            ranked.append((similarity + .12 * (year in years) + .06 * typed, title))
        ranked.sort(reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return None
        return ranked[0][1]

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
    cast = [row for row in movie_credits.get("cast", []) if row.get("media_type") in {"movie", "tv"}]
    crew = [row for row in movie_credits.get("crew", []) if row.get("media_type") in {"movie", "tv"}]
    lines = [
        f"Имя: {person.get('name', 'неизвестно')}.",
        f"Русское имя: {person.get('name_ru') or person.get('name', 'неизвестно')}.",
        f"Известен в области: {person.get('known_for_department') or 'не указано'}.",
        f"Дата рождения: {person.get('birthday') or 'не указана'}.",
        f"Место рождения: {person.get('place_of_birth') or 'не указано'}.",
        f"Биография: {person.get('biography') or 'не указана'}",
    ]
    if person.get("biography_ru"):
        lines.append("Биография на русском: " + person["biography_ru"])
    if cast:
        lines.append("Фильмы и сериалы как актёр: " + "; ".join(
            f"{row.get('title') or row.get('name')} ({(row.get('release_date') or row.get('first_air_date') or '')[:4]}) — {row.get('character') or 'роль не указана'}"
            for row in cast[:35] if row.get("title") or row.get("name")
        ))
    if crew:
        lines.append("Работа в съёмочной группе: " + "; ".join(
            f"{row.get('title') or row.get('name')} ({(row.get('release_date') or row.get('first_air_date') or '')[:4]}) — {row.get('job') or 'участник'}"
            for row in crew[:35] if row.get("title") or row.get("name")
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
        self._ingest_locks = [Lock() for _ in range(32)]

    def _wiki_document(self, wiki: Wikipedia, name: str, kind: str, year: str, ident: str) -> bool:
        title = wiki.search_title(name, kind, year) if name else None
        article = wiki.article(title) if title else None
        if not article:
            return False
        text, url = article
        source = "Wikipedia RU" if wiki.language == "ru" else "Wikipedia"
        self.db.put_document(
            source=source, source_id=title, entity_type=kind,
            entity_id=ident, title=title, url=url, text=text,
        )
        self.db.remove_other_source_documents(kind, ident, source, url)
        return True

    def enrich_russian(self, kind: str, entity_id: int) -> bool:
        ident = str(entity_id)
        item = self.db.get_entity(kind, ident)
        if not item:
            return False
        name = item.get("title_ru") or item.get("name_ru") or item.get("title") or item.get("name")
        year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
        return self._wiki_document(self.wiki_ru, name, kind, year, ident)

    def ingest(self, kind: str, entity_id: int, *, refresh: bool = False) -> dict:
        if kind not in {"movie", "tv", "person"}:
            raise ValueError("kind must be movie, tv or person")
        # Selection can start loading before Ask is clicked. A second request
        # for the same item waits for both article fetches and reuses the result.
        with self._ingest_locks[entity_id % len(self._ingest_locks)]:
            return self._ingest(kind, entity_id, refresh)

    def _ingest(self, kind: str, entity_id: int, refresh: bool) -> dict:
        ident = str(entity_id)
        stored = self.db.get_entity(kind, ident)
        if stored and not refresh:
            return {"type": kind, "id": entity_id, "title": stored.get("title") or stored.get("name"), "cached": True}
        fetch, document = {
            "movie": (self.tmdb.movie, movie_document),
            "tv": (self.tmdb.tv, tv_document),
            "person": (self.tmdb.person, person_document),
        }[kind]
        item = fetch(entity_id)
        title = item.get("title") or item.get("name") or ident
        localized = item.get("title_ru") or item.get("name_ru") or title
        original = item.get("original_title") or item.get("original_name") or title
        english = original if re.search(r"[a-z]", original, re.I) else title
        published = item.get("release_date") or item.get("first_air_date") or item.get("birthday")
        year = (published or "")[:4] if kind != "person" else ""
        self.db.put_document(
            source="TMDB", source_id=ident, entity_type=kind, entity_id=ident,
            title=f"{title} / {localized} — TMDB" if localized != title else f"{title} — TMDB",
            url=f"https://www.themoviedb.org/{kind}/{entity_id}",
            text=document(item), published_at=published,
        )
        with ThreadPoolExecutor(max_workers=2) as workers:
            en = workers.submit(self._wiki_document, self.wiki, english, kind, year, ident)
            ru = workers.submit(self._wiki_document, self.wiki_ru, localized, kind, year, ident)
            wiki_added, ru_added = en.result(), ru.result()
        self.db.put_entity(kind, ident, title, item)
        return {"type": kind, "id": entity_id, "title": localized, "cached": False,
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
