from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from threading import Event, Thread
import hashlib
import logging
import json
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import PLOT_QUESTION, generate, stream_generate, warm_model
from .config import settings
from .credits import cast_credit, role_query
from .db import Database
from .demo_catalog import seed_demo, starter_cards
from .evidence import question_parts
from .search import SearchEngine
from .sources import Ingestor, SourceError, mentioned_title, title_key
from .wiki_catalog import WikiCatalog


db = Database(settings.db_path)
search_engine = SearchEngine(db, settings)
ingestor = Ingestor(db, settings)
wiki_catalog = WikiCatalog(db, ingestor.wiki_ru, ingestor.wiki)

app = FastAPI(title="Movie RAG", version="0.1.0", description="Поиск и ответы о фильмах с источниками",
              docs_url=None, redoc_url=None)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
log = logging.getLogger(__name__)
_search_ready = Event()


@app.on_event("startup")
def preload_answer_model() -> None:
    if settings.demo_mode or settings.offline:
        seed_demo(db)
    Thread(target=warm_model, args=(settings,), daemon=True, name="qwen-warmup").start()
    if settings.semantic_enabled:
        Thread(target=preload_search_model, daemon=True, name="search-warmup").start()


def preload_search_model() -> None:
    try:
        search_engine.warm()
        log.info("Search encoder and vector index are ready")
    except Exception:
        log.exception("Could not preload the semantic search encoder")
    finally:
        _search_ready.set()


def diverse_hits(hits: list[dict], limit: int = 6, per_url_limit: int = 2) -> list[dict]:
    """Keep distinct sources and take later chunks from each source in rounds."""
    selected: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for hit in hits:
        grouped.setdefault(hit["url"], []).append(hit)
    sources = list(grouped.values())
    # Round-robin later chunks so that a second language/version of an article
    # is not crowded out by adjacent chunks from the first one.
    for depth in range(per_url_limit):
        for source_hits in sources:
            if depth < len(source_hits):
                selected.append(source_hits[depth])
                if len(selected) == limit:
                    return selected
    return selected


class IngestRequest(BaseModel):
    type: str = Field(pattern="^(movie|tv|person)$")
    id: int = Field(gt=0)
    refresh: bool = False


@app.get("/", include_in_schema=False)
def home():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in ("style.css", "app.js"):
        digest = hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:12]
        html = html.replace(f'/static/{name}"', f'/static/{name}?v={digest}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/docs", include_in_schema=False)
def docs():
    return HTMLResponse((STATIC / "api.html").read_text(encoding="utf-8"))


def _effective_mode(mode: str) -> str:
    return mode if settings.semantic_enabled and _search_ready.is_set() else "bm25"


@app.get("/health")
def health():
    return {"ok": True, "tmdb_configured": settings.has_tmdb_auth,
            "database": db.stats(), "semantic_error": search_engine.semantic_error,
            "demo_mode": settings.offline, "semantic_enabled": settings.semantic_enabled,
            "starter": starter_cards() if settings.demo_mode or settings.offline else [],
            "ready": True}


def _local_discover(q: str) -> list[dict]:
    hits = db.bm25(q, limit=18)
    details = db.chunk_details([ident for ident, _ in hits])
    results = []
    seen = set()
    for ident, _ in hits:
        hit = details.get(ident)
        if not hit:
            continue
        kind, entity_id = hit["entity_type"], hit["entity_id"]
        if (kind, entity_id) in seen:
            continue
        item = db.get_entity(kind, entity_id)
        if not item:
            continue
        seen.add((kind, entity_id))
        title = item.get("title_ru") or item.get("name_ru") or item.get("title") or item.get("name") or entity_id
        results.append({
            "type": kind, "id": int(entity_id), "title": title,
            "original_title": item.get("original_title") or item.get("original_name") or title,
            "year": (item.get("release_date") or item.get("first_air_date") or item.get("year") or "")[:4],
            "summary": item.get("overview_ru") or item.get("overview") or "",
            "poster_path": item.get("poster_path") or item.get("profile_path"),
        })
        if len(results) == 6:
            break
    return results


@app.get("/discover")
def discover(q: str = Query(min_length=2, max_length=120)):
    local = _local_discover(q)
    if settings.offline:
        return {"query": q, "results": local}
    query = mentioned_title(q) or q
    if settings.has_tmdb_auth:
        try:
            if results := ingestor.tmdb.search(query):
                return {"query": q, "results": results}
        except SourceError:
            log.info("TMDB unavailable; trying Wikipedia")
    try:
        remote = wiki_catalog.discover(query)
    except SourceError as exc:
        return {"query": q, "results": local, "notice": str(exc)}
    combined = {}
    local_titles = [item for item in local if title_key(item["title"]).startswith(title_key(query))]
    for item in [*remote, *(local_titles if remote else local)]:
        combined.setdefault((item["type"], item["id"]), item)
    return {"query": q, "results": list(combined.values())[:8]}


@app.post("/ingest")
def ingest(body: IngestRequest):
    item = db.get_entity(body.type, str(body.id))
    if settings.offline:
        if not item or not db.entity_chunk_ids(body.type, str(body.id)):
            raise HTTPException(status_code=404, detail="Этого объекта нет в локальной базе.")
        if body.refresh:
            raise HTTPException(status_code=409, detail="Для обновления статьи отключите режим без интернета.")
        return {"type": body.type, "id": body.id, "title": item.get("title") or item.get("name"), "cached": True}
    try:
        if not settings.has_tmdb_auth or (item and item.get("wiki_page")):
            return wiki_catalog.ingest(body.type, body.id, refresh=body.refresh)
        return ingestor.ingest(body.type, body.id, refresh=body.refresh)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc
    except SourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/search")
def search(
    q: str = Query(min_length=2, max_length=500), top_n: int = Query(5, ge=1, le=30),
    mode: str = Query("hybrid", pattern="^(hybrid|bm25|semantic)$"),
    entity_type: str | None = Query(None, pattern="^(movie|tv|person)$"),
    entity_id: int | None = Query(None, gt=0),
):
    entity = (entity_type, str(entity_id)) if entity_type and entity_id else None
    effective = _effective_mode(mode)
    hits = search_engine.search(q, limit=top_n, mode=effective, entity=entity)
    return {"query": q, "mode": effective, "results": hits,
            "semantic_error": search_engine.semantic_error}


def _catalog_fact_hit(q: str, entity: tuple[str, str], item: dict) -> dict | None:
    role = role_query(q)
    facts = item.get("wiki_facts", {})
    wiki_url = item.get("wiki_source_url")
    if role:
        credit = cast_credit(item, entity[0], role)
        if credit and (hit := db.catalog_chunk(entity)):
            actor, character = credit
            hit.update(text=f"Персонаж: {character}. Исполнитель: {actor}.",
                       catalog_answer=f"Эту роль исполняет {actor} [1].", score=1.0)
            return hit
        cast = facts.get("cast", [])
        credit = cast_credit({"credits": {"cast": cast}}, "movie", role, fuzzy=True)
        if not credit and wiki_url and not settings.offline and entity[0] in {"movie", "tv"}:
            cast, extra_url = wiki_catalog.extra_cast(*entity)
            credit = cast_credit({"credits": {"cast": cast}}, "movie", role, fuzzy=True)
            if credit:
                wiki_url = extra_url
        if credit and wiki_url and (hit := db.catalog_chunk(entity, url=wiki_url)):
            actor, character = credit
            hit.update(text=f"Персонаж: {character}. Исполнитель: {actor}.",
                       catalog_answer=f"Эту роль исполняет {actor} [1].", score=1.0)
            return hit
    if not wiki_url:
        return None
    parts = question_parts(q)
    fact = None
    evidence = None
    if len(parts) == 1 and entity[0] == "movie" and re.match(r"^\s*кто (?:снял|режисс[её]р)\b", q, re.I):
        if director := facts.get("director"):
            fact = f"Режиссёр фильма — {director}."
    if len(parts) == 1 and entity[0] in {"movie", "tv"} and re.match(r"^\s*о ч[её]м\b", q, re.I):
        fact = facts.get("premise")
    if entity[0] == "person" and re.match(r"^\s*чем .*извест", q, re.I) and all(
        re.match(r"(?:чем .*извест|назови .*фильм)", part, re.I) for part in parts
    ):
        evidence = facts.get("known_for")
        if evidence:
            titles = list(dict.fromkeys(title.strip() for title in re.findall(r"«([^»]+)»", evidence)))[:5]
            fact = "Среди известных работ — " + ", ".join(f"«{title}»" for title in titles) + "." if titles else evidence
    if fact and (hit := db.catalog_chunk(entity, url=wiki_url)):
        hit.update(text=evidence or fact, catalog_answer=f"{fact} [1]", score=1.0)
        return hit
    return None


def _answer_hits(q: str, entity_type: str | None, entity_id: int | None, mode: str) -> list[dict]:
    entity = (entity_type, str(entity_id)) if entity_type and entity_id else None
    if not entity and (title := mentioned_title(q)):
        # Only resolve exact titles. Remakes and namesakes should be selected
        # by the user, since search popularity is not evidence of intent.
        year_match = re.search(r"\s*\(((?:19|20)\d{2})\)$", title)
        year = year_match.group(1) if year_match else None
        name = title[:year_match.start()] if year_match else title
        key = title_key(name)
        try:
            candidates = discover(name)["results"]
        except HTTPException:
            candidates = []
        exact = [
            item for item in candidates if key in {title_key(item["title"]), title_key(item["original_title"])}
            and (not year or item["year"] == year)
        ]
        if len(exact) != 1:
            raise HTTPException(status_code=409, detail="Выберите точный фильм, сериал или человека в поиске.")
        entity = (exact[0]["type"], str(exact[0]["id"]))
    if not entity:
        entity = db.match_entity(q)
    if not entity and (settings.demo_mode or settings.offline):
        matches = _local_discover(q)
        if len(matches) == 1:
            entity = (matches[0]["type"], str(matches[0]["id"]))
    if not entity:
        raise HTTPException(status_code=409, detail="Выберите фильм, сериал или человека в поиске либо укажите точное название в кавычках.")
    if not db.entity_chunk_ids(*entity):
        ingest(IngestRequest(type=entity[0], id=int(entity[1])))
    if (item := db.get_entity(*entity)) and (fact_hit := _catalog_fact_hit(q, entity, item)):
        return [fact_hit]
    hits = search_engine.search(q, limit=24, mode=mode, entity=entity)
    parts = question_parts(q)[:3]
    if len(parts) == 1:
        selected = diverse_hits(hits, limit=10, per_url_limit=5)
    else:
        # Extra BM25 lookups are cheap and do not re-encode the subquestions.
        submode = "semantic" if mode == "semantic" else "bm25"
        branches = [search_engine.search(part, limit=4, mode=submode, entity=entity) for part in parts]
        priority = [hit for round_hits in zip_longest(*branches) for hit in round_hits if hit]
        unique = {}
        for hit in [*priority, *hits]:
            unique.setdefault(hit["chunk_id"], hit)
        selected = diverse_hits(list(unique.values()), limit=15, per_url_limit=7)
    if PLOT_QUESTION.search(q):
        ids = [hit["chunk_id"] for hit in selected[:6] if hit["source"].startswith("Wikipedia")]
        windows = db.chunk_windows(ids)
        for hit in selected:
            if window := windows.get(hit["chunk_id"]):
                hit["context_window"] = window
    return selected



def _answer_sources(hits: list[dict]) -> list[dict]:
    return [
        {"number": i, "title": hit["title"], "url": hit["url"], "source": hit["source"],
         "excerpt": hit["text"][:550], "score": hit["score"], "chunk_id": hit["chunk_id"]}
        for i, hit in enumerate(hits, 1)
    ]


@app.get("/ask")
def ask(
    q: str = Query(min_length=2, max_length=500),
    entity_type: str | None = Query(None, pattern="^(movie|tv|person)$"),
    entity_id: int | None = Query(None, gt=0),
    mode: str = Query("hybrid", pattern="^(hybrid|bm25|semantic)$"),
):
    mode = _effective_mode(mode)
    hits = _answer_hits(q, entity_type, entity_id, mode)
    scorer = search_engine.score_passages if mode != "bm25" else None
    answer, answer_mode = generate(q, hits, settings, scorer)
    sources = _answer_sources(hits)
    return {"query": q, "answer": answer, "answer_mode": answer_mode,
            "sources": sources, "semantic_error": search_engine.semantic_error}


@app.get("/ask/stream")
def ask_stream(
    q: str = Query(min_length=2, max_length=500),
    entity_type: str | None = Query(None, pattern="^(movie|tv|person)$"),
    entity_id: int | None = Query(None, gt=0),
    mode: str = Query("hybrid", pattern="^(hybrid|bm25|semantic)$"),
):
    mode = _effective_mode(mode)
    hits = _answer_hits(q, entity_type, entity_id, mode)

    def events():
        yield json.dumps({"type": "sources", "sources": _answer_sources(hits),
                          "semantic_error": search_engine.semantic_error}, ensure_ascii=False) + "\n"
        scorer = search_engine.score_passages if mode != "bm25" else None
        for event in stream_generate(q, hits, settings, scorer):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
