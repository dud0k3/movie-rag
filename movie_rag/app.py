from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import CONFLICT_QUESTION, PLOT_QUESTION, generate, warm_model
from .config import settings
from .db import Database
from .search import SearchEngine
from .sources import Ingestor, SourceError


db = Database(settings.db_path)
search_engine = SearchEngine(db, settings)
ingestor = Ingestor(db, settings)

app = FastAPI(title="Movie RAG", version="0.1.0", description="Поиск и ответы о фильмах с источниками")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
log = logging.getLogger(__name__)
_warmup_lock = Lock()
_warmup_done: set[str] = set()


@app.on_event("startup")
def preload_answer_model() -> None:
    # Keep startup responsive while loading the local model before the first ask.
    Thread(target=preload_qwen, daemon=True, name="qwen-warmup").start()
    Thread(target=preload_search_model, daemon=True, name="search-warmup").start()


def preload_qwen() -> None:
    try:
        warm_model(settings)
    finally:
        with _warmup_lock:
            _warmup_done.add("qwen")


def preload_search_model() -> None:
    try:
        search_engine.warm()
        log.info("Search encoder and vector index are ready")
    except Exception:
        log.exception("Could not preload the semantic search encoder")
    finally:
        with _warmup_lock:
            _warmup_done.add("search")


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
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "tmdb_configured": settings.has_tmdb_auth,
            "database": db.stats(), "semantic_error": search_engine.semantic_error,
            "ready": len(_warmup_done) == 2}


@app.get("/discover")
def discover(q: str = Query(min_length=2, max_length=120)):
    try:
        results = ingestor.tmdb.search(q)
        if not results:
            local_hits = db.bm25(q, limit=12)
            details = db.chunk_details([ident for ident, _ in local_hits])
            seen: set[tuple[str, str]] = set()
            for ident, _ in local_hits:
                hit = details.get(ident)
                if not hit:
                    continue
                kind, entity_id = hit["entity_type"], hit["entity_id"]
                key = (kind, entity_id)
                if key in seen:
                    continue
                item = db.get_entity(kind, entity_id)
                if not item:
                    continue
                seen.add(key)
                title = item.get("title") or item.get("name") or str(entity_id)
                results.append({
                    "type": kind, "id": int(entity_id), "title": title,
                    "original_title": item.get("original_title") or item.get("original_name") or title,
                    "year": (item.get("release_date") or item.get("first_air_date") or "")[:4],
                    "summary": item.get("overview_ru") or item.get("overview") or "",
                    "poster_path": item.get("poster_path"),
                })
                if len(results) == 6:
                    break
        return {"query": q, "results": results}
    except SourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ingest")
def ingest(body: IngestRequest):
    try:
        return ingestor.ingest(body.type, body.id, refresh=body.refresh)
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
    hits = search_engine.search(q, limit=top_n, mode=mode, entity=entity)
    return {"query": q, "mode": mode, "results": hits,
            "semantic_error": search_engine.semantic_error}


@app.get("/ask")
def ask(
    q: str = Query(min_length=2, max_length=500),
    entity_type: str | None = Query(None, pattern="^(movie|tv|person)$"),
    entity_id: int | None = Query(None, gt=0),
    mode: str = Query("hybrid", pattern="^(hybrid|bm25|semantic)$"),
):
    entity = (entity_type, str(entity_id)) if entity_type and entity_id else None
    if entity and not db.get_entity(*entity):
        try:
            ingestor.ingest(entity[0], int(entity[1]))
        except SourceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    plot_question = bool(PLOT_QUESTION.search(q))
    conflict_question = bool(CONFLICT_QUESTION.search(q)) and bool(entity and entity[0] == "person")
    hits = diverse_hits(
        search_engine.search(q, limit=18, mode=mode, entity=entity),
        per_url_limit=6 if plot_question or conflict_question else 2,
    )
    answer, answer_mode = generate(q, hits, settings)
    sources = [
        {"number": i, "title": hit["title"], "url": hit["url"], "source": hit["source"],
         "excerpt": hit["text"][:550], "score": hit["score"], "chunk_id": hit["chunk_id"]}
        for i, hit in enumerate(hits, 1)
    ]
    return {"query": q, "answer": answer, "answer_mode": answer_mode,
            "sources": sources, "semantic_error": search_engine.semantic_error}
