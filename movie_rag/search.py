"""BM25 plus multilingual semantic retrieval with reciprocal rank fusion."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import threading

import numpy as np

from .config import Settings
from .db import Database


log = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, db: Database, config: Settings):
        self.db = db
        self.config = config
        self._lock = threading.RLock()
        self._model = None
        self._ids: np.ndarray | None = None
        self._vectors: np.ndarray | None = None
        self._faiss = None
        self._index = None
        self._db_signature: tuple[int, int] | None = None
        self.semantic_error: str | None = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def warm(self) -> None:
        """Prime the on-disk vector index and query encoder before first search."""
        with self._lock:
            self.rebuild()
            self._load_model().encode(["кино"], normalize_embeddings=True, show_progress_bar=False)

    def score_passages(self, query: str, passages: list[str]) -> list[float]:
        """Semantic similarity for short evidence passages using the warm encoder."""
        if not passages:
            return []
        with self._lock:
            embeddings = np.asarray(
                self._load_model().encode(
                    [query, *passages], normalize_embeddings=True,
                    show_progress_bar=False, batch_size=32,
                ), dtype=np.float32,
            )
        return (embeddings[1:] @ embeddings[0]).tolist()

    def rebuild(self, force: bool = False) -> int:
        with self._lock:
            signature = self.db.chunk_signature()
            if not force and self._ids is not None and self._db_signature == signature:
                return len(self._ids)
            rows = self.db.all_chunks()
            ids = np.array([int(row["id"]) for row in rows], dtype=np.int64)
            if not len(ids):
                self._ids = ids
                self._vectors = np.empty((0, 0), dtype=np.float32)
                self._index = None
                self._db_signature = signature
                return 0
            path = self.config.vector_path
            reusable: dict[int, np.ndarray] = {}
            if not force and path.exists():
                try:
                    with np.load(path) as saved:
                        old_ids = saved["ids"]
                        old_vectors = saved["vectors"]
                        if np.array_equal(ids, old_ids):
                            self._ids = old_ids
                            self._vectors = old_vectors
                            self._db_signature = signature
                            self._make_index()
                            return len(ids)
                        if old_vectors.ndim == 2 and len(old_ids) == len(old_vectors):
                            reusable = {int(ident): old_vectors[pos] for pos, ident in enumerate(old_ids)}
                except (OSError, ValueError, KeyError):
                    pass
            vector_dim = next((vector.shape[0] for vector in reusable.values()), 0)
            vectors = np.empty((len(ids), vector_dim), dtype=np.float32) if vector_dim else None
            missing_positions = []
            for pos, ident in enumerate(ids):
                previous = reusable.get(int(ident))
                if vectors is None or previous is None or previous.shape[0] != vector_dim:
                    missing_positions.append(pos)
                else:
                    vectors[pos] = previous
            if missing_positions:
                model = self._load_model()
                encoded = np.asarray(model.encode(
                    [rows[pos]["text"] for pos in missing_positions], batch_size=16,
                    show_progress_bar=False, normalize_embeddings=True,
                ), dtype=np.float32)
                if vectors is not None and vectors.shape[1] != encoded.shape[1]:
                    encoded = np.asarray(model.encode(
                        [row["text"] for row in rows], batch_size=16,
                        show_progress_bar=False, normalize_embeddings=True,
                    ), dtype=np.float32)
                    vectors = encoded
                else:
                    if vectors is None:
                        vectors = np.empty((len(ids), encoded.shape[1]), dtype=np.float32)
                    vectors[missing_positions] = encoded
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, ids=ids, vectors=vectors)
            temp.replace(path)
            self._ids, self._vectors = ids, vectors
            self._db_signature = signature
            self._make_index()
            return len(ids)

    def _make_index(self):
        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self._vectors.shape[1])
            self._index.add(self._vectors)
        except ImportError:
            self._faiss = None
            self._index = None

    def vector(self, query: str, limit: int = 30, entity: tuple[str, str] | None = None) -> list[tuple[int, float]]:
        try:
            self.rebuild()
            if self._ids is None or not len(self._ids):
                return []
            q = np.asarray(self._load_model().encode([query], normalize_embeddings=True), dtype=np.float32)
            if self._index is not None and entity is None:
                scores, positions = self._index.search(q, min(limit, len(self._ids)))
                pairs = [(int(self._ids[pos]), float(score)) for score, pos in zip(scores[0], positions[0]) if pos >= 0]
            else:
                scores = self._vectors @ q[0]
                if entity:
                    allowed = self.db.entity_chunk_ids(*entity)
                    positions = [pos for pos, ident in enumerate(self._ids) if int(ident) in allowed]
                    positions = sorted(positions, key=lambda pos: float(scores[pos]), reverse=True)[:limit]
                else:
                    positions = np.argsort(-scores)[:limit]
                pairs = [(int(self._ids[pos]), float(scores[pos])) for pos in positions]
            self.semantic_error = None
            return pairs[:limit]
        except Exception as exc:
            # Lexical search remains available if the local embedding model is not installed yet.
            self.semantic_error = f"Семантический поиск временно недоступен: {type(exc).__name__}: {exc}"
            log.exception("Semantic search unavailable")
            return []

    def search(
        self, query: str, *, limit: int = 5, mode: str = "hybrid",
        entity: tuple[str, str] | None = None,
    ) -> list[dict]:
        if mode not in {"hybrid", "bm25", "semantic"}:
            raise ValueError("mode must be hybrid, bm25 or semantic")
        limit = max(1, min(limit, 30))
        candidates = max(30, limit * 5)
        retrieval_query = query
        lowered_query = query.lower()
        if re.search(r"спрыг|прыгнул|выпал|суицид|самоубий|покончил", lowered_query):
            # Users describe the same plot event in many ways; an article may
            # say “покончила с собой” or “выпала из окна”, not “спрыгнула”.
            retrieval_query += " самоубийство покончила с собой выпала из окна погибла"
        if re.search(r"умер|погиб|убил|убийств|смерт|скончал", lowered_query):
            retrieval_query += " смерть погиб убийство убил умер"
        if re.search(r"финал|концовк|чем законч|что в конце", lowered_query):
            retrieval_query += " финал концовка развязка в конце"
        if re.search(r"почему|зачем", lowered_query) and re.search(r"сделал|поступил|ушёл|ушел|предал|убил|умер|погиб|спрыг|выбрал", lowered_query):
            retrieval_query += " причина мотив решение последствия"
        if re.search(r"кто\s+(?:такой|такая|такое|это)|кем\s+(?:является|приходится)", lowered_query):
            retrieval_query += " роль персонаж ученик ученица студент девушка сын дочь актёр актриса"
        if re.search(r"что означа|значени\w*|символиз|смысл", lowered_query):
            retrieval_query += " значение символ смысл талисман предмет образ значение в фильме"
        if re.search(r"двойник|двойн|клонир|клон|копи[яи]", lowered_query):
            retrieval_query += " клон клонирование копия двойник"
        if entity and entity[0] == "person" and re.search(
            r"конфликт|скандал|инцидент|ссор|разноглас|драк|стычк|обвин|арест|задерж|судебн|хулиган",
            lowered_query,
        ):
            retrieval_query += " инциденты скандал арест суд драка хулиганство укусил ударил обвинение"
        lexical = self.db.bm25(retrieval_query, candidates, entity) if mode in {"hybrid", "bm25"} else []
        semantic = self.vector(retrieval_query, candidates, entity) if mode in {"semantic", "hybrid"} else []
        ranks: dict[int, dict] = {}
        for rank, (ident, score) in enumerate(lexical, start=1):
            ranks.setdefault(ident, {"score": 0.0, "bm25_score": None, "semantic_score": None})
            ranks[ident]["score"] += (1.5 if mode == "hybrid" else 1.0) / (60 + rank)
            ranks[ident]["bm25_score"] = score
        for rank, (ident, score) in enumerate(semantic, start=1):
            ranks.setdefault(ident, {"score": 0.0, "bm25_score": None, "semantic_score": None})
            ranks[ident]["score"] += 1.0 / (60 + rank)
            ranks[ident]["semantic_score"] = score
        best = sorted(ranks, key=lambda ident: ranks[ident]["score"], reverse=True)[:limit]
        details = self.db.chunk_details(best)
        return [{"chunk_id": ident, **details[ident], **ranks[ident]} for ident in best]
