from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .chunking import chunks, context_window


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,
  published_at TEXT,
  fetched_at TEXT NOT NULL,
  sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  text TEXT NOT NULL,
  UNIQUE(document_id, position)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, title, text, tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS entities (
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  name TEXT NOT NULL,
  payload TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY(entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_documents_entity ON documents(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def put_entity(self, kind: str, entity_id: str, name: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                """INSERT INTO entities VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                  name=excluded.name, payload=excluded.payload, fetched_at=excluded.fetched_at""",
                (kind, str(entity_id), name, json.dumps(payload, ensure_ascii=False), now),
            )

    def get_entity(self, kind: str, entity_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM entities WHERE entity_type=? AND entity_id=?",
                (kind, str(entity_id)),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def match_entity(self, query: str) -> tuple[str, str] | None:
        """Find an exact, unambiguous title or name in a free-form question."""
        from .sources import title_key

        question = f" {title_key(query)} "
        with self.connect() as db:
            rows = db.execute(
                """SELECT entity_type, entity_id, name,
                          json_extract(payload, '$.title_ru') AS title_ru,
                          json_extract(payload, '$.name_ru') AS name_ru,
                          json_extract(payload, '$.original_title') AS original_title,
                          json_extract(payload, '$.original_name') AS original_name
                   FROM entities WHERE entity_type IN ('movie', 'tv', 'person')"""
            ).fetchall()
        matches = {}
        for row in rows:
            for alias in list(row)[2:]:
                key = title_key(alias or "")
                if len(key) >= 4 and f" {key} " in question:
                    ident = (row["entity_type"], row["entity_id"])
                    matches[ident] = max(matches.get(ident, 0), len(key))
        if not matches:
            return None
        longest = max(matches.values())
        best = [entity for entity, length in matches.items() if length == longest]
        return best[0] if len(best) == 1 else None

    def put_document(
        self, *, source: str, source_id: str, entity_type: str, entity_id: str,
        title: str, url: str, text: str, published_at: str | None = None,
    ) -> tuple[int, bool]:
        normalized = " ".join(text.split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            old = db.execute("SELECT id, sha256 FROM documents WHERE url=?", (url,)).fetchone()
            if old and old["sha256"] == digest:
                db.execute("UPDATE documents SET fetched_at=? WHERE id=?", (now, old["id"]))
                return old["id"], False
            if old:
                ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE document_id=?", (old["id"],))]
                for chunk_id in ids:
                    db.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
                db.execute("DELETE FROM chunks WHERE document_id=?", (old["id"],))
                db.execute(
                    """UPDATE documents SET source=?, source_id=?, entity_type=?, entity_id=?,
                    title=?, text=?, published_at=?, fetched_at=?, sha256=? WHERE id=?""",
                    (source, source_id, entity_type, str(entity_id), title, normalized,
                     published_at, now, digest, old["id"]),
                )
                doc_id = old["id"]
            else:
                cursor = db.execute(
                    """INSERT INTO documents
                    (source, source_id, entity_type, entity_id, title, url, text,
                     published_at, fetched_at, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (source, source_id, entity_type, str(entity_id), title, url,
                     normalized, published_at, now, digest),
                )
                doc_id = cursor.lastrowid
            for pos, piece in enumerate(chunks(normalized)):
                cursor = db.execute(
                    "INSERT INTO chunks (document_id, position, text) VALUES (?, ?, ?)",
                    (doc_id, pos, piece),
                )
                db.execute(
                    "INSERT INTO chunks_fts (chunk_id, title, text) VALUES (?, ?, ?)",
                    (cursor.lastrowid, title, piece),
                )
            return doc_id, True

    def remove_other_source_documents(self, entity_type: str, entity_id: str, source: str, keep_url: str) -> None:
        """Remove stale article matches when source resolution changes."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT id FROM documents WHERE entity_type=? AND entity_id=?
                AND source=? AND url<>?""",
                (entity_type, str(entity_id), source, keep_url),
            ).fetchall()
            for row in rows:
                chunk_ids = db.execute("SELECT id FROM chunks WHERE document_id=?", (row["id"],)).fetchall()
                for chunk in chunk_ids:
                    db.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk["id"],))
                db.execute("DELETE FROM documents WHERE id=?", (row["id"],))

    def all_chunks(self) -> list[sqlite3.Row]:
        with self.connect() as db:
            return db.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()

    def chunk_count(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def chunk_signature(self) -> tuple[int, int]:
        """Cheap index freshness marker; chunk IDs change whenever text is replaced."""
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
        return int(row[0]), int(row[1])

    def entity_chunk_ids(self, entity_type: str, entity_id: str) -> set[int]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE d.entity_type=? AND d.entity_id=?""",
                (entity_type, entity_id),
            ).fetchall()
        return {int(row[0]) for row in rows}

    def stats(self) -> dict:
        with self.connect() as db:
            return {
                "documents": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "movies": db.execute("SELECT COUNT(*) FROM entities WHERE entity_type='movie'").fetchone()[0],
                "series": db.execute("SELECT COUNT(*) FROM entities WHERE entity_type='tv'").fetchone()[0],
                "people": db.execute("SELECT COUNT(*) FROM entities WHERE entity_type='person'").fetchone()[0],
            }

    def chunk_details(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT c.id, c.text, d.title, d.url, d.source, d.entity_type,
                d.entity_id, d.published_at FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE c.id IN ({placeholders})""", ids,
            ).fetchall()
        return {int(r["id"]): dict(r) for r in rows}

    def chunk_windows(self, ids: list[int]) -> dict[int, str]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT c.id, c.text, prev.text AS prev, following.text AS following
                    FROM chunks c
                    LEFT JOIN chunks prev
                        ON prev.document_id=c.document_id AND prev.position=c.position-1
                    LEFT JOIN chunks following
                        ON following.document_id=c.document_id AND following.position=c.position+1
                    WHERE c.id IN ({placeholders})""", ids,
            ).fetchall()
        return {row["id"]: context_window(row["prev"] or "", row["text"], row["following"] or "")
                for row in rows if row["prev"] or row["following"]}

    def catalog_chunk(self, entity: tuple[str, str]) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT c.id AS chunk_id, c.text, d.title, d.url, d.source, d.entity_type,
                          d.entity_id, d.published_at
                   FROM documents d JOIN chunks c ON c.document_id=d.id
                   WHERE d.entity_type=? AND d.entity_id=? AND d.source='TMDB' AND c.position=0""",
                entity,
            ).fetchone()
        return dict(row) if row else None

    def bm25(self, query: str, limit: int = 30, entity: tuple[str, str] | None = None) -> list[tuple[int, float]]:
        import re
        stopwords = {"кто", "что", "как", "где", "когда", "какой", "какая", "какие",
                     "фильм", "фильма", "фильме", "фильму", "про", "его", "ее", "её",
                     "the", "movie", "film", "was", "who", "and"}
        terms = [term for term in re.findall(r"\w+", query.lower(), re.UNICODE)
                 if len(term) >= 2 and term not in stopwords][:16]
        if not terms:
            return []
        expanded = set(terms)
        for term in terms:
            # Russian FTS does not equate е and ё, including in inflected titles.
            expanded.add(term.replace("ё", "е"))
            for position, letter in enumerate(term):
                if letter == "е":
                    expanded.add(term[:position] + "ё" + term[position + 1:])
            # Common diminutive and case forms of Marat should retrieve the
            # TMDB cast entry, which lists the character as “Marat Suvorov”.
            if term in {"маратик", "маратика", "маратику", "маратиком", "маратике", "maratik", "maratika", "maratiku", "maratikom", "maratike"}:
                expanded.update({"марат", "marat"})
        expression = " OR ".join(f'"{term}"' for term in sorted(expanded))
        sql = """SELECT CAST(f.chunk_id AS INTEGER) AS id, bm25(chunks_fts, 0, 8, 1) AS score
        FROM chunks_fts f JOIN chunks c ON c.id=CAST(f.chunk_id AS INTEGER)
        JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ?"""
        params: list = [expression]
        if entity:
            sql += " AND d.entity_type=? AND d.entity_id=?"
            params.extend(entity)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        with self.connect() as db:
            rows = db.execute(sql, params).fetchall()
            if entity and len(rows) < min(limit, 4):
                # Scoped fallback for Russian inflection. Keep exact matches
                # first, and never broaden the search to other titles.
                endings = r"(?:ами|ями|ого|его|ому|ему|ыми|ими|ая|яя|ую|юю|ых|их|ой|ей|ам|ям|ом|ем|ов|ев|[аяуюыиео])$"
                stems = {re.sub(endings, "", term) for term in expanded
                         if re.fullmatch(r"[а-яё]{5,}", term)}
                stems = {stem for stem in stems if len(stem) >= 4}
                if stems:
                    prefix_match = " OR ".join(f'"{stem}"*' for stem in sorted(stems))
                    extra = db.execute(sql, [prefix_match, *params[1:]]).fetchall()
                    exact_ids = {row["id"] for row in rows}
                    rows.extend(row for row in extra if row["id"] not in exact_ids)
        return [(int(r["id"]), float(r["score"])) for r in rows[:limit]]
