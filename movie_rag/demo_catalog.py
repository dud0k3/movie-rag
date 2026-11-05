"""A small, attributed starter corpus for running the service without TMDB."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from .db import Database


def _records() -> list[dict]:
    path = Path(__file__).with_name("demo_catalog.json")
    return json.loads(path.read_text(encoding="utf-8"))


def starter_cards() -> list[dict]:
    return [{"type": row["type"], "id": row["id"], "title": row["title"],
             "original_title": row["original_title"], "year": (row["date"] or "")[:4],
             "poster_path": None} for row in _records()]


def seed_demo(db: Database) -> None:
    """Index the prepared summaries with the same ingestion path as live articles."""
    for record in _records():
        kind, ident = record["type"], str(record["id"])
        if db.get_entity(kind, ident):
            continue
        title = record["title"]
        article = record["article"]
        url = "https://ru.wikipedia.org/wiki/" + quote(article.replace(" ", "_"))
        item = {"title": title, "original_title": record["original_title"], "demo_article": article,
                "demo_url": url}
        if kind == "movie":
            item["release_date"] = record["date"]
        elif kind == "tv":
            item.update(name=title, first_air_date=record["date"])
        else:
            item["name"] = title
        db.put_document(
            source="Wikipedia RU · демо-выжимка", source_id=article, entity_type=kind,
            entity_id=ident, title=title, url=url, text=record["text"], published_at=record["date"],
        )
        db.put_entity(kind, ident, title, item)
