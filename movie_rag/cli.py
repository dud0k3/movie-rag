from __future__ import annotations

import argparse
import json

from .app import db, ingestor, search_engine


def main():
    parser = argparse.ArgumentParser(description="Movie RAG commands")
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed", help="Load popular foreign films from TMDB")
    seed.add_argument("--count", type=int, default=100)
    commands.add_parser("reindex", help="Build/rebuild the semantic vector index")
    commands.add_parser("enrich-ru", help="Add Russian Wikipedia articles to cached entities")
    commands.add_parser("stats", help="Show database counts")
    args = parser.parse_args()
    if args.command == "seed":
        def progress(index, total, item):
            suffix = f"ERROR {item['error']}" if "error" in item else item.get("title", "")
            print(f"[{index}/{total}] {suffix}", flush=True)
        result = ingestor.seed_popular(args.count, progress=progress)
        print(f"Loaded {result['loaded']}/{result['requested']} movies")
    elif args.command == "reindex":
        print(f"Indexed {search_engine.rebuild(force=True)} chunks")
    elif args.command == "enrich-ru":
        with db.connect() as connection:
            entities = connection.execute("SELECT entity_type, entity_id, name FROM entities ORDER BY entity_type, entity_id").fetchall()
        loaded = 0
        for index, entity in enumerate(entities, 1):
            found = ingestor.enrich_russian(entity["entity_type"], int(entity["entity_id"]))
            loaded += found
            print(f"[{index}/{len(entities)}] {entity['name']}: {'Wikipedia RU' if found else 'не найдено'}", flush=True)
        print(f"Loaded {loaded}/{len(entities)} Russian articles")
    else:
        print(json.dumps(db.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
