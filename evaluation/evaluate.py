"""Entity-level Recall@K: does search retrieve a chunk for the expected film?"""

import json
from pathlib import Path

from movie_rag.app import db, search_engine


def main():
    cases = json.loads((Path(__file__).parent / "questions.json").read_text())
    loaded = [case for case in cases if db.get_entity("movie", str(case["movie_id"]))]
    if not loaded:
        raise SystemExit("No evaluation films are loaded. Run the seed command first.")
    output = {"questions_total": len(cases), "questions_loaded": len(loaded), "modes": {}}
    for mode in ("bm25", "semantic", "hybrid"):
        rows = []
        for case in loaded:
            hits = search_engine.search(case["q"], limit=10, mode=mode)
            ranks = [rank for rank, hit in enumerate(hits, 1)
                     if hit["entity_type"] == "movie" and hit["entity_id"] == str(case["movie_id"])]
            rows.append({"question": case["q"], "expected_movie_id": case["movie_id"],
                         "rank": min(ranks) if ranks else None})
        output["modes"][mode] = {
            "recall_at_1": sum(row["rank"] == 1 for row in rows) / len(rows),
            "recall_at_5": sum(row["rank"] is not None and row["rank"] <= 5 for row in rows) / len(rows),
            "recall_at_10": sum(row["rank"] is not None and row["rank"] <= 10 for row in rows) / len(rows),
            "rows": rows,
        }
    output["semantic_error"] = search_engine.semantic_error
    path = Path(__file__).parent / "results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"questions_loaded": output["questions_loaded"],
                      "metrics": {mode: {k: v for k, v in data.items() if k != "rows"}
                                  for mode, data in output["modes"].items()},
                      "semantic_error": output["semantic_error"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
