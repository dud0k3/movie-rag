"""Run the same fixed set against the API's retrieval and streaming code."""
import argparse
import json
import math
import re
from pathlib import Path
from time import perf_counter


def matches(text, patterns):
    return all(re.search(pattern, text, re.I) for pattern in patterns)


def p95(values):
    ordered = sorted(values)
    return round(ordered[math.ceil(len(ordered) * .95) - 1], 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="Fetch missing benchmark entities from TMDB and Wikipedia")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cinema_cases.json"))
    args = parser.parse_args()
    from movie_rag.app import _answer_hits, db, ingestor, search_engine, settings
    from movie_rag.answer import stream_generate, warm_model

    cases = json.loads(args.cases.read_text())
    if args.prepare:
        for kind, ident in dict.fromkeys(tuple(case["entity"]) for case in cases):
            if not db.get_entity(kind, str(ident)):
                print(ingestor.ingest(kind, ident), flush=True)
    missing = [c["id"] for c in cases if not db.get_entity(c["entity"][0], str(c["entity"][1]))]
    if missing:
        raise SystemExit(f"Use --prepare to fetch missing entities: {', '.join(missing)}")
    search_engine.warm()
    warm_model(settings)
    rows = []
    for case in cases:
        question, (kind, ident), facts = case["q"], case["entity"], case["facts"]
        start = perf_counter()
        hits = _answer_hits(question, kind, ident, "hybrid")
        retrieval = perf_counter() - start
        first = None
        for event in stream_generate(question, hits, settings, search_engine.score_passages):
            if first is None and event["type"] in ("delta", "done"):
                first = perf_counter() - start
        assert event["type"] == "done"
        answer = event["answer"]
        total = perf_counter() - start
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        evidence = " ".join(hit["text"] for hit in hits)
        covered = sum(matches(answer, [pattern]) for pattern in facts)
        contradictions = [pattern for pattern in case.get("forbidden", []) if matches(answer, [pattern])]
        rows.append({
            "id": case["id"], "answer": answer, "mode": event["answer_mode"],
            "facts_covered": covered, "facts_expected": len(facts),
            "known_contradictions": contradictions,
            "fact_check": (covered == len(facts) and not contradictions) if facts else bool(re.search(
                r"не\s+(?:удалось подтвердить|указ|упомин|раскры|сообщ|извест|наш|удалось|привед|фигурир)|нет\b|отсутств", answer, re.I,
            ) and not re.search(r"\b\d{4}\b", answer)),
            "evidence_recall": matches(evidence, facts) if facts else None,
            "citation_ids_valid": bool(cited) and cited <= set(range(1, len(hits) + 1)) if facts else None,
            "cited_urls": sorted({hits[i-1]["url"] for i in cited if 1 <= i <= len(hits)}),
            "retrieval_s": round(retrieval, 3), "first_text_s": round(first, 3),
            "total_s": round(total, 3), **event.get("usage", {}),
        })
        print(case["id"], rows[-1]["fact_check"], rows[-1]["total_s"], answer, flush=True)
    answerable = [r for r in rows if r["evidence_recall"] is not None]
    report = {
        "model": settings.ollama_model, "corpus": db.stats(), "questions": len(rows),
        "fact_checks_passed": sum(r["fact_check"] for r in rows),
        "fact_coverage": sum(r["facts_covered"] for r in rows) / sum(r["facts_expected"] for r in rows),
        "evidence_recall": sum(r["evidence_recall"] for r in answerable) / len(answerable),
        "valid_citation_ids": sum(r["citation_ids_valid"] for r in answerable),
        "retrieval_p95_s": p95([r["retrieval_s"] for r in rows]),
        "first_text_p95_s": p95([r["first_text_s"] for r in rows]),
        "total_p95_s": p95([r["total_s"] for r in rows]),
        "complete_under_5s": sum(r["total_s"] <= 5 for r in rows),
        "passed_under_5s": sum(r["total_s"] <= 5 and r["fact_check"] for r in rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
