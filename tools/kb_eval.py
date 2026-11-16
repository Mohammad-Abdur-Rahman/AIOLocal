"""Validate the local knowledge corpus with project-shaped questions.

This is the check that matters after downloading documents: does the library
actually answer questions about this project, and does automatic folder routing
pick the right folder instead of the whole library?

It reads the real SQLite knowledge database (no server required), optionally
scans the inbox first, rebuilds folder summaries and prints:

* folder hit@3 — the expected folder was routed or appears in the top 3 results;
* chunk hit@5 — a top-5 chunk contains an expected term;
* negative control — an unrelated question must return nothing.

Exit code is 1 when a threshold is missed, so it can be used in CI.

Usage::

    python tools/kb_eval.py --scan
    python tools/kb_eval.py --data-dir "%LOCALAPPDATA%\\AIOLocal"
    python tools/kb_eval.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiolocal.config import Paths, Settings  # noqa: E402
from aiolocal.knowledge import KnowledgeBase  # noqa: E402

QUERIES = Path(__file__).resolve().parent / "kb_eval_queries.json"


def folder_match(expected: str, slug: str) -> bool:
    """A routed parent folder still owns a document in a nested child folder."""
    return (
        slug == expected
        or slug.startswith(expected + "/")
        or expected.startswith(slug + "/")
    )


def combined(item: dict) -> str:
    return " ".join(
        str(item.get(key) or "") for key in ("title", "heading", "text")
    ).lower()


def evaluate(kb: KnowledgeBase, payload: dict, *, limit: int, embed_dim: int) -> dict:
    positives = []
    for case in payload.get("positive") or []:
        started = time.perf_counter()
        result = kb.search_auto(case["query"], limit=limit, embed_dim=embed_dim)
        elapsed = time.perf_counter() - started
        items = result.get("items") or []
        routing = result.get("routing") or {}
        routed = routing.get("collections") or []
        result_folders = [str(item.get("collection") or "") for item in items[:3]]
        routed_hit = any(folder_match(case["folder"], slug) for slug in routed)
        result_hit = any(folder_match(case["folder"], slug) for slug in result_folders)
        terms = [str(term).lower() for term in case.get("terms") or []]
        text = "\n".join(combined(item) for item in items[:limit])
        matched_terms = [term for term in terms if term in text]
        chunk_hit = bool(matched_terms) if terms else bool(items)
        positives.append(
            {
                "query": case["query"],
                "expected_folder": case["folder"],
                "folder_hit": routed_hit or result_hit,
                "routed_folder_hit": routed_hit,
                "chunk_hit": chunk_hit,
                "matched_terms": matched_terms,
                "routed": routed,
                "fallback": routing.get("fallback"),
                "abstained": routing.get("abstained"),
                "top": [
                    {
                        "collection": item.get("collection"),
                        "title": item.get("title"),
                        "score": item.get("score"),
                    }
                    for item in items[:3]
                ],
                "ms": round(elapsed * 1000, 1),
            }
        )

    negatives = []
    for query in payload.get("negative") or []:
        result = kb.search_auto(query, limit=limit, embed_dim=embed_dim)
        items = result.get("items") or []
        routing = result.get("routing") or {}
        abstained = bool(routing.get("abstained")) or not items
        negatives.append(
            {
                "query": query,
                "pass": abstained,
                "items": len(items),
                "routing": routing.get("mode"),
                "top": [
                    {"collection": item.get("collection"), "title": item.get("title")}
                    for item in items[:2]
                ],
            }
        )

    folder_rate = (
        sum(1 for row in positives if row["folder_hit"]) / len(positives)
        if positives
        else 0.0
    )
    chunk_rate = (
        sum(1 for row in positives if row["chunk_hit"]) / len(positives)
        if positives
        else 0.0
    )
    negative_rate = (
        sum(1 for row in negatives if row["pass"]) / len(negatives)
        if negatives
        else 1.0
    )
    return {
        "positives": positives,
        "negatives": negatives,
        "metrics": {
            "queries": len(positives),
            "folder_hit_at_3": round(folder_rate, 4),
            "chunk_hit_at_5": round(chunk_rate, 4),
            "negative_control": round(negative_rate, 4),
            "average_ms": round(
                sum(row["ms"] for row in positives) / len(positives), 1
            )
            if positives
            else 0.0,
        },
    }


def print_report(report: dict) -> None:
    print("Query                                                            folder  chunk   route")
    print("-" * 100)
    for row in report["positives"]:
        print(
            f"{row['query'][:64]:<64} "
            f"{'ok ' if row['folder_hit'] else 'MISS':<6} "
            f"{'ok ' if row['chunk_hit'] else 'MISS':<6} "
            f"{', '.join(row['routed'][:3]) or ('global' if row['fallback'] else '-')}"
        )
        if not row["folder_hit"] or not row["chunk_hit"]:
            print(
                f"    expected {row['expected_folder']} | terms {row['matched_terms']} | "
                f"top {[(t['collection'], t['title']) for t in row['top']]}"
            )
    print()
    for row in report["negatives"]:
        print(
            f"negative: {row['query'][:64]:<64} "
            f"{'pass' if row['pass'] else 'FAIL'} ({row['items']} items)"
        )
        if not row["pass"]:
            print(f"    top {row['top']}")
    metrics = report["metrics"]
    print()
    print(
        f"folder hit@3: {metrics['folder_hit_at_3']:.1%}  "
        f"chunk hit@5: {metrics['chunk_hit_at_5']:.1%}  "
        f"negative control: {metrics['negative_control']:.1%}  "
        f"avg {metrics['average_ms']} ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the AIOLocal knowledge corpus.")
    parser.add_argument("--data-dir", default="", help="app data folder (default: automatic)")
    parser.add_argument("--db", default="", help="override the knowledge.db path")
    parser.add_argument("--queries", default=str(QUERIES), help="query JSON file")
    parser.add_argument("--scan", action="store_true", help="scan the inbox before evaluating")
    parser.add_argument("--limit", type=int, default=5, help="results per query (default 5)")
    parser.add_argument("--embed-dim", type=int, default=0, help="override embedding dimension")
    parser.add_argument("--no-rebuild", action="store_true", help="skip folder summary rebuild")
    parser.add_argument("--folder-threshold", type=float, default=0.90)
    parser.add_argument("--chunk-threshold", type=float, default=0.85)
    parser.add_argument("--json", default="", help="write the full report to this file")
    args = parser.parse_args(argv)

    paths = Paths.create(args.data_dir or None)
    db_path = Path(args.db).expanduser().resolve() if args.db else paths.knowledge_db_path
    if not db_path.exists():
        print(f"knowledge database not found: {db_path}", file=sys.stderr)
        print("Run tools/kb_sync.py sync and pass --scan.", file=sys.stderr)
        return 2
    kb = KnowledgeBase(db_path, files_dir=paths.knowledge_files)
    settings = Settings()
    embed_dim = max(64, min(int(args.embed_dim or settings.embed_dim), 2048))
    try:
        if args.scan:
            print(f"Scanning {paths.knowledge_inbox} …", flush=True)
            result = kb.scan_inbox(
                paths.knowledge_inbox, embed_dim=embed_dim, limit=0
            )
            print(
                f"  added {result['added']}, skipped {result['skipped']}, "
                f"failed {len(result['failed'])} in {result['elapsed_ms'] / 1000:.1f}s"
            )
            for failure in result["failed"][:10]:
                print(f"  ! {failure['file']}: {failure['error']}")
        if not args.no_rebuild:
            rebuilt = kb.rebuild_folder_summaries(embed_dim=embed_dim)
            print(f"Folder index: {rebuilt['updated']} summaries, {rebuilt['empty']} empty")
        stats = kb.stats()
        print(
            f"Corpus: {stats['documents']} documents, {stats['chunks']} chunks, "
            f"{stats['collections']} folders"
        )
        payload = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        report = evaluate(kb, payload, limit=args.limit, embed_dim=embed_dim)
        print_report(report)
        if args.json:
            Path(args.json).write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(f"Report written to {args.json}")
    finally:
        kb.close()

    metrics = report["metrics"]
    failed = (
        metrics["folder_hit_at_3"] < args.folder_threshold
        or metrics["chunk_hit_at_5"] < args.chunk_threshold
        or metrics["negative_control"] < 1.0
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
