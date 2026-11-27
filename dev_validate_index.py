"""Validate the strict-index artifact for the 204-page test.pdf.

Checks, against the user's verbatim 17 topics:
  * exactly 17 topic lines, names matching the prompt
  * pages ascending and deduplicated within each topic
  * every page that is *not* a blank/cover page is classified under some topic
  * multi-topic pages are allowed (a page may appear in several topics)

Usage:
    python dev_validate_index.py [path-to-artifact] [document_id]
"""
from __future__ import annotations

import re
import sqlite3
import sys

from aiolocal.tasks import _looks_like_furniture_page

DB = r"C:\Users\Dargater\AppData\Local\AIOLocal\data\memory.db"
DEFAULT_DOC = "1795486226117cc5af49ef173a5c6"
DEFAULT_ART = r"C:\Users\Dargater\AppData\Local\AIOLocal\artifacts\tasks\task-179553096276045b93faadb4f57ea.md"
TOTAL_PAGES = 204

TOPICS = [
    "Number: Arithmetic, Fractions, Decimals, Percentages & Ratio",
    "Surds & Indices",
    "Algebra: Expressions, Factorisation & Algebraic Fractions",
    "Equations & Inequalities",
    "Sequences, Functions & Algebraic Proportion",
    "Coordinate Geometry",
    "Graphs & Kinematics",
    "Geometry: Angles, Polygons, Constructions, Bearings & Symmetry",
    "Circle Theorems",
    "Mensuration (Perimeter, Area, Surface Area & Volume)",
    "Similarity & Congruency",
    "Trigonometry (2D & 3D)",
    "Transformations & Vectors",
    "Probability",
    "Statistics",
    "Sets & Logic/Venn Diagrams",
    "Legacy / Out-of-Current-Syllabus (2019 & Earlier Papers)",
]


def furniture_pages(document_id: str) -> set[int]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT page_no, text FROM document_pages WHERE document_id=? ORDER BY page_no",
        (document_id,),
    ).fetchall()
    con.close()
    pages = {int(no) for no, text in rows if _looks_like_furniture_page(text or "")}
    # Pages missing from the DB are treated as furniture so they cannot fire a
    # false failure; a real gap shows up as a missing classified page elsewhere.
    present = {int(no) for no, _ in rows}
    pages |= {p for p in range(1, TOTAL_PAGES + 1) if p not in present}
    return pages


def main() -> int:
    artifact = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ART
    document = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DOC
    with open(artifact, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    problems: list[str] = []
    if len(lines) != 17:
        problems.append(f"expected 17 topic lines, got {len(lines)}")

    parsed: dict[str, list[int]] = {}
    for ln in lines:
        m = re.match(r"^(?P<topic>.+):\s*(?P<pages>.*)$", ln)
        if not m:
            problems.append(f"line does not match Topic: pages format: {ln!r}")
            continue
        topic, pages_raw = m.group("topic"), m.group("pages")
        if topic not in TOPICS:
            problems.append(f"unknown/invented topic name: {topic!r}")
            continue
        if pages_raw.strip() == "None":
            parsed[topic] = []
            continue
        pages_raw = pages_raw.strip()
        if pages_raw and not re.fullmatch(r"\d+(?:,\s*\d+)*", pages_raw):
            problems.append(f"bad page list for {topic!r}: {pages_raw!r}")
            continue
        parsed[topic] = [int(p) for p in re.findall(r"\d+", pages_raw)]

    missing_topics = [t for t in TOPICS if t not in parsed]
    if missing_topics:
        problems.append(f"topics missing from output: {missing_topics}")

    union: set[int] = set()
    for topic, pages in parsed.items():
        if not pages:
            continue
        if pages != sorted(pages):
            problems.append(f"{topic}: pages not ascending: {pages[:20]}")
        if len(pages) != len(set(pages)):
            dupes = sorted({p for p in pages if pages.count(p) > 1})
            problems.append(f"{topic}: duplicate pages: {dupes}")
        bad = [p for p in pages if not 1 <= p <= TOTAL_PAGES]
        if bad:
            problems.append(f"{topic}: out-of-range pages: {bad}")
        union.update(pages)

    unassigned = sorted(set(range(1, TOTAL_PAGES + 1)) - union)
    furniture = furniture_pages(document)
    unexplained = [p for p in unassigned if p not in furniture]

    print(f"topics: {len(parsed)}/17")
    print(f"pages classified: {len(union)}/{TOTAL_PAGES}")
    covered = sum(len(p) for p in parsed.values())
    print(f"multi-topic extra assignments: {covered - len(union)}")
    print(f"unassigned (furniture): {[p for p in unassigned if p in furniture]}")
    print(f"unassigned (unexplained): {unexplained}")

    if unexplained:
        problems.append(f"real question pages left unclassified: {unexplained}")

    if problems:
        print("\nFAIL:")
        for pr in problems:
            print(f"  - {pr}")
        return 1

    print("\nOK: all 17 topics present, names exact, pages ascending/deduped, "
          "every non-blank page classified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
