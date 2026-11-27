"""Diagnose the 8 pages the repair loop failed to classify."""
from __future__ import annotations

import json
import sqlite3

import httpx

from aiolocal.tasks import _looks_like_furniture_page, STRICT_PAGE_PROMPT, STRICT_INDEX_SYSTEM

DB = r"C:\Users\Dargater\AppData\Local\AIOLocal\data\memory.db"
DOC = "1795486226117cc5af49ef173a5c6"
BASE = "http://127.0.0.1:8000/v3"
MODEL = "qwen3-vl-8b-instruct-int4-ov"
TARGETS = [1, 2, 3, 27, 29, 62, 159, 186]

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


def ask(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": STRICT_INDEX_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = httpx.post(BASE + "/chat/completions", json=payload, timeout=120, trust_env=False)
    return r.json()["choices"][0]["message"].get("content") or ""


def main() -> int:
    con = sqlite3.connect(DB)
    for page in TARGETS:
        row = con.execute(
            "SELECT text FROM document_pages WHERE document_id=? AND page_no=?", (DOC, page)
        ).fetchone()
        text = (row[0] if row else "") or ""
        furn = _looks_like_furniture_page(text)
        snippet = " ".join(text.split())[:120]
        print(f"\n===== p{page}: furniture={furn} len={len(text)}")
        print(f"  text: {snippet}")
        prompt = STRICT_PAGE_PROMPT.format(
            topics="\n".join(f"- {t}" for t in TOPICS),
            page=page,
            segment=text[:12000],
        )
        out = ask(prompt)
        print(f"  model says: {out!r}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
