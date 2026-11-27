"""Strict-index smoke test through the real HTTP API.

Creates a throwaway conversation, attaches an already-analysed document, runs
the 17-topic strict classification job from the user's verbatim prompt, then
validates the produced index: 17 lines in the user's order, sorted and
deduplicated per topic, `None` for empty topics, and every page accounted for.

Usage:
    python dev_strict.py [document_id]
"""
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, r"C:\Users\Dargater\Desktop\OpenCode Project")
from aiolocal.config import Paths
from aiolocal.security import load_or_create_token
from aiolocal.tasks import _looks_like_furniture_page

BASE = "http://127.0.0.1:7322"
TOKEN = load_or_create_token(Paths.create(None).token_path)
ARTIFACTS = os.path.expandvars(r"%LOCALAPPDATA%\AIOLocal\artifacts\tasks")
MEMORY_DB = os.path.expandvars(r"%LOCALAPPDATA%\AIOLocal\data\memory.db")
PROMPT_FILE = os.path.join(
    r"C:\Users\Dargater\Desktop\OpenCode Project", "prompts", "topic-index-4024.md"
)

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


def user_instruction() -> str:
    """The user's own prompt, minus the file's title and note headers."""
    text = open(PROMPT_FILE, encoding="utf-8").read()
    marker = "You are a precise mathematics exam-indexing assistant."
    at = text.find(marker)
    return text[at:].strip() if at >= 0 else text.strip()


def call(method, path, body=None, headers=None):
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def furniture_pages(document_id: str, total: int) -> set[int]:
    """Blank/cover pages that may legitimately stay unclassified."""
    con = sqlite3.connect(MEMORY_DB)
    rows = con.execute(
        "SELECT page_no, text FROM document_pages WHERE document_id=? ORDER BY page_no",
        (document_id,),
    ).fetchall()
    con.close()
    pages = {int(no) for no, text in rows if _looks_like_furniture_page(text or "")}
    present = {int(no) for no, _ in rows}
    pages |= {p for p in range(1, total + 1) if p not in present}
    return pages


def wait_for_model(call_fn, model: str, timeout: float = 240.0) -> bool:
    """Wait until the upstream lists ``model``.

    ``healthz`` answers as soon as the app is up, which can be before the model
    server has finished loading. Starting a task in that window makes every
    first call fail, so wait for the actual model list instead.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            upstream = (call_fn("GET", "/api/v1/health") or {}).get("upstream") or {}
        except Exception:
            upstream = {}
        if model in (upstream.get("models") or []):
            return True
        time.sleep(3.0)
    return False


def main() -> int:
    document = sys.argv[1] if len(sys.argv) > 1 else "1795486226117cc5af49ef173a5c6"
    instruction = user_instruction()
    print(f"instruction: {len(instruction)} chars, {len(TOPICS)} topics", flush=True)

    health = call("GET", "/api/v1/health")
    print("upstream:", json.dumps(health.get("upstream"))[:300], flush=True)

    vision_model = "qwen3-vl-8b-instruct-int4-ov"
    if not wait_for_model(call, vision_model):
        print(f"warning: {vision_model} not listed after waiting", flush=True)

    conv = call("POST", "/api/v1/conversations")
    conv_id = conv["id"]
    print("conversation:", conv_id, flush=True)
    scoped = call("POST", f"/api/v1/documents/scope/{document}?conversation_id={conv_id}")
    print("scoped:", json.dumps(scoped)[:300], flush=True)

    started = time.time()
    task = call(
        "POST",
        "/api/v1/tasks",
        {
            "conversation_id": conv_id,
            "document_ids": [document],
            "instruction": instruction,
            "topics": TOPICS,
            "title": "topic index (strict)",
            "model": "qwen3-vl-8b-instruct-int4-ov",
        },
    )["task"]
    task_id = task["id"]
    print(
        f"task: {task_id} strict={task.get('strict')} units={task.get('total_units')}",
        flush=True,
    )

    last = None
    while True:
        task = call("GET", f"/api/v1/tasks/{task_id}")
        key = (task.get("cursor"), task.get("state"))
        if key != last:
            print(
                f"[{time.time()-started:7.1f}s] {task.get('state')} "
                f"{task.get('cursor')}/{task.get('total_units')} err={task.get('error')!r}",
                flush=True,
            )
            last = key
        if task.get("state") in ("done", "error", "cancelled", "interrupted"):
            break
        time.sleep(1.0)

    if task.get("state") != "done":
        print("TASK DID NOT FINISH:", json.dumps(task, indent=2)[:2000])
        return 1

    path = os.path.join(ARTIFACTS, f"task-{task_id}.md")
    print("artifact:", path, os.path.exists(path), flush=True)
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    print(f"lines: {len(lines)}", flush=True)
    for ln in lines:
        print(" ", ln)

    # Validate the deliverable.
    problems = []
    total_pages = int(task.get("total_units") or 0)
    if len(lines) != len(TOPICS):
        problems.append(f"expected {len(TOPICS)} lines, got {len(lines)}")
    seen: list[int] = []
    for index, ln in enumerate(lines):
        # The topic name itself may contain colons; the page list is the last
        # ":"-separated field that starts with a digit.
        match = re.match(r"^(?P<name>.+?):\s*(?P<rest>\d[\d,\s]*|None)\s*$", ln)
        if not match:
            problems.append(f"bad line format: {ln!r}")
            continue
        name, rest = match.group("name").strip(), match.group("rest").strip()
        if index < len(TOPICS) and name != TOPICS[index]:
            problems.append(f"topic {index + 1} is {name!r}, expected {TOPICS[index]!r}")
        if name not in TOPICS:
            problems.append(f"unknown topic {name!r}")
        if rest != "None":
            pages = [int(p) for p in rest.replace(" ", "").split(",") if p]
            if pages != sorted(pages):
                problems.append(f"unsorted pages for {name}")
            if len(pages) != len(set(pages)):
                problems.append(f"duplicate page within {name}")
            seen.extend(pages)
    missing = sorted(set(range(1, total_pages + 1)) - set(seen))
    out_of_range = sorted({p for p in seen if p < 1 or p > total_pages})
    furniture = furniture_pages(document, total_pages)
    unexplained = [p for p in missing if p not in furniture]
    print(
        f"pages assigned: {len(set(seen))}/{total_pages}, "
        f"multi-topic repeats={len(seen) - len(set(seen))}",
        flush=True,
    )
    print(
        f"unassigned furniture (blank/cover, expected): "
        f"{[p for p in missing if p in furniture]}",
        flush=True,
    )
    print(f"unassigned unexplained: {unexplained[:40]}", flush=True)
    if out_of_range:
        problems.append(f"out-of-range pages {out_of_range}")
    if unexplained:
        problems.append(f"real question pages left unclassified: {unexplained}")

    print("RESULT:", "PASS" if not problems else "FAIL " + "; ".join(problems), flush=True)
    print(f"elapsed: {time.time()-started:.1f}s", flush=True)
    print(f"task id: {task_id}", flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
