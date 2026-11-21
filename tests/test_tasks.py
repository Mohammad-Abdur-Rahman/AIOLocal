from __future__ import annotations

import time
from pathlib import Path

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.db import Database
from aiolocal.memory import store
from aiolocal.tasks import ResponseTaskWorker, looks_like_bulk_task


def make_env(tmp_path: Path) -> tuple[ResponseTaskWorker, Database, Paths]:
    paths = Paths.create(tmp_path / "data")
    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b"))
    db = Database(paths.db_path)
    worker = ResponseTaskWorker(paths, settings_store, db)
    return worker, db, paths


def add_document(db: Database, pages: int, *, title: str = "Questions") -> str:
    summary_id = store.insert_summary(
        db,
        title=f"Document: {title}",
        body="# Document\n\n## Overview\n_analysis complete_",
        topic_slug="documents",
        kind="document",
        source="document",
    )
    with db.write() as c:
        for page in range(1, pages + 1):
            c.execute(
                """INSERT INTO document_pages(document_id, page_no, text, tokens, created_at)
                   VALUES(?,?,?,?,?)""",
                (summary_id, page, f"Page {page}: question {page} about topic {page % 3}", 8, 1),
            )
            c.execute(
                """INSERT INTO document_page_notes(document_id, page_no, note, prev_summary,
                       tokens, complete, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (summary_id, page, f"Page {page}: question {page}", None, 8, 1, 1),
            )
    return summary_id


def wait_for(worker: ResponseTaskWorker, task_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        current = worker.get(task_id)
        assert current is not None
        last = current
        if current["state"] in ("done", "error", "cancelled", "interrupted"):
            return current
        time.sleep(0.02)
    raise AssertionError(f"task did not finish: {last}")


def test_bulk_detector_only_fires_on_exhaustive_document_jobs():
    assert looks_like_bulk_task(
        "Categorise every question in this document by topic", total_pages=80
    )
    assert looks_like_bulk_task(
        "Go through all pages and list each question", total_pages=30
    )
    # A normal question about a big document must stay single-shot.
    assert not looks_like_bulk_task("What does page 12 say?", total_pages=584)
    assert not looks_like_bulk_task("Categorise every question", total_pages=5)


def test_task_appends_sections_and_never_rewrites(tmp_path: Path):
    worker, db, _ = make_env(tmp_path)
    document_id = add_document(db, 12)
    prompts: list[str] = []

    def fake_ask(settings, model, prompt, *, max_tokens):
        prompts.append(prompt)
        if "table of contents" in prompt.lower():
            return "## Introduction\n\nThis draft groups the questions by topic."
        return "COVERED: 1-6\n\n## Topic A\n- question"

    worker._ask = fake_ask
    task = worker.start(
        conversation_id="chat-1",
        instruction="Categorise every question by topic",
        topics=["A", "B"],
        document_ids=[document_id],
        batch=6,
    )
    current = wait_for(worker, task["id"])
    assert current["state"] == "done", current
    assert current["cursor"] == 12
    assert current["summary_id"]

    draft = worker.draft(task["id"])
    assert draft is not None
    assert len(draft["chunks"]) == 2
    # The draft is the concatenation of what was produced, in order.
    assert draft["text"].count("## Topic A") == 2
    assert "COVERED:" not in draft["text"]

    # The final section prompt only ever contains the unfinished segment, never
    # the whole document.
    assert all(len(prompt) < 12_000 for prompt in prompts)

    summary = store.get_summary(db, current["summary_id"])
    assert summary is not None
    assert summary["kind"] == "task"
    assert "Section index" in summary["body"]

    path = worker.paths.artifact_dir / "tasks" / f"task-{task['id']}.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "## Draft" in text
    # The draft body itself: two sections, no leftover marker.
    body = text.split("## Draft", 1)[1]
    assert body.count("## Topic A") == 2
    assert "COVERED:" not in body


def test_a_failed_segment_does_not_advance_the_cursor(tmp_path: Path):
    worker, db, _ = make_env(tmp_path)
    document_id = add_document(db, 6)

    worker._ask = lambda settings, model, prompt, *, max_tokens: ""
    task = worker.start(
        conversation_id="chat-2",
        instruction="List every question",
        document_ids=[document_id],
        batch=3,
    )
    current = wait_for(worker, task["id"])
    assert current["state"] == "error"
    assert current["cursor"] == 0  # nothing was skipped

    # A later good run resumes from the same unit and appends.
    worker._ask = lambda settings, model, prompt, *, max_tokens: "COVERED: 4-6\n- item"
    assert worker.resume(task["id"], batch=3)
    current = wait_for(worker, task["id"])
    assert current["state"] == "done"
    assert current["cursor"] == 6
    assert len(worker.draft(task["id"])["chunks"]) == 2


def test_pause_stops_the_run_and_resume_appends(tmp_path: Path):
    worker, db, _ = make_env(tmp_path)
    document_id = add_document(db, 6)

    paused = {"id": ""}
    calls = {"n": 0}

    def pause_after_first(settings, model, prompt, *, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1 and paused["id"]:
            worker.pause(paused["id"])
        return "COVERED: 1-3\n- item"

    worker._ask = pause_after_first
    task = worker.start(
        conversation_id="chat-3",
        instruction="List every question",
        document_ids=[document_id],
        batch=3,
    )
    paused["id"] = task["id"]

    deadline = time.time() + 10
    state = None
    while time.time() < deadline:
        state = worker.get(task["id"])
        assert state is not None
        if state["state"] == "paused":
            break
        time.sleep(0.02)
    assert state["state"] == "paused", state

    # A paused run is stopped, not auto-resumed.
    time.sleep(0.1)
    assert worker.get(task["id"])["state"] == "paused"
    first = worker.draft(task["id"])
    assert first is not None
    assert len(first["chunks"]) == 1

    worker._ask = lambda settings, model, prompt, *, max_tokens: "COVERED: 4-6\n- item"
    assert worker.resume(task["id"], batch=3)
    current = wait_for(worker, task["id"])
    assert current["state"] == "done"
    second = worker.draft(task["id"])
    assert [c["ordinal"] for c in second["chunks"]] == [0, 1]
    assert [c["unit_from"] for c in second["chunks"]] == [1, 4]
    assert second["chunks"][0]["content"] == first["chunks"][0]["content"]
