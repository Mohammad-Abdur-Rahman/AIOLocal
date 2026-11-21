from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from pypdf import PdfWriter

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.db import Database
from aiolocal.documents import DocumentAnalyzer, DocumentError
from aiolocal.memory import store


def make_pdf(path: Path, pages: int = 5) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def make_analyzer(tmp_path: Path, **settings_kwargs) -> tuple[DocumentAnalyzer, Database, Paths]:
    paths = Paths.create(tmp_path / "data")
    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b", **settings_kwargs))
    db = Database(paths.db_path)
    return DocumentAnalyzer(paths, settings_store, db), db, paths


def wait_for(analyzer: DocumentAnalyzer, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = analyzer.get(job_id)
        if current and current["state"] in ("done", "error", "cancelled"):
            return current
        time.sleep(0.02)
    raise AssertionError("document job did not finish")


def test_pdf_is_analysed_one_page_per_call(tmp_path: Path):
    analyzer, db, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "lesson.pdf", pages=5)
    calls: list[tuple[int, int, int]] = []

    analyzer._render_pages = lambda path, first, last, dpi: [b"page"] * (last - first + 1)

    def vision(settings, model, title, first, last, images):
        calls.append((first, last, len(images)))
        return f"Page {first}: notes for page {first}."

    analyzer._vision_batch = vision

    job = analyzer.start(pdf, "lesson.pdf")
    current = wait_for(analyzer, job["id"])

    assert current["state"] == "done", current
    assert current["pages_total"] == 5
    assert current["pages_done"] == 5
    assert current["batches_total"] == 5
    assert current["batches_done"] == 5
    assert calls == [(1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 5, 1)]
    # The overview is a roll-up of completed notes, so every analysed page is
    # represented and no whole-document re-summarisation call is made.
    assert "Page 1" in current["final"]
    assert "Page 5" in current["final"]
    deadline = time.time() + 5
    while pdf.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert not pdf.exists()

    summary = store.get_summary(db, current["summary_id"])
    assert summary is not None
    assert summary["kind"] == "document"
    assert "Page 1" in summary["body"]
    assert "Page 5" in summary["body"]
    assert "## Page map" in summary["body"]
    assert "### Pages 1-3" not in summary["body"]
    with db.read() as c:
        pages = c.execute(
            "SELECT COUNT(*) FROM document_pages WHERE document_id=?",
            (current["summary_id"],),
        ).fetchone()[0]
        sections = c.execute(
            "SELECT COUNT(*) FROM document_sections WHERE document_id=?",
            (current["summary_id"],),
        ).fetchone()[0]
    assert pages == 5
    assert sections == 1


def test_page_rows_survive_and_are_retrievable(tmp_path: Path):
    analyzer, db, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "facts.pdf", pages=20)

    analyzer._render_pages = lambda path, first, last, dpi: [b"page"]
    analyzer._vision_batch = (
        lambda settings, model, title, first, last, images: f"FACT-{first}"
    )
    analyzer._consolidate = lambda job, settings, notes: "OVERVIEW"

    job = analyzer.start(pdf, "facts.pdf")
    current = wait_for(analyzer, job["id"])
    assert current["state"] == "done", current

    from aiolocal.memory.retrieve import retrieve

    result = retrieve(
        db, "what does page 17 say about FACT-17", settings=analyzer.settings_store.load()
    )
    bodies = "\n".join(item.body for item in result.items)
    assert "FACT-17" in bodies
    assert any(item.kind == "document_page" for item in result.items)


def test_resume_continues_after_the_last_stored_page(tmp_path: Path, monkeypatch):
    paths = Paths.create(tmp_path / "data")
    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b"))
    db = Database(paths.db_path)
    pdf = make_pdf(tmp_path / "resume.pdf", pages=4)
    summary_id = store.insert_summary(
        db, title="Document: resume", body="# Document: resume", topic_slug="documents",
        kind="document", source="document",
    )
    with db.write() as c:
        for page in (1, 2):
            c.execute(
                "INSERT INTO document_pages(document_id, page_no, text, tokens, created_at) VALUES(?,?,?,?,?)",
                (summary_id, page, f"OLD-{page}", 2, 1),
            )
        c.execute(
            """INSERT INTO document_jobs(id, filename, path, title, model, state, summary_id,
                   pages_total, pages_done, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            ("job-resume", "resume.pdf", str(pdf), "resume", "qwen3.5-9b",
             "interrupted", summary_id, 4, 2, 1, 1),
        )

    calls: list[int] = []

    def vision(self, settings, model, title, first, last, images):
        calls.append(first)
        return f"NEW-{first}"

    monkeypatch.setattr(DocumentAnalyzer, "_render_pages", lambda self, path, first, last, dpi: [b"page"])
    monkeypatch.setattr(DocumentAnalyzer, "_vision_batch", vision)

    analyzer = DocumentAnalyzer(paths, settings_store, db)
    current = wait_for(analyzer, "job-resume")
    assert current["state"] == "done", current
    assert calls == [3, 4]
    assert current["pages_done"] == 4
    with db.read() as c:
        rows = c.execute(
            "SELECT page_no, text FROM document_pages WHERE document_id=? ORDER BY page_no",
            (summary_id,),
        ).fetchall()
    assert [r["page_no"] for r in rows] == [1, 2, 3, 4]
    assert rows[1]["text"] == "OLD-2"


def test_cancel_stops_before_the_final_consolidation(tmp_path: Path):
    analyzer, db, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "cancel.pdf", pages=4)
    seen: list[str] = []
    gate = threading.Event()
    holder: list[str] = []

    analyzer._render_pages = lambda path, first, last, dpi: [b"page"] * (last - first + 1)

    def vision(settings, model, title, first, last, images):
        gate.wait(5)
        seen.append(f"{first}-{last}")
        if holder:
            analyzer.cancel(holder[0])
        return "partial"

    analyzer._vision_batch = vision

    job = analyzer.start(pdf, "cancel.pdf")
    holder.append(job["id"])
    gate.set()
    current = wait_for(analyzer, job["id"])

    assert current["state"] == "cancelled"
    assert current["final"] == ""
    summary = store.get_summary(db, current["summary_id"])
    assert summary is not None
    # The source is kept so the job can be resumed later.
    assert pdf.exists()


def test_non_vision_chat_model_falls_back_to_a_vision_model(tmp_path: Path):
    analyzer, _, _ = make_analyzer(tmp_path)
    analyzer.settings_store.save(Settings(chat_model="qwen3-4b"))
    pdf = make_pdf(tmp_path / "plain.pdf", pages=1)

    job = analyzer.start(pdf, "plain.pdf")

    assert job["model"] == "qwen3.5-9b"
    analyzer.close()


def test_start_requires_a_vision_model(tmp_path: Path, monkeypatch):
    analyzer, _, _ = make_analyzer(tmp_path)
    import aiolocal.documents as documents

    monkeypatch.setattr(documents, "CATALOG", [])
    pdf = make_pdf(tmp_path / "plain.pdf", pages=1)

    with pytest.raises(DocumentError, match="vision model"):
        analyzer.start(pdf, "plain.pdf")
    assert not pdf.exists()


def test_failure_before_any_page_leaves_no_stub_summary(tmp_path: Path):
    analyzer, db, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "broken.pdf", pages=2)

    def boom(path, first, last, dpi):
        raise DocumentError("render failed")

    analyzer._render_pages = boom
    before = len(store.list_summaries(db, limit=200))
    job = analyzer.start(pdf, "broken.pdf")
    current = wait_for(analyzer, job["id"])

    assert current["state"] == "error"
    assert current["summary_id"] == ""
    assert len(store.list_summaries(db, limit=200)) == before


def test_only_pdf_files_are_accepted(tmp_path: Path):
    analyzer, _, _ = make_analyzer(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")

    with pytest.raises(DocumentError, match="only PDF"):
        analyzer.start(text, "notes.txt")
    assert not text.exists()


def test_saved_batch_and_cap_values_are_ignored(tmp_path: Path, monkeypatch):
    analyzer, _, _ = make_analyzer(tmp_path)
    pdf = make_pdf(tmp_path / "stale.pdf", pages=5)
    calls: list[tuple[int, int]] = []

    # Simulate an old app.json that still asks for 3 pages per call and a 2-page
    # cap. Settings normally forces these fields, so tamper with the loaded object
    # directly to prove documents.py cannot be talked into batching or stopping.
    settings = analyzer.settings_store.load()
    settings.document_pages_per_batch = 3
    settings.document_max_pages = 2
    monkeypatch.setattr(analyzer.settings_store, "load", lambda: settings)

    analyzer._render_pages = lambda path, first, last, dpi: [b"page"] * (last - first + 1)

    def vision(user_settings, model, title, first, last, images):
        calls.append((first, last))
        return f"Page {first}"

    analyzer._vision_batch = vision
    analyzer._consolidate = lambda job, user_settings, notes: "DONE"

    job = analyzer.start(pdf, "stale.pdf")
    current = wait_for(analyzer, job["id"])

    assert current["state"] == "done", current
    assert current["pages_total"] == 5
    assert current["pages_done"] == 5
    assert calls == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]


def test_old_settings_json_is_migrated_to_one_page_and_no_cap(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.path.write_text(
        json.dumps(
            {
                "config_version": 1,
                "chat_model": "qwen3.5-9b",
                "document_pages_per_batch": 3,
                "document_max_pages": 300,
            }
        ),
        encoding="utf-8",
    )

    settings = store.load()

    assert settings.document_pages_per_batch == 1
    assert settings.document_max_pages == 0
    assert settings.config_version == 2
    rewritten = json.loads(store.path.read_text(encoding="utf-8"))
    assert rewritten["document_pages_per_batch"] == 1
    assert rewritten["document_max_pages"] == 0
    assert rewritten["config_version"] == 2


def test_document_endpoint_streams_upload_and_finishes(env):
    import io

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)

    response = env["client"].post(
        "/api/v1/documents/analyze?filename=lesson.pdf",
        content=buffer.getvalue(),
        headers={"X-AIO-Token": env["token"], "Content-Type": "application/pdf"},
    )
    assert response.status_code == 200, response.text
    job = response.json()["job"]

    deadline = time.time() + 15
    current = job
    while time.time() < deadline:
        current = env["client"].get(
            f"/api/v1/documents/jobs/{job['id']}", headers={"X-AIO-Token": env["token"]}
        ).json()
        if current["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)

    assert current["state"] == "done", current
    assert current["pages_total"] == 2
    assert current["summary_id"]
    summary = store.get_summary(env["app"].state.ctx.db, current["summary_id"])
    assert summary is not None
    assert summary["kind"] == "document"


def test_deleted_custom_chat_model_falls_back_to_a_vision_model(tmp_path: Path):
    import shutil

    from aiolocal.models import GRAPH_TEMPLATE, ModelManager

    analyzer, _, paths = make_analyzer(tmp_path)
    manager = ModelManager(paths, analyzer.settings_store)
    folder = manager.models_root() / "custom-vision"
    folder.mkdir(parents=True)
    (folder / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (folder / "openvino_model.bin").write_bytes(b"0" * 2_000_000)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "graph.pbtxt").write_text(GRAPH_TEMPLATE, encoding="utf-8")

    entry = manager.register_custom_folder("custom-vision")
    manager.assign_role(entry["id"], "chat", device="CPU")
    assert analyzer.settings_store.load().chat_model == entry["id"]

    # The user deletes the folder by hand instead of removing it in the UI.
    shutil.rmtree(folder)
    manager._installed_cache.clear()
    analyzer.models = manager

    assert analyzer._vision_model(analyzer.settings_store.load()) == "qwen3.5-9b"
