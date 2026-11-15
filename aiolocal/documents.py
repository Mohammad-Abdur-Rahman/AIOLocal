"""PDF page vision analysis into durable, page-level document memory.

A document is analysed one page per vision call and every page is stored as its own
row before the next page is read. That makes the analysis resumable, cancellable
and complete: the model-visible summary is a small index, while the pages and
section summaries stay searchable on disk.

The pipeline is hierarchical, in the same shape as the knowledge base:

* ``document_pages`` - the exact text produced for every page;
* ``document_sections`` - summaries of ~8 pages (and ~80 pages for long documents);
* one ``summaries`` row of kind ``document`` - the root overview and page map.

Nothing is thrown away because a context window is full. Only what is injected
into a prompt is budgeted, and that happens in ``memory/assemble.py``.

Rendering needs ``pypdfium2`` and Pillow. The uploaded PDF is kept until the job
finishes successfully so a crash or restart can resume; it is deleted afterwards
unless ``document_keep_pages`` is enabled.
"""

from __future__ import annotations

import base64
import io
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .config import Paths, Settings, SettingsStore
from .db import Database, now_ms
from .llm import api_url
from .memory import store
from .memory import semantic
from .memory.text import embed, estimate_tokens, new_id
from .models import CATALOG

VISION_SYSTEM = (
    "You are the document indexing engine of a local AI workspace. You read PDF "
    "page images carefully and write durable memory notes. Never invent content: "
    "when a page is unclear, say so. Preserve exact names, numbers, dates, units, "
    "definitions, equations and code."
)

BATCH_PROMPT = """Index page(s) {first}-{last} of "{title}" for long-term memory.

For EACH page, write a section exactly like "Page N" and cover:
- the subject and purpose of the page;
- key facts, names, dates, numbers, definitions and conclusions;
- diagrams, charts and tables: describe axes, labels, values and relationships;
- equations or code: transcribe them exactly;
- anything a reader would need to find this page again (terms, titles, headings).

Finish the whole answer with a "Memory-worthy points" bullet list of the five
most important takeaways across these pages. Be concise but complete. If a page
is blank or illegible, say so instead of guessing."""

FINAL_PROMPT = """Consolidate the notes below into ONE overview of "{title}" ({pages} pages).

Write 250-700 words in these sections:
1. Overview - what the document is and what it covers.
2. Key facts - the most important details, numbers, names and definitions.
3. Structure - the main parts and what each contains.
4. Practical details - equations, code, tables, instructions or warnings.
5. Retrieval keywords - 8-15 terms a person would use to find this document.

Do not invent anything that is not in the notes. If the notes are partial, say
which parts are missing. Return plain Markdown only.

NOTES:
{notes}"""

REDUCE_PROMPT = """Summarise these sections of "{title}" in 150-250 words.
Keep names, numbers, definitions and conclusions. Do not invent anything.

SECTIONS:
{notes}"""

SECTION_PAGES = 8
SECTION_GROUP = 10
MAX_SECTION_CHARS = 2200
MAX_BODY_CHARS = 20_000


@dataclass
class DocumentJob:
    id: str
    filename: str
    path: str
    title: str
    state: str = "queued"
    model: str = ""
    summary_id: str = ""
    pages_total: int = 0
    pages_done: int = 0
    batches_total: int = 0
    batches_done: int = 0
    final: str = ""
    error: str = ""
    created_at: int = field(default_factory=now_ms)
    updated_at: int = field(default_factory=now_ms)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "title": self.title,
            "state": self.state,
            "model": self.model,
            "summary_id": self.summary_id,
            "pages_total": self.pages_total,
            "pages_done": self.pages_done,
            "batches_total": self.batches_total,
            "batches_done": self.batches_done,
            "final": self.final,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DocumentError(Exception):
    """The document job could not be started or completed."""


class DocumentAnalyzer:
    """Runs PDF jobs on private worker threads."""

    def __init__(
        self,
        paths: Paths,
        settings_store: SettingsStore,
        db: Database,
        *,
        publish: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self.db = db
        self.publish = publish
        self.dir = paths.data_dir / "documents"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, DocumentJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._closing = False
        self._restore()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._lock:
            self._closing = True
            for job in self._jobs.values():
                job.cancel.set()
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=2.0)
        with self._lock:
            for job in self._jobs.values():
                if job.state in ("queued", "running"):
                    job.state = "interrupted"
                    job.updated_at = now_ms()
                    self._save_job(job)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.public() for j in jobs[: max(1, min(int(limit), 100))]]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.public() if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.cancel.set()
            return True

    def _restore(self) -> None:
        """Reload persisted jobs so a restart can resume instead of starting over."""
        try:
            with self.db.read() as c:
                rows = c.execute(
                    "SELECT * FROM document_jobs ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
        except Exception:
            return
        resumable: list[DocumentJob] = []
        for row in rows:
            job = DocumentJob(
                id=row["id"],
                filename=row["filename"],
                path=row["path"],
                title=row["title"],
                state=row["state"],
                model=row["model"] or "",
                summary_id=row["summary_id"] or "",
                pages_total=int(row["pages_total"] or 0),
                pages_done=int(row["pages_done"] or 0),
                error=row["error"] or "",
                created_at=int(row["created_at"]),
                updated_at=int(row["updated_at"]),
            )
            with self._lock:
                self._jobs[job.id] = job
            if job.state in ("queued", "running", "interrupted"):
                if Path(job.path).is_file():
                    resumable.append(job)
                else:
                    job.state = "interrupted"
                    job.error = "the uploaded file is no longer available; attach it again to continue"
                    self._save_job(job)
        for job in resumable:
            job.state = "queued"
            with self._lock:
                thread = threading.Thread(
                    target=self._run, args=(job,), name=f"doc-{job.id}", daemon=True
                )
                self._threads[job.id] = thread
            thread.start()

    # ------------------------------------------------------------------ start
    def start(self, upload: Path, filename: str) -> dict:
        """Take ownership of an uploaded PDF and return the queued job."""
        settings = self.settings_store.load()
        model = self._vision_model(settings)
        if not model:
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            raise DocumentError(
                "No vision model is available. Download and load Qwen3.5 9B or "
                "Qwen3-VL 4B in the Models view first."
            )
        suffix = Path(filename or upload.name).suffix.lower()
        if suffix != ".pdf":
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            raise DocumentError("only PDF files can be analysed page by page")
        try:
            import pypdfium2 as pdfium  # noqa: F401
        except ImportError as exc:
            try:
                upload.unlink(missing_ok=True)
            except OSError:
                pass
            raise DocumentError(
                "PDF rendering is not installed in this build (pypdfium2/Pillow missing)."
            ) from exc

        job_id = new_id()
        clean = Path(filename or upload.name).name[:180] or "document.pdf"
        job = DocumentJob(
            id=job_id,
            filename=clean,
            path=str(upload),
            title=clean.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or clean,
            model=model,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._save_job(job)
            thread = threading.Thread(target=self._run, args=(job,), name=f"doc-{job_id}", daemon=True)
            self._threads[job_id] = thread
        thread.start()
        self._emit(job)
        return job.public()

    def _save_job(self, job: DocumentJob) -> None:
        job.updated_at = now_ms()
        try:
            with self.db.write() as c:
                c.execute(
                    """INSERT INTO document_jobs(id, filename, path, title, model, state,
                           summary_id, pages_total, pages_done, error, created_at, updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                           state=excluded.state, summary_id=excluded.summary_id,
                           pages_total=excluded.pages_total, pages_done=excluded.pages_done,
                           error=excluded.error, updated_at=excluded.updated_at""",
                    (
                        job.id, job.filename, job.path, job.title, job.model, job.state,
                        job.summary_id, job.pages_total, job.pages_done,
                        (job.error or "")[:1000], job.created_at, job.updated_at,
                    ),
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ worker
    def _run(self, job: DocumentJob) -> None:
        settings = self.settings_store.load()
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(job.path)
            try:
                total = len(pdf)
            finally:
                pdf.close()
            if total <= 0:
                raise DocumentError("this PDF has no pages")
            cap = int(settings.document_max_pages or 0)
            job.pages_total = total if cap <= 0 else min(total, cap)
            if cap > 0 and total > cap:
                job.error = (
                    f"only the first {cap} of {total} pages were analysed; "
                    "set document_max_pages to 0 for the whole document"
                )
        except DocumentError as exc:
            self._fail(job, str(exc))
            return
        except Exception as exc:
            self._fail(job, f"could not open the PDF ({type(exc).__name__}: {exc})")
            return

        step = max(1, int(settings.document_pages_per_batch or 1))
        job.batches_total = (job.pages_total + step - 1) // step
        if job.state != "running":
            job.state = "running"
        self._save_job(job)
        self._emit(job)

        header = (
            f"# Document: {job.title}\n"
            f"Source: {job.filename} ({job.pages_total} pages). "
            "Analysed page by page with the local vision model.\n\n"
            "## Overview\n_Analysis in progress._\n"
        )
        try:
            if not job.summary_id:
                job.summary_id = store.insert_summary(
                    self.db,
                    title=f"Document: {job.title}"[:200],
                    body=header,
                    topic_slug="documents",
                    kind="document",
                    salience=0.72,
                    confidence=0.8,
                    source="document",
                    entities=["document", job.title],
                    embed_dim=settings.embed_dim,
                )
                self._save_job(job)
            else:
                store.update_summary_body(
                    self.db, job.summary_id, header, title=f"Document: {job.title}",
                    embed_dim=settings.embed_dim,
                )
        except Exception as exc:
            self._fail(job, f"could not create the memory entry ({type(exc).__name__}: {exc})")
            return
        try:
            semantic.index_summary(
                self.db, settings, job.summary_id, f"Document: {job.title}", header
            )
        except Exception:
            pass

        completed = self._pages_done(job.summary_id)
        job.pages_done = completed
        self._save_job(job)
        self._emit(job)

        try:
            for start in range(completed + 1, job.pages_total + 1, step):
                if job.cancel.is_set():
                    break
                end = min(job.pages_total, start + step - 1)
                images = self._render_pages(job.path, start, end, settings.document_dpi)
                if not images:
                    raise DocumentError(f"could not render pages {start}-{end}")
                text = self._vision_batch(
                    settings, job.model, job.title, start, end, images
                )
                self._store_pages(job, start, end, text, settings.embed_dim)
                job.pages_done = end
                job.batches_done = (end + step - 1) // step
                job.state = "running"
                self._save_job(job)
                self._emit(job)

            if job.cancel.is_set():
                job.state = "interrupted" if self._closing else "cancelled"
                self._save_job(job)
                self._emit(job)
                return

            self._build_sections(job, settings)
            final = self._compose_overview(job, settings)
            job.final = final
            body = self._compose_body(job, final)
            store.update_summary_body(
                self.db, job.summary_id, body,
                title=f"Document: {job.title}", embed_dim=settings.embed_dim,
            )
            try:
                semantic.index_summary(
                    self.db, settings, job.summary_id, f"Document: {job.title}", body
                )
            except Exception:
                pass
            job.state = "done"
            job.error = ""
            self._save_job(job)
            self._emit(job)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"[:1000]
            pages = self._pages_done(job.summary_id)
            try:
                if job.summary_id and pages:
                    self._build_sections(job, settings)
                    final = self._compose_overview(job, settings)
                    body = self._compose_body(job, final)
                    store.update_summary_body(
                        self.db, job.summary_id, body,
                        title=f"Document: {job.title}", embed_dim=settings.embed_dim,
                    )
                    semantic.index_summary(
                        self.db, settings, job.summary_id, f"Document: {job.title}", body
                    )
                elif job.summary_id:
                    store.delete_summary(self.db, job.summary_id)
                    job.summary_id = ""
            except Exception:
                pass
            job.state = "error"
            self._save_job(job)
            self._emit(job)
        finally:
            with self._lock:
                self._threads.pop(job.id, None)
            if job.state == "done" and not settings.document_keep_pages:
                try:
                    Path(job.path).unlink(missing_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------ pages
    def _pages_done(self, document_id: str) -> int:
        if not document_id:
            return 0
        with self.db.read() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(page_no),0) FROM document_pages WHERE document_id=?",
                (document_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def _store_pages(
        self, job: DocumentJob, first: int, last: int, text: str, embed_dim: int
    ) -> None:
        pieces = _split_page_text(text, first, last)
        with self.db.write() as c:
            for page_no in range(first, last + 1):
                page_text = pieces.get(page_no, "").strip()
                if not page_text:
                    continue
                c.execute(
                    """INSERT INTO document_pages(document_id, page_no, text, tokens, created_at)
                       VALUES(?,?,?,?,?)
                       ON CONFLICT(document_id, page_no) DO UPDATE SET
                           text=excluded.text, tokens=excluded.tokens, created_at=excluded.created_at""",
                    (job.summary_id, page_no, page_text, estimate_tokens(page_text), now_ms()),
                )
        for page_no in range(first, last + 1):
            page_text = pieces.get(page_no, "").strip()
            if not page_text:
                continue
            try:
                store.put_embedding(
                    self.db,
                    "document_page",
                    f"{job.summary_id}:{page_no}",
                    "hash",
                    embed(page_text[:2000], embed_dim),
                )
            except Exception:
                pass

    def _build_sections(self, job: DocumentJob, settings: Settings) -> None:
        if not job.summary_id or not job.pages_total:
            return
        with self.db.read() as c:
            rows = c.execute(
                "SELECT page_no, text FROM document_pages WHERE document_id=? ORDER BY page_no",
                (job.summary_id,),
            ).fetchall()
        if not rows:
            return
        by_page = {int(r["page_no"]): r["text"] for r in rows}
        sections: list[tuple[int, int, int, str]] = []
        ordinal = 0
        for start in range(1, job.pages_total + 1, SECTION_PAGES):
            end = min(job.pages_total, start + SECTION_PAGES - 1)
            collected = [by_page[n] for n in range(start, end + 1) if by_page.get(n)]
            if not collected:
                continue
            summary = _section_summary(start, end, collected)
            sections.append((1, ordinal, start, end, summary))
            ordinal += 1
        # A second level keeps very long documents navigable without a giant map.
        if len(sections) > SECTION_GROUP:
            group_ordinal = 0
            for index in range(0, len(sections), SECTION_GROUP):
                group = sections[index : index + SECTION_GROUP]
                start = group[0][2]
                end = group[-1][3]
                text = _section_summary(
                    start, end, [g[4] for g in group], limit=1800
                )
                sections.append((2, group_ordinal, start, end, text))
                group_ordinal += 1

        with self.db.write() as c:
            c.execute("DELETE FROM document_sections WHERE document_id=?", (job.summary_id,))
            for level, ordinal, start, end, summary in sections:
                c.execute(
                    """INSERT INTO document_sections(document_id, ordinal, level, page_from,
                           page_to, summary, tokens, created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        job.summary_id, ordinal, level, start, end, summary,
                        estimate_tokens(summary), now_ms(),
                    ),
                )
        for level, ordinal, start, end, summary in sections:
            try:
                store.put_embedding(
                    self.db,
                    "document_section",
                    f"{job.summary_id}:{level}:{ordinal}",
                    "hash",
                    embed(summary, settings.embed_dim),
                )
            except Exception:
                pass
        job.batches_done = job.batches_total
        self._save_job(job)

    # ------------------------------------------------------------------ summaries
    def _compose_overview(self, job: DocumentJob, settings: Settings) -> str:
        with self.db.read() as c:
            rows = c.execute(
                """SELECT level, ordinal, page_from, page_to, summary
                   FROM document_sections WHERE document_id=?
                   ORDER BY level DESC, ordinal""",
                (job.summary_id,),
            ).fetchall()
        if not rows:
            return "_No page text was produced._"
        top_level = int(rows[0]["level"])
        notes = [
            (int(r["page_from"]), int(r["page_to"]), r["summary"])
            for r in rows
            if int(r["level"]) == top_level
        ]
        joined = "\n\n".join(f"### Pages {a}-{b}\n{t}" for a, b, t in notes)
        if len(joined) > 14_000:
            grouped: list[tuple[int, int, str]] = []
            buffer: list[tuple[int, int, str]] = []
            size = 0
            for note in notes:
                cost = len(note[2]) + 40
                if buffer and size + cost > 10_000:
                    grouped.append((buffer[0][0], buffer[-1][1], self._reduce_notes(job, settings, buffer)))
                    buffer, size = [], 0
                buffer.append(note)
                size += cost
            if buffer:
                grouped.append((buffer[0][0], buffer[-1][1], self._reduce_notes(job, settings, buffer)))
            notes = grouped
            joined = "\n\n".join(f"### Pages {a}-{b}\n{t}" for a, b, t in notes)
        return self._consolidate(job, settings, notes) or (
            "_The consolidating model was unavailable; section summaries are stored separately._"
        )

    def _reduce_notes(self, job: DocumentJob, settings: Settings, notes: list[tuple[int, int, str]]) -> str:
        text = "\n\n".join(t for _, _, t in notes)
        prompt = REDUCE_PROMPT.format(title=job.title, notes=text[:12_000])
        out = self._ask_text(settings, job.model, prompt, max_tokens=600)
        if out:
            return out
        return text[:1600]

    def _consolidate(
        self, job: DocumentJob, settings: Settings, notes: list[tuple[int, int, str]]
    ) -> str:
        notes_text = "\n\n".join(f"### Pages {a}-{b}\n{text}" for a, b, text in notes)
        prompt = FINAL_PROMPT.format(title=job.title, pages=job.pages_total, notes=notes_text[:14_000])
        out = self._ask_text(settings, job.model, prompt, max_tokens=1400)
        if out:
            return out
        overview = notes[0][2][:1200] if notes else ""
        return (
            f"## Overview\n{overview}\n\n"
            f"## Key facts\nSee the section summaries. The consolidating model was unavailable, "
            f"so this document's memory contains {len(notes)} indexed sections covering "
            f"{job.pages_total} pages."
        )

    def _ask_text(self, settings: Settings, model: str, prompt: str, *, max_tokens: int) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = api_url(settings.upstream_base_url, "chat/completions")
        try:
            with self._client(settings) as client:
                response = client.post(url, json=payload)
            if response.status_code >= 400:
                return ""
            content = response.json()["choices"][0]["message"].get("content") or ""
            return str(content).strip()[:16_000]
        except Exception:
            return ""

    def _compose_body(self, job: DocumentJob, final: str) -> str:
        with self.db.read() as c:
            rows = c.execute(
                """SELECT level, ordinal, page_from, page_to, summary
                   FROM document_sections WHERE document_id=?
                   ORDER BY level ASC, ordinal ASC""",
                (job.summary_id,),
            ).fetchall()
        top = [r for r in rows if int(r["level"]) == 1]
        parts = [
            f"# Document: {job.title}",
            f"Source: {job.filename} ({job.pages_total} pages). "
            "Analysed page by page with the local vision model. "
            f"{len(top)} section summaries and every page are stored and searchable.",
            "## Overview",
            final or "_No overview was produced._",
            "",
            "## Page map",
        ]
        shown = 0
        for row in top:
            if shown >= 40:
                break
            summary = " ".join(str(row["summary"]).split())[:180]
            parts.append(f"- Pages {int(row['page_from'])}-{int(row['page_to'])}: {summary}")
            shown += 1
        if len(top) > shown:
            parts.append(f"- … {len(top) - shown} more sections are searchable by page.")
        body = "\n".join(parts)
        if len(body) > MAX_BODY_CHARS:
            body = body[: MAX_BODY_CHARS - 40] + "\n… page map truncated"
        return body

    def _fail(self, job: DocumentJob, message: str) -> None:
        job.state = "error"
        job.error = message[:1000]
        self._save_job(job)
        self._emit(job)

    def _emit(self, job: DocumentJob) -> None:
        if not self.publish:
            return
        try:
            self.publish("document_job", job.public())
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers
    def _vision_model(self, settings: Settings) -> str:
        preferred = str(settings.chat_model or "").strip()
        for entry in CATALOG:
            if entry["id"] == preferred and entry.get("vision") and entry.get("task") == "text_generation":
                return preferred
        for entry in CATALOG:
            if entry.get("vision") and entry.get("task") == "text_generation":
                return entry["id"]
        return ""

    def _client(self, settings: Settings) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(600.0, connect=5.0, write=60.0),
            trust_env=False,
        )

    def _render_pages(self, path: str, first: int, last: int, dpi: int) -> list[bytes]:
        import pypdfium2 as pdfium

        images: list[bytes] = []
        pdf = pdfium.PdfDocument(path)
        try:
            for index in range(first - 1, last):
                page = pdf[index]
                bitmap = None
                pil = None
                try:
                    bitmap = page.render(scale=max(1.0, dpi / 72.0))
                    pil = bitmap.to_pil().convert("RGB")
                    buffer = io.BytesIO()
                    pil.save(buffer, format="JPEG", quality=84, optimize=True)
                    images.append(buffer.getvalue())
                finally:
                    if pil is not None:
                        pil.close()
                    if bitmap is not None:
                        bitmap.close()
                    page.close()
        finally:
            pdf.close()
        return images

    def _vision_batch(
        self,
        settings: Settings,
        model: str,
        title: str,
        first: int,
        last: int,
        images: list[bytes],
    ) -> str:
        parts: list[dict] = [
            {"type": "text", "text": BATCH_PROMPT.format(title=title, first=first, last=last)}
        ]
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": parts},
            ],
            "temperature": 0.2,
            "max_tokens": 1600,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = api_url(settings.upstream_base_url, "chat/completions")
        with self._client(settings) as client:
            response = client.post(url, json=payload)
        if response.status_code >= 400:
            raise DocumentError(
                f"the model server rejected pages {first}-{last} "
                f"(HTTP {response.status_code}): {response.text[:400]}"
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            raise DocumentError(f"the model returned an unreadable answer ({type(exc).__name__})") from exc
        text = str(content).strip()
        if not text:
            raise DocumentError(f"the model returned an empty analysis for pages {first}-{last}")
        return text


def _split_page_text(text: str, first: int, last: int) -> dict[int, str]:
    """Split a multi-page vision answer into per-page notes when possible."""
    if first == last:
        return {first: text}
    matches = list(re.finditer(r"(?im)^\s*(?:#{1,6}\s*)?page\s+(\d+)\b.*$", text))
    if not matches:
        return {first: text}
    pieces: dict[int, str] = {}
    for index, match in enumerate(matches):
        page_no = int(match.group(1))
        if page_no < first or page_no > last:
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            pieces[page_no] = body
    return pieces or {first: text}


def _section_summary(
    start: int, end: int, texts: list[str], *, limit: int = MAX_SECTION_CHARS
) -> str:
    body = " ".join(" ".join(t.split())[:420] for t in texts)
    body = body[:limit].strip()
    return f"Pages {start}-{end}: {body}"
