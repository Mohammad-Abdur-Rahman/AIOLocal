"""Iterative response drafting for tasks larger than one context window.

A long job (for example "categorise every question in this 584-page document
by these topics") cannot be answered in one call without truncating the source
or the answer. This worker drafts in bounded segments:

1. Work on a slice of units and stop at a hard limit.
2. Append the produced text to an append-only draft. Nothing already written is
   ever rewritten.
3. Keep only the *unfinished* segment in the next prompt. The system context
   states which units are complete so the model can decide to skip repeated
   items instead of re-emitting them.
4. When every unit is covered, assemble the accumulated chunks into one
   presentable Markdown file and record a short index summary.

The draft is the deliverable. The summary is only a pointer to it.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .config import Paths, Settings, SettingsStore
from .db import Database, now_ms
from .llm import api_url
from .memory import store
from .memory.text import estimate_tokens, new_id, normalise, truncate_tokens

TASK_SYSTEM = (
    "You are completing a long task in parts. The user's instruction and topics "
    "apply to the whole job. Each call gives you ONLY the next segment of source "
    "units and a note describing what earlier calls already wrote. "
    "Never repeat, restate or re-categorise an item that earlier calls completed; "
    "use the completed note to decide what to skip. Never invent items. "
    "Quote names, numbers and page references exactly."
)

BATCH_PROMPT = """TASK
{instruction}

TOPICS
{topics}

COMPLETED SO FAR
{completed}

NEXT SEGMENT (units {first}-{last} of {total})
{segment}

Write only the result for units {first}-{last} above. Begin with a line
"COVERED: {first}-{last}". Then the result, following the user's requested format.
If part of this segment was already completed above, skip it silently.
Do not write a conclusion or summary of the whole job here."""

INDEX_PROMPT = """Task: {instruction}

Below are the headings and opening lines of the {count} draft sections that
together answer this task. Write a 120-250 word introduction and a short table
of contents. Do not repeat the body and do not invent sections.

SECTIONS
{sections}"""

BULK_WORDS = {
    "every", "each", "all", "entire", "whole", "complete", "full",
    "categorise", "categorize", "classify", "categorised", "categorized",
    "list", "enumerate", "extract", "map", "audit", "go through", "work through",
    "page by page", "question by question", "one by one",
}
TASK_VERBS = {
    "categorise", "categorize", "classify", "list", "extract", "enumerate",
    "summarise", "summarize", "answer", "audit", "map", "rewrite", "translate",
    "grade", "label", "group", "sort", "analyse", "analyze", "review",
}
DEFAULT_BATCH = 6
MAX_BATCH = 20


def looks_like_bulk_task(text: str, *, total_pages: int) -> bool:
    """Conservative detector: only very large, explicitly exhaustive jobs.

    A normal question about a document must stay a normal chat turn, so this
    requires both an exhaustive phrase and a subject at least ~20 pages long.
    """
    if total_pages < 20:
        return False
    words = set(re.findall(r"[a-z]+", normalise(text).lower()))
    if not (words & TASK_VERBS):
        return False
    lowered = normalise(text).lower()
    return any(marker in lowered for marker in BULK_WORDS) or "all of" in lowered


# --------------------------------------------------------------------- models
@dataclass
class ResponseTask:
    id: str
    conversation_id: str
    instruction: str
    title: str = "Long task"
    topics: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    model: str = ""
    cursor: int = 0
    total_units: int = 0
    state: str = "queued"
    error: str = ""
    summary_id: str = ""
    created_at: int = 0
    updated_at: int = 0
    pause: threading.Event = field(default_factory=threading.Event, repr=False)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "instruction": self.instruction,
            "title": self.title,
            "topics": self.topics,
            "document_ids": self.document_ids,
            "model": self.model,
            "cursor": self.cursor,
            "total_units": self.total_units,
            "state": self.state,
            "error": self.error,
            "summary_id": self.summary_id,
            "percent": round(100.0 * self.cursor / self.total_units, 1)
            if self.total_units
            else 0.0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ResponseTaskWorker:
    """Runs drafting jobs on private daemon threads."""

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
        self.dir = paths.artifact_dir / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, ResponseTask] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._closing = False
        self._persisted: dict[str, int] = {}
        self._restore()

    # ------------------------------------------------------------------ public
    def start(
        self,
        *,
        conversation_id: str,
        instruction: str,
        topics: list[str] | None = None,
        document_ids: list[str] | None = None,
        model: str = "",
        title: str = "",
        batch: int = DEFAULT_BATCH,
    ) -> dict:
        units = self._collect_units(document_ids or [])
        if not units:
            raise ValueError("this task needs at least one analysed document")
        task = ResponseTask(
            id=new_id(),
            conversation_id=conversation_id,
            instruction=instruction.strip()[:8000],
            title=(title or instruction.strip().splitlines()[0])[:120] or "Long task",
            topics=[str(t)[:80] for t in (topics or [])][:40],
            document_ids=[str(d) for d in (document_ids or [])][:12],
            units=units,
            model=model,
            total_units=len(units),
            state="queued",
            created_at=now_ms(),
        )
        with self._lock:
            self._tasks[task.id] = task
        self._save(task)
        thread = threading.Thread(
            target=self._run, args=(task, max(1, min(int(batch), MAX_BATCH))),
            name=f"task-{task.id}", daemon=True,
        )
        with self._lock:
            self._threads[task.id] = thread
        thread.start()
        return task.public()

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
        return task.public() if task else None

    def list(self, *, conversation_id: str = "", limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM response_tasks"
        args: list[object] = []
        if conversation_id:
            sql += " WHERE conversation_id=?"
            args.append(conversation_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 200)))
        with self.db.read() as c:
            rows = c.execute(sql, tuple(args)).fetchall()
        return [self._row_public(r) for r in rows]

    def pause(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.state not in ("queued", "running"):
                return False
            task.pause.set()
            if task.state == "queued":
                task.state = "paused"
                self._save(task)
            self._emit(task)
            return True

    def _finish_pause(self, task_id: str) -> None:
        """Release a paused task once its worker thread has stopped.

        Called from the worker's ``finally`` block. The continuation thread
        restarts from the persisted cursor, so a pause is exactly a boundary
        between two appended segments.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.state != "paused" or self._closing:
                return
            self._threads.pop(task.id, None)
        self.resume(task_id)

    def resume(self, task_id: str, *, batch: int = DEFAULT_BATCH) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.state not in ("paused", "interrupted", "error"):
                return False
            task.pause.clear()
            task.cancel.clear()
            task.state = "running"
            if task.error:
                task.error = ""
            self._save(task)
            self._emit(task)
            thread = self._threads.get(task.id)
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._run, args=(task, max(1, min(int(batch), MAX_BATCH))),
                    name=f"task-{task.id}", daemon=True,
                )
                self._threads[task.id] = thread
                thread.start()
            return True

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.cancel.set()
            task.pause.clear()
            return True

    def draft(self, task_id: str) -> dict | None:
        """Return the assembled draft without waiting for the job to finish."""
        with self.db.read() as c:
            task_row = c.execute("SELECT * FROM response_tasks WHERE id=?", (task_id,)).fetchone()
            if not task_row:
                return None
            rows = c.execute(
                "SELECT * FROM response_task_chunks WHERE task_id=? ORDER BY ordinal",
                (task_id,),
            ).fetchall()
        return {
            "task": self._row_public(task_row),
            "chunks": [
                {
                    "ordinal": int(r["ordinal"]),
                    "unit_from": int(r["unit_from"]),
                    "unit_to": int(r["unit_to"]),
                    "content": str(r["content"]),
                    "tokens": int(r["tokens"] or 0),
                }
                for r in rows
            ],
            "text": _join_chunks([str(r["content"]) for r in rows]),
        }

    def close(self) -> None:
        with self._lock:
            self._closing = True
            for task in self._tasks.values():
                task.cancel.set()
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=2.0)
        with self._lock:
            for task in self._tasks.values():
                if task.state in ("queued", "running"):
                    task.state = "interrupted"
                    task.updated_at = now_ms()
                    self._save(task)

    # ----------------------------------------------------------------- internal
    def _collect_units(self, document_ids: list[str]) -> list[str]:
        if not document_ids:
            return []
        marks = ",".join("?" * len(document_ids))
        with self.db.read() as c:
            rows = c.execute(
                f"""SELECT document_id, page_no FROM document_pages
                    WHERE document_id IN ({marks}) ORDER BY document_id, page_no""",
                tuple(document_ids),
            ).fetchall()
        return [f"{r['document_id']}:{int(r['page_no'])}" for r in rows]

    def _persisted_cursor(self, task: ResponseTask) -> int:
        """Highest cursor whose segment is already durably appended.

        ``task.cursor`` moves inside the loop before the append is visible to
        other connections, so a freshly started continuation thread must trust
        the committed chunks instead of the in-memory value.
        """
        with self.db.read() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(unit_to), 0) FROM response_task_chunks WHERE task_id=?",
                (task.id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def _run(self, task: ResponseTask, batch: int) -> None:
        try:
            settings = self.settings_store.load()
            model = task.model or settings.filing_model or settings.chat_model
            task.cursor = self._persisted_cursor(task)
            task.state = "running"
            self._save(task)
            self._emit(task)
            while task.cursor < task.total_units:
                if task.cancel.is_set():
                    break
                if task.pause.is_set():
                    task.state = "paused"
                    self._save(task)
                    self._emit(task)
                    while task.pause.is_set() and not task.cancel.is_set():
                        task.pause.wait(timeout=0.25)
                    if task.cancel.is_set():
                        break
                    task.state = "running"
                    self._save(task)
                    self._emit(task)

                start = task.cursor
                end = min(task.total_units, start + batch)
                segment, first_unit, last_unit = self._segment(task.units, start, end)
                if not segment:
                    task.cursor = end
                    self._save(task)
                    continue
                completed = self._completed_note(task)
                prompt = BATCH_PROMPT.format(
                    instruction=task.instruction,
                    topics="\n".join(f"- {t}" for t in task.topics) or "(none supplied)",
                    completed=completed or "(nothing yet)",
                    first=first_unit,
                    last=last_unit,
                    total=task.total_units,
                    segment=segment[:12_000],
                )
                content = self._ask(settings, model, prompt, max_tokens=1600)
                if not content.strip():
                    # A model failure must not advance the cursor: resuming will
                    # retry this exact segment instead of skipping it. Clear any
                    # stale error from a previous attempt first.
                    task.error = ""
                    break
                self._append_chunk(task, start, end, first_unit, last_unit, content)
                task.cursor = end
                task.error = ""
                self._save(task)
                self._emit(task)

            if task.cancel.is_set():
                task.state = "interrupted" if self._closing else "cancelled"
                self._save(task)
                self._emit(task)
                return

            if task.cursor < task.total_units:
                # Unfinished work is a *failure* here, not a pause: the user can
                # retry it, and the persisted draft is kept either way. Reporting
                # "paused" would make a broken segment look like a deliberate stop.
                task.state = "error"
                if not task.error:
                    task.error = (
                        "Stopped after segment "
                        f"{task.cursor}/{task.total_units}: the model returned no text. "
                        "The draft is saved; retry to continue from here."
                    )
                self._save(task)
                self._emit(task)
                return

            self._present_and_finish(task, settings, model)
        except Exception as exc:
            task.state = "error"
            task.error = f"{type(exc).__name__}: {exc}"[:1000]
            self._save(task)
            self._emit(task)
        finally:
            # Never leave a task reporting "running" after its worker thread has
            # gone: a reply that cannot finish is an error the user can retry.
            if task.state == "running" and not task.pause.is_set():
                task.state = "error"
                if not task.error:
                    task.error = "The task worker stopped unexpectedly."
                self._save(task)
                self._emit(task)
            with self._lock:
                current = self._threads.get(task.id)
                if current is threading.current_thread():
                    self._threads.pop(task.id, None)
            self._finish_pause(task.id)

    def _segment(self, units: list[str], start: int, end: int) -> tuple[str, int, int]:
        pairs = [u.split(":", 1) for u in units[start:end]]
        if not pairs:
            return "", start + 1, end
        by_doc: dict[str, list[int]] = {}
        for doc_id, page in pairs:
            if page.isdigit():
                by_doc.setdefault(doc_id, []).append(int(page))
        blocks: list[str] = []
        for doc_id, pages in by_doc.items():
            marks = ",".join("?" * len(pages))
            with self.db.read() as c:
                rows = c.execute(
                    f"""SELECT page_no, text FROM document_pages
                        WHERE document_id=? AND page_no IN ({marks}) ORDER BY page_no""",
                    (doc_id, *pages),
                ).fetchall()
            blocks.extend(f"### Page {int(r['page_no'])}\n{r['text']}" for r in rows)
        return "\n\n".join(blocks), start + 1, end

    def _completed_note(self, task: ResponseTask) -> str:
        """A compact statement of finished work, for the model to skip.

        Only the tail of the draft is quoted, never the whole text: the point is
        to let the model decide what is already covered while keeping the prompt
        bounded.
        """
        with self.db.read() as c:
            row = c.execute(
                """SELECT unit_to, content FROM response_task_chunks
                   WHERE task_id=? ORDER BY ordinal DESC LIMIT 1""",
                (task.id,),
            ).fetchone()
            count = c.execute(
                "SELECT COUNT(*) FROM response_task_chunks WHERE task_id=?", (task.id,)
            ).fetchone()
        if not row:
            return ""
        tail = " ".join(str(row["content"]).split())[-600:]
        return (
            f"{int(count[0]) if count else 0} earlier sections are already written. "
            f"They cover units 1-{int(row['unit_to'])}. "
            f"Last section ends with: ...{tail}"
        )

    def _append_chunk(
        self,
        task: ResponseTask,
        start: int,
        end: int,
        first_unit: int,
        last_unit: int,
        content: str,
    ) -> None:
        body = _strip_covered(content).strip()
        if not body:
            body = content.strip()
        with self.db.write() as c:
            # Ordinal = number of sections already written, never the cursor:
            # two segments of different sizes must still produce 0,1,2,...
            row = c.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM response_task_chunks WHERE task_id=?",
                (task.id,),
            ).fetchone()
            c.execute(
                """INSERT INTO response_task_chunks(task_id, ordinal, unit_from, unit_to,
                       content, tokens, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    task.id,
                    int(row[0] if row else 0),
                    first_unit,
                    last_unit,
                    body,
                    estimate_tokens(body),
                    now_ms(),
                ),
            )

    def _present_and_finish(self, task: ResponseTask, settings: Settings, model: str) -> None:
        summary_id = self._present(task, settings, model)
        task.summary_id = summary_id or ""
        task.state = "done"
        task.error = ""
        task.updated_at = now_ms()
        self._save(task)
        self._emit(task)

    def _present(self, task: ResponseTask, settings: Settings, model: str) -> str:
        with self.db.read() as c:
            rows = c.execute(
                "SELECT content FROM response_task_chunks WHERE task_id=? ORDER BY ordinal",
                (task.id,),
            ).fetchall()
        chunks = [str(r["content"]) for r in rows]
        if not chunks:
            return ""
        sections = "\n".join(_section_head(chunk) for chunk in chunks)
        intro = self._ask(
            settings,
            model,
            INDEX_PROMPT.format(
                instruction=task.instruction,
                count=len(chunks),
                sections=sections[:12_000],
            ),
            max_tokens=700,
        )
        header = [
            f"# {task.title}",
            "",
            f"**Task.** {task.instruction}",
            "",
        ]
        if task.topics:
            header.append("**Topics.** " + ", ".join(task.topics))
            header.append("")
        header.append(f"**Source.** {task.total_units} units across {len(task.document_ids)} document(s).")
        header.append("")
        if intro:
            header.extend(["## Introduction", intro, ""])
        body = _join_chunks(chunks)
        text = "\n".join(header) + "\n## Draft\n\n" + body + "\n"
        path = self.dir / f"task-{task.id}.md"
        try:
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass

        title = f"Task draft: {task.title}"[:200]
        summary_body = (
            f"# {task.title}\n\n"
            f"**Task.** {task.instruction}\n\n"
            f"**Draft.** {len(chunks)} sections, {task.total_units} units, "
            f"stored at `{path.name}`.\n\n"
            "## Introduction\n"
            f"{intro or '_The introducing model was unavailable; the draft is complete._'}\n\n"
            "## Section index\n"
            + "\n".join(f"- {_section_head(chunk, 160)}" for chunk in chunks[:80])
        )
        try:
            summary_id = store.insert_summary(
                self.db,
                title=title,
                body=summary_body,
                topic_slug="tasks",
                kind="task",
                salience=0.7,
                confidence=0.8,
                source="task",
                entities=["task"],
                embed_dim=settings.embed_dim,
            )
            return summary_id
        except Exception:
            return ""

    def _ask(self, settings: Settings, model: str, prompt: str, *, max_tokens: int) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": TASK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0), trust_env=False) as client:
                response = client.post(
                    api_url(settings.upstream_base_url, "chat/completions"), json=payload
                )
            if response.status_code >= 400:
                return ""
            return str(response.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------ storage
    def _save(self, task: ResponseTask) -> None:
        task.updated_at = now_ms()
        self._persisted[task.id] = task.cursor
        try:
            with self.db.write() as c:
                c.execute(
                    """INSERT INTO response_tasks(id, conversation_id, request_id, title,
                           instruction, topics, document_ids, units, model, cursor,
                           total_units, state, error, summary_id, created_at, updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                           cursor=excluded.cursor, total_units=excluded.total_units,
                           state=excluded.state, error=excluded.error,
                           summary_id=excluded.summary_id, updated_at=excluded.updated_at""",
                    (
                        task.id, task.conversation_id, "", task.title[:200],
                        task.instruction, json.dumps(task.topics, ensure_ascii=False),
                        json.dumps(task.document_ids, ensure_ascii=False),
                        json.dumps(task.units, ensure_ascii=False),
                        task.model, task.cursor, task.total_units, task.state,
                        (task.error or "")[:1000], task.summary_id,
                        task.created_at, task.updated_at,
                    ),
                )
        except Exception:
            pass

    def _restore(self) -> None:
        try:
            with self.db.read() as c:
                rows = c.execute(
                    "SELECT * FROM response_tasks ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
        except Exception:
            return
        for row in rows:
            task = self._from_row(row)
            self._persisted[task.id] = task.cursor
            with self._lock:
                self._tasks[task.id] = task
            if task.state in ("queued", "running", "interrupted"):
                # An interrupted job resumes where the cursor stopped; completed
                # chunks are never rewritten.
                thread = threading.Thread(
                    target=self._run, args=(task, DEFAULT_BATCH),
                    name=f"task-{task.id}", daemon=True,
                )
                with self._lock:
                    self._threads[task.id] = thread
                thread.start()

    def _from_row(self, row) -> ResponseTask:
        def _json(value) -> list:
            try:
                parsed = json.loads(value or "[]")
                return parsed if isinstance(parsed, list) else []
            except (TypeError, ValueError):
                return []

        return ResponseTask(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            instruction=str(row["instruction"] or ""),
            title=str(row["title"] or "Long task"),
            topics=[str(t) for t in _json(row["topics"])],
            document_ids=[str(d) for d in _json(row["document_ids"])],
            units=[str(u) for u in _json(row["units"])],
            model=str(row["model"] or ""),
            cursor=int(row["cursor"] or 0),
            total_units=int(row["total_units"] or 0),
            state=str(row["state"] or "queued"),
            error=str(row["error"] or ""),
            summary_id=str(row["summary_id"] or ""),
            created_at=int(row["created_at"] or 0),
            updated_at=int(row["updated_at"] or 0),
        )

    def _row_public(self, row) -> dict:
        return self._from_row(row).public()

    def _emit(self, task: ResponseTask) -> None:
        if not self.publish:
            return
        try:
            self.publish("response_task", task.public())
        except Exception:
            pass


# ------------------------------------------------------------------ formatting
_COVERED_RE = re.compile(
    r"(?i)^\s*COVERED\s*:\s*[0-9]+(?:\s*[-\u2013]\s*[0-9]+)?\s*$", re.MULTILINE
)


def _strip_covered(text: str) -> str:
    return _COVERED_RE.sub("", text).lstrip()


def _section_head(text: str, limit: int = 120) -> str:
    cleaned = _strip_covered(text)
    for line in cleaned.splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return "(empty section)"


def _join_chunks(chunks: list[str]) -> str:
    """Concatenate chunks without rewriting them.

    Only the per-section ``COVERED:`` marker is removed and a repeated heading
    inside a single chunk is dropped. Identical headings in *different* chunks
    are kept: those are separate units that happen to share a topic, and
    merging them would lose content.
    """
    out: list[str] = []
    for chunk in chunks:
        body = _strip_covered(chunk).strip()
        if not body:
            continue
        lines = body.splitlines()
        first = lines[0].strip() if lines else ""
        if first and len(lines) > 1 and lines[1].strip().startswith("#"):
            # A repeated heading inside one chunk is a formatting slip, not
            # distinct content.
            body = "\n".join(lines[1:]).strip()
        out.append(body)
    return "\n\n".join(out)
