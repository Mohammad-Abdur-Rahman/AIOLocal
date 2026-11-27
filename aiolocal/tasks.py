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
import logging
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
from .memory.text import estimate_tokens, new_id, normalise, truncate_tokens

log = logging.getLogger(__name__)

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

STRICT_INDEX_SYSTEM = (
    "You are a page classification engine for a local document workspace. "
    "You assign every page of a document to one or more topics from the list "
    "you are given: a single page may test several topics, especially when it "
    "holds parts of different questions. Output only the classification lines, "
    "with no preamble, no explanation, no markdown and no closing remark."
)

STRICT_INDEX_PROMPT = """TASK
{instruction}

TOPICS (use these names exactly as written)
{topics}

EXPECTED PAGES: {expected}

PAGES TO CLASSIFY (units {first}-{last} of {total})
{segment}

For EACH expected page above, output one line per topic that the page tests, in
this format:
Topic Name: <page number>

Rules:
- Use a topic name from the list above, spelled exactly as given.
- A page may test more than one topic: write a separate line for each topic
  that appears on it. A question split across two pages belongs to both pages.
- Every expected page must appear at least once. Never skip a page and never
  invent one.
- The page text may be a description of the page rather than a transcription.
  Classify from what it describes; do not answer None merely because a
  description shows no formula.
- If no topic fits a page (a cover page, a blank page), write: None: <page number>
- Keep the lines in ascending page order. No other text."""

STRICT_PAGE_PROMPT = """Classify this single page of a mathematics paper.

TOPICS (use these names exactly as written):
{topics}

PAGE {page}:
{segment}

Reply with exactly one line in the form "Topic Name: {page}", where the topic
name comes from the list above. The page text may be a description of the page
rather than a transcription; classify from what it describes and do not answer
None merely because no formula is shown. If the page is a cover page, a blank
page, or holds no mathematics question, reply with "None: {page}". Output
nothing else."""

STRICT_DESCRIPTION_PROMPT = """The text below DESCRIBES page {page} of a mathematics
exam paper. It is a description rather than a transcription, so judge the page by
the question it describes, not by whether a formula is printed in the description.

TOPICS (use these names exactly as written):
{topics}

DESCRIPTION OF PAGE {page}:
{segment}

Reply with exactly one line in the form "Topic Name: {page}", where the topic
name comes from the list above. A question about matrices, vectors or
transformation matrices belongs to the Legacy topic. If the description says the
page is blank or holds no mathematics question, reply with "None: {page}".
Output nothing else."""

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
STRICT_INDEX_BATCH = 10
STRICT_SEGMENT_CHARS = 12_000
# A batch over ten pages should never need more than a few hundred tokens. A
# tighter cap bounds the damage when the model starts looping (it hits the cap
# fast instead of spinning for minutes) while still leaving room for many lines.
STRICT_MAX_TOKENS = 900
# One page needs one short answer, but the model sometimes spends a sentence
# before it ("The page is about..."), so leave room for that.
STRICT_PAGE_MAX_TOKENS = 160
# How many times a single page may be retried before it is left unassigned. Each
# retry is a fresh single-page question, so a low number keeps a stuck page from
# stalling the whole job.
STRICT_REPAIR_DEPTH = 3
# After the main pass, unassigned pages get one more sweep with a fresh retry
# budget. A batch occasionally drops a page and a repair can lose one to a
# transient model failure; the sweep means an otherwise-good job still ends up
# covering every page that holds a question, without asking the user to re-run.
STRICT_SWEEP_ROUNDS = 2


def looks_like_bulk_task(text: str, *, total_pages: int, topics: list[str] | None = None) -> bool:
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


def looks_like_strict_index(
    instruction: str,
    total_pages: int,
    topics: list[str] | None = None,
) -> bool:
    """Detect a page-by-page classification job.

    A strict-index job is different from a prose drafting job: it must emit one
    classification line per page, aggregated per topic. It is recognised by the
    combination of

    * a classifying verb ("classify", "categorise", "label", "assign", ...),
    * an exhaustive scope ("every page", "each page", "all pages", ...), and
    * a real topic list (at least five non-empty names).

    All three are required so an ordinary question about a document stays a
    normal chat turn.
    """
    if total_pages < 20:
        return False

    lowered = normalise(instruction).lower()
    words = set(re.findall(r"[a-z]+", lowered))

    classify_verbs = {
        "classify", "classifies", "categorise", "categorize", "categorised",
        "categorized", "label", "labels", "assign", "assigns", "map", "maps",
        "group", "groups", "sort", "sorts", "tag", "tags", "bucket", "buckets",
    }
    if not (words & classify_verbs):
        return False

    exhaustive = (
        "every page", "each page", "all pages", "every question", "each question",
        "every item", "each item", "page by page", "question by question",
        "entire document", "whole document", "all of the pages",
    )
    if not any(marker in lowered for marker in exhaustive):
        return False

    clean = [str(topic).strip() for topic in (topics or []) if str(topic).strip()]
    return len(clean) >= 5


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

    @property
    def strict(self) -> bool:
        """True when this job must classify every page into the given topics.

        Derived from the instruction and topics rather than stored, so a task
        restored from an older database behaves identically without a migration.
        """
        return looks_like_strict_index(
            self.instruction, total_pages=self.total_units, topics=self.topics
        )

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
            "strict": self.strict,
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
        instruction = instruction.strip()[:12000]
        clean_topics = [str(t)[:80] for t in (topics or [])][:40]
        # A strict-index job classifies pages, so it uses its own batch size and
        # ignores a caller's drafting batch (which is tuned for prose sections).
        strict = looks_like_strict_index(
            instruction, total_pages=len(units), topics=clean_topics
        )
        if strict and batch == DEFAULT_BATCH:
            batch = STRICT_INDEX_BATCH
        task = ResponseTask(
            id=new_id(),
            conversation_id=conversation_id,
            instruction=instruction,
            title=(title or instruction.splitlines()[0])[:120] or "Long task",
            topics=clean_topics,
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
        task_public = self._row_public(task_row)
        raw_chunks = [str(r["content"]) for r in rows]
        if task_public.get("strict"):
            # A strict job's readable text is the aggregated index, not the raw
            # per-segment answers. This also lets a half-finished job be read.
            assigned = _aggregate_strict(
                "\n".join(raw_chunks),
                list(task_public.get("topics") or []),
                int(task_public.get("total_units") or 0),
            )
            text = "\n".join(
                f"{topic}: {', '.join(str(p) for p in pages) if pages else 'None'}"
                for topic, pages in assigned
            )
        else:
            text = _join_chunks(raw_chunks)
        return {
            "task": task_public,
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
            "text": text,
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
            log.info(
                "task %s started: %s, %d units from %d, model %s",
                task.id,
                "strict-index" if task.strict else "prose",
                task.total_units,
                task.cursor,
                model,
            )
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
                log.info(
                    "task %s segment %d-%d (%d/%d units)",
                    task.id,
                    first_unit,
                    last_unit,
                    end,
                    task.total_units,
                )
                if task.strict:
                    # Strict-index jobs emit classification lines, not prose, so
                    # they use their own prompt and a self-healing classifier.
                    content = self._classify_segment(
                        settings, model, task, start, end
                    )
                else:
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
                if task.strict:
                    # After repair, content may legitimately contain "None: X" lines.
                    # We only fail if no topic lines were emitted at all.
                    topic_lines = [ln for ln in content.splitlines() if _STRICT_LINE_RE.match(ln)]
                    if not topic_lines:
                        task.error = (
                            f"Segment {first_unit}-{last_unit} produced no topic lines. "
                            "Retry to continue from here."
                        )
                        break
                self._append_chunk(task, start, end, first_unit, last_unit, content)
                task.cursor = end
                task.error = ""
                self._save(task)
                self._emit(task)
                log.info(
                    "task %s segment %d-%d done: %d chars (%d/%d units)",
                    task.id,
                    first_unit,
                    last_unit,
                    len(content),
                    end,
                    task.total_units,
                )

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
                log.warning(
                    "task %s stopped at %d/%d units: %s",
                    task.id,
                    task.cursor,
                    task.total_units,
                    task.error,
                )
                return

            if task.strict:
                # The main pass is done; give any page it missed one more chance
                # before the index is written, so a complete document yields a
                # complete index without a manual re-run.
                self._sweep_strict(settings, model, task)
                if task.cancel.is_set():
                    task.state = "interrupted" if self._closing else "cancelled"
                    self._save(task)
                    self._emit(task)
                    return

            self._present_and_finish(task, settings, model)
        except Exception as exc:
            task.state = "error"
            task.error = f"{type(exc).__name__}: {exc}"[:1000]
            self._save(task)
            self._emit(task)
            log.exception("task %s failed: %s", task.id, task.error)
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

    # ------------------------------------------------------------- unit mapping
    @staticmethod
    def _unit_doc_page(task: ResponseTask, index: int) -> tuple[str, int]:
        """Return (document_id, page_number) for a 0-based unit index.

        ``task.units`` entries are ``"<document_id>:<page>"`` strings. A malformed
        entry falls back to the unit ordinal so a single-document job still maps
        to the right page instead of crashing the whole classification.
        """
        raw = task.units[index] if 0 <= index < len(task.units) else ""
        doc_id, _, page = raw.partition(":")
        if page.isdigit():
            return doc_id, int(page)
        return doc_id, index + 1

    def _page_text(self, task: ResponseTask, index: int) -> str:
        doc_id, page = self._unit_doc_page(task, index)
        with self.db.read() as c:
            row = c.execute(
                "SELECT text FROM document_pages WHERE document_id=? AND page_no=?",
                (doc_id, page),
            ).fetchone()
        return str(row["text"]) if row else ""

    # ----------------------------------------------------- strict classification
    def _classify_segment(
        self, settings: Settings, model: str, task: ResponseTask, start: int, end: int
    ) -> str:
        """Classify every page in ``[start, end)`` into the user's topics.

        Strategy (cheap first, self-healing second):

        1. Ask the model to classify the whole batch in one call.
        2. Discard a *degenerate* answer (the model stuck repeating one line).
        3. For every page the batch did not cover, classify it on its own, retrying
           a few times. This repairs silent skips without paying a per-page call
           for the many pages the batch already got right.

        The returned text is a flat list of ``Topic Name: page`` lines (multi-topic
        pages contribute several lines) so the caller can append it unchanged.
        """
        indices = list(range(start, min(end, task.total_units)))
        pages = [self._unit_doc_page(task, i)[1] for i in indices]
        page_set = set(pages)
        # Covers and blank pages carry no question. Detect them up front so the
        # batch answer cannot attach a topic to them (the model sometimes files a
        # cover under "Legacy") and so repair can record "None" without a call.
        furniture = {
            page
            for index, page in zip(indices, pages)
            if _looks_like_furniture_page(self._page_text(task, index))
        }

        batch_lines: list[tuple[str, int]] = []
        segment, first_unit, last_unit = self._segment(task.units, start, end)
        if segment:
            prompt = STRICT_INDEX_PROMPT.format(
                instruction=task.instruction,
                topics="\n".join(f"- {t}" for t in task.topics),
                expected=", ".join(str(p) for p in sorted(page_set)),
                first=first_unit,
                last=last_unit,
                total=task.total_units,
                segment=segment[:STRICT_SEGMENT_CHARS],
            )
            batch = self._ask_retry(
                settings,
                model,
                prompt,
                max_tokens=STRICT_MAX_TOKENS,
                system=STRICT_INDEX_SYSTEM,
            )
            if batch and not _is_degenerate(batch):
                for name, plist in _parse_strict_lines(batch):
                    if name.strip().lower() in _UNASSIGNED_NAMES:
                        continue
                    key = _topic_key(name, task.topics)
                    if not key:
                        continue
                    for p in plist:
                        if p in page_set and p not in furniture:
                            batch_lines.append((key, p))

        covered = {p for _, p in batch_lines}
        for index in indices:
            page = self._unit_doc_page(task, index)[1]
            if page in covered:
                continue
            repaired = self._repair_unit(settings, model, task, index, depth=0)
            if repaired:
                name, p = repaired
                batch_lines.append((name, p))

        # Deduplicate identical (topic, page) pairs and order by page then topic.
        unique = sorted(set(batch_lines), key=lambda t: (t[1], t[0]))
        return "\n".join(f"{topic}: {page}" for topic, page in unique) + "\n"

    def _sweep_strict(
        self, settings: Settings, model: str, task: ResponseTask
    ) -> None:
        """Re-attempt pages the finished pass left unassigned.

        The batch and its repair already cover almost every page, but a page can
        still be lost to a batch skip that repair could not fix, or to a model
        reload partway through. Sweeping at the end gives each such page a fresh
        retry budget, so a good job does not need a manual re-run to be complete.

        Results are appended as one extra chunk spanning the whole job. That span
        is deliberate: the resume cursor is recovered from ``MAX(unit_to)``, so a
        sweep chunk quoting only its own pages could make a resumed job rewind and
        redo work.
        """
        for round_no in range(STRICT_SWEEP_ROUNDS):
            with self.db.read() as c:
                rows = c.execute(
                    "SELECT content FROM response_task_chunks WHERE task_id=? ORDER BY ordinal",
                    (task.id,),
                ).fetchall()
            assigned = _aggregate_strict(
                "\n".join(str(r["content"]) for r in rows),
                task.topics,
                task.total_units,
            )
            covered = {page for _, pages in assigned for page in pages}
            todo = [
                index
                for index in range(task.total_units)
                if self._unit_doc_page(task, index)[1] not in covered
                and not _looks_like_furniture_page(self._page_text(task, index))
            ]
            if not todo:
                return
            log.info(
                "task %s sweep %d: %d page(s) still unclassified",
                task.id,
                round_no + 1,
                len(todo),
            )
            lines: list[str] = []
            for index in todo:
                if task.cancel.is_set():
                    break
                repaired = self._repair_unit(settings, model, task, index, depth=0)
                if repaired:
                    name, page = repaired
                    lines.append(f"{name}: {page}")
            if not lines:
                continue
            body = "\n".join(lines)
            with self.db.write() as c:
                row = c.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM response_task_chunks "
                    "WHERE task_id=?",
                    (task.id,),
                ).fetchone()
                c.execute(
                    """INSERT INTO response_task_chunks(task_id, ordinal, unit_from,
                           unit_to, content, tokens, created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        task.id,
                        int(row[0] if row else 0),
                        1,
                        task.total_units,
                        body,
                        estimate_tokens(body),
                        now_ms(),
                    ),
                )
            self._save(task)
            self._emit(task)

    def _repair_unit(
        self,
        settings: Settings,
        model: str,
        task: ResponseTask,
        index: int,
        depth: int,
        deadline: float | None = None,
    ) -> tuple[str, int] | None:
        """Classify one page on its own, retrying until it answers about *this* page.

        Returns ``(topic_name, page)`` where ``topic_name`` is canonical (or the
        literal ``"None"`` for a cover/blank page), or ``None`` when the retry
        budget is exhausted and the caller should leave the page unassigned.
        ``deadline`` caps the wall-clock time spent on one page, so a wedged
        model server cannot stall a 204-page job on a single page.
        """
        if deadline is None:
            deadline = time.time() + 150.0
        _, page = self._unit_doc_page(task, index)
        text = self._page_text(task, index)
        if not text.strip() or _looks_like_furniture_page(text):
            return ("None", page)

        if depth > STRICT_REPAIR_DEPTH or time.time() > deadline:
            log.warning(
                "task %s: page %d unclassified after %d tries", task.id, page, depth
            )
            return None

        template = (
            STRICT_DESCRIPTION_PROMPT if _looks_like_description(text)
            else STRICT_PAGE_PROMPT
        )
        prompt = template.format(
            topics="\n".join(f"- {t}" for t in task.topics),
            page=page,
            segment=text[:STRICT_SEGMENT_CHARS],
        )
        content = self._ask_retry(
            settings,
            model,
            prompt,
            max_tokens=STRICT_PAGE_MAX_TOKENS,
            system=STRICT_INDEX_SYSTEM,
        )

        # Normalise once: strip bullets and a trailing full stop, which the model
        # adds often ("Legacy ...: 1.") and which would otherwise hide the line.
        lines = [
            ln.strip().lstrip("-*\u2022 ").strip().rstrip(".").strip()
            for ln in content.splitlines()
            if ln.strip()
        ]

        # First pass: a "Topic Name: <page>" (or "None: <page>") line about this page.
        for raw in lines:
            match = _STRICT_LINE_RE.match(raw)
            if not match:
                continue
            if int(re.split(r"[,\s]+", match.group("pages").strip())[0] or 0) != page:
                continue
            name = match.group("topic").strip().rstrip(".").strip()
            if name.lower() in _UNASSIGNED_NAMES:
                # A substantial real page answered "None" is usually a miss; give
                # it another chance before accepting it as genuinely empty.
                if _substantial(text) and depth < STRICT_REPAIR_DEPTH:
                    return self._repair_unit(
                        settings, model, task, index, depth + 1, deadline
                    )
                return ("None", page)
            key = _topic_key(name, task.topics)
            if key:
                return (key, page)

        # Second pass: the model sometimes omits the page number on a single-page
        # prompt ("Mensuration" or just "None"). The page is known, so attach it
        # rather than discarding a real answer. Lines that *do* carry a (wrong)
        # page number are skipped: a mismatched page is a genuine disagreement,
        # not a formatting slip.
        for raw in lines:
            if _STRICT_LINE_RE.match(raw):
                continue
            if raw.lower() in _UNASSIGNED_NAMES:
                if _substantial(text) and depth < STRICT_REPAIR_DEPTH:
                    return self._repair_unit(
                        settings, model, task, index, depth + 1, deadline
                    )
                return ("None", page)
            key = _topic_key(raw, task.topics)
            if key:
                return (key, page)

        # Last resort: the topic may be buried in a sentence rather than alone on
        # a line. Accept it only when the answer names exactly one topic.
        buried = _topic_in_text(content, task.topics)
        if buried:
            return (buried, page)

        log.warning(
            "task %s: page %d unusable answer %r", task.id, page, content[:200]
        )
        return self._repair_unit(
            settings, model, task, index, depth + 1, deadline
        )

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
        if task.strict:
            # A strict-index job's deliverable is the aggregated index, not a
            # drafted prose document, so it is assembled from every chunk.
            summary_id = self._present_strict(task, settings)
        else:
            summary_id = self._present(task, settings, model)
        task.summary_id = summary_id or ""
        task.state = "done"
        task.error = ""
        task.updated_at = now_ms()
        self._save(task)
        self._emit(task)
        log.info(
            "task %s done: %d units, summary %s",
            task.id,
            task.total_units,
            task.summary_id or "(none)",
        )

    def _present_strict(self, task: ResponseTask, settings: Settings) -> str:
        """Aggregate every segment into the strict "Topic Name: pages" index.

        The file written to disk is exactly the deliverable: one line per topic,
        pages ascending and deduplicated, ``None`` when a topic matched nothing.
        A page may appear under several topics when it holds sub-parts that test
        different topics; it is deduplicated only *within* a topic. A page the
        model skipped is reported as unassigned instead of silently vanishing.
        """
        with self.db.read() as c:
            rows = c.execute(
                "SELECT content FROM response_task_chunks WHERE task_id=? ORDER BY ordinal",
                (task.id,),
            ).fetchall()
        raw = "\n".join(str(r["content"]) for r in rows)

        assigned = _aggregate_strict(raw, task.topics, task.total_units)
        lines = [f"{topic}: {', '.join(str(p) for p in pages) if pages else 'None'}"
                 for topic, pages in assigned]
        text = "\n".join(lines) + "\n"

        path = self.dir / f"task-{task.id}.md"
        # Ensure the directory exists (it may not be created automatically).
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass

        unassigned = [
            page
            for page in range(1, task.total_units + 1)
            if not any(page in pages for _, pages in assigned)
        ]
        summary_body = (
            f"# {task.title}\n\n"
            f"**Task.** {task.instruction}\n\n"
            f"**Result.** {task.total_units} pages classified into {len(task.topics)} "
            f"topics; stored at `{path.name}`.\n\n"
            "## Index\n"
            + text
        )
        if unassigned:
            preview = ", ".join(str(p) for p in unassigned[:40])
            more = "" if len(unassigned) <= 40 else f" … (+{len(unassigned) - 40} more)"
            summary_body += (
                f"\n## Not classified\nPages the model did not assign: {preview}{more}\n"
            )
        try:
            return store.insert_summary(
                self.db,
                title=f"Index: {task.title}"[:200],
                body=summary_body,
                topic_slug="tasks",
                kind="task",
                salience=0.75,
                confidence=0.8,
                source="task",
                entities=["task", "index"],
                embed_dim=settings.embed_dim,
            )
        except Exception:
            return ""

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
        # Ensure the directory exists (it may not be created automatically).
        path.parent.mkdir(parents=True, exist_ok=True)
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

    def _ask(
        self,
        settings: Settings,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        system: str = TASK_SYSTEM,
        timeout: float = 120.0,
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout, connect=5.0), trust_env=False
            ) as client:
                response = client.post(
                    api_url(settings.upstream_base_url, "chat/completions"), json=payload
                )
            if response.status_code >= 400:
                log.warning("task model returned HTTP %d: %s", response.status_code,
                            response.text[:200])
                return ""
            return str(response.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            log.warning("task model call failed: %s", exc)
            return ""

    def _ask_retry(
        self,
        settings: Settings,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        system: str = TASK_SYSTEM,
        attempts: int = 3,
    ) -> str:
        """``_ask`` with retries for transient upstream trouble.

        A busy or reloading local model server can drop a request (HTTP 404
        "graph definition not found") or answer with an empty body. Retrying
        turns that into a normal answer instead of a silently unclassified page.
        A *fast* failure is usually the server still loading, so wait longer;
        a slow one already burned the timeout, so retry promptly.
        """
        answer = ""
        for attempt in range(1, max(1, attempts) + 1):
            started = time.time()
            answer = self._ask(
                settings, model, prompt, max_tokens=max_tokens, system=system
            )
            if answer:
                return answer
            if attempt < attempts:
                fast = (time.time() - started) < 5.0
                time.sleep(2.0 if fast else 0.5)
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
                # chunks are never rewritten. A strict-index job keeps its own
                # smaller batch so a restart cannot change the segment size.
                batch = STRICT_INDEX_BATCH if task.strict else DEFAULT_BATCH
                thread = threading.Thread(
                    target=self._run, args=(task, batch),
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


# --------------------------------------------------------------- strict index
# "Algebra: 3, 5, 7" and "Algebra: 3" are both accepted; a bare number is not,
# because it carries no topic and would silently drop the page.
_STRICT_LINE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?P<topic>.+?)\s*[:=\u2014]\s*"
    r"(?P<pages>[0-9][0-9,\s\u2013-]*)\s*$"
)

# Words a model uses when it cannot place a page. These are not topics: in the
# required output "None" is the placeholder for an empty page list.
_UNASSIGNED_NAMES = {
    "none", "n a", "na", "n/a", "unclassified", "unassigned", "other", "unknown",
}


def _is_degenerate(text: str) -> bool:
    """True when the answer is the model stuck in a repetition loop.

    A healthy batch over several pages produces many distinct ``Topic: page``
    lines (they differ by page number). A loop emits the *same* line over and
    over. Six or more lines with a single line repeated across a third of them
    is a loop, not a real classification.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    return max(lines.count(ln) for ln in set(lines)) >= max(5, len(lines) * 0.35)


def _substantial(text: str) -> bool:
    """True when a page holds enough text to be a real question, not a divider."""
    return len(text.strip()) >= 160


_DESCRIPTION_MARKERS = (
    "visible content:",
    "purpose:",
    "no diagrams, charts, tables, or equations",
    "no diagrams, charts, tables or equations",
)


def _looks_like_description(text: str) -> bool:
    """True when the stored page text describes the page instead of transcribing it.

    Sparse pages are sometimes read by a vision pass that writes "Subject: ...
    Visible content: ..." rather than the question text. Such a description still
    names what the page asks, so it must be classified from its subject and not
    treated as an empty page.
    """
    s = " ".join(text.split()).lower()
    return any(marker in s for marker in _DESCRIPTION_MARKERS)


def _looks_like_furniture_page(text: str) -> bool:
    """True for a blank or exam-cover page (no mathematics to classify).

    The test is deliberately narrow. The last page of every paper ends with the
    standard "Permission to reproduce items ... copyright" notice, and that page
    usually still holds a real question — so the notice alone must never count
    as furniture. A cover page is recognised by the paper header plus its
    instructions, and a blank page by its own label.
    """
    s = " ".join(text.split()).lower()
    if "blank page" in s:
        return True
    if "visually blank" in s:
        return True
    if "cambridge" in s and "mathematics (syllabus d)" in s and (
        "read these instructions" in s
        or "candidates answer on the question paper" in s
        or "additional materials" in s
    ):
        return True
    return False


def _parse_strict_lines(text: str) -> list[tuple[str, list[int]]]:
    """Pull ``(topic, pages)`` pairs out of one model answer.

    Tolerates the shapes a model actually produces: markdown bullets, bold topic
    names, "and" before the last page, and en dashes in ranges. Anything that
    does not name a topic and at least one page is ignored rather than guessed.
    """
    pairs: list[tuple[str, list[int]]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[-*\u2022]+\s*", "", stripped)
        stripped = stripped.replace("**", "").replace("`", "")
        match = _STRICT_LINE_RE.match(stripped)
        if not match:
            continue
        topic = match.group("topic").strip().rstrip(".").strip()
        if not topic:
            continue
        pages: list[int] = []
        for chunk in re.split(r"[,\s]+", match.group("pages")):
            chunk = chunk.strip()
            if not chunk:
                continue
            if re.fullmatch(r"[0-9]+", chunk):
                pages.append(int(chunk))
                continue
            span = re.fullmatch(r"([0-9]+)[\u2013-]([0-9]+)", chunk)
            if span:
                low, high = int(span.group(1)), int(span.group(2))
                if 0 < low <= high <= 100_000:
                    pages.extend(range(low, high + 1))
        if pages:
            pairs.append((topic, pages))
    return pairs


_TOPIC_STOPWORDS = {
    "and", "of", "the", "a", "an", "for", "with", "in", "to", "out",
    "current", "earlier", "papers", "syllabus", "amp",
}

# Terms models use that no topic name mentions verbatim, mapped to the canonical
# topic that owns them. Checked before the word-overlap fallback so that, e.g.,
# "Exponential Equations & Logarithms" (solving 2^x = 100) lands on
# Surds & Indices rather than Equations & Inequalities purely by sharing the
# word "equations". Each term is matched against the squashed topic name.
#
# Terms are deliberately specific: "exponentialequation" (not bare
# "exponential") so that "Exponential Graphs" keeps falling through to the
# word-overlap fallback and, correctly, onto Graphs & Kinematics.
_TOPIC_KEYWORD_ALIASES = {
    "exponentialequation": "Surds & Indices",
    "logarith": "Surds & Indices",      # logarithm / logarithms
    "standardform": "Surds & Indices",  # standard form, A x 10^n
    "rationalis": "Surds & Indices",    # rationalising denominators
    "pythagor": "Trigonometry (2D & 3D)",  # Pythagoras' theorem
    "sohcahtoa": "Trigonometry (2D & 3D)",
    "sinerule": "Trigonometry (2D & 3D)",
    "cosinerule": "Trigonometry (2D & 3D)",
    "matric": "Legacy / Out-of-Current-Syllabus (2019 & Earlier Papers)",
    "linearprogramming": "Legacy / Out-of-Current-Syllabus (2019 & Earlier Papers)",
}


def _topic_tokens(text: str) -> set[str]:
    """Lowercase word tokens with punctuation and filler words removed.

    Used to recognise a *shortened* topic ("Number: Arithmetic, Percentages &
    Ratio" for the full "Number: Arithmetic, Fractions, Decimals, Percentages &
    Ratio") where a plain substring test fails because middle words were dropped.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w and w not in _TOPIC_STOPWORDS}


def _topic_exact(name: str, topics: list[str]) -> str:
    """Match one candidate string against the topic list (no prefix stripping)."""
    squashed = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not squashed:
        return ""
    exact = [
        topic
        for topic in topics
        if re.sub(r"[^a-z0-9]+", "", topic.lower()) == squashed
    ]
    if exact:
        return exact[0]

    if len(squashed) >= 6:
        subs = [
            topic
            for topic in topics
            if squashed in re.sub(r"[^a-z0-9]+", "", topic.lower())
            or re.sub(r"[^a-z0-9]+", "", topic.lower()) in squashed
        ]
        if len(subs) == 1:
            return subs[0]

    name_tokens = _topic_tokens(name)
    if name_tokens:
        supersets = [
            topic for topic in topics if name_tokens <= _topic_tokens(topic)
        ]
        if len(supersets) == 1:
            return supersets[0]
    return ""


def _topic_keyword_alias(name: str, topics: list[str]) -> str:
    """Map a distinctive term the model invented onto the topic that owns it.

    Only applies when the alias resolves to exactly one canonical topic, so a
    name that trips two aliases ("sine rule" plus "cosine rule" both point to
    Trigonometry, fine; a name mixing unrelated terms stays unassigned).
    """
    squashed = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not squashed:
        return ""
    hits = {
        topic
        for term, topic in _TOPIC_KEYWORD_ALIASES.items()
        if term in squashed and topic in topics
    }
    return hits.pop() if len(hits) == 1 else ""


def _topic_nearest(name: str, topics: list[str]) -> str:
    """Generic closest-topic fallback for wording no exact match covers.

    Scores each topic by how many content words the model's name shares with
    it and returns the single clear winner. Only reached after exact,
    substring, superset and keyword-alias matching have all failed; a name
    that shares words with two topics equally is left unassigned rather than
    guessed.
    """
    name_words = _topic_tokens(name)
    if not name_words:
        return ""
    best: list[tuple[int, str]] = []
    for topic in topics:
        overlap = len(name_words & _topic_tokens(topic))
        if overlap:
            best.append((overlap, topic))
    if not best:
        return ""
    best.sort(key=lambda pair: (-pair[0], pair[1]))
    top_score = best[0][0]
    winners = [topic for score, topic in best if score == top_score]
    return winners[0] if len(winners) == 1 else ""


def _topic_key(name: str, topics: list[str]) -> str:
    """Map a model-written topic onto the user's spelling.

    Models re-case, re-punctuate, pluralise and *shorten* a long topic by
    dropping middle words ("Number: Arithmetic, Percentages & Ratio"). They also
    append a subtopic label after another colon ("Number: Arithmetic: Compound
    Interest"). This tries the full string first, then progressively drops the
    trailing colon segments, and only ever returns a genuine topic:

    1. exact squashed-equal (ignoring punctuation/case);
    2. a unique substring match;
    3. a unique topic whose word set is a superset of the name's word set;
    4. a keyword alias for a term no topic names verbatim (e.g. "logarithm");
    5. the unique closest topic by shared content words.

    An ambiguous short word ("Geometry" fits both "Coordinate Geometry" and the
    "Geometry: Angles, ..." topic) is never guessed, and an unknown name returns
    "" so the deliverable keeps exactly the user's topic lines.
    """
    candidates = [name]
    parts = name.split(":")
    for end in range(len(parts) - 1, 0, -1):
        candidates.append(":".join(parts[:end]))
    for candidate in candidates:
        key = _topic_exact(candidate.strip(), topics)
        if key:
            return key
    key = _topic_keyword_alias(name, topics)
    if key:
        return key
    return _topic_nearest(name, topics)


def _topic_in_text(text: str, topics: list[str]) -> str:
    """Last resort: find one topic named somewhere inside a sentence.

    The model sometimes wraps its answer in prose ("This page is about
    statistics, so the line is Statistics: 12"). A whole-line match misses that,
    so scan the answer for exactly one identifiable topic. If the text names two
    or more topics the answer is ambiguous and nothing is returned.
    """
    squashed = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not squashed:
        return ""
    named = [
        topic
        for topic in topics
        if re.sub(r"[^a-z0-9]+", "", topic.lower()) in squashed
    ]
    if len(named) == 1:
        return named[0]
    if named:
        return ""
    words = _topic_tokens(text)
    if not words:
        return ""
    supersets = [topic for topic in topics if _topic_tokens(topic) <= words]
    if len(supersets) == 1:
        return supersets[0]
    if len(supersets) > 1:
        return ""
    key = _topic_keyword_alias(text, topics)
    if key:
        return key
    return _topic_nearest(text, topics)


def _aggregate_strict(
    raw: str, topics: list[str], total_units: int
) -> list[tuple[str, list[int]]]:
    """Fold every segment's answer into one ordered topic-to-pages index.

    Ordering follows the user's topic list so the file reads the way it was
    asked for. Pages are sorted and deduplicated *within* a topic. A page may
    appear in several topics: a multi-part question can test two topics on the
    same page, and dropping one of them would lose a real classification. A
    topic with no pages still gets its ``None`` line because the required output
    has one line per topic.

    A page the model could not place is *not* given a "None" topic: in this
    format ``None`` is the placeholder for an empty page list, not a topic name.
    Such a page simply stays unassigned and is reported by the caller.
    """
    buckets: dict[str, set[int]] = {topic: set() for topic in topics}

    for name, pages in _parse_strict_lines(raw):
        if name.strip().lower() in _UNASSIGNED_NAMES:
            continue
        key = _topic_key(name, topics)
        if not key:
            continue
        buckets[key].update(p for p in pages if 1 <= p <= total_units)

    return [(topic, sorted(buckets[topic])) for topic in topics]
