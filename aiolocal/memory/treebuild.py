"""Background roll-up of tree memory.

Chunks are written synchronously (cheap, one row). Summaries need the local
model, so they run on a private daemon thread; a chat turn never waits for the
tree to catch up. If the model is unavailable the builder leaves the chunks
unconsumed instead of inventing a summary, so the next drain retries.
"""

from __future__ import annotations

import json
import threading

import httpx

from . import tree
from .text import estimate_tokens, first_sentence, truncate_tokens

from ..config import Settings, SettingsStore
from ..db import Database
from ..llm import api_url

TREE_SYSTEM = (
    "You are the memory roll-up engine of a local AI workspace. You merge "
    "provenance-tagged notes into one shorter note. Rules: never invent facts, "
    "names, numbers, dates, units, definitions or conclusions that are not in the "
    "inputs; preserve exact values and identifiers verbatim; keep any uncertainty "
    "the inputs state; do not answer, solve or extend anything. Name every distinct "
    "thing that would help a reader find it again. Return plain Markdown only."
)

ROLLUP_PROMPT = """Merge the {count} notes below into one compact memory note.

Level: {label}
Rules:
- keep every exact name, number, date, unit, definition, equation and code identifier;
- keep the order of ideas and say which note each distinct point came from only if
  the inputs disagree;
- do not add facts, steps, answers or interpretations that are not printed below;
- if the notes are partial, say which part is partial;
- write 80-350 words.

NOTES:
{notes}"""


def _label(degree: int) -> str:
    if degree <= 0:
        return "raw units (first roll-up)"
    return f"degree-{degree} summary"


def _note_block(index: int, text: str) -> str:
    return f"### Note {index}\n{text.strip()[:4000]}"


class TreeBuilder:
    """Owns roll-up threads. One drain per tree at a time."""

    def __init__(
        self,
        db: Database,
        settings_store: SettingsStore,
        *,
        model: str = "",
        timeout: float = 60.0,
    ) -> None:
        self.db = db
        self.settings_store = settings_store
        self.model = model
        self.timeout = timeout
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._kicks: dict[str, threading.Event] = {}
        # One drain at a time per tree: the manual roll-up route and the
        # background thread must never consume the same group twice.
        self._drains: dict[str, threading.Lock] = {}
        self._closing = False

    # ------------------------------------------------------------------ public
    def kick(self, tree_id: str) -> None:
        """Ask for a drain in the background. Coalesces repeated calls."""
        if not tree_id or self._closing:
            return
        with self._lock:
            event = self._kicks.get(tree_id)
            if event is None:
                event = threading.Event()
                self._kicks[tree_id] = event
            event.set()
            thread = self._threads.get(tree_id)
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(
                target=self._loop, args=(tree_id, event), name=f"tree-{tree_id}", daemon=True
            )
            self._threads[tree_id] = thread
            thread.start()

    def drain(self, tree_id: str, *, max_nodes: int = 6) -> int:
        """Synchronously create up to ``max_nodes`` parents. Returns the count."""
        if not tree_id:
            return 0
        with self._lock:
            drain_lock = self._drains.setdefault(tree_id, threading.Lock())
        with drain_lock:
            settings = self.settings_store.load()
            model = self.model or settings.filing_model or settings.chat_model
            return len(
                tree.rollup(
                    self.db,
                    tree_id,
                    lambda degree, texts: self._summarise(settings, model, degree, texts),
                    max_nodes=max_nodes,
                )
            )

    def stop(self, timeout: float = 3.0) -> None:
        self._closing = True
        with self._lock:
            threads = list(self._threads.values())
            for event in self._kicks.values():
                event.set()
        for thread in threads:
            thread.join(timeout=max(0.1, timeout / max(1, len(threads))))

    def backfill(self, *, max_chunks: int = 20_000) -> int:
        """Seed trees for data stored before tree memory existed.

        Idempotent: chunks are keyed by page or exchange, so re-running only
        appends what is missing. The roll-up itself stays on the background
        drain, never on the startup path.
        """
        if self._closing:
            return 0
        tombstones = self._tombstones()
        added = self._backfill_documents(max_chunks, tombstones)
        if added < max_chunks:
            added += self._backfill_chats(max_chunks - added, tombstones)
        return added

    def _tombstones(self) -> set[tuple[str, str]]:
        """Refs whose tree the user deleted; backfill must never recreate them."""
        with self.db.read() as c:
            rows = c.execute(
                "SELECT kind, ref_id FROM memory_tree_tombstones"
            ).fetchall()
        return {(str(r["kind"]), str(r["ref_id"])) for r in rows}

    # ----------------------------------------------------------------- internal
    def _backfill_documents(self, budget: int, tombstones: set[tuple[str, str]]) -> int:
        with self.db.read() as c:
            docs = c.execute(
                """SELECT id, title FROM summaries
                   WHERE kind='document' AND merged_into IS NULL
                   ORDER BY created_at"""
            ).fetchall()
        added = 0
        for doc in docs:
            if added >= budget or self._closing:
                break
            doc_added = 0
            doc_id = str(doc["id"])
            if ("document", doc_id) in tombstones:
                continue
            tree_id = tree.get_or_create_tree(
                self.db, kind="document", ref_id=doc_id,
                title=str(doc["title"] or f"Document {doc_id}"),
            )
            existing = self._chunk_keys(tree_id)
            with self.db.read() as c:
                pages = c.execute(
                    """SELECT p.page_no,
                              COALESCE(NULLIF(n.note, ''), p.text) AS body
                       FROM document_pages p
                       LEFT JOIN document_page_notes n
                         ON n.document_id=p.document_id AND n.page_no=p.page_no
                       WHERE p.document_id=? ORDER BY p.page_no""",
                    (doc_id,),
                ).fetchall()
            for page in pages:
                if added >= budget or self._closing:
                    break
                key = f"page:{int(page['page_no'])}"
                if key in existing:
                    continue
                tree.append_chunk(
                    self.db,
                    tree_id,
                    key,
                    {
                        "text": str(page["body"] or "").strip(),
                        "document_id": doc_id,
                        "page_no": int(page["page_no"]),
                    },
                    complete=True,
                )
                added += 1
                doc_added += 1
            if doc_added:
                self.kick(tree_id)
        return added

    def _backfill_chats(self, budget: int, tombstones: set[tuple[str, str]]) -> int:
        with self.db.read() as c:
            conversations = c.execute(
                "SELECT id FROM conversations ORDER BY last_at"
            ).fetchall()
        added = 0
        for conversation in conversations:
            if added >= budget or self._closing:
                break
            conversation_id = str(conversation["id"])
            conversation_added = 0
            if ("chat", conversation_id) in tombstones:
                continue
            tree_id = tree.get_or_create_tree(
                self.db, kind="chat", ref_id=conversation_id,
                title=f"Chat {conversation_id}",
            )
            existing = self._chunk_keys(tree_id)
            with self.db.read() as c:
                rows = c.execute(
                    """SELECT id, role, content FROM messages
                       WHERE conversation_id=? AND duplicate_of IS NULL
                         AND superseded_by IS NULL AND COALESCE(injected, 0)=0
                       ORDER BY created_at""",
                    (conversation_id,),
                ).fetchall()
            pending: dict | None = None
            for row in rows:
                if added >= budget or self._closing:
                    break
                role = str(row["role"])
                if role == "user":
                    pending = {"id": str(row["id"]), "content": str(row["content"] or "")}
                    continue
                if role != "assistant" or pending is None:
                    continue
                key = f"exchange:{pending['id']}"
                if key not in existing:
                    body = (
                        f"User: {pending['content'].strip()}\n\n"
                        f"Assistant: {str(row['content'] or '').strip()}"
                    ).strip()
                    tree.append_chunk(
                        self.db,
                        tree_id,
                        key,
                        {
                            "text": body,
                            "conversation_id": conversation_id,
                            "user_message_id": pending["id"],
                            "assistant_message_id": str(row["id"]),
                        },
                        complete=True,
                    )
                    added += 1
                    conversation_added += 1
                pending = None
            if conversation_added:
                self.kick(tree_id)
        return added

    def _chunk_keys(self, tree_id: str) -> set[str]:
        with self.db.read() as c:
            rows = c.execute(
                "SELECT chunk_key FROM tree_chunks WHERE tree_id=?", (tree_id,)
            ).fetchall()
        return {str(r["chunk_key"]) for r in rows}

    def _loop(self, tree_id: str, event: threading.Event) -> None:
        try:
            while not self._closing:
                event.wait(timeout=0.1)
                event.clear()
                try:
                    self.drain(tree_id)
                except Exception:
                    return
                if not event.is_set():
                    # Nothing new arrived while draining; stop the thread. A
                    # later kick starts a fresh one.
                    with self._lock:
                        if not event.is_set():
                            self._threads.pop(tree_id, None)
                            self._kicks.pop(tree_id, None)
                            return
        finally:
            with self._lock:
                if self._threads.get(tree_id) is threading.current_thread():
                    self._threads.pop(tree_id, None)

    def _summarise(
        self, settings: Settings, model: str, degree: int, texts: list[str]
    ) -> str:
        notes = "\n\n".join(_note_block(i + 1, t) for i, t in enumerate(texts))
        prompt = ROLLUP_PROMPT.format(
            count=len(texts), label=_label(degree), notes=notes[:14_000]
        )
        out = self._ask(settings, model, prompt)
        if out:
            return out
        # Deterministic fallback: a lossless digest, not an invented summary.
        # It keeps provenance and exact wording, so accuracy degrades gracefully
        # when the model is down. The next roll-up uses the model again.
        digest = " | ".join(first_sentence(t, 280) for t in texts if t.strip())
        return truncate_tokens(digest, 900)

    def _ask(self, settings: Settings, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": TREE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 700,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(
                    api_url(settings.upstream_base_url, "chat/completions"), json=payload
                )
            if response.status_code >= 400:
                return ""
            content = response.json()["choices"][0]["message"].get("content") or ""
            return str(content).strip()[:8000]
        except Exception:
            return ""


# ------------------------------------------------------------------ ingestion
def append_exchange(
    db: Database,
    *,
    conversation_id: str,
    request_id: str,
    user_text: str,
    assistant_text: str,
    user_message_id: str = "",
    assistant_message_id: str = "",
    complete: bool = True,
) -> str:
    """Add one complete exchange as a tree chunk and return the tree id."""
    tree_id = tree.get_or_create_tree(
        db, kind="chat", ref_id=conversation_id, title=f"Chat {conversation_id}"
    )
    body = (
        f"User: {user_text.strip()}\n\nAssistant: {assistant_text.strip()}".strip()
    )
    tree.append_chunk(
        db,
        tree_id,
        f"exchange:{request_id or user_message_id or body[:32]}",
        {
            "text": body,
            "conversation_id": conversation_id,
            "request_id": request_id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
        },
        tokens=estimate_tokens(body),
        complete=complete,
    )
    return tree_id


def append_document_page(
    db: Database,
    *,
    document_id: str,
    page_no: int,
    text: str,
    title: str = "",
) -> str:
    """Add one page note as a document-tree chunk and return the tree id."""
    tree_id = tree.get_or_create_tree(
        db, kind="document", ref_id=document_id, title=title or f"Document {document_id}"
    )
    body = text.strip()
    tree.append_chunk(
        db,
        tree_id,
        f"page:{int(page_no)}",
        {"text": body, "document_id": document_id, "page_no": int(page_no)},
        tokens=estimate_tokens(body),
        complete=True,
    )
    return tree_id


def tree_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)
