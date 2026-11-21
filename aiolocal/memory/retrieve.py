"""Memory read path.

The design goal is a deterministic, fast retrieval that never blocks a reply. Every
stage is best-effort: if one fails, the reply proceeds with less memory. The
retrieval plan is logged so the Memory Inspector can explain what the model saw.

Scoring is reciprocal rank fusion (RRF) across independent ranked lists. BM25 and
cosine scores are not comparable, so fusing ranks avoids per-corpus calibration as
the database grows.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import store, semantic, tree
from .text import embed, estimate_tokens, keywords, normalise

from ..config import Settings
from ..db import Database, now_ms

RRF_K = 60
MAX_SUMMARIES = 8
MAX_EXACT_MESSAGES = 12
EXPAND_PER_SUMMARY = 6
GATE_NORMALISED = 0.06
MAX_TREE_ITEMS = 6

_PAGE_RANGE_RE = re.compile(
    r"(?im)^\s*(?:pages?|p\.)\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*$"
)
_PAGE_RANGE_INLINE_RE = re.compile(
    r"(?i)\b(?:pages?|p\.)\s*(\d+)\s*(?:[-–]\s*(\d+))?\b"
)


def parse_page_refs(query: str) -> list[tuple[int, int]]:
    """Extract explicit page references from a query.

    Supports "page 12", "pages 3-5" and inline forms such as
    "what does page 12 say about X". Ranges are normalised so the smaller
    number comes first. Duplicates are removed while preserving order.
    """
    text = str(query or "")
    refs: list[tuple[int, int]] = []
    for match in _PAGE_RANGE_INLINE_RE.finditer(text):
        first = int(match.group(1))
        second = int(match.group(2)) if match.group(2) else first
        low, high = (first, second) if first <= second else (second, first)
        if low < 1:
            continue
        refs.append((low, high))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def document_pages_range(
    db: Database, document_id: str, first: int, last: int, *, limit: int = 50
) -> list[MemoryItem]:
    """Return exact page excerpts for a page range, bypassing vector thresholds."""
    if not document_id:
        return []
    low, high = (first, last) if first <= last else (last, first)
    low = max(1, low)
    items: list[MemoryItem] = []
    with db.read() as c:
        rows = c.execute(
            """SELECT p.page_no, p.text, p.created_at AS page_at, d.title
               FROM document_pages p JOIN summaries d ON d.id=p.document_id
               WHERE p.document_id=? AND p.page_no BETWEEN ? AND ?
               ORDER BY p.page_no LIMIT ?""",
            (document_id, low, high, max(1, min(int(limit), 200))),
        ).fetchall()
    for row in rows:
        body = str(row["text"])[:6000]
        items.append(
            MemoryItem(
                kind="document_page",
                id=f"{document_id}:{int(row['page_no'])}",
                title=f"{row['title']} — page {row['page_no']}",
                body=body,
                score=1.0,
                created_at=int(row["page_at"] or 0),
                salience=0.66,
                tokens=estimate_tokens(body),
                summary_kind="document_page",
            )
        )
    return items

_CONTINUATION_PHRASES = {
    "make it shorter", "make it longer", "again", "why", "why?", "continue", "go on",
    "fix that", "try again", "same", "and then", "what about that", "explain more",
    "more detail", "elaborate", "rewrite", "rephrase", "simplify", "expand",
    "keep going", "do it", "yes", "no", "ok", "okay", "thanks", "thank you",
}
# One-word follow-ups must stay continuations without turning every short lookup
# (for example "integral") into one.
_CONTINUATION_WORDS = {
    "again", "continue", "more", "less", "same", "why", "how", "what", "huh", "ok",
    "okay", "yes", "no", "thanks", "thank", "you", "it", "that", "this", "those",
    "these", "them", "there", "then", "go", "on",
}


@dataclass
class MemoryItem:
    kind: str  # "summary" | "exact"
    id: str
    title: str
    body: str
    score: float = 0.0
    topic: str | None = None
    message_ids: list[str] = field(default_factory=list)
    created_at: int = 0
    salience: float = 0.5
    tokens: int = 0
    summary_kind: str = ""  # summaries only: episodic/document/...

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "summary_kind": self.summary_kind,
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "score": round(self.score, 5),
            "topic": self.topic,
            "message_ids": self.message_ids,
            "created_at": self.created_at,
            "salience": round(self.salience, 3),
            "tokens": self.tokens or estimate_tokens(self.body),
        }


@dataclass
class RetrievalResult:
    intent: str
    items: list[MemoryItem]
    plan: dict
    latency_ms: int

    @property
    def token_count(self) -> int:
        return sum(i.tokens or estimate_tokens(i.body) for i in self.items)


def classify_intent(query: str) -> str:
    text = normalise(query).lower()
    if not text:
        return "continuation"
    if text in _CONTINUATION_PHRASES:
        return "continuation"
    words = text.replace("?", "").replace("!", "").split()
    if words and all(word in _CONTINUATION_WORDS for word in words):
        return "continuation"
    if len(words) <= 4:
        deictic = {"it", "that", "this", "those", "these", "them", "again", "more", "less"}
        if deictic & set(words):
            return "continuation"
    return "lookup"


def _recency_boost(created_at: int) -> float:
    if not created_at:
        return 1.0
    age_days = max(0.0, (now_ms() - created_at) / 86_400_000.0)
    return 1.0 + 0.15 * math.exp(-age_days / 45.0)


def _topic_hits(db: Database, query: str) -> list[str]:
    words = set(keywords(query, limit=10))
    if not words:
        return []
    slugs: list[str] = []
    for topic in store.list_topics(db):
        hay = f"{topic['slug']} {topic['name']}".lower()
        if any(w in hay for w in words):
            slugs.append(topic["slug"])
    return slugs


def _topics_for_slugs(db: Database, slugs: list[str], limit: int = 8) -> list[str]:
    if not slugs:
        return []
    q = ",".join("?" for _ in slugs)
    with db.read() as c:
        rows = c.execute(
            f"""SELECT s.id FROM summaries s JOIN topics t ON t.id=s.topic_id
                WHERE t.slug IN ({q}) AND s.merged_into IS NULL
                ORDER BY s.salience DESC, s.created_at DESC LIMIT ?""",
            (*slugs, limit),
        ).fetchall()
    return [r["id"] for r in rows]


def _vector_rank(
    query: str, rows: list[tuple[str, object]], dim: int, threshold: float, limit: int
) -> list[str]:
    pairs = [(owner_id, np.asarray(vec, dtype=np.float32)) for owner_id, vec in rows]
    pairs = [(owner_id, vec) for owner_id, vec in pairs if vec.size == dim]
    if not pairs:
        return []
    qv = embed(query, dim)
    return _rank_pairs(qv, pairs, threshold, limit)


def _rank_pairs(
    qv: np.ndarray, pairs: list[tuple[str, np.ndarray]], threshold: float, limit: int
) -> list[str]:
    qnorm = float(np.linalg.norm(qv))
    if qnorm == 0 or not pairs:
        return []
    matrix = np.vstack([vec for _, vec in pairs])
    norms = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(norms * qnorm, 1e-9)
    sims = (matrix @ qv) / denom
    order = np.argsort(-sims)[:limit]
    return [pairs[i][0] for i in order if float(sims[i]) > threshold]


def _vector_summaries(db: Database, query: str, settings: Settings, limit: int = 40) -> list[str]:
    return _vector_lists(
        db, "summary", query, settings, threshold=0.02, limit=limit, cap=50_000
    )


def _vector_messages(db: Database, query: str, settings: Settings, limit: int = 30) -> list[str]:
    return _vector_lists(
        db, "message", query, settings, threshold=0.10, limit=limit, cap=20_000
    )


def _vector_lists(
    db: Database,
    owner_type: str,
    query: str,
    settings: Settings,
    *,
    threshold: float,
    limit: int,
    cap: int,
) -> list[str]:
    """Fuse semantic vectors (when configured) with the always-present hash vectors."""
    out: list[str] = []
    seen: set[str] = set()

    def add(ids: list[str]) -> None:
        for owner_id in ids:
            if owner_id not in seen:
                seen.add(owner_id)
                out.append(owner_id)

    semantic_model = semantic.model_name(settings)
    if semantic_model != "hash":
        try:
            rows = store.get_embeddings(db, owner_type, semantic_model, limit=cap)
            if rows:
                vector = semantic.embed_text(settings, query)
                if vector is not None and vector.size:
                    pairs = [(owner_id, np.asarray(vec, dtype=np.float32)) for owner_id, vec in rows]
                    pairs = [(owner_id, vec) for owner_id, vec in pairs if vec.size == vector.size]
                    add(_rank_pairs(vector, pairs, threshold, limit))
        except Exception:
            pass

    try:
        rows = store.get_embeddings(db, owner_type, "hash", limit=cap)
        add(_vector_rank(query, rows, settings.embed_dim, threshold, limit))
    except Exception:
        pass
    return out[:limit]


def _rrf(lists: list[list[str]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in lists:
        for rank, key in enumerate(ranked):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


DOCUMENT_WORDS = {
    "pdf", "document", "documents", "page", "pages", "paper", "report",
    "attachment", "attached", "file", "chapter", "section", "figure", "table",
}


def _tree_items(
    db: Database,
    query: str,
    *,
    conversation_id: str | None,
    document_ids: list[str] | None,
    limit: int = MAX_TREE_ITEMS,
) -> tuple[list[MemoryItem], list[list[str]]]:
    """Back-trace chat and opted-in document trees.

    Returns the items plus ``[[tree_id, node_id], ...]`` usage pairs so the
    gateway can strengthen branches after the reply is actually delivered.
    """
    searches: list[tuple[str, str]] = []
    if conversation_id:
        chat = tree.tree_by_ref(db, kind="chat", ref_id=conversation_id)
        if chat:
            searches.append(("chat", str(chat["id"])))
    for document_id in (document_ids or [])[:6]:
        if not document_id:
            continue
        doc_tree = tree.tree_by_ref(db, kind="document", ref_id=str(document_id))
        if doc_tree:
            searches.append(("document", str(doc_tree["id"])))

    items: list[MemoryItem] = []
    usage: list[list[str]] = []
    for kind, tree_id in searches:
        try:
            hits = tree.backtrace(db, tree_id, query, limit=limit)
        except Exception:
            continue
        for hit in hits:
            if hit.kind == "chunk":
                if kind == "document" and hit.page_from:
                    title = f"Document page {hit.page_from}"
                    summary_kind = "tree_page"
                else:
                    title = "Exact exchange"
                    summary_kind = "tree_chunk"
                item_id = f"tree-chunk:{tree_id}:{hit.id}"
            else:
                title = f"Memory summary, degree {hit.degree}"
                summary_kind = "tree_node"
                item_id = f"tree-node:{hit.id}"
                usage.append([tree_id, hit.id])
            items.append(
                MemoryItem(
                    kind="summary",
                    id=item_id,
                    title=title,
                    body=hit.body,
                    score=hit.score,
                    created_at=0,
                    salience=0.6,
                    tokens=hit.tokens or estimate_tokens(hit.body),
                    summary_kind=summary_kind,
                )
            )

    items.sort(key=lambda i: i.score, reverse=True)
    # Keep at most `limit` items overall but never let one tree crowd out the
    # other: they answer different questions.
    return items[: max(1, limit)], usage


def _norm_body(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()[:240]


def _dedupe(items: list[MemoryItem]) -> list[MemoryItem]:
    """Drop items whose text is already present, keeping the higher score.

    Tree chunks duplicate the archived messages; injecting both wastes budget and
    makes the model repeat itself.
    """
    best: dict[str, MemoryItem] = {}
    for item in items:
        key = _norm_body(item.body)
        if not key:
            continue
        current = best.get(key)
        if current is None or item.score > current.score:
            best[key] = item
    return sorted(best.values(), key=lambda i: i.score, reverse=True)


def _document_items(
    db: Database,
    query: str,
    settings: Settings,
    *,
    document_ids: list[str] | None = None,
    section_limit: int = 2,
    page_limit: int = 4,
) -> list[MemoryItem]:
    """Retrieve section summaries and exact pages from uploaded documents.

    The document summary is only an index now; the facts live in these rows. Page
    text is returned verbatim so the model can quote it instead of guessing from an
    overview. ``document_ids`` scopes the search to documents this chat opted into;
    ``None`` means unrestricted (Memory Inspector only).
    """
    allowed = None if document_ids is None else {str(d) for d in document_ids}
    if allowed is not None and not allowed:
        return []

    def permitted(doc_id: str) -> bool:
        return allowed is None or doc_id in allowed

    section_ids = [
        key for key in _vector_lists(
            db, "document_section", query, settings,
            threshold=0.10, limit=60, cap=40_000,
        )
        if permitted(key.split(":", 1)[0])
    ]
    page_ids = [
        key for key in _vector_lists(
            db, "document_page", query, settings,
            threshold=0.14, limit=60, cap=80_000,
        )
        if permitted(key.split(":", 1)[0])
    ]
    items: list[MemoryItem] = []
    seen_sections: set[str] = set()
    seen_pages: set[str] = set()
    with db.read() as c:
        for rank, key in enumerate(section_ids):
            if len(seen_sections) >= section_limit:
                break
            parts = key.split(":")
            if len(parts) != 3:
                continue
            doc_id, level, ordinal = parts[0], parts[1], parts[2]
            if key in seen_sections:
                continue
            row = c.execute(
                """SELECT s.page_from, s.page_to, s.summary, s.created_at AS section_at,
                          d.title, d.created_at AS doc_at
                   FROM document_sections s JOIN summaries d ON d.id=s.document_id
                   WHERE s.document_id=? AND s.level=? AND s.ordinal=?""",
                (doc_id, int(level) if level.isdigit() else 1, int(ordinal) if ordinal.isdigit() else 0),
            ).fetchone()
            if not row:
                continue
            seen_sections.add(key)
            title = f"{row['title']} — pages {row['page_from']}-{row['page_to']}"
            body = row["summary"]
            items.append(
                MemoryItem(
                    kind="document_section",
                    id=key,
                    title=title,
                    body=body,
                    score=1.0 / (RRF_K + rank + 1),
                    created_at=int(row["section_at"] or row["doc_at"] or 0),
                    salience=0.62,
                    tokens=estimate_tokens(body) + estimate_tokens(title),
                    summary_kind="document_section",
                )
            )
        for rank, key in enumerate(page_ids):
            if len(seen_pages) >= page_limit:
                break
            parts = key.split(":")
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            if key in seen_pages:
                continue
            row = c.execute(
                """SELECT p.page_no, p.text, p.created_at AS page_at, d.title
                   FROM document_pages p JOIN summaries d ON d.id=p.document_id
                   WHERE p.document_id=? AND p.page_no=?""",
                (parts[0], int(parts[1])),
            ).fetchone()
            if not row:
                continue
            seen_pages.add(key)
            title = f"{row['title']} — page {row['page_no']}"
            body = str(row["text"])[:6000]
            items.append(
                MemoryItem(
                    kind="document_page",
                    id=key,
                    title=title,
                    body=body,
                    score=1.0 / (RRF_K + rank + 1),
                    created_at=int(row["page_at"] or 0),
                    salience=0.66,
                    tokens=estimate_tokens(body) + estimate_tokens(title),
                    summary_kind="document_page",
                )
            )
    return items


def _group_exact(db: Database, message_ids: list[str], fused: dict[str, float]) -> list[MemoryItem]:
    rows = store.messages_by_ids(db, message_ids)
    if not rows:
        return []
    rows.sort(key=lambda r: r["created_at"])
    groups: list[list[dict]] = []
    for row in rows:
        if (
            groups
            and groups[-1][-1]["conversation_id"] == row["conversation_id"]
            and row["created_at"] - groups[-1][-1]["created_at"] < 30 * 60 * 1000
        ):
            groups[-1].append(row)
        else:
            groups.append([row])

    items: list[MemoryItem] = []
    for group in groups:
        body = "\n".join(f"{r['role']}: {r['content']}" for r in group)
        best = max(fused.get(r["id"], 0.0) for r in group)
        first, last = group[0], group[-1]
        date = time.strftime("%Y-%m-%d %H:%M", time.localtime(first["created_at"] / 1000))
        items.append(
            MemoryItem(
                kind="exact",
                id=first["conversation_id"],
                title=f"Exact excerpt — {date}",
                body=body,
                score=best,
                message_ids=[r["id"] for r in group],
                created_at=first["created_at"],
                salience=0.5,
                tokens=estimate_tokens(body),
            )
        )
    items.sort(key=lambda i: i.score, reverse=True)
    return items[:MAX_EXACT_MESSAGES]


def retrieve(
    db: Database,
    query: str,
    *,
    settings: Settings,
    conversation_id: str | None = None,
    exclude_message_ids: set[str] | None = None,
    request_id: str | None = None,
    force_lookup: bool = False,
    document_ids: list[str] | None = None,
) -> RetrievalResult:
    t0 = time.perf_counter()
    q = normalise(query)
    exclude = exclude_message_ids or set()
    # The Memory Inspector search box always searches. Intent classification only
    # exists to decide whether a normal chat turn needs retrieval at all.
    intent = "lookup" if force_lookup else classify_intent(q)
    plan: dict = {
        "intent": intent,
        "query": q[:400],
        "sources": {},
        "gate": None,
        "selected": [],
        "latency_ms": 0,
    }

    if not q or intent == "continuation":
        plan["skipped"] = "continuation"
        latency = int((time.perf_counter() - t0) * 1000)
        plan["latency_ms"] = latency
        if request_id:
            try:
                store.log_retrieval(
                    db,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    query=q,
                    plan=plan,
                    latency_ms=latency,
                )
            except Exception:  # pragma: no cover
                pass
        return RetrievalResult(intent=intent, items=[], plan=plan, latency_ms=latency)

    summaries: dict[str, dict] = {}
    lists: list[list[str]] = []

    try:
        vec_ids = _vector_summaries(db, q, settings)
        lists.append(vec_ids)
        plan["sources"]["summary_vector"] = vec_ids[:10]
    except Exception as exc:  # pragma: no cover - defensive
        plan["sources"]["summary_vector_error"] = str(exc)[:200]
        vec_ids = []

    try:
        fts_sum = [sid for sid, _ in store.search_summaries_fts(db, q, limit=40)]
        lists.append(fts_sum)
        plan["sources"]["summary_fts"] = fts_sum[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["summary_fts_error"] = str(exc)[:200]
        fts_sum = []

    try:
        fts_msg = [mid for mid, _ in store.search_messages_fts(db, q, limit=40)]
        fts_msg = [m for m in fts_msg if m not in exclude]
        lists.append(fts_msg)
        plan["sources"]["message_fts"] = fts_msg[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["message_fts_error"] = str(exc)[:200]
        fts_msg = []

    try:
        topic_ids = _topics_for_slugs(db, _topic_hits(db, q))
        lists.append(topic_ids)
        plan["sources"]["topic"] = topic_ids[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["topic_error"] = str(exc)[:200]
        topic_ids = []

    try:
        vec_msg = [m for m in _vector_messages(db, q, settings) if m not in exclude]
        lists.append(vec_msg)
        plan["sources"]["message_vector"] = vec_msg[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["message_vector_error"] = str(exc)[:200]
        vec_msg = []

    fused_summary = _rrf(lists[:2] + [topic_ids])
    fused_message = _rrf([fts_msg, vec_msg])

    summary_rows = {s["id"]: s for s in store.summaries_by_ids(db, list(fused_summary))}
    max_possible = 3.0 / (RRF_K + 1)
    top_raw = max(fused_summary.values(), default=0.0)
    plan["gate"] = round(top_raw / max_possible, 4) if max_possible else 0.0

    ordered: list[MemoryItem] = []
    allowed_docs = None if document_ids is None else {str(d) for d in document_ids}
    for sid, rrf in sorted(fused_summary.items(), key=lambda kv: kv[1], reverse=True):
        row = summary_rows.get(sid)
        if not row:
            continue
        # Document memory is opt-in per chat: a document summary is never
        # injected unless the user attached that document to this conversation.
        if (
            str(row.get("kind") or "") == "document"
            and allowed_docs is not None
            and sid not in allowed_docs
        ):
            continue
        salience = float(row.get("salience") or 0.5)
        score = rrf * (0.75 + 0.25 * salience) * _recency_boost(int(row.get("created_at") or 0))
        body = row.get("body") or ""
        ordered.append(
            MemoryItem(
                kind="summary",
                id=sid,
                title=row.get("title") or "Summary",
                body=body,
                score=score,
                topic=row.get("topic_slug"),
                created_at=int(row.get("created_at") or 0),
                salience=salience,
                tokens=estimate_tokens(body) + estimate_tokens(row.get("title") or ""),
                summary_kind=str(row.get("kind") or ""),
            )
        )
    ordered = ordered[:MAX_SUMMARIES]

    if top_raw / max_possible < GATE_NORMALISED:
        plan["gate"] = "below_threshold"
        ordered = []

    # Documents are stored as pages + sections. Pull exact excerpts when the query
    # is document-shaped, or whenever a document summary was already retrieved.
    # Explicit page references ("pages 3-5", "page 12") resolve directly and
    # bypass the similarity thresholds, so page-number queries are exact.
    doc_items: list[MemoryItem] = []
    try:
        words = set(re.findall(r"[a-z]+", q.lower()))
        has_doc = any(i.summary_kind == "document" for i in ordered)
        page_refs = parse_page_refs(q)
        # An explicit page reference is a direct lookup, not a similarity
        # search: it must resolve even when no summary matched the wording.
        if page_refs:
            doc_ids = [
                i.id for i in ordered
                if i.summary_kind == "document" or str(i.id).startswith("doc:")
            ]
            if not doc_ids:
                # Fall back to the most recent *attached* document when the
                # query names pages but no document summary matched the text.
                candidates = allowed_docs
                if candidates is None:
                    candidates = {
                        str(r["id"])
                        for r in store.list_summaries(db, limit=50)
                        if str(r.get("kind") or "") == "document"
                    }
                doc_ids = list(candidates)[:3]
            seen_pages: set[str] = set()
            for doc_id in doc_ids[:3]:
                if allowed_docs is not None and doc_id not in allowed_docs:
                    continue
                for low, high in page_refs:
                    for item in document_pages_range(db, doc_id, low, high):
                        if item.id not in seen_pages:
                            seen_pages.add(item.id)
                            doc_items.append(item)
            if doc_items:
                plan["sources"]["document_pages_explicit"] = [i.id for i in doc_items]
        if (not doc_items) and (has_doc or (words & DOCUMENT_WORDS)):
            doc_items = _document_items(db, q, settings, document_ids=document_ids)
            if doc_items:
                plan["sources"]["document"] = [i.id for i in doc_items]
    except Exception as exc:  # pragma: no cover - defensive
        plan["sources"]["document_error"] = str(exc)[:200]

    # Append-only tree memory: back-trace from the query through the summary
    # levels down to exact chunks. This is the primary long-term path; the flat
    # ranked lists above remain as a fallback for summaries and direct hits.
    tree_items: list[MemoryItem] = []
    try:
        tree_items, tree_usage = _tree_items(
            db,
            q,
            conversation_id=conversation_id,
            document_ids=document_ids,
            limit=MAX_TREE_ITEMS,
        )
        if tree_items:
            plan["sources"]["tree"] = [i.id for i in tree_items]
        if tree_usage:
            plan["tree_usage"] = tree_usage
    except Exception as exc:  # pragma: no cover - defensive
        plan["sources"]["tree_error"] = str(exc)[:200]

    # Expand the messages behind the selected summaries, plus direct message hits.
    expand_ids: list[str] = []
    for item in ordered:
        for mid in store.summary_message_ids(db, item.id, limit=EXPAND_PER_SUMMARY):
            if mid not in exclude and mid not in expand_ids:
                expand_ids.append(mid)
    for mid in list(fused_message):
        if mid not in exclude and mid not in expand_ids:
            expand_ids.append(mid)

    exact_items: list[MemoryItem] = []
    if expand_ids:
        try:
            exact_items = _group_exact(db, expand_ids[:MAX_EXACT_MESSAGES * 4], fused_message)
        except Exception as exc:  # pragma: no cover
            plan["exact_error"] = str(exc)[:200]

    items = _dedupe(tree_items + doc_items + exact_items + ordered)
    plan["selected"] = [i.to_dict() for i in items]
    latency = int((time.perf_counter() - t0) * 1000)
    plan["latency_ms"] = latency

    if request_id:
        try:
            store.log_retrieval(
                db,
                request_id=request_id,
                conversation_id=conversation_id,
                query=q,
                plan=plan,
                latency_ms=latency,
            )
        except Exception:  # pragma: no cover - logging must never break a reply
            pass

    return RetrievalResult(intent=intent, items=items, plan=plan, latency_ms=latency)
