"""Append-only tree memory with bounded back-trace retrieval.

The tree is a Merkle-like roll-up of immutable units:

    chunk, chunk, chunk            -> degree-1 node
    node1, node2, node3 (degree 1) -> degree-2 node
    ...                            -> degree-N root

Nothing is rewritten. A unit is *consumed* when it becomes a child
(``parent_id`` set); consumed units remain readable forever, so a query can
always descend from a high-level summary back to the exact chunk. The only
destructive path is an explicit user delete, which removes a whole tree and
leaves a tombstone so the startup backfill cannot recreate it.

Retrieval is a beam search, not a flat top-k. It scores every node at the
current frontier, keeps the *K* best branches, descends into their children,
and scores again. Several paths are compared at every level, and the result
carries the full path plus leaf provenance so the caller can show why an item
was selected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable

from .text import cosine, embed, estimate_tokens, keywords, new_id

from ..db import Database, now_ms

FANOUT = 3
MAX_BEAM = 3
MAX_DEPTH = 32


# --------------------------------------------------------------------- models
@dataclass(frozen=True)
class Chunk:
    tree_id: str
    chunk_key: str
    ordinal: int
    payload: dict
    tokens: int
    complete: bool
    parent_id: str | None
    created_at: int

    @property
    def text(self) -> str:
        return chunk_text(self.payload)


@dataclass(frozen=True)
class Node:
    id: str
    tree_id: str
    parent_id: str | None
    degree: int
    ordinal: int
    summary: str
    unit_keys: tuple[str, ...]
    page_from: int | None
    page_to: int | None
    tokens: int
    state: str
    active: bool
    created_at: int


@dataclass(frozen=True)
class TreeHit:
    """One retrieval result with its provenance path."""

    kind: str  # "node" | "chunk"
    id: str
    body: str
    score: float
    degree: int
    path: tuple[str, ...]
    page_from: int | None = None
    page_to: int | None = None
    tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "id": self.id,
            "body": self.body,
            "score": round(self.score, 5),
            "degree": self.degree,
            "path": list(self.path),
            "page_from": self.page_from,
            "page_to": self.page_to,
            "tokens": self.tokens,
        }


# ------------------------------------------------------------------ utilities
def chunk_text(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return ""
    if payload.get("text"):
        return str(payload["text"])
    parts = [payload.get("user"), payload.get("assistant")]
    return "\n\n".join(str(p) for p in parts if p)


def _as_chunk(row) -> Chunk:
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    return Chunk(
        tree_id=str(row["tree_id"]),
        chunk_key=str(row["chunk_key"]),
        ordinal=int(row["ordinal"]),
        payload=payload if isinstance(payload, dict) else {},
        tokens=int(row["tokens"] or 0),
        complete=bool(row["complete"]),
        parent_id=row["parent_id"],
        created_at=int(row["created_at"] or 0),
    )


def _as_node(row) -> Node:
    try:
        keys = json.loads(row["unit_keys"] or "[]")
    except (TypeError, ValueError):
        keys = []
    return Node(
        id=str(row["id"]),
        tree_id=str(row["tree_id"]),
        parent_id=row["parent_id"],
        degree=int(row["degree"] or 1),
        ordinal=int(row["ordinal"] or 0),
        summary=str(row["summary"] or ""),
        unit_keys=tuple(str(k) for k in keys) if isinstance(keys, list) else (),
        page_from=row["page_from"],
        page_to=row["page_to"],
        tokens=int(row["tokens"] or 0),
        state=str(row["state"] or "complete"),
        active=bool(row["active"]),
        created_at=int(row["created_at"] or 0),
    )


# ---------------------------------------------------------------------- trees
def create_tree(
    db: Database, *, kind: str = "chat", title: str = "", ref_id: str = ""
) -> str:
    tree_id = new_id()
    now = now_ms()
    with db.write() as c:
        c.execute(
            """INSERT INTO memory_trees(id, kind, title, ref_id, node_count, leaf_count,
                   created_at, updated_at)
               VALUES(?,?,?,?,0,0,?,?)""",
            (tree_id, kind, title[:200], ref_id or None, now, now),
        )
    return tree_id


def get_tree(db: Database, tree_id: str) -> dict | None:
    if not tree_id:
        return None
    with db.read() as c:
        row = c.execute("SELECT * FROM memory_trees WHERE id=?", (tree_id,)).fetchone()
    return dict(row) if row else None


def tree_by_ref(db: Database, *, kind: str, ref_id: str) -> dict | None:
    """Return the tree attached to a conversation/document, if it exists."""
    if not ref_id:
        return None
    with db.read() as c:
        row = c.execute(
            "SELECT * FROM memory_trees WHERE kind=? AND ref_id=? ORDER BY created_at LIMIT 1",
            (kind, ref_id),
        ).fetchone()
    return dict(row) if row else None


def get_or_create_tree(
    db: Database, *, kind: str, ref_id: str, title: str = ""
) -> str:
    existing = tree_by_ref(db, kind=kind, ref_id=ref_id)
    if existing:
        return str(existing["id"])
    return create_tree(db, kind=kind, title=title, ref_id=ref_id)


def list_trees(db: Database, *, kind: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM memory_trees"
    args: list[object] = []
    if kind:
        sql += " WHERE kind=?"
        args.append(kind)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    args.append(max(1, min(int(limit), 200)))
    with db.read() as c:
        rows = c.execute(sql, tuple(args)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ deletion
def delete_tree_rows(c, tree_id: str) -> None:
    """Delete one tree's rows on an existing connection.

    Kept separate so a conversation/document delete can cascade inside its own
    transaction without opening a nested write.
    """
    c.execute("DELETE FROM tree_node_usage WHERE tree_id=?", (tree_id,))
    c.execute("DELETE FROM tree_chunks WHERE tree_id=?", (tree_id,))
    c.execute("DELETE FROM tree_nodes WHERE tree_id=?", (tree_id,))
    c.execute("DELETE FROM memory_trees WHERE id=?", (tree_id,))


def _tombstone(c, kind: str, ref_id: str, deleted_at: int) -> None:
    c.execute(
        """INSERT INTO memory_tree_tombstones(kind, ref_id, deleted_at)
           VALUES(?,?,?)
           ON CONFLICT(kind, ref_id) DO UPDATE SET deleted_at=excluded.deleted_at""",
        (kind, ref_id, deleted_at),
    )


def is_tombstoned(db: Database, *, kind: str, ref_id: str) -> bool:
    """True when the user deleted this ref's tree, so backfill must skip it."""
    if not ref_id:
        return False
    with db.read() as c:
        row = c.execute(
            "SELECT 1 FROM memory_tree_tombstones WHERE kind=? AND ref_id=?",
            (kind, ref_id),
        ).fetchone()
    return row is not None


def clear_tombstone(c, *, kind: str, ref_id: str) -> None:
    """Drop a tombstone when its conversation/document no longer exists."""
    c.execute(
        "DELETE FROM memory_tree_tombstones WHERE kind=? AND ref_id=?",
        (kind, ref_id),
    )


def delete_tree(db: Database, tree_id: str) -> dict | None:
    """Delete one tree and everything under it.

    This is the only destructive path in tree memory and therefore only runs on
    an explicit user action. The ref is tombstoned so the startup backfill does
    not silently recreate what was just deleted.
    """
    with db.write() as c:
        row = c.execute(
            "SELECT kind, ref_id FROM memory_trees WHERE id=?", (tree_id,)
        ).fetchone()
        if not row:
            return None
        kind = str(row["kind"] or "chat")
        ref_id = str(row["ref_id"] or "")
        nodes = c.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE tree_id=?", (tree_id,)
        ).fetchone()[0]
        chunks = c.execute(
            "SELECT COUNT(*) FROM tree_chunks WHERE tree_id=?", (tree_id,)
        ).fetchone()[0]
        delete_tree_rows(c, tree_id)
        if ref_id:
            _tombstone(c, kind, ref_id, now_ms())
    return {
        "deleted": tree_id,
        "kind": kind,
        "ref_id": ref_id,
        "nodes": int(nodes or 0),
        "chunks": int(chunks or 0),
    }


def delete_trees(db: Database, *, kind: str | None = None) -> dict:
    """Delete every tree of a kind (or all trees).

    Every conversation/document that could be re-imported is tombstoned, not
    just the refs that currently have a tree, so "delete all" stays deleted.
    """
    if kind not in (None, "chat", "document"):
        raise ValueError("kind must be chat, document or empty")
    deleted = 0
    nodes = 0
    chunks = 0
    with db.write() as c:
        sql = "SELECT id, kind, ref_id FROM memory_trees"
        args: tuple[object, ...] = ()
        if kind:
            sql += " WHERE kind=?"
            args = (kind,)
        rows = c.execute(sql, args).fetchall()
        refs: set[tuple[str, str]] = set()
        for row in rows:
            ref = str(row["ref_id"] or "")
            row_kind = str(row["kind"] or "chat")
            if ref:
                refs.add((row_kind, ref))
            nodes += int(
                c.execute(
                    "SELECT COUNT(*) FROM tree_nodes WHERE tree_id=?", (row["id"],)
                ).fetchone()[0]
                or 0
            )
            chunks += int(
                c.execute(
                    "SELECT COUNT(*) FROM tree_chunks WHERE tree_id=?", (row["id"],)
                ).fetchone()[0]
                or 0
            )
            delete_tree_rows(c, str(row["id"]))
            deleted += 1
        # Also tombstone refs that have no tree yet, so the next startup cannot
        # import them into a brand-new tree.
        if kind in (None, "chat"):
            for row in c.execute("SELECT id FROM conversations").fetchall():
                refs.add(("chat", str(row["id"])))
        if kind in (None, "document"):
            for row in c.execute(
                "SELECT id FROM summaries WHERE kind='document' AND merged_into IS NULL"
            ).fetchall():
                refs.add(("document", str(row["id"])))
        now = now_ms()
        for row_kind, ref in refs:
            _tombstone(c, row_kind, ref, now)
    return {
        "deleted": deleted,
        "kind": kind or "all",
        "nodes": nodes,
        "chunks": chunks,
    }


# --------------------------------------------------------------------- chunks
def append_chunk(
    db: Database,
    tree_id: str,
    chunk_key: str,
    payload: dict,
    *,
    tokens: int | None = None,
    complete: bool = True,
) -> Chunk | None:
    """Add one immutable unit. Re-appending the same key is a no-op update.

    The ordinal is assigned once, so a resumed job that re-archives the same
    exchange cannot reorder the tree.
    """
    if not tree_id or not chunk_key:
        return None
    text = chunk_text(payload)
    if tokens is None:
        tokens = estimate_tokens(text)
    now = now_ms()
    with db.write() as c:
        existing = c.execute(
            "SELECT ordinal FROM tree_chunks WHERE tree_id=? AND chunk_key=?",
            (tree_id, chunk_key),
        ).fetchone()
        if existing is not None:
            c.execute(
                """UPDATE tree_chunks SET payload=?, tokens=?, complete=?
                   WHERE tree_id=? AND chunk_key=?""",
                (
                    json.dumps(payload, ensure_ascii=False),
                    int(tokens),
                    1 if complete else 0,
                    tree_id,
                    chunk_key,
                ),
            )
            ordinal = int(existing["ordinal"])
        else:
            row = c.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM tree_chunks WHERE tree_id=?",
                (tree_id,),
            ).fetchone()
            ordinal = int(row[0] if row else 0)
            c.execute(
                """INSERT INTO tree_chunks(tree_id, chunk_key, ordinal, payload, tokens,
                       complete, parent_id, created_at)
                   VALUES(?,?,?,?,?,?,NULL,?)""",
                (
                    tree_id,
                    chunk_key,
                    ordinal,
                    json.dumps(payload, ensure_ascii=False),
                    int(tokens),
                    1 if complete else 0,
                    now,
                ),
            )
        c.execute(
            """UPDATE memory_trees SET
                   leaf_count=(SELECT COUNT(*) FROM tree_chunks WHERE tree_id=?),
                   updated_at=? WHERE id=?""",
            (tree_id, now, tree_id),
        )
        row = c.execute(
            "SELECT * FROM tree_chunks WHERE tree_id=? AND chunk_key=?",
            (tree_id, chunk_key),
        ).fetchone()
    return _as_chunk(row) if row else None


def mark_chunk_complete(db: Database, tree_id: str, chunk_key: str) -> bool:
    with db.write() as c:
        cur = c.execute(
            "UPDATE tree_chunks SET complete=1 WHERE tree_id=? AND chunk_key=?",
            (tree_id, chunk_key),
        )
        c.execute("UPDATE memory_trees SET updated_at=? WHERE id=?", (now_ms(), tree_id))
    return cur.rowcount > 0


def pending_chunks(db: Database, tree_id: str, limit: int = 300) -> list[Chunk]:
    with db.read() as c:
        rows = c.execute(
            """SELECT * FROM tree_chunks WHERE tree_id=? AND complete=1
               ORDER BY ordinal LIMIT ?""",
            (tree_id, max(1, int(limit))),
        ).fetchall()
    return [_as_chunk(r) for r in rows]


# -------------------------------------------------------------------- roll-up
def _next_group(
    db: Database, tree_id: str
) -> tuple[int, list[tuple[str, str, int | None, int | None]]] | None:
    """Find the lowest degree with FANOUT unconsumed complete children.

    Returns ``(degree, [(key, text, page_from, page_to), ...])`` or ``None``.
    Either three chunks (degree 0) or three nodes of the same degree.
    """
    with db.read() as c:
        rows = c.execute(
            """SELECT chunk_key, payload
               FROM tree_chunks
               WHERE tree_id=? AND parent_id IS NULL AND complete=1
               ORDER BY ordinal LIMIT ?""",
            (tree_id, FANOUT),
        ).fetchall()
        if len(rows) >= FANOUT:
            children = []
            for r in rows:
                try:
                    payload = json.loads(r["payload"] or "{}")
                except (TypeError, ValueError):
                    payload = {}
                page_no = payload.get("page_no") if isinstance(payload, dict) else None
                children.append((str(r["chunk_key"]), chunk_text(payload), page_no, page_no))
            return 0, children

        degrees = [
            int(r[0])
            for r in c.execute(
                """SELECT DISTINCT degree FROM tree_nodes
                   WHERE tree_id=? AND parent_id IS NULL AND state='complete' AND active=1
                   ORDER BY degree""",
                (tree_id,),
            ).fetchall()
        ]
        for degree in degrees:
            rows = c.execute(
                """SELECT id, summary, page_from, page_to FROM tree_nodes
                   WHERE tree_id=? AND degree=? AND parent_id IS NULL
                     AND state='complete' AND active=1
                   ORDER BY ordinal LIMIT ?""",
                (tree_id, degree, FANOUT),
            ).fetchall()
            if len(rows) >= FANOUT:
                return degree, [
                    (str(r["id"]), str(r["summary"]), r["page_from"], r["page_to"])
                    for r in rows
                ]
    return None


def rollup(
    db: Database,
    tree_id: str,
    summarise: Callable[[int, list[str]], str],
    *,
    max_nodes: int = 8,
) -> list[str]:
    """Create parents for every complete group of FANOUT children.

    ``summarise(degree, texts)`` returns the parent summary. A degree of 0
    means the children are raw chunks; 1 means degree-1 nodes, and so on.
    An empty return stops the roll-up so a failed model call never creates a
    node without provenance.
    """
    created: list[str] = []
    for _ in range(max(0, int(max_nodes))):
        group = _next_group(db, tree_id)
        if not group:
            break
        degree, children = group
        keys = [k for k, _, _, _ in children]
        texts = [t for _, t, _, _ in children]
        summary = summarise(degree, texts)
        if not summary or not str(summary).strip():
            break
        summary = str(summary).strip()
        pages = [p for _, _, p, _ in children if p is not None]
        page_from = min(pages) if pages else None
        page_to = max([p for _, _, _, p in children if p is not None], default=None)
        node_id = new_id()
        now = now_ms()
        with db.write() as c:
            # Two callers can summarise the same group (a manual roll-up and the
            # background drain). Only the first may consume it: re-check that
            # every child is still free inside the write transaction, otherwise
            # the second would create a duplicate parent for the same units.
            if degree == 0:
                marks = ",".join("?" * len(keys))
                free = c.execute(
                    f"""SELECT COUNT(*) FROM tree_chunks
                        WHERE tree_id=? AND parent_id IS NULL AND complete=1
                          AND chunk_key IN ({marks})""",
                    (tree_id, *keys),
                ).fetchone()[0]
            else:
                marks = ",".join("?" * len(keys))
                free = c.execute(
                    f"""SELECT COUNT(*) FROM tree_nodes
                        WHERE tree_id=? AND parent_id IS NULL AND state='complete'
                          AND active=1 AND id IN ({marks})""",
                    (tree_id, *keys),
                ).fetchone()[0]
            if int(free or 0) != len(keys):
                break
            row = c.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM tree_nodes "
                "WHERE tree_id=? AND degree=?",
                (tree_id, degree + 1),
            ).fetchone()
            ordinal = int(row[0] if row else 0)
            c.execute(
                """INSERT INTO tree_nodes(id, tree_id, parent_id, degree, ordinal,
                       summary, unit_keys, page_from, page_to, tokens, state, active,
                       source_hash, created_at)
                   VALUES(?,?,NULL,?,?,?,?,?,?,?,'complete',1,NULL,?)""",
                (
                    node_id,
                    tree_id,
                    degree + 1,
                    ordinal,
                    summary,
                    json.dumps(keys, ensure_ascii=False),
                    page_from,
                    page_to,
                    estimate_tokens(summary),
                    now,
                ),
            )
            if degree == 0:
                for key in keys:
                    c.execute(
                        """UPDATE tree_chunks SET parent_id=?
                           WHERE tree_id=? AND chunk_key=?""",
                        (node_id, tree_id, key),
                    )
            else:
                for key in keys:
                    c.execute("UPDATE tree_nodes SET parent_id=? WHERE id=?", (node_id, key))
            c.execute(
                """UPDATE memory_trees SET
                       node_count=(SELECT COUNT(*) FROM tree_nodes WHERE tree_id=?),
                       updated_at=? WHERE id=?""",
                (tree_id, now, tree_id),
            )
            # Root = highest-degree active node with no parent.
            root = c.execute(
                """SELECT id FROM tree_nodes
                   WHERE tree_id=? AND parent_id IS NULL AND state='complete' AND active=1
                   ORDER BY degree DESC, ordinal DESC LIMIT 1""",
                (tree_id,),
            ).fetchone()
            c.execute(
                "UPDATE memory_trees SET root_id=? WHERE id=?",
                (root["id"] if root else None, tree_id),
            )
        created.append(node_id)
    return created


# ------------------------------------------------------------------ back-trace
def _load_nodes(db: Database, tree_id: str) -> dict[str, Node]:
    with db.read() as c:
        rows = c.execute(
            "SELECT * FROM tree_nodes WHERE tree_id=? AND active=1", (tree_id,)
        ).fetchall()
    return {str(r["id"]): _as_node(r) for r in rows}


def _load_chunks(db: Database, tree_id: str) -> dict[str, Chunk]:
    with db.read() as c:
        rows = c.execute(
            "SELECT * FROM tree_chunks WHERE tree_id=? AND complete=1", (tree_id,)
        ).fetchall()
    return {str(r["chunk_key"]): _as_chunk(r) for r in rows}


def _load_usage(db: Database, tree_id: str) -> dict[str, float]:
    with db.read() as c:
        rows = c.execute(
            "SELECT node_id, relevance FROM tree_node_usage WHERE tree_id=?", (tree_id,)
        ).fetchall()
    return {str(r["node_id"]): float(r["relevance"] or 0.0) for r in rows}


def _score(query_vec, query_words: set[str], text: str) -> float:
    if not text:
        return 0.0
    vec = embed(text, int(query_vec.size))
    cos = max(0.0, cosine(query_vec, vec))
    if query_words:
        overlap = len(query_words & set(keywords(text, 40))) / float(len(query_words))
    else:
        overlap = 0.0
    return 0.6 * cos + 0.4 * overlap


def backtrace(
    db: Database,
    tree_id: str,
    query: str,
    *,
    branches: int = MAX_BEAM,
    limit: int = 8,
    include_chunks: bool = True,
) -> list[TreeHit]:
    """Score the tree from the root down, comparing several branches per level.

    The beam is deliberate: a high-level summary can be a poor lexical match
    while the exact chunk underneath answers the query precisely. Keeping more
    than one branch means the leaf is not discarded by one unlucky parent.
    """
    text = str(query or "").strip()
    if not tree_id or not text:
        return []
    nodes = _load_nodes(db, tree_id)
    chunks = _load_chunks(db, tree_id) if include_chunks else {}
    if not nodes and not chunks:
        return []

    qvec = embed(text)
    qwords = set(keywords(text, 24))
    usage = _load_usage(db, tree_id)
    branches = max(1, min(int(branches), 8))
    limit = max(1, min(int(limit), 50))

    hits: list[TreeHit] = []
    seen: set[str] = set()

    def add_node(node: Node, score: float, path: tuple[str, ...]) -> None:
        if node.id in seen:
            return
        seen.add(node.id)
        hits.append(
            TreeHit(
                kind="node",
                id=node.id,
                body=node.summary,
                score=score,
                degree=node.degree,
                path=path,
                page_from=node.page_from,
                page_to=node.page_to,
                tokens=node.tokens,
            )
        )

    def add_chunk(chunk: Chunk, score: float, path: tuple[str, ...]) -> None:
        key = f"chunk:{chunk.chunk_key}"
        if key in seen or not chunk.text:
            return
        seen.add(key)
        hits.append(
            TreeHit(
                kind="chunk",
                id=chunk.chunk_key,
                body=chunk.text,
                score=score,
                degree=0,
                path=path,
                page_from=chunk.payload.get("page_no") if isinstance(chunk.payload, dict) else None,
                page_to=chunk.payload.get("page_no") if isinstance(chunk.payload, dict) else None,
                tokens=chunk.tokens,
            )
        )

    # Roots are the unconsumed top nodes. Leftover lower-degree roots are
    # included so a partially rolled-up tree is still searchable.
    frontier = [n for n in nodes.values() if n.parent_id is None]
    if not frontier:
        frontier = list(nodes.values())

    visited: set[str] = set()
    for _ in range(MAX_DEPTH):
        live = [n for n in frontier if n.id not in visited]
        if not live:
            break
        scored: list[tuple[float, Node]] = []
        for node in live:
            visited.add(node.id)
            score = _score(qvec, qwords, node.summary)
            score *= 1.0 + min(usage.get(node.id, 0.0), 0.5)
            scored.append((score, node))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        keep = scored[:branches]
        for score, node in keep:
            add_node(node, score, _path_of(node, nodes))
        next_frontier: list[Node] = []
        for score, node in keep:
            for key in node.unit_keys:
                child = nodes.get(key)
                if child is not None and child.active:
                    next_frontier.append(child)
                    continue
                chunk = chunks.get(key)
                if chunk is not None:
                    add_chunk(chunk, _score(qvec, qwords, chunk.text), _path_of(node, nodes) + (chunk.chunk_key,))
        frontier = next_frontier
        if not frontier:
            break

    # Unconsumed chunks are not under any node yet; score them directly so a
    # freshly written exchange is retrievable before the first roll-up.
    for chunk in chunks.values():
        if chunk.parent_id is None:
            add_chunk(chunk, _score(qvec, qwords, chunk.text), (chunk.chunk_key,))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def _path_of(node: Node, nodes: dict[str, Node]) -> tuple[str, ...]:
    path = [node.id]
    seen = {node.id}
    parent = nodes.get(node.parent_id) if node.parent_id else None
    while parent is not None and parent.id not in seen:
        path.append(parent.id)
        seen.add(parent.id)
        parent = nodes.get(parent.parent_id) if parent.parent_id else None
    return tuple(reversed(path))


def record_usage(db: Database, tree_id: str, node_ids: Iterable[str]) -> None:
    """Strengthen the relevance of branches that produced a used answer."""
    ids = [str(n) for n in node_ids if n]
    if not tree_id or not ids:
        return
    now = now_ms()
    with db.write() as c:
        for node_id in ids:
            c.execute(
                """INSERT INTO tree_node_usage(tree_id, node_id, hits, relevance, last_used)
                   VALUES(?,?,1,0.05,?)
                   ON CONFLICT(tree_id, node_id) DO UPDATE SET
                       hits=hits+1,
                       relevance=MIN(1.0, relevance+0.05),
                       last_used=excluded.last_used""",
                (tree_id, node_id, now),
            )


# ----------------------------------------------------------------------- views
def outline(db: Database, tree_id: str, *, max_chunks: int = 60) -> dict:
    """Return the tree as a nested structure for the Memory view."""
    tree = get_tree(db, tree_id)
    if not tree:
        return {"tree": None, "nodes": [], "chunks": []}
    nodes = _load_nodes(db, tree_id)
    chunks = _load_chunks(db, tree_id)
    usage = _load_usage(db, tree_id)

    def node_dict(node: Node) -> dict:
        return {
            "id": node.id,
            "degree": node.degree,
            "ordinal": node.ordinal,
            "parent": node.parent_id,
            "summary": node.summary,
            "tokens": node.tokens,
            "page_from": node.page_from,
            "page_to": node.page_to,
            "active": node.active,
            "relevance": round(usage.get(node.id, 0.0), 3),
            "children": list(node.unit_keys),
        }

    node_list = sorted(nodes.values(), key=lambda n: (n.degree, n.ordinal))
    chunk_list = sorted(chunks.values(), key=lambda ch: ch.ordinal)[:max(1, int(max_chunks))]
    return {
        "tree": tree,
        "nodes": [node_dict(n) for n in node_list],
        "chunks": [
            {
                "key": ch.chunk_key,
                "ordinal": ch.ordinal,
                "tokens": ch.tokens,
                "complete": ch.complete,
                "consumed": ch.parent_id is not None,
                "text": ch.text[:200],
            }
            for ch in chunk_list
        ],
    }


def stats(db: Database, tree_id: str) -> dict:
    with db.read() as c:
        node_row = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(MAX(degree),0) AS depth FROM tree_nodes WHERE tree_id=?",
            (tree_id,),
        ).fetchone()
        chunk_row = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(complete),0) AS done FROM tree_chunks WHERE tree_id=?",
            (tree_id,),
        ).fetchone()
        pending_row = c.execute(
            """SELECT COUNT(*) FROM tree_chunks
               WHERE tree_id=? AND parent_id IS NULL AND complete=1""",
            (tree_id,),
        ).fetchone()
    return {
        "nodes": int(node_row["n"] if node_row else 0),
        "depth": int(node_row["depth"] if node_row else 0),
        "chunks": int(chunk_row["n"] if chunk_row else 0),
        "chunks_complete": int(chunk_row["done"] if chunk_row else 0),
        "chunks_unrolled": int(pending_row[0] if pending_row else 0),
    }
