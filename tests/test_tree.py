from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aiolocal.config import Paths
from aiolocal.db import Database
from aiolocal.memory import store, tree
from aiolocal.memory.treebuild import TreeBuilder


def digest(degree: int, texts: list[str]) -> str:
    return f"degree-{degree}: " + " | ".join(texts)


def make_tree(db: Database, kind: str, ref_id: str = "", count: int = 3) -> str:
    tree_id = tree.create_tree(db, kind=kind, title="test", ref_id=ref_id)
    for index in range(count):
        tree.append_chunk(
            db, tree_id, f"c{index}", {"text": f"unit {index} about topic{index}"}, tokens=8
        )
    return tree_id


def test_rollup_still_works_after_adding_delete_helpers(tmp_path: Path):
    """The deletion work must not regress normal rollup behaviour."""
    db = Database(tmp_path / "tree.db")
    tree_id = make_tree(db, "chat", count=9)
    created = tree.rollup(db, tree_id, digest, max_nodes=10)
    assert len(created) == 4
    stats = tree.stats(db, tree_id)
    assert stats["nodes"] == 4
    assert stats["chunks"] == 9


def test_delete_tree_removes_rows_and_tombstones_the_ref(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    tree_id = make_tree(db, "chat", ref_id="conv-1", count=3)
    assert tree.stats(db, tree_id)["chunks"] == 3

    result = tree.delete_tree(db, tree_id)
    assert result is not None
    assert result["deleted"] == tree_id
    assert result["chunks"] == 3
    assert result["ref_id"] == "conv-1"

    # The tree and its rows are gone...
    assert tree.get_tree(db, tree_id) is None
    assert tree.stats(db, tree_id)["chunks"] == 0
    # ...and the ref is tombstoned so the startup backfill cannot recreate it.
    assert tree.is_tombstoned(db, kind="chat", ref_id="conv-1") is True


def test_delete_tree_returns_none_for_unknown_id(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    assert tree.delete_tree(db, "does-not-exist") is None


def test_delete_tree_leaves_other_trees_alone(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    keep = make_tree(db, "chat", ref_id="keep", count=2)
    drop = make_tree(db, "chat", ref_id="drop", count=2)

    tree.delete_tree(db, drop)

    assert tree.stats(db, keep)["chunks"] == 2
    assert tree.stats(db, drop)["chunks"] == 0
    assert tree.is_tombstoned(db, kind="chat", ref_id="drop") is True
    # A tree that was not deleted must not become tombstoned.
    assert tree.is_tombstoned(db, kind="chat", ref_id="keep") is False


def test_deleting_a_conversation_cascades_to_its_tree(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    conversation_id = "conv-cascade"
    store.ensure_conversation(db, conversation_id)
    tree_id = make_tree(db, "chat", ref_id=conversation_id, count=3)
    assert tree.tree_by_ref(db, kind="chat", ref_id=conversation_id) is not None

    assert store.delete_conversation(db, conversation_id) is True

    # The tree is part of the chat, so it must not survive the chat.
    assert tree.tree_by_ref(db, kind="chat", ref_id=conversation_id) is None
    assert tree.stats(db, tree_id)["chunks"] == 0
    # The tombstone is dropped with the ref, because the ref no longer exists.
    assert tree.is_tombstoned(db, kind="chat", ref_id=conversation_id) is False


def test_deleting_a_document_cascades_to_its_tree(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    document_id = store.insert_summary(
        db, title="Doc", body="# Doc", kind="document", source="document"
    )
    tree_id = make_tree(db, "document", ref_id=document_id, count=2)

    assert store.delete_summary(db, document_id) is True

    assert tree.tree_by_ref(db, kind="document", ref_id=document_id) is None
    assert tree.stats(db, tree_id)["chunks"] == 0
    assert tree.is_tombstoned(db, kind="document", ref_id=document_id) is False


def test_backfill_skips_a_tombstoned_ref(tmp_path: Path):
    """A deleted tree must stay deleted across a restart/backfill."""
    paths = Paths.create(tmp_path / "data")
    from aiolocal.config import Settings, SettingsStore

    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b"))
    db = Database(paths.db_path)

    conversation_id = "conv-tomb"
    store.ensure_conversation(db, conversation_id)
    store.archive_message(
        db, conversation_id=conversation_id, role="user", content="Remember this."
    )
    store.archive_message(
        db, conversation_id=conversation_id, role="assistant", content="Noted."
    )

    builder = TreeBuilder(db, settings_store)
    builder.kick = lambda tree_id: None
    # Backfill creates the tree the first time it sees this conversation. A
    # chunk is one user/assistant exchange, so both messages are required.
    assert builder.backfill() == 1
    first = tree.tree_by_ref(db, kind="chat", ref_id=conversation_id)
    assert first is not None

    # Now the user deletes that tree.
    tree.delete_tree(db, str(first["id"]))
    assert tree.is_tombstoned(db, kind="chat", ref_id=conversation_id) is True

    # A later backfill must not recreate what was just deleted.
    assert builder.backfill() == 0
    assert tree.tree_by_ref(db, kind="chat", ref_id=conversation_id) is None


def test_clear_tombstone_allows_backfill_again(tmp_path: Path):
    """Dropping the tombstone is the explicit "restore this memory" path."""
    paths = Paths.create(tmp_path / "data")
    from aiolocal.config import Settings, SettingsStore

    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b"))
    db = Database(paths.db_path)

    conversation_id = "conv-restore"
    store.ensure_conversation(db, conversation_id)
    store.archive_message(
        db, conversation_id=conversation_id, role="user", content="Bring me back."
    )
    store.archive_message(
        db, conversation_id=conversation_id, role="assistant", content="Restored."
    )
    builder = TreeBuilder(db, settings_store)
    builder.kick = lambda tree_id: None
    assert builder.backfill() == 1

    existing = tree.tree_by_ref(db, kind="chat", ref_id=conversation_id)
    tree.delete_tree(db, str(existing["id"]))
    assert builder.backfill() == 0

    with db.write() as c:
        tree.clear_tombstone(c, kind="chat", ref_id=conversation_id)
    assert tree.is_tombstoned(db, kind="chat", ref_id=conversation_id) is False

    assert builder.backfill() == 1
    assert tree.tree_by_ref(db, kind="chat", ref_id=conversation_id) is not None


def test_delete_trees_by_kind_clears_everything_of_that_kind(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    make_tree(db, "chat", ref_id="c1", count=1)
    make_tree(db, "chat", ref_id="c2", count=1)
    doc_id = store.insert_summary(
        db, title="Doc", body="# Doc", kind="document", source="document"
    )
    make_tree(db, "document", ref_id=doc_id, count=1)

    result = tree.delete_trees(db, kind="chat")

    assert result["deleted"] == 2
    assert tree.is_tombstoned(db, kind="chat", ref_id="c1") is True
    # The document tree is untouched by a chat-only delete.
    assert tree.tree_by_ref(db, kind="document", ref_id=doc_id) is not None


# --------------------------------------------------------------- concurrent roll-up
def test_two_rollups_cannot_consume_the_same_group(tmp_path: Path):
    """A manual roll-up racing the background drain must not duplicate a parent."""
    db = Database(tmp_path / "tree.db")
    tree_id = make_tree(db, "chat", count=3)

    def slow(degree: int, texts: list[str]) -> str:
        time.sleep(0.05)
        return digest(degree, texts)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(tree.rollup, db, tree_id, slow, max_nodes=1)
        second = pool.submit(tree.rollup, db, tree_id, slow, max_nodes=1)
        created = len(first.result()) + len(second.result())

    assert created == 1
    assert tree.stats(db, tree_id)["nodes"] == 1
    with db.read() as c:
        parents = c.execute(
            "SELECT parent_id FROM tree_chunks WHERE tree_id=? AND parent_id IS NOT NULL",
            (tree_id,),
        ).fetchall()
    assert len(parents) == 3
    assert 1 == len({row["parent_id"] for row in parents})


def test_builder_drain_is_serialised_per_tree(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    from aiolocal.config import Settings, SettingsStore

    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="fake"))
    db = Database(paths.db_path)
    tree_id = make_tree(db, "chat", count=3)
    builder = TreeBuilder(db, settings_store, model="fake")
    calls = []

    def slow(settings, model, degree, texts):
        calls.append(texts)
        time.sleep(0.05)
        return digest(degree, texts)

    builder._summarise = slow
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(builder.drain, tree_id, max_nodes=1)
        second = pool.submit(builder.drain, tree_id, max_nodes=1)
        total = first.result() + second.result()

    assert total == 1
    assert len(calls) == 1
    assert tree.stats(db, tree_id)["nodes"] == 1


# ------------------------------------------------------------------- back-trace
def test_backtrace_returns_the_exact_matching_chunk_first(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    tree_id = tree.create_tree(db, kind="chat", title="t")
    facts = {
        "c0": "User: Which planet is largest?\n\nAssistant: Jupiter is the largest planet.",
        "c1": "User: Explain photosynthesis.\n\nAssistant: Chlorophyll absorbs light in chloroplasts.",
        "c2": "User: Who wrote Hamlet?\n\nAssistant: William Shakespeare wrote Hamlet.",
    }
    for key, text in facts.items():
        tree.append_chunk(db, tree_id, key, {"text": text})
    tree.rollup(db, tree_id, lambda d, ts: f"level {d}: " + " ".join(ts), max_nodes=4)

    hits = tree.backtrace(db, tree_id, "photosynthesis chlorophyll", limit=6)
    assert hits
    assert hits[0].kind == "chunk"
    assert "chlorophyll" in hits[0].body.lower()
    # The path is a real root-to-leaf route, not a flat score.
    assert len(hits[0].path) >= 2


def test_backtrace_still_finds_chunks_before_the_first_rollup(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    tree_id = make_tree(db, "chat", count=2)
    hits = tree.backtrace(db, tree_id, "topic1", limit=4)
    assert hits and hits[0].kind == "chunk"
    assert "topic1" in hits[0].body


def test_backtrace_document_chunk_keeps_its_page_number(tmp_path: Path):
    db = Database(tmp_path / "tree.db")
    tree_id = tree.create_tree(db, kind="document", title="doc")
    tree.append_chunk(
        db,
        tree_id,
        "page:42",
        {"text": "Question 7 asks for the area of a trapezium.", "page_no": 42},
    )
    hits = tree.backtrace(db, tree_id, "area of a trapezium", limit=3)
    assert hits
    assert hits[0].kind == "chunk"
    assert hits[0].page_from == 42
    assert hits[0].page_to == 42