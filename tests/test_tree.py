from __future__ import annotations

import json
from pathlib import Path

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.db import Database
from aiolocal.memory import store, tree
from aiolocal.memory.treebuild import TreeBuilder


def make_tree(tmp_path: Path, count: int) -> tuple[Database, str]:
    db = Database(tmp_path / "tree.db")
    tree_id = tree.create_tree(db, kind="chat", title="test")
    for index in range(count):
        tree.append_chunk(
            db,
            tree_id,
            f"c{index}",
            {"text": f"unit {index} about topic{index}"},
            tokens=8,
        )
    return db, tree_id


def digest(degree: int, texts: list[str]) -> str:
    return f"degree-{degree}: " + " | ".join(texts)


def test_three_chunks_make_one_summary_and_nine_make_a_second_level(tmp_path: Path):
    db, tree_id = make_tree(tmp_path, 9)
    created = tree.rollup(db, tree_id, digest, max_nodes=10)
    assert len(created) == 4  # 3 degree-1 nodes + 1 degree-2 root
    stats = tree.stats(db, tree_id)
    assert stats["nodes"] == 4
    assert stats["depth"] == 2
    assert stats["chunks"] == 9
    assert stats["chunks_unrolled"] == 0

    detail = tree.outline(db, tree_id)
    by_degree = {}
    for node in detail["nodes"]:
        by_degree.setdefault(node["degree"], []).append(node)
    assert len(by_degree[1]) == 3
    assert len(by_degree[2]) == 1
    root = by_degree[2][0]
    assert root["parent"] is None
    assert len(root["children"]) == 3
    # Provenance is mandatory: every node names the units it covers.
    for node in detail["nodes"]:
        assert node["children"]
        assert node["summary"]


def test_rollup_never_creates_a_node_without_model_text(tmp_path: Path):
    db, tree_id = make_tree(tmp_path, 3)
    assert tree.rollup(db, tree_id, lambda degree, texts: "", max_nodes=5) == []
    assert tree.stats(db, tree_id)["nodes"] == 0
    # Chunks stay unconsumed so a later successful run can still roll them up.
    with db.read() as c:
        parent = c.execute(
            "SELECT parent_id FROM tree_chunks WHERE tree_id=? AND chunk_key='c0'",
            (tree_id,),
        ).fetchone()[0]
    assert parent is None


def test_reappending_a_chunk_keeps_its_ordinal_and_does_not_duplicate(tmp_path: Path):
    db, tree_id = make_tree(tmp_path, 2)
    first = tree.append_chunk(db, tree_id, "c0", {"text": "unit 0 rewritten"})
    assert first is not None
    assert first.ordinal == 0
    assert tree.stats(db, tree_id)["chunks"] == 2
    assert first.text == "unit 0 rewritten"


def test_backtrace_descends_to_the_exact_chunk(tmp_path: Path):
    db, tree_id = make_tree(tmp_path, 9)
    tree.rollup(db, tree_id, digest, max_nodes=10)
    hits = tree.backtrace(db, tree_id, "topic7", limit=6)
    assert hits
    bodies = "\n".join(hit.body for hit in hits)
    assert "topic7" in bodies
    exact = [hit for hit in hits if hit.kind == "chunk" and "topic7" in hit.body]
    assert exact, [hit.to_dict() for hit in hits]
    # The exact chunk carries the full provenance path, not just its own id.
    assert exact[0].path[-1] == "c7"
    assert len(exact[0].path) > 1


def test_backtrace_compares_more_than_one_branch(tmp_path: Path):
    db, tree_id = make_tree(tmp_path, 9)
    tree.rollup(db, tree_id, digest, max_nodes=10)
    hits = tree.backtrace(db, tree_id, "topic1", branches=3, limit=10)
    degrees = {hit.degree for hit in hits}
    # Both a summary level and a leaf are considered, so one unlucky parent
    # cannot hide the answer.
    assert 0 in degrees
    assert any(degree >= 1 for degree in degrees)


def test_usage_strengthens_a_branch_without_rewriting_the_node(tmp_path: Path):
    db, tree_id = make_tree(tmp_path, 3)
    node_id = tree.rollup(db, tree_id, digest, max_nodes=1)[0]
    before = tree.outline(db, tree_id)["nodes"][0]["summary"]
    tree.record_usage(db, tree_id, [node_id])
    tree.record_usage(db, tree_id, [node_id])
    detail = tree.outline(db, tree_id)
    assert detail["nodes"][0]["relevance"] > 0
    assert detail["nodes"][0]["summary"] == before


def test_backfill_seeds_existing_data_once_and_prefers_page_notes(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    settings_store = SettingsStore(paths)
    settings_store.save(Settings(chat_model="qwen3.5-9b"))
    db = Database(paths.db_path)

    store.ensure_conversation(db, "old-chat")
    store.archive_message(
        db, conversation_id="old-chat", role="user", content="We chose Tokio."
    )
    store.archive_message(
        db, conversation_id="old-chat", role="assistant", content="Noted for later."
    )

    document_id = store.insert_summary(
        db,
        title="Document: Old manual",
        body="# Old manual",
        kind="document",
        source="document",
    )
    with db.write() as c:
        for page in (1, 2):
            c.execute(
                """INSERT INTO document_pages(document_id, page_no, text, tokens, created_at)
                   VALUES(?,?,?,?,?)""",
                (document_id, page, f"raw page {page}", 4, 1),
            )
            c.execute(
                """INSERT INTO document_page_notes(document_id, page_no, note, prev_summary,
                       tokens, complete, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (document_id, page, f"noted page {page}", None, 4, 1, 1),
            )

    builder = TreeBuilder(db, settings_store)
    builder.kick = lambda tree_id: None  # keep the test deterministic

    assert builder.backfill() == 3

    chat_tree = tree.tree_by_ref(db, kind="chat", ref_id="old-chat")
    assert chat_tree is not None
    chat_chunks = tree.pending_chunks(db, str(chat_tree["id"]))
    assert len(chat_chunks) == 1
    assert chat_chunks[0].chunk_key.startswith("exchange:")
    assert "We chose Tokio." in chat_chunks[0].text
    assert "Noted for later." in chat_chunks[0].text

    doc_tree = tree.tree_by_ref(db, kind="document", ref_id=document_id)
    assert doc_tree is not None
    doc_chunks = tree.pending_chunks(db, str(doc_tree["id"]))
    assert [chunk.text for chunk in doc_chunks] == ["noted page 1", "noted page 2"]

    # Running again appends nothing and rewrites nothing.
    assert builder.backfill() == 0
    assert tree.stats(db, str(doc_tree["id"]))["chunks"] == 2
