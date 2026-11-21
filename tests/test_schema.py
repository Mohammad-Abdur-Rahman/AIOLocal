from __future__ import annotations

import sqlite3
from pathlib import Path

from aiolocal.db import Database, SCHEMA_VERSION


def make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "memory.db")


def test_schema_has_tree_scope_and_task_tables(tmp_path: Path):
    db = make_db(tmp_path)
    with db.read() as c:
        names = {
            str(r[0])
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    for table in (
        "memory_trees",
        "tree_nodes",
        "tree_chunks",
        "tree_node_usage",
        "document_page_notes",
        "conversation_documents",
        "response_tasks",
        "response_task_chunks",
    ):
        assert table in names
    with db.read() as c:
        version = int(c.execute("PRAGMA user_version").fetchone()[0])
    assert version == SCHEMA_VERSION == 5


def test_old_database_is_upgraded_in_place(tmp_path: Path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, source TEXT NOT NULL DEFAULT 'native',"
        " external_id TEXT, started_at INTEGER NOT NULL, last_at INTEGER NOT NULL, meta TEXT)"
    )
    conn.execute("PRAGMA user_version=3")
    conn.commit()
    conn.close()
    db = Database(path)
    with db.read() as c:
        names = {str(r[0]) for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = int(c.execute("PRAGMA user_version").fetchone()[0])
    assert "tree_nodes" in names
    assert "conversation_documents" in names
    assert version == SCHEMA_VERSION


def test_a_v4_database_keeps_its_rows_and_gains_the_new_tables(tmp_path: Path):
    path = tmp_path / "v4.db"
    db = Database(path)
    with db.write() as c:
        c.execute(
            "INSERT INTO conversations(id, title, started_at, last_at) VALUES(?,?,?,?)",
            ("old-conv", "Old chat", 1, 2),
        )
        c.execute(
            """INSERT INTO messages(id, conversation_id, role, content, content_hash,
                   tokens, created_at) VALUES(?,?,?,?,?,?,?)""",
            ("old-msg", "old-conv", "user", "remember this", "hash", 3, 1),
        )
    db.close()

    # Emulate a v4 file: the tree/task tables did not exist yet.
    conn = sqlite3.connect(path)
    for table in (
        "memory_trees",
        "tree_nodes",
        "tree_chunks",
        "tree_embeddings",
        "tree_node_usage",
        "document_page_notes",
        "conversation_documents",
        "response_tasks",
        "response_task_chunks",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute("PRAGMA user_version=4")
    conn.commit()
    conn.close()

    upgraded = Database(path)
    with upgraded.read() as c:
        assert c.execute("SELECT content FROM messages WHERE id='old-msg'").fetchone()[0] == "remember this"
        names = {str(r[0]) for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = int(c.execute("PRAGMA user_version").fetchone()[0])
    assert version == SCHEMA_VERSION
    assert {"tree_nodes", "conversation_documents", "response_tasks"} <= names
