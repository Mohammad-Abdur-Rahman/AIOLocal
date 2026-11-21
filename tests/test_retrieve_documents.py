from __future__ import annotations

from pathlib import Path

from aiolocal.config import Paths, Settings
from aiolocal.db import Database
from aiolocal.memory import store
from aiolocal.memory.retrieve import (
    _document_items,
    document_pages_range,
    parse_page_refs,
    retrieve,
)
from aiolocal.memory.text import embed


def make_db(tmp_path: Path) -> tuple[Database, Settings]:
    paths = Paths.create(tmp_path / "data")
    return Database(paths.db_path), Settings()


def add_document(db: Database, settings: Settings, title: str, pages: int) -> str:
    document_id = store.insert_summary(
        db,
        title=f"Document: {title}",
        body=f"# {title}\n\n## Overview\nPages about {title}.",
        topic_slug="documents",
        kind="document",
        source="document",
        embed_dim=settings.embed_dim,
    )
    with db.write() as c:
        for page in range(1, pages + 1):
            text = f"Page {page} of {title}: alpha beta topic {page}."
            c.execute(
                """INSERT INTO document_pages(document_id, page_no, text, tokens, created_at)
                   VALUES(?,?,?,?,?)""",
                (document_id, page, text, 8, 1),
            )
    return document_id


def test_parse_page_refs_handles_single_range_and_inline_forms():
    assert parse_page_refs("page 12") == [(12, 12)]
    assert parse_page_refs("pages 3-5") == [(3, 5)]
    assert parse_page_refs("What does page 12 say about alpha?") == [(12, 12)]
    assert parse_page_refs("compare pages 9-4") == [(4, 9)]
    assert parse_page_refs("pages 3-5 and page 3") == [(3, 5), (3, 3)]
    assert parse_page_refs("What is the conclusion?") == []


def test_document_pages_range_is_exact_and_ordered(tmp_path: Path):
    db, settings = make_db(tmp_path)
    document_id = add_document(db, settings, "Manual", 8)

    items = document_pages_range(db, document_id, 6, 4)
    assert [item.id for item in items] == [
        f"{document_id}:4",
        f"{document_id}:5",
        f"{document_id}:6",
    ]
    assert "Page 4" in items[0].body
    assert all(item.kind == "document_page" for item in items)

    # A hard cap keeps a huge range from flooding the context.
    capped = document_pages_range(db, document_id, 1, 8, limit=3)
    assert len(capped) == 3


def test_explicit_page_reference_resolves_for_an_attached_document(tmp_path: Path):
    db, settings = make_db(tmp_path)
    attached = add_document(db, settings, "Attached", 6)
    other = add_document(db, settings, "Other", 6)

    result = retrieve(db, "pages 2-3", settings=settings, document_ids=[attached])
    ids = {item.id for item in result.items}
    assert f"{attached}:2" in ids
    assert f"{attached}:3" in ids
    assert f"{other}:2" not in ids
    assert result.plan["sources"]["document_pages_explicit"] == [
        f"{attached}:2",
        f"{attached}:3",
    ]

    # No opt-in means no document pages, even with a perfect page reference.
    empty = retrieve(db, "page 2", settings=settings, document_ids=[])
    assert not any(str(item.id).startswith(f"{attached}:") for item in empty.items)
    assert not any(str(item.id).startswith(f"{other}:") for item in empty.items)


def test_document_items_are_scoped_to_opted_in_documents(tmp_path: Path):
    db, settings = make_db(tmp_path)
    attached = add_document(db, settings, "Attached", 3)
    other = add_document(db, settings, "Other", 3)

    with db.read() as c:
        rows = c.execute(
            "SELECT document_id, page_no, text FROM document_pages ORDER BY document_id, page_no"
        ).fetchall()
    for row in rows:
        store.put_embedding(
            db,
            "document_page",
            f"{row['document_id']}:{row['page_no']}",
            "hash",
            embed(str(row["text"]), settings.embed_dim),
        )

    can_read = _document_items(
        db, "alpha beta topic", settings, document_ids=[attached], page_limit=6
    )
    assert can_read
    assert all(str(item.id).startswith(f"{attached}:") for item in can_read)

    cannot_read = _document_items(db, "alpha beta topic", settings, document_ids=[])
    assert cannot_read == []

    unrestricted = _document_items(db, "alpha beta topic", settings, document_ids=None)
    doc_ids = {str(item.id).split(":", 1)[0] for item in unrestricted}
    assert doc_ids == {attached, other}


def test_page_reference_uses_the_most_recent_attached_document_when_no_summary_matches(
    tmp_path: Path,
):
    db, settings = make_db(tmp_path)
    document_id = add_document(db, settings, "Astronomy", 5)

    # The wording shares no terms with the overview, so no summary is ranked; the
    # page reference must still resolve inside the attached document.
    result = retrieve(db, "page 4", settings=settings, document_ids=[document_id])
    assert any(item.id == f"{document_id}:4" for item in result.items)
