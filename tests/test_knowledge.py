from __future__ import annotations

from pathlib import Path

from aiolocal.config import Paths
from aiolocal.knowledge import KnowledgeBase, normalise_slug


def make_kb(tmp_path: Path) -> KnowledgeBase:
    paths = Paths.create(tmp_path / "data")
    return KnowledgeBase(paths.knowledge_db_path, files_dir=paths.knowledge_files)


def write_inbox(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_nested_slug_is_normalised_without_escaping():
    assert normalise_slug("AI / Papers!") == "ai/papers"
    assert normalise_slug("../etc/passwd") == "etc/passwd"
    assert normalise_slug("") == "general"


def test_inbox_nested_folders_create_parent_collections(tmp_path: Path):
    kb = make_kb(tmp_path)
    inbox = tmp_path / "inbox"
    write_inbox(inbox, "ai/vision/flow.md", "# Optical flow\n\nLucas-Kanade assumes small motion.")
    write_inbox(inbox, "cooking/bread.md", "# Sourdough\n\nBake at 240 C with steam.")

    result = kb.scan_inbox(inbox, embed_dim=128)
    assert result["added"] == 2, result

    collections = {c["slug"]: c for c in kb.list_collections()}
    assert "ai" in collections and "ai/vision" in collections
    assert collections["ai"]["parent_slug"] is None
    assert collections["ai/vision"]["parent_slug"] == "ai"
    assert collections["ai/vision"]["documents"] == 1
    assert collections["ai"]["subtree_documents"] == 1
    assert kb.collection_id("ai") is not None


def test_scan_limit_counts_only_new_documents(tmp_path: Path):
    kb = make_kb(tmp_path)
    inbox = tmp_path / "inbox"
    for index in range(45):
        write_inbox(inbox, f"bulk/doc-{index:02d}.md", f"# Note {index}\n\nunique-token-{index}")

    result = kb.scan_inbox(inbox, embed_dim=128)
    assert result["added"] == 45, result
    assert result["seen"] == 45
    # A second pass must skip all of them instead of consuming a 40-file budget.
    again = kb.scan_inbox(inbox, embed_dim=128, limit=40)
    assert again["added"] == 0
    assert again["skipped"] == 45
    assert again["more"] is False


def test_folder_summaries_route_auto_search_and_prefilter(tmp_path: Path):
    kb = make_kb(tmp_path)
    inbox = tmp_path / "inbox"
    write_inbox(
        inbox,
        "ai/vision/flow.md",
        "# Optical flow constraints\n\n"
        "The Lucas-Kanade method estimates dense optical flow under brightness constancy.",
    )
    write_inbox(
        inbox,
        "cooking/bread.md",
        "# Sourdough bread\n\nFerment the dough overnight and bake it with steam at 240 C.",
    )
    kb.scan_inbox(inbox, embed_dim=128)
    rebuilt = kb.rebuild_folder_summaries(embed_dim=128)
    assert rebuilt["updated"] >= 3, rebuilt

    routed = kb.route("how does optical flow brightness constancy work", embed_dim=128)
    assert routed["mode"] == "folder", routed
    assert routed["collections"][0].startswith("ai"), routed

    result = kb.search_auto(
        "how does optical flow brightness constancy work", embed_dim=128
    )
    assert result["items"], result
    assert result["items"][0]["collection"].startswith("ai")
    assert all(not item["collection"].startswith("cooking") for item in result["items"])

    explicit = kb.search(
        "sourdough steam", collections=["cooking"], embed_dim=128
    )
    assert explicit
    assert all(item["collection"].startswith("cooking") for item in explicit)


def test_auto_search_abstains_on_unrelated_question(tmp_path: Path):
    kb = make_kb(tmp_path)
    inbox = tmp_path / "inbox"
    write_inbox(inbox, "cooking/bread.md", "# Sourdough\n\nBake with steam at 240 C.")
    kb.scan_inbox(inbox, embed_dim=128)
    kb.rebuild_folder_summaries(embed_dim=128)

    result = kb.search_auto(
        "quantum chromodynamics lattice spacing renormalisation", embed_dim=128
    )
    assert result["items"] == [], result
    assert result["routing"].get("abstained") or result["routing"].get("fallback")


def test_deleting_a_parent_folder_deletes_nested_documents(tmp_path: Path):
    kb = make_kb(tmp_path)
    inbox = tmp_path / "inbox"
    write_inbox(inbox, "ai/vision/flow.md", "# Flow\n\nOptical flow.")
    write_inbox(inbox, "ai/speech/asr.md", "# ASR\n\nSpeech recognition.")
    write_inbox(inbox, "cooking/bread.md", "# Bread\n\nSteam.")
    kb.scan_inbox(inbox, embed_dim=128)
    kb.rebuild_folder_summaries(embed_dim=128)

    assert kb.delete_collection("ai") is True

    assert kb.collection_id("ai") is None
    assert kb.collection_id("ai/vision") is None
    assert kb.collection_id("ai/speech") is None
    assert kb.stats()["documents"] == 1
    assert kb.stats()["summarised_folders"] == 1


def test_source_metadata_is_inherited_from_kbmeta(tmp_path: Path):
    kb = make_kb(tmp_path)
    inbox = tmp_path / "inbox"
    (inbox / "dev").mkdir(parents=True)
    (inbox / "dev" / ".kbmeta.json").write_text(
        '{"source":"fastapi","source_url":"https://github.com/fastapi/fastapi",'
        '"license":"MIT"}',
        encoding="utf-8",
    )
    write_inbox(inbox, "dev/fastapi/tutorial.md", "# Tutorial\n\nCreate an app.")

    kb.scan_inbox(inbox, embed_dim=128)

    docs = kb.list_documents("dev/fastapi")
    assert len(docs) == 1
    assert docs[0]["source"] == "fastapi"
    assert docs[0]["license"] == "MIT"
