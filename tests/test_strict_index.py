from __future__ import annotations

import re
from pathlib import Path

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.db import Database
from aiolocal.memory import store
from aiolocal.tasks import (
    ResponseTask,
    ResponseTaskWorker,
    _aggregate_strict,
    _is_degenerate,
    _looks_like_description,
    _looks_like_furniture_page,
    _parse_strict_lines,
    _topic_in_text,
    _topic_key,
    looks_like_bulk_task,
    looks_like_strict_index,
)

TOPICS = [
    "Algebra",
    "Calculus",
    "Geometry",
    "Number Theory",
    "Statistics",
    "Trigonometry",
    "Combinatorics",
    "Complex Analysis",
    "Differential Equations",
    "Linear Algebra",
    "Discrete Mathematics",
    "Mathematical Analysis",
    "Real Analysis",
    "Abstract Algebra",
    "Topology",
    "Logic",
    "Probability Theory",
]


# ------------------------------------------------------------------- detection
def test_bulk_task_still_detected_for_large_documents():
    assert looks_like_bulk_task(
        "Summarise every page of this document", total_pages=120
    )
    # A small document is never a bulk job, however it is phrased.
    assert not looks_like_bulk_task(
        "Summarise every page of this document", total_pages=5
    )
    # A plain question about a big document stays a chat turn.
    assert not looks_like_bulk_task("What is on page 40?", total_pages=200)


def test_strict_index_requires_exhaustive_wording_and_topics():
    instruction = "Classify every page into exactly one of these topics:\n" + "\n".join(
        f"- {t}" for t in TOPICS
    )
    assert looks_like_strict_index(instruction, total_pages=204, topics=TOPICS)

    # No topics supplied: too ambiguous to take over the reply.
    assert not looks_like_strict_index(instruction, total_pages=204, topics=None)
    # Not an exhaustive instruction.
    assert not looks_like_strict_index(
        "Classify page 12 into one of these topics", total_pages=204, topics=TOPICS
    )
    # Too few pages to be worth a job.
    assert not looks_like_strict_index(instruction, total_pages=8, topics=TOPICS)


# --------------------------------------------------------------------- parsing
def test_parse_accepts_the_shapes_a_model_actually_emits():
    text = "\n".join(
        [
            "Algebra: 1, 3, 5",
            "- Calculus: 2",
            "**Geometry**: 4, 6",
            "Number Theory = 7",
            "None: 8",
            "this line is prose and must be ignored",
            "Topology: 9-11",
        ]
    )
    pairs = dict(_parse_strict_lines(text))
    assert pairs["Algebra"] == [1, 3, 5]
    assert pairs["Calculus"] == [2]
    assert pairs["Geometry"] == [4, 6]
    assert pairs["Number Theory"] == [7]
    assert pairs["None"] == [8]
    assert pairs["Topology"] == [9, 10, 11]
    assert "this line is prose and must be ignored" not in pairs


def test_parse_ignores_a_line_without_pages():
    """A heading with no page numbers must not consume a page silently."""
    assert _parse_strict_lines("Algebra: none here") == []
    assert _parse_strict_lines("Algebra:") == []


# ----------------------------------------------------------------- aggregation
def test_aggregate_outputs_one_line_per_topic_in_the_users_order():
    raw = "Calculus: 1\nAlgebra: 2"
    result = _aggregate_strict(raw, ["Algebra", "Calculus"], total_units=2)
    assert [topic for topic, _ in result] == ["Algebra", "Calculus"]
    assert dict(result) == {"Algebra": [2], "Calculus": [1]}


def test_aggregate_merges_repeated_segments_and_dedupes_pages():
    raw = "Algebra: 1, 2\nAlgebra: 2, 3"
    result = _aggregate_strict(raw, ["Algebra"], total_units=3)
    assert result == [("Algebra", [1, 2, 3])]


def test_aggregate_maps_model_spelling_onto_the_users_topic():
    """'number-theory' must land in the user's 'Number Theory' bucket."""
    raw = "number-theory: 5\nNUMBER THEORY: 6"
    result = _aggregate_strict(raw, ["Number Theory"], total_units=6)
    assert result == [("Number Theory", [5, 6])]


def test_topic_key_maps_a_shortened_topic_by_dropped_middle_words():
    topics = [
        "Number: Arithmetic, Fractions, Decimals, Percentages & Ratio",
        "Mensuration (Perimeter, Area, Surface Area & Volume)",
        "Coordinate Geometry",
        "Geometry: Angles, Polygons, Constructions, Bearings & Symmetry",
    ]
    # Middle words dropped ("Fractions, Decimals") must still resolve.
    assert _topic_key("Number: Arithmetic, Percentages & Ratio", topics) == topics[0]
    assert _topic_key("Number: Arithmetic", topics) == topics[0]
    assert _topic_key("Mensuration", topics) == topics[1]
    # A bare ambiguous word is never guessed.
    assert _topic_key("Geometry", topics) == ""
    # An invented topic is dropped.
    assert _topic_key("Astrology", topics) == ""


def test_aggregate_gives_empty_topics_a_none_placeholder():
    raw = "Algebra: 1"
    result = _aggregate_strict(raw, ["Algebra", "Topology"], total_units=2)
    assert result == [("Algebra", [1]), ("Topology", [])]


def test_aggregate_keeps_a_page_in_every_topic_it_tests():
    """A multi-part page belongs to each topic it tests, not just the first."""
    raw = "Algebra: 1\nCalculus: 1"
    result = _aggregate_strict(raw, ["Algebra", "Calculus"], total_units=1)
    assert dict(result) == {"Algebra": [1], "Calculus": [1]}


def test_aggregate_drops_invented_topics_but_keeps_user_lines():
    """The deliverable must have exactly the user's topic lines."""
    raw = "Algebra: 1\nAstrology: 2"
    result = _aggregate_strict(raw, ["Algebra", "Calculus"], total_units=2)
    assert [topic for topic, _ in result] == ["Algebra", "Calculus"]
    assert dict(result) == {"Algebra": [1], "Calculus": []}


def test_aggregate_accepts_the_real_topic_names_with_colons():
    topics = [
        "Number: Arithmetic, Fractions, Decimals, Percentages & Ratio",
        "Surds & Indices",
        "Sets & Logic/Venn Diagrams",
    ]
    raw = (
        "Number: Arithmetic, Fractions, Decimals, Percentages & Ratio: 2, 3\n"
        "Surds & Indices: 4\n"
        "Sets & Logic/Venn Diagrams: 5\n"
    )
    result = _aggregate_strict(raw, topics, total_units=5)
    assert dict(result) == {topics[0]: [2, 3], topics[1]: [4], topics[2]: [5]}


def test_aggregate_drops_pages_that_do_not_exist():
    raw = "Algebra: 1, 99"
    result = _aggregate_strict(raw, ["Algebra"], total_units=2)
    assert result == [("Algebra", [1])]


def test_aggregate_treats_none_wording_as_unassigned_not_a_topic():
    """'None' is the empty-list placeholder, never a topic of its own.

    A page the model could not place must not create an extra output line; it
    stays unassigned so the caller can report it.
    """
    raw = "None: 1\nunclassified: 2\nAlgebra: 3"
    result = _aggregate_strict(raw, ["Algebra"], total_units=3)
    assert result == [("Algebra", [3])]
    covered = {page for _, pages in result for page in pages}
    assert covered == {3}


# ------------------------------------------------- loop / furniture detection
def test_degenerate_detector_flags_a_repetition_loop():
    loop = "\n".join(["Graphs & Kinematics: 31"] * 40)
    assert _is_degenerate(loop)
    # A normal multi-page batch has distinct page numbers, so it is not a loop.
    ok = "\n".join(f"Algebra: {p}" for p in range(1, 11))
    assert not _is_degenerate(ok)
    # Short answers are never treated as loops.
    assert not _is_degenerate("Algebra: 1\nAlgebra: 1")


def test_furniture_detector_matches_blank_and_cover_pages():
    assert _looks_like_furniture_page("### BLANK PAGE\nPermission to reproduce ...")
    assert _looks_like_furniture_page(
        "Cambridge O Level MATHEMATICS (SYLLABUS D) 4024/21 2 hours 30 minutes "
        "READ THESE INSTRUCTIONS FIRST Candidates answer on the Question Paper."
    )
    assert not _looks_like_furniture_page("### Q7. A bag contains 10 tiles ...")


def test_copyright_notice_alone_is_not_furniture():
    """The last page of a paper carries the copyright notice *and* a question."""
    page = (
        "(c) Transformation M is a reflection in the line x = 2. Draw and label "
        "triangle D. Permission to reproduce items where third-party owned material "
        "protected by copyright is included has been sought and cleared where possible."
    )
    assert not _looks_like_furniture_page(page)
    assert _looks_like_furniture_page("This page is visually blank.")


# --------------------------------------------------- batch + repair pipeline
def _tiny_env(tmp_path: Path, pages: list[str]):
    paths = Paths.create(tmp_path / "data")
    store_ = SettingsStore(paths)
    store_.save(Settings(chat_model="qwen3.5-9b"))
    db = Database(paths.db_path)
    worker = ResponseTaskWorker(paths, store_, db)
    doc = store.insert_summary(
        db,
        title="Document: Q",
        body="# Document\n\n## Overview\n_done_",
        topic_slug="documents",
        kind="document",
        source="document",
    )
    with db.write() as c:
        for no, text in enumerate(pages, start=1):
            c.execute(
                """INSERT INTO document_pages(document_id, page_no, text, tokens, created_at)
                   VALUES(?,?,?,?,?)""",
                (doc, no, text, 8, 1),
            )
    task = ResponseTask(
        id="t1",
        conversation_id="c",
        instruction="Classify every page into these topics",
        topics=["Algebra", "Statistics"],
        document_ids=[doc],
        units=[f"{doc}:{no}" for no in range(1, len(pages) + 1)],
        total_units=len(pages),
    )
    return worker, task


def _covered(text: str) -> set[int]:
    return {p for _, pages in _parse_strict_lines(text) for p in pages}


def test_classify_segment_repairs_pages_the_batch_skipped(tmp_path: Path):
    worker, task = _tiny_env(tmp_path, [f"Page {n}: question {n}" for n in range(1, 7)])

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:  # the batch call
            return "Algebra: 1\nStatistics: 2\nAlgebra: 3\nStatistics: 4"
        n = int(re.search(r"PAGE (\d+)", prompt).group(1))
        return f"{'Statistics' if n % 2 else 'Algebra'}: {n}"

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 6)
    assert _covered(text) == {1, 2, 3, 4, 5, 6}
    # A multi-topic page is still allowed; nothing is invented.
    assert {k for k, _ in _parse_strict_lines(text)} <= {"Algebra", "Statistics"}


def test_classify_segment_discards_a_looping_batch(tmp_path: Path):
    worker, task = _tiny_env(tmp_path, [f"Page {n}: question {n}" for n in range(1, 5)])

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Statistics: 2\n" * 60  # a repetition loop
        n = int(re.search(r"PAGE (\d+)", prompt).group(1))
        return f"Algebra: {n}"

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 4)
    # The loop is thrown away and every page is re-classified one by one.
    assert _covered(text) == {1, 2, 3, 4}


def test_classify_segment_marks_furniture_pages_none(tmp_path: Path):
    pages = ["Page 1: question 1", "### BLANK PAGE\ncopyright", "Page 3: question 3"]
    worker, task = _tiny_env(tmp_path, pages)

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Algebra: 1\nAlgebra: 3"
        n = int(re.search(r"PAGE (\d+)", prompt).group(1))
        return f"Algebra: {n}"

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 3)
    pairs = _parse_strict_lines(text)
    none_pages = {p for k, v in pairs if k == "None" for p in v}
    topic_pages = {p for k, v in pairs if k != "None" for p in v}
    assert none_pages == {2}  # recorded, but not a topic
    assert topic_pages == {1, 3}


def test_repair_accepts_a_topic_answer_without_a_page_number(tmp_path: Path):
    """A single-page prompt sometimes yields just the topic; attach the page."""
    worker, task = _tiny_env(tmp_path, ["Page 1: question 1", "Page 2: question 2"])

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Algebra: 1"  # page 2 skipped by the batch
        return "Statistics"      # topic only, no page number

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 2)
    assert ("Statistics", [2]) in _parse_strict_lines(text)


def test_repair_tolerates_a_trailing_full_stop(tmp_path: Path):
    """'Topic: 2.' must not be thrown away because of the final period."""
    worker, task = _tiny_env(tmp_path, ["Page 1: question 1", "Page 2: question 2"])

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Algebra: 1"
        return "Statistics: 2."

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 2)
    assert ("Statistics", [2]) in _parse_strict_lines(text)


def test_repair_retries_none_on_a_substantial_page(tmp_path: Path):
    """A real question answered 'None' is retried instead of being dropped."""
    long_page = "Page 2: " + ("solve the simultaneous equations " * 10)
    worker, task = _tiny_env(tmp_path, ["Page 1: q1", long_page])
    calls = {"n": 0}

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Algebra: 1"
        calls["n"] += 1
        return "None: 2" if calls["n"] == 1 else "Algebra: 2"

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 2)
    assert ("Algebra", [2]) in _parse_strict_lines(text)
    assert calls["n"] >= 2

def test_topic_in_text_accepts_one_buried_topic():
    topics = ["Statistics", "Probability", "Coordinate Geometry", "Geometry: Angles & Polygons"]
    assert _topic_in_text("This page is about statistics.", topics) == "Statistics"
    # Two topics named in one sentence is ambiguous.
    assert _topic_in_text("It mixes probability and statistics.", topics) == ""
    # Nothing recognisable.
    assert _topic_in_text("I cannot help with that.", topics) == ""


def test_ask_retry_recovers_from_a_transient_empty_answer(tmp_path: Path):
    worker, _task = _tiny_env(tmp_path, ["Page 1: q1"])
    calls = {"n": 0}

    def flaky(settings, model, prompt, *, max_tokens, system=""):
        calls["n"] += 1
        return "" if calls["n"] == 1 else "Algebra: 1"

    worker._ask = flaky
    out = worker._ask_retry(worker.settings_store.load(), "m", "p", max_tokens=20)
    assert out == "Algebra: 1"
    assert calls["n"] == 2


def test_repair_finds_a_topic_inside_a_sentence(tmp_path: Path):
    """A prose answer that names exactly one topic is still usable."""
    worker, task = _tiny_env(tmp_path, ["Page 1: q1", "Page 2: q2"])

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Algebra: 1"
        return "The page is about statistics, so the line is Statistics."

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 2)
    assert ("Statistics", [2]) in _parse_strict_lines(text)


def test_topic_key_drops_a_trailing_subtopic_label():
    topics = [
        "Number: Arithmetic, Fractions, Decimals, Percentages & Ratio",
        "Geometry: Angles, Polygons, Constructions, Bearings & Symmetry",
        "Mensuration (Perimeter, Area, Surface Area & Volume)",
    ]
    assert _topic_key("Number: Arithmetic: Compound Interest", topics) == topics[0]
    assert _topic_key("Number: Compound Interest", topics) == topics[0]
    assert _topic_key("Geometry: Angles", topics) == topics[1]
    assert _topic_key("Mensuration: Volume of a cone", topics) == topics[2]


def test_topic_key_maps_invented_wording_onto_its_topic():
    """Page 197's answer "Exponential Equations & Logarithms" must resolve."""
    topics = [
        "Surds & Indices",
        "Equations & Inequalities",
        "Graphs & Kinematics",
        "Trigonometry (2D & 3D)",
        "Legacy / Out-of-Current-Syllabus (2019 & Earlier Papers)",
    ]
    # Keyword alias: "exponential equation" + "logarithm" both belong to
    # Surds & Indices, even though "equations" also overlaps with
    # Equations & Inequalities.
    assert _topic_key("Exponential Equations & Logarithms", topics) == topics[0]
    # But "Exponential Graphs" is not an equation: it falls through to the
    # word-overlap fallback and, correctly, onto Graphs & Kinematics.
    assert _topic_key("Exponential Graphs", topics) == topics[2]
    # A term no topic names verbatim resolves via the alias map.
    assert _topic_key("Standard Form", topics) == topics[0]
    assert _topic_key("Pythagoras' theorem", topics) == topics[3]
    # Matrices are a legacy-syllabus item per the user's rules.
    assert _topic_key("Matrices", topics) == topics[4]
    # An invented topic with no keyword and no word overlap stays unassigned.
    assert _topic_key("Astrology", topics) == ""


def test_topic_key_picks_the_unique_closest_topic_by_shared_words():
    topics = [
        "Surds & Indices",
        "Equations & Inequalities",
        "Statistics",
    ]
    # "Inequalities" alone shares words with exactly one topic.
    assert _topic_key("Solving Inequalities", topics) == topics[1]
    # "Data & Averages" shares nothing with any topic: left unassigned, not guessed.
    assert _topic_key("Data & Averages", topics) == ""


def test_description_style_page_is_classified_from_its_subject(tmp_path: Path):
    desc = (
        "Page 2 Subject: A question from an examination paper. Purpose: To ask the "
        "examinee to explain the meaning of each element in a matrix denoted by P. "
        "Visible content: (ii) Explain what each element in P represents. No diagrams, "
        "charts, tables, or equations are visible on this page. " + "padding " * 30
    )
    worker, task = _tiny_env(tmp_path, ["Page 1: q1", desc])

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        if "PAGES TO CLASSIFY" in prompt:
            return "Algebra: 1"
        if "DESCRIBES page" in prompt:
            return "Statistics: 2"
        return "None: 2"

    worker._ask = fake_ask
    text = worker._classify_segment(worker.settings_store.load(), "m", task, 0, 2)
    assert ("Statistics", [2]) in _parse_strict_lines(text)


def test_description_detector():
    assert _looks_like_description("Subject: a question. Visible content: (ii) Explain ...")
    assert _looks_like_description("No diagrams, charts, tables, or equations are visible.")
    assert not _looks_like_description("### Q7. A bag contains 10 tiles ...")


def test_sweep_classifies_pages_the_main_pass_missed(tmp_path: Path):
    """A page lost by the batch is recovered by the end-of-job sweep."""
    worker, task = _tiny_env(
        tmp_path, [f"Page {n}: question {n}" for n in range(1, 4)]
    )
    worker._append_chunk(task, 0, 3, 1, 3, "Algebra: 1")

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        n = int(re.search(r"PAGE (\d+)", prompt).group(1))
        return f"Statistics: {n}"

    worker._ask = fake_ask
    worker._sweep_strict(worker.settings_store.load(), "m", task)

    with worker.db.read() as c:
        rows = c.execute(
            "SELECT content FROM response_task_chunks WHERE task_id=? ORDER BY ordinal",
            (task.id,),
        ).fetchall()
    assigned = _aggregate_strict(
        "\n".join(str(r["content"]) for r in rows), task.topics, task.total_units
    )
    covered = {p for _, pages in assigned for p in pages}
    assert covered == {1, 2, 3}


def test_sweep_chunk_does_not_rewind_the_resume_cursor(tmp_path: Path):
    """Sweep output spans the whole job so a resume cannot redo early pages."""
    worker, task = _tiny_env(
        tmp_path, [f"Page {n}: question {n}" for n in range(1, 6)]
    )
    worker._append_chunk(task, 0, 5, 1, 5, "Algebra: 1")

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        n = int(re.search(r"PAGE (\d+)", prompt).group(1))
        return f"Statistics: {n}"

    worker._ask = fake_ask
    worker._sweep_strict(worker.settings_store.load(), "m", task)
    assert worker._persisted_cursor(task) == task.total_units


def test_sweep_skips_furniture_and_leaves_it_unassigned(tmp_path: Path):
    worker, task = _tiny_env(tmp_path, ["Page 1: q1", "### BLANK PAGE\ncopyright"])
    worker._append_chunk(task, 0, 2, 1, 2, "Algebra: 1")
    calls: list[str] = []

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        calls.append(prompt)
        return "Statistics: 2"

    worker._ask = fake_ask
    worker._sweep_strict(worker.settings_store.load(), "m", task)
    assert calls == []  # a blank page is never sent to the model


def test_sweep_is_a_no_op_when_everything_is_classified(tmp_path: Path):
    worker, task = _tiny_env(tmp_path, ["Page 1: q1", "Page 2: q2"])
    worker._append_chunk(task, 0, 2, 1, 2, "Algebra: 1\nStatistics: 2")
    calls: list[str] = []

    def fake_ask(settings, model, prompt, *, max_tokens, system=""):
        calls.append(prompt)
        return "Algebra: 1"

    worker._ask = fake_ask
    worker._sweep_strict(worker.settings_store.load(), "m", task)
    assert calls == []
