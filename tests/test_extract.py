"""Unit tests for the layout-aware extractor in ``aiolocal/extract.py``.

These tests use small synthetic pages plus ``test.pdf`` (the real exam paper at
the project root) when it is available, so they run offline without docling or
MinerU installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiolocal.extract import (
    LayoutDocument,
    Line,
    PageLayout,
    PageResult,
    blocks_for,
    furniture,
    is_furniture,
    merge_pages,
    needs_vision,
    parse_docling_json,
    parse_mineru_content_list,
    read_page,
    render_markdown,
)

PDF = Path("test.pdf")


def _line(text, x0=50.0, y0=100.0, x1=200.0, y1=110.0, size=10.0):
    return Line(text=text, x0=x0, y0=y0, x1=x1, y1=y1, size=size)


# ------------------------------------------------------------------ bands / cells


def test_bands_merge_same_line():
    from aiolocal.extract import _bands

    # Two glyph runs at slightly different baselines belong to one visual line.
    chars = [
        (10, 100, 30, 106, "H"),
        (30, 101, 45, 107, "i"),
        (10, 120, 30, 126, "B"),
    ]
    bands = _bands(chars)
    assert len(bands) == 2
    merged = next(b for b in bands if len(b["chars"]) == 2)
    texts = "".join(c[4] for c in sorted(merged["chars"], key=lambda c: c[0]))
    assert texts == "Hi"


def test_cells_splits_table_columns():
    from aiolocal.extract import _cells

    chars = [
        (50, 100, 90, 106, "A"),
        (50, 100, 90, 106, "d"),
        (50, 100, 90, 106, "u"),
        (50, 100, 90, 106, "l"),
        (50, 100, 90, 106, "t"),
        (150, 100, 180, 106, "$"),
        (150, 100, 180, 106, "2"),
        (150, 100, 180, 106, "6"),
    ]
    texts, starts = _cells(chars)
    assert texts == ("Adult", "$26")
    assert len(starts) == 2


def test_cells_reinserts_word_spaces():
    from aiolocal.extract import _cells

    # A gap between WORD_GAP and CELL_GAP becomes a space inside one cell.
    chars = [
        (50, 100, 60, 106, "C"),
        (50, 100, 60, 106, "h"),
        (50, 100, 60, 106, "i"),
        (50, 100, 60, 106, "l"),
        (50, 100, 60, 106, "d"),
        (66, 100, 76, 106, "("),
        (66, 100, 76, 106, "A"),
        (66, 100, 76, 106, "g"),
        (66, 100, 76, 106, "e"),
        (66, 100, 76, 106, ")"),
    ]
    texts, _ = _cells(chars)
    assert texts == ("Child (Age)",)


# ----------------------------------------------------------- furniture detection


def test_is_furniture_running_header():
    layout = PageLayout(page_no=1, width=595.0, height=842.0)
    running = {"page # of #"}
    line = _line("Page 3 of 584", x0=50, y0=20, x1=150, y1=30)
    assert is_furniture(line, running, layout) is True


def test_is_furniture_not_body():
    layout = PageLayout(page_no=1, width=595.0, height=842.0)
    running = {"completely unrelated header"}
    line = _line("UCLES 93 4024/22", x0=50, y0=400, x1=250, y1=410)
    assert is_furniture(line, running, layout) is False


def test_furniture_needs_repetition():
    pages = {
        1: [_line("Page 1 of 584")],
        2: [_line("Page 2 of 584")],
        3: [_line("Page 3 of 584")],
        4: [_line("unrelated body text here")],
    }
    found = furniture(pages)
    assert any("page" in s and "#" in s for s in found)


# ----------------------------------------------------------- block classification


def test_blocks_for_question_and_table():
    layout = PageLayout(page_no=1, width=595.0, height=842.0)
    layout.lines = [
        _line("1 The table below shows the cost to visit a zoo.", x0=50),
        _line("Adult   $26.00", x0=204),
        _line("Child   $15.50", x0=204),
        _line("(a) On 30th April, the family visits.", x0=73),
        _line("Calculate the total cost.", x0=73),
    ]
    # Two rows share a cell anchor -> a table.
    layout.lines[1].cells = ("Adult", "$26.00")
    layout.lines[1].cell_x = (50.0, 150.0)
    layout.lines[2].cells = ("Child", "$15.50")
    layout.lines[2].cell_x = (50.0, 150.0)
    blocks = blocks_for(layout, set())
    kinds = [b["kind"] for b in blocks]
    assert "table" in kinds


def test_blocks_for_heading_by_size():
    layout = PageLayout(page_no=1, width=595.0, height=842.0)
    body = [_line("body text line", size=10.0) for _ in range(6)]
    layout.lines = [_line("SECTION A", size=14.0)] + body
    blocks = blocks_for(layout, set())
    assert blocks[0]["kind"] == "heading"
    assert blocks[1]["kind"] == "text"


def test_blocks_for_figure_flag():
    layout = PageLayout(page_no=1, width=595.0, height=842.0, figure_area=0.15)
    layout.lines = [_line("a line of text")]
    blocks = blocks_for(layout, set())
    assert any(b["kind"] == "figure" for b in blocks)


# ----------------------------------------------------------- markdown rendering


def test_render_markdown_question():
    blocks = [
        {"kind": "question", "number": 3, "text": "Calculate p.", "page_no": 1},
        {
            "kind": "table",
            "rows": [["Adult", "$26"], ["Child", "$15"]],
            "page_no": 1,
        },
    ]
    md = render_markdown(blocks)
    assert "### Q3. Calculate p." in md
    assert "| Adult | $26 |" in md
    # A Markdown table needs its separator directly under the header and no
    # blank line between rows.
    table = md.split("\n\n")[1]
    assert table.splitlines()[0] == "| Adult | $26 |"
    assert table.splitlines()[1] == "| --- | --- |"
    assert table.splitlines()[2] == "| Child | $15 |"


def test_blocks_for_cover_page_has_no_invented_question():
    layout = PageLayout(page_no=1, width=595.0, height=842.0)
    layout.lines = [
        _line("Cambridge Ordinary Level", x0=50),
        _line("MATHEMATICS (SYLLABUS D) 4024/22", x0=50),
        _line("Paper 2 May/June 2019", x0=50),
        _line("2 hours 30 minutes", x0=50),
        _line("READ THESE INSTRUCTIONS FIRST", x0=50),
        _line("Answer all questions.", x0=50),
    ]
    blocks = blocks_for(layout, set())
    assert [b["kind"] for b in blocks] == ["text"] * len(layout.lines)
    assert not any(b["kind"] == "question" for b in blocks)


def test_render_markdown_figure():
    blocks = [{"kind": "figure", "area": 0.12, "page_no": 1}]
    md = render_markdown(blocks)
    assert "Figure on this page" in md


# ----------------------------------------------------------- needs_vision


def test_needs_vision_sparse_text():
    page = PageResult(page_no=1, markdown="hi", source="native")
    assert needs_vision(page, "auto") is True


def test_needs_vision_off():
    page = PageResult(page_no=1, markdown="", source="native")
    assert needs_vision(page, "off") is False


def test_needs_vision_figure_threshold():
    rich = PageResult(
        page_no=1, markdown="x" * 500, source="native", flags={"figure_area": 0.05}
    )
    assert needs_vision(rich, "auto") is False  # below 10%
    assert needs_vision(rich, "all") is True  # above 2%
    assert needs_vision(rich, "off") is False


# ----------------------------------------------------------- external parser import


def test_parse_docling_json_minimal():
    data = {
        "pages": [{"page_no": 1}],
        "texts": [
            {
                "text": "Hello world",
                "prov": [{"page_no": 1, "bbox": {"t": 100, "l": 50}, "coord_origin": "top-left"}],
            }
        ],
    }
    results = parse_docling_json(data)
    assert results is not None
    assert 1 in results
    assert results[1].source == "docling"


def test_parse_docling_json_table():
    data = {
        "pages": [{"page_no": 1}],
        "tables": [
            {
                "text": "",
                "prov": [{"page_no": 1, "bbox": {}, "coord_origin": "top-left"}],
                "data": {
                    "grid": [
                        {"cells": [{"text": "A"}, {"text": "1"}]},
                        {"cells": [{"text": "B"}, {"text": "2"}]},
                    ]
                },
            }
        ],
    }
    results = parse_docling_json(data)
    assert results is not None
    table = [b for b in results[1].blocks if b["kind"] == "table"][0]
    assert table["rows"] == [["A", "1"], ["B", "2"]]


def test_parse_mineru_content_list():
    data = [
        {"page_idx": 0, "type": "heading", "text": "Section A"},
        {"page_idx": 0, "type": "table", "text": "", "table_body": "<tr><td>A</td><td>1</td></tr>"},
        {"page_idx": 1, "type": "text", "text": "page two body"},
    ]
    results = parse_mineru_content_list(data)
    assert results is not None
    assert 1 in results and 2 in results
    assert results[1].source == "mineru"
    assert results[1].blocks[0]["kind"] == "heading"
    assert results[2].blocks[0]["kind"] == "text"


def test_merge_pages_prefers_external_when_richer():
    native = {
        1: PageResult(page_no=1, markdown="short", source="native", blocks=[]),
        2: PageResult(page_no=2, markdown="x" * 500, source="native", blocks=[]),
    }
    external = {
        1: PageResult(
            page_no=1,
            markdown="x" * 1000,
            source="docling",
            blocks=[{"kind": "table", "rows": [["a", "b"]]}],
        ),
        2: PageResult(page_no=2, markdown="tiny", source="docling", blocks=[]),
    }
    merged = merge_pages(native, external)
    assert merged[1].source == "docling"  # richer
    assert merged[2].source == "native"  # external not richer


# ----------------------------------------------------------- real PDF integration


@pytest.mark.skipif(not PDF.exists(), reason="test.pdf not present")
def test_read_page_real_pdf():
    with LayoutDocument(str(PDF)) as doc:
        result = doc.native_page(2)
    assert result.source == "native"
    assert result.chars > 0
    assert result.flags["chars"] > 0


@pytest.mark.skipif(not PDF.exists(), reason="test.pdf not present")
def test_needs_vision_blank_page():
    with LayoutDocument(str(PDF)) as doc:
        # Page 78 is a blank page in test.pdf -> sparse -> vision needed.
        result = doc.native_page(78)
    assert needs_vision(result, "auto") is True
