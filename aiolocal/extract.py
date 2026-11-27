"""Document extraction: read a PDF's real structure, not just its characters.

A plain ``get_text_range()`` call returns the right characters in roughly the
right order but throws away everything that makes a page readable: which line is
a heading, where a table starts, which block is a diagram, and whether the page
is one column or two. This module keeps that structure.

Three backends are supported, in increasing order of cost:

``native``
    Built in, no extra install. Each page is split into lines using the real
    character boxes from pdfium, then into blocks (question, heading, table,
    equation, figure, text). Page furniture (running headers, ``Page 3 of 584``)
    is detected across the document and removed. A page with a large vector
    drawing is flagged so the caller can ask the vision model about it.

``docling`` / ``mineru``
    Optional external parsers. When the user has installed one and pointed
    ``docling_path`` / ``mineru_path`` at it, its output is imported as per-page
    Markdown so tables, formulas and reading order come from a layout model.
    Both are best-effort: any failure falls back to ``native`` rather than
    failing the upload.

Memory-tree compatibility is deliberate: a page stays the atomic unit, so page
numbers keep their meaning for indexing and back-tracing. The improvement is
inside the page - structured Markdown instead of a wall of text - plus a block
list stored alongside it.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200
# A vector drawing covering this share of the page is worth a vision pass: the
# text layer says "the diagram shows..." but only the picture has the labels.
FIGURE_VISION_AUTO = 0.10
FIGURE_VISION_ALL = 0.02
# A chart is often many small bars, so the union bounding box and the object
# count catch diagrams the summed "large object" area misses.
FIGURE_VISION_UNION = 0.12
FIGURE_VISION_OBJECTS = 50
FIGURE_MIN_SIDE = 24.0
BAND_PAD = 2.0
# Horizontal gaps in glyph boxes: cell separators are wide (table columns),
# word gaps are smaller but still visibly larger than intra-word spacing.
CELL_GAP = 12.0
WORD_GAP = 3.0
# Within one page, question numbers run in order. A number that jumps far
# ahead of the previous one is an axis tick or a data value, not a question.
MAX_QUESTION_STEP = 6

MATH_CHARS = set("∫∑√≤≥×÷±πθ°∝⇒⇔∈∉⊂⊆∪∩≠≈∞−→←↔")

_QUESTION_RE = re.compile(r"^(\d{1,2})\s*[.)]?\s+(?=[A-Z([0-9])")
_QUESTION_ALONE_RE = re.compile(r"^(\d{1,2})\s*[.)]?$")
_PART_RE = re.compile(r"^\(([a-z])\)\s*")
_ROMAN_RE = re.compile(r"^\(([ivx]{1,4})\)\s*")
_PAGE_NO_RE = re.compile(r"^(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?$", re.IGNORECASE)
_DOTS_RE = re.compile(r"[.·]{8,}")
_WS_RE = re.compile(r"[ \t]+")


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    cells: tuple[str, ...] = ()
    cell_x: tuple[float, ...] = ()

    @property
    def top(self) -> float:
        return self.y1

    @property
    def bottom(self) -> float:
        return self.y0


@dataclass
class PageLayout:
    page_no: int
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    figure_area: float = 0.0
    # All vector drawing objects on the page: count and the union of their
    # bounding boxes. A chart made of many small bars has a tiny summed
    # "large object" area but a large object count, so both are tracked.
    figure_objects: int = 0
    figure_union: float = 0.0
    # True when the drawings are a decorative page border, not a diagram.
    figure_frame: bool = False

    @property
    def chars(self) -> int:
        return sum(len(line.text) for line in self.lines)

    @property
    def median_size(self) -> float:
        sizes = sorted(line.size for line in self.lines if line.size > 0)
        if not sizes:
            return 0.0
        return sizes[len(sizes) // 2]


@dataclass
class PageResult:
    page_no: int
    markdown: str
    source: str
    blocks: list[dict] = field(default_factory=list)
    flags: dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.markdown.strip())

    def layout_json(self) -> str:
        payload = {"blocks": self.blocks, "flags": self.flags, "source": self.source}
        try:
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return ""


# --------------------------------------------------------------------- geometry
def _bands(chars: list[tuple[float, float, float, float, str]]) -> list[dict]:
    """Group characters into visual lines by vertical overlap.

    PDF text is drawn as separate runs at slightly different baselines
    (superscripts, mixed fonts), so a fixed gap threshold splits one printed
    line into two or three. Overlap of the vertical extents is stable instead.
    """
    bands: list[dict] = []
    for char in sorted(chars, key=lambda c: -c[3]):
        placed = False
        for band in bands:
            overlap = min(band["top"], char[3]) - max(band["bottom"], char[1])
            shortest = min(band["top"] - band["bottom"], char[3] - char[1])
            if overlap > 0.35 * shortest:
                band["top"] = max(band["top"], char[3])
                band["bottom"] = min(band["bottom"], char[1])
                band["chars"].append(char)
                placed = True
                break
        if not placed:
            bands.append({"top": char[3], "bottom": char[1], "chars": [char]})
    bands.sort(key=lambda b: (-b["top"], b["bottom"]))
    return bands


def _cells(
    chars: list[tuple[float, float, float, float, str]]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Split a line into cells at wide horizontal gaps (table columns).

    Space glyphs in a PDF often carry a zero-height box, so they are absent
    from the character list. Word spaces are therefore re-inserted from the
    horizontal gap between neighbouring glyphs: a gap wider than the cell
    separator (``CELL_GAP``) starts a new column, a smaller-but-meaningful gap
    (``WORD_GAP``) becomes a space, and anything tighter is inside a word.
    """
    chars = sorted(chars, key=lambda c: c[0])
    texts: list[str] = []
    starts: list[float] = []
    current: list[str] = []
    current_x = 0.0
    last_x1 = None
    last_size = 0.0
    for x0, y0, x1, y1, ch in chars:
        size = y1 - y0
        if last_x1 is not None:
            gap = x0 - last_x1
            if gap >= CELL_GAP and current:
                text = _WS_RE.sub(" ", "".join(current)).strip()
                if text:
                    texts.append(text)
                    starts.append(current_x)
                current = []
                current_x = x0
            elif gap >= max(WORD_GAP, 0.30 * (last_size or size)) and current:
                current.append(" ")
        if not current:
            current_x = x0
        current.append(ch)
        last_x1 = x1 if last_x1 is None else max(last_x1, x1)
        last_size = size
    text = _WS_RE.sub(" ", "".join(current)).strip()
    if text:
        texts.append(text)
        starts.append(current_x)
    return tuple(texts), tuple(starts)


def read_page(pdf, page_no: int) -> PageLayout:
    """Read one page: lines with geometry, cell splits, and figure coverage."""
    page = pdf[page_no - 1]
    width, height = page.get_size()
    layout = PageLayout(page_no=page_no, width=width, height=height)
    textpage = None
    try:
        textpage = page.get_textpage()
        chars: list[tuple[float, float, float, float, str]] = []
        try:
            count = textpage.count_chars()
        except Exception:
            count = 0
        for i in range(count):
            try:
                x0, y0, x1, y1 = textpage.get_charbox(i)
            except Exception:
                continue
            # Zero-height boxes are hyphenation artifacts and page markers; they
            # duplicate a real line and would split it in two bands.
            if x1 - x0 <= 0 or y1 - y0 < 1.0:
                continue
            try:
                ch = textpage.get_text_range(i, 1)
            except Exception:
                ch = ""
            chars.append((x0, y0, x1, y1, ch))
        for band in _bands(chars):
            band_chars = band["chars"]
            xs0 = min(c[0] for c in band_chars)
            xs1 = max(c[2] for c in band_chars)
            try:
                raw = textpage.get_text_bounded(
                    left=max(0.0, xs0 - BAND_PAD),
                    bottom=max(0.0, band["bottom"] - BAND_PAD),
                    right=min(width, xs1 + BAND_PAD),
                    top=min(height, band["top"] + BAND_PAD),
                )
            except Exception:
                raw = "".join(c[4] for c in sorted(band_chars, key=lambda c: c[0]))
            text = _WS_RE.sub(" ", (raw or "").replace("\r", " ").replace("\n", " ")).strip()
            if not text:
                continue
            cells, cell_x = _cells(band_chars)
            layout.lines.append(
                Line(
                    text=text,
                    x0=xs0,
                    y0=band["bottom"],
                    x1=xs1,
                    y1=band["top"],
                    size=max(c[3] - c[1] for c in band_chars),
                    cells=cells,
                    cell_x=cell_x,
                )
            )
    finally:
        if textpage is not None:
            try:
                textpage.close()
            except Exception:
                pass
        # Vector drawings are the figures in a born-digital exam paper: graphs,
        # triangles, histograms, circle diagrams. A chart is often many small
        # bars (a tiny summed "large object" area), so the object count and the
        # union bounding box matter as much as the summed area. A decorative
        # page border is a thin frame of many segments around the whole page and
        # is detected so it is not mistaken for a diagram.
        try:
            large_area = 0.0
            boxes: list[tuple[float, float, float, float]] = []
            for obj in page.get_objects():
                try:
                    if obj.type != 2:
                        continue
                    bx0, by0, bx1, by1 = obj.get_bounds()
                except Exception:
                    continue
                boxes.append((bx0, by0, bx1, by1))
                if bx1 - bx0 >= FIGURE_MIN_SIDE and by1 - by0 >= FIGURE_MIN_SIDE:
                    large_area += (bx1 - bx0) * (by1 - by0)
            if width * height > 0:
                layout.figure_area = min(1.0, large_area / (width * height))
            layout.figure_objects = len(boxes)
            if boxes and width > 0 and height > 0:
                ux0 = min(b[0] for b in boxes)
                uy0 = min(b[1] for b in boxes)
                ux1 = max(b[2] for b in boxes)
                uy1 = max(b[3] for b in boxes)
                wfrac = (ux1 - ux0) / width
                hfrac = (uy1 - uy0) / height
                layout.figure_union = max(0.0, min(1.0, wfrac * hfrac))
                layout.figure_frame = (
                    wfrac >= 0.80 and hfrac >= 0.80 and len(boxes) >= 120
                )
        except Exception:
            layout.figure_area = 0.0
            layout.figure_objects = 0
            layout.figure_union = 0.0
            layout.figure_frame = False
        try:
            page.close()
        except Exception:
            pass
    return layout

# ------------------------------------------------------------------ furniture
def _signature(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\d+", "#", lowered)
    lowered = re.sub(r"[^a-z#]+", " ", lowered).strip()
    return lowered[:60]


def _core(text: str) -> str:
    return " ".join(_signature(text).split()[:3])


def furniture(lines_by_page: dict[int, list[Line]]) -> set[str]:
    """Running headers/footers, found by repetition across pages."""
    pages = len(lines_by_page)
    if pages < 3:
        return set()
    counts: dict[str, int] = {}
    cores: dict[str, int] = {}
    for lines in lines_by_page.values():
        seen: set[str] = set()
        seen_core: set[str] = set()
        for line in lines:
            if len(line.text) > 90:
                continue
            sig = _signature(line.text)
            if sig and sig not in seen:
                seen.add(sig)
                counts[sig] = counts.get(sig, 0) + 1
            core = _core(line.text)
            if core and core not in seen_core:
                seen_core.add(core)
                cores[core] = cores.get(core, 0) + 1
    threshold = max(3, int(pages * 0.35))
    found = {sig for sig, n in counts.items() if n >= threshold and len(sig) >= 4}
    found |= {core for core, n in cores.items() if n >= threshold and len(core) >= 4}
    return found


def is_furniture(line: Line, running: set[str], layout: PageLayout) -> bool:
    if _PAGE_NO_RE.match(line.text.strip()):
        # Bare numbers are furniture only in the top/bottom margins; a question
        # number can legitimately be a lone "12" in the body.
        return line.y1 > layout.height * 0.91 or line.y0 < layout.height * 0.09
    if _signature(line.text) in running or _core(line.text) in running:
        return True
    return False


# ------------------------------------------------------------------ structure
def _question_margin(lines: list[Line]) -> float:
    """The left edge where body text starts; question numbers sit there.

    Furniture has already been removed, so the leftmost remaining line is the
    real page margin. Questions align with it; indented body text starts a
    little to the right.
    """
    if not lines:
        return 0.0
    return float(min(line.x0 for line in lines))


def _is_question(line: Line, width: float, margin: float, median: float = 0.0) -> int:
    text = line.text.strip()
    if abs(line.x0 - margin) > 3:
        return 0
    # A question number is printed at body size. A tiny lone glyph in the
    # margin (a border artifact or a registration mark) is not a question.
    if median > 0 and line.size < median * 0.70:
        return 0
    if line.x1 - line.x0 > width * 0.85 and len(line.cells) <= 1:
        return 0
    match = _QUESTION_RE.match(text) or _QUESTION_ALONE_RE.match(text)
    if not match:
        return 0
    number = int(match.group(1))
    if number < 1 or number > 99:
        return 0
    rest = text[match.end() :].strip()
    # A line of pure data ("26.00") is not a question heading.
    if rest and not re.search(r"[A-Za-z(]", rest):
        return 0
    # A lone number with no text is a question only when it is a short line
    # in the number column, not a page-wide data row.
    if not rest and line.x1 - line.x0 > width * 0.5:
        return 0
    return number


def _columns(lines: list[Line], width: float) -> list[Line]:
    """Return lines in column order when the page has a clean two-column gutter."""
    if len(lines) < 8:
        return lines
    for guess in (0.45, 0.5, 0.55):
        gutter = width * guess
        left = [ln for ln in lines if ln.x1 <= gutter + 2]
        right = [ln for ln in lines if ln.x0 >= gutter - 2]
        crossing = len(lines) - len(left) - len(right)
        if len(left) >= 0.3 * len(lines) and len(right) >= 0.3 * len(lines) and crossing <= 0.05 * len(lines):
            return left + right
    return lines


def _table_run(lines: list[Line], start: int) -> int:
    """Length of the table starting at ``start`` (0 when it is not one)."""
    first = lines[start]
    if len(first.cells) < 2 or len(first.cell_x) != len(first.cells):
        return 0
    width = len(first.cells)
    anchor = first.cell_x
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if len(line.cells) != width or len(line.cell_x) != width:
            break
        # Every cell but the last must start at the same x as the first row:
        # that is what makes it a column, not a sentence with a wide space.
        if any(abs(line.cell_x[i] - anchor[i]) > 10 for i in range(width - 1)):
            break
        end += 1
    return end - start if end - start >= 2 else 0


def _math_density(text: str) -> float:
    if not text:
        return 0.0
    marks = sum(1 for ch in text if ch in MATH_CHARS)
    return marks / max(1, len(text))


def _filter_question_sequence(blocks: list[dict]) -> None:
    """Remove question numbers that break the page's increasing sequence.

    An exam page lists questions in ascending order, so a lone ``15`` that
    appears after ``4`` (a histogram axis tick, a data value) is dropped. A
    first number is always kept because there is nothing to compare it with.
    """
    positions = [i for i, b in enumerate(blocks) if b.get("kind") == "question"]
    keep: set[int] = set()
    last = 0
    for pos in positions:
        number = int(blocks[pos].get("number") or 0)
        if last and (number <= last or number > last + MAX_QUESTION_STEP):
            continue
        keep.add(pos)
        last = number
    if len(keep) == len(positions):
        return
    filtered = [b for i, b in enumerate(blocks) if i in keep or b.get("kind") != "question"]
    blocks[:] = filtered


def _is_cover_page(lines: list[Line]) -> bool:
    """True for an exam cover/instructions page, which has no questions.

    A cover page carries the paper code, the time allowed and the rubric. Its
    header box and "2 hours 30 minutes" line look like a question to the
    geometry heuristics, so the page is kept as plain text instead.
    """
    text = " ".join(line.text for line in lines).lower()
    if "read these instructions first" in text:
        return True
    return (
        "cambridge" in text
        and "mathematics" in text
        and "paper" in text
        and ("time" in text or "hours" in text)
    )


def blocks_for(layout: PageLayout, running: set[str]) -> list[dict]:
    """Turn a page's lines into typed blocks in reading order."""
    body = [ln for ln in layout.lines if not is_furniture(ln, running, layout)]
    body = [ln for ln in body if re.search(r"\w", ln.text)]
    if not body:
        return []
    if _is_cover_page(body):
        # No question numbers, tables or headings: a cover page is metadata.
        return [
            {"kind": "text", "text": line.text, "page_no": layout.page_no}
            for line in body
        ]
    margin = _question_margin(body)
    ordered = _columns(body, layout.width)
    median = layout.median_size or 10.0
    blocks: list[dict] = []
    index = 0
    while index < len(ordered):
        line = ordered[index]
        span = _table_run(ordered, index)
        if span:
            rows = [ln.cells for ln in ordered[index : index + span]]
            blocks.append(
                {
                    "kind": "table",
                    "rows": [list(row) for row in rows],
                    "page_no": layout.page_no,
                }
            )
            index += span
            continue
        question = _is_question(line, layout.width, margin, median)
        if question:
            rest = line.text.strip()
            rest = re.sub(r"^\d{1,2}\s*[.)]?\s*", "", rest).strip()
            blocks.append(
                {
                    "kind": "question",
                    "number": question,
                    "text": rest,
                    "page_no": layout.page_no,
                }
            )
            index += 1
            continue
        if line.size >= median * 1.18 and len(line.text) <= 90 and not (
            _PART_RE.match(line.text) or _ROMAN_RE.match(line.text)
        ):
            blocks.append(
                {
                    "kind": "heading",
                    "level": 2 if line.size >= median * 1.35 else 3,
                    "text": line.text,
                    "page_no": layout.page_no,
                }
            )
            index += 1
            continue
        kind = "equation" if _math_density(line.text) >= 0.12 and len(line.text) < 160 else "text"
        blocks.append({"kind": kind, "text": line.text, "page_no": layout.page_no})
        index += 1
    _filter_question_sequence(blocks)
    if not layout.figure_frame and (
        layout.figure_union >= 0.05 or layout.figure_area >= FIGURE_VISION_ALL
    ):
        blocks.append(
            {
                "kind": "figure",
                "area": round(layout.figure_area, 3),
                "page_no": layout.page_no,
            }
        )
    return blocks


def _table_markdown(rows: list[list[str]]) -> list[str]:
    cleaned = [[cell.replace("|", "\\|").strip() for cell in row] for row in rows]
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    out = ["| " + " | ".join(padded[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in padded[1:]:
        out.append("| " + " | ".join(row) + " |")
    return out


def render_markdown(blocks: list[dict]) -> str:
    """Structured Markdown for one page; readable by a model and a human."""
    out: list[str] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "question":
            text = str(block.get("text") or "").strip()
            number = block.get("number")
            out.append(f"### Q{number}" + (f". {text}" if text else ""))
        elif kind == "heading":
            level = int(block.get("level") or 2)
            out.append("#" * max(3, min(level + 1, 6)) + " " + str(block.get("text") or ""))
        elif kind == "table":
            rows = block.get("rows") or []
            if rows:
                # One Markdown table is one block: the rows must stay adjacent.
                # Joining the page's blocks with blank lines would otherwise put
                # an empty line between every row and break the table.
                out.append("\n".join(_table_markdown(rows)))
        elif kind == "equation":
            out.append(str(block.get("text") or ""))
        elif kind == "text":
            out.append(str(block.get("text") or ""))
        elif kind == "figure":
            area = float(block.get("area") or 0)
            out.append(f"> Figure on this page (vector drawing, {area:.0%} of the page).")
    return "\n\n".join(part for part in out if part.strip())


def flags_for(layout: PageLayout, blocks: list[dict]) -> dict:
    kinds = [str(b.get("kind")) for b in blocks]
    return {
        "chars": layout.chars,
        "figure_area": round(layout.figure_area, 3),
        "figure_objects": layout.figure_objects,
        "figure_union": round(layout.figure_union, 3),
        "figure_frame": layout.figure_frame,
        "questions": [int(b["number"]) for b in blocks if b.get("kind") == "question"],
        "tables": kinds.count("table"),
        "equations": kinds.count("equation"),
        "headings": kinds.count("heading"),
        "lines": len(layout.lines),
    }


# ---------------------------------------------------------------------- native
class LayoutDocument:
    """A PDF opened once for structured, page-by-page reading."""

    def __init__(self, path: str) -> None:
        import pypdfium2 as pdfium

        self.path = str(path)
        self.pdf = pdfium.PdfDocument(self.path)
        self.page_count = len(self.pdf)
        self._layouts: dict[int, PageLayout] = {}
        self._running: set[str] | None = None

    def layout(self, page_no: int) -> PageLayout:
        cached = self._layouts.get(page_no)
        if cached is None:
            cached = read_page(self.pdf, page_no)
            self._layouts[page_no] = cached
        return cached

    def running_furniture(self) -> set[str]:
        """Sample the document, not every page, to find repeated headers."""
        if self._running is not None:
            return self._running
        sample: dict[int, list[Line]] = {}
        step = max(1, self.page_count // 24)
        for page_no in range(1, self.page_count + 1, step):
            sample[page_no] = self.layout(page_no).lines
        if self.page_count > 2:
            for page_no in (1, self.page_count):
                sample[page_no] = self.layout(page_no).lines
        self._running = furniture(sample)
        return self._running

    def native_page(self, page_no: int) -> PageResult:
        layout = self.layout(page_no)
        running = self.running_furniture()
        blocks = blocks_for(layout, running)
        markdown = render_markdown(blocks)
        if not markdown.strip() and layout.lines:
            # No confidence in the structure: fall back to the ordered text so
            # nothing is silently lost.
            markdown = "\n".join(ln.text for ln in layout.lines if not is_furniture(ln, running, layout))
        return PageResult(
            page_no=page_no,
            markdown=markdown,
            source="native",
            blocks=blocks,
            flags=flags_for(layout, blocks),
        )

    def page_results(self) -> dict[int, PageResult]:
        return {page_no: self.native_page(page_no) for page_no in range(1, self.page_count + 1)}

    def close(self) -> None:
        try:
            self.pdf.close()
        except Exception:
            pass

    def __enter__(self) -> "LayoutDocument":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------ external parsers
def _grid_to_rows(grid) -> list[list[str]]:
    rows: list[list[str]] = []
    if not isinstance(grid, list):
        return rows
    for row in grid:
        cells: list[str] = []
        if isinstance(row, dict):
            row = row.get("cells") or []
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, dict):
                text = cell.get("text") or cell.get("content") or ""
            else:
                text = str(cell or "")
            cells.append(str(text).strip())
        if any(cells):
            rows.append(cells)
    return rows


def _docling_items(data: dict) -> list[dict]:
    items: list[dict] = []
    for key in ("texts", "tables", "pictures", "key_value_items", "form_items"):
        for item in data.get(key) or []:
            if not isinstance(item, dict):
                continue
            prov = item.get("prov") or []
            if not isinstance(prov, list) or not prov:
                continue
            ref = prov[0] if isinstance(prov[0], dict) else {}
            try:
                page_no = int(ref.get("page_no") or 0)
            except (TypeError, ValueError):
                continue
            if page_no <= 0:
                continue
            bbox = ref.get("bbox") if isinstance(ref.get("bbox"), dict) else {}
            try:
                top = float(bbox.get("t", bbox.get("top", 0)))
                left = float(bbox.get("l", bbox.get("left", 0)))
            except (TypeError, ValueError):
                top, left = 0.0, 0.0
            origin = str(ref.get("coord_origin") or data.get("origin") or "").upper()
            label = str(item.get("label") or key[:-1])
            text = item.get("text") or item.get("orig") or ""
            rows = _grid_to_rows((item.get("data") or {}).get("grid")) if key == "tables" else []
            items.append(
                {
                    "page_no": page_no,
                    "label": label,
                    "text": str(text),
                    "rows": rows,
                    "top": top,
                    "left": left,
                    "top_origin": "TOPLEFT" in origin,
                    "level": item.get("level") or 0,
                }
            )
    return items


def parse_docling_json(data: dict) -> dict[int, PageResult] | None:
    """Convert a DoclingDocument JSON export into per-page Markdown."""
    if not isinstance(data, dict):
        return None
    page_numbers = [
        int(p.get("page_no") or 0) for p in (data.get("pages") or []) if isinstance(p, dict)
    ]
    page_numbers = [p for p in page_numbers if p > 0]
    shift = 1 if page_numbers and min(page_numbers) == 0 else 0
    items = _docling_items(data)
    if not items:
        return None
    grouped: dict[int, list[dict]] = {}
    for item in items:
        page_no = item["page_no"] + shift
        item["page_no"] = page_no
        grouped.setdefault(page_no, []).append(item)
    results: dict[int, PageResult] = {}
    for page_no, page_items in grouped.items():
        page_items.sort(
            key=lambda i: (i["top"] if i["top_origin"] else -i["top"], i["left"])
        )
        blocks: list[dict] = []
        for item in page_items:
            label = item["label"].lower()
            if label in ("section_header", "title"):
                blocks.append(
                    {"kind": "heading", "level": 2 + int(item.get("level") or 0), "text": item["text"]}
                )
            elif label == "table":
                rows = item["rows"]
                blocks.append({"kind": "table", "rows": rows} if rows else {"kind": "text", "text": item["text"]})
            elif label in ("picture", "figure", "chart"):
                blocks.append({"kind": "figure", "area": 0.0, "caption": item["text"]})
            elif label in ("formula", "equation"):
                blocks.append({"kind": "equation", "text": item["text"]})
            else:
                if item["text"].strip():
                    blocks.append({"kind": "text", "text": item["text"]})
        markdown = render_markdown(blocks)
        results[page_no] = PageResult(
            page_no=page_no,
            markdown=markdown,
            source="docling",
            blocks=blocks,
            flags={
                "tables": sum(1 for b in blocks if b["kind"] == "table"),
                "equations": sum(1 for b in blocks if b["kind"] == "equation"),
                "lines": len(blocks),
            },
        )
    return results or None


def parse_mineru_content_list(data) -> dict[int, PageResult] | None:
    """Convert MinerU's ``content_list.json`` (0-based ``page_idx``) to pages."""
    if not isinstance(data, list):
        return None
    grouped: dict[int, list[dict]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            page_index = int(entry.get("page_idx", entry.get("page_no", -1)))
        except (TypeError, ValueError):
            continue
        if page_index < 0:
            continue
        page_no = page_index + 1
        kind = str(entry.get("type") or "text").lower()
        text = str(entry.get("text") or "").strip()
        blocks = grouped.setdefault(page_no, [])
        if kind in ("title", "heading"):
            if text:
                blocks.append({"kind": "heading", "level": 2, "text": text})
        elif kind == "table":
            body = entry.get("table_body") or ""
            rows = _html_table(str(body)) or _grid_to_rows(entry.get("table_cells"))
            if rows:
                blocks.append({"kind": "table", "rows": rows})
            elif text:
                blocks.append({"kind": "text", "text": text})
        elif kind in ("image", "figure"):
            caption = entry.get("img_caption") or entry.get("image_caption") or []
            if isinstance(caption, list):
                caption = " ".join(str(c) for c in caption)
            blocks.append({"kind": "figure", "area": 0.0, "caption": str(caption)})
        elif kind in ("equation", "interline_equation", "formula"):
            if text:
                blocks.append({"kind": "equation", "text": text})
        elif text:
            blocks.append({"kind": "text", "text": text})
    if not grouped:
        return None
    results: dict[int, PageResult] = {}
    for page_no, blocks in grouped.items():
        markdown = render_markdown(blocks)
        results[page_no] = PageResult(
            page_no=page_no,
            markdown=markdown,
            source="mineru",
            blocks=blocks,
            flags={
                "tables": sum(1 for b in blocks if b["kind"] == "table"),
                "equations": sum(1 for b in blocks if b["kind"] == "equation"),
                "lines": len(blocks),
            },
        )
    return results


def _html_table(html: str) -> list[list[str]]:
    if "<tr" not in html.lower():
        return []
    rows: list[list[str]] = []
    for raw_row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", html):
        cells = [
            re.sub(r"(?is)<[^>]+>", "", cell).strip()
            for cell in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", raw_row)
        ]
        if any(cells):
            rows.append(cells)
    return rows


# ------------------------------------------------------------------ processes
def _run_cli(args: list[str], *, timeout: int) -> bool:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            creationflags=0x08000000 if __import__("os").name == "nt" else 0,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _find_json(root: Path, wanted: str) -> Path | None:
    matches = [p for p in root.rglob("*.json") if wanted in p.name.lower()]
    if not matches:
        matches = list(root.rglob("*.json"))
    return matches[0] if matches else None


def docling_pages(path: str, executable: str) -> dict[int, PageResult] | None:
    """Run the Docling CLI and import its JSON, or None if it is unusable."""
    exe = shutil.which(executable) or (executable if Path(executable).expanduser().is_file() else "")
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="aiolocal-docling-") as tmp:
        out = Path(tmp)
        ok = _run_cli([exe, str(path), "--to", "json", "--output", str(out)], timeout=3600)
        candidate = _find_json(out, "json")
        if not ok or candidate is None:
            return None
        try:
            data = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            return None
        return parse_docling_json(data)


def mineru_pages(path: str, executable: str) -> dict[int, PageResult] | None:
    """Run MinerU and import its ``content_list.json``, or None."""
    exe = shutil.which(executable) or (executable if Path(executable).expanduser().is_file() else "")
    if not exe:
        return None
    name = Path(exe).name.lower()
    with tempfile.TemporaryDirectory(prefix="aiolocal-mineru-") as tmp:
        out = Path(tmp)
        attempts = [[exe, "parse", str(path), "--output", str(out)]]
        if "kit" not in name:
            attempts.append([exe, "parse", str(path), "--output", str(out), "--pages", "all"])
        for args in attempts:
            if _run_cli(args, timeout=3600):
                break
        candidate = None
        for pattern in ("*content_list*.json", "*.json"):
            matches = list(out.rglob(pattern))
            if matches:
                candidate = matches[0]
                break
        if candidate is None:
            return None
        try:
            data = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            return None
        return parse_mineru_content_list(data)


# -------------------------------------------------------------------- choosing
def merge_pages(
    native: dict[int, PageResult], external: dict[int, PageResult] | None
) -> dict[int, PageResult]:
    """Use the external parser where it is clearly richer, native elsewhere."""
    if not external:
        return native
    merged = dict(native)
    for page_no, page in external.items():
        base = native.get(page_no)
        if base is None:
            merged[page_no] = page
            continue
        structured = any(b.get("kind") in ("table", "figure") for b in page.blocks)
        if page.chars >= base.chars * 1.1 or (structured and page.chars >= base.chars * 0.8):
            page.flags = {**base.flags, **page.flags}
            merged[page_no] = page
    return merged


def extract_pages(
    path: str,
    *,
    parser: str = "native",
    docling_path: str = "",
    mineru_path: str = "",
) -> dict[int, PageResult] | None:
    """Read the whole PDF once, choosing backends according to settings."""
    parser = (parser or "native").strip().lower()
    try:
        doc = LayoutDocument(path)
    except Exception as exc:
        log.warning("layout reader could not open %s: %s", path, exc)
        return None
    try:
        native = doc.page_results()
        if parser in ("docling", "mineru", "hybrid", "auto"):
            external: dict[int, PageResult] | None = None
            if parser in ("docling", "hybrid", "auto") and docling_path:
                external = docling_pages(path, docling_path)
                if external:
                    log.info("docling read %d pages from %s", len(external), Path(path).name)
                else:
                    log.info("docling unavailable for %s; using native layout", Path(path).name)
            if external is None and parser in ("mineru", "hybrid", "auto") and mineru_path:
                external = mineru_pages(path, mineru_path)
                if external:
                    log.info("mineru read %d pages from %s", len(external), Path(path).name)
                else:
                    log.info("mineru unavailable for %s; using native layout", Path(path).name)
            if parser in ("docling", "mineru"):
                return external or native
            return merge_pages(native, external)
        return native
    finally:
        doc.close()


def needs_vision(page: PageResult, mode: str) -> bool:
    """Whether a page needs a picture pass (scans, charts, formatted tables).

    A born-digital page whose text layer is clean is kept on the native path.
    Vision is used when the text layer cannot carry the page:

    * the page is sparse (a scan or a blank page);
    * it is a chart/diagram: many vector objects, or drawings covering a real
      share of the page (but not a decorative full-page border);
    * ``all`` mode also sends pages with even small drawings.
    """
    mode = (mode or "auto").strip().lower()
    if mode in ("off", "false", "0", "no"):
        return False
    if page.chars < MIN_TEXT_CHARS:
        return True
    flags = page.flags or {}
    area = float(flags.get("figure_area") or 0.0)
    union = float(flags.get("figure_union") or 0.0)
    objects = int(flags.get("figure_objects") or 0)
    frame = bool(flags.get("figure_frame") or False)
    if mode in ("all", "true", "1", "yes"):
        if frame:
            return False
        return area >= FIGURE_VISION_ALL or union >= 0.02 or objects >= 8
    if frame:
        # A decorative border over a text page is not a diagram.
        return False
    return (
        area >= FIGURE_VISION_AUTO
        or union >= FIGURE_VISION_UNION
        or (objects >= FIGURE_VISION_OBJECTS and union >= 0.05)
    )
