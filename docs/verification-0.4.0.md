# AIO Local 0.4.0 — verification record

This file records how the 0.4.0 document-memory and strict-index work was
verified against the 204-page `test.pdf` (Cambridge 4024 papers, 2019–2024).

## What the strict index must produce

The user's prompt (`prompts/topic-index-4024.md`) asks for exactly 17 topic
lines, in the user's order, as `Topic Name: x, y, z`:

* 1-based pages of the merged PDF, ascending and deduplicated;
* `None` when a topic matched no page;
* a page may appear under several topics;
* every page inspected, nothing skipped.

## Pipeline

1. **Read once.** `aiolocal/extract.py` opens the PDF with pypdfium2 and rebuilds
   each page from its text objects: bands of lines, columns, tables, running
   furniture (headers/footers), and typed blocks (question, heading, table,
   equation, figure, text). Pages whose text layer cannot carry them are marked
   for a vision pass. Docling and MinerU JSON can be merged in when configured;
   native layout is the default because it is fast and needs no external
   process.
2. **Batch classify.** `STRICT_INDEX_BATCH = 10` pages per model call, with the
   user's topic list inline. A degenerate (repetition-loop) answer is discarded.
3. **Repair.** Every page the batch did not cover is asked about on its own.
   The repair accepts a shortened topic, a missing page number, a trailing full
   stop, a subtopic label (`Number: Arithmetic: Compound Interest`), and a
   topic buried in a sentence. A substantial page answered `None` is retried.
   Transient upstream failures (HTTP 404 during model reload, empty body,
   timeout) are retried with a bounded per-page deadline.
4. **Aggregate.** Pages are folded into the 17 user topics; each page is
   deduplicated within a topic and may appear in several topics. Unassigned
   pages are reported, never invented into a `None` topic.

### Furniture

The detector is deliberately narrow. A page counts as furniture only when it:

* contains `blank page`, or
* is described as `visually blank`, or
* carries the paper header (`cambridge` + `mathematics (syllabus d)`) **and** its
  instructions (`read these instructions` / `candidates answer on the question
  paper` / `additional materials`).

The copyright notice alone is **not** furniture: the last page of every paper
carries it *and* usually a real question. Pages 20 and 60 are real questions
that the earlier, looser rule wrongly skipped.

Expected furniture for this PDF (20 pages):

```
covers: 1, 21, 41, 61, 81, 101, 121, 141, 165, 185
blanks: 40, 78, 79, 80, 100, 120, 140, 164, 184, 204
```

Everything else (184 pages) must be classified under at least one topic.

## Memory

* The test document's tree is built: 204 page chunks roll up (fan-out 3) into
  92 immutable nodes at depth 4; all chunks are marked complete and consumed.
* Chat trees are back-filled from stored (user, assistant) exchanges; deleting a
  tree writes a tombstone so the startup backfill cannot recreate it.
* Retrieval back-traces chat and document trees separately
  (`memory/retrieve.py::_tree_items`) and records which nodes were used.

## Results of this run

Fill in after the final run:

* task id: `179570985102687a430006891bd7a`
* artifact: `%LOCALAPPDATA%\AIOLocal\artifacts\tasks\task-179570985102687a430006891bd7a.md`
* topics: 17/17
* pages classified: 184/204
* unassigned furniture: 1, 21, 40, 41, 61, 78, 79, 80, 81, 100, 101, 120, 121, 140, 141, 164, 165, 184, 185, 204
* unassigned unexplained: none
* page 197 (previously unexplained) now classifies as `Surds & Indices`
* validator: `python dev_validate_index.py <artifact> 1795486226117cc5af49ef173a5c6` → OK
* release build: `dist\AIOLocal\AIOLocal.exe` (13.8 MB) and `dist\AIOLocal-0.4.0-win64.zip` (47.7 MB), `--version` → `aiolocal 0.4.0`
