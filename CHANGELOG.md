# Changelog

All notable changes to AIO Local are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-11-18

A bug-fix release for memory deletion, document storage, knowledge folders and
automatic routing. No new model or external service is required.

### Fixed

- **Memory deletion now cascades.** Deleting a memory also removes its summaries,
  topic links, raw messages, FTS rows and vectors, and prunes topics that become
  empty. A forgotten conversation can no longer reappear through derived data.
- **Every PDF page is stored.** Documents keep all pages with no storage cap;
  cancelling a job no longer discards finished pages or the uploaded PDF, and
  interrupted jobs resume from the last completed page.
- **Document memory is reliable in chat.** Pages are grouped into sections (about
  8 pages each, about 80 pages per part) with one durable summary row, and the
  document body is guaranteed a context budget instead of being trimmed away.
- **Document jobs are always one page per vision call.** The saved settings can no longer
  reintroduce three-page batches or a page cap; an old `app.json` is migrated to
  `document_pages_per_batch: 1` and `document_max_pages: 0` on first load.
- **Cancelled document jobs can be dismissed.** The progress bar now shows a **Hide** button
  for cancelled, failed, interrupted and completed jobs.
- **The memory badge matches the Memory view.** It reports "memory items" with a tooltip and
  inspector breakdown for summary rows, page/section excerpts and history, so "5 items" next
  to two summary rows is explained rather than looking like a counting bug.
- **Nested knowledge folders work end to end.** Inbox folders map to nested
  collection slugs (`ai/papers`), missing parents are created automatically, and
  deleting a folder recursively removes child folders, documents, pages,
  sections, vectors and jobs.
- **Inbox scans no longer stop at 40 files.** Scans are unlimited by default,
  support `.mdx`, count only new ingests toward an optional limit, and report
  live progress.
- **Automatic routing finds the right folder.** Folder summaries now include a
  rarity-weighted term profile built from every chunk, and routing matches query
  terms with bounded morphology and inverse-folder-frequency weighting. For
  example, a pypdfium2 rendering question routes to `dev/pdfium` instead of an
  unrelated framework that merely mentions "page".
- **Weak matches can abstain.** Candidates are re-ranked by query-term coverage;
  unrelated questions return nothing instead of the least-bad chunk.
- **Folder search is bounded.** Vector candidates are prefiltered and fetched in
  batches, so a large knowledge base does not load every chunk vector for one
  query.

### Added

- `tools/kb_sync.py` and `tools/kb_manifest.json` install a curated Tier-1
  reference corpus with Git commit, source URL and licence metadata.
- `tools/kb_eval.py` and `tools/kb_eval_queries.json` verify folder routing,
  chunk retrieval and negative controls after a sync.
- Knowledge UI: nested folder tree with summaries and document counts, **Auto**
  routing, rebuild index, enrich now, worker status, scan progress and delete.
- Control endpoints for folder rebuild, immediate enrichment, automatic search,
  worker status and optional scan limits.
- The gateway accepts `X-AIO-Knowledge: auto`, reports the chosen route in the
  plan, and wraps knowledge excerpts as untrusted reference data.
- **Custom OpenVINO models.** The Models view accepts a Hugging Face repository URL
  or a folder copied into the models directory. Only OpenVINO IR layouts are
  registered; ONNX, GGUF and SafeTensors folders are ignored with a reason. A new
  folder stays pending until a role is chosen (chat, memory summary, embedding,
  image generation or image editing), then settings, OVMS config, dropdowns and
  `opencode.json` are updated automatically. Models are scanned at startup and on
  demand; removing one keeps the folder on disk.

### Changed

- PDF vision analysis now uses one page per model call for accuracy instead of
  batching three pages together.
- `document_max_pages` defaults to `0` and is no longer a cap; every PDF page is
  analysed.
- **Pillow is raised to 13.0.0** (`Pillow>=13,<14`), clearing the three
  Dependabot alerts for CVE-2026-112902, CVE-2026-128066 and CVE-2026-128067
  (TIFF libtiff encoder memory disclosure, TIFF scanline overflow, and TIFF/EXIF
  IFD memory amplification).
- Folder summaries are rebuilt automatically after a scan that adds documents.

### Validation

- Full test suite: 130 passed.
- Corpus check on the reference library (385 documents, 6,072 chunks, 56
  folders): folder hit@3 100%, chunk hit@5 100%, negative control 100%.

## [0.2.0] - 2026-11-10

First public release: local chat through OVMS, exact and summary memory, the
Knowledge library, image generation, model management, the GIMP dock, and a
Linux source-build path.

[0.2.1]: https://github.com/Mohammad-Abdur-Rahman/AIOLocal/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Mohammad-Abdur-Rahman/AIOLocal/releases/tag/v0.2.0
