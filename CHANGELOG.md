# Changelog

All notable changes to AIO Local are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-20

A memory redesign. The flat top-k index is still present as a fallback, but long-term
memory now uses an append-only tree that never prunes and can be back-traced from a
question to exact source units. Documents are stored as ordered pages with incremental
notes, and exhaustive document tasks run as resumable background jobs. No new model or
external service is required; existing databases are upgraded in place.

### Added

- **Append-only memory tree.** Every three complete units roll up into one summary
  (chunks to degree-1, three degree-1 nodes to degree-2, and so on). Nodes and chunks
  are immutable; a consumed unit keeps its row and its parent link forever.
- **Beam back-trace retrieval.** A query is scored across several branches at each
  level and descends to the exact chunk, returning the provenance path. Chat trees and
  document trees are separate, so one cannot crowd out the other.
- **Per-chat document opt-in.** A chat only sees documents the user attaches in the
  chat menu. Document summaries and page excerpts are filtered at retrieval time, and
  the Memory view lists the same document scope the model received.
- **Page-addressed document memory.** A document is an ordered array of pages. Each
  page note is appended with the previous note as context instead of rewriting the
  whole document, and the overview is a roll-up of tree roots plus uncovered page
  notes, giving every page exactly one place in the summary.
- **Direct page queries.** "page 12", "pages 3-5" and inline forms resolve exact page
  text, including several ranges in one request (capped at 200 pages).
- **Resumable long tasks.** Exhaustive document jobs ("categorise every question")
  route to a background worker that appends draft sections, stores the draft, keeps
  only the unfinished segment in context, and continues from a persisted cursor after
  pause, failure or restart. A final pass makes the accumulated draft presentable.
- **Document pause.** Analyse jobs can be paused as well as cancelled, and resume
  contiguously from the last completed page.
- **Tree and task views.** The Memory view can browse tree outlines, node relevance and
  provenance; the chat shows task progress with a draft viewer and pause/resume/cancel.
- **Startup backfill.** Conversations and documents created before this release enter
  their trees once, in the background, without blocking startup.

### Changed

- **Roll-ups never invent text.** If the model returns nothing, the chunks stay
  unconsumed and a later drain retries. When a model is unavailable entirely, a
  deterministic lossless digest preserves wording and provenance instead of guessing.
- **Tree usage is recorded separately.** Retrieval feedback updates `tree_node_usage`,
  so relevance can grow without rewriting a node.
- **OVMS stops with the app.** The app-owned OVMS process is held in a Windows job
  object with kill-on-close, plus a PID file, `atexit` and console-control handlers.
  An external OVMS the app did not start is left running.
- **Document analysis spends its time on facts, not formatting.** Each page is read
  with streaming already off and thinking disabled; the prompt now asks for compact,
  non-repetitive notes and no longer generates a redundant per-page "Visible points"
  list. The PDF is opened once per job instead of once per page. Output is unchanged
  in substance - every visible fact is still transcribed - with less generation per
  page, so large documents finish sooner.

### Fixed

- A failed task segment no longer advances the cursor or silently skips work; retry
  continues from the same unit and appends to the existing draft.
- A paused task no longer restarts from the wrong ordinal; draft sections are numbered
  by what was written, not by cursor position.
- Document overviews no longer re-summarise the whole document (the main source of
  invented content) and partial jobs report honestly which pages are missing.
- Restored document jobs now wait until OVMS has started before resuming, so a launch
  against a stopped model server produces one clean failure instead of a race on every
  start.

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
- **The chat dropdown now controls the whole app.** Choosing a model saves `chat_model`,
  updates the header pill immediately, rewrites `opencode.json`, and is used by chat,
  document analysis, memory summaries, images and OpenCode. If the chosen model is installed
  but not loaded, the app loads it for you instead of asking for a manual unload/load.
- **Removed custom models stay removed.** Removing a model clears its role references in
  `chat_model`, `filing_model`, `embedding_model`, `image_model` and `image_edit_model`,
  drops it from auto-load, regenerates the OVMS and OpenCode configs, and records the folder
  as ignored so a restart or status poll cannot re-add it. The explicit **Scan models folder**
  button is still able to bring it back. Delete an OpenVINO folder by hand and the next start
  repairs the stale references automatically.
- **Missing custom folders are visible and removable.** Custom models whose folder was moved
  or deleted are flagged "folder missing" with a **Remove from list** button (pending models
  have one in the role prompt too), and are no longer offered in the chat dropdown or in
  pending-role prompts.
- **A stale or unloaded vision model can no longer fail every document job.** Document
  analysis only uses an assigned, installed custom model, prefers a vision model OVMS is
  actually serving, and turns the opaque
  `Mediapipe graph definition with requested name is not found` response into an actionable
  "load it in Models or pick a loaded model in Chat" error. The same translation applies to
  chat and OpenCode requests through the gateway.
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
  demand; removing one keeps the folder on disk and persists the removal.

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

### Security

- **API errors no longer echo exception text.** Internal failures are logged with
  a traceback and every endpoint returns a short, curated message, so file
  paths, library details and stack traces cannot leak to the browser
  (`py/stack-trace-exposure`, CWE-209).
- **DuckDuckGo redirect checks parse the host.** The result unwrapper compares
  the URL hostname (`duckduckgo.com` or `*.duckduckgo.com`) instead of using a
  substring test on the netloc, closing
  `py/incomplete-url-substring-sanitization` (CWE-20).

### Validation

- Full test suite: 145 passed.
- Corpus check on the reference library (385 documents, 6,072 chunks, 56
  folders): folder hit@3 100%, chunk hit@5 100%, negative control 100%.

## [0.2.0] - 2026-11-10

First public release: local chat through OVMS, exact and summary memory, the
Knowledge library, image generation, model management, the GIMP dock, and a
Linux source-build path.

[0.3.0]: https://github.com/Mohammad-Abdur-Rahman/AIOLocal/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Mohammad-Abdur-Rahman/AIOLocal/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Mohammad-Abdur-Rahman/AIOLocal/releases/tag/v0.2.0
