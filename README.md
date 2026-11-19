# AIO Local

**A local, memory-first AI workspace for Intel laptops.** Chat, image generation, model
management, and a transparent long-term memory system — all on `127.0.0.1`, with no cloud calls.

This is **v0.2.1**, a bug-fix release on top of the first public release. It is a real,
working product: chat, memory, a knowledge/reference library, local image generation, a GIMP
dock, and a source-build path for Linux. What exists is listed below; what does not exist is
listed just as plainly.

---

## What's fixed in 0.2.1

- **Deleting a memory now removes everything derived from it.** Summaries, topic links, raw
  messages, FTS rows, vectors and now-empty topics are deleted in one cascade, so a deleted
  conversation cannot reappear through a summary or an embedding.
- **PDFs keep every page and can resume.** Each page is rendered and analysed separately, stored
  without a page cap, and survives cancel/resume. Long documents are grouped into sections
  (about 8 pages per section, about 80 pages per part) with one durable memory row, so chat can
  answer from any page instead of a truncated summary.
- **Knowledge folders are real folders.** Nested inbox folders map to nested collection slugs
  (`ai/papers`), parents are created automatically, and deleting a folder cascades to child
  folders, documents, pages, sections, vectors and jobs.
- **Automatic routing finds the right folder.** Folder summaries now carry a rarity-weighted
  term profile built from every chunk, query terms are matched with bounded morphology, and
  matches are weighted by inverse folder frequency. A pypdfium2 question routes to
  `dev/pdfium`, not to an unrelated framework that happens to mention "page".
- **Scans no longer stop at the first 40 files.** Inbox scans are unlimited by default, support
  `.mdx`, count only new files toward an optional limit, and report progress while they run.
- **Search can abstain.** Candidates are re-ranked by query-term coverage, and an unrelated
  question returns nothing instead of the least-bad chunk.
- **The reference corpus is reproducible.** `tools/kb_sync.py` installs the Tier-1 corpus with
  provenance and licence metadata; `tools/kb_eval.py` verifies routing, chunk retrieval and
  negative controls (385 documents / 6,072 chunks: 100% folder hit@3, 100% chunk hit@5,
  100% negative control on this machine).
- **Document jobs are one page per call, always.** The saved settings cannot reintroduce
  batching or a page cap; old `app.json` files are migrated once. After a job is cancelled or
  fails, the progress bar can be dismissed instead of staying on screen.
- **Memory counts match reality.** The chat badge now says "memory items" and its tooltip
  explains the split between summary rows and the exact page/section/history excerpts that
  were injected but are not listed as summaries in the Memory view.
- **Custom OpenVINO models can be imported.** Add a Hugging Face repository URL or copy an
  OpenVINO IR folder into the models directory. The app scans on startup, prompts for a role
  (chat, memory summary, embedding, image generation or image editing), then configures the
  matching settings, OVMS config, dropdowns and `opencode.json` exactly like a built-in model.
  Removing a model from the list keeps the folder on disk, clears every setting that pointed
  at it, and stays removed after a restart; a folder deleted by hand is flagged as missing and
  repaired at startup.
- **The Chat dropdown is the app-wide model switch.** It saves the selected model, updates
  the header pill, reloads `opencode.json`, and is used by chat, document analysis, memory
  summaries and OpenCode. A selected-but-unloaded model is loaded automatically.

---

## What works in v0.2

| Area | Status |
|---|---|
| Local server + browser UI | ✅ |
| Chat with streaming through any OpenAI-compatible upstream (OVMS by default) | ✅ |
| Exact memory DB (append-only, deduplicated, SQLite WAL) | ✅ |
| Summary memory + topics + many-to-many links | ✅ |
| Hybrid retrieval (BM25 + local hashed embeddings + reciprocal rank fusion) | ✅ |
| Optional OVMS semantic embeddings fused with hash retrieval, with re-embed job | ✅ |
| Optional LLM filing with deterministic heuristic fallback | ✅ |
| Token-budgeted prompt assembly with the documented trim order | ✅ |
| Memory Inspector (see exactly what the model saw) | ✅ |
| Model catalog, device choice, load/unload via OVMS config reload | ✅ |
| Custom OpenVINO IR import (Hugging Face URL or local folder) with role-based auto-configuration | ✅ |
| Image generation/edit proxy + local gallery | ✅ |
| Image attachments to a vision chat model (Qwen3-VL 4B) | ✅ |
| PDF chat attachments: every page rendered and analysed one page per vision call, grouped into section memory and resumable jobs | ✅ |
| Offline math rendering (bundled KaTeX) and PDF text extraction (pypdf) | ✅ |
| Local-only security: loopback bind, per-install token, origin/host checks, CSP | ✅ |
| Windows build (PyInstaller onedir) and Linux source build | ✅ |
| Independent Knowledge library (separate DB, FTS5 + vectors, nested folders, automatic routing, inbox, opt-in per message) | ✅ |
| Code workspace (OpenCode lifecycle + workspace preview) | ✅ |
| Reverse engineering (REA MCP stdio client, tool console, lifecycle) | ✅ |
| Embedded GIMP dock (Win32 host + embed + external fallback) | ✅ |
| GIMP diagnostics: log capture, plugin port/weight checks, model-list repair | ✅ |
| Optional PhotoGIMP installer from the official GPL-3.0 release (Windows) | ✅ |
| Audio / video | ⛔ planned (v2 — Stable Diffusion-class model for 16 GB) |

The UI groups Knowledge, Code, Reverse engineering and GIMP under **Workspace** because they are
real views. Only **Audio & video** is under **Planned modules**.

---

## Requirements

- Windows 10/11 x64, 8 GB RAM minimum (16 GB recommended for the 9B model).
- Linux x86_64 is supported as a **source build** (Python 3.11+). Only Windows binaries are published.
- **App bundle** includes server, UI, memory, KB, web search, image proxy, and the GIMP/REA/OpenCode lifecycle managers. It does **not** include OVMS/GIMP/REA/OpenCode binaries or model weights (those are GB-scale/externally licensed).
- **Automatic setup**: every launch checks the six runtime components (OVMS, portable Node.js, OpenCode, private Python, GIMP 3, Intel OpenVINO GIMP plugin). The Setup page downloads, verifies and unpacks anything missing into app-owned folders under the data directory, with per-component progress and logs. No system-wide install is required.
- **Automatic hosting**: the app can start/stop/reload its own server (`uvicorn`), manage the GIMP dock (`Win32 SetParent`), and launch REA (`npx -y rea-agents mcp`) when the user presses Start. Settings allow custom executable paths.
- **Specs-aware model notes**: the Models page shows licence tag, size (`~X GB`), install/load status, and device options for each model. The 9B chat model is recommended for 16 GB + GPU/NPU; the small filing model (`LFM2.5-1.2B`) fits easily alongside.
- **Models** are not bundled (several GB each). Pull them from the Models page; Stable Diffusion-class generation (planned v2) needs its own disk budget.

Without OVMS the app still starts, the UI works, and memory is fully testable — chat returns a clear
`UPSTREAM_UNAVAILABLE` message instead of a fake answer.

---

## Run from source

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"

# start (opens your browser)
python -m aiolocal

# or without opening a browser
python -m aiolocal --no-browser --port 7317
```

Then open <http://127.0.0.1:7317>.

### Linux source build

There is no Linux binary release. Build from source on Linux x86_64:

```bash
git clone <your-repository-url> aiolocal
cd aiolocal
chmod +x packaging/run-linux.sh
./packaging/run-linux.sh
```

Or manually:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m aiolocal --no-browser --port 7317
```

Install the external pieces with your distribution instead of the Windows Setup
downloaders:

- **OVMS**: install the Intel OpenVINO Model Server package or container, then point
  **Settings → Upstream URL** at it. The app does not ship a Linux OVMS binary.
- **Node / OpenCode**: install Node 20+ and OpenCode with your package manager; set
  their paths in Settings if they are not on `PATH`.
- **GIMP**: install GIMP 3. The Intel OpenVINO GIMP plugin is Windows-oriented; on Linux,
  install it manually into your own GIMP profile. GIMP launches in its own window on
  Linux (the Win32 dock is Windows-only).
- **PhotoGIMP**: use the official manual instructions from
  <https://github.com/Diolinux/PhotoGIMP>; the managed installer is Windows-only.

Data lives under `~/.local/share/AIOLocal/` (or `$XDG_DATA_HOME/AIOLocal`).

### First-launch setup

OVMS is required for local models and is **not bundled** (it is a separate, externally
licensed server). Open **Setup** on first launch and press **Install all**. The app downloads
and verifies its own copies of OVMS, Node.js, OpenCode, Python, GIMP and the Intel OpenVINO
GIMP plugin into `data\tools` under its data folder. You can also install or repair one
component at a time and read its full log from the same page.

Then:

1. Open **Models**, download a model, choose a device and press **Load**. Model weights are
   the only large downloads you choose explicitly; they are never bundled.
2. To add a model that is not in the catalog, paste a Hugging Face URL in **Custom OpenVINO
   models** or copy its folder into the models directory. Only OpenVINO IR layouts are
   accepted (`.xml` + `.bin` plus `config.json`, `openvino_config.json`, `model_index.json` or
   `graph.pbtxt`); ONNX, GGUF and SafeTensors are rejected. Choose its role and the app wires
   up Chat/OpenCode, memory summaries, embeddings, or image generation/editing.
3. Press **Start**. The app writes the OVMS config itself, reproduces OVMS's `setupvars`
   environment (its bundled Python backend needs `PYTHONHOME`/`PATH`), and shows a live
   startup status with the log tail if anything fails.
4. Model downloads show real byte progress; server startup shows an indeterminate progress
   bar plus the current stage (starting → loading model → ready).

The previous build wrote a config with a `task` key, which OVMS 2026 rejects, and launched
`ovms.exe` without its Python environment. Both are fixed: GenAI models are configured with
`graph_path`, and the generated graph is patched per device so GPU/CPU/NPU changes actually
take effect.

### Direct OpenVINO GenAI (no server)

OpenVINO GenAI does have an in-process Python API (`openvino_genai.LLMPipeline`) that needs
no OVMS server. This build does **not** bundle that engine; OVMS remains the default because
it provides one OpenAI-compatible endpoint for chat, embeddings and image generation, plus
model reload without restarting the app. Bundling a second engine would add another large
native dependency and a different streaming API, so it is not claimed here.

---

## Build the Windows binary

```powershell
.\packaging\build.ps1
```

Output: `dist\AIOLocal\AIOLocal.exe` (and a zip next to it). The build is a `onedir` bundle so
startup is fast and the web assets are visible for auditing. Copy the whole `dist\AIOLocal`
folder to another Windows PC; there is no installer. The first launch creates its own data
and tool folders, then the Setup page installs anything the machine is missing.

---

## Data and privacy

Everything lives in `%LOCALAPPDATA%\AIOLocal\` on Windows and
`~/.local/share/AIOLocal/` on Linux:

```
config\app.json      settings (no secrets)
config\token.bin     DPAPI-encrypted local API token
config\secrets.bin   DPAPI-encrypted optional keys (Brave Search, etc.)
data\memory.db       conversations, summaries, jobs, embeddings
data\knowledge.db    separate knowledge base (FTS5 + chunk vectors)
knowledge\inbox\    raw JSON/MD files picked up by the filing LLM
artifacts\images\    generated images
logs\                rotating logs (prompts are not logged at info level)
models\              OVMS model repository (relocatable)
tools\               app-owned runtimes installed by Setup (OVMS, Node, OpenCode, Python, GIMP)
opencode\            isolated OpenCode data, config, cache and state
workspace\           code workspace
```

- The server binds **127.0.0.1 only** and rejects non-local `Host`/`Origin` headers.
- Every API call needs the per-install token; the UI receives it in a same-origin page.
- Model downloads, Setup component downloads, and explicit web-search or knowledge-sync requests are
  the only outbound traffic. There is no telemetry, analytics, remote logging, or auto-update.

See `SECURITY.md` for the threat model and reporting instructions.

---

## Licence

AGPL-3.0-or-later. See `LICENSE` and `NOTICE`.

**Before uploading to GitHub:**
- Confirm the source is complete (`git add -A`; no `node_modules`, `.venv`, `dist`, or model weights).
- `NOTICE` already lists all third-party licences and the AGPL obligations. Do not bundle Open WebUI, OVMS binaries, or model files.
- The AGPL requires that anyone who receives the binary also gets the source. Provide the repository link and the exact version tag in releases.
- The `websearch.py` feature references third-party backends (DuckDuckGo/SeArXNG/Brave); include a note in the release that Brave needs a user-provided API key and the app does not ship keys.
- Stable Diffusion 1.5 weights use CreativeML OpenRAIL-M; do not bundle them and mention that restriction in the release notes.

Third-party components keep their own licences; `NOTICE` lists them and the obligations. In
particular, Stable Diffusion 1.5 weights are **CreativeML OpenRAIL-M**, not Apache-2.0, and carry
use restrictions that pass through to you.

---

## Project layout

```
aiolocal/
  __main__.py        launcher (server + browser)
  app.py             FastAPI app, middleware, static files
  config.py          paths, settings, token bootstrap
  db.py              SQLite, migrations, single-writer transactions
  security.py        DPAPI, token checks, origin/host validation, safe paths
  gateway.py         OpenAI-compatible /v1/chat/completions + memory
  control.py         /api/v1 control plane
  llm.py             upstream client and SSE tee
  models.py          OVMS catalog, config generation, load/unload
  images.py          image proxy and gallery
  events.py          server-sent events
  memory/            text, store, retrieve, assemble, filing, jobs, semantic
  documents.py       PDF pages -> vision model -> durable document memory
  knowledge.py       separate reference library, folder routing, scan/enrich jobs
  photogimp.py       optional PhotoGIMP manager (Windows)
  web/               no-build HTML/CSS/JS UI
tests/               unit + integration tests (fake upstream, no GPU needed)
tools/               kb_sync.py and kb_eval.py for the reference corpus
packaging/           PyInstaller spec and build script
PLAN/                the full architecture plan (design source of truth)
```
