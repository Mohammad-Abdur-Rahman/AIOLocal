# Contributing to AIO Local

Thanks for taking the time to contribute. AIO Local is a local-first, single-user
desktop app: chat, memory, documents, knowledge, images and OpenCode all run on
the same machine through OpenVINO Model Server (OVMS). Small, focused changes
with a regression test are the easiest to review and merge.

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Ways to help

- Report a reproducible bug with the bug report form.
- Request a feature with the feature request form.
- Improve documentation, error messages or the UI copy.
- Add a regression test for a bug you found.
- Improve OpenVINO model compatibility or device handling.
- Review pull requests and reproduce reported issues.

Security problems must **not** be reported in a public issue. Follow
[SECURITY.md](SECURITY.md) and use a private security advisory.

## Development setup

Requirements: Windows 10/11 x64, Python 3.11+ (3.12 is used for releases), and
Git. OVMS and model weights are only needed to exercise real inference; the test
suite uses fakes.

```powershell
git clone https://github.com/Mohammad-Abdur-Rahman/AIOLocal.git
cd AIOLocal
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the app from source without opening a browser:

```powershell
.\.venv\Scripts\python.exe -m aiolocal --no-browser
```

The UI is served at `http://127.0.0.1:7317`. App data lives in
`%LOCALAPPDATA%\AIOLocal`; logs are in `%LOCALAPPDATA%\AIOLocal\logs`. Use
`--data-dir` to run against a throwaway profile:

```powershell
.\.venv\Scripts\python.exe -m aiolocal --no-browser --port 7399 --data-dir "$env:TEMP\aiolocal-dev"
```

On Linux the source build works with the same commands (use
`.venv/bin/python`). Windows-only features such as the GIMP dock report that
they are unavailable.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Rules for tests:

- Every bug fix needs a regression test that fails before the fix.
- Tests must not download models or require OVMS, a GPU or network access.
- Use `tmp_path` and a temporary data directory; never write to the real
  `%LOCALAPPDATA%\AIOLocal` profile.
- Keep tests deterministic; no sleeps longer than a few hundred milliseconds.

The full suite currently runs in under a minute.

## Project layout

```
aiolocal/
  app.py          FastAPI app, lifespan and module wiring
  control.py      /api/v1 endpoints used by the web UI
  gateway.py      OpenAI-compatible gateway and memory pipeline
  documents.py    PDF rendering and one-page-per-call vision jobs
  models.py       catalog, OVMS config, custom OpenVINO models
  config.py       Settings, Paths and migration of app.json
  memory/         exact memory, summaries, filing worker
  knowledge.py    knowledge folders, ingestion and routing
  code.py         isolated OpenCode server and opencode.json
  reverse.py      REA over MCP stdio
  web/            index.html, app.js, styles.css (no build step)
tests/            pytest suite
packaging/        PyInstaller spec and Windows build script
```

## Code guidelines

- Python 3.11+ syntax, type hints on public functions, and the existing
  `from __future__ import annotations` convention.
- Keep the web UI dependency-free: plain HTML, CSS and JavaScript. No bundler,
  framework or CDN.
- Never put user or exception text directly into an HTTP error response. Log
  the exception and return a curated, actionable message (see
  `aiolocal/errors.py`).
- Do not log prompts, tokens, secrets or full request bodies.
- Filesystem joins must go through `safe_path()` / `safe_join()` and
  allow-listed IDs.
- Do not commit model weights, `dist/`, `build/`, virtual environments, logs or
  screenshots with personal data.
- Update `CHANGELOG.md` for user-visible changes and `README.md` when behaviour
  or setup steps change.

## Pull requests

1. Fork the repository and create a branch from `main`
   (`fix/pdf-one-page`, `feat/custom-model-roles`, and so on).
2. Keep the change focused. Separate refactors from behaviour changes.
3. Add or update tests and run the full suite.
4. Describe the problem, the fix and how you tested it. Include a screenshot
   for UI changes.
5. Link the issue the pull request closes (`Closes #123`).
6. Make sure CI is green and there are no new CodeQL alerts.

Maintainers may ask for a smaller change or a test before merging. By
contributing you agree that your contribution is licensed under the project's
[AGPL-3.0-or-later](LICENSE) licence.

## Release checklist (maintainers)

This is the process used for `0.2.1`:

1. Update `__version__` and `pyproject.toml`.
2. Update `CHANGELOG.md`; keep the comparison links current.
3. Run `pytest -q` and `pip check`.
4. Build with `.\packaging\build.ps1`.
5. Smoke-test the bundle with a fresh `--data-dir` and check `/healthz`, the UI
   and `/api/v1/state`.
6. Write `dist\SHA256SUMS.txt` and the release notes.
7. Commit, tag (`vX.Y.Z`), push `main` and the tag, then publish the GitHub
   release with the zip and checksum attached.
8. Confirm Dependabot and CodeQL are clean for the new commit.
