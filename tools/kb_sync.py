"""Download a curated reference corpus into the AIOLocal knowledge inbox.

The knowledge base is only useful when it contains real project material. This
tool fetches Tier-1 documentation with a sparse Git clone (or plain HTTPS for a
few single-page references), copies text files into ``inbox/<folder>`` and writes
``.kbmeta.json`` beside them so the scanner records the source URL, licence and
commit.

It never talks to the running app; it only writes into the inbox folder. The app
picks the files up on its next scan, or run ``tools/kb_eval.py --scan``.

Usage::

    python tools/kb_sync.py list
    python tools/kb_sync.py sync
    python tools/kb_sync.py sync --only fastapi,sqlite --force
    python tools/kb_sync.py sync --dest C:\\temp\\inbox
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "kb_manifest.json"
TEXT_EXTENSIONS = {
    ".md", ".markdown", ".mdx", ".txt", ".rst", ".json", ".jsonl", ".ndjson",
    ".csv", ".tsv", ".html", ".htm",
}
IGNORED_DIRS = {
    ".git", ".github", ".hg", ".svn", "__pycache__", "node_modules", ".venv",
    "venv", "_build", "build", "dist",
}
MAX_FILE_BYTES = 8 * 1024 * 1024

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
    re.compile(r"<\|\s*(im_start|system|assistant)\s*\|>", re.I),
    re.compile(r"begin\s+system\s+prompt", re.I),
]


def default_inbox() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "AIOLocal" / "knowledge" / "inbox"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )


def sparse_checkout(repo: Path, includes: list[dict]) -> None:
    paths = [str(item["path"]) for item in includes if item.get("path")]
    if not paths:
        return
    try:
        run_git(["sparse-checkout", "set", "--skip-checks", *paths], repo)
    except subprocess.CalledProcessError:
        # Older Git versions do not know --skip-checks but can still handle dirs
        # and files separately.
        directories = [p for p in paths if not Path(p).suffix]
        files = [p for p in paths if Path(p).suffix]
        if directories:
            run_git(["sparse-checkout", "set", *directories], repo)
        for item in files:
            run_git(["checkout", "HEAD", "--", item], repo)


def iter_include_files(repo: Path, include: dict):
    relative = str(include.get("path") or "")
    source = repo / relative
    if not source.exists():
        return
    if source.is_file():
        if source.suffix.lower() in TEXT_EXTENSIONS:
            yield source, Path(source.name)
        return
    for root, dirs, files in os.walk(source):
        dirs[:] = [
            d for d in dirs
            if d.lower() not in IGNORED_DIRS and not d.startswith(".")
        ]
        root_path = Path(root)
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = root_path / name
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path, path.relative_to(source)


def scan_injection(text: str) -> list[str]:
    hits: list[str] = []
    sample = text[:400_000]
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(sample)
        if match:
            hits.append(match.group(0)[:120])
        if len(hits) >= 5:
            break
    return hits


def copy_license(repo: Path, target: Path) -> str:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt"):
        source = repo / name
        if source.is_file():
            target.mkdir(parents=True, exist_ok=True)
            (target / "LICENSE-upstream.txt").write_bytes(source.read_bytes())
            return name
    return ""


def sync_git(source: dict, dest_root: Path, *, force: bool, dry_run: bool) -> dict:
    folder = dest_root / source["folder"]
    if folder.exists() and force and not dry_run:
        shutil.rmtree(folder)
    if dry_run:
        return {
            "name": source["name"],
            "folder": source["folder"],
            "state": "planned",
            "files": "?",
        }
    folder.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f"aiolocal-kb-{source['name']}-"))
    copy = temp / "repo"
    try:
        run_git(
            [
                "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "--quiet", source["url"], str(copy),
            ]
        )
        sparse_checkout(copy, source.get("includes") or [])
        commit = run_git(["rev-parse", "HEAD"], copy).stdout.strip()
        copied = 0
        warnings: list[dict] = []
        seen_names: set[Path] = set()
        for include in source.get("includes") or []:
            for path, relative in iter_include_files(copy, include):
                target = folder / relative
                if target in seen_names:
                    continue
                seen_names.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    target.write_text(text, encoding="utf-8", errors="replace")
                except OSError as exc:
                    warnings.append({"file": str(relative), "error": str(exc)[:200]})
                    continue
                copied += 1
                hits = scan_injection(text)
                if hits:
                    warnings.append({"file": str(relative), "injection": hits})
        licence_file = copy_license(copy, folder)
        metadata = {
            "source": source["name"],
            "source_url": source["url"],
            "commit": commit,
            "license": source.get("license"),
            "license_file": licence_file,
            "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": copied,
            "warnings": warnings[:100],
        }
        (folder / ".kbmeta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return {
            "name": source["name"],
            "folder": source["folder"],
            "state": "ok",
            "files": copied,
            "warnings": len(warnings),
            "commit": commit[:12],
        }
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def safe_filename(url: str) -> str:
    name = Path(urllib.parse.urlsplit(url).path).name or "reference.html"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:160] or "reference.html"


def sync_urls(source: dict, dest_root: Path, *, force: bool, dry_run: bool) -> dict:
    folder = dest_root / source["folder"]
    if folder.exists() and force and not dry_run:
        shutil.rmtree(folder)
    if dry_run:
        return {
            "name": source["name"],
            "folder": source["folder"],
            "state": "planned",
            "files": len(source.get("urls") or []),
        }
    folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    warnings: list[dict] = []
    entries = []
    for item in source.get("urls") or []:
        url = item["url"]
        name = safe_filename(url)
        target = folder / name
        request = urllib.request.Request(
            url, headers={"User-Agent": "AIOLocal-KB-Sync/0.2 (+local reference library)"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                warnings.append({"file": name, "error": "file is too large"})
                continue
            text = raw.decode("utf-8", errors="replace")
            target.write_text(text, encoding="utf-8", errors="replace")
        except Exception as exc:
            warnings.append({"file": name, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        copied += 1
        entries.append({"url": url, "title": item.get("title") or name, "file": name})
        hits = scan_injection(text)
        if hits:
            warnings.append({"file": name, "injection": hits})
    metadata = {
        "source": source["name"],
        "source_url": source.get("url") or (source.get("urls") or [{}])[0].get("url"),
        "commit": "",
        "license": source.get("license"),
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": copied,
        "urls": entries,
        "warnings": warnings[:100],
    }
    (folder / ".kbmeta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "name": source["name"],
        "folder": source["folder"],
        "state": "ok",
        "files": copied,
        "warnings": len(warnings),
        "commit": "-",
    }


def command_list(manifest: dict) -> int:
    print(f"Tier {manifest.get('tier')} corpus:")
    for source in manifest.get("sources") or []:
        kind = source.get("type") or "git"
        where = source.get("url") or f"{len(source.get('urls') or [])} URLs"
        print(f"  {source['name']:<18} {source['folder']:<22} {kind:<4} {where}")
    return 0


def command_sync(args) -> int:
    manifest = load_manifest()
    only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
    sources = [
        source for source in manifest.get("sources") or []
        if not only or source["name"] in only or source["folder"] in only
    ]
    if not sources:
        print("No matching sources in the manifest.", file=sys.stderr)
        return 2
    dest_root = Path(args.dest).expanduser().resolve() if args.dest else default_inbox()
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest_root}")
    results = []
    started = time.time()
    for source in sources:
        kind = source.get("type") or "git"
        print(f"-> {source['name']} ({kind}) ...", flush=True)
        try:
            if kind == "url":
                result = sync_urls(source, dest_root, force=args.force, dry_run=args.dry_run)
            else:
                result = sync_git(source, dest_root, force=args.force, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()[:400]
            result = {
                "name": source["name"],
                "folder": source["folder"],
                "state": "failed",
                "error": message,
            }
        except Exception as exc:
            result = {
                "name": source["name"],
                "folder": source["folder"],
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:400],
            }
        results.append(result)
        if result["state"] == "failed":
            print(f"  failed: {result.get('error')}")
        else:
            print(
                f"  {result['state']}: {result.get('files')} files"
                + (f", {result.get('warnings')} warnings" if result.get("warnings") else "")
            )
    elapsed = time.time() - started
    ok = [r for r in results if r["state"] in ("ok", "planned")]
    failed = [r for r in results if r["state"] == "failed"]
    print(f"Done: {len(ok)} sources, {len(failed)} failed, {elapsed:.1f}s")
    if failed:
        for result in failed:
            print(f"  ! {result['name']}: {result.get('error')}", file=sys.stderr)
        return 1
    print("Next: python tools/kb_eval.py --scan")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the AIOLocal reference corpus.")
    parser.add_argument("command", choices=("list", "sync"), nargs="?", default="list")
    parser.add_argument("--dest", default="", help="destination inbox (default: app data inbox)")
    parser.add_argument("--only", default="", help="comma-separated source names or folders")
    parser.add_argument("--force", action="store_true", help="replace existing source folders")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    if args.command == "list":
        return command_list(manifest)
    return command_sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
