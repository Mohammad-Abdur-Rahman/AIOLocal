from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiolocal.code import CodeWorkspace
from aiolocal.config import Paths, SettingsStore


def _workspace(tmp_path: Path, monkeypatch, which=None) -> CodeWorkspace:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    if which is not None:
        monkeypatch.setattr("aiolocal.code.shutil.which", which)
    paths = Paths.create(tmp_path / "data")
    return CodeWorkspace(paths, SettingsStore(paths), base_url="")


def test_opencode_desktop_cli_is_detected(tmp_path: Path, monkeypatch):
    local = tmp_path / "Local"
    exe = local / "Programs" / "ai.opencode.desktop" / "resources" / "opencode-cli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZstub")

    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr("aiolocal.code.shutil.which", lambda name: None)

    paths = Paths.create(tmp_path / "data")
    workspace = CodeWorkspace(paths, SettingsStore(paths), base_url="http://127.0.0.1:8000")
    assert workspace.executable() == str(exe)


def _broken_npm_shim(root: Path) -> Path:
    """The layout npm leaves when @opencode/cli's postinstall never ran.

    ``bin/opencode.exe`` is a small shell script, not a PE image, and the npm
    ``opencode.cmd`` shim on PATH launches it. Windows then shows an
    "Unsupported 16-Bit Application" dialog.
    """
    placeholder = root / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
    placeholder.parent.mkdir(parents=True)
    placeholder.write_text(
        'echo "Error: postinstall was not run." >&2\nexit 1\n', encoding="utf-8"
    )
    shim = root / "opencode.cmd"
    shim.write_text(
        "@ECHO off\n"
        "SET dp0=%~dp0\n"
        '"%dp0%\\node_modules\\@opencode\\cli\\bin\\opencode.exe"   %*\n',
        encoding="utf-8",
    )
    return shim


@pytest.mark.skipif(os.name != "nt", reason="Windows loader check")
def test_broken_npm_shim_is_skipped(tmp_path: Path, monkeypatch):
    shim = _broken_npm_shim(tmp_path / "npm")
    workspace = _workspace(tmp_path, monkeypatch, lambda name: str(shim))

    assert workspace.executable() is None
    assert "not a working program" in workspace._discovery_note


@pytest.mark.skipif(os.name != "nt", reason="Windows loader check")
def test_portable_copy_beats_a_broken_path_shim(tmp_path: Path, monkeypatch):
    shim = _broken_npm_shim(tmp_path / "npm")
    workspace = _workspace(tmp_path, monkeypatch, lambda name: str(shim))
    portable = workspace.paths.tools_dir / "opencode" / "opencode.exe"
    portable.parent.mkdir(parents=True)
    portable.write_bytes(b"MZstub")

    assert workspace.executable() == str(portable)
