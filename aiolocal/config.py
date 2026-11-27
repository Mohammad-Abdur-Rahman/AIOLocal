"""Paths, settings and first-run bootstrap.

Everything is local. Settings are plain JSON with no secrets; the API token is stored
separately and encrypted with Windows DPAPI (see :mod:`aiolocal.security`).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    """Per-user data folder on Windows and Linux.

    Windows keeps the familiar ``%LOCALAPPDATA%\\AIOLocal``. On Linux the XDG
    data directory is used so a source build behaves like a normal desktop app
    and never writes into the checkout.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "AIOLocal"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    config_dir: Path
    db_path: Path
    token_path: Path
    artifact_dir: Path
    image_dir: Path
    log_dir: Path
    models_dir: Path
    tools_dir: Path
    knowledge_dir: Path
    knowledge_inbox: Path
    knowledge_files: Path
    knowledge_db_path: Path
    secret_path: Path

    @classmethod
    def create(cls, root: str | os.PathLike[str] | None = None) -> "Paths":
        data = Path(root).expanduser().resolve() if root else default_data_dir()
        config = data / "config"
        models = data / "models"
        tools = data / "tools"
        knowledge = data / "knowledge"
        knowledge_inbox = knowledge / "inbox"
        knowledge_files = knowledge / "files"
        for d in (
            data,
            config,
            data / "data",
            data / "artifacts",
            data / "logs",
            models,
            tools,
            knowledge,
            knowledge_inbox,
            knowledge_files,
        ):
            d.mkdir(parents=True, exist_ok=True)
        image_dir = data / "artifacts" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=data,
            config_dir=config,
            db_path=data / "data" / "memory.db",
            token_path=config / "token.bin",
            artifact_dir=data / "artifacts",
            image_dir=image_dir,
            log_dir=data / "logs",
            models_dir=models,
            tools_dir=tools,
            knowledge_dir=knowledge,
            knowledge_inbox=knowledge_inbox,
            knowledge_files=knowledge_files,
            knowledge_db_path=data / "data" / "knowledge.db",
            secret_path=config / "secrets.bin",
        )


@dataclass
class Settings:
    # Upstream (OVMS by default)
    upstream_base_url: str = "http://127.0.0.1:8000"
    chat_model: str = "qwen3.5-9b"
    filing_model: str = ""  # empty -> deterministic heuristic filing
    # Context / memory
    context_tokens: int = 8192
    output_reserve_tokens: int = 1024
    max_output_tokens: int = 1024
    enable_thinking: bool = False
    short_history_turns: int = 6
    memory_enabled: bool = True
    memory_hint: bool = True
    embedding_backend: str = "hash"  # "hash" or "ovms"
    embedding_model: str = "qwen3-embedding-0.6b"
    embed_dim: int = 256
    # Document vision (PDF pages -> main vision model -> summary memory)
    # Kept for old app.json files, but always one page per call and no page cap.
    document_pages_per_batch: int = 1
    document_dpi: int = 140
    document_max_pages: int = 0
    document_keep_pages: bool = False
    # Explicit image roles, so a custom model can be selected from the UI.
    image_model: str = "sd15"
    image_edit_model: str = "sd15-inpaint"
    # Bumped when a saved setting needs a one-time correction.
    config_version: int = 2
    # Local OVMS
    ovms_path: str = ""
    models_dir: str = ""
    auto_load: list[str] = field(default_factory=list)
    ovms_auto_start: bool = True
    # Generation
    temperature: float = 0.7
    # Knowledge base (separate from conversation memory; opt-in per message)
    knowledge_enabled: bool = True
    knowledge_top_k: int = 6
    knowledge_enrich_batch: int = 20
    knowledge_auto_folders: int = 3
    docling_path: str = ""
    mineru_path: str = ""
    # Document vision mode: "auto" (text layer first, vision only for sparse /
    # figure-heavy pages), "all" (vision on every page with a drawing), or "off".
    document_vision_mode: str = "auto"
    knowledge_scan_seconds: int = 120
    # Web search (off by default; each message must opt in)
    web_search_enabled: bool = False
    web_search_backend: str = "duckduckgo"  # duckduckgo | searxng | brave
    searxng_url: str = ""
    web_search_max_results: int = 5
    # Code workspace (OpenCode)
    opencode_path: str = ""
    opencode_port: int = 4096
    workspace_dir: str = ""
    # Reverse engineering (REA over MCP stdio)
    rea_command: str = ""
    rea_version: str = ""
    ghidra_path: str = ""
    # GIMP dock
    gimp_path: str = ""
    gimp_embed: bool = True
    # Diagnostics
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.context_tokens = max(1024, min(int(self.context_tokens), 131072))
        self.output_reserve_tokens = max(64, min(int(self.output_reserve_tokens), 32768))
        self.max_output_tokens = max(16, min(int(self.max_output_tokens), 32768))
        self.short_history_turns = max(0, min(int(self.short_history_turns), 50))
        self.embed_dim = max(64, min(int(self.embed_dim), 2048))
        # PDF accuracy: exactly one page per vision call, and every page is read.
        # The fields remain for compatibility with old app.json files; the values
        # are forced so a stale config cannot reintroduce batching or a page cap.
        self.document_pages_per_batch = 1
        self.document_dpi = max(72, min(int(self.document_dpi), 220))
        self.document_max_pages = 0
        if self.embedding_backend not in ("hash", "ovms"):
            self.embedding_backend = "hash"
        self.temperature = max(0.0, min(float(self.temperature), 2.0))
        self.knowledge_top_k = max(1, min(int(self.knowledge_top_k), 20))
        self.knowledge_enrich_batch = max(1, min(int(self.knowledge_enrich_batch), 100))
        self.knowledge_auto_folders = max(1, min(int(self.knowledge_auto_folders), 6))
        self.knowledge_scan_seconds = max(0, min(int(self.knowledge_scan_seconds), 3600))
        self.web_search_max_results = max(1, min(int(self.web_search_max_results), 10))
        self.opencode_port = max(1024, min(int(self.opencode_port), 65535))
        if self.web_search_backend not in ("duckduckgo", "searxng", "brave"):
            self.web_search_backend = "duckduckgo"
        if self.document_vision_mode not in ("auto", "all", "off"):
            self.document_vision_mode = "auto"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Settings":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in (raw or {}).items() if k in allowed and v is not None}
        obj = cls(**clean)
        if not isinstance(obj.auto_load, list):
            obj.auto_load = []
        obj.auto_load = [str(x) for x in obj.auto_load][:32]
        return obj


class SettingsStore:
    """Reads/writes ``config/app.json`` atomically."""

    def __init__(self, paths: Paths) -> None:
        self.path = paths.config_dir / "app.json"
        self._paths = paths

    def load(self) -> Settings:
        if not self.path.exists():
            s = Settings()
            self.save(s)
            return s
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("settings root must be an object")
            migrated = False
            try:
                version = int(raw.get("config_version") or 0)
            except (TypeError, ValueError):
                version = 0
            if version < 2:
                # Older builds shipped 3 pages per call and a 300-page cap. Even
                # though Settings now forces the corrected values, rewrite the file
                # so the UI and the data folder do not keep advertising them.
                raw["document_pages_per_batch"] = 1
                raw["document_max_pages"] = 0
                raw["config_version"] = 2
                migrated = True
            settings = Settings.from_dict(raw)
            if migrated:
                self.save(settings)
            return settings
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Corrupt settings should never prevent the app from starting.
            backup = self.path.with_suffix(".corrupt.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            s = Settings()
            self.save(s)
            return s

    def save(self, settings: Settings) -> None:
        payload = json.dumps(settings.to_dict(), indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def models_dir(self, settings: Settings) -> Path:
        if settings.models_dir:
            return Path(settings.models_dir).expanduser()
        return self._paths.models_dir
