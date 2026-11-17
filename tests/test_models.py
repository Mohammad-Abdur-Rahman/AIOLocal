from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.models import (
    CATALOG,
    GRAPH_TEMPLATE,
    ModelManager,
    inspect_ov_folder,
    parse_hf_repo,
)


def make_manager(tmp_path: Path) -> ModelManager:
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings(models_dir=str(tmp_path / "models")))
    return ModelManager(paths, store)


def model_dir(manager: ModelManager) -> Path:
    entry = next(m for m in CATALOG if m["id"] == "qwen3.5-9b")
    directory = manager.model_dir(entry)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "graph.pbtxt").write_text(GRAPH_TEMPLATE, encoding="utf-8")
    return directory


def test_generated_config_uses_graph_path_and_devices(tmp_path: Path):
    manager = make_manager(tmp_path)
    directory = model_dir(manager)
    (directory / "openvino_language_model.bin").write_bytes(b"0" * 2_000_000)

    manager.load("qwen3.5-9b", "CPU")
    payload = json.loads(manager.config_path.read_text(encoding="utf-8"))
    config = payload["model_config_list"][0]["config"]

    assert config["name"] == "qwen3.5-9b"
    assert config["graph_path"] == "graph-CPU.pbtxt"
    assert "task" not in config
    assert "plugin_config" not in config
    graph = (directory / "graph-CPU.pbtxt").read_text(encoding="utf-8")
    assert 'device: "CPU"' in graph
    # The original graph is left untouched.
    assert 'device: "GPU"' in (directory / "graph.pbtxt").read_text(encoding="utf-8")

    manager.load("qwen3.5-9b", "GPU")
    payload = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert payload["model_config_list"][0]["config"]["graph_path"] == "graph-GPU.pbtxt"


def test_installed_rejects_partial_downloads(tmp_path: Path):
    manager = make_manager(tmp_path)
    directory = model_dir(manager)
    entry = {"repo": "OpenVINO/Qwen3.5-9B-int4-ov"}

    # config.json but no weights is a partial download, not an installed model.
    assert manager.installed(entry) is False

    (directory / "openvino_language_model.bin").write_bytes(b"0" * 2_000_000)
    manager._installed_cache.clear()  # what a completed pull does
    assert manager.installed(entry) is True

    # Files inside the Hugging Face cache do not count as model weights.
    cached = directory / ".cache" / "huggingface" / "download"
    cached.mkdir(parents=True)
    (cached / "partial.bin").write_bytes(b"0" * 3_000_000)
    (directory / "openvino_language_model.bin").unlink()
    manager._installed_cache.clear()
    assert manager.installed(entry) is False


def test_installed_check_is_cached_briefly(tmp_path: Path):
    manager = make_manager(tmp_path)
    directory = model_dir(manager)
    entry = {"repo": "OpenVINO/Qwen3.5-9B-int4-ov"}
    assert manager.installed(entry) is False

    # A direct folder change is not seen until the short cache expires; this keeps
    # the 1.5 s status polling from re-walking every model folder each time.
    (directory / "openvino_language_model.bin").write_bytes(b"0" * 2_000_000)
    assert manager.installed(entry) is False
    manager._installed_cache[entry["repo"]] = (0.0, False)
    assert manager.installed(entry) is True


def test_ovms_environment_is_sanitised(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)
    root = tmp_path / "OVMS"
    (root / "python" / "Scripts").mkdir(parents=True)
    (root / "espeak-ng-data").mkdir()
    exe = root / "ovms.exe"
    exe.write_bytes(b"stub")

    monkeypatch.setenv("PYTHONPATH", r"C:\polluted")
    monkeypatch.setenv("PYTHONHOME", r"C:\wrong-python")
    env = manager._ovms_env(exe)

    assert "PYTHONPATH" not in env
    assert env["PYTHONHOME"] == str(root / "python")
    assert env["ESPEAK_DATA_PATH"] == str(root / "espeak-ng-data")
    assert env["OVMS_DIR"] == str(root)
    assert env["PATH"].startswith(str(root))
    assert env["PYTHONNOUSERSITE"] == "1"


def test_ovms_executable_auto_detection(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)
    exe = tmp_path / "ovms.exe"
    exe.write_bytes(b"stub")
    monkeypatch.setattr(manager, "_ovms_detected", lambda: [str(exe)])
    assert manager._ovms_executable() == exe

def test_lfm_defaults_to_cpu_with_npu_fallback():
    from aiolocal.models import catalog_entry

    entry = catalog_entry("lfm2.5-1.2b")
    assert entry is not None
    assert entry["default_device"] == "CPU"
    assert entry["devices"][0] == "CPU"
    assert "CPU" in entry["fallback_devices"]


def make_custom_folder(
    manager: ModelManager,
    name: str,
    *,
    config: dict | None = None,
    model_index: dict | None = None,
    xml: str = "openvino_model",
    graph: bool = True,
) -> Path:
    directory = manager.models_root() / name
    directory.mkdir(parents=True)
    (directory / f"{xml}.xml").write_text("<net/>", encoding="utf-8")
    (directory / f"{xml}.bin").write_bytes(b"0" * 2_000_000)
    (directory / "config.json").write_text(json.dumps(config or {}), encoding="utf-8")
    if model_index is not None:
        (directory / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    if graph:
        (directory / "graph.pbtxt").write_text(GRAPH_TEMPLATE, encoding="utf-8")
    return directory


def test_inspect_ov_folder_rejects_non_openvino_weights(tmp_path: Path):
    manager = make_manager(tmp_path)
    folder = manager.models_root() / "gguf-model"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.gguf").write_bytes(b"0" * 1024)

    info = inspect_ov_folder(folder)

    assert info["ok"] is False
    assert "OpenVINO IR" in info["reason"]
    assert "not supported" in info["reason"]


def test_inspect_ov_folder_requires_metadata(tmp_path: Path):
    manager = make_manager(tmp_path)
    folder = manager.models_root() / "bare-ir"
    folder.mkdir(parents=True)
    (folder / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (folder / "openvino_model.bin").write_bytes(b"0" * 1024)

    info = inspect_ov_folder(folder)

    assert info["ok"] is False
    assert "config.json" in info["reason"]


def test_inspect_ov_folder_detects_text_embedding_and_image(tmp_path: Path):
    manager = make_manager(tmp_path)

    text = make_custom_folder(manager, "custom-text", config={"max_position_embeddings": 16384})
    embedding = make_custom_folder(manager, "bge-embedding", config={"model_type": "embedding"})
    image = make_custom_folder(manager, "sd-custom", model_index={"_class_name": "StableDiffusionPipeline"})

    text_info = inspect_ov_folder(text)
    assert text_info["ok"] is True
    assert text_info["task"] == "text_generation"
    assert text_info["context"] == 16384

    embed_info = inspect_ov_folder(embedding)
    assert embed_info["ok"] is True
    assert embed_info["task"] == "embeddings"

    image_info = inspect_ov_folder(image)
    assert image_info["ok"] is True
    assert image_info["task"] == "image_generation"


def test_parse_hf_repo_accepts_urls_and_rejects_other_hosts():
    assert parse_hf_repo("https://huggingface.co/OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov") == (
        "OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov"
    )
    assert parse_hf_repo("OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov") == (
        "OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov"
    )
    with pytest.raises(ValueError, match="huggingface.co"):
        parse_hf_repo("https://example.com/OpenVINO/model")
    with pytest.raises(ValueError, match="organisation"):
        parse_hf_repo("https://huggingface.co/only-a-name")


def test_scan_registers_pending_model_once(tmp_path: Path):
    manager = make_manager(tmp_path)
    make_custom_folder(manager, "detected-model")

    fresh = manager.scan_custom_models(force=True)

    assert len(fresh) == 1
    assert fresh[0]["folder"] == "detected-model"
    assert fresh[0]["role_status"] == "pending"
    assert [m["id"] for m in manager.pending_entries()] == [fresh[0]["id"]]

    # A second scan sees the same folder and does not register a duplicate.
    assert manager.scan_custom_models(force=True) == []
    assert len(manager.entries()) == len(CATALOG) + 1

    # A non-OpenVINO folder is ignored and explained.
    broken = manager.models_root() / "not-openvino"
    broken.mkdir(parents=True)
    (broken / "model.onnx").write_bytes(b"0")
    manager.scan_custom_models(force=True)
    assert "not-openvino" in manager._scan_ignored


def test_assign_role_configures_chat_settings(tmp_path: Path):
    manager = make_manager(tmp_path)
    make_custom_folder(manager, "custom-chat")
    entry = manager.scan_custom_models(force=True)[0]

    result = manager.assign_role(entry["id"], "chat", device="CPU")

    assert result["ok"] is True
    assert result["role"] == "chat"
    assert result["device"] == "CPU"
    settings = manager.settings_store.load()
    assert settings.chat_model == entry["id"]
    assert entry["id"] in settings.auto_load
    stored = manager.custom_entry(entry["id"])
    assert stored is not None
    assert stored["role_status"] == "assigned"
    assert stored["role_key"] == "chat"
    assert manager.status()["models"][-1]["custom"] is True


def test_assign_image_editing_role_sets_edit_model(tmp_path: Path):
    manager = make_manager(tmp_path)
    make_custom_folder(
        manager,
        "inpaint-custom",
        model_index={"_class_name": "StableDiffusionInpaintPipeline"},
    )
    entry = manager.scan_custom_models(force=True)[0]

    result = manager.assign_role(entry["id"], "inpaint")

    assert result["role"] == "image_edit"
    settings = manager.settings_store.load()
    assert settings.image_edit_model == entry["id"]
    assert settings.image_model != entry["id"]


def test_assign_role_rejects_unknown_custom_model(tmp_path: Path):
    manager = make_manager(tmp_path)

    with pytest.raises(ValueError, match="not a custom model"):
        manager.assign_role("qwen3.5-9b", "chat")
