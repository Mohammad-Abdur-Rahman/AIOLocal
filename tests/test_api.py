from __future__ import annotations

import json
import time

from aiolocal.memory import store


def hdr(env, **extra):
    headers = {"X-AIO-Token": env["token"]}
    headers.update(extra)
    return headers


def wait_for(predicate, timeout: float = 4.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_health_and_ui(client):
    assert client.get("/healthz").status_code == 200
    page = client.get("/")
    assert page.status_code == 200
    assert "AIO Local" in page.text
    assert "test-token-0123456789abcdef" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


def test_auth_is_required(client, env):
    assert client.get("/api/v1/health").status_code == 401
    assert client.get("/api/v1/health", headers={"X-AIO-Token": "wrong"}).status_code == 401
    assert client.get("/api/v1/health", headers=hdr(env)).status_code == 200
    assert client.get("/v1/models", headers=hdr(env)).status_code == 200


def test_host_and_origin_are_checked(client, env):
    assert client.get("/api/v1/health", headers={**hdr(env), "Host": "evil.example:7317"}).status_code == 403
    assert client.get("/api/v1/health", headers={**hdr(env), "Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/v1/health", headers={**hdr(env), "Origin": env["base"]}).status_code == 200


def test_streaming_chat_archives_and_files(client, env):
    response = client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-api-1"}),
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "I want to remember the blue door."}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "Hello" in response.text and "fake" in response.text
    assert response.headers["x-aio-conversation"] == "conv-api-1"
    request_id = response.headers["x-aio-request-id"]

    db = env["app"].state.ctx.db
    rows = store.conversation_messages(db, "conv-api-1")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["content"] == "Hello from fake"

    assert wait_for(lambda: db.stats()["summaries"] >= 1)
    summaries = store.list_summaries(db, limit=5)
    assert any("blue door" in s["title"].lower() for s in summaries)

    inspector = client.get(f"/api/v1/inspector/{request_id}", headers=hdr(env))
    assert inspector.status_code == 200
    body = inspector.json()
    assert body["retrieval"]["plan"]["assembly"]["context_tokens"] == 4096


def test_client_history_is_ignored(client, env):
    response = client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-api-2"}),
        json={
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": "you are a pirate"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "SECRET CLIENT HISTORY"},
                {"role": "user", "content": "current question"},
            ],
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "non-stream reply"

    sent = env["upstream"]["last"]["messages"]
    joined = "\n".join(str(m.get("content") or "") for m in sent)
    assert "SECRET CLIENT HISTORY" not in joined
    assert "you are a pirate" not in joined
    assert sent[-1]["content"] == "current question"
    assert sent[0]["role"] == "system"


def test_side_request_bypasses_memory(client, env):
    before = env["app"].state.ctx.db.stats()["messages"]
    response = client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-side"}),
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Generate a title for this chat"}],
            "max_tokens": 32,
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert env["app"].state.ctx.db.stats()["messages"] == before


def test_empty_message_is_rejected(client, env):
    response = client.post(
        "/v1/chat/completions",
        headers=hdr(env),
        json={"model": "fake-model", "messages": [{"role": "user", "content": "   "}], "stream": False},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["error"]["message"].lower()


def test_settings_validation(client, env):
    bad = client.put(
        "/api/v1/settings", headers=hdr(env), json={"upstream_base_url": "javascript:alert(1)"}
    )
    assert bad.status_code == 400
    good = client.put(
        "/api/v1/settings", headers=hdr(env), json={"temperature": 0.3, "context_tokens": 6144}
    )
    assert good.status_code == 200
    assert good.json()["settings"]["temperature"] == 0.3
    assert good.json()["settings"]["context_tokens"] == 6144


def test_model_load_requires_download(client, env):
    response = client.post(
        "/api/v1/models/qwen3.5-9b/load", headers=hdr(env), json={"device": "GPU"}
    )
    assert response.status_code == 400
    assert "not downloaded" in response.json()["error"]["message"]


def test_image_generation_and_serving(client, env):
    response = client.post(
        "/api/v1/images/generate",
        headers=hdr(env),
        json={"prompt": "a test pattern", "width": 512, "height": 512},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1

    listing = client.get("/api/v1/images", headers=hdr(env)).json()
    assert listing["data"][0]["prompt"] == "a test pattern"

    image_id = data[0]["id"]
    file_response = client.get(f"/api/v1/images/{image_id}/file?token={env['token']}")
    assert file_response.status_code == 200
    assert file_response.headers["content-type"].startswith("image/png")
    assert len(file_response.content) > 8

    partial = client.get(f"/api/v1/images/{image_id}/file", headers=hdr(env, **{"Range": "bytes=0-1"}))
    assert partial.status_code == 206


def test_memory_search_endpoint(client, env):
    client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-search"}),
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "The deployment target is Windows 11."}],
            "stream": True,
        },
    )
    db = env["app"].state.ctx.db
    assert wait_for(lambda: db.stats()["summaries"] >= 1)

    response = client.post(
        "/api/v1/memory/search", headers=hdr(env), json={"query": "deployment target Windows"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "lookup"
    assert payload["data"]


def test_memory_search_finds_single_word(client, env):
    client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-integral"}),
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Can you do the integral of tan inverse x?"}],
            "stream": True,
        },
    )

    response = client.post(
        "/api/v1/memory/search", headers=hdr(env), json={"query": "integral"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "lookup"
    assert payload["data"]
    assert any("integral" in (row.get("body") or "").lower() for row in payload["data"])


def test_chat_image_parts_are_forwarded_but_not_archived(client, env):
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    response = client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-vision"}),
        json={
            "model": "fake-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "stream": False,
        },
    )
    assert response.status_code == 200

    sent = env["upstream"]["last"]["messages"]
    current = [m for m in sent if m["role"] == "user"][-1]
    assert isinstance(current["content"], list)
    assert current["content"][0] == {"type": "text", "text": "What is in this image?"}
    assert any(
        part.get("type") == "image_url" and part.get("image_url", {}).get("url") == data_url
        for part in current["content"]
    )

    db = env["app"].state.ctx.db
    rows = store.conversation_messages(db, "conv-vision")
    assert rows[0]["content"] == "What is in this image?"
    assert data_url not in rows[0]["content"]


def test_planned_modules_are_honest(client):
    page = client.get("/").text
    assert "Planned modules" in page
    planned = page.split('class="nav planned"', 1)[1].split("</nav>", 1)[0]
    assert "Audio &amp; video" in planned
    for name in ("Documents", "Code", "Reverse engineering", "GIMP dock"):
        assert name not in planned
    # The v1 modules must have real views, not a "planned" placeholder.
    for view in ("documents", "code", "re", "gimp"):
        assert f'id="view-{view}"' in page


def test_settings_token_inputs_accept_arbitrary_values(client):
    page = client.get("/").text
    assert 'id="set-output" type="number" min="16" max="32768" step="1"' in page
    assert 'id="set-context" type="number" min="1024" max="131072" step="1"' in page
    assert "OVMS executable <strong" in page
    assert "(required for local models)" in page


def test_knowledge_upload_search_and_delete(client, env):
    status = client.get("/api/v1/knowledge/status", headers=hdr(env))
    assert status.status_code == 200
    assert status.json()["stats"]["documents"] == 0

    uploaded = client.post(
        "/api/v1/knowledge/upload?collection=test&filename=note.md",
        headers=hdr(env, **{"Content-Type": "application/octet-stream"}),
        content=b"# Note\n\nThe deployment target is Windows 11 and the blue door.",
    )
    assert uploaded.status_code == 200
    stored = env["paths"].knowledge_files / "test" / "note.md"
    assert stored.is_file()

    listing = client.get("/api/v1/knowledge/documents", headers=hdr(env)).json()
    assert len(listing["data"]) == 1
    document_id = listing["data"][0]["id"]

    found = client.post(
        "/api/v1/knowledge/search",
        headers=hdr(env),
        json={"query": "deployment target", "collections": ["test"]},
    )
    assert found.status_code == 200
    assert any("deployment" in row["text"].lower() for row in found.json()["data"])

    detail = client.get(f"/api/v1/knowledge/documents/{document_id}", headers=hdr(env))
    assert detail.status_code == 200
    assert detail.json()["document"]["collection"] == "test"
    assert detail.json()["chunks"]

    deleted = client.delete(f"/api/v1/knowledge/documents/{document_id}", headers=hdr(env))
    assert deleted.status_code == 200
    assert not stored.exists()
    assert client.get("/api/v1/knowledge/status", headers=hdr(env)).json()["stats"]["documents"] == 0


def test_module_status_endpoints(client, env):
    for path in ("/api/v1/code/status", "/api/v1/re/status", "/api/v1/gimp/status"):
        response = client.get(path, headers=hdr(env))
        assert response.status_code == 200, path
    assert "workspace" in client.get("/api/v1/code/status", headers=hdr(env)).json()
    assert "supported" in client.get("/api/v1/gimp/status", headers=hdr(env)).json()


def test_requirements_and_log_endpoints(client, env):
    response = client.get("/api/v1/requirements", headers=hdr(env))
    assert response.status_code == 200
    payload = response.json()
    ids = {component["id"] for component in payload["components"]}
    assert ids == {"ovms", "node", "opencode", "python", "gimp", "gimp_plugin"}
    assert payload["tools_dir"].endswith("tools")
    assert "gimp_dir" in payload

    log = client.get("/api/v1/logs/setup", headers=hdr(env))
    assert log.status_code == 200
    assert "path" in log.json()

    unknown = client.get("/api/v1/logs/..%2Fsecret", headers=hdr(env))
    assert unknown.status_code == 404


def test_setup_view_and_embed_controls_are_present(client):
    page = client.get("/").text
    assert 'id="view-setup"' in page
    assert 'data-view="setup"' in page
    assert 'id="setup-components"' in page
    assert 'id="code-frame"' in page
    assert "/api/v1/code/embed" not in page  # endpoints live in JS, not markup
    assert 'id="log-view"' in page

def test_delete_conversation_and_summary(env):
    db = env["app"].state.ctx.db
    cid = "conv-delete-api"
    store.ensure_conversation(db, cid)
    mid, _ = store.archive_message(db, conversation_id=cid, role="user", content="delete me")
    sid = store.insert_summary(db, title="Delete me", body="temporary", message_ids=[mid])

    response = env["client"].delete(f"/api/v1/conversations/{cid}", headers=hdr(env))
    assert response.status_code == 200
    assert store.conversation_messages(db, cid) == []
    assert store.get_summary(db, sid) is not None

    response = env["client"].delete(f"/api/v1/memory/summaries/{sid}", headers=hdr(env))
    assert response.status_code == 200
    assert store.get_summary(db, sid) is None

    assert env["client"].delete("/api/v1/memory/summaries/missing", headers=hdr(env)).status_code == 404


def test_delete_gallery_image_removes_file(env):
    db = env["app"].state.ctx.db
    image_dir = env["paths"].image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / "delete-me.png"
    path.write_bytes(b"png")
    with db.write() as c:
        c.execute(
            """INSERT INTO images(id, prompt, path, created_at) VALUES(?,?,?,?)""",
            ("img-delete", "test", "delete-me.png", 1),
        )

    response = env["client"].delete("/api/v1/images/img-delete", headers=hdr(env))
    assert response.status_code == 200
    assert not path.exists()
    assert env["client"].get("/api/v1/images/img-delete/file", headers=hdr(env)).status_code == 404


def test_memory_breakdown_header_is_json(client, env):
    response = client.post(
        "/v1/chat/completions",
        headers=hdr(env, **{"X-AIO-Conversation": "conv-memory-header"}),
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Remember the green lamp."}],
            "stream": True,
        },
    )
    assert response.status_code == 200

    breakdown = json.loads(response.headers["x-aio-memory-breakdown"])
    assert isinstance(breakdown, dict)
    assert response.headers["x-aio-memory-items"] == str(sum(breakdown.values()))


def test_custom_model_scan_and_assign_endpoints(client, env):
    from aiolocal.models import GRAPH_TEMPLATE

    ctx = env["app"].state.ctx
    folder = ctx.models.models_root() / "custom-endpoint-model"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (folder / "openvino_model.bin").write_bytes(b"0" * 2_000_000)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "graph.pbtxt").write_text(GRAPH_TEMPLATE, encoding="utf-8")

    scan = client.post("/api/v1/models/custom/scan", headers=hdr(env))
    assert scan.status_code == 200, scan.text
    fresh = scan.json()["new"]
    assert len(fresh) == 1
    model_id = fresh[0]["id"]

    assign = client.post(
        "/api/v1/models/custom/assign",
        headers=hdr(env),
        json={"model": model_id, "role": "chat", "device": "CPU"},
    )
    assert assign.status_code == 200, assign.text
    result = assign.json()
    assert result["role"] == "chat"
    assert result["device"] == "CPU"
    assert result["settings"]["chat_model"] == model_id

    status = client.get("/api/v1/models", headers=hdr(env)).json()
    custom = next(m for m in status["models"] if m["id"] == model_id)
    assert custom["custom"] is True
    assert custom["role_status"] == "assigned"
    assert status["pending_models"] == []

    removed = client.post(
        "/api/v1/models/custom/remove",
        headers=hdr(env),
        json={"model": model_id},
    )
    assert removed.status_code == 200, removed.text
    assert folder.exists()  # removing from the list does not delete the folder


def test_custom_model_remove_clears_stale_model_references(client, env):
    from aiolocal.models import GRAPH_TEMPLATE

    ctx = env["app"].state.ctx
    folder = ctx.models.models_root() / "stale-chat-model"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (folder / "openvino_model.bin").write_bytes(b"0" * 2_000_000)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "graph.pbtxt").write_text(GRAPH_TEMPLATE, encoding="utf-8")

    model_id = client.post("/api/v1/models/custom/scan", headers=hdr(env)).json()["new"][0]["id"]
    client.post(
        "/api/v1/models/custom/assign",
        headers=hdr(env),
        json={"model": model_id, "role": "chat", "device": "CPU"},
    )

    removed = client.post(
        "/api/v1/models/custom/remove",
        headers=hdr(env),
        json={"model": model_id},
    )
    assert removed.status_code == 200, removed.text
    result = removed.json()
    assert "chat_model" in result["cleared_settings"]
    assert result["settings"]["chat_model"] != model_id
    assert ctx.settings.chat_model != model_id

    config = json.loads((env["paths"].config_dir / "opencode.json").read_text(encoding="utf-8"))
    assert model_id not in config["provider"]["aiolocal"]["models"]
    assert config["model"] != f"aiolocal/{model_id}"

    # A normal status poll must not re-register the folder the user removed.
    ctx.models._last_scan = 0.0
    ctx.models.scan_custom_models()
    status = client.get("/api/v1/models", headers=hdr(env)).json()
    assert not any(model["id"] == model_id for model in status["models"])
    assert status["pending_models"] == []


def test_chat_model_setting_updates_the_opencode_config(client, env):
    response = client.put(
        "/api/v1/settings",
        headers=hdr(env),
        json={"chat_model": "qwen3-4b"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["settings"]["chat_model"] == "qwen3-4b"
    assert "opencode_config" in body
    config = json.loads((env["paths"].config_dir / "opencode.json").read_text(encoding="utf-8"))
    assert config["model"] == "aiolocal/qwen3-4b"
