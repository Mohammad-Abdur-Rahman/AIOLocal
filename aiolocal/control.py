"""Control plane used by the bundled UI.

These routes are local-only, token-protected and never expose prompts in logs. They are
separate from the OpenAI-compatible gateway so a future client can use `/v1` without
needing the control API.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .context import AppContext
from .db import now_ms
from .errors import safe_failure
from .llm import validate_base_url
from .memory import store, tree
from .memory.retrieve import retrieve
from .security import valid_id

router = APIRouter(prefix="/api/v1")


def _err(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "request_error", "code": code}},
    )


def _fail(
    status: int, exc: BaseException, action: str, message: str, code: str
) -> JSONResponse:
    """Log ``exc`` and respond with a curated message that never contains it."""
    return _err(status, safe_failure(exc, action, message), code)


@router.get("/health")
async def health(request: Request):
    ctx: AppContext = request.app.state.ctx
    upstream = await ctx.upstream.health()
    return {
        "ok": True,
        "version": __version__,
        "uptime_s": int((now_ms() - ctx.started_at) / 1000),
        "upstream": upstream,
        "stats": ctx.db.stats(),
    }


@router.get("/state")
async def state(request: Request):
    ctx: AppContext = request.app.state.ctx
    upstream = await ctx.upstream.health()
    return {
        "version": __version__,
        "settings": ctx.settings.to_dict(),
        "stats": ctx.db.stats(),
        "jobs": store.job_stats(ctx.db),
        "models": ctx.models.status(upstream),
        "paths": {
            "data": str(ctx.paths.data_dir),
            "database": str(ctx.paths.db_path),
            "images": str(ctx.paths.image_dir),
            "logs": str(ctx.paths.log_dir),
        },
    }


# ------------------------------------------------------------------------- settings
@router.get("/settings")
async def get_settings(request: Request):
    ctx: AppContext = request.app.state.ctx
    return {"settings": ctx.settings.to_dict()}


@router.put("/settings")
async def put_settings(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    if not isinstance(body, dict):
        return _err(400, "body must be an object", "invalid_body")

    merged = {**ctx.settings.to_dict(), **body}
    try:
        from .config import Settings

        new_settings = Settings.from_dict(merged)
        new_settings.upstream_base_url = validate_base_url(new_settings.upstream_base_url)
    except (ValueError, TypeError) as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "validating settings",
                "Some settings are invalid. Check numeric fields, URLs, paths and model IDs.",
            ),
            "invalid_settings",
        )

    if new_settings.models_dir:
        try:
            from pathlib import Path

            Path(new_settings.models_dir).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return _err(
                400,
                safe_failure(
                    exc,
                    "creating the models folder",
                    "That folder cannot be used for models. Choose a writable folder.",
                ),
                "invalid_models_dir",
            )

    previous_chat = ctx.settings.chat_model
    ctx.update_settings(new_settings)
    result: dict = {"ok": True, "settings": new_settings.to_dict()}
    if new_settings.chat_model != previous_chat and ctx.code_ws is not None:
        # OpenCode's config names the default model, so it must follow the chat
        # dropdown without waiting for a restart.
        try:
            result["opencode_config"] = str(ctx.code_ws.write_config())
        except OSError as exc:
            result["opencode_error"] = safe_failure(
                exc,
                "writing the OpenCode config",
                "The settings were saved, but the OpenCode config could not be written. See the app log.",
            )
    ctx.bus.publish("settings_changed", {})
    return result


# ------------------------------------------------------------------------- chats
@router.get("/conversations")
async def conversations(request: Request, limit: int = 100):
    ctx: AppContext = request.app.state.ctx
    return {"data": store.list_conversations(ctx.db, limit=max(1, min(int(limit), 300)))}


@router.post("/conversations")
async def new_conversation(request: Request):
    ctx: AppContext = request.app.state.ctx
    from .memory.text import new_id

    conversation_id = new_id()
    store.ensure_conversation(ctx.db, conversation_id, source="api")
    return {"id": conversation_id}


@router.get("/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(conversation_id):
        return _err(400, "invalid conversation id", "invalid_id")
    return {"data": store.conversation_messages(ctx.db, conversation_id, limit=1000)}


@router.delete("/conversations/{conversation_id}")
async def conversation_delete(conversation_id: str, request: Request, with_summaries: bool = False):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(conversation_id):
        return _err(400, "invalid conversation id", "invalid_id")
    if not store.delete_conversation(
        ctx.db, conversation_id, with_summaries=with_summaries
    ):
        return _err(404, "conversation not found", "not_found")
    return {"ok": True, "deleted": conversation_id, "with_summaries": with_summaries}


# --------------------------------------------------------------------------- tasks
@router.get("/tasks")
async def tasks_list(request: Request, conversation_id: str = "", limit: int = 50):
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None:
        return _err(503, "task drafting is not running", "unavailable")
    return {
        "data": ctx.tasks.list(
            conversation_id=conversation_id, limit=max(1, min(int(limit), 200))
        )
    }


@router.post("/tasks")
async def tasks_start(request: Request):
    """Start a drafting job explicitly (the chat route auto-detects bulk jobs)."""
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None:
        return _err(503, "task drafting is not running", "unavailable")
    try:
        body = await request.json()
    except Exception:
        return _err(400, "request body is not valid JSON", "invalid_json")
    conversation_id = str(body.get("conversation_id") or "")
    instruction = str(body.get("instruction") or "").strip()
    if not conversation_id or not instruction:
        return _err(400, "conversation_id and instruction are required", "invalid_request")
    document_ids = [str(d) for d in (body.get("document_ids") or [])]
    if not document_ids:
        document_ids = store.conversation_documents(ctx.db, conversation_id)
    if not document_ids:
        return _err(409, "attach at least one analysed document to this chat first", "no_documents")
    topics = [str(t) for t in (body.get("topics") or [])]
    try:
        task = ctx.tasks.start(
            conversation_id=conversation_id,
            instruction=instruction,
            topics=topics,
            document_ids=document_ids,
            model=str(body.get("model") or ""),
            title=str(body.get("title") or ""),
        )
    except Exception as exc:
        return _err(409, f"could not start the task ({exc})", "task_unavailable")
    return {"ok": True, "task": task}


@router.get("/tasks/{task_id}")
async def tasks_detail(task_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None:
        return _err(503, "task drafting is not running", "unavailable")
    if not valid_id(task_id):
        return _err(400, "invalid task id", "invalid_id")
    task = ctx.tasks.get(task_id)
    if task is None:
        task_row = next(
            (t for t in ctx.tasks.list(limit=200) if t["id"] == task_id), None
        )
        if task_row is None:
            return _err(404, "task not found", "not_found")
        task = task_row
    return task


@router.get("/tasks/{task_id}/draft")
async def tasks_draft(task_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None:
        return _err(503, "task drafting is not running", "unavailable")
    if not valid_id(task_id):
        return _err(400, "invalid task id", "invalid_id")
    draft = ctx.tasks.draft(task_id)
    if draft is None:
        return _err(404, "task not found", "not_found")
    return draft


@router.post("/tasks/{task_id}/pause")
async def tasks_pause(task_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None or not valid_id(task_id):
        return _err(400, "invalid task", "invalid_id")
    if not ctx.tasks.pause(task_id):
        return _err(409, "only a queued or running task can be paused", "invalid_state")
    return {"ok": True, "state": "paused"}


@router.post("/tasks/{task_id}/resume")
async def tasks_resume(task_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None or not valid_id(task_id):
        return _err(400, "invalid task", "invalid_id")
    if not ctx.tasks.resume(task_id):
        return _err(409, "only a paused, interrupted or failed task can be resumed", "invalid_state")
    return {"ok": True, "state": "running"}


@router.post("/tasks/{task_id}/cancel")
async def tasks_cancel(task_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if ctx.tasks is None or not valid_id(task_id):
        return _err(400, "invalid task", "invalid_id")
    if not ctx.tasks.cancel(task_id):
        return _err(404, "task not found", "not_found")
    return {"ok": True, "state": "cancelling"}


# ------------------------------------------------------------------------- memory
@router.get("/memory/trees")
async def memory_trees(request: Request, kind: str = "", limit: int = 50):
    ctx: AppContext = request.app.state.ctx
    kind = kind.strip().lower()
    if kind not in ("", "chat", "document"):
        return _err(400, "kind must be chat or document", "invalid_kind")
    return {
        "data": tree.list_trees(
            ctx.db, kind=kind or None, limit=max(1, min(int(limit), 200))
        )
    }


@router.get("/memory/trees/{tree_id}")
async def memory_tree_detail(tree_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(tree_id):
        return _err(400, "invalid tree id", "invalid_id")
    detail = tree.outline(ctx.db, tree_id)
    if not detail.get("tree"):
        return _err(404, "tree not found", "not_found")
    detail["stats"] = tree.stats(ctx.db, tree_id)
    return detail


@router.post("/memory/trees/{tree_id}/rollup")
async def memory_tree_rollup(tree_id: str, request: Request, max_nodes: int = 6):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(tree_id):
        return _err(400, "invalid tree id", "invalid_id")
    if ctx.trees is None:
        return _err(503, "the tree builder is not running", "unavailable")
    created = ctx.trees.drain(tree_id, max_nodes=max(1, min(int(max_nodes), 50)))
    return {"ok": True, "created": created, "stats": tree.stats(ctx.db, tree_id)}


@router.delete("/memory/trees/{tree_id}")
async def memory_tree_delete(tree_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(tree_id):
        return _err(400, "invalid tree id", "invalid_id")
    result = tree.delete_tree(ctx.db, tree_id)
    if not result:
        return _err(404, "tree not found", "not_found")
    return {"ok": True, **result}


@router.post("/memory/trees/clear")
async def memory_trees_clear(request: Request, kind: str = ""):
    """Delete every chat or document tree (both when kind is empty)."""
    ctx: AppContext = request.app.state.ctx
    kind = kind.strip().lower()
    if kind not in ("", "chat", "document"):
        return _err(400, "kind must be chat, document or empty", "invalid_kind")
    result = tree.delete_trees(ctx.db, kind=kind or None)
    return {"ok": True, **result}


@router.get("/memory/summaries")
async def summaries(request: Request, limit: int = 50, offset: int = 0, topic: str | None = None):
    ctx: AppContext = request.app.state.ctx
    if topic and len(topic) > 80:
        return _err(400, "invalid topic", "invalid_topic")
    return {
        "data": store.list_summaries(
            ctx.db, limit=max(1, min(int(limit), 200)), offset=max(0, int(offset)), topic=topic
        )
    }


@router.get("/memory/summaries/{summary_id}")
async def summary_detail(summary_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(summary_id):
        return _err(400, "invalid summary id", "invalid_id")
    summary = store.get_summary(ctx.db, summary_id)
    if not summary:
        return _err(404, "summary not found", "not_found")
    message_ids = store.summary_message_ids(ctx.db, summary_id, limit=50)
    messages = store.messages_by_ids(ctx.db, message_ids)
    return {"summary": summary, "messages": messages}


@router.delete("/memory/summaries/{summary_id}")
async def summary_delete(summary_id: str, request: Request, with_messages: bool = True):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(summary_id):
        return _err(400, "invalid summary id", "invalid_id")
    if ctx.documents:
        for job in ctx.documents.recent(limit=100):
            if job.get("summary_id") == summary_id and job.get("state") in ("queued", "running"):
                ctx.documents.cancel(job["id"])
    if not store.delete_summary(ctx.db, summary_id, with_messages=with_messages):
        return _err(404, "summary not found", "not_found")
    return {"ok": True, "deleted": summary_id, "with_messages": with_messages}


@router.get("/memory/topics")
async def topics(request: Request):
    ctx: AppContext = request.app.state.ctx
    return {"data": store.list_topics(ctx.db)}


@router.get("/memory/jobs")
async def jobs(request: Request, limit: int = 50):
    ctx: AppContext = request.app.state.ctx
    with ctx.db.read() as c:
        rows = c.execute(
            """SELECT id, kind, state, attempts, last_error, run_after, created_at
               FROM memory_jobs ORDER BY created_at DESC LIMIT ?""",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    return {"data": [dict(r) for r in rows], "counts": store.job_stats(ctx.db)}


@router.post("/memory/search")
async def memory_search(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    query = str((body or {}).get("query") or "").strip()
    if not query:
        return _err(400, "query is required", "empty_query")
    if len(query) > 4000:
        return _err(400, "query is too long", "query_too_long")
    result = retrieve(ctx.db, query, settings=ctx.settings, request_id=None, force_lookup=True)
    return {
        "intent": result.intent,
        "latency_ms": result.latency_ms,
        "plan": result.plan,
        "data": [item.to_dict() for item in result.items],
    }


@router.post("/memory/consolidate")
async def consolidate_now(request: Request):
    ctx: AppContext = request.app.state.ctx
    job_id = store.queue_job(ctx.db, "consolidate", {})
    if ctx.worker:
        ctx.worker.notify()
    return {"ok": True, "job": job_id}


@router.post("/memory/reembed")
async def memory_reembed(request: Request, limit: int = 5000):
    """Queue semantic re-embedding of existing memory with the configured model."""
    ctx: AppContext = request.app.state.ctx
    from .memory import semantic

    if not semantic.enabled(ctx.settings):
        return _err(
            400,
            "Semantic embeddings are off. Choose the OVMS embedding backend in Settings first.",
            "semantic_disabled",
        )
    if ctx.worker:
        job_id = ctx.worker.enqueue_reembed(limit)
    else:
        job_id = store.queue_job(ctx.db, "reembed", {"limit": int(limit)})
    return {
        "ok": True,
        "job": job_id,
        "model": semantic.model_name(ctx.settings),
        "limit": max(1, min(int(limit), 50_000)),
    }


@router.get("/inspector/{request_id}")
async def inspector(request_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(request_id):
        return _err(400, "invalid request id", "invalid_id")
    row = store.retrieval_plan(ctx.db, request_id)
    if not row:
        return _err(404, "no retrieval record for that request", "not_found")
    with ctx.db.read() as c:
        messages = c.execute(
            """SELECT id, role, content, created_at FROM messages
               WHERE request_id=? AND injected=0 ORDER BY created_at ASC""",
            (request_id,),
        ).fetchall()
    return {"retrieval": row, "messages": [dict(m) for m in messages]}


# ------------------------------------------------------------------------- models
@router.get("/models")
async def models(request: Request):
    ctx: AppContext = request.app.state.ctx
    upstream = await ctx.upstream.health()
    return ctx.models.status(upstream)


@router.post("/models/{model_id}/load")
async def load_model(model_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        body = {}
    device = str((body or {}).get("device") or "").upper()
    entry = ctx.models.entry(model_id)
    if entry is None:
        return _err(400, "Unknown model. Refresh the Models view and try again.", "model_error")
    if ctx.models.pull_active(model_id):
        return _err(400, "That model is still downloading. Wait for it to finish.", "model_error")
    if not ctx.models.installed(entry):
        return _err(400, "That model is not downloaded yet. Download it in Models first.", "model_error")
    if device and device not in (entry.get("devices") or []):
        allowed = ", ".join(entry.get("devices") or [])
        return _err(400, f"That device is not supported for this model. Choose {allowed}.", "model_error")
    candidates = [device]
    if entry:
        for fallback in entry.get("fallback_devices") or []:
            fallback = str(fallback).upper()
            if fallback and fallback not in candidates:
                candidates.append(fallback)
    try:
        result = ctx.models.load(model_id, device)
    except ValueError as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "loading the model",
                "The model could not be loaded. Check that its folder contains a complete "
                "OpenVINO graph and that the OVMS log does not report a compile error.",
            ),
            "model_error",
        )

    last_reload: dict = {}
    requested = result.get("device", device)
    verify = bool(entry and entry.get("fallback_devices"))
    for candidate in candidates:
        if candidate != requested:
            try:
                result = ctx.models.load(model_id, candidate)
            except ValueError as exc:
                last_reload = {
                    "ok": False,
                    "message": safe_failure(
                        exc,
                        f"loading the model on {candidate}",
                        f"The model could not be loaded on {candidate}.",
                    ),
                }
                continue
        ok, message = await ctx.models.reload(ctx.upstream)
        last_reload = {"ok": ok, "message": message}
        if not ok:
            continue
        if not verify:
            result["verified"] = True
            result["reload"] = last_reload
            break
        # OVMS reload returns 200 even when a graph fails to initialise. Wait for
        # the model to appear in /v1/models before declaring success.
        loaded = await _wait_for_model(ctx, model_id, timeout=35.0 if candidate == requested else 45.0)
        result["verified"] = loaded
        if loaded:
            result["reload"] = last_reload
            if candidate != requested:
                result["fallback"] = {
                    "from": requested,
                    "to": candidate,
                    "reason": f"{requested} did not become available; loaded on {candidate} instead.",
                }
                result["message"] = result["fallback"]["reason"]
            break
    result["reload"] = last_reload
    if not result.get("verified") and entry and entry.get("fallback_devices"):
        result["message"] = (
            f"{model_id} did not become available on {requested} or its fallback devices. "
            "Check the OVMS log for the compilation error."
        )
    ctx.bus.publish(
        "model_loaded",
        {"model": model_id, "device": result.get("device", requested), "fallback": result.get("fallback")},
    )
    return result


async def _wait_for_model(ctx: AppContext, model_id: str, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        health = await ctx.upstream.health()
        if health.get("ok") and model_id in (health.get("models") or []):
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.75)


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    result = ctx.models.unload(model_id)
    ok, message = await ctx.models.reload(ctx.upstream)
    ctx.bus.publish("model_unloaded", {"model": model_id})
    return {**result, "reload": {"ok": ok, "message": message}}


@router.post("/models/{model_id}/pull")
async def pull_model(model_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx

    def publish(kind: str, data: dict) -> None:
        ctx.bus.publish(kind, data)

    try:
        result = ctx.models.start_pull(model_id, publish=publish)
    except ValueError as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "queueing the model download",
                "The download could not be started. Check that the model exists and that "
                "the models folder is writable.",
            ),
            "model_error",
        )
    return result


# ------------------------------------------------------------------------- custom models
@router.post("/models/custom/download")
async def custom_model_download(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        body = {}
    url = str((body or {}).get("url") or (body or {}).get("repo") or "").strip()
    if not url:
        return _err(400, "enter a Hugging Face repository URL", "invalid_repo")

    def publish(kind: str, data: dict) -> None:
        ctx.bus.publish(kind, data)

    try:
        result = ctx.models.start_custom_pull(url, publish=publish)
    except ValueError as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "starting the custom model download",
                "Enter a full Hugging Face model URL, for example "
                "https://huggingface.co/OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov.",
            ),
            "model_error",
        )
    return result


@router.post("/models/custom/scan")
async def custom_model_scan(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        fresh = ctx.models.scan_custom_models(force=True)
    except Exception as exc:
        return _err(
            500,
            safe_failure(
                exc,
                "scanning the models folder",
                "The models folder could not be scanned. Check that it exists and is readable.",
            ),
            "scan_error",
        )
    return {
        "ok": True,
        "new": fresh,
        "pending": ctx.models.pending_entries(),
        "models_dir": str(ctx.models.models_root()),
    }


@router.post("/models/custom/assign")
async def custom_model_assign(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    if not isinstance(body, dict):
        return _err(400, "body must be an object", "invalid_body")
    model_id = str(body.get("model") or body.get("id") or "").strip()
    role = str(body.get("role") or "").strip()
    if not model_id or not role:
        return _err(400, "model and role are required", "invalid_body")
    try:
        result = ctx.models.assign_role(
            model_id,
            role,
            vision=body.get("vision") if isinstance(body.get("vision"), bool) else None,
            device=str(body.get("device") or ""),
        )
    except ValueError as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "assigning the custom model",
                "The model could not be assigned. Check that its folder is still a valid "
                "OpenVINO IR model.",
            ),
            "model_error",
        )

    from .config import Settings

    try:
        new_settings = Settings.from_dict(result.get("settings") or {})
        ctx.update_settings(new_settings)
    except (ValueError, TypeError) as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "applying the role settings",
                "The model was found, but the resulting settings were invalid. See the app log.",
            ),
            "invalid_settings",
        )

    if result.get("opencode_needs_reload") and ctx.code_ws is not None:
        try:
            result["opencode_config"] = str(ctx.code_ws.write_config())
        except OSError as exc:
            result["opencode_error"] = safe_failure(
                exc,
                "writing the OpenCode config",
                "The role was assigned, but the OpenCode config could not be written. See the app log.",
            )
    if ctx.models.ovms_running():
        ok, message = await ctx.models.reload(ctx.upstream)
        result["reload"] = {"ok": ok, "message": message}
    ctx.bus.publish("model_loaded", {"model": model_id, "role": result.get("role")})
    ctx.bus.publish("settings_changed", {})
    return result


@router.post("/models/custom/remove")
async def custom_model_remove(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        body = await request.json()
    except Exception:
        body = {}
    model_id = str((body or {}).get("model") or "").strip()
    if not model_id:
        return _err(400, "model is required", "invalid_body")
    try:
        result = ctx.models.remove_custom(model_id)
    except ValueError as exc:
        return _err(
            400,
            safe_failure(
                exc,
                "removing the custom model",
                "That model could not be removed from the list.",
            ),
            "model_error",
        )
    if result.get("settings"):
        from .config import Settings

        try:
            ctx.update_settings(Settings.from_dict(result["settings"]))
        except (ValueError, TypeError):
            pass
    if ctx.code_ws is not None:
        try:
            result["opencode_config"] = str(ctx.code_ws.write_config())
        except OSError as exc:
            result["opencode_error"] = safe_failure(
                exc,
                "writing the OpenCode config",
                "The model was removed, but the OpenCode config could not be written. See the app log.",
            )
    if ctx.models.ovms_running():
        ok, message = await ctx.models.reload(ctx.upstream)
        result["reload"] = {"ok": ok, "message": message}
    ctx.bus.publish("settings_changed", {})
    return result


@router.post("/ovms/start")
async def ovms_start(request: Request):
    ctx: AppContext = request.app.state.ctx
    try:
        return ctx.models.start_ovms()
    except ValueError as exc:
        return _fail(
            400,
            exc,
            "starting OVMS",
            "OVMS could not start. Check the OVMS path, the model config and the selected port.",
            "ovms_error",
        )
    except OSError as exc:
        return _fail(
            500,
            exc,
            "starting OVMS",
            "OVMS could not be launched. Check that its executable exists and that the port is free.",
            "ovms_error",
        )


@router.post("/ovms/stop")
async def ovms_stop(request: Request):
    ctx: AppContext = request.app.state.ctx
    return ctx.models.stop_ovms()


@router.post("/ovms/reload")
async def ovms_reload(request: Request):
    ctx: AppContext = request.app.state.ctx
    ctx.models.write_config()
    ok, message = await ctx.models.reload(ctx.upstream)
    return {"ok": ok, "message": message}


# ------------------------------------------------------------------------- knowledge
def _kb_unavailable():
    return _err(503, "the knowledge base is not available", "kb_unavailable")


@router.get("/knowledge/status")
async def knowledge_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    payload = {
        "stats": ctx.kb.stats(),
        "collections": ctx.kb.list_collections(),
        "inbox": str(ctx.paths.knowledge_inbox),
        "files": str(ctx.paths.knowledge_files),
    }
    if ctx.kb_worker is not None:
        payload["worker"] = ctx.kb_worker.status()
    return payload


@router.get("/knowledge/collections")
async def knowledge_collections(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    return {"data": ctx.kb.list_collections()}


@router.post("/knowledge/collections")
async def knowledge_collection_create(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    slug = str((body or {}).get("slug") or "").strip()
    name = str((body or {}).get("name") or "").strip() or None
    if not slug:
        return _err(400, "collection name is required", "empty_collection")
    if len(slug) > 80:
        return _err(400, "collection name is too long", "invalid_collection")
    collection_id = ctx.kb.ensure_collection(slug, name)
    return {"ok": True, "id": collection_id, "collections": ctx.kb.list_collections()}


@router.delete("/knowledge/collections/{slug:path}")
async def knowledge_collection_delete(slug: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    if not ctx.kb.delete_collection(slug):
        return _err(404, "collection not found", "not_found")
    return {"ok": True, "collections": ctx.kb.list_collections()}


@router.post("/knowledge/folders/rebuild")
async def knowledge_folders_rebuild(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    try:
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(
            ctx.kb.rebuild_folder_summaries, embed_dim=ctx.settings.embed_dim
        )
    except Exception as exc:
        return _fail(
            500,
            exc,
            "rebuilding folder summaries",
            "Folder summaries could not be rebuilt. Check the app log for details.",
            "folder_rebuild_failed",
        )
    return {"ok": True, **result, "collections": ctx.kb.list_collections()}


@router.get("/knowledge/documents")
async def knowledge_documents(
    request: Request, collection: str | None = None, limit: int = 100
):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    return {"data": ctx.kb.list_documents(collection, limit=max(1, min(int(limit), 500)))}


@router.get("/knowledge/documents/{document_id}")
async def knowledge_document(document_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    if not valid_id(document_id):
        return _err(400, "invalid document id", "invalid_id")
    doc = ctx.kb.get_document(document_id)
    if not doc:
        return _err(404, "document not found", "not_found")
    return {"document": doc, "chunks": ctx.kb.list_chunks(document_id, limit=200)}


@router.delete("/knowledge/documents/{document_id}")
async def knowledge_document_delete(document_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    if not valid_id(document_id):
        return _err(400, "invalid document id", "invalid_id")
    if not ctx.kb.delete_document(document_id):
        return _err(404, "document not found", "not_found")
    return {"ok": True}


@router.post("/knowledge/upload")
async def knowledge_upload(request: Request, collection: str = "general", filename: str = ""):
    """Stream a raw-body upload straight to disk.

    The browser sends the bytes with ``Content-Type: application/octet-stream`` and
    passes the original filename in the query string. Nothing is buffered in memory,
    so a 400 MB JSONL dataset works exactly like a small Markdown file.
    """
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    from .knowledge import MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, KnowledgeError

    clean_name = Path(str(filename or "upload.txt")).name
    suffix = Path(clean_name).suffix.lower()
    if not clean_name or suffix not in SUPPORTED_EXTENSIONS:
        shown = suffix or "(none)"
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return _err(
            400,
            f"unsupported file type {shown}; supported: {supported}",
            "invalid_file",
        )
    try:
        temporary, target, slug = ctx.kb.upload_target(
            clean_name, collection or "general"
        )
    except KnowledgeError as exc:
        return _fail(
            400,
            exc,
            "preparing the knowledge upload",
            "That file cannot be imported. Use a text, Markdown, JSON or PDF file with a valid name.",
            "invalid_file",
        )
    received = 0
    try:
        with open(temporary, "wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                received += len(chunk)
                if received > MAX_FILE_BYTES:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                    limit = MAX_FILE_BYTES // (1024 * 1024)
                    return _err(
                        400,
                        f"file is too large (max {limit} MB)",
                        "invalid_file",
                    )
                handle.write(chunk)
        if received == 0:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return _err(400, "the uploaded file is empty", "invalid_file")
        os.replace(temporary, target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return _fail(
            500,
            exc,
            "storing the knowledge upload",
            "The upload could not be stored. Check that the knowledge folder is writable and has free space.",
            "upload_failed",
        )

    try:
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(
            ctx.kb.ingest_path,
            target,
            slug,
            embed_dim=ctx.settings.embed_dim,
            docling_path=ctx.settings.docling_path,
        )
    except Exception as exc:
        return _fail(
            400,
            exc,
            "importing the knowledge document",
            "The document could not be imported. Check that it contains readable text and is not encrypted.",
            "import_failed",
        )
    ctx.bus.publish("knowledge_document", {"state": "added", "document": result})
    return {
        "ok": True,
        "document": result,
        "bytes": received,
        "stats": ctx.kb.stats(),
    }


@router.post("/knowledge/search")
async def knowledge_search(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    body = body or {}
    query = str(body.get("query") or "").strip()
    if not query:
        return _err(400, "query is required", "empty_query")
    collections = body.get("collections") or []
    if not isinstance(collections, list):
        return _err(400, "collections must be a list", "invalid_collections")
    try:
        limit = max(1, min(int(body.get("limit") or ctx.settings.knowledge_top_k), 20))
    except (TypeError, ValueError):
        limit = ctx.settings.knowledge_top_k
    auto = bool(body.get("auto")) or any(str(c).strip().lower() == "auto" for c in collections)
    try:
        if auto:
            routed = ctx.kb.search_auto(
                query,
                limit=limit,
                embed_dim=ctx.settings.embed_dim,
                max_folders=ctx.settings.knowledge_auto_folders,
            )
            return {
                "data": routed.get("items") or [],
                "routing": routed.get("routing") or {},
                "query": query,
            }
        rows = ctx.kb.search(
            query,
            collections=[str(c) for c in collections if str(c).strip()],
            limit=limit,
            embed_dim=ctx.settings.embed_dim,
        )
    except Exception as exc:
        return _fail(
            500,
            exc,
            "searching the knowledge base",
            "The knowledge search failed. Check the app log; the knowledge index may need a rebuild.",
            "search_failed",
        )
    return {"data": rows, "query": query}


@router.post("/knowledge/enrich")
async def knowledge_enrich(request: Request):
    """Run one enrichment batch now instead of waiting for the worker cycle."""
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    if ctx.kb_worker is None:
        return _err(503, "the knowledge worker is not running", "worker_unavailable")
    try:
        from starlette.concurrency import run_in_threadpool

        count = await run_in_threadpool(ctx.kb_worker.enrich, ctx.settings)
    except Exception as exc:
        return _fail(
            500,
            exc,
            "enriching knowledge documents",
            "Enrichment failed. Check the app log; the memory chat model may not be loaded.",
            "enrich_failed",
        )
    return {"ok": True, "enriched": count, "stats": ctx.kb.stats()}


@router.post("/knowledge/scan")
async def knowledge_scan(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.kb:
        return _kb_unavailable()
    limit = 0
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("limit") is not None:
            limit = max(0, min(int(body.get("limit") or 0), 100000))
    except Exception:
        limit = 0
    try:
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(
            ctx.kb.scan_inbox,
            ctx.paths.knowledge_inbox,
            embed_dim=ctx.settings.embed_dim,
            limit=limit,
            docling_path=ctx.settings.docling_path,
        )
        if result.get("added"):
            await run_in_threadpool(
                ctx.kb.rebuild_folder_summaries, embed_dim=ctx.settings.embed_dim
            )
    except Exception as exc:
        return _fail(
            500,
            exc,
            "scanning the knowledge inbox",
            "The inbox could not be scanned. Check that the inbox folder exists and is readable.",
            "scan_failed",
        )
    if ctx.kb_worker:
        ctx.kb_worker.notify()
    return {"ok": True, **result, "stats": ctx.kb.stats()}


# ------------------------------------------------------------------------- code workspace
@router.get("/code/status")
async def code_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    return ctx.code_ws.status()


@router.post("/code/start")
async def code_start(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    try:
        from starlette.concurrency import run_in_threadpool

        ctx.code_ws.write_config()
        result = await run_in_threadpool(ctx.code_ws.start)
    except ValueError as exc:
        return _fail(
            400,
            exc,
            "starting the code workspace",
            "OpenCode could not start. Check the workspace path and that the OpenCode runtime is installed.",
            "code_error",
        )
    except OSError as exc:
        return _fail(
            500,
            exc,
            "starting OpenCode",
            "OpenCode could not be started. Check that its executable exists and that the port is free.",
            "code_error",
        )
    ctx.bus.publish("code_status", {"running": True, "port": result.get("port")})
    return result


@router.post("/code/embed")
async def code_embed(request: Request):
    """Return a ready-to-embed OpenCode URL, starting the isolated server if needed."""
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    try:
        from starlette.concurrency import run_in_threadpool

        if not ctx.code_ws.running():
            ctx.code_ws.write_config()
            await run_in_threadpool(ctx.code_ws.start)
            ctx.bus.publish("code_status", {"running": True})
        status = ctx.code_ws.status()
    except ValueError as exc:
        return _fail(
            400,
            exc,
            "preparing the embedded code workspace",
            "OpenCode could not start. Check the workspace path and that the OpenCode runtime is installed.",
            "code_error",
        )
    except OSError as exc:
        return _fail(
            500,
            exc,
            "starting OpenCode for embedding",
            "OpenCode could not be started. Check that its executable exists and that the port is free.",
            "code_error",
        )
    return {
        "ok": True,
        "running": status["running"],
        "embed_url": status["embed_url"],
        "url": status["url"],
        "port": status["port"],
        "isolated": status["isolated"],
        "data_dir": status["data_dir"],
        "workspace": status["workspace"],
    }


@router.post("/code/stop")
async def code_stop(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    result = ctx.code_ws.stop()
    ctx.bus.publish("code_status", {"running": False})
    return result


@router.post("/code/workspace")
async def code_set_workspace(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    try:
        root = ctx.code_ws.set_workspace(str((body or {}).get("path") or ""))
    except ValueError as exc:
        return _fail(
            400,
            exc,
            "setting the code workspace",
            "That workspace folder cannot be used. Choose an existing folder that you can write to.",
            "invalid_workspace",
        )
    from dataclasses import replace

    ctx.update_settings(replace(ctx.settings, workspace_dir=str(root)))
    return {"ok": True, "workspace": str(root)}


@router.post("/code/config")
async def code_write_config(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    path = ctx.code_ws.write_config()
    return {
        "ok": True,
        "path": str(path),
        "note": "This file contains the loopback API token; do not copy it into a project.",
    }


@router.get("/code/files")
async def code_file_browser(request: Request, limit: int = 800):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    return {
        "root": str(ctx.code_ws.workspace()),
        "files": ctx.code_ws.list_files(limit=limit),
    }


@router.get("/code/file")
async def code_read_file(request: Request, path: str = ""):
    ctx: AppContext = request.app.state.ctx
    if not ctx.code_ws:
        return _err(503, "the code workspace is not available", "code_unavailable")
    try:
        return ctx.code_ws.read_file(path)
    except ValueError as exc:
        return _fail(
            400,
            exc,
            "reading a workspace file",
            "That file cannot be read. Check that it is inside the workspace and is a text file.",
            "read_error",
        )


# ------------------------------------------------------------------------- reverse engineering
@router.get("/re/status")
async def re_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.reverse:
        return _err(503, "the reverse-engineering module is not available", "re_unavailable")
    return ctx.reverse.status()


@router.post("/re/start")
async def re_start(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.reverse:
        return _err(503, "the reverse-engineering module is not available", "re_unavailable")
    try:
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(ctx.reverse.start)
    except Exception as exc:
        return _fail(
            500,
            exc,
            "starting REA",
            "REA could not start. Check the app log and the REA runtime settings.",
            "re_error",
        )
    ctx.bus.publish("re_status", {"running": True})
    return result


@router.post("/re/stop")
async def re_stop(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.reverse:
        return _err(503, "the reverse-engineering module is not available", "re_unavailable")
    result = ctx.reverse.stop()
    ctx.bus.publish("re_status", {"running": False})
    return result


@router.get("/re/tools")
async def re_tools(request: Request, refresh: int = 0):
    ctx: AppContext = request.app.state.ctx
    if not ctx.reverse:
        return _err(503, "the reverse-engineering module is not available", "re_unavailable")
    try:
        from starlette.concurrency import run_in_threadpool

        tools = await run_in_threadpool(ctx.reverse.tools, bool(refresh))
    except Exception as exc:
        return _fail(
            400,
            exc,
            "listing REA tools",
            "The REA tool list could not be read. Check that REA is installed and its MCP server starts.",
            "re_error",
        )
    return {"data": tools}


@router.post("/re/call")
async def re_call(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.reverse:
        return _err(503, "the reverse-engineering module is not available", "re_unavailable")
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    name = str((body or {}).get("name") or "").strip()
    arguments = (body or {}).get("arguments") or {}
    if not name:
        return _err(400, "tool name is required", "empty_tool")
    if not isinstance(arguments, dict):
        return _err(400, "arguments must be a JSON object", "invalid_arguments")
    try:
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(ctx.reverse.call, name, arguments)
    except Exception as exc:
        return _fail(
            400,
            exc,
            "calling an REA tool",
            "The REA tool call failed. Check the tool arguments and the app log.",
            "re_error",
        )
    return {"ok": True, "result": result}


# ------------------------------------------------------------------------- GIMP dock
@router.get("/gimp/status")
async def gimp_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    return ctx.gimp.status()


@router.post("/gimp/launch")
async def gimp_launch(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    try:
        body = await request.json()
    except Exception:
        body = {}
    image_id = (body or {}).get("image_id") or None
    embed = bool((body or {}).get("embed", ctx.settings.gimp_embed))
    try:
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(
            ctx.gimp.launch, image_id=str(image_id) if image_id else None, embed=embed
        )
    except Exception as exc:
        return _fail(
            400,
            exc,
            "launching the GIMP dock",
            "The GIMP dock could not launch. Check that GIMP is installed and that the plugin is repaired.",
            "gimp_error",
        )
    ctx.bus.publish("gimp_status", {"running": True})
    return result


@router.post("/gimp/release")
async def gimp_release(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    return ctx.gimp.release()


@router.post("/gimp/open")
async def gimp_open(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    image_id = str((body or {}).get("image_id") or "").strip()
    if not image_id:
        return _err(400, "image_id is required", "empty_image")
    try:
        return ctx.gimp.open_image(image_id)
    except Exception as exc:
        return _fail(
            400,
            exc,
            "opening an image in GIMP",
            "The image could not be opened in GIMP. Check that the image still exists and that GIMP is running.",
            "gimp_error",
        )


@router.post("/gimp/repair")
async def gimp_repair(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(ctx.gimp.repair)


@router.get("/gimp/photogimp")
async def photogimp_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    from .photogimp import PhotoGIMP

    return PhotoGIMP(ctx.paths, ctx.gimp).status()


@router.post("/gimp/photogimp/install")
async def photogimp_install(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    from starlette.concurrency import run_in_threadpool

    from .photogimp import PhotoGIMP, PhotoGimpError

    try:
        result = await run_in_threadpool(PhotoGIMP(ctx.paths, ctx.gimp).install)
    except PhotoGimpError as exc:
        return _fail(
            400,
            exc,
            "installing PhotoGIMP",
            "PhotoGIMP could not be installed. Check that GIMP is installed and that the download is reachable.",
            "photogimp_error",
        )
    except Exception as exc:
        return _fail(
            500,
            exc,
            "installing PhotoGIMP",
            "PhotoGIMP could not be installed. See the app log for details.",
            "photogimp_error",
        )
    ctx.bus.publish("gimp_status", {"photogimp": "installed"})
    return result


@router.post("/gimp/photogimp/remove")
async def photogimp_remove(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.gimp:
        return _err(503, "the GIMP dock is not available", "gimp_unavailable")
    from starlette.concurrency import run_in_threadpool

    from .photogimp import PhotoGIMP

    try:
        result = await run_in_threadpool(PhotoGIMP(ctx.paths, ctx.gimp).remove)
    except Exception as exc:
        return _fail(
            500,
            exc,
            "removing PhotoGIMP",
            "PhotoGIMP could not be removed. Check that its files are not open in GIMP.",
            "photogimp_error",
        )
    ctx.bus.publish("gimp_status", {"photogimp": "removed"})
    return result


# ------------------------------------------------------------------------- documents
MAX_DOCUMENT_BYTES = 512 * 1024 * 1024


def _documents_unavailable():
    return _err(503, "document analysis is not available", "documents_unavailable")


@router.get("/documents/jobs")
async def document_jobs(request: Request, limit: int = 20):
    ctx: AppContext = request.app.state.ctx
    if not ctx.documents:
        return _documents_unavailable()
    return {"data": ctx.documents.recent(limit=limit)}


@router.get("/documents/jobs/{job_id}")
async def document_job(job_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.documents:
        return _documents_unavailable()
    job = ctx.documents.get(job_id)
    if not job:
        return _err(404, "document job not found", "not_found")
    return job


@router.post("/documents/jobs/{job_id}/cancel")
async def document_job_cancel(job_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.documents:
        return _documents_unavailable()
    if not ctx.documents.cancel(job_id):
        return _err(404, "document job not found", "not_found")
    return {"ok": True}


@router.post("/documents/jobs/{job_id}/pause")
async def document_job_pause(job_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.documents:
        return _documents_unavailable()
    if not ctx.documents.pause(job_id):
        return _err(409, "only a queued or running job can be paused", "invalid_state")
    return {"ok": True, "state": "paused"}


@router.post("/documents/jobs/{job_id}/resume")
async def document_job_resume(job_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.documents:
        return _documents_unavailable()
    if not ctx.documents.resume(job_id):
        return _err(409, "only a paused job can be resumed", "invalid_state")
    return {"ok": True, "state": "running"}


@router.get("/documents/options")
async def document_options(request: Request, conversation_id: str = ""):
    """List analysed documents and whether this chat has opted into each."""
    ctx: AppContext = request.app.state.ctx
    return {
        "data": store.document_options(ctx.db, conversation_id),
        "conversation_id": conversation_id,
    }


@router.get("/documents/{document_id}/pages")
async def document_pages(
    document_id: str,
    request: Request,
    first: int = 1,
    last: int = 0,
    limit: int = 40,
):
    """Read stored page notes directly. Multi-page ranges are returned in order."""
    ctx: AppContext = request.app.state.ctx
    if not valid_id(document_id):
        return _err(400, "invalid document id", "invalid_id")
    summary = store.get_summary(ctx.db, document_id)
    if not summary or str(summary.get("kind") or "") != "document":
        return _err(404, "document not found", "not_found")
    low = max(1, int(first))
    high = int(last) if int(last) >= low else low
    cap = max(1, min(int(limit), 200))
    if high - low + 1 > cap:
        high = low + cap - 1
    with ctx.db.read() as c:
        rows = c.execute(
            """SELECT page_no, text, tokens FROM document_pages
               WHERE document_id=? AND page_no BETWEEN ? AND ?
               ORDER BY page_no""",
            (document_id, low, high),
        ).fetchall()
        total = c.execute(
            "SELECT COUNT(*) FROM document_pages WHERE document_id=?", (document_id,)
        ).fetchone()
    return {
        "document_id": document_id,
        "title": summary.get("title"),
        "first": low,
        "last": high,
        "pages_total": int(total[0]) if total else 0,
        "pages": [
            {"page_no": int(r["page_no"]), "text": str(r["text"]), "tokens": int(r["tokens"] or 0)}
            for r in rows
        ],
    }


@router.post("/documents/scope/{document_id}")
async def document_scope_add(document_id: str, request: Request, conversation_id: str = ""):
    """Opt this chat into a document. Only these documents are ever retrieved."""
    ctx: AppContext = request.app.state.ctx
    if not valid_id(document_id):
        return _err(400, "invalid document id", "invalid_id")
    if not conversation_id:
        return _err(400, "conversation_id is required", "invalid_conversation")
    summary = store.get_summary(ctx.db, document_id)
    if not summary or str(summary.get("kind") or "") != "document":
        return _err(404, "document not found", "not_found")
    store.attach_document(ctx.db, conversation_id, document_id)
    return {"ok": True, "attached": document_id, "conversation_id": conversation_id}


@router.delete("/documents/scope/{document_id}")
async def document_scope_remove(document_id: str, request: Request, conversation_id: str = ""):
    ctx: AppContext = request.app.state.ctx
    if not valid_id(document_id):
        return _err(400, "invalid document id", "invalid_id")
    if not conversation_id:
        return _err(400, "conversation_id is required", "invalid_conversation")
    removed = store.detach_document(ctx.db, conversation_id, document_id)
    return {"ok": True, "removed": removed, "conversation_id": conversation_id}


@router.post("/documents/analyze")
async def document_analyze(request: Request, filename: str = ""):
    """Stream a PDF upload to disk, then analyse every page with the vision model."""
    ctx: AppContext = request.app.state.ctx
    if not ctx.documents:
        return _documents_unavailable()
    from .documents import DocumentError

    name = Path(str(filename or "document.pdf")).name[:180] or "document.pdf"
    if Path(name).suffix.lower() != ".pdf":
        return _err(400, "only PDF files can be analysed page by page", "invalid_file")
    target = ctx.documents.dir / f"{time.time_ns()}-{re.sub(r'[^A-Za-z0-9._-]+', '_', name)}"
    temporary = target.with_suffix(target.suffix + ".part")
    received = 0
    try:
        with open(temporary, "wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                received += len(chunk)
                if received > MAX_DOCUMENT_BYTES:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                    limit = MAX_DOCUMENT_BYTES // (1024 * 1024)
                    return _err(
                        400,
                        f"the PDF is too large (max {limit} MB)",
                        "invalid_file",
                    )
                handle.write(chunk)
        if received == 0:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return _err(400, "the uploaded PDF is empty", "invalid_file")
        os.replace(temporary, target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return _fail(
            500,
            exc,
            "storing the document upload",
            "The PDF could not be stored. Check that the documents folder is writable and has free space.",
            "upload_failed",
        )

    try:
        from starlette.concurrency import run_in_threadpool

        job = await run_in_threadpool(ctx.documents.start, target, name)
    except DocumentError as exc:
        return _fail(
            400,
            exc,
            "starting the document job",
            "The PDF could not be analysed. Check that it is a valid, unencrypted PDF and that a vision model is installed.",
            "document_error",
        )
    except Exception as exc:
        return _fail(
            500,
            exc,
            "starting the document job",
            "The document job could not start. See the app log for details.",
            "document_error",
        )
    return {"ok": True, "job": job, "bytes": received}


# ------------------------------------------------------------------------- web search
@router.get("/web/status")
async def web_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.web_search:
        return {"available": False, "reason": "web search is not configured"}
    ok, reason = ctx.web_search.available()
    return {"available": bool(ok), "reason": reason}


@router.post("/web/search")
async def web_search(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.web_search:
        return _err(503, "web search is not available", "web_unavailable")
    try:
        body = await request.json()
    except Exception:
        return _err(400, "body must be JSON", "invalid_json")
    query = str((body or {}).get("query") or "").strip()
    if not query:
        return _err(400, "query is required", "empty_query")
    try:
        from starlette.concurrency import run_in_threadpool

        results = await run_in_threadpool(ctx.web_search.search, query)
    except Exception as exc:
        return _fail(
            400,
            exc,
            "running a web search",
            "The web search failed. Check your connection and the search backend settings.",
            "web_error",
        )
    return {"data": results, "query": query}

# ------------------------------------------------------------------------- requirements
@router.get("/requirements")
async def requirements_status(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.requirements:
        return _err(503, "the setup manager is not available", "setup_unavailable")
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(ctx.requirements.summary)


@router.post("/requirements/check")
async def requirements_check(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.requirements:
        return _err(503, "the setup manager is not available", "setup_unavailable")
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(ctx.requirements.check)


@router.post("/requirements/install-all")
async def requirements_install_all(request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.requirements:
        return _err(503, "the setup manager is not available", "setup_unavailable")
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(ctx.requirements.install_missing)


@router.post("/requirements/{component_id}/install")
async def requirements_install(component_id: str, request: Request):
    ctx: AppContext = request.app.state.ctx
    if not ctx.requirements:
        return _err(503, "the setup manager is not available", "setup_unavailable")
    from starlette.concurrency import run_in_threadpool

    try:
        return await run_in_threadpool(ctx.requirements.install, component_id)
    except ValueError as exc:
        return _fail(
            400,
            exc,
            "installing a setup component",
            "That component cannot be installed. Check the component ID and the setup log.",
            "install_error",
        )


@router.get("/requirements/{component_id}/log")
async def requirements_log(component_id: str, request: Request, lines: int = 160):
    ctx: AppContext = request.app.state.ctx
    if not ctx.requirements:
        return _err(503, "the setup manager is not available", "setup_unavailable")
    if component_id not in {d["id"] for d in ctx.requirements._definitions()}:
        return _err(404, "unknown component", "not_found")
    return {
        "component": component_id,
        "path": str(ctx.requirements.component_log(component_id)),
        "lines": ctx.requirements.log_tail(component_id, lines),
    }


@router.get("/logs/{name}")
async def component_log(name: str, request: Request, lines: int = 200):
    ctx: AppContext = request.app.state.ctx
    allowed = {"aiolocal", "ovms", "opencode", "setup", "gimp", "rea"}
    if name.startswith("setup-") and re.fullmatch(r"setup-[a-z0-9_]{1,40}", name):
        pass
    elif name not in allowed:
        return _err(404, "unknown log", "not_found")
    path = ctx.paths.log_dir / f"{name}.log"
    if not path.is_file():
        return {"name": name, "path": str(path), "lines": []}
    try:
        data = path.read_bytes()[-512_000:]
    except OSError as exc:
        return _fail(
            500,
            exc,
            "reading an app log",
            "The log file could not be read. Check that the logs folder is readable.",
            "log_error",
        )
    text = data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
    rows = [line for line in text.split("\n") if line.strip()]
    return {
        "name": name,
        "path": str(path),
        "lines": rows[-max(1, min(int(lines), 2000)):],
    }


# ------------------------------------------------------------------------- system
@router.get("/system/info")
async def system_info(request: Request):
    ctx: AppContext = request.app.state.ctx
    usage = shutil.disk_usage(ctx.paths.data_dir)
    return {
        "app_version": __version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
        "disk_free_gb": round(usage.free / (1024 ** 3), 1),
        "data_dir": str(ctx.paths.data_dir),
        "catalog_size": len(ctx.models.entries()) if ctx.models else 0,
        "time": int(time.time()),
    }


@router.get("/events")
async def events(request: Request):
    ctx: AppContext = request.app.state.ctx
    subscriber = ctx.bus.subscribe()

    async def generator():
        try:
            async for frame in ctx.bus.stream(subscriber):
                yield frame
                if await request.is_disconnected():
                    break
        finally:
            ctx.bus.unsubscribe(subscriber)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )

