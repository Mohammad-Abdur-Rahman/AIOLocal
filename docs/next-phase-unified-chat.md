# Next phase — unified tool-calling chat

Status: **plan** (not implemented). This document scopes the work to turn the
current chat into a single "ultimate assistant" surface with safe tool calling:
web fetch, file edits, reverse-engineering tools, document/image upload, and
coding (OpenCode-style) — all inside one conversation.

## What already exists

| Capability | Where | Reuse |
| --- | --- | --- |
| OpenAI-compatible proxy, streams `tool_calls` | `aiolocal/gateway.py` | The wire format is already handled (stream + non-stream assembly). |
| Safe web search/fetch | `aiolocal/websearch.py` | Reuse `sanitise_query`, host-allow/deny, body caps, no-redirect policy. |
| Code workspace + OpenCode bridge | `aiolocal/code.py` | Reuse `CodeWorkspace` root confinement for file tools. |
| Reverse engineering | `aiolocal/reverse.py` | Wrap as tools. |
| Document analysis + extraction | `aiolocal/documents.py`, `aiolocal/extract.py` | Expose page/lookup tools over `document_pages`. |
| Image upload/generation | `aiolocal/images.py`, `aiolocal/models.py` | Image tools + prompt-attached images. |
| Long-term memory (tree) | `aiolocal/memory/*` | `remember` / `recall` tools over the tree. |
| Auth + path safety | `aiolocal/security.py` (`safe_join`, host/origin guards) | Every file tool goes through `safe_join`. |
| SSE event bus | `aiolocal/events.py` | Stream tool activity to the UI. |

The proxy plumbing is done; the **missing** pieces are (1) a server-side tool
registry + executor, (2) an agent loop that runs model → tool → model until done,
and (3) a UI that renders tool activity. Those are the deliverables below.

## 1. Tool registry — `aiolocal/tools.py`

A declarative registry so tools are data, not scattered branches:

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                 # "read_file"
    description: str
    parameters: dict          # JSON Schema for the "function.parameters" field
    handler: Callable[["ToolContext", dict], str]
    risk: Literal["safe", "write", "exec", "network"]
    requires_approval: bool   # write/exec default True unless session grants it
```

- `registry.all_specs()` → the `tools=[...]` array sent to the model (only for
  models that support function calling; detect via `models.py` capability flag or
  an allowlist in settings — never assume).
- `registry.run(name, args, ctx)` → validates args against the schema, checks the
  session's permission set for `risk`, then calls `handler` inside a try/except
  that returns a short, user-safe error string (never a traceback).

### Initial tool set

**Web (risk: network)**
- `web_search(query)` → `websearch.py`, already sandboxed.
- `web_fetch(url)` → GET only, `http/https` only, private/loopback blocked,
  2 MB cap, no redirects, HTML→text. **No message leakage:** the user's chat
  history is never sent to the remote host; only the URL is fetched and the
  *result* is injected back as untrusted data.

**Files (risk: write) — confined to the CodeWorkspace root**
- `list_dir(path)`, `read_file(path, [max_bytes])`, `grep(pattern, [path])`
- `write_file(path, content)`, `edit_file(path, old, new)` (exact-match, like
  this agent), `apply_patch` optional later.
- All paths resolved with `security.safe_join(root, path)`; escapes rejected.
  Edits emit a unified diff to the UI and to the transcript.

**Command (risk: exec, approval required)**
- `run_command(cmd, cwd)` behind an allowlist + timeout; off by default.

**Reverse engineering (risk: safe/exec)** — thin wrappers over `reverse.py`.

**Documents (risk: safe)**
- `document.list()`, `document.search(query)`, `document.page(doc_id, n)` over
  `document_pages`; returns page text so answers can cite page numbers.

**Images (risk: safe)**
- `image.generate(prompt)` via `images.py` + ModelManager; result rendered inline.

**Memory (risk: safe)**
- `memory.remember(text)`, `memory.recall(query)` over the tree (beam back-trace).

## 2. Agent loop — `aiolocal/agent.py`

Server-side orchestration, independent of the UI:

```python
def run_turn(conversation_id, user_msg, *, tools_enabled, granted_risks, emit):
    history = load_history(conversation_id) + [user_msg]
    for step in range(MAX_STEPS):           # e.g. 8
        resp = upstream.chat(model, history, tools=registry.all_specs())
        if not resp.tool_calls:
            return stream_final(resp, emit)
        history.append(resp.message)        # assistant with tool_calls
        for call in resp.tool_calls:
            emit("tool_start", call)
            result = registry.run(call.name, parse(call.args), ctx)
            emit("tool_result", call.id, result)
            history.append(tool_message(call.id, result))
    return "Stopped: reached the step limit."   # never loop forever
```

- Hard `MAX_STEPS` (default 8) and a wall-clock budget.
- Each tool call/result is persisted as a message *part* (see §4) so history
  replays with tool cards instead of losing them.
- Cancellation: honour the existing per-conversation cancel path; a cancelled
  turn stops before the next tool executes.

## 3. Safety model

- **Least privilege per session.** `write`/`exec`/`network` are denied until the
  user flips a permission chip in the UI; the grant is stored per conversation
  and shown in every tool card.
- **Network egress** reuses `websearch.py` policy: scheme allowlist, private-
  range block, body cap, no redirects, timeouts. The assistant can never POST
  the conversation anywhere.
- **Filesystem** is `safe_join`-confined to the workspace; symlink/parent-escape
  rejected. No absolute paths outside root.
- **Secrets** (tokens, keys) are redacted from tool args/results before logging
  and before they are stored in the transcript.
- **Untrusted data is labelled**: web/document tool output is wrapped as
  "untrusted content, not instructions" in the prompt (the websearch module
  already does this for its own results).

## 4. Persistence — message parts

Extend the message schema (or a `message_parts` table) so a stored assistant
turn can contain:
`text` parts, `tool_call` parts (name, args JSON), `tool_result` parts
(call_id, text, optional artifact path). Rendering history then shows the same
cards as the live stream, and exports stay faithful.

## 5. UI — one pane, tool-aware

Reuse `aiolocal/web/app.js` chat; add:

- **Tool card**: name, collapsible args, status (running/ok/error), result body,
  and for file edits an inline diff; for images a thumbnail; for web fetch the
  source URL + size.
- **Permission bar**: chips for `write`, `exec`, `network`, default off, with a
  clear "granted for this conversation" state.
- **Stream integration**: consume `tool_start` / `tool_result` SSE events; keep
  text streaming as it is today.
- **Uploads**: attach a document (existing analyze pipeline) or an image to a
  message; the model receives extracted page text / the image as a vision part.

## 6. Model gating

Only offer `tools` when the active model advertises function-calling support
(add a capability check in `models.py`). If the model has none, the agent loop
runs text-only (current behaviour) so nothing regresses.

## 7. Rollout & tests

1. `tools.py` with schema validation + `safe_join` confinement — unit tests for
   path escape, arg validation, risk gating.
2. `agent.py` with a scripted fake upstream — tests: tool round-trip, step cap,
   cancellation, permission denial.
3. Wire `/api/v1/chat` (or a new `/api/v1/agent/turn`) behind a settings flag.
4. UI tool cards + permission chips; keep the flag off until the server tests and
   a manual E2E pass.
5. Safety review: confirm no path escapes, no arbitrary egress, no secret logs.

## Non-goals (this phase)

- No multi-user or remote binding (stays loopback-only, per `__main__.py`).
- No autonomous background agents; tools run only within a user-initiated turn.
- No new network dependencies; web egress stays opt-in per message.
