"""Agent-facing MCP operations for Magpie.

The browser has a process-local "open workspace" for presentation. MCP callers
must never rely on or mutate that selection: every workspace-scoped tool below
routes directly to a stable workspace id.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from mcp.types import ToolAnnotations

from . import engine as engine_mod
from . import server


GUIDE = """\
Magpie is a persistent field for developing thoughts, not an authority that
settles truth automatically. Choose a workspace explicitly for every field
operation. Use contribute for raw speech or notes; it queues asynchronous
atomization, so poll get_field to observe resulting cards. Use
get_conversation_map for the compact digest of themes, recurring ideas, typed
follow-ups, and sparse connections. collide and organize change the field.
recall_workspace searches the private Raven memory graph through Magpie;
adopt_memory imports one suggestion as an open local card.
Only the human-facing browser may judge, dismiss, or create a verification
receipt; those operations are intentionally absent from MCP.
"""

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
LOCAL_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _require_store() -> server.Storage:
    store = server.STORE
    if store is None:
        raise RuntimeError("workspace storage is not initialized")
    return store


def _workspace(workspace_id: str) -> server.Workspace:
    workspace_id = str(workspace_id or "").strip()
    if not workspace_id:
        raise ValueError("workspace_id is required")
    return _require_store().load_workspace(workspace_id)


def _workspace_summary(workspace: server.Workspace) -> dict[str, Any]:
    return {
        "id": workspace.id,
        "name": workspace.name,
        "question": workspace.question,
        "context_version": workspace.context_version,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at,
        "open_in_browser": workspace.id == server.WORKSPACE_ID,
    }


def _target_engine(workspace_id: str) -> tuple[Any, bool]:
    _workspace(workspace_id)
    return server._engine_for_workspace(workspace_id)


def _save(
    workspace_id: str,
    engine: Any,
    *,
    event_kind: str,
    event_payload: dict[str, Any],
    increment_context: bool = True,
) -> server.Workspace:
    saved = server._persist_engine(
        workspace_id,
        engine,
        event_kind=event_kind,
        event_payload=event_payload,
        increment_context=increment_context,
    )
    if saved is None:
        raise RuntimeError("workspace persistence is unavailable")
    return saved


def _state_for(workspace_id: str) -> dict[str, Any]:
    with server.LOCK:
        engine, _is_current = _target_engine(workspace_id)
        state = dict(engine.state())
        workspace = _workspace(workspace_id)
        state["workspace"] = {
            **_workspace_summary(workspace),
            "bank_revision": _require_store().bank_revision(),
        }
        state["providers"] = server._provider_status()
        state["memory_shelf"] = server._memory_state(workspace_id)
        return state


_DIGEST_LIST_KEYS = (
    "themes",
    "recurring_ideas",
    "open_questions",
    "decisions",
    "constraints",
    "experiments",
    "tasks",
    "between_ideas",
)


def _conversation_map(workspace_id: str) -> dict[str, Any]:
    """Return the additive conversation digest without exposing full field state."""

    state = _state_for(workspace_id)
    raw_digest = state.get("digest")
    digest = dict(raw_digest) if isinstance(raw_digest, dict) else {}
    for key in _DIGEST_LIST_KEYS:
        if not isinstance(digest.get(key), list):
            digest[key] = []
    # Connections are deliberately a sparse secondary surface. Enforce that
    # contract at the agent boundary even while the internal digest evolves.
    digest["between_ideas"] = digest["between_ideas"][:3]
    return {"workspace_id": workspace_id, "digest": digest}


def _create_workspace(name: str, question: str = "") -> dict[str, Any]:
    store = _require_store()
    name = str(name or "").strip()
    question = str(question or "").strip()
    if not name:
        raise ValueError("name is required")
    engine = engine_mod.Engine()
    if question:
        engine.seed(question)
    with server.LOCK:
        workspace = store.create_workspace(
            name,
            question=question,
            snapshot=engine.state(),
        )
        workspace = _save(
            workspace.id,
            engine,
            event_kind="workspace.created",
            event_payload={"name": name, "question": question, "source": "mcp"},
        )
    return {"workspace": _workspace_summary(workspace)}


def _seed(workspace_id: str, question: str) -> dict[str, Any]:
    question = str(question or "").strip()
    if not question:
        raise ValueError("question is required")
    with server.LOCK:
        engine, _is_current = _target_engine(workspace_id)
        engine.seed(question)
        workspace = _save(
            workspace_id,
            engine,
            event_kind="question.changed",
            event_payload={"question": question, "source": "mcp"},
        )
    return {"workspace": _workspace_summary(workspace), "question": question}


def _contribute(workspace_id: str, text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("text is required")
    with server.LOCK:
        engine, _is_current = _target_engine(workspace_id)
        workspace = _save(
            workspace_id,
            engine,
            event_kind="thought.submitted",
            event_payload={"text": text, "source": "mcp"},
        )
    threading.Thread(
        target=server._run_atomize,
        args=(text, workspace_id),
        daemon=True,
        name=f"magpie-atomize-{workspace_id[-8:]}",
    ).start()
    return {
        "queued": True,
        "workspace_id": workspace_id,
        "context_version": workspace.context_version,
        "submitted_text": text,
    }


def _collide(workspace_id: str, a: str, b: str) -> dict[str, Any]:
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a or not b:
        raise ValueError("a and b are required")
    if a == b:
        raise ValueError("a and b must identify different cards")
    with server.LOCK:
        _target_engine(workspace_id)
        card = server._collide_and_fuse(a, b, workspace_id)
    return {"queued": True, "workspace_id": workspace_id, "card": card}


def _section_key(engine: Any, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError("section is required")
    for key, section in engine.sections.items():
        if key == value or str(section.get("name") or "") == value:
            return key
    return value


def _organize(
    workspace_id: str,
    action: str,
    *,
    card_id: str | None = None,
    section: str | None = None,
    name: str | None = None,
    key: str | None = None,
    color: str = "#c9b8a0",
) -> dict[str, Any]:
    action = str(action or "").strip().lower()
    allowed = {"keep", "release", "move", "create_section"}
    if action not in allowed:
        raise ValueError(f"action must be one of: {', '.join(sorted(allowed))}")
    with server.LOCK:
        engine, _is_current = _target_engine(workspace_id)
        payload: dict[str, Any]
        if action in {"keep", "release"}:
            card_id = str(card_id or "").strip()
            if not card_id:
                raise ValueError("card_id is required")
            card = engine._card(card_id)
            want_pinned = action == "keep"
            if card.pinned != want_pinned:
                card = engine.keep(card_id)
            payload = {"card": server._card_dict(card)}
            event_kind = "idea.pinned" if want_pinned else "idea.released"
            event_payload = {"card_id": card_id, "source": "mcp"}
        elif action == "move":
            card_id = str(card_id or "").strip()
            if not card_id:
                raise ValueError("card_id is required")
            resolved = _section_key(engine, str(section or ""))
            card = engine.move(card_id, resolved)
            payload = {"card": server._card_dict(card)}
            event_kind = "idea.moved"
            event_payload = {
                "card_id": card_id,
                "section": resolved,
                "source": "mcp",
            }
        else:
            name = str(name or "").strip()
            if not name:
                raise ValueError("name is required")
            section_key = str(key or name.lower().replace(" ", "-")).strip()
            color = str(color or "").strip()
            if not _HEX_COLOR.match(color):
                raise ValueError("color must be a hex literal like #c9b8a0")
            created = engine.add_section(section_key, name, color)
            payload = {"section": dict(created)}
            event_kind = "section.created"
            event_payload = {
                "key": section_key,
                "name": name,
                "source": "mcp",
            }
        if action != "create_section":
            server._sync_projection_presentation(workspace_id, card)
        workspace = _save(
            workspace_id,
            engine,
            event_kind=event_kind,
            event_payload=event_payload,
        )
    return {
        **payload,
        "workspace_id": workspace.id,
        "context_version": workspace.context_version,
    }


def _harvest(workspace_id: str) -> dict[str, Any]:
    with server.LOCK:
        engine, _is_current = _target_engine(workspace_id)
        brief = engine.harvest()
        workspace = _save(
            workspace_id,
            engine,
            event_kind="workspace.harvested",
            event_payload={
                "cards": len(brief.get("cards") or []),
                "source": "mcp",
            },
            increment_context=False,
        )
    server._ensure_runtime()
    path = Path(server.RUNTIME_DIR) / f"harvest-{workspace.id}-{int(time.time())}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(brief, fh, indent=2, default=str)
        fh.flush()
        server.os.fsync(fh.fileno())
    server.os.replace(tmp, path)
    return {"workspace_id": workspace.id, "brief": brief, "path": str(path)}


def _recall(workspace_id: str, query: str = "", limit: int = 10) -> dict[str, Any]:
    _workspace(workspace_id)
    return server._recall_workspace(workspace_id, query, limit=limit)


def _adopt(
    workspace_id: str,
    memory_id: str,
    section: str | None = None,
) -> dict[str, Any]:
    _workspace(workspace_id)
    return server._adopt_raven_memory(
        workspace_id, memory_id, section=section
    )


def register_tools(mcp: Any) -> None:
    """Register the deliberately bounded Magpie MCP surface."""

    @mcp.tool(annotations=READ_ONLY)
    def guide() -> dict[str, Any]:
        """Explain Magpie's model, safe tool use, and provenance boundary."""
        return {
            "guide": GUIDE,
            "human_only_operations": ["judge", "kill", "verify", "resolve"],
        }

    @mcp.tool(annotations=READ_ONLY)
    def list_workspaces() -> dict[str, Any]:
        """List stable workspace ids without changing the browser selection."""
        store = _require_store()
        with server.LOCK:
            return {
                "workspaces": [
                    _workspace_summary(workspace)
                    for workspace in store.list_workspaces()
                ]
            }

    @mcp.tool(annotations=LOCAL_WRITE)
    def create_workspace(name: str, question: str = "") -> dict[str, Any]:
        """Create an isolated workspace without opening it in the browser."""
        return _create_workspace(name, question)

    @mcp.tool(annotations=READ_ONLY)
    def get_field(workspace_id: str) -> dict[str, Any]:
        """Read one workspace's field without changing any active workspace."""
        return _state_for(workspace_id)

    @mcp.tool(annotations=READ_ONLY)
    def get_conversation_map(workspace_id: str) -> dict[str, Any]:
        """Read its compact digest, themes, recurring ideas, and typed follow-ups."""
        return _conversation_map(workspace_id)

    @mcp.tool(annotations=READ_ONLY)
    def recall_workspace(
        workspace_id: str, query: str = "", limit: int = 10
    ) -> dict[str, Any]:
        """Search Raven through Magpie for workspace-relevant memories."""
        return _recall(workspace_id, query, limit)

    @mcp.tool(annotations=LOCAL_WRITE)
    def adopt_memory(
        workspace_id: str,
        memory_id: str,
        section: str | None = None,
    ) -> dict[str, Any]:
        """Import one Raven suggestion as an open card in this workspace."""
        return _adopt(workspace_id, memory_id, section)

    @mcp.tool(annotations=LOCAL_WRITE)
    def seed_field(workspace_id: str, question: str) -> dict[str, Any]:
        """Set the central question for one explicitly named workspace."""
        return _seed(workspace_id, question)

    @mcp.tool(annotations=LOCAL_WRITE)
    def contribute(workspace_id: str, text: str) -> dict[str, Any]:
        """Queue raw speech or notes for asynchronous atomization."""
        return _contribute(workspace_id, text)

    @mcp.tool(annotations=LOCAL_WRITE)
    def collide(workspace_id: str, a: str, b: str) -> dict[str, Any]:
        """Queue a synthesis attempt between two cards in one workspace."""
        return _collide(workspace_id, a, b)

    @mcp.tool(annotations=LOCAL_WRITE)
    def organize(
        workspace_id: str,
        action: str,
        card_id: str | None = None,
        section: str | None = None,
        name: str | None = None,
        key: str | None = None,
        color: str = "#c9b8a0",
    ) -> dict[str, Any]:
        """Keep, release, move a card, or create a section; never dismiss it."""
        return _organize(
            workspace_id,
            action,
            card_id=card_id,
            section=section,
            name=name,
            key=key,
            color=color,
        )

    @mcp.tool(annotations=LOCAL_WRITE)
    def harvest(workspace_id: str) -> dict[str, Any]:
        """Write and return the current structured brief for one workspace."""
        return _harvest(workspace_id)
