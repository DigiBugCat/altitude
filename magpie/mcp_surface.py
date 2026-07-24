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
Altitude is a persistent LADDER of positions, not a pile of cards, and not an
authority that settles truth automatically. Choose a workspace explicitly for
every field operation.

The model has three floors. Receipts are evidence events. Claims (altitude 0)
are atomic, receipt-checkable propositions. Frames (altitude 1+) are
abstractions standing on claims; a frame is NEVER directly supported or
refuted — its support is computed from the floor below on every read, so no
level can ever drift from the level under it.

Reading: get_field(workspace_id, altitude) returns one floor, defaulting to the
top; every position carries its supports, its computed support summary, and
when it was last grounded. descend(position_id) swaps to a frame's floor,
folded instances included. get_conversation_map gives the compact digest.
harvest(altitude) writes the decision-ready brief.

Going up: propose_click(a, b) asks whether two positions are two instances of
ONE frame. Most pairs are not, and "no" is the expected answer. A click that
passes the gates lands in the emergence inbox (pending_clicks) — never in the
field. Only a human confirms it, and confirming means "organize these
together", not "this is true": the new frame is created OPEN with no receipt
and both instances stay alive one floor down, folded rather than archived.
unfold(frame_id) reverses that at any time.

Going down: derive(frame_id) proposes the atomic claims that would make a frame
true. They arrive as visibly ungrounded slots — structure awaiting evidence.
Derivation asserts nothing.

reconsider_pair(a, b) is the deliberate retry door for a settled non-click.
recall_workspace searches private Raven memory; adopt_memory imports one
suggestion as a quarantined sub-ladder awaiting a human.

Only the human-facing browser may judge, dismiss, confirm a click, or create a
verification receipt. Those operations are intentionally absent from MCP: this
surface can propose structure, never settle it.
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
    # §3.2: `between_ideas` becomes `frames`. The truncation to 3 survives — it
    # was already the right instinct; only the thing being counted changed,
    # from sparse machine connections to confirmed levels of the ladder.
    "frames",
)


def _conversation_map(workspace_id: str) -> dict[str, Any]:
    """Return the additive conversation digest without exposing full field state."""

    state = _state_for(workspace_id)
    raw_digest = state.get("digest")
    digest = dict(raw_digest) if isinstance(raw_digest, dict) else {}
    for key in _DIGEST_LIST_KEYS:
        if not isinstance(digest.get(key), list):
            digest[key] = []
    digest.pop("between_ideas", None)
    digest["frames"] = digest["frames"][:3]
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


def _get_field(workspace_id: str, altitude: int | None = None) -> dict[str, Any]:
    """§3.2 — the field at one altitude, every position carrying its structure."""
    _workspace(workspace_id)
    field = server._api_field(
        {"workspace_id": workspace_id, "altitude": altitude}
    )
    state = _state_for(workspace_id)
    # The digest and workspace envelope stay attached: agents used `get_field`
    # as their whole read surface before altitude existed, and removing that
    # would make the tool strictly less useful than the thing it replaces.
    return {
        **field,
        "question": state.get("question"),
        "sections": state.get("sections"),
        "digest": state.get("digest"),
        "workspace": state.get("workspace"),
        "memory_shelf": state.get("memory_shelf"),
    }


def _descend(workspace_id: str, position_id: str) -> dict[str, Any]:
    _workspace(workspace_id)
    position_id = str(position_id or "").strip()
    if not position_id:
        raise ValueError("position_id is required")
    return server._api_descend(
        {"workspace_id": workspace_id, "position_id": position_id}
    )


def _propose_click(workspace_id: str, a: str, b: str) -> dict[str, Any]:
    """§2.1/§2.4 — request a recognition. It can only reach the inbox.

    This replaces the ``collide`` tool. An agent can ask whether two positions
    are one idea; it cannot put anything in the field by asking, and it cannot
    confirm the answer — that verdict is human-only and lives at
    ``resolve_click``, which is deliberately absent from this surface.
    """
    _workspace(workspace_id)
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a or not b:
        raise ValueError("a and b are required")
    if a == b:
        raise ValueError("a and b must identify different positions")
    return server._api_propose_click({"workspace_id": workspace_id, "a": a, "b": b})


def _pending_clicks(
    workspace_id: str, include_rejected: bool = False
) -> dict[str, Any]:
    _workspace(workspace_id)
    return server._api_pending_clicks(
        {"workspace_id": workspace_id, "include_rejected": bool(include_rejected)}
    )


def _reconsider_pair(workspace_id: str, a: str, b: str) -> dict[str, Any]:
    _workspace(workspace_id)
    return server._api_reconsider_pair(
        {"workspace_id": workspace_id, "a": str(a or ""), "b": str(b or "")}
    )


def _unfold(workspace_id: str, frame_id: str) -> dict[str, Any]:
    _workspace(workspace_id)
    frame_id = str(frame_id or "").strip()
    if not frame_id:
        raise ValueError("frame_id is required")
    return server._api_unfold(
        {"workspace_id": workspace_id, "frame_id": frame_id}
    )


def _derive(workspace_id: str, frame_id: str) -> dict[str, Any]:
    """§1.4 — fill structure beneath a frame. Asserts nothing.

    Every claim this creates arrives as an ungrounded slot: `open`, never
    grounded, scanner-ineligible and bank-ineligible until a receipt lands or a
    human pins it. An agent can state what evidence *would* settle a frame; it
    still cannot supply that evidence through MCP.
    """
    _workspace(workspace_id)
    frame_id = str(frame_id or "").strip()
    if not frame_id:
        raise ValueError("frame_id is required")
    return server._run_derive(frame_id, workspace_id)


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


def _harvest(
    workspace_id: str, altitude: int | None = None, max_items: int = 12
) -> dict[str, Any]:
    """§3.3 — the decision-ready brief at an altitude, hard-capped per section."""
    with server.LOCK:
        engine, _is_current = _target_engine(workspace_id)
        floor = None if altitude is None else max(0, int(altitude))
        brief = engine.harvest(altitude=floor, max_items=max(1, int(max_items)))
        workspace = _save(
            workspace_id,
            engine,
            event_kind="workspace.harvested",
            event_payload={
                "altitude": brief.get("altitude"),
                "spine": len(brief.get("spine") or []),
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
            "human_only_operations": [
                "judge", "kill", "verify", "resolve", "resolve_click",
            ],
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
    def get_field(
        workspace_id: str, altitude: int | None = None
    ) -> dict[str, Any]:
        """Read one floor of the ladder; omit altitude for the top."""
        return _get_field(workspace_id, altitude)

    @mcp.tool(annotations=READ_ONLY)
    def descend(workspace_id: str, position_id: str) -> dict[str, Any]:
        """Read the floor beneath one frame, folded instances included."""
        return _descend(workspace_id, position_id)

    @mcp.tool(annotations=READ_ONLY)
    def pending_clicks(
        workspace_id: str, include_rejected: bool = False
    ) -> dict[str, Any]:
        """List open emergence-inbox candidates; optionally near-misses too."""
        return _pending_clicks(workspace_id, include_rejected)

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
    def propose_click(workspace_id: str, a: str, b: str) -> dict[str, Any]:
        """Ask whether two positions are instances of one frame; inbox only."""
        return _propose_click(workspace_id, a, b)

    @mcp.tool(annotations=LOCAL_WRITE)
    def reconsider_pair(workspace_id: str, a: str, b: str) -> dict[str, Any]:
        """Deliberately reopen a settled non-click pair, visibly versioned."""
        return _reconsider_pair(workspace_id, a, b)

    @mcp.tool(annotations=LOCAL_WRITE)
    def derive(workspace_id: str, frame_id: str) -> dict[str, Any]:
        """Propose the atomic claims that would ground a frame, as empty slots."""
        return _derive(workspace_id, frame_id)

    @mcp.tool(annotations=LOCAL_WRITE)
    def unfold(workspace_id: str, frame_id: str) -> dict[str, Any]:
        """Release a frame's instances and vacate the frame position."""
        return _unfold(workspace_id, frame_id)

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
    def harvest(
        workspace_id: str,
        altitude: int | None = None,
        max_items: int = 12,
    ) -> dict[str, Any]:
        """Write and return the decision-ready brief at one altitude."""
        return _harvest(workspace_id, altitude, max_items)
