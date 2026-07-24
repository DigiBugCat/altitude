"""magpie.server — stdlib HTTP front end for the magpie field.

Serves the static field UI from ``birds/magpie/app/`` and a small JSON API over
the deterministic :mod:`magpie.engine`.  Every engine access happens under a
single process-wide lock; every mutation is persisted immediately.

Run:  python3 -m magpie.server [--port 7351]
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import struct
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from . import engine as engine_mod
from . import providers as providers_mod
from . import workers
from .raven_client import RavenClient
from .storage import Storage, Workspace

# ``python -m magpie.server`` executes this file as ``__main__``. Runtime
# adapters import ``magpie.server`` for the shared engine globals; alias the
# module early so that does not create a second, uninitialized module instance.
if __name__ == "__main__":
    sys.modules["magpie.server"] = sys.modules[__name__]

# A container publishes its port from outside the namespace, so a loopback
# bind would make the published port dead. Default stays loopback-only.
HOST = os.environ.get("MAGPIE_HOST") or "127.0.0.1"
PORT = int(os.environ.get("MAGPIE_PORT") or 7351)

BIRD_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BIRD_DIR / "app"


def _runtime_dir() -> Path:
    """Resolve persistent state without assuming a host or checkout layout."""
    configured = os.environ.get("MAGPIE_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return root / "magpie"


RUNTIME_DIR = _runtime_dir()
STATE_PATH = RUNTIME_DIR / "state.json"
DB_PATH = RUNTIME_DIR / "magpie.sqlite3"

# SPEC §2.2 — the metabolism is a structural inspector, not a generator.
# 30.0s from 3.0s: the old cadence drained the pair space exhaustively, and
# "10 contributions → 45 pairs → 45 cards was the loop working as written".
METABOLISM_PERIOD = 30.0
# No scan within 60s of the last human contribution: recognition waits for the
# field to settle rather than racing the person typing into it.
QUIESCENCE_WINDOW = 60.0
# Fleet-wide LLM call budget across workspaces, per day.
FLEET_CLICK_BUDGET_PER_DAY = 50
# §2.2 anti-starvation: N new claims with zero scans forces one wave.
STARVATION_CLAIMS = 10
# §3.3/§7.3(c): 14 days without grounding flags a position in the brief.
STALENESS_WINDOW = 14 * 24 * 3600.0

EMBEDDING_MODEL = os.environ.get("MAGPIE_EMBEDDING_MODEL") or "magpie-default"
EMBEDDING_VERSION = os.environ.get("MAGPIE_EMBEDDING_VERSION") or "v1"

AUTO_CONNECTIONS = (
    os.environ.get("MAGPIE_AUTO_CONNECTIONS", "").strip().lower()
    in {"1", "true", "yes", "on"}
)

# --------------------------------------------------------------------------
# shared mutable state
# --------------------------------------------------------------------------

LOCK = threading.Lock()
ENGINE: Any = None
STORE: Storage | None = None
WORKSPACE_ID: str | None = None
_STOP = threading.Event()

# §2.2 metabolism accounting. Touched only from the single metabolism thread
# and from tests, so it needs no lock of its own.
_FLEET_BUDGET_DAY: int = -1
_FLEET_BUDGET_SPENT: int = 0
_STARVATION_COUNTS: dict[str, int] = {}


class ServiceUnavailable(RuntimeError):
    """A configured integration is temporarily unavailable."""


@dataclass(frozen=True)
class VerificationRequest:
    """Bounded input for a future tool-using verification runtime."""

    workspace_id: str | None
    card: dict
    question: str


class VerificationHook(Protocol):
    """Submission boundary for a future Codex App Server adapter.

    Submission starts work and returns an opaque job id. It deliberately does
    not return a verdict: observations must come back through an explicit,
    trusted receipt-ingestion path before ``Engine.resolve`` is called.
    """

    def submit(self, request: VerificationRequest) -> str: ...


VERIFICATION_HOOK: VerificationHook | None = None
RAVEN = RavenClient.from_env()
RAVEN_LAST_ERROR: str | None = None
RAVEN_OUTBOX_PERIOD = 1.0
RAVEN_MAX_ATTEMPTS = 8


def _ensure_runtime() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _persist_engine(
    workspace_id: str | None,
    engine: Any,
    *,
    event_kind: str | None = None,
    event_payload: dict | None = None,
    increment_context: bool = False,
) -> Workspace | None:
    """Commit one workspace snapshot and optional retrieval-trigger event.

    ``STORE is None`` is retained only for isolated unit tests that exercise
    the engine/server seam without booting persistence.
    """
    if STORE is None or workspace_id is None:
        _ensure_runtime()
        engine_mod.save(engine, str(STATE_PATH))
        return None
    snapshot = engine.state()
    with STORE.transaction():
        workspace = STORE.save_workspace(
            workspace_id,
            snapshot,
            question=engine.question,
            increment_context=increment_context,
        )
        # The snapshot stays authoritative; these are indices written in the
        # same transaction so the index can never commit without the snapshot
        # it indexes (§5).
        STORE.sync_positions(workspace_id, snapshot)
        STORE.sync_click_ledger(workspace_id, snapshot)
        if event_kind:
            STORE.append_event(
                event_kind,
                event_payload or {},
                workspace_id=workspace_id,
                context_version=workspace.context_version,
            )
    return workspace


def _persist(
    *,
    event_kind: str | None = None,
    event_payload: dict | None = None,
    increment_context: bool = False,
) -> Workspace | None:
    """Commit the currently open workspace. Caller MUST hold ``LOCK``."""
    return _persist_engine(
        WORKSPACE_ID,
        ENGINE,
        event_kind=event_kind,
        event_payload=event_payload,
        increment_context=increment_context,
    )


def _engine_for_workspace(workspace_id: str | None) -> tuple[Any, bool]:
    """Return ``(engine, is_current)`` under ``LOCK``."""
    if workspace_id is None or workspace_id == WORKSPACE_ID or STORE is None:
        return ENGINE, True
    workspace = STORE.load_workspace(workspace_id)
    return engine_mod.Engine.from_state(workspace.snapshot), False


def _required_workspace_id(body: dict) -> str:
    """Resolve an explicitly-routed workspace at every mutating HTTP boundary."""
    workspace_id = str(body.get("workspace_id") or "").strip()
    if not workspace_id:
        raise ValueError("workspace_id required")
    if STORE is not None:
        STORE.load_workspace(workspace_id)
    elif workspace_id != WORKSPACE_ID:
        raise KeyError(workspace_id)
    return workspace_id


def _export_class_for(
    position: Any,
    data: dict[str, Any],
    *,
    source: str,
) -> str | None:
    """SPEC §4 — which of the three export classes this card qualifies for.

    Returning ``None`` means *never banked*.  This is the contamination fix:
    the old ``_bank_card`` enqueued a Raven write for every card, so machine
    fusions were written to shared memory and recalled back later. Only
    human-rooted claims and human-confirmed frames leave the workspace.

    ``position`` is the live engine :class:`~magpie.engine.Position`, not the
    storage index row: banking runs before the snapshot is persisted, so the
    index is a tick stale here and an eligibility gate must never read stale
    structure.  The index exists to make the decision *inspectable* after the
    fact (``outbox.export_class``), not to make it.
    """
    origin = getattr(position, "origin", None)
    floor_kind = getattr(position, "floor_kind", "claim")
    external = bool(getattr(position, "external", False))
    confirmed_by = getattr(position, "confirmed_by", None)

    # Recall-quarantined material is not ours to re-bank: writing it back would
    # close the loop that made the field recall its own echo (§4).
    if external:
        return None
    # A derived-ungrounded claim is structure awaiting evidence, not an
    # assertion (§1.4); an unpromoted machine candidate is not material at all.
    if origin in ("derivation", "recall"):
        return None
    if floor_kind == "frame":
        # A fold stays workspace-local unless a human confirmed it (§1.6, §4).
        return "human_curated_frame" if str(confirmed_by or "").strip() else None
    if str(data.get("state")) in ("supported", "refuted") and str(
        data.get("receipt") or ""
    ).strip():
        return "settled_claim"
    if source == "atomize" and origin in (None, "human"):
        return "human_root"
    return None


def _bank_card(
    workspace_id: str | None,
    card: Any,
    *,
    source: str,
    engine: Any = None,
) -> Any | None:
    """Bank a card locally and queue its canonical Raven write.

    The local idea-bank row is written for every card — it is workspace-local
    provenance. The *Raven* write is gated on an export class (§4): machine
    candidates, derived-ungrounded claims, and unpromoted folds are banked
    locally and never queued outward.

    Raven delivery is intentionally asynchronous: the workspace stays usable
    if the private memory service is disabled or temporarily unavailable.
    """
    if STORE is None or workspace_id is None:
        return None
    data = _card_dict(card)
    text = str(data.get("text") or "").strip()
    if not text:
        return None
    position = None
    if engine is not None:
        try:
            position = engine.position(str(data.get("id") or ""))
        except KeyError:
            position = None
    idea, _created = STORE.upsert_idea(
        text,
        kind=str(data.get("artifact_type") or data.get("kind") or "claim"),
        metadata={
            "provenance": data.get("foot") or "",
            "artifact_type": data.get("artifact_type") or "claim",
        },
    )
    STORE.link_idea(
        workspace_id,
        idea.id,
        local_ref=str(data.get("id") or "") or None,
        source=source,
        metadata={"state": data.get("state"), "archived": data.get("archived", False)},
    )
    local_ref = str(data.get("id") or "").strip()
    export_class = _export_class_for(position, data, source=source)
    if export_class is None:
        return idea
    STORE.enqueue_raven_remember(
        text,
        export_class=export_class,
        workspace_id=workspace_id,
        source="human" if export_class == "human_root" else None,
        tags=["magpie", f"workspace:{workspace_id}"],
        episode_id=workspace_id,
        dedupe_key=f"magpie-card:{workspace_id}:{local_ref or idea.id}",
    )
    return idea


def _bank_occurrence(
    workspace_id: str | None,
    card: Any,
    occurrence_text: str,
    *,
    relation: str,
    source: str = "atomize",
) -> Any | None:
    """Persist repeat/refinement provenance without creating Raven noise."""
    if STORE is None or workspace_id is None:
        return None
    data = _card_dict(card)
    clean = str(occurrence_text or "").strip()
    if not clean:
        return None
    idea, _created = STORE.upsert_idea(
        clean,
        kind=str(data.get("artifact_type") or data.get("kind") or "observation"),
        metadata={"canonical_card_id": data.get("id"), "relation": relation},
    )
    STORE.link_idea(
        workspace_id,
        idea.id,
        local_ref=str(data.get("id") or "") or None,
        source=f"{source}:{relation}",
        metadata={
            "canonical_card_id": data.get("id"),
            "relation": relation,
            "state": data.get("state"),
        },
    )
    return idea


def _raven_projection_dict(projection: Any) -> dict[str, Any]:
    return {
        "id": projection.id,
        "memory_id": projection.raven_memory_id,
        "local_ref": projection.local_ref,
        "section": projection.section,
        "mass": projection.mass,
        "pinned": projection.pinned,
        "hidden": projection.hidden,
        "note": projection.local_note,
        "status": projection.local_status,
        "metadata": projection.metadata,
        "created_at": projection.created_at,
        "updated_at": projection.updated_at,
    }


def _raven_exposure_dict(exposure: Any) -> dict[str, Any]:
    memory = dict(exposure.metadata.get("memory") or {})
    return {
        "memory_id": exposure.raven_memory_id,
        "status": exposure.status,
        "reason": exposure.reason,
        "context_version": exposure.context_version,
        "first_shown_at": exposure.first_shown_at,
        "last_shown_at": exposure.last_shown_at,
        "memory": memory,
    }


def _memory_state(workspace_id: str | None) -> dict[str, Any]:
    """Return Magpie-shaped memory state without exposing Raven credentials."""
    if STORE is None or workspace_id is None:
        return {
            "enabled": RAVEN.enabled,
            "status": "disabled" if not RAVEN.enabled else "unavailable",
            "error": RAVEN_LAST_ERROR,
            "suggestions": [],
            "projections": [],
        }
    exposures = STORE.list_raven_exposures(workspace_id)
    suggestions = [
        _raven_exposure_dict(exposure)
        for exposure in exposures
        if exposure.status == "suggested"
    ]
    return {
        "enabled": RAVEN.enabled,
        "status": (
            "disabled"
            if not RAVEN.enabled
            else ("unavailable" if RAVEN_LAST_ERROR else "ready")
        ),
        "error": RAVEN_LAST_ERROR,
        "suggestions": suggestions,
        "projections": [
            _raven_projection_dict(projection)
            for projection in STORE.list_raven_projections(
                workspace_id, include_hidden=False
            )
        ],
    }


def _sync_projection_presentation(
    workspace_id: str | None, card: Any
) -> None:
    """Mirror local card layout into its Raven projection, if one exists."""
    if STORE is None or workspace_id is None or card is None:
        return
    data = _card_dict(card)
    local_ref = str(data.get("id") or "").strip()
    if not local_ref:
        return
    projection = next(
        (
            item
            for item in STORE.list_raven_projections(workspace_id)
            if item.local_ref == local_ref
        ),
        None,
    )
    if projection is None:
        return
    STORE.upsert_raven_projection(
        workspace_id,
        projection.raven_memory_id,
        local_ref=local_ref,
        section=str(data.get("section") or projection.section),
        mass=float(data.get("mass") or 0),
        pinned=bool(data.get("pinned")),
        hidden=bool(data.get("archived")),
        local_note=projection.local_note,
        local_status=projection.local_status,
        metadata=projection.metadata,
    )


def _card_dict(card: Any) -> dict:
    """Normalize a Card (dataclass or dict) into a plain dict."""
    if isinstance(card, dict):
        return dict(card)
    if hasattr(card, "__dict__"):
        return {k: v for k, v in vars(card).items() if not k.startswith("_")}
    fields = (
        "id", "kind", "text", "section", "mass", "state",
        "receipt", "foot", "pinned", "born", "parents",
    )
    return {f: getattr(card, f, None) for f in fields}


def _section_names(engine: Any | None = None) -> list[str]:
    """Section display names from one engine state. Caller holds ``LOCK``."""
    snap = (engine or ENGINE).state()
    sections = snap.get("sections") or {}
    names: list[str] = []
    if isinstance(sections, dict):
        for key, val in sections.items():
            if isinstance(val, dict):
                names.append(str(val.get("name") or key))
            else:
                names.append(str(val))
    elif isinstance(sections, list):
        for val in sections:
            if isinstance(val, dict):
                names.append(str(val.get("name") or val.get("key")))
            else:
                names.append(str(val))
    return [n for n in names if n]


def _section_key_for(name: str, engine: Any | None = None) -> str | None:
    """Best-effort resolve a display name back to a section key. Holds ``LOCK``."""
    snap = (engine or ENGINE).state()
    sections = snap.get("sections") or {}
    if isinstance(sections, dict):
        for key, val in sections.items():
            disp = val.get("name") if isinstance(val, dict) else val
            if key == name or disp == name:
                return key
    elif isinstance(sections, list):
        for val in sections:
            if isinstance(val, dict) and (val.get("key") == name or val.get("name") == name):
                return val.get("key")
    return None


_THEME_COLORS = (
    "#b8c99d",
    "#d9b08c",
    "#a8c5d6",
    "#d1b3c4",
    "#c7b7e5",
    "#e0c879",
)


def _ensure_theme(engine: Any, name: str | None) -> str | None:
    """Resolve or create a concise theme while preserving legacy sections.

    Sections are the persisted layout primitive. The thematic product model
    deliberately reuses that durable shape so old workspaces need no migration.
    """
    clean = " ".join(str(name or "").split()).strip()
    if not clean:
        return None
    existing = _section_key_for(clean, engine)
    if existing:
        return existing
    if clean.lower() in {"field", "inbox"}:
        return "field" if "field" in engine.sections else None
    key = re.sub(r"[^a-z0-9]+", "-", clean.lower()).strip("-")[:48] or "theme"
    base = key
    suffix = 2
    while key in engine.sections:
        key = f"{base[:44]}-{suffix}"
        suffix += 1
    color = _THEME_COLORS[len(engine.sections) % len(_THEME_COLORS)]
    engine.add_section(key, clean[:80], color)
    return key


# --------------------------------------------------------------------------
# worker paths (run off the request thread)
# --------------------------------------------------------------------------


def _run_atomize(text: str, workspace_id: str | None = None) -> None:
    target_workspace = workspace_id if workspace_id is not None else WORKSPACE_ID
    try:
        with LOCK:
            target_engine, _is_current = _engine_for_workspace(target_workspace)
            snap = target_engine.state()
            sections = [
                str(s.get("name") or s.get("key"))
                for s in (snap.get("sections") or [])
                if isinstance(s, dict)
            ]
            existing_cards = [
                {
                    "id": str(card.get("id") or ""),
                    "text": str(card.get("text") or ""),
                    "section": str(card.get("section") or ""),
                    "artifact_type": str(card.get("artifact_type") or "claim"),
                    "occurrence_count": int(card.get("occurrence_count") or 1),
                }
                for card in (snap.get("cards") or [])
                if isinstance(card, dict) and not card.get("archived")
            ]
        try:
            claims = workers.atomize(
                text, sections, existing_cards=existing_cards
            )
        except TypeError as exc:
            # Rolling/test compatibility for a two-argument atomizer.
            if "existing_cards" not in str(exc):
                raise
            claims = workers.atomize(text, sections)
    except Exception:
        traceback.print_exc()
        claims = [{
            "text": text,
            "section": None,
            "artifact_type": "question" if text.rstrip().endswith("?") else "observation",
            "relation": "new",
            "foot": "atomize failed · raw",
        }]

    if not claims:
        claims = [{
            "text": text,
            "section": None,
            "artifact_type": "question" if text.rstrip().endswith("?") else "observation",
            "relation": "new",
            "foot": "atomize empty · raw",
        }]

    with LOCK:
        target_engine, _is_current = _engine_for_workspace(target_workspace)
        transaction = (
            STORE.transaction()
            if STORE is not None and target_workspace is not None
            else nullcontext()
        )
        with transaction:
            created_cards = []
            repeated_cards = []
            for claim in claims:
                try:
                    artifact_text = str(claim.get("text") or text).strip()
                    relation = str(claim.get("relation") or "new").lower()
                    canonical_id = str(claim.get("canonical_id") or "").strip()
                    canonical = None
                    if canonical_id:
                        candidate = target_engine.cards.get(canonical_id)
                        if candidate is not None and not candidate.archived:
                            canonical = candidate
                    if canonical is None:
                        canonical = target_engine.find_canonical(artifact_text)
                    if canonical is not None and relation in {
                        "repeat", "refinement"
                    }:
                        card = target_engine.record_occurrence(
                            canonical.id,
                            artifact_text,
                            relation=relation,
                            foot=str(claim.get("foot") or ""),
                        )
                        _bank_occurrence(
                            target_workspace,
                            card,
                            artifact_text,
                            relation=relation,
                        )
                        repeated_cards.append(card)
                        continue
                    # Exact duplicates are always consolidated even when a
                    # provider forgets to label them as repeats.
                    if canonical is not None and relation != "contradiction":
                        card = target_engine.record_occurrence(
                            canonical.id,
                            artifact_text,
                            relation="repeat",
                            foot=str(claim.get("foot") or ""),
                        )
                        _bank_occurrence(
                            target_workspace,
                            card,
                            artifact_text,
                            relation="repeat",
                        )
                        repeated_cards.append(card)
                        continue
                    section = _ensure_theme(
                        target_engine,
                        claim.get("theme") or claim.get("section"),
                    )
                    card = target_engine.propose(
                        artifact_text,
                        section=section,
                        artifact_type=str(
                            claim.get("artifact_type") or "observation"
                        ),
                        foot=claim.get("foot", ""),
                    )
                except Exception:
                    traceback.print_exc()
                    continue
                created_cards.append(card)
                # Persistence errors intentionally escape: the outer transaction
                # must roll back rather than commit a snapshot without its bank row.
                _bank_card(
                    target_workspace, card, source="atomize",
                    engine=target_engine,
                )
            try:
                target_engine.enforce_cap()
            except Exception:
                traceback.print_exc()
            _persist_engine(
                target_workspace,
                target_engine,
                event_kind="claims.created",
                event_payload={
                    "submitted_text": text,
                    "card_ids": [card.id for card in created_cards],
                    "repeated_card_ids": sorted(
                        {card.id for card in repeated_cards}
                    ),
                },
                increment_context=True,
            )


def _run_recognize(
    a_id: str,
    b_id: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """SPEC §2.2/§2.4 — ask about one pair; land a candidate or an attempt row.

    This is the *only* place a recognition request is issued, shared by the
    background scanner and the human-initiated ``propose_click``. It never
    materializes field content: the sole possible outputs are a row in the
    emergence inbox and a row in the never-retry ledger.

    Runs OFF the lock while the provider is in flight, then re-resolves the
    engine — a workspace switch or a competing tick may have landed in between,
    and the pair must be re-validated against current state rather than the
    state that motivated the ask.
    """
    target_workspace = workspace_id if workspace_id is not None else WORKSPACE_ID
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(target_workspace)
        try:
            a = _card_dict(target_engine._card(a_id))
            b = _card_dict(target_engine._card(b_id))
        except KeyError:
            return {"click": False, "error": "unknown position"}
        question = target_engine.question

    try:
        out = workers.recognize(a, b, question)
    except Exception:
        traceback.print_exc()
        # An unexpected worker crash is an outage, not a judgment about the
        # pair: it must record `failed`, which does not consume it (§2.3).
        out = {"click": False, "failed": True, "provenance": "recognizer crashed"}

    with LOCK:
        target_engine, _is_current = _engine_for_workspace(target_workspace)
        transaction = (
            STORE.transaction()
            if STORE is not None and target_workspace is not None
            else nullcontext()
        )
        with transaction:
            result: dict[str, Any]
            event_kind = "click.no_click"
            event_payload: dict[str, Any] = {"position_ids": [a_id, b_id]}
            try:
                if out.get("failed"):
                    target_engine.record_attempt(a_id, b_id, "failed")
                    result = {"click": False, "failed": True}
                    event_kind = "click.failed"
                elif not out.get("click"):
                    target_engine.record_attempt(a_id, b_id, "no_click")
                    result = {"click": False, "failed": False}
                else:
                    proposal = engine_mod.ClickProposal(
                        abstraction=str(out.get("abstraction") or ""),
                        specializer_a=str(out.get("specializer_a") or ""),
                        specializer_b=str(out.get("specializer_b") or ""),
                        scope_boundary=str(out.get("scope_boundary") or ""),
                    )
                    try:
                        candidate = target_engine.propose_click(
                            a_id, b_id, proposal
                        )
                    except engine_mod.GateFailure as exc:
                        # propose_click already wrote the `gate_failed` row. A
                        # failed gate is not a card: it emits nothing (§1.3).
                        result = {
                            "click": False,
                            "gate_failed": exc.gate,
                            "detail": str(exc),
                        }
                        event_kind = "click.gate_failed"
                        event_payload["gate"] = exc.gate
                    else:
                        result = {"click": True, "candidate": candidate.to_dict()}
                        event_kind = "click.proposed"
                        event_payload["candidate_id"] = candidate.id
            except Exception as exc:
                traceback.print_exc()
                result = {"click": False, "error": str(exc)}
                event_kind = "click.failed"
            event_payload["provenance"] = str(out.get("provenance") or "")
            _persist_engine(
                target_workspace,
                target_engine,
                event_kind=event_kind,
                event_payload=event_payload,
            )
    return result


# --------------------------------------------------------------------------
# metabolism
# --------------------------------------------------------------------------


def _embedding_ranker(workspace_id: str | None, engine: Any):
    """Build the §2.2 cosine ranker over ``idea_embeddings``, or ``None``.

    Embeddings are a **bounded ranking index only** — never an acceptance
    threshold, never a materialization trigger. Returning ``None`` when no
    vectors exist is deliberate: with no ranking signal every cosine reads 0.0,
    which is below ``CLICK_FLOOR``, so the tick does nothing rather than
    falling back to "rank by something else". Ranking absence must degrade to
    silence, not to a different selection rule.
    """
    if STORE is None or workspace_id is None:
        return None
    vectors: dict[str, list[float]] = {}
    try:
        occurrences = STORE.list_workspace_ideas(workspace_id)
    except Exception:
        return None
    for item in occurrences:
        local_ref = str(item.get("local_ref") or "")
        if not local_ref or local_ref in vectors:
            continue
        embedding = STORE.get_embedding(
            item["idea"].id, model=EMBEDDING_MODEL, version=EMBEDDING_VERSION
        )
        if embedding is None:
            continue
        try:
            values = list(
                struct.unpack(f"<{embedding.dimensions}f", embedding.vector)
            )
        except struct.error:
            continue
        norm = math.sqrt(sum(v * v for v in values))
        if norm <= 0.0:
            continue
        vectors[local_ref] = [v / norm for v in values]
    if not vectors:
        return None

    def cosine(a_id: str, b_id: str) -> float:
        va, vb = vectors.get(a_id), vectors.get(b_id)
        if va is None or vb is None or len(va) != len(vb):
            return 0.0
        return sum(x * y for x, y in zip(va, vb))

    return cosine


def _fleet_budget_remaining() -> int:
    """§2.2 — the fleet-wide daily cap on recognition spend across workspaces."""
    global _FLEET_BUDGET_DAY, _FLEET_BUDGET_SPENT
    day = int(time.time() // 86400)
    if day != _FLEET_BUDGET_DAY:
        _FLEET_BUDGET_DAY = day
        _FLEET_BUDGET_SPENT = 0
    return max(0, FLEET_CLICK_BUDGET_PER_DAY - _FLEET_BUDGET_SPENT)


def _last_contribution_at(workspace_id: str | None, engine: Any) -> float:
    """Most recent human input, for the §2.2 quiescence gate."""
    stamps = [
        p.occupant.last_seen
        for p in engine.all_positions()
        if p.origin == "human"
    ]
    return max(stamps) if stamps else 0.0


def _inspect_workspace(workspace_id: str | None) -> dict[str, Any]:
    """One structural-inspection tick for one workspace (§2.2).

    The loop **materializes nothing**. It may do exactly three things: expire
    stale click candidates, vacate frames that lost all their instances, and
    select AT MOST ONE pair to ask a question about. Card generation is not in
    its vocabulary — that is the whole point of the redesign, and the reason
    there is no branch here that calls ``propose()``.

    Returns a small report so the caller can spend the fleet budget honestly.
    """
    global _STARVATION_COUNTS
    report: dict[str, Any] = {"scanned": False, "pair": None, "reason": ""}
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)

        # Structural upkeep — no inference, no new material, so it runs on
        # every tick regardless of quiescence or budget.
        expired = target_engine.expire_candidates()
        vacated = target_engine.vacate_empty_frames()
        if expired or vacated:
            _persist_engine(
                workspace_id,
                target_engine,
                event_kind="field.inspected",
                event_payload={
                    "expired_candidates": [c.id for c in expired],
                    "vacated_frames": [p.id for p in vacated],
                },
            )
        report["expired"] = len(expired)
        report["vacated"] = len(vacated)

        # Emission rate is governed by inbox capacity, not timer frequency
        # (§2.2/§2.4): a fourth candidate is not generated until one resolves.
        open_candidates = len(target_engine.open_candidates())
        if open_candidates >= engine_mod.INBOX_CAP:
            report["reason"] = "inbox full"
            return report

        # §2.2 anti-starvation: N new claims with zero scans forces exactly one
        # wave, so budget exhaustion or a workspace that never goes quiet can
        # never silently disable recognition forever.
        claim_count = sum(
            1 for p in target_engine.all_positions()
            if p.floor_kind == "claim" and p.origin == "human"
        )
        seen = _STARVATION_COUNTS.get(workspace_id or "", 0)
        starving = claim_count - seen >= STARVATION_CLAIMS

        idle = time.time() - _last_contribution_at(workspace_id, target_engine)
        if not starving and idle < QUIESCENCE_WINDOW:
            report["reason"] = "field not quiescent"
            return report
        if _fleet_budget_remaining() <= 0:
            report["reason"] = "fleet budget exhausted"
            return report

        pair = target_engine.scan_candidates(
            _embedding_ranker(workspace_id, target_engine)
        )
        if pair is None:
            report["reason"] = "no eligible pair above CLICK_FLOOR"
            # A scan that legitimately found nothing still counts as a wave:
            # otherwise a quiet field would re-trigger starvation forever.
            _STARVATION_COUNTS[workspace_id or ""] = claim_count
            return report
        _STARVATION_COUNTS[workspace_id or ""] = claim_count
        report["pair"] = pair
    return report


def _metabolism_loop() -> None:
    """SPEC §2.2 — the background loop as a structural inspector.

    Cadence 30s (from 3.0s), quiescence-gated, inbox-governed, fleet-budgeted.
    Per tick, per workspace, it selects at most one pair and asks about it. The
    old loop drained the pair space exhaustively at 3-second intervals and
    materialized a card for every pair; nothing here can do that, because
    ``_run_recognize`` cannot produce field material at all.
    """
    global _FLEET_BUDGET_SPENT
    while not _STOP.wait(METABOLISM_PERIOD):
        try:
            with LOCK:
                workspace_ids = (
                    [workspace.id for workspace in STORE.list_workspaces()]
                    if STORE is not None
                    else [WORKSPACE_ID]
                )
            for workspace_id in workspace_ids:
                report = _inspect_workspace(workspace_id)
                pair = report.get("pair")
                if not pair:
                    continue
                _FLEET_BUDGET_SPENT += 1
                _run_recognize(pair[0], pair[1], workspace_id)
        except Exception:
            traceback.print_exc()


def _card_from_snapshot(workspace_id: str, local_ref: str) -> dict[str, Any] | None:
    if STORE is None:
        return None
    workspace = STORE.load_workspace(workspace_id)
    raw = workspace.snapshot.get("cards") or {}
    if isinstance(raw, dict):
        card = raw.get(local_ref)
        return dict(card) if isinstance(card, dict) else None
    for card in raw:
        if isinstance(card, dict) and card.get("id") == local_ref:
            return dict(card)
    return None


def _outbox_local_ref(dedupe_key: str, workspace_id: str) -> str | None:
    prefix = f"magpie-card:{workspace_id}:"
    if not dedupe_key.startswith(prefix):
        return None
    return dedupe_key[len(prefix):] or None


def _deliver_raven_remember(item: Any) -> None:
    """Deliver one private Raven write and bind its workspace projection."""
    global RAVEN_LAST_ERROR
    if STORE is None:
        return
    payload = dict(item.payload)
    local_ref = (
        _outbox_local_ref(item.dedupe_key, item.workspace_id)
        if item.workspace_id
        else None
    )
    card = (
        _card_from_snapshot(item.workspace_id, local_ref)
        if item.workspace_id and local_ref
        else None
    )
    parents = [str(value) for value in (card or {}).get("parents") or [] if value]
    if parents and item.workspace_id:
        derived_from = [
            STORE.completed_raven_memory_id_for_local_ref(
                item.workspace_id, parent
            )
            for parent in parents
        ]
        if any(memory_id is None for memory_id in derived_from):
            error = "waiting for parent Raven memories"
            if item.attempts >= RAVEN_MAX_ATTEMPTS:
                STORE.mark_raven_remember(item.id, "failed", error=error)
                return
            delay = min(60.0, 2.0 ** min(item.attempts, 6))
            STORE.mark_raven_remember(
                item.id,
                "pending",
                error=error,
                available_at=time.time() + delay,
            )
            return
        payload.pop("source", None)
        payload["hints"] = {"derived_from": derived_from}
    result = RAVEN.remember(**payload)
    if result.ok:
        memory_id = str((result.value or {}).get("id") or "").strip()
        if not memory_id:
            result = type(result)(
                ok=False,
                value=result.value,
                error="Raven remember returned no memory id",
                unavailable=False,
                disabled=False,
            )
        else:
            STORE.mark_raven_remember(
                item.id, "completed", raven_memory_id=memory_id
            )
            if item.workspace_id:
                section = str((card or {}).get("section") or "field")
                STORE.upsert_raven_projection(
                    item.workspace_id,
                    memory_id,
                    local_ref=local_ref,
                    section=section,
                    mass=float((card or {}).get("mass") or 1.0),
                    pinned=bool((card or {}).get("pinned")),
                    hidden=bool((card or {}).get("archived")),
                    local_status="adopted",
                    metadata={
                        "origin": "magpie",
                        "kind": (card or {}).get("kind"),
                    },
                )
            RAVEN_LAST_ERROR = None
            return
    error = str(result.error or "Raven remember failed")[:500]
    RAVEN_LAST_ERROR = error
    if result.disabled:
        # Configuration may be added while Magpie stays up. Do not burn retries.
        STORE.mark_raven_remember(
            item.id,
            "pending",
            error=error,
            available_at=time.time() + 30.0,
        )
    elif item.attempts >= RAVEN_MAX_ATTEMPTS:
        STORE.mark_raven_remember(item.id, "failed", error=error)
    else:
        delay = min(300.0, 2.0 ** min(item.attempts, 8))
        STORE.mark_raven_remember(
            item.id,
            "pending",
            error=error,
            available_at=time.time() + delay,
        )


def _raven_outbox_loop() -> None:
    while not _STOP.wait(RAVEN_OUTBOX_PERIOD):
        try:
            if STORE is None or not RAVEN.enabled:
                continue
            for item in STORE.claim_raven_remembers(limit=4, lease_seconds=120):
                _deliver_raven_remember(item)
        except Exception:
            traceback.print_exc()


def _recall_workspace(
    workspace_id: str,
    query: str = "",
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Proxy a Raven recall and persist only workspace-local shelf exposure."""
    global RAVEN_LAST_ERROR
    if STORE is None:
        raise RuntimeError("workspace storage is not initialized")
    workspace = STORE.load_workspace(workspace_id)
    clean_query = str(query or workspace.question or "").strip()
    if not clean_query:
        raise ValueError("query required when the workspace has no question")
    limit = max(1, min(int(limit), 25))
    context_version = workspace.context_version
    result = RAVEN.recall(clean_query, limit=limit, expand=1)
    if not result.ok:
        RAVEN_LAST_ERROR = str(result.error or "Raven recall failed")
        if result.unavailable:
            raise ServiceUnavailable(RAVEN_LAST_ERROR)
        raise ValueError(RAVEN_LAST_ERROR)
    current = STORE.load_workspace(workspace_id)
    if current.context_version != context_version:
        return {
            "workspace_id": workspace_id,
            "query": clean_query,
            "stale": True,
            "suggestions": [],
        }
    existing = {
        exposure.raven_memory_id: exposure
        for exposure in STORE.list_raven_exposures(workspace_id)
    }
    projections = {
        projection.raven_memory_id
        for projection in STORE.list_raven_projections(workspace_id)
        if not projection.hidden
    }
    suggestions: list[dict[str, Any]] = []
    for memory in (result.value or {}).get("results") or []:
        if not isinstance(memory, dict):
            continue
        memory_id = str(memory.get("id") or "").strip()
        content = str(memory.get("content") or "").strip()
        if not memory_id or not content or memory_id in projections:
            continue
        # §4 — the suppression registry and the durable dismissal list. Both
        # are workspace-local and reversible; neither mutates Raven.
        if STORE.recall_is_blocked(workspace_id, memory_id):
            continue
        prior = existing.get(memory_id)
        if prior is not None and prior.status == "dismissed":
            continue
        exposure = STORE.upsert_raven_exposure(
            workspace_id,
            memory_id,
            status="suggested",
            reason=f"Recall for: {clean_query}",
            context_version=context_version,
            metadata={"memory": memory, "query": clean_query},
        )
        suggestions.append(_raven_exposure_dict(exposure))
    RAVEN_LAST_ERROR = None
    return {
        "workspace_id": workspace_id,
        "query": clean_query,
        "stale": False,
        "suggestions": suggestions,
    }


def _adopt_one_position(
    workspace_id: str,
    engine: Any,
    text: str,
    *,
    section: str | None,
    foot: str,
    supports: list[str] | None = None,
) -> Any:
    """Adopt one memory as a QUARANTINED position, deduped by fingerprint.

    SPEC §4: everything adopted arrives ``external=True`` and
    ``needs_human`` — excluded from the click scanner until a human pins it.
    Recall can inform without breeding.

    Dedupe is by ``idea_fingerprint`` onto an existing position rather than
    minting a duplicate: the one job position ids cannot do alone
    (Appendix B #2). Without it a re-adopted memory could mint a fresh position
    and walk around the never-retry ledger.
    """
    clean = str(text or "").strip()
    existing = (
        STORE.find_position_by_fingerprint(workspace_id, clean)
        if STORE is not None
        else None
    )
    if existing is not None and existing.position_id in engine.positions:
        return engine.position(existing.position_id).occupant
    supports = [sid for sid in (supports or []) if sid in engine.positions]
    card = engine.propose(
        clean,
        section=section,
        state="needs_human",
        foot=foot,
        origin="recall",
        external=True,
        floor_kind="frame" if supports else "claim",
        supports=supports or None,
    )
    return card


def _adopt_raven_floor(
    workspace_id: str,
    engine: Any,
    memory: dict[str, Any],
) -> list[Any]:
    """Reconstruct the floor of a banked ``human_curated_frame`` (§4, Delta 2).

    A frame was banked with ``hints={"derived_from": [floor memory ids]}``, so
    the ladder is recoverable: adopt those floor memories first as quarantined
    claim positions, and the caller stands the frame on them with its support
    edges intact. Recall arrives AT an altitude — never as loose extra ideas
    dumped on one floor.

    A floor member that cannot be fetched is simply absent; the frame then
    stands on what was actually recovered rather than on a fabricated edge.
    """
    hints = dict(memory.get("hints") or {})
    floor_ids = [
        str(value).strip()
        for value in (hints.get("derived_from") or [])
        if str(value or "").strip()
    ]
    if not floor_ids:
        return []
    out: list[Any] = []
    for floor_id in floor_ids:
        if STORE is not None and STORE.recall_is_blocked(workspace_id, floor_id):
            continue
        node: dict[str, Any] = {}
        fetched = RAVEN.get(floor_id, depth=0)
        if fetched.ok:
            node = dict((fetched.value or {}).get("node") or {})
        floor_text = str(node.get("content") or "").strip()
        if not floor_text:
            continue
        child = _adopt_one_position(
            workspace_id,
            engine,
            floor_text,
            section=None,
            foot=f"Raven floor · {node.get('kind', 'memory')}",
        )
        position = engine.position(child.id)
        # Grounding dates arrive intact: the floor's history is what makes the
        # reconstructed sub-ladder honest rather than freshly minted (§4).
        grounded = node.get("last_grounded_at") or node.get("created_at")
        if grounded is not None:
            try:
                position.last_grounded_at = float(grounded)
            except (TypeError, ValueError):
                pass
        if STORE is not None:
            STORE.upsert_raven_projection(
                workspace_id,
                floor_id,
                local_ref=child.id,
                section=child.section,
                mass=child.mass,
                local_status="adopted",
                metadata={"memory": node, "origin": "raven", "floor": True},
            )
        out.append(child)
    return out


def _adopt_raven_memory(
    workspace_id: str,
    memory_id: str,
    *,
    section: str | None = None,
) -> dict[str, Any]:
    """Create a quarantined local sub-ladder for a Raven memory (§4).

    Never re-remembers: adoption is inbound only.
    """
    if STORE is None:
        raise RuntimeError("workspace storage is not initialized")
    memory_id = str(memory_id or "").strip()
    if not memory_id:
        raise ValueError("memory_id required")
    existing_projection = STORE.get_raven_projection(workspace_id, memory_id)
    if (
        existing_projection is not None
        and existing_projection.local_ref
        and not existing_projection.hidden
    ):
        existing_card = _card_from_snapshot(
            workspace_id, existing_projection.local_ref
        )
        if existing_card is not None and not existing_card.get("archived"):
            return {
                "workspace_id": workspace_id,
                "card": existing_card,
                "projection": _raven_projection_dict(existing_projection),
                "already_adopted": True,
            }
    exposure = next(
        (
            item
            for item in STORE.list_raven_exposures(workspace_id)
            if item.raven_memory_id == memory_id
        ),
        None,
    )
    memory = dict((exposure.metadata if exposure else {}).get("memory") or {})
    if not memory:
        fetched = RAVEN.get(memory_id, depth=1)
        if not fetched.ok:
            if fetched.unavailable:
                raise ServiceUnavailable(str(fetched.error or "Raven unavailable"))
            raise ValueError(str(fetched.error or "Raven memory not found"))
        memory = dict((fetched.value or {}).get("node") or {})
    content = str(memory.get("content") or "").strip()
    if not content:
        raise ValueError("Raven memory has no content")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        section_key = str(section or "").strip() or None
        if section_key:
            section_key = next(
                (
                    key
                    for key, value in target_engine.sections.items()
                    if key == section_key or str(value.get("name") or "") == section_key
                ),
                section_key,
            )
        with STORE.transaction():
            floor_cards = _adopt_raven_floor(
                workspace_id, target_engine, memory
            )
            card = _adopt_one_position(
                workspace_id,
                target_engine,
                content,
                section=section_key,
                foot=(
                    f"Raven · {memory.get('kind', 'memory')} · "
                    f"{memory.get('state', 'open')} · "
                    f"{float(memory.get('effective_confidence') or 0):.2f}"
                ),
                supports=[child.id for child in floor_cards],
            )
            idea, _created = STORE.upsert_idea(
                content,
                kind=str(card.kind or "claim"),
                metadata={"raven_memory_id": memory_id},
            )
            STORE.link_idea(
                workspace_id,
                idea.id,
                local_ref=card.id,
                source="raven",
                metadata={"raven_memory_id": memory_id},
            )
            projection = STORE.upsert_raven_projection(
                workspace_id,
                memory_id,
                local_ref=card.id,
                section=card.section,
                mass=card.mass,
                pinned=card.pinned,
                local_status="adopted",
                metadata={"memory": memory, "origin": "raven"},
            )
            STORE.upsert_raven_exposure(
                workspace_id,
                memory_id,
                status="adopted",
                reason="Imported into the workspace field",
                context_version=STORE.load_workspace(workspace_id).context_version,
                metadata={"memory": memory},
            )
            target_engine.enforce_cap()
            _persist_engine(
                workspace_id,
                target_engine,
                event_kind="memory.adopted",
                event_payload={"memory_id": memory_id, "card_id": card.id},
                increment_context=True,
            )
    return {
        "workspace_id": workspace_id,
        "card": _card_dict(card),
        "projection": _raven_projection_dict(projection),
    }


def _dismiss_raven_memory(workspace_id: str, memory_id: str) -> dict[str, Any]:
    if STORE is None:
        raise RuntimeError("workspace storage is not initialized")
    prior = next(
        (
            item
            for item in STORE.list_raven_exposures(workspace_id)
            if item.raven_memory_id == memory_id
        ),
        None,
    )
    with STORE.transaction():
        exposure = STORE.upsert_raven_exposure(
            workspace_id,
            memory_id,
            status="dismissed",
            reason="Dismissed from this workspace shelf",
            context_version=STORE.load_workspace(workspace_id).context_version,
            metadata={} if prior is None else prior.metadata,
        )
        # §4 — durable by memory id, not just an exposure row recall rewrites:
        # a human-dismissed exposure never resurfaces in this workspace,
        # regardless of future recall scoring.
        STORE.dismiss_memory(
            workspace_id, memory_id, reason="dismissed from the workspace shelf"
        )
    return {"workspace_id": workspace_id, "exposure": _raven_exposure_dict(exposure)}


# --------------------------------------------------------------------------
# provider status
# --------------------------------------------------------------------------


def _provider_status() -> list[dict]:
    out: list[dict] = []
    try:
        chain = providers_mod.get_chain()
    except Exception:
        return out
    providers = getattr(chain, "providers", None) or []
    last = getattr(chain, "last_provider", None)
    for p in providers:
        name = getattr(p, "name", None) or p.__class__.__name__
        entry = {"name": name, "last": name == last}
        avail = getattr(p, "available", None)
        if callable(avail):
            try:
                entry["available"] = bool(avail())
            except Exception:
                entry["available"] = False
        out.append(entry)
    return out


def _workspace_dict(workspace: Workspace) -> dict:
    return {
        "id": workspace.id,
        "name": workspace.name,
        "question": workspace.question,
        "context_version": workspace.context_version,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at,
        "current": workspace.id == WORKSPACE_ID,
    }


def _current_workspace() -> Workspace | None:
    if STORE is None or WORKSPACE_ID is None:
        return None
    return STORE.load_workspace(WORKSPACE_ID)


def _api_workspaces(_body: dict) -> dict:
    if STORE is None:
        return {"workspaces": []}
    return {"workspaces": [_workspace_dict(ws) for ws in STORE.list_workspaces()]}


def _api_workspace_current(_body: dict) -> dict:
    workspace = _current_workspace()
    return {"workspace": None if workspace is None else _workspace_dict(workspace)}


def _api_workspace_create(body: dict) -> dict:
    global ENGINE, WORKSPACE_ID
    if STORE is None:
        raise RuntimeError("workspace storage is not initialized")
    name = str(body.get("name") or "").strip()
    if not name:
        raise ValueError("workspace name required")
    question = str(body.get("question") or "").strip()
    fresh = engine_mod.Engine()
    if question:
        fresh.seed(question)
    with LOCK:
        with STORE.transaction():
            if ENGINE is not None and WORKSPACE_ID is not None:
                _persist()
            workspace = STORE.create_workspace(
                name,
                question=question,
                snapshot=fresh.state(),
            )
            STORE.set_current_workspace(workspace.id)
            workspace = _persist_engine(
                workspace.id,
                fresh,
                event_kind="workspace.created",
                event_payload={"name": name, "question": question},
                increment_context=True,
            ) or workspace
        WORKSPACE_ID = workspace.id
        ENGINE = fresh
        state = fresh.state()
    return {"workspace": _workspace_dict(workspace), "state": state}


def _api_workspace_open(body: dict) -> dict:
    global ENGINE, WORKSPACE_ID
    if STORE is None:
        raise RuntimeError("workspace storage is not initialized")
    workspace_id = str(body.get("id") or "").strip()
    if not workspace_id:
        raise ValueError("workspace id required")
    with LOCK:
        workspace = STORE.load_workspace(workspace_id)
        resumed = engine_mod.Engine.from_state(workspace.snapshot)
        with STORE.transaction():
            if ENGINE is not None and WORKSPACE_ID is not None:
                _persist()
            workspace = STORE.set_current_workspace(workspace_id)
            STORE.append_event(
                "workspace.resumed",
                {"bank_revision": STORE.bank_revision()},
                workspace_id=workspace.id,
                context_version=workspace.context_version,
            )
        ENGINE = resumed
        WORKSPACE_ID = workspace.id
        state = resumed.state()
    return {"workspace": _workspace_dict(workspace), "state": state}


# --------------------------------------------------------------------------
# API handlers — each returns a JSON-serializable dict (merged into ok:true)
# --------------------------------------------------------------------------


def _api_state(body: dict) -> dict:
    requested = str(body.get("workspace_id") or "").strip() or None
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(requested)
        snap = target_engine.state()
        workspace = (
            _current_workspace()
            if requested is None or requested == WORKSPACE_ID
            else (STORE.load_workspace(requested) if STORE is not None else None)
        )
    snap = dict(snap)
    snap["providers"] = _provider_status()
    if workspace is not None:
        snap["workspace"] = {
            **_workspace_dict(workspace),
            "bank_revision": STORE.bank_revision() if STORE is not None else 0,
        }
        snap["memory_shelf"] = _memory_state(workspace.id)
    return snap


def _api_seed(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    question = (body.get("question") or "").strip()
    if not question:
        raise ValueError("question required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        target_engine.seed(question)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="question.changed",
            event_payload={"question": question},
            increment_context=True,
        )
    return {"question": question}


def _api_propose(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("text required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        workspace = _persist_engine(
            workspace_id,
            target_engine,
            event_kind="thought.submitted",
            event_payload={"text": text},
            increment_context=True,
        )
    threading.Thread(
        target=_run_atomize, args=(text, workspace_id), daemon=True
    ).start()
    return {
        "queued": True,
        "workspace_id": workspace_id,
        "context_version": None if workspace is None else workspace.context_version,
    }


def _position_dict(engine: Any, position: Any) -> dict[str, Any]:
    """One position on the wire, with everything derived actually derived.

    ``altitude`` and ``support`` are recomputed here on every call rather than
    read from the position (§1.5): the wire format is a projection of the
    floor, so no client can ever be handed a support figure the floor does not
    currently justify.
    """
    out = position.to_dict()
    out["altitude"] = engine.altitude(position.id)
    out["lineage_mass"] = engine.lineage_mass(position.id)
    out["staleness"] = engine.staleness(position.id)
    if position.floor_kind == "frame":
        out["support"] = engine.frame_support(position.id)
        out["instance_count"] = sum(
            1 for sid in position.supports
            if sid in engine.positions
            and engine.positions[sid].folded_under == position.id
        )
        out["ungrounded_slots"] = [
            slot.id for slot in engine.ungrounded_slots(position.id)
        ]
    else:
        out["support_state"] = position.support_state
        out["receipt"] = position.receipt
    return out


def _api_field(body: dict) -> dict:
    """§3.1/§3.2 — the field at one altitude, defaulting to the top."""
    workspace_id = _required_workspace_id(body)
    raw_altitude = body.get("altitude")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        alts = target_engine.altitudes()
        max_altitude = max(alts.values()) if alts else 0
        altitude = (
            max_altitude if raw_altitude is None else max(0, int(raw_altitude))
        )
        positions = [
            _position_dict(target_engine, p)
            for p in target_engine.live_positions(altitude)
        ]
    return {
        "workspace_id": workspace_id,
        "altitude": altitude,
        "max_altitude": max_altitude,
        "positions": positions,
    }


def _api_descend(body: dict) -> dict:
    """§3.1/§3.2 — the frame's floor, folded instances included."""
    workspace_id = _required_workspace_id(body)
    position_id = str(body.get("position_id") or body.get("id") or "").strip()
    if not position_id:
        raise ValueError("position_id required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        frame = target_engine.position(position_id)
        floor = [
            _position_dict(target_engine, child)
            for child in target_engine.descend(position_id)
        ]
        header = _position_dict(target_engine, frame)
    return {"workspace_id": workspace_id, "frame": header, "floor": floor}


def _api_propose_click(body: dict) -> dict:
    """§2.1/§2.4 — the human-initiated recognition request.

    This is what ``/api/collide`` became. The difference is the whole thesis:
    the old endpoint created a card in the field and then asked a model to
    write it; this one asks a question whose only possible destinations are the
    emergence inbox and the never-retry ledger.
    """
    workspace_id = _required_workspace_id(body)
    a = str(body.get("a") or "").strip()
    b = str(body.get("b") or "").strip()
    if not a or not b:
        raise ValueError("a and b required")
    if a == b:
        raise ValueError("a and b must identify different positions")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        target_engine.position(a)
        target_engine.position(b)
        if target_engine.pair_consumed(a, b):
            raise ValueError("pair already attempted; use reconsider_pair")
        if len(target_engine.open_candidates()) >= engine_mod.INBOX_CAP:
            raise ValueError("emergence inbox is full")
    result = _run_recognize(a, b, workspace_id)
    return {"workspace_id": workspace_id, **result}


def _api_pending_clicks(body: dict) -> dict:
    """§2.4/§7.2 — the emergence inbox, plus near-miss drilldown on request."""
    workspace_id = _required_workspace_id(body)
    include_rejected = bool(body.get("include_rejected"))
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        candidates = [c.to_dict() for c in target_engine.open_candidates()]
        for candidate in candidates:
            candidate["instances"] = [
                _card_dict(target_engine._card(candidate["position_a"])),
                _card_dict(target_engine._card(candidate["position_b"])),
            ]
        out: dict[str, Any] = {
            "workspace_id": workspace_id,
            "candidates": candidates,
            "capacity": engine_mod.INBOX_CAP,
            "click_budget_remaining": target_engine.click_budget_remaining(),
        }
        if include_rejected:
            # Near-misses are private metrics with an operator inspection
            # surface (Appendix B #4) — never field spam.
            out["rejected"] = [
                dict(row) for row in target_engine.click_attempts.values()
                if row["outcome"] in ("gate_failed", "no_click", "declined",
                                      "expired", "failed")
            ]
        _persist_engine(workspace_id, target_engine)
    return out


def _api_resolve_click(body: dict) -> dict:
    """§2.4 — accept / accept-with-edit / decline one inbox candidate."""
    workspace_id = _required_workspace_id(body)
    candidate_id = str(body.get("candidate_id") or "").strip()
    verdict = str(body.get("verdict") or "").strip().lower()
    if not candidate_id:
        raise ValueError("candidate_id required")
    if verdict not in ("accept", "decline"):
        raise ValueError("verdict must be accept or decline")
    confirmed_by = str(body.get("confirmed_by") or "").strip()
    if verdict == "accept" and not confirmed_by:
        # A click is confirmed by a human or not at all (§1.6): the
        # confirmation is provenance, and unattributed provenance is none.
        raise ValueError("confirmed_by required to accept a click")
    text = body.get("text")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        transaction = (
            STORE.transaction()
            if STORE is not None and workspace_id is not None
            else nullcontext()
        )
        with transaction:
            if verdict == "decline":
                candidate = target_engine.decline_click(candidate_id)
                payload: dict[str, Any] = {"candidate": candidate.to_dict()}
                event_kind = "click.declined"
            else:
                frame = target_engine.confirm_click(
                    candidate_id,
                    confirmed_by=confirmed_by,
                    text=str(text).strip() if text else None,
                )
                payload = {
                    "frame": _position_dict(target_engine, frame),
                    "folded": [
                        sid for sid in frame.supports
                        if target_engine.positions[sid].folded_under == frame.id
                    ],
                }
                event_kind = "click.confirmed"
                # A confirmed fold is the one machine-adjacent artifact a human
                # signed, so it is the one that may leave the workspace (§4).
                _bank_card(
                    workspace_id,
                    frame.occupant,
                    source="click",
                    engine=target_engine,
                )
            target_engine.enforce_cap()
            _persist_engine(
                workspace_id,
                target_engine,
                event_kind=event_kind,
                event_payload={
                    "candidate_id": candidate_id,
                    "confirmed_by": confirmed_by or None,
                },
                increment_context=True,
            )
    return {"workspace_id": workspace_id, **payload}


def _api_reconsider_pair(body: dict) -> dict:
    """§2.3 — the deliberate, provenance-visible retry door."""
    workspace_id = _required_workspace_id(body)
    a = str(body.get("a") or "").strip()
    b = str(body.get("b") or "").strip()
    if not a or not b:
        raise ValueError("a and b required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        row = target_engine.reconsider_pair(a, b)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="click.reconsidered",
            event_payload={"position_ids": [a, b],
                           "operation_version": row["operation_version"]},
            increment_context=True,
        )
    return {"workspace_id": workspace_id, "attempt": row}


def _api_unfold(body: dict) -> dict:
    """§1.6 — always cheap: instances return, the frame position is vacated."""
    workspace_id = _required_workspace_id(body)
    frame_id = str(body.get("frame_id") or body.get("id") or "").strip()
    if not frame_id:
        raise ValueError("frame_id required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        released = target_engine.unfold(frame_id)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="frame.unfolded",
            event_payload={
                "frame_id": frame_id,
                "released": [p.id for p in released],
            },
            increment_context=True,
        )
        out = [_position_dict(target_engine, p) for p in released]
    return {"workspace_id": workspace_id, "frame_id": frame_id, "released": out}


def _run_derive(frame_id: str, workspace_id: str | None = None) -> dict[str, Any]:
    """§1.4 — the explicit downward operator, off the request thread.

    NEVER background-initiated: there is no call to this from
    ``_metabolism_loop``, and adding one would be the "background-initiated
    derivation" the spec excludes by design.
    """
    target_workspace = workspace_id if workspace_id is not None else WORKSPACE_ID
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(target_workspace)
        frame = target_engine.position(frame_id)
        if frame.floor_kind != "frame":
            raise ValueError("derive applies to frames only")
        frame_card = _card_dict(frame.occupant)
        question = target_engine.question

    try:
        out = workers.derive(frame_card, question)
    except Exception:
        traceback.print_exc()
        out = {"ok": False, "claims": [], "provenance": "derivation crashed"}

    with LOCK:
        target_engine, _is_current = _engine_for_workspace(target_workspace)
        transaction = (
            STORE.transaction()
            if STORE is not None and target_workspace is not None
            else nullcontext()
        )
        with transaction:
            created: list[Any] = []
            if out.get("ok") and out.get("claims"):
                created = target_engine.derive(frame_id, list(out["claims"]))
                for card in created:
                    # Derived-ungrounded claims are bank-ineligible (§1.4/§4);
                    # `_bank_card` refuses them by origin. The call stays so the
                    # local idea-bank row (workspace-local provenance) is
                    # written for every position the same way.
                    _bank_card(
                        target_workspace, card, source="derivation",
                        engine=target_engine,
                    )
            target_engine.enforce_cap()
            _persist_engine(
                target_workspace,
                target_engine,
                event_kind="frame.derived" if created else "derivation.empty",
                event_payload={
                    "frame_id": frame_id,
                    "card_ids": [card.id for card in created],
                    "provenance": str(out.get("provenance") or ""),
                },
                increment_context=True,
            )
            slots = [
                _position_dict(target_engine, slot)
                for slot in target_engine.ungrounded_slots(frame_id)
            ]
    return {
        "workspace_id": target_workspace,
        "frame_id": frame_id,
        "created": [_card_dict(card) for card in created],
        "ungrounded_slots": slots,
        "provenance": str(out.get("provenance") or ""),
    }


def _api_derive(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    frame_id = str(body.get("frame_id") or body.get("id") or "").strip()
    if not frame_id:
        raise ValueError("frame_id required")
    return _run_derive(frame_id, workspace_id)


def _api_verify(body: dict) -> dict:
    """Submit verification if a runtime adapter has been explicitly installed."""
    workspace_id = _required_workspace_id(body)
    card_id = body.get("id")
    if not card_id:
        raise ValueError("id required")
    hook = VERIFICATION_HOOK
    if hook is None:
        raise ValueError("verification runtime is not configured")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        card = target_engine._card(card_id)
        request = VerificationRequest(
            workspace_id=workspace_id,
            card=_card_dict(card),
            question=target_engine.question,
        )
        target_engine.request_verify(card_id)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="verification.started",
            event_payload={"card_id": card_id},
            increment_context=True,
        )
    try:
        job_id = hook.submit(request)
    except Exception:
        with LOCK:
            target_engine, _is_current = _engine_for_workspace(workspace_id)
            target_engine.reopen(card_id, foot="verification submission failed")
            _persist_engine(
                workspace_id,
                target_engine,
                event_kind="verification.submission_failed",
                event_payload={"card_id": card_id},
                increment_context=True,
            )
        raise
    if not job_id or not str(job_id).strip():
        with LOCK:
            target_engine, _is_current = _engine_for_workspace(workspace_id)
            target_engine.reopen(card_id, foot="verification submission failed")
            _persist_engine(
                workspace_id,
                target_engine,
                event_kind="verification.submission_failed",
                event_payload={"card_id": card_id},
                increment_context=True,
            )
        raise ValueError("verification hook returned no job id")
    return {"queued": True, "job_id": str(job_id), "card": request.card}


def _api_judge(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    card_id, verdict = body.get("id"), body.get("verdict")
    if not card_id or verdict not in ("yes", "no", "unknown"):
        raise ValueError("id and verdict (yes|no|unknown) required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        res = target_engine.judge(card_id, verdict)
        target_engine.enforce_cap()
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="claim.resolved" if verdict in ("yes", "no") else "claim.split",
            event_payload={"card_id": card_id, "verdict": verdict},
            increment_context=True,
        )
    if isinstance(res, list):
        return {"cards": [_card_dict(c) for c in res]}
    return {"card": _card_dict(res)}


def _api_keep(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    card_id = body.get("id")
    if not card_id:
        raise ValueError("id required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        res = target_engine.keep(card_id)
        _sync_projection_presentation(workspace_id, res)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="idea.pinned" if res.pinned else "idea.released",
            event_payload={"card_id": card_id},
            increment_context=True,
        )
    return {"card": _card_dict(res)} if res is not None else {}


def _api_kill(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    card_id = body.get("id")
    if not card_id:
        raise ValueError("id required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        res = target_engine.kill(card_id)
        _sync_projection_presentation(workspace_id, res)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="idea.dismissed",
            event_payload={"card_id": card_id},
            increment_context=True,
        )
    return {"card": _card_dict(res)} if res is not None else {}


def _api_move(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    card_id, section = body.get("id"), body.get("section")
    if not card_id or not section:
        raise ValueError("id and section required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        key = _section_key_for(section, target_engine) or section
        res = target_engine.move(card_id, key)
        _sync_projection_presentation(workspace_id, res)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="idea.moved",
            event_payload={"card_id": card_id, "section": key},
            increment_context=True,
        )
    return {"card": _card_dict(res)} if res is not None else {}


_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _api_section(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("name required")
    key = body.get("key") or name.strip().lower().replace(" ", "-")
    # Colors land in a CSS style attribute in the UI. Validate at the boundary
    # so nothing but a hex literal is ever persisted into engine state.
    color = str(body.get("color") or "#c9b8a0").strip()
    if not _HEX_COLOR.match(color):
        raise ValueError("color must be a hex literal like #c9b8a0")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        if key in target_engine.sections:
            raise ValueError("a field with that name already exists")
        target_engine.add_section(key, name, color)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="section.created",
            event_payload={"key": key, "name": name},
        )
    return {"key": key, "name": name, "color": color}


def _api_section_rename(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    key = str(body.get("key") or "").strip()
    name = str(body.get("name") or "").strip()
    if not key or not name:
        raise ValueError("key and name required")
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        section = target_engine.rename_section(key, name)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="section.renamed",
            event_payload={"key": key, "name": name},
            increment_context=True,
        )
    return dict(section)


def _api_harvest(body: dict) -> dict:
    """§3.3 — the decision-ready brief, not a dump.

    ``altitude`` selects the spine's floor; ``None`` means "everything from
    floor 0 up", which is still capped per section. Full state remains at
    ``/api/state``, a debugging surface rather than a deliverable.
    """
    workspace_id = _required_workspace_id(body)
    raw_altitude = body.get("altitude")
    altitude = None if raw_altitude is None else max(0, int(raw_altitude))
    max_items = int(body.get("max_items") or 12)
    with LOCK:
        target_engine, _is_current = _engine_for_workspace(workspace_id)
        brief = target_engine.harvest(altitude=altitude, max_items=max_items)
        _persist_engine(
            workspace_id,
            target_engine,
            event_kind="workspace.harvested",
            event_payload={
                "altitude": brief.get("altitude"),
                "spine": len(brief.get("spine") or []),
            },
        )
    _ensure_runtime()
    path = RUNTIME_DIR / f"harvest-{workspace_id}-{int(time.time())}.json"
    tmp = path.with_suffix(".json.tmp")
    # Same atomicity contract as engine.save: fsync the payload before the
    # rename, so a crash mid-write can never leave a truncated harvest behind.
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(brief, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return {"brief": brief, "path": str(path)}


def _api_recall(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    return _recall_workspace(
        workspace_id,
        str(body.get("query") or ""),
        limit=int(body.get("limit") or 10),
    )


def _api_recall_adopt(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    return _adopt_raven_memory(
        workspace_id,
        str(body.get("memory_id") or ""),
        section=body.get("section"),
    )


def _api_recall_dismiss(body: dict) -> dict:
    workspace_id = _required_workspace_id(body)
    memory_id = str(body.get("memory_id") or "").strip()
    if not workspace_id or not memory_id:
        raise ValueError("workspace_id and memory_id required")
    return _dismiss_raven_memory(workspace_id, memory_id)


def _api_voice_session(body: dict) -> dict:
    """Mint a short-lived private ElevenLabs Agent connection URL.

    The long-lived ElevenLabs key remains server-side. The selected workspace
    is validated here and is also passed by the browser as a conversation
    dynamic variable, so the agent can supply the required workspace_id to MCP.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID", "").strip()
    if not api_key or not agent_id:
        raise ServiceUnavailable(
            "ElevenLabs voice configuration is incomplete "
            "(ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID are required)"
        )
    workspace_id = _required_workspace_id(body)

    query = urlencode({"agent_id": agent_id})
    request = Request(
        "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
        f"?{query}",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ServiceUnavailable(
            "ElevenLabs rejected the voice session request"
        ) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ServiceUnavailable("ElevenLabs voice service is unavailable") from exc
    if not isinstance(payload, dict):
        raise ServiceUnavailable("ElevenLabs returned an invalid voice response")
    signed_url = str(payload.get("signed_url") or "").strip()
    parsed_url = urlparse(signed_url)
    if parsed_url.scheme != "wss" or not parsed_url.netloc:
        raise ServiceUnavailable("ElevenLabs returned an invalid signed voice URL")
    return {"signed_url": signed_url, "agent_id": agent_id}


ROUTES: dict[str, Callable[[dict], dict]] = {
    "/api/workspaces": _api_workspace_create,
    "/api/workspaces/open": _api_workspace_open,
    "/api/seed": _api_seed,
    "/api/propose": _api_propose,
    # §2.1: `/api/collide` becomes `propose_click` — a human-initiated request
    # for the same recognition attempt, landing in the same inbox, never in
    # the field. The old route is gone rather than aliased: an alias would be
    # the second channel the spec excludes by design.
    "/api/click/propose": _api_propose_click,
    "/api/click/pending": _api_pending_clicks,
    "/api/click/resolve": _api_resolve_click,
    "/api/click/reconsider": _api_reconsider_pair,
    "/api/derive": _api_derive,
    "/api/field": _api_field,
    "/api/descend": _api_descend,
    "/api/unfold": _api_unfold,
    "/api/verify": _api_verify,
    "/api/judge": _api_judge,
    "/api/keep": _api_keep,
    "/api/kill": _api_kill,
    "/api/move": _api_move,
    "/api/section": _api_section,
    "/api/section/rename": _api_section_rename,
    "/api/harvest": _api_harvest,
    "/api/recall": _api_recall,
    "/api/recall/adopt": _api_recall_adopt,
    "/api/recall/dismiss": _api_recall_dismiss,
    "/api/voice/session": _api_voice_session,
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "magpie/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing --------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        pass

    def _send(self, code: int, payload: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _ok(self, obj: dict) -> None:
        out = {"ok": True}
        out.update(obj or {})
        self._json(200, out)

    def _err(self, code: int, message: str) -> None:
        self._json(code, {"ok": False, "error": message})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("body must be a JSON object")
        return obj

    # -- verbs -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/state":
            try:
                query = dict(
                    part.split("=", 1) if "=" in part else (part, "")
                    for part in parsed.query.split("&")
                    if part
                )
                self._ok(_api_state({"workspace_id": query.get("workspace_id", "")}))
            except Exception as exc:
                traceback.print_exc()
                self._err(500, str(exc))
            return
        if path == "/api/workspaces":
            try:
                self._ok(_api_workspaces({}))
            except Exception as exc:
                traceback.print_exc()
                self._err(500, str(exc))
            return
        if path == "/api/workspaces/current":
            try:
                self._ok(_api_workspace_current({}))
            except Exception as exc:
                traceback.print_exc()
                self._err(500, str(exc))
            return
        if path.startswith("/api/"):
            self._err(405, "use POST")
            return
        self._serve_static(path)

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        handler = ROUTES.get(path)
        if handler is None:
            self._err(404, "no such endpoint")
            return
        try:
            body = self._read_body()
        except Exception as exc:
            self._err(400, f"bad JSON body: {exc}")
            return
        try:
            self._ok(handler(body))
        except KeyError as exc:
            self._err(404, f"unknown id: {exc}")
        except ValueError as exc:
            self._err(400, str(exc))
        except ServiceUnavailable as exc:
            self._err(503, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._err(500, str(exc))

    # -- static ----------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        target = (APP_DIR / rel).resolve()
        if target.is_dir():
            target = target / "index.html"
        try:
            target.relative_to(APP_DIR.resolve())
        except ValueError:
            self._err(403, "forbidden")
            return
        if not target.is_file():
            self._err(404, "not found")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in (
            "application/javascript", "application/json",
        ):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


# --------------------------------------------------------------------------
# boot
# --------------------------------------------------------------------------


def build_engine() -> Any:
    """Compatibility helper for tests and legacy callers."""
    _ensure_runtime()
    if STATE_PATH.exists():
        try:
            return engine_mod.load(str(STATE_PATH))
        except Exception:
            traceback.print_exc()
    return engine_mod.Engine()


def _initialize_storage() -> tuple[Storage, Workspace, Any]:
    """Open SQLite, import legacy state once, and restore the active workspace."""
    _ensure_runtime()
    store = Storage(DB_PATH)
    workspaces = store.list_workspaces()
    if not workspaces:
        if STATE_PATH.exists():
            workspace = store.migrate_legacy_state(STATE_PATH)
        else:
            engine = engine_mod.Engine()
            workspace = store.create_workspace(
                "Untitled workspace",
                snapshot=engine.state(),
            )
    else:
        current_id = store.current_workspace_id()
        try:
            workspace = (
                store.load_workspace(current_id)
                if current_id is not None
                else workspaces[0]
            )
        except KeyError:
            workspace = workspaces[0]
    store.set_current_workspace(workspace.id)
    engine = engine_mod.Engine.from_state(workspace.snapshot)
    return store, workspace, engine


def create_runtime(*, access: str = "local") -> Any:
    """Create the one-port AviaryMCP/browser ASGI runtime."""
    from .http_runtime import create_runtime as build_runtime

    return build_runtime(access=access)


def serve(port: int = PORT, host: str = HOST) -> None:
    global ENGINE, STORE, WORKSPACE_ID
    _STOP.clear()
    STORE, workspace, ENGINE = _initialize_storage()
    WORKSPACE_ID = workspace.id
    with LOCK:
        _persist()

    raven_outbox = threading.Thread(
        target=_raven_outbox_loop,
        daemon=True,
        name="magpie-raven-outbox",
    )
    metabolism: threading.Thread | None = None
    if AUTO_CONNECTIONS:
        metabolism = threading.Thread(
            target=_metabolism_loop,
            daemon=True,
            name="magpie-metabolism",
        )
        metabolism.start()
    raven_outbox.start()
    # ``local`` access is loopback-only by SDK invariant: the browser field and
    # JSON plane carry no auth. A container binds beyond loopback, so it must
    # declare that posture explicitly rather than quietly widening the bind.
    runtime = create_runtime(access=os.environ.get("MAGPIE_ACCESS") or "local")
    print(
        f"magpie serving http://{host}:{port} "
        f"(app={APP_DIR}, mcp=/mcp, db={DB_PATH}, workspace={WORKSPACE_ID})"
    )
    try:
        runtime.run(
            transport="http",
            host=host,
            port=port,
            path="/mcp",
            stateless_http=True,
            show_banner=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        if metabolism is not None:
            metabolism.join(timeout=METABOLISM_PERIOD + 1)
        raven_outbox.join(timeout=RAVEN_OUTBOX_PERIOD + 1)
        with LOCK:
            _persist()
        if STORE is not None:
            STORE.close()
            STORE = None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="magpie.server")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default=HOST)
    args = ap.parse_args(argv)
    serve(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
