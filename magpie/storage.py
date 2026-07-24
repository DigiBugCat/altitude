"""SQLite persistence for Magpie workspaces and the shared idea bank.

The repository stores opaque JSON engine snapshots so the deterministic engine
can evolve independently from the database schema.  Cross-workspace concepts
(ideas, occurrences, retrieval events, memory exposures, and embeddings) are
stored relationally.

This module intentionally performs no inference and calls no embedding API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 3
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def new_id(prefix: str) -> str:
    """Return a stable opaque identifier generated without database state."""

    return f"{prefix}_{uuid.uuid4().hex}"


def normalize_idea_text(text: str) -> str:
    """Normalize text for exact (not semantic) idea deduplication."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _SPACE_RE.sub(" ", " ".join(_WORD_RE.findall(normalized))).strip()


def idea_fingerprint(text: str) -> str:
    normalized = normalize_idea_text(text)
    if not normalized:
        raise ValueError("idea text must contain searchable characters")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _decode(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    question: str
    snapshot: dict[str, Any]
    metadata: dict[str, Any]
    context_version: int
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Idea:
    id: str
    text: str
    fingerprint: str
    kind: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Event:
    id: str
    workspace_id: str | None
    kind: str
    payload: dict[str, Any]
    context_version: int | None
    status: str
    attempts: int
    created_at: float
    available_at: float
    claimed_at: float | None
    completed_at: float | None
    error: str | None


@dataclass(frozen=True)
class Exposure:
    id: str
    workspace_id: str
    idea_id: str
    status: str
    reason: str
    context_version: int
    first_shown_at: float
    last_shown_at: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Embedding:
    idea_id: str
    model: str
    version: str
    dimensions: int
    vector: bytes
    encoding: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class RavenProjection:
    """Workspace-local presentation state for one canonical Raven memory."""

    id: str
    workspace_id: str
    raven_memory_id: str
    local_ref: str | None
    section: str
    mass: float
    pinned: bool
    hidden: bool
    local_note: str
    local_status: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class RavenExposure:
    """A record that a canonical Raven memory was shown in a workspace."""

    id: str
    workspace_id: str
    raven_memory_id: str
    status: str
    reason: str
    context_version: int
    first_shown_at: float
    last_shown_at: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RavenRemember:
    """One durable outbound ``remember`` request to Raven."""

    id: str
    workspace_id: str | None
    dedupe_key: str
    payload: dict[str, Any]
    status: str
    attempts: int
    raven_memory_id: str | None
    created_at: float
    available_at: float
    claimed_at: float | None
    completed_at: float | None
    error: str | None


class Storage:
    """Thread-safe SQLite repository.

    One connection is shared under a re-entrant lock.  Every mutating method is
    transactional; callers can group several operations with ``transaction()``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = 5_000,
        now=time.time,
    ):
        self.path = os.fspath(path)
        self.now = now
        self._lock = threading.RLock()
        parent = os.path.dirname(os.path.abspath(self.path))
        if self.path != ":memory:":
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path,
            timeout=max(0, busy_timeout_ms) / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run operations atomically; nested calls participate via a savepoint."""

        with self._lock:
            nested = self._conn.in_transaction
            marker = f"sp_{uuid.uuid4().hex}" if nested else ""
            self._conn.execute(f"SAVEPOINT {marker}" if nested else "BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                if nested:
                    self._conn.execute(f"ROLLBACK TO {marker}")
                    self._conn.execute(f"RELEASE {marker}")
                else:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute(f"RELEASE {marker}" if nested else "COMMIT")

    def _migrate(self) -> None:
        with self._lock:
            db = self._conn
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version < 1:
                try:
                    db.executescript(
                        """
                    BEGIN IMMEDIATE;
                    CREATE TABLE workspaces (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        question TEXT NOT NULL DEFAULT '',
                        snapshot_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        context_version INTEGER NOT NULL DEFAULT 0
                            CHECK (context_version >= 0),
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE ideas (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        fingerprint TEXT NOT NULL UNIQUE,
                        kind TEXT NOT NULL DEFAULT 'claim',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE workspace_idea_occurrences (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                            ON DELETE CASCADE,
                        idea_id TEXT NOT NULL REFERENCES ideas(id)
                            ON DELETE RESTRICT,
                        local_ref TEXT,
                        source TEXT NOT NULL DEFAULT 'workspace',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX occurrences_workspace_idx
                        ON workspace_idea_occurrences(workspace_id, created_at);
                    CREATE INDEX occurrences_idea_idx
                        ON workspace_idea_occurrences(idea_id, created_at);

                    CREATE TABLE events (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT REFERENCES workspaces(id)
                            ON DELETE CASCADE,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        context_version INTEGER,
                        status TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending','processing','completed','failed')),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        available_at REAL NOT NULL,
                        claimed_at REAL,
                        completed_at REAL,
                        error TEXT
                    );
                    CREATE INDEX events_queue_idx
                        ON events(status, available_at, created_at);
                    CREATE INDEX events_workspace_idx
                        ON events(workspace_id, status, created_at);

                    CREATE TABLE memory_exposures (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                            ON DELETE CASCADE,
                        idea_id TEXT NOT NULL REFERENCES ideas(id)
                            ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        reason TEXT NOT NULL DEFAULT '',
                        context_version INTEGER NOT NULL,
                        first_shown_at REAL NOT NULL,
                        last_shown_at REAL NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(workspace_id, idea_id)
                    );

                    CREATE TABLE idea_embeddings (
                        idea_id TEXT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
                        model TEXT NOT NULL,
                        version TEXT NOT NULL,
                        dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                        vector BLOB NOT NULL,
                        encoding TEXT NOT NULL DEFAULT 'float32-le',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (idea_id, model, version)
                    );
                    CREATE INDEX embeddings_model_idx
                        ON idea_embeddings(model, version, dimensions);

                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                    )
                except BaseException:
                    if db.in_transaction:
                        db.execute("ROLLBACK")
                    raise
                version = 1
            if version < 2:
                with self.transaction() as tx:
                    tx.execute(
                        """CREATE TABLE IF NOT EXISTS app_settings (
                               key TEXT PRIMARY KEY,
                               value TEXT NOT NULL
                           )"""
                    )
                    tx.execute(
                        """INSERT OR IGNORE INTO app_settings(key,value)
                           VALUES ('bank_revision','0')"""
                    )
                    tx.execute("PRAGMA user_version = 2")
                version = 2
            if version < 3:
                with self.transaction() as tx:
                    tx.execute(
                        """CREATE TABLE IF NOT EXISTS raven_projections (
                               id TEXT PRIMARY KEY,
                               workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                                   ON DELETE CASCADE,
                               raven_memory_id TEXT NOT NULL,
                               local_ref TEXT,
                               section TEXT NOT NULL DEFAULT 'field',
                               mass REAL NOT NULL DEFAULT 1
                                   CHECK (mass >= 0),
                               pinned INTEGER NOT NULL DEFAULT 0
                                   CHECK (pinned IN (0,1)),
                               hidden INTEGER NOT NULL DEFAULT 0
                                   CHECK (hidden IN (0,1)),
                               local_note TEXT NOT NULL DEFAULT '',
                               local_status TEXT NOT NULL DEFAULT '',
                               metadata_json TEXT NOT NULL DEFAULT '{}',
                               created_at REAL NOT NULL,
                               updated_at REAL NOT NULL,
                               UNIQUE(workspace_id,raven_memory_id)
                           )"""
                    )
                    tx.execute(
                        """CREATE INDEX IF NOT EXISTS raven_projections_workspace_idx
                           ON raven_projections(
                               workspace_id,hidden,section,pinned,updated_at
                           )"""
                    )
                    tx.execute(
                        """CREATE TABLE IF NOT EXISTS raven_exposures (
                               id TEXT PRIMARY KEY,
                               workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                                   ON DELETE CASCADE,
                               raven_memory_id TEXT NOT NULL,
                               status TEXT NOT NULL DEFAULT 'suggested',
                               reason TEXT NOT NULL DEFAULT '',
                               context_version INTEGER NOT NULL DEFAULT 0
                                   CHECK (context_version >= 0),
                               first_shown_at REAL NOT NULL,
                               last_shown_at REAL NOT NULL,
                               metadata_json TEXT NOT NULL DEFAULT '{}',
                               UNIQUE(workspace_id,raven_memory_id)
                           )"""
                    )
                    tx.execute(
                        """CREATE INDEX IF NOT EXISTS raven_exposures_workspace_idx
                           ON raven_exposures(workspace_id,last_shown_at)"""
                    )
                    tx.execute(
                        """CREATE TABLE IF NOT EXISTS raven_remember_outbox (
                               id TEXT PRIMARY KEY,
                               workspace_id TEXT REFERENCES workspaces(id)
                                   ON DELETE SET NULL,
                               dedupe_key TEXT NOT NULL UNIQUE,
                               payload_json TEXT NOT NULL,
                               status TEXT NOT NULL DEFAULT 'pending'
                                   CHECK (status IN (
                                       'pending','processing','completed','failed'
                                   )),
                               attempts INTEGER NOT NULL DEFAULT 0
                                   CHECK (attempts >= 0),
                               raven_memory_id TEXT,
                               created_at REAL NOT NULL,
                               available_at REAL NOT NULL,
                               claimed_at REAL,
                               completed_at REAL,
                               error TEXT
                           )"""
                    )
                    tx.execute(
                        """CREATE INDEX IF NOT EXISTS raven_outbox_queue_idx
                           ON raven_remember_outbox(
                               status,available_at,created_at
                           )"""
                    )
                    tx.execute(
                        """CREATE INDEX IF NOT EXISTS raven_outbox_workspace_idx
                           ON raven_remember_outbox(
                               workspace_id,status,created_at
                           )"""
                    )
                    tx.execute("PRAGMA user_version = 3")

    # -- workspaces -----------------------------------------------------

    def create_workspace(
        self,
        name: str,
        *,
        question: str = "",
        snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> Workspace:
        workspace_id = workspace_id or new_id("ws")
        ts = float(self.now())
        snap = dict(snapshot or {})
        if question and "question" not in snap:
            snap["question"] = question
        with self.transaction() as db:
            db.execute(
                """INSERT INTO workspaces
                   (id,name,question,snapshot_json,metadata_json,context_version,
                    created_at,updated_at) VALUES (?,?,?,?,?,0,?,?)""",
                (workspace_id, str(name), str(question), _json(snap),
                 _json(metadata or {}), ts, ts),
            )
        return self.load_workspace(workspace_id)

    def list_workspaces(self) -> list[Workspace]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM workspaces ORDER BY updated_at DESC, id"
            ).fetchall()
        return [self._workspace(row) for row in rows]

    def current_workspace_id(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM app_settings WHERE key='current_workspace_id'"
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_current_workspace(self, workspace_id: str) -> Workspace:
        workspace = self.load_workspace(workspace_id)
        with self.transaction() as db:
            db.execute(
                """INSERT INTO app_settings(key,value)
                   VALUES ('current_workspace_id',?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (workspace_id,),
            )
        return workspace

    def bank_revision(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM app_settings WHERE key='bank_revision'"
            ).fetchone()
        return 0 if row is None else int(row["value"])

    def load_workspace(self, workspace_id: str) -> Workspace:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        if row is None:
            raise KeyError(workspace_id)
        return self._workspace(row)

    def save_workspace(
        self,
        workspace_id: str,
        snapshot: dict[str, Any],
        *,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_context_version: int | None = None,
        increment_context: bool = True,
    ) -> Workspace:
        """Save a snapshot with optional optimistic concurrency protection."""

        ts = float(self.now())
        assignments = ["snapshot_json=?", "updated_at=?"]
        params: list[Any] = [_json(snapshot), ts]
        if question is not None:
            assignments.append("question=?")
            params.append(str(question))
        if metadata is not None:
            assignments.append("metadata_json=?")
            params.append(_json(metadata))
        if increment_context:
            assignments.append("context_version=context_version+1")
        where = "id=?"
        params.append(workspace_id)
        if expected_context_version is not None:
            where += " AND context_version=?"
            params.append(int(expected_context_version))
        with self.transaction() as db:
            cur = db.execute(
                f"UPDATE workspaces SET {','.join(assignments)} WHERE {where}", params
            )
            if cur.rowcount != 1:
                exists = db.execute(
                    "SELECT 1 FROM workspaces WHERE id=?", (workspace_id,)
                ).fetchone()
                if exists:
                    raise RuntimeError("workspace context version conflict")
                raise KeyError(workspace_id)
        return self.load_workspace(workspace_id)

    # -- idea bank and occurrences -------------------------------------

    def upsert_idea(
        self,
        text: str,
        *,
        kind: str = "claim",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Idea, bool]:
        """Return ``(canonical_idea, created)`` using normalized exact identity."""

        clean_text = str(text or "").strip()
        fingerprint = idea_fingerprint(clean_text)
        ts = float(self.now())
        idea_id = new_id("idea")
        with self.transaction() as db:
            cur = db.execute(
                """INSERT INTO ideas
                   (id,text,fingerprint,kind,metadata_json,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(fingerprint) DO NOTHING""",
                (idea_id, clean_text, fingerprint, str(kind), _json(metadata or {}),
                 ts, ts),
            )
            created = cur.rowcount == 1
            if created:
                db.execute(
                    """UPDATE app_settings
                       SET value=CAST(value AS INTEGER)+1
                       WHERE key='bank_revision'"""
                )
            row = db.execute(
                "SELECT * FROM ideas WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        return self._idea(row), created

    def link_idea(
        self,
        workspace_id: str,
        idea_id: str,
        *,
        local_ref: str | None = None,
        source: str = "workspace",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        occurrence_id = new_id("occ")
        with self.transaction() as db:
            db.execute(
                """INSERT INTO workspace_idea_occurrences
                   (id,workspace_id,idea_id,local_ref,source,metadata_json,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (occurrence_id, workspace_id, idea_id, local_ref, str(source),
                 _json(metadata or {}), float(self.now())),
            )
        return occurrence_id

    def list_workspace_ideas(self, workspace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT o.id occurrence_id,o.local_ref,o.source,
                          o.metadata_json occurrence_metadata,o.created_at appeared_at,
                          i.*
                   FROM workspace_idea_occurrences o
                   JOIN ideas i ON i.id=o.idea_id
                   WHERE o.workspace_id=?
                   ORDER BY o.created_at,o.id""",
                (workspace_id,),
            ).fetchall()
        return [
            {
                "occurrence_id": row["occurrence_id"],
                "local_ref": row["local_ref"],
                "source": row["source"],
                "occurrence_metadata": _decode(row["occurrence_metadata"], {}),
                "appeared_at": row["appeared_at"],
                "idea": self._idea(row),
            }
            for row in rows
        ]

    # -- durable event queue -------------------------------------------

    def append_event(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        workspace_id: str | None = None,
        context_version: int | None = None,
        available_at: float | None = None,
    ) -> Event:
        event_id = new_id("evt")
        ts = float(self.now())
        with self.transaction() as db:
            db.execute(
                """INSERT INTO events
                   (id,workspace_id,kind,payload_json,context_version,status,
                    attempts,created_at,available_at)
                   VALUES (?,?,?,?,?,'pending',0,?,?)""",
                (event_id, workspace_id, str(kind), _json(payload or {}),
                 context_version, ts, ts if available_at is None else available_at),
            )
            row = db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._event(row)

    def poll_events(
        self,
        *,
        limit: int = 10,
        workspace_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> list[Event]:
        """Atomically claim events, reclaiming work abandoned after its lease."""

        if limit < 1:
            return []
        if lease_seconds <= 0:
            raise ValueError("event lease must be positive")
        now = float(self.now())
        condition = "status='pending' AND available_at<=?"
        params: list[Any] = [now]
        if workspace_id is not None:
            condition += " AND workspace_id=?"
            params.append(workspace_id)
        params.append(int(limit))
        with self.transaction() as db:
            db.execute(
                """UPDATE events
                   SET status='pending',claimed_at=NULL,
                       available_at=CASE WHEN available_at>? THEN ? ELSE available_at END,
                       error=CASE WHEN error IS NULL OR error=''
                                  THEN 'claim lease expired' ELSE error END
                   WHERE status='processing' AND claimed_at IS NOT NULL
                     AND claimed_at<=?""",
                (now, now, now - float(lease_seconds)),
            )
            rows = db.execute(
                f"SELECT id FROM events WHERE {condition} "
                "ORDER BY created_at,id LIMIT ?",
                params,
            ).fetchall()
            ids = [row["id"] for row in rows]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            db.execute(
                f"""UPDATE events SET status='processing',claimed_at=?,
                    attempts=attempts+1 WHERE id IN ({placeholders})""",
                [now, *ids],
            )
            claimed = db.execute(
                f"SELECT * FROM events WHERE id IN ({placeholders}) "
                "ORDER BY created_at,id",
                ids,
            ).fetchall()
        return [self._event(row) for row in claimed]

    def mark_event(
        self,
        event_id: str,
        status: str,
        *,
        error: str | None = None,
        available_at: float | None = None,
    ) -> Event:
        """Mark an event completed/failed, or return it to the pending queue."""

        if status not in ("pending", "completed", "failed"):
            raise ValueError("event status must be pending, completed, or failed")
        now = float(self.now())
        completed_at = now if status in ("completed", "failed") else None
        with self.transaction() as db:
            cur = db.execute(
                """UPDATE events SET status=?,error=?,completed_at=?,
                   claimed_at=CASE WHEN ?='pending' THEN NULL ELSE claimed_at END,
                   available_at=CASE WHEN ?='pending' THEN ? ELSE available_at END
                   WHERE id=?""",
                (status, error, completed_at, status, status,
                 now if available_at is None else available_at, event_id),
            )
            if cur.rowcount != 1:
                raise KeyError(event_id)
            row = db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._event(row)

    # -- memory shelf state --------------------------------------------

    def upsert_exposure(
        self,
        workspace_id: str,
        idea_id: str,
        *,
        status: str = "suggested",
        reason: str = "",
        context_version: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Exposure:
        ts = float(self.now())
        exposure_id = new_id("exp")
        with self.transaction() as db:
            db.execute(
                """INSERT INTO memory_exposures
                   (id,workspace_id,idea_id,status,reason,context_version,
                    first_shown_at,last_shown_at,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(workspace_id,idea_id) DO UPDATE SET
                     status=excluded.status,
                     reason=excluded.reason,
                     context_version=excluded.context_version,
                     last_shown_at=excluded.last_shown_at,
                     metadata_json=excluded.metadata_json""",
                (exposure_id, workspace_id, idea_id, str(status), str(reason),
                 int(context_version), ts, ts, _json(metadata or {})),
            )
            row = db.execute(
                "SELECT * FROM memory_exposures WHERE workspace_id=? AND idea_id=?",
                (workspace_id, idea_id),
            ).fetchone()
        return self._exposure(row)

    def list_exposures(self, workspace_id: str) -> list[Exposure]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memory_exposures WHERE workspace_id=?
                   ORDER BY last_shown_at DESC,id""",
                (workspace_id,),
            ).fetchall()
        return [self._exposure(row) for row in rows]

    # -- Raven workspace projections ----------------------------------

    def upsert_raven_projection(
        self,
        workspace_id: str,
        raven_memory_id: str,
        *,
        local_ref: str | None = None,
        section: str = "field",
        mass: float = 1.0,
        pinned: bool = False,
        hidden: bool = False,
        local_note: str = "",
        local_status: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RavenProjection:
        """Create or replace the local presentation of a Raven memory.

        Raven owns the memory content and epistemic state.  This row contains
        only workspace-specific layout and workflow state.
        """

        memory_id = str(raven_memory_id or "").strip()
        if not memory_id:
            raise ValueError("raven memory id is required")
        mass_value = float(mass)
        if mass_value < 0:
            raise ValueError("projection mass must be non-negative")
        ts = float(self.now())
        projection_id = new_id("rproj")
        with self.transaction() as db:
            db.execute(
                """INSERT INTO raven_projections
                   (id,workspace_id,raven_memory_id,local_ref,section,mass,
                    pinned,hidden,local_note,local_status,metadata_json,
                    created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(workspace_id,raven_memory_id) DO UPDATE SET
                     local_ref=excluded.local_ref,
                     section=excluded.section,
                     mass=excluded.mass,
                     pinned=excluded.pinned,
                     hidden=excluded.hidden,
                     local_note=excluded.local_note,
                     local_status=excluded.local_status,
                     metadata_json=excluded.metadata_json,
                     updated_at=excluded.updated_at""",
                (
                    projection_id, workspace_id, memory_id, local_ref,
                    str(section), mass_value, int(bool(pinned)), int(bool(hidden)),
                    str(local_note), str(local_status), _json(metadata or {}), ts, ts,
                ),
            )
            row = db.execute(
                """SELECT * FROM raven_projections
                   WHERE workspace_id=? AND raven_memory_id=?""",
                (workspace_id, memory_id),
            ).fetchone()
        return self._raven_projection(row)

    def get_raven_projection(
        self, workspace_id: str, raven_memory_id: str
    ) -> RavenProjection | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM raven_projections
                   WHERE workspace_id=? AND raven_memory_id=?""",
                (workspace_id, raven_memory_id),
            ).fetchone()
        return None if row is None else self._raven_projection(row)

    def list_raven_projections(
        self,
        workspace_id: str,
        *,
        include_hidden: bool = True,
        section: str | None = None,
    ) -> list[RavenProjection]:
        conditions = ["workspace_id=?"]
        params: list[Any] = [workspace_id]
        if not include_hidden:
            conditions.append("hidden=0")
        if section is not None:
            conditions.append("section=?")
            params.append(str(section))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM raven_projections
                    WHERE {' AND '.join(conditions)}
                    ORDER BY pinned DESC,updated_at DESC,id""",
                params,
            ).fetchall()
        return [self._raven_projection(row) for row in rows]

    def upsert_raven_exposure(
        self,
        workspace_id: str,
        raven_memory_id: str,
        *,
        status: str = "suggested",
        reason: str = "",
        context_version: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> RavenExposure:
        """Record local shelf exposure without sending feedback to Raven."""

        memory_id = str(raven_memory_id or "").strip()
        if not memory_id:
            raise ValueError("raven memory id is required")
        if context_version < 0:
            raise ValueError("context version must be non-negative")
        ts = float(self.now())
        exposure_id = new_id("rexp")
        with self.transaction() as db:
            db.execute(
                """INSERT INTO raven_exposures
                   (id,workspace_id,raven_memory_id,status,reason,context_version,
                    first_shown_at,last_shown_at,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(workspace_id,raven_memory_id) DO UPDATE SET
                     status=excluded.status,
                     reason=excluded.reason,
                     context_version=excluded.context_version,
                     last_shown_at=excluded.last_shown_at,
                     metadata_json=excluded.metadata_json""",
                (
                    exposure_id, workspace_id, memory_id, str(status), str(reason),
                    int(context_version), ts, ts, _json(metadata or {}),
                ),
            )
            row = db.execute(
                """SELECT * FROM raven_exposures
                   WHERE workspace_id=? AND raven_memory_id=?""",
                (workspace_id, memory_id),
            ).fetchone()
        return self._raven_exposure(row)

    def list_raven_exposures(self, workspace_id: str) -> list[RavenExposure]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM raven_exposures WHERE workspace_id=?
                   ORDER BY last_shown_at DESC,id""",
                (workspace_id,),
            ).fetchall()
        return [self._raven_exposure(row) for row in rows]

    # -- Raven remember outbox -----------------------------------------

    def enqueue_raven_remember(
        self,
        content: str,
        *,
        workspace_id: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        episode_id: str | None = None,
        hints: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        available_at: float | None = None,
    ) -> RavenRemember:
        """Durably queue one Raven ``remember`` call.

        ``dedupe_key`` prevents duplicate local enqueue operations. Raven's
        current API does not accept an idempotency key, so delivery is
        at-least-once if a process dies after the remote write but before the
        local completion commit.
        """

        clean_content = str(content or "").strip()
        if not clean_content:
            raise ValueError("remember content is required")
        key = str(dedupe_key or new_id("remember"))
        payload: dict[str, Any] = {"content": clean_content}
        if source is not None:
            payload["source"] = str(source)
        if tags is not None:
            payload["tags"] = [str(tag) for tag in tags]
        if episode_id is not None:
            payload["episode_id"] = str(episode_id)
        if hints is not None:
            payload["hints"] = dict(hints)
        ts = float(self.now())
        outbox_id = new_id("rout")
        with self.transaction() as db:
            db.execute(
                """INSERT INTO raven_remember_outbox
                   (id,workspace_id,dedupe_key,payload_json,status,attempts,
                    created_at,available_at)
                   VALUES (?,?,?,?, 'pending',0,?,?)
                   ON CONFLICT(dedupe_key) DO NOTHING""",
                (
                    outbox_id, workspace_id, key, _json(payload), ts,
                    ts if available_at is None else float(available_at),
                ),
            )
            row = db.execute(
                "SELECT * FROM raven_remember_outbox WHERE dedupe_key=?", (key,)
            ).fetchone()
        return self._raven_remember(row)

    def claim_raven_remembers(
        self,
        *,
        limit: int = 10,
        lease_seconds: float = 300.0,
    ) -> list[RavenRemember]:
        """Atomically claim due outbox items and reclaim expired leases."""

        if limit < 1:
            return []
        if lease_seconds <= 0:
            raise ValueError("outbox lease must be positive")
        now = float(self.now())
        with self.transaction() as db:
            db.execute(
                """UPDATE raven_remember_outbox
                   SET status='pending',claimed_at=NULL,
                       available_at=CASE
                           WHEN available_at>? THEN ? ELSE available_at END,
                       error=CASE WHEN error IS NULL OR error=''
                           THEN 'claim lease expired' ELSE error END
                   WHERE status='processing' AND claimed_at IS NOT NULL
                     AND claimed_at<=?""",
                (now, now, now - float(lease_seconds)),
            )
            rows = db.execute(
                """SELECT id FROM raven_remember_outbox
                   WHERE status='pending' AND available_at<=?
                   ORDER BY attempts,available_at,created_at,id LIMIT ?""",
                (now, int(limit)),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            db.execute(
                f"""UPDATE raven_remember_outbox
                    SET status='processing',claimed_at=?,attempts=attempts+1
                    WHERE id IN ({placeholders})""",
                [now, *ids],
            )
            claimed = db.execute(
                f"""SELECT * FROM raven_remember_outbox
                    WHERE id IN ({placeholders}) ORDER BY created_at,id""",
                ids,
            ).fetchall()
        return [self._raven_remember(row) for row in claimed]

    def mark_raven_remember(
        self,
        outbox_id: str,
        status: str,
        *,
        raven_memory_id: str | None = None,
        error: str | None = None,
        available_at: float | None = None,
    ) -> RavenRemember:
        if status not in ("pending", "completed", "failed"):
            raise ValueError(
                "outbox status must be pending, completed, or failed"
            )
        if status == "completed" and not str(raven_memory_id or "").strip():
            raise ValueError("completed remember requires a raven memory id")
        now = float(self.now())
        completed_at = now if status in ("completed", "failed") else None
        with self.transaction() as db:
            cur = db.execute(
                """UPDATE raven_remember_outbox
                   SET status=?,raven_memory_id=COALESCE(?,raven_memory_id),
                       error=?,completed_at=?,
                       claimed_at=CASE WHEN ?='pending' THEN NULL ELSE claimed_at END,
                       available_at=CASE WHEN ?='pending' THEN ? ELSE available_at END
                   WHERE id=?""",
                (
                    status, raven_memory_id, error, completed_at, status, status,
                    now if available_at is None else float(available_at), outbox_id,
                ),
            )
            if cur.rowcount != 1:
                raise KeyError(outbox_id)
            row = db.execute(
                "SELECT * FROM raven_remember_outbox WHERE id=?", (outbox_id,)
            ).fetchone()
        return self._raven_remember(row)

    def completed_raven_memory_id_for_local_ref(
        self,
        workspace_id: str,
        local_ref: str,
    ) -> str | None:
        """Return the canonical Raven id recorded for one local card write.

        A Raven projection is unique by remote memory id, so canonical Raven
        deduplication may make several local cards share one projection. The
        completed outbox row remains unique by local card dedupe key and is
        therefore the durable source for local-card ancestry.
        """

        dedupe_key = f"magpie-card:{workspace_id}:{local_ref}"
        with self._lock:
            row = self._conn.execute(
                """SELECT raven_memory_id FROM raven_remember_outbox
                   WHERE workspace_id=? AND dedupe_key=? AND status='completed'
                     AND raven_memory_id IS NOT NULL
                   LIMIT 1""",
                (workspace_id, dedupe_key),
            ).fetchone()
        if row is None:
            return None
        memory_id = str(row["raven_memory_id"] or "").strip()
        return memory_id or None

    # -- embeddings -----------------------------------------------------

    def put_embedding(
        self,
        idea_id: str,
        *,
        model: str,
        version: str,
        dimensions: int,
        vector: bytes,
        encoding: str = "float32-le",
        metadata: dict[str, Any] | None = None,
    ) -> Embedding:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        raw = bytes(vector)
        if not raw:
            raise ValueError("embedding vector must not be empty")
        if not model or not version:
            raise ValueError("embedding model and version are required")
        if encoding in ("float32-le", "float32-be") and len(raw) != dimensions * 4:
            raise ValueError(
                f"{encoding} embedding must contain exactly {dimensions * 4} bytes"
            )
        ts = float(self.now())
        with self.transaction() as db:
            existing = db.execute(
                """SELECT dimensions,encoding FROM idea_embeddings
                   WHERE model=? AND version=? AND idea_id<>? LIMIT 1""",
                (model, version, idea_id),
            ).fetchone()
            if existing is not None and (
                int(existing["dimensions"]) != int(dimensions)
                or str(existing["encoding"]) != str(encoding)
            ):
                raise ValueError(
                    "embedding model/version must use one dimension and encoding"
                )
            db.execute(
                """INSERT INTO idea_embeddings
                   (idea_id,model,version,dimensions,vector,encoding,metadata_json,
                    created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(idea_id,model,version) DO UPDATE SET
                     dimensions=excluded.dimensions,
                     vector=excluded.vector,
                     encoding=excluded.encoding,
                     metadata_json=excluded.metadata_json,
                     updated_at=excluded.updated_at""",
                (idea_id, str(model), str(version), int(dimensions),
                 sqlite3.Binary(raw), str(encoding), _json(metadata or {}), ts, ts),
            )
            row = db.execute(
                """SELECT * FROM idea_embeddings
                   WHERE idea_id=? AND model=? AND version=?""",
                (idea_id, model, version),
            ).fetchone()
        return self._embedding(row)

    def get_embedding(
        self, idea_id: str, *, model: str, version: str
    ) -> Embedding | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM idea_embeddings
                   WHERE idea_id=? AND model=? AND version=?""",
                (idea_id, model, version),
            ).fetchone()
        return None if row is None else self._embedding(row)

    # -- legacy import --------------------------------------------------

    def migrate_legacy_state(
        self,
        path: str | os.PathLike[str],
        *,
        workspace_name: str = "Imported workspace",
        workspace_id: str | None = None,
    ) -> Workspace:
        """Import a legacy ``state.json`` without modifying or deleting it."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as fh:
            snapshot = json.load(fh)
        if not isinstance(snapshot, dict):
            raise ValueError("legacy state must be a JSON object")
        question = str(snapshot.get("question", ""))
        with self.transaction():
            workspace = self.create_workspace(
                workspace_name,
                question=question,
                snapshot=snapshot,
                metadata={"imported_from": str(source.resolve())},
                workspace_id=workspace_id,
            )
            for card in snapshot.get("cards", []):
                if not isinstance(card, dict) or not str(card.get("text", "")).strip():
                    continue
                idea, _created = self.upsert_idea(
                    str(card["text"]),
                    kind=str(card.get("kind", "claim")),
                    metadata={"legacy": True},
                )
                self.link_idea(
                    workspace.id,
                    idea.id,
                    local_ref=str(card.get("id", "")) or None,
                    source="legacy",
                    metadata={
                        "state": card.get("state", "open"),
                        "archived": bool(card.get("archived", False)),
                    },
                )
        return self.load_workspace(workspace.id)

    # -- row conversion -------------------------------------------------

    @staticmethod
    def _workspace(row: sqlite3.Row) -> Workspace:
        return Workspace(
            id=row["id"], name=row["name"], question=row["question"],
            snapshot=_decode(row["snapshot_json"], {}),
            metadata=_decode(row["metadata_json"], {}),
            context_version=row["context_version"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _idea(row: sqlite3.Row) -> Idea:
        return Idea(
            id=row["id"], text=row["text"], fingerprint=row["fingerprint"],
            kind=row["kind"], metadata=_decode(row["metadata_json"], {}),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"], workspace_id=row["workspace_id"], kind=row["kind"],
            payload=_decode(row["payload_json"], {}),
            context_version=row["context_version"], status=row["status"],
            attempts=row["attempts"], created_at=row["created_at"],
            available_at=row["available_at"], claimed_at=row["claimed_at"],
            completed_at=row["completed_at"], error=row["error"],
        )

    @staticmethod
    def _exposure(row: sqlite3.Row) -> Exposure:
        return Exposure(
            id=row["id"], workspace_id=row["workspace_id"], idea_id=row["idea_id"],
            status=row["status"], reason=row["reason"],
            context_version=row["context_version"],
            first_shown_at=row["first_shown_at"],
            last_shown_at=row["last_shown_at"],
            metadata=_decode(row["metadata_json"], {}),
        )

    @staticmethod
    def _raven_projection(row: sqlite3.Row) -> RavenProjection:
        return RavenProjection(
            id=row["id"], workspace_id=row["workspace_id"],
            raven_memory_id=row["raven_memory_id"], local_ref=row["local_ref"],
            section=row["section"], mass=float(row["mass"]),
            pinned=bool(row["pinned"]), hidden=bool(row["hidden"]),
            local_note=row["local_note"], local_status=row["local_status"],
            metadata=_decode(row["metadata_json"], {}),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _raven_exposure(row: sqlite3.Row) -> RavenExposure:
        return RavenExposure(
            id=row["id"], workspace_id=row["workspace_id"],
            raven_memory_id=row["raven_memory_id"], status=row["status"],
            reason=row["reason"], context_version=row["context_version"],
            first_shown_at=row["first_shown_at"],
            last_shown_at=row["last_shown_at"],
            metadata=_decode(row["metadata_json"], {}),
        )

    @staticmethod
    def _raven_remember(row: sqlite3.Row) -> RavenRemember:
        return RavenRemember(
            id=row["id"], workspace_id=row["workspace_id"],
            dedupe_key=row["dedupe_key"],
            payload=_decode(row["payload_json"], {}),
            status=row["status"], attempts=row["attempts"],
            raven_memory_id=row["raven_memory_id"],
            created_at=row["created_at"], available_at=row["available_at"],
            claimed_at=row["claimed_at"], completed_at=row["completed_at"],
            error=row["error"],
        )

    @staticmethod
    def _embedding(row: sqlite3.Row) -> Embedding:
        return Embedding(
            idea_id=row["idea_id"], model=row["model"], version=row["version"],
            dimensions=row["dimensions"], vector=bytes(row["vector"]),
            encoding=row["encoding"], metadata=_decode(row["metadata_json"], {}),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
