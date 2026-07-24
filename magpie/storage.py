"""SQLite persistence for Altitude workspaces and the shared idea bank.

The repository stores opaque JSON engine snapshots so the deterministic engine
can evolve independently from the database schema.  Cross-workspace concepts
(ideas, occurrences, retrieval events, memory exposures, and embeddings) are
stored relationally.

Schema v4 adds the position layer (SPEC-ALTITUDE §1.2, §5).  **The snapshot
remains authoritative; the position tables are indices.**  ``sync_positions``
projects a snapshot into ``positions`` / ``position_supports`` /
``occupant_revisions`` so that durable ids, load-bearing edges, occupant
revision history, and ``last_grounded_at`` are queryable without deserializing
every workspace — but nothing reads structure back out of them to rebuild an
engine.  That one-way rule is what keeps the tables from becoming a second
source of truth that could drift from the floor (§1.5).

Two derived quantities are conspicuously absent from the schema: altitude and
frame support.  Both are computed from the floor on every read (§1.5) and
storing either would create exactly the drift the law forbids.

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

SCHEMA_VERSION = 4
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# §4 — typed export classes.  Eligibility to reach shared memory is an
# inspectable property of each queued write, not a condition buried in a
# function.  A write with no class is never queued at all; these three are the
# whole allowlist.
EXPORT_CLASSES = ("human_root", "human_curated_frame", "settled_claim")

# §1.2 mirrors of the engine vocabulary, kept here so the index can constrain
# its own columns without importing the engine (storage stays inference-free).
FLOOR_KINDS = ("claim", "frame")
POSITION_STATUSES = ("live", "folded", "vacated", "retired")
POSITION_ORIGINS = ("human", "click", "derivation", "recall")


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
    export_class: str = "human_root"


@dataclass(frozen=True)
class PositionRow:
    """The durable index row for one position (§1.2).

    Altitude and frame support are deliberately absent: both are derived from
    the floor on every read (§1.5), and a stored copy could drift.
    """

    workspace_id: str
    position_id: str
    floor_kind: str
    origin: str
    status: str
    folded_under: str | None
    external: bool
    pinned_by_human: bool
    last_grounded_at: float | None
    occupant_text: str
    occupant_fingerprint: str
    artifact_type: str
    support_state: str | None      # claims only; None for frames (§1.1)
    receipt: str | None            # claims only
    supports: list[str]            # load-bearing edges, one floor DOWN
    provenance: list[str]          # legacy `parents` — NEVER support edges (§5)
    lineage: list[str]
    confirmed_by: str | None
    confirmed_at: float | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class OccupantRevision:
    """One entry of a position's occupant history (§1.2).

    Rephrasing never touches the structure; this table is where the wording that
    was replaced goes, so the position keeps its history without keeping it in
    the live occupant.
    """

    id: str
    workspace_id: str
    position_id: str
    revision: int
    text: str
    fingerprint: str
    relation: str
    foot: str
    recorded_at: float


@dataclass(frozen=True)
class Suppression:
    """A locally suppressed Raven memory id (§4).

    Reversible and local: it neither deletes the memory nor mutates Raven's
    global epistemic state.
    """

    raven_memory_id: str
    reason: str
    workspace_id: str | None
    created_at: float


@dataclass(frozen=True)
class Dismissal:
    """A durable workspace-local dismissal by memory id (§4).

    A human-dismissed exposure never resurfaces in that workspace, regardless of
    future recall scoring.
    """

    workspace_id: str
    raven_memory_id: str
    reason: str
    created_at: float


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
                version = 3
            if version < 4:
                self._migrate_v4()

    def _migrate_v4(self) -> None:
        """SPEC §5 — schema v4: the position layer, plus the §4 export gate.

        Additive and conservative.  Existing outbox rows are backfilled to
        ``human_root`` rather than being invented into frames: a legacy write
        was a card the old ``_bank_card`` sent unconditionally, and calling it a
        curated frame would fabricate the human confirmation §5 requires.
        """

        with self.transaction() as tx:
            tx.execute(
                """CREATE TABLE IF NOT EXISTS positions (
                       workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                           ON DELETE CASCADE,
                       position_id TEXT NOT NULL,
                       floor_kind TEXT NOT NULL DEFAULT 'claim'
                           CHECK (floor_kind IN ('claim','frame')),
                       origin TEXT NOT NULL DEFAULT 'human'
                           CHECK (origin IN
                               ('human','click','derivation','recall')),
                       status TEXT NOT NULL DEFAULT 'live'
                           CHECK (status IN
                               ('live','folded','vacated','retired')),
                       folded_under TEXT,
                       external INTEGER NOT NULL DEFAULT 0
                           CHECK (external IN (0,1)),
                       pinned_by_human INTEGER NOT NULL DEFAULT 0
                           CHECK (pinned_by_human IN (0,1)),
                       last_grounded_at REAL,
                       occupant_text TEXT NOT NULL DEFAULT '',
                       occupant_fingerprint TEXT NOT NULL DEFAULT '',
                       artifact_type TEXT NOT NULL DEFAULT 'claim',
                       -- Claims only.  A frame stores NULL in both columns:
                       -- frame support is computed from the floor (§1.5) and a
                       -- frame can never carry a receipt at all (§1.1).
                       support_state TEXT,
                       receipt TEXT,
                       provenance_json TEXT NOT NULL DEFAULT '[]',
                       lineage_json TEXT NOT NULL DEFAULT '[]',
                       confirmed_by TEXT,
                       confirmed_at REAL,
                       created_at REAL NOT NULL,
                       updated_at REAL NOT NULL,
                       PRIMARY KEY (workspace_id, position_id),
                       CHECK (floor_kind <> 'frame'
                              OR (support_state IS NULL AND receipt IS NULL))
                   )"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS positions_workspace_idx
                   ON positions(workspace_id,status,floor_kind)"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS positions_stale_idx
                   ON positions(workspace_id,last_grounded_at)"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS positions_fingerprint_idx
                   ON positions(workspace_id,occupant_fingerprint)"""
            )
            # Support edges live in their own table so the load-bearing
            # structure is queryable in both directions.  `provenance` stays a
            # JSON column on the position: §5 is explicit that legacy `parents`
            # are NOT support edges, and giving them the same shape here would
            # invite exactly the confusion the backfill rule forbids.
            tx.execute(
                """CREATE TABLE IF NOT EXISTS position_supports (
                       workspace_id TEXT NOT NULL,
                       position_id TEXT NOT NULL,
                       supports_id TEXT NOT NULL,
                       ordinal INTEGER NOT NULL DEFAULT 0,
                       PRIMARY KEY (workspace_id, position_id, supports_id),
                       FOREIGN KEY (workspace_id, position_id)
                           REFERENCES positions(workspace_id, position_id)
                           ON DELETE CASCADE
                   )"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS position_supports_floor_idx
                   ON position_supports(workspace_id,supports_id)"""
            )
            tx.execute(
                """CREATE TABLE IF NOT EXISTS occupant_revisions (
                       id TEXT PRIMARY KEY,
                       workspace_id TEXT NOT NULL,
                       position_id TEXT NOT NULL,
                       revision INTEGER NOT NULL CHECK (revision >= 0),
                       text TEXT NOT NULL,
                       fingerprint TEXT NOT NULL DEFAULT '',
                       relation TEXT NOT NULL DEFAULT 'refinement',
                       foot TEXT NOT NULL DEFAULT '',
                       recorded_at REAL NOT NULL,
                       UNIQUE (workspace_id, position_id, revision)
                   )"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS occupant_revisions_position_idx
                   ON occupant_revisions(workspace_id,position_id,revision)"""
            )
            # §2.3 never-retry ledger.  Keyed on position ids so rewording an
            # occupant can neither resurrect nor silently reopen a settled
            # non-click.  Retry is `operation_version + 1`, a paper-trailed
            # human act.
            tx.execute(
                """CREATE TABLE IF NOT EXISTS click_attempts (
                       workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                           ON DELETE CASCADE,
                       position_a TEXT NOT NULL,
                       position_b TEXT NOT NULL,
                       operation_version INTEGER NOT NULL DEFAULT 1
                           CHECK (operation_version >= 1),
                       outcome TEXT NOT NULL
                           CHECK (outcome IN ('no_click','gate_failed',
                               'declined','clicked','expired','failed',
                               'reconsidered')),
                       detail TEXT NOT NULL DEFAULT '',
                       attempted_at REAL NOT NULL,
                       PRIMARY KEY
                           (workspace_id, position_a, position_b,
                            operation_version),
                       CHECK (position_a <= position_b)
                   )"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS click_attempts_outcome_idx
                   ON click_attempts(workspace_id,outcome,attempted_at)"""
            )
            tx.execute(
                """CREATE TABLE IF NOT EXISTS click_candidates (
                       id TEXT PRIMARY KEY,
                       workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                           ON DELETE CASCADE,
                       position_a TEXT NOT NULL,
                       position_b TEXT NOT NULL,
                       abstraction TEXT NOT NULL,
                       specializer_a TEXT NOT NULL DEFAULT '',
                       specializer_b TEXT NOT NULL DEFAULT '',
                       scope_boundary TEXT NOT NULL DEFAULT '',
                       status TEXT NOT NULL DEFAULT 'open'
                           CHECK (status IN
                               ('open','accepted','declined','expired')),
                       created_at REAL NOT NULL,
                       resolved_at REAL
                   )"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS click_candidates_open_idx
                   ON click_candidates(workspace_id,status,created_at)"""
            )
            # §4 cleanup of existing damage.  Global (workspace_id NULL) or
            # workspace-scoped, reversible, and never a Raven mutation.
            tx.execute(
                """CREATE TABLE IF NOT EXISTS suppression_registry (
                       raven_memory_id TEXT NOT NULL,
                       workspace_id TEXT,
                       reason TEXT NOT NULL DEFAULT '',
                       created_at REAL NOT NULL,
                       UNIQUE (raven_memory_id, workspace_id)
                   )"""
            )
            tx.execute(
                """CREATE INDEX IF NOT EXISTS suppression_memory_idx
                   ON suppression_registry(raven_memory_id)"""
            )
            tx.execute(
                """CREATE TABLE IF NOT EXISTS dismissals (
                       workspace_id TEXT NOT NULL REFERENCES workspaces(id)
                           ON DELETE CASCADE,
                       raven_memory_id TEXT NOT NULL,
                       reason TEXT NOT NULL DEFAULT '',
                       created_at REAL NOT NULL,
                       PRIMARY KEY (workspace_id, raven_memory_id)
                   )"""
            )
            # §4 — the export gate, as a column rather than a caller condition.
            columns = {
                str(row["name"])
                for row in tx.execute(
                    "PRAGMA table_info(raven_remember_outbox)"
                ).fetchall()
            }
            if "export_class" not in columns:
                tx.execute(
                    """ALTER TABLE raven_remember_outbox
                       ADD COLUMN export_class TEXT NOT NULL
                       DEFAULT 'human_root'"""
                )
            tx.execute("PRAGMA user_version = 4")

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

    # -- positions: the durable index (§1.2, §5) ------------------------

    def sync_positions(
        self, workspace_id: str, snapshot: dict[str, Any]
    ) -> list[PositionRow]:
        """Project a snapshot's positions into the index.  One direction only.

        The snapshot is authoritative (§5).  This method never writes back into
        it and nothing reconstructs an engine from these rows, so the index can
        never become a second source of truth that drifts from the floor.

        Positions are upserted, never deleted: a position id is durable and
        never reused (§1.2), and ``vacated``/``retired`` are states the row
        records rather than reasons to drop it.  ``created_at`` therefore
        survives every later sync.

        Two derived quantities are refused entry deliberately.  A frame's
        ``support_state`` and ``receipt`` are forced to NULL regardless of what
        the snapshot's occupant payload claims — the same structural refusal the
        engine makes in ``Position.support_state`` (§1.1, §1.5).  A forged
        supported frame cannot be laundered into the database by way of the
        index.
        """

        ts = float(self.now())
        rows = [
            row
            for row in (snapshot.get("positions") or [])
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        ]
        with self.transaction() as db:
            for row in rows:
                position_id = str(row["id"]).strip()
                occupant = dict(row.get("occupant") or {})
                declared_supports = [
                    str(value) for value in (row.get("supports") or []) if value
                ]
                floor_kind = str(row.get("floor_kind") or "claim")
                if floor_kind not in FLOOR_KINDS:
                    floor_kind = "claim"
                # A position standing on a floor IS a frame — that is what
                # `altitude(p) = 1 + max(...)` means (§1.2).  Deriving frame-ness
                # from the edges rather than trusting the declared label closes
                # the obvious forgery: relabel a frame as a claim and its
                # occupant's `supported` + receipt would otherwise be stored as
                # if the floor had never decided anything.  The index must not
                # be more credulous than the engine.
                if declared_supports:
                    floor_kind = "frame"
                origin = str(row.get("origin") or "human")
                if origin not in POSITION_ORIGINS:
                    origin = "human"
                status = str(row.get("status") or "live")
                if status not in POSITION_STATUSES:
                    status = "live"
                text = str(occupant.get("text") or "")
                try:
                    fingerprint = idea_fingerprint(text)
                except ValueError:
                    fingerprint = ""
                # THE FLOOR DECIDES.  A frame's support is computed, so the
                # index stores nothing it could contradict.
                if floor_kind == "frame":
                    support_state = None
                    receipt = None
                else:
                    support_state = str(occupant.get("state") or "open")
                    receipt = occupant.get("receipt")
                    receipt = str(receipt) if str(receipt or "").strip() else None
                artifact_type = str(occupant.get("artifact_type") or "claim")
                supports = declared_supports
                db.execute(
                    """INSERT INTO positions
                       (workspace_id,position_id,floor_kind,origin,status,
                        folded_under,external,pinned_by_human,last_grounded_at,
                        occupant_text,occupant_fingerprint,artifact_type,
                        support_state,receipt,provenance_json,lineage_json,
                        confirmed_by,confirmed_at,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(workspace_id,position_id) DO UPDATE SET
                         floor_kind=excluded.floor_kind,
                         origin=excluded.origin,
                         status=excluded.status,
                         folded_under=excluded.folded_under,
                         external=excluded.external,
                         pinned_by_human=excluded.pinned_by_human,
                         last_grounded_at=excluded.last_grounded_at,
                         occupant_text=excluded.occupant_text,
                         occupant_fingerprint=excluded.occupant_fingerprint,
                         artifact_type=excluded.artifact_type,
                         support_state=excluded.support_state,
                         receipt=excluded.receipt,
                         provenance_json=excluded.provenance_json,
                         lineage_json=excluded.lineage_json,
                         confirmed_by=excluded.confirmed_by,
                         confirmed_at=excluded.confirmed_at,
                         updated_at=excluded.updated_at""",
                    (
                        workspace_id, position_id, floor_kind, origin, status,
                        row.get("folded_under"),
                        int(bool(row.get("external", False))),
                        int(bool(row.get("pinned_by_human", False))),
                        row.get("last_grounded_at"),
                        text, fingerprint, artifact_type,
                        support_state, receipt,
                        _json([str(v) for v in (row.get("provenance") or [])]),
                        _json([str(v) for v in (row.get("lineage") or [])]),
                        row.get("confirmed_by"), row.get("confirmed_at"),
                        ts, ts,
                    ),
                )
                db.execute(
                    """DELETE FROM position_supports
                       WHERE workspace_id=? AND position_id=?""",
                    (workspace_id, position_id),
                )
                for ordinal, supports_id in enumerate(supports):
                    db.execute(
                        """INSERT OR REPLACE INTO position_supports
                           (workspace_id,position_id,supports_id,ordinal)
                           VALUES (?,?,?,?)""",
                        (workspace_id, position_id, supports_id, ordinal),
                    )
                self._sync_occupant_revisions(
                    db, workspace_id, position_id, row, occupant
                )
        return self.list_positions(workspace_id)

    def _sync_occupant_revisions(
        self,
        db: sqlite3.Connection,
        workspace_id: str,
        position_id: str,
        row: dict[str, Any],
        occupant: dict[str, Any],
    ) -> None:
        """Append-only occupant history for one position (§1.2).

        The engine caps a card's in-memory ``evolution`` list, so the snapshot
        eventually forgets the oldest rewordings.  This table does not: it is
        append-only and keyed by revision ordinal, which is the point of making
        the position durable in the first place.
        """

        revisions = [
            item for item in (row.get("occupant_revisions") or [])
            if isinstance(item, dict)
        ]
        known = {
            str(r["fingerprint"])
            for r in db.execute(
                """SELECT fingerprint FROM occupant_revisions
                   WHERE workspace_id=? AND position_id=?""",
                (workspace_id, position_id),
            ).fetchall()
        }
        next_revision = int(
            db.execute(
                """SELECT COALESCE(MAX(revision),-1)+1 FROM occupant_revisions
                   WHERE workspace_id=? AND position_id=?""",
                (workspace_id, position_id),
            ).fetchone()[0]
        )
        # The wording each revision replaced, then the wording standing now.
        history = [
            (str(item.get("from") or ""), "superseded",
             str(item.get("foot") or ""), float(item.get("ts") or 0.0))
            for item in revisions
        ]
        history.append(
            (str(occupant.get("text") or ""), "current",
             str(occupant.get("foot") or ""),
             float(occupant.get("last_seen") or occupant.get("born") or 0.0))
        )
        for text, relation, foot, recorded_at in history:
            if not str(text).strip():
                continue
            try:
                fingerprint = idea_fingerprint(text)
            except ValueError:
                continue
            if fingerprint in known:
                continue
            known.add(fingerprint)
            db.execute(
                """INSERT INTO occupant_revisions
                   (id,workspace_id,position_id,revision,text,fingerprint,
                    relation,foot,recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    new_id("rev"), workspace_id, position_id, next_revision,
                    str(text), fingerprint, relation, foot, recorded_at,
                ),
            )
            next_revision += 1

    def list_positions(
        self,
        workspace_id: str,
        *,
        floor_kind: str | None = None,
        status: str | None = None,
    ) -> list[PositionRow]:
        conditions = ["workspace_id=?"]
        params: list[Any] = [workspace_id]
        if floor_kind is not None:
            conditions.append("floor_kind=?")
            params.append(str(floor_kind))
        if status is not None:
            conditions.append("status=?")
            params.append(str(status))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM positions WHERE {' AND '.join(conditions)}
                    ORDER BY created_at,position_id""",
                params,
            ).fetchall()
            edges = self._conn.execute(
                """SELECT position_id,supports_id FROM position_supports
                   WHERE workspace_id=? ORDER BY position_id,ordinal""",
                (workspace_id,),
            ).fetchall()
        supports: dict[str, list[str]] = {}
        for edge in edges:
            supports.setdefault(str(edge["position_id"]), []).append(
                str(edge["supports_id"])
            )
        return [self._position(row, supports.get(str(row["position_id"]), []))
                for row in rows]

    def get_position(
        self, workspace_id: str, position_id: str
    ) -> PositionRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM positions WHERE workspace_id=? AND position_id=?",
                (workspace_id, position_id),
            ).fetchone()
            if row is None:
                return None
            edges = self._conn.execute(
                """SELECT supports_id FROM position_supports
                   WHERE workspace_id=? AND position_id=? ORDER BY ordinal""",
                (workspace_id, position_id),
            ).fetchall()
        return self._position(row, [str(edge["supports_id"]) for edge in edges])

    def find_position_by_fingerprint(
        self, workspace_id: str, text: str
    ) -> PositionRow | None:
        """§4 — adoption-time dedupe onto an existing position.

        The one job position ids cannot do alone (Appendix B #2): a re-adopted
        memory must land on the position it already occupies rather than minting
        a fresh one, or it could mint its way around the never-retry ledger.
        """

        try:
            fingerprint = idea_fingerprint(text)
        except ValueError:
            return None
        with self._lock:
            row = self._conn.execute(
                """SELECT position_id FROM positions
                   WHERE workspace_id=? AND occupant_fingerprint=?
                     AND status<>'retired'
                   ORDER BY created_at,position_id LIMIT 1""",
                (workspace_id, fingerprint),
            ).fetchone()
        if row is None:
            return None
        return self.get_position(workspace_id, str(row["position_id"]))

    def list_occupant_revisions(
        self, workspace_id: str, position_id: str
    ) -> list[OccupantRevision]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM occupant_revisions
                   WHERE workspace_id=? AND position_id=?
                   ORDER BY revision""",
                (workspace_id, position_id),
            ).fetchall()
        return [self._occupant_revision(row) for row in rows]

    def stale_positions(
        self, workspace_id: str, *, older_than: float
    ) -> list[PositionRow]:
        """§1.2/§3.3 — positions whose ground has not been touched recently.

        A never-grounded position is stale by construction, which is why NULL
        sorts in rather than out: an ungrounded ceiling is exactly what the
        staleness surface exists to make visible (§7.4).
        """

        with self._lock:
            rows = self._conn.execute(
                """SELECT position_id FROM positions
                   WHERE workspace_id=? AND status IN ('live','folded')
                     AND (last_grounded_at IS NULL OR last_grounded_at<?)
                   ORDER BY last_grounded_at IS NOT NULL,last_grounded_at,
                            position_id""",
                (workspace_id, float(older_than)),
            ).fetchall()
        out = []
        for row in rows:
            position = self.get_position(workspace_id, str(row["position_id"]))
            if position is not None:
                out.append(position)
        return out

    # -- click ledger and emergence inbox (§2.3, §2.4) ------------------

    def record_click_attempt(
        self,
        workspace_id: str,
        position_a: str,
        position_b: str,
        outcome: str,
        *,
        operation_version: int = 1,
        detail: str = "",
    ) -> dict[str, Any]:
        """Write one never-retry row.  Order-independent by construction."""

        a, b = sorted((str(position_a), str(position_b)))
        ts = float(self.now())
        with self.transaction() as db:
            db.execute(
                """INSERT INTO click_attempts
                   (workspace_id,position_a,position_b,operation_version,
                    outcome,detail,attempted_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(workspace_id,position_a,position_b,
                               operation_version)
                   DO UPDATE SET outcome=excluded.outcome,
                                 detail=excluded.detail,
                                 attempted_at=excluded.attempted_at""",
                (workspace_id, a, b, int(operation_version), str(outcome),
                 str(detail), ts),
            )
            row = db.execute(
                """SELECT * FROM click_attempts
                   WHERE workspace_id=? AND position_a=? AND position_b=?
                     AND operation_version=?""",
                (workspace_id, a, b, int(operation_version)),
            ).fetchone()
        return dict(row)

    def list_click_attempts(
        self, workspace_id: str, *, position_a: str | None = None,
        position_b: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["workspace_id=?"]
        params: list[Any] = [workspace_id]
        if position_a is not None and position_b is not None:
            a, b = sorted((str(position_a), str(position_b)))
            conditions.extend(["position_a=?", "position_b=?"])
            params.extend([a, b])
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM click_attempts
                    WHERE {' AND '.join(conditions)}
                    ORDER BY attempted_at,operation_version""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def sync_click_ledger(
        self, workspace_id: str, snapshot: dict[str, Any]
    ) -> None:
        """Project the engine's attempt ledger and inbox into their indices."""

        with self.transaction() as db:
            for row in (snapshot.get("click_attempts") or []):
                if not isinstance(row, dict):
                    continue
                try:
                    a, b = sorted(
                        (str(row["position_a"]), str(row["position_b"]))
                    )
                except (KeyError, TypeError):
                    continue
                db.execute(
                    """INSERT INTO click_attempts
                       (workspace_id,position_a,position_b,operation_version,
                        outcome,detail,attempted_at)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(workspace_id,position_a,position_b,
                                   operation_version)
                       DO UPDATE SET outcome=excluded.outcome,
                                     detail=excluded.detail,
                                     attempted_at=excluded.attempted_at""",
                    (
                        workspace_id, a, b,
                        int(row.get("operation_version", 1)),
                        str(row.get("outcome") or "no_click"),
                        str(row.get("detail") or ""),
                        float(row.get("attempted_at") or 0.0),
                    ),
                )
            for row in (snapshot.get("click_candidates") or []):
                if not isinstance(row, dict) or not row.get("id"):
                    continue
                status = str(row.get("status") or "open")
                db.execute(
                    """INSERT INTO click_candidates
                       (id,workspace_id,position_a,position_b,abstraction,
                        specializer_a,specializer_b,scope_boundary,status,
                        created_at,resolved_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         status=excluded.status,
                         abstraction=excluded.abstraction,
                         specializer_a=excluded.specializer_a,
                         specializer_b=excluded.specializer_b,
                         scope_boundary=excluded.scope_boundary,
                         resolved_at=excluded.resolved_at""",
                    (
                        str(row["id"]), workspace_id,
                        str(row.get("position_a") or ""),
                        str(row.get("position_b") or ""),
                        str(row.get("abstraction") or ""),
                        str(row.get("specializer_a") or ""),
                        str(row.get("specializer_b") or ""),
                        str(row.get("scope_boundary") or ""),
                        status,
                        float(row.get("created_at") or 0.0),
                        None if status == "open" else float(self.now()),
                    ),
                )

    def list_click_candidates(
        self, workspace_id: str, *, status: str | None = "open"
    ) -> list[dict[str, Any]]:
        conditions = ["workspace_id=?"]
        params: list[Any] = [workspace_id]
        if status is not None:
            conditions.append("status=?")
            params.append(str(status))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM click_candidates
                    WHERE {' AND '.join(conditions)} ORDER BY created_at,id""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    # -- suppression and dismissal (§4) --------------------------------

    def suppress_memory(
        self,
        raven_memory_id: str,
        *,
        workspace_id: str | None = None,
        reason: str = "",
    ) -> Suppression:
        """Tag a memory id so it stops resurfacing.  Local and reversible."""

        memory_id = str(raven_memory_id or "").strip()
        if not memory_id:
            raise ValueError("raven memory id is required")
        ts = float(self.now())
        with self.transaction() as db:
            db.execute(
                """INSERT INTO suppression_registry
                   (raven_memory_id,workspace_id,reason,created_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(raven_memory_id,workspace_id)
                   DO UPDATE SET reason=excluded.reason""",
                (memory_id, workspace_id, str(reason), ts),
            )
            row = db.execute(
                """SELECT * FROM suppression_registry
                   WHERE raven_memory_id=? AND workspace_id IS ?""",
                (memory_id, workspace_id),
            ).fetchone()
        return self._suppression(row)

    def unsuppress_memory(
        self, raven_memory_id: str, *, workspace_id: str | None = None
    ) -> bool:
        """Reverse a suppression.  The registry is a lens, not a deletion."""

        with self.transaction() as db:
            cur = db.execute(
                """DELETE FROM suppression_registry
                   WHERE raven_memory_id=? AND workspace_id IS ?""",
                (str(raven_memory_id), workspace_id),
            )
        return cur.rowcount > 0

    def is_suppressed(
        self, raven_memory_id: str, *, workspace_id: str | None = None
    ) -> bool:
        with self._lock:
            row = self._conn.execute(
                """SELECT 1 FROM suppression_registry
                   WHERE raven_memory_id=?
                     AND (workspace_id IS NULL OR workspace_id=?)
                   LIMIT 1""",
                (str(raven_memory_id), workspace_id),
            ).fetchone()
        return row is not None

    def list_suppressions(
        self, *, workspace_id: str | None = None
    ) -> list[Suppression]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM suppression_registry
                   WHERE workspace_id IS NULL OR workspace_id=?
                   ORDER BY created_at,raven_memory_id""",
                (workspace_id,),
            ).fetchall()
        return [self._suppression(row) for row in rows]

    def dismiss_memory(
        self, workspace_id: str, raven_memory_id: str, *, reason: str = ""
    ) -> Dismissal:
        """§4 — durable workspace-local dismissal.

        Unlike an exposure row, which recall rewrites on every pass, this is
        terminal for the workspace: a human-dismissed memory never resurfaces
        there regardless of future recall scoring.
        """

        memory_id = str(raven_memory_id or "").strip()
        if not memory_id:
            raise ValueError("raven memory id is required")
        ts = float(self.now())
        with self.transaction() as db:
            db.execute(
                """INSERT INTO dismissals
                   (workspace_id,raven_memory_id,reason,created_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(workspace_id,raven_memory_id)
                   DO UPDATE SET reason=excluded.reason""",
                (workspace_id, memory_id, str(reason), ts),
            )
            row = db.execute(
                """SELECT * FROM dismissals
                   WHERE workspace_id=? AND raven_memory_id=?""",
                (workspace_id, memory_id),
            ).fetchone()
        return self._dismissal(row)

    def is_dismissed(self, workspace_id: str, raven_memory_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """SELECT 1 FROM dismissals
                   WHERE workspace_id=? AND raven_memory_id=? LIMIT 1""",
                (workspace_id, str(raven_memory_id)),
            ).fetchone()
        return row is not None

    def list_dismissals(self, workspace_id: str) -> list[Dismissal]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM dismissals WHERE workspace_id=?
                   ORDER BY created_at,raven_memory_id""",
                (workspace_id,),
            ).fetchall()
        return [self._dismissal(row) for row in rows]

    def recall_is_blocked(self, workspace_id: str, raven_memory_id: str) -> bool:
        """One question the recall path asks before showing anything (§4)."""

        memory_id = str(raven_memory_id or "").strip()
        if not memory_id:
            return True
        return self.is_suppressed(
            memory_id, workspace_id=workspace_id
        ) or self.is_dismissed(workspace_id, memory_id)

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
        export_class: str,
        workspace_id: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        episode_id: str | None = None,
        hints: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        available_at: float | None = None,
    ) -> RavenRemember:
        """Durably queue one Raven ``remember`` call.

        ``export_class`` is required and must be one of :data:`EXPORT_CLASSES`
        (§4).  This is the Raven contamination fix at its only chokepoint: the
        old ``_bank_card`` enqueued a write for *every* card, so machine
        fusions reached shared memory and were recalled back later.  There is
        no unclassified write, and therefore no way for a machine candidate, a
        derived-ungrounded claim, or an unpromoted fold to reach Raven — a
        caller that cannot name a class is a caller that must not be banking.

        ``dedupe_key`` prevents duplicate local enqueue operations. Raven's
        current API does not accept an idempotency key, so delivery is
        at-least-once if a process dies after the remote write but before the
        local completion commit.
        """

        clean_content = str(content or "").strip()
        if not clean_content:
            raise ValueError("remember content is required")
        export = str(export_class or "").strip()
        if export not in EXPORT_CLASSES:
            raise ValueError(
                "export_class must be one of "
                f"{', '.join(EXPORT_CLASSES)}; machine-generated material is "
                "never banked"
            )
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
                    created_at,available_at,export_class)
                   VALUES (?,?,?,?, 'pending',0,?,?,?)
                   ON CONFLICT(dedupe_key) DO NOTHING""",
                (
                    outbox_id, workspace_id, key, _json(payload), ts,
                    ts if available_at is None else float(available_at),
                    export,
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

    def backfill_positions(self, workspace_id: str) -> dict[str, Any]:
        """SPEC §5 — the conservative, one-pass-per-workspace backfill.

        The rule this method exists to enforce: **never auto-convert receipted
        syntheses into folds.**  Combination provenance is not identity
        recognition, and fabricating ``supports`` edges from legacy ``parents``
        would seed the ladder with exactly the structure the click gates exist
        to prevent.  So:

        - Legacy ``parents`` are written as *provenance*, never as ``supports``.
          The position rows this pass creates have empty support edges even
          where the old card had two parents.
        - Legacy ``kind == 'synthesis'`` becomes a provisional annotation and a
          row in the migration inbox, awaiting human confirmation through the
          standard gates.  It does **not** become ``floor_kind='frame'``.
        - Archived parents of receipted syntheses are un-archived to floor 0 —
          recovering material the old law destroyed — but left unfolded pending
          that confirmation.
        - Syntheses without receipts revert to ``needs_human`` claims, floor 0.
        - Every historical pair, from live *and* archived cards, is written to
          ``click_attempts`` with ``outcome='no_click'``,
          ``operation_version=1``: migration itself seeds the never-retry
          memory.  Historical retries collapse to one row.

        Returns the migration inbox and counters.  It mutates the snapshot's
        card states only where §5 mandates (un-archiving recovered parents,
        reverting receiptless syntheses), and is idempotent: a second pass finds
        the same rows and rewrites them identically.
        """

        workspace = self.load_workspace(workspace_id)
        snapshot = dict(workspace.snapshot or {})
        cards = [
            dict(card) for card in (snapshot.get("cards") or [])
            if isinstance(card, dict) and str(card.get("id") or "").strip()
        ]
        by_id = {str(card["id"]): card for card in cards}
        ts = float(self.now())

        pending_frames: list[dict[str, Any]] = []
        recovered: list[str] = []
        reverted: list[str] = []
        pair_rows: set[tuple[str, str]] = set()

        for card in cards:
            parents = [
                str(value) for value in (card.get("parents") or []) if value
            ]
            # Historical pairs seed the ledger regardless of what became of the
            # card: memory must be separate from output (§2.3), so an archived
            # child no longer resurrects its pair.
            for i, a in enumerate(parents):
                for b in parents[i + 1:]:
                    if a != b:
                        pair_rows.add(tuple(sorted((a, b))))
            is_synthesis = str(card.get("kind") or "") == "synthesis"
            if not is_synthesis:
                continue
            has_receipt = bool(str(card.get("receipt") or "").strip())
            if has_receipt and str(card.get("state")) == "supported":
                pending_frames.append({
                    "position_id": str(card["id"]),
                    "text": str(card.get("text") or ""),
                    "provenance": parents,
                    "receipt": str(card.get("receipt") or ""),
                })
                for parent_id in parents:
                    parent = by_id.get(parent_id)
                    if parent is not None and parent.get("archived"):
                        parent["archived"] = False
                        recovered.append(parent_id)
            else:
                # No receipt, no terminal state.  The law does not bend for
                # history: what was never grounded reverts to needs_human.
                if card.get("state") in ("supported", "refuted"):
                    card["state"] = "needs_human"
                    card["receipt"] = None
                    reverted.append(str(card["id"]))

        snapshot["cards"] = cards
        with self.transaction():
            self.save_workspace(
                workspace_id, snapshot, increment_context=False
            )
            for a, b in sorted(pair_rows):
                self.record_click_attempt(
                    workspace_id, a, b, "no_click",
                    operation_version=1, detail="migration: historical pair",
                )
            with self.transaction() as db:
                for entry in pending_frames:
                    db.execute(
                        """INSERT INTO positions
                           (workspace_id,position_id,floor_kind,origin,status,
                            folded_under,external,pinned_by_human,
                            last_grounded_at,occupant_text,
                            occupant_fingerprint,artifact_type,support_state,
                            receipt,provenance_json,lineage_json,confirmed_by,
                            confirmed_at,created_at,updated_at)
                           VALUES (?,?,'claim','human','live',NULL,0,0,NULL,?,
                                   ?,'claim','needs_human',NULL,?,'[]',NULL,
                                   NULL,?,?)
                           ON CONFLICT(workspace_id,position_id) DO UPDATE SET
                             provenance_json=excluded.provenance_json,
                             updated_at=excluded.updated_at""",
                        (
                            workspace_id, entry["position_id"],
                            entry["text"],
                            idea_fingerprint(entry["text"])
                            if entry["text"].strip() else "",
                            _json(entry["provenance"]), ts, ts,
                        ),
                    )
        return {
            "workspace_id": workspace_id,
            "migration_inbox": pending_frames,
            "recovered_parents": sorted(set(recovered)),
            "reverted_syntheses": sorted(set(reverted)),
            "seeded_attempts": len(pair_rows),
        }

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
            export_class=(
                row["export_class"]
                if "export_class" in row.keys()
                else "human_root"
            ),
        )

    @staticmethod
    def _position(row: sqlite3.Row, supports: list[str]) -> PositionRow:
        return PositionRow(
            workspace_id=row["workspace_id"], position_id=row["position_id"],
            floor_kind=row["floor_kind"], origin=row["origin"],
            status=row["status"], folded_under=row["folded_under"],
            external=bool(row["external"]),
            pinned_by_human=bool(row["pinned_by_human"]),
            last_grounded_at=row["last_grounded_at"],
            occupant_text=row["occupant_text"],
            occupant_fingerprint=row["occupant_fingerprint"],
            artifact_type=row["artifact_type"],
            support_state=row["support_state"], receipt=row["receipt"],
            supports=list(supports),
            provenance=_decode(row["provenance_json"], []),
            lineage=_decode(row["lineage_json"], []),
            confirmed_by=row["confirmed_by"], confirmed_at=row["confirmed_at"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _occupant_revision(row: sqlite3.Row) -> OccupantRevision:
        return OccupantRevision(
            id=row["id"], workspace_id=row["workspace_id"],
            position_id=row["position_id"], revision=row["revision"],
            text=row["text"], fingerprint=row["fingerprint"],
            relation=row["relation"], foot=row["foot"],
            recorded_at=row["recorded_at"],
        )

    @staticmethod
    def _suppression(row: sqlite3.Row) -> Suppression:
        return Suppression(
            raven_memory_id=row["raven_memory_id"],
            workspace_id=row["workspace_id"], reason=row["reason"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _dismissal(row: sqlite3.Row) -> Dismissal:
        return Dismissal(
            workspace_id=row["workspace_id"],
            raven_memory_id=row["raven_memory_id"], reason=row["reason"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _embedding(row: sqlite3.Row) -> Embedding:
        return Embedding(
            idea_id=row["idea_id"], model=row["model"], version=row["version"],
            dimensions=row["dimensions"], vector=bytes(row["vector"]),
            encoding=row["encoding"], metadata=_decode(row["metadata_json"], {}),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
