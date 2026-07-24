import sqlite3

import pytest

from magpie.storage import SCHEMA_VERSION, Storage


def test_raven_projection_is_workspace_local_and_upserts_layout_state(tmp_path):
    now = [10.0]
    with Storage(tmp_path / "db.sqlite3", now=lambda: now[0]) as store:
        first_ws = store.create_workspace("First")
        second_ws = store.create_workspace("Second")
        first = store.upsert_raven_projection(
            first_ws.id,
            "mem_1",
            local_ref="card_1",
            section="shelf",
            mass=2.5,
            metadata={"recall_score": 0.8},
        )
        other = store.upsert_raven_projection(second_ws.id, "mem_1")

        now[0] = 20.0
        changed = store.upsert_raven_projection(
            first_ws.id,
            "mem_1",
            local_ref="card_1",
            section="field",
            mass=3,
            pinned=True,
            hidden=True,
            local_note="Use in the launch section",
            local_status="accepted",
        )

        assert changed.id == first.id
        assert changed.created_at == first.created_at == 10.0
        assert changed.updated_at == 20.0
        assert changed.pinned is True
        assert changed.hidden is True
        assert changed.local_note == "Use in the launch section"
        assert changed.local_status == "accepted"
        assert changed.metadata == {}
        assert other.id != changed.id
        assert store.list_raven_projections(first_ws.id, include_hidden=False) == []
        assert store.list_raven_projections(
            first_ws.id, section="field"
        ) == [changed]
        assert store.get_raven_projection(first_ws.id, "missing") is None


def test_raven_projection_validates_external_identity_and_mass(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        with pytest.raises(ValueError, match="memory id"):
            store.upsert_raven_projection(ws.id, "")
        with pytest.raises(ValueError, match="non-negative"):
            store.upsert_raven_projection(ws.id, "mem_1", mass=-0.1)


def test_raven_exposure_stays_local_and_preserves_first_seen(tmp_path):
    now = [100.0]
    with Storage(tmp_path / "db.sqlite3", now=lambda: now[0]) as store:
        ws = store.create_workspace("A")
        first = store.upsert_raven_exposure(
            ws.id,
            "mem_1",
            reason="retrieved for question",
            context_version=2,
        )
        now[0] = 110.0
        second = store.upsert_raven_exposure(
            ws.id,
            "mem_1",
            status="dismissed",
            reason="not useful here",
            context_version=3,
            metadata={"local_only": True},
        )

        assert second.id == first.id
        assert second.first_shown_at == 100.0
        assert second.last_shown_at == 110.0
        assert second.status == "dismissed"
        assert second.metadata == {"local_only": True}
        assert store.list_raven_exposures(ws.id) == [second]


def test_raven_remember_outbox_dedupes_claims_retries_and_completes(tmp_path):
    now = [100.0]
    with Storage(tmp_path / "db.sqlite3", now=lambda: now[0]) as store:
        ws = store.create_workspace("A")
        first = store.enqueue_raven_remember(
            "A durable thought",
            export_class="human_root",
            workspace_id=ws.id,
            source="human",
            tags=["magpie"],
            hints={"derived_from": ["mem_parent"]},
            dedupe_key="card:c1:v1",
        )
        duplicate = store.enqueue_raven_remember(
            "This payload must not replace the original",
            export_class="human_root",
            workspace_id=ws.id,
            dedupe_key="card:c1:v1",
        )
        assert duplicate == first
        assert first.payload == {
            "content": "A durable thought",
            "source": "human",
            "tags": ["magpie"],
            "hints": {"derived_from": ["mem_parent"]},
        }

        claimed = store.claim_raven_remembers()
        assert [item.id for item in claimed] == [first.id]
        assert claimed[0].attempts == 1
        assert store.claim_raven_remembers() == []

        now[0] = 500.0
        reclaimed = store.claim_raven_remembers(lease_seconds=300)
        assert reclaimed[0].id == first.id
        assert reclaimed[0].attempts == 2
        pending = store.mark_raven_remember(
            first.id, "pending", error="temporary", available_at=600
        )
        assert pending.status == "pending"
        assert store.claim_raven_remembers() == []

        now[0] = 600.0
        assert store.claim_raven_remembers()[0].attempts == 3
        done = store.mark_raven_remember(
            first.id, "completed", raven_memory_id="mem_created"
        )
        assert done.status == "completed"
        assert done.raven_memory_id == "mem_created"
        assert done.completed_at == 600.0


def test_completed_raven_outbox_item_requires_memory_id(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        item = store.enqueue_raven_remember(
            "Remember me", export_class="human_root"
        )
        with pytest.raises(ValueError, match="requires a raven memory id"):
            store.mark_raven_remember(item.id, "completed")


def test_completed_raven_outbox_preserves_each_local_ref_binding(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        for local_ref in ("c1", "c2"):
            item = store.enqueue_raven_remember(
                "Canonical duplicate",
                export_class="human_root",
                workspace_id=ws.id,
                dedupe_key=f"magpie-card:{ws.id}:{local_ref}",
            )
            store.mark_raven_remember(
                item.id, "completed", raven_memory_id="mem-canonical"
            )

        assert (
            store.completed_raven_memory_id_for_local_ref(ws.id, "c1")
            == "mem-canonical"
        )
        assert (
            store.completed_raven_memory_id_for_local_ref(ws.id, "c2")
            == "mem-canonical"
        )
        assert (
            store.completed_raven_memory_id_for_local_ref(ws.id, "missing")
            is None
        )


def test_raven_outbox_does_not_starve_fresh_items_behind_retries(tmp_path):
    now = [100.0]
    with Storage(tmp_path / "db.sqlite3", now=lambda: now[0]) as store:
        retried = store.enqueue_raven_remember(
            "Waiting descendant", export_class="human_root",
            dedupe_key="retrying"
        )
        claimed = store.claim_raven_remembers(limit=1)
        assert claimed[0].id == retried.id
        store.mark_raven_remember(
            retried.id, "pending", error="waiting", available_at=100.0
        )
        fresh = store.enqueue_raven_remember(
            "Fresh root", export_class="human_root", dedupe_key="fresh"
        )

        next_item = store.claim_raven_remembers(limit=1)[0]

        assert next_item.id == fresh.id
        assert next_item.attempts == 1


def test_outbox_rows_carry_their_export_class_for_inspection(tmp_path):
    """§4 — eligibility is an inspectable property of each queued write.

    The point of the column is that you can audit what left the workspace
    without re-deriving the decision from a function body.
    """
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        store.enqueue_raven_remember(
            "An atomized human contribution",
            export_class="human_root",
            workspace_id=ws.id,
            dedupe_key="root",
        )
        store.enqueue_raven_remember(
            "A human-confirmed frame",
            export_class="human_curated_frame",
            workspace_id=ws.id,
            hints={"derived_from": ["mem-a", "mem-b"]},
            dedupe_key="frame",
        )
        store.enqueue_raven_remember(
            "A receipt-resolved claim",
            export_class="settled_claim",
            workspace_id=ws.id,
            dedupe_key="settled",
        )

        classes = {
            item.export_class for item in store.claim_raven_remembers(limit=10)
        }
        assert classes == {
            "human_root", "human_curated_frame", "settled_claim"
        }


def test_v2_database_migrates_additively_to_raven_schema(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                question TEXT NOT NULL DEFAULT '',
                snapshot_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                context_version INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO app_settings(key,value) VALUES ('bank_revision','0');
            INSERT INTO workspaces(
                id,name,question,snapshot_json,metadata_json,context_version,
                created_at,updated_at
            ) VALUES ('ws_old','Old','','{}','{}',0,1,1);
            PRAGMA user_version = 2;
            """
        )

    with Storage(path) as store:
        assert store.load_workspace("ws_old").name == "Old"
        projection = store.upsert_raven_projection("ws_old", "mem_1")
        assert projection.raven_memory_id == "mem_1"

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "raven_projections",
            "raven_exposures",
            "raven_remember_outbox",
        } <= tables
