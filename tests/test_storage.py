import json
import sqlite3
import struct

import pytest

from magpie.storage import SCHEMA_VERSION, Storage, idea_fingerprint


def test_workspace_persists_and_resumes_after_reopen(tmp_path):
    path = tmp_path / "magpie.sqlite3"
    with Storage(path) as store:
        ws = store.create_workspace("Checkout", question="Why?", snapshot={"cards": []})
        saved = store.save_workspace(
            ws.id,
            {"question": "Why?", "cards": [{"id": "c1", "text": "Latency"}]},
            expected_context_version=0,
        )
        assert saved.context_version == 1

    with Storage(path) as reopened:
        loaded = reopened.load_workspace(ws.id)
        assert loaded.name == "Checkout"
        assert loaded.snapshot["cards"][0]["text"] == "Latency"
        assert loaded.context_version == 1
        assert reopened.list_workspaces()[0].id == ws.id


def test_workspace_snapshots_and_occurrences_are_isolated(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        a = store.create_workspace("A", snapshot={"cards": ["a"]})
        b = store.create_workspace("B", snapshot={"cards": ["b"]})
        idea, _ = store.upsert_idea("A shared point")
        store.link_idea(a.id, idea.id, local_ref="a1")

        assert store.load_workspace(a.id).snapshot == {"cards": ["a"]}
        assert store.load_workspace(b.id).snapshot == {"cards": ["b"]}
        assert len(store.list_workspace_ideas(a.id)) == 1
        assert store.list_workspace_ideas(b.id) == []


def test_workspace_save_detects_stale_context_version(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        store.save_workspace(ws.id, {"rev": 1}, expected_context_version=0)
        with pytest.raises(RuntimeError, match="context version conflict"):
            store.save_workspace(ws.id, {"rev": 2}, expected_context_version=0)
        assert store.load_workspace(ws.id).snapshot == {"rev": 1}


def test_idea_upsert_deduplicates_normalized_exact_text(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        assert store.bank_revision() == 0
        first, created1 = store.upsert_idea("  Mobile validation—errors! ")
        second, created2 = store.upsert_idea("mobile validation errors")

        assert created1 is True
        assert created2 is False
        assert first.id == second.id
        assert first.fingerprint == idea_fingerprint("MOBILE validation errors.")
        assert store.bank_revision() == 1


def test_current_workspace_selection_persists(tmp_path):
    path = tmp_path / "db.sqlite3"
    with Storage(path) as store:
        a = store.create_workspace("A")
        b = store.create_workspace("B")
        assert store.current_workspace_id() is None
        store.set_current_workspace(b.id)
        assert store.current_workspace_id() == b.id

    with Storage(path) as reopened:
        assert reopened.current_workspace_id() == b.id
        reopened.set_current_workspace(a.id)
        assert reopened.current_workspace_id() == a.id


def test_event_queue_claim_mark_and_retry(tmp_path):
    clock = iter([10, 11, 12, 13, 14, 15, 16, 17]).__next__
    with Storage(tmp_path / "db.sqlite3", now=clock) as store:
        ws = store.create_workspace("A")
        first = store.append_event(
            "thought.added", {"card": "c1"}, workspace_id=ws.id, context_version=2
        )
        store.append_event("later", workspace_id=ws.id, available_at=100)

        claimed = store.poll_events(workspace_id=ws.id)
        assert [event.id for event in claimed] == [first.id]
        assert claimed[0].status == "processing"
        assert claimed[0].attempts == 1
        assert store.poll_events(workspace_id=ws.id) == []

        retry = store.mark_event(first.id, "pending", error="temporary")
        assert retry.status == "pending"
        claimed_again = store.poll_events(workspace_id=ws.id)
        assert claimed_again[0].attempts == 2
        done = store.mark_event(first.id, "completed")
        assert done.status == "completed"
        assert done.completed_at is not None


def test_event_queue_reclaims_expired_processing_lease(tmp_path):
    now = [100.0]
    with Storage(tmp_path / "db.sqlite3", now=lambda: now[0]) as store:
        ws = store.create_workspace("A")
        event = store.append_event("retrieve", workspace_id=ws.id)
        assert store.poll_events(workspace_id=ws.id, lease_seconds=30)[0].id == event.id

        now[0] = 129.0
        assert store.poll_events(workspace_id=ws.id, lease_seconds=30) == []
        now[0] = 131.0
        reclaimed = store.poll_events(workspace_id=ws.id, lease_seconds=30)
        assert [item.id for item in reclaimed] == [event.id]
        assert reclaimed[0].attempts == 2


def test_memory_exposure_upsert_preserves_first_seen(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        idea, _ = store.upsert_idea("Remember this")
        first = store.upsert_exposure(
            ws.id, idea.id, reason="initial question", context_version=1
        )
        second = store.upsert_exposure(
            ws.id, idea.id, status="dismissed", reason="not relevant",
            context_version=2,
        )
        assert second.id == first.id
        assert second.first_shown_at == first.first_shown_at
        assert second.status == "dismissed"
        assert store.list_exposures(ws.id) == [second]


def test_embeddings_are_versioned_by_model_and_round_trip_blobs(tmp_path):
    vector_a = struct.pack("<3f", 0.1, 0.2, 0.3)
    vector_b = struct.pack("<2f", 0.8, 0.9)
    with Storage(tmp_path / "db.sqlite3") as store:
        idea, _ = store.upsert_idea("Vectorize me")
        store.put_embedding(
            idea.id, model="nomic", version="v1", dimensions=3, vector=vector_a
        )
        store.put_embedding(
            idea.id, model="other", version="2026-01", dimensions=2,
            vector=vector_b, encoding="float32-le",
        )
        one = store.get_embedding(idea.id, model="nomic", version="v1")
        two = store.get_embedding(idea.id, model="other", version="2026-01")

        assert one is not None and one.vector == vector_a and one.dimensions == 3
        assert two is not None and two.vector == vector_b and two.dimensions == 2
        assert store.get_embedding(idea.id, model="nomic", version="v2") is None


def test_embeddings_validate_blob_size_and_model_shape(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        first, _ = store.upsert_idea("First")
        second, _ = store.upsert_idea("Second")
        with pytest.raises(ValueError, match="exactly 12 bytes"):
            store.put_embedding(
                first.id, model="embed", version="v1", dimensions=3, vector=b"bad"
            )
        store.put_embedding(
            first.id,
            model="embed",
            version="v1",
            dimensions=3,
            vector=struct.pack("<3f", 0.1, 0.2, 0.3),
        )
        with pytest.raises(ValueError, match="one dimension and encoding"):
            store.put_embedding(
                second.id,
                model="embed",
                version="v1",
                dimensions=2,
                vector=struct.pack("<2f", 0.1, 0.2),
            )


def test_legacy_migration_imports_snapshot_and_bank_without_deleting_source(tmp_path):
    legacy = tmp_path / "state.json"
    state = {
        "question": "Old question",
        "cards": [
            {"id": "c1", "text": "One point", "kind": "claim", "state": "open"},
            {"id": "c2", "text": "one point!", "kind": "claim", "state": "open"},
        ],
    }
    legacy.write_text(json.dumps(state), encoding="utf-8")

    with Storage(tmp_path / "db.sqlite3") as store:
        workspace = store.migrate_legacy_state(legacy)
        assert workspace.question == "Old question"
        assert workspace.snapshot == state
        occurrences = store.list_workspace_ideas(workspace.id)
        assert len(occurrences) == 2
        assert occurrences[0]["idea"].id == occurrences[1]["idea"].id

    assert legacy.exists()
    assert json.loads(legacy.read_text(encoding="utf-8")) == state


def test_schema_version_and_wal_are_initialized(tmp_path):
    path = tmp_path / "db.sqlite3"
    with Storage(path):
        pass
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
