import json
import sqlite3
import struct

import pytest

from magpie.engine import ClickProposal, Engine
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


# --------------------------------------------------------------------------
# §1.2 / §5 — position persistence
# --------------------------------------------------------------------------


def _laddered_engine():
    """A two-instance fold with a derived slot beneath it, honestly built."""
    engine = Engine(now=iter(range(1000, 2000)).__next__)
    a = engine.propose("Checkout stalls on mobile networks")
    b = engine.propose("Signup stalls on mobile networks")
    for _ in range(8):
        engine.propose("An unrelated human contribution")
    candidate = engine.propose_click(
        a.id,
        b.id,
        ClickProposal(
            abstraction="Radio handover interrupts long requests",
            specializer_a="in the case of the checkout flow",
            specializer_b="in the case of the signup flow",
            scope_boundary="excludes desktop throughput questions",
        ),
    )
    frame = engine.confirm_click(candidate.id, confirmed_by="andrew")
    engine.derive(
        frame.id,
        [{"text": "p95 mobile latency exceeds 800ms",
          "falsification": "a week of p95 under 400ms"}],
    )
    return engine, a, b, frame


def test_positions_persist_durable_ids_supports_and_grounding(tmp_path):
    engine, a, b, frame = _laddered_engine()
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_positions(ws.id, engine.state())

        frame_row = store.get_position(ws.id, frame.id)
        assert frame_row.floor_kind == "frame"
        assert frame_row.origin == "click"
        # The load-bearing edges survive, in order, as their own relation.
        assert frame_row.supports[:2] == [a.id, b.id]
        assert frame_row.confirmed_by == "andrew"
        # Lineage is the union of floor-0 roots covered — both instances plus
        # the slot derived beneath the frame (§1.6).
        assert {a.id, b.id} <= set(frame_row.lineage)

        # Instances are FOLDED, never archived (§1.6) — the whole point of the
        # deletion in resolve(). They remain individually addressable.
        assert store.get_position(ws.id, a.id).status == "folded"
        assert store.get_position(ws.id, a.id).folded_under == frame.id
        assert store.get_position(ws.id, b.id).status == "folded"

        # A derived slot is ungrounded by construction (§1.4).
        derived = [
            row for row in store.list_positions(ws.id)
            if row.origin == "derivation"
        ]
        assert len(derived) == 1
        assert derived[0].last_grounded_at is None
        assert derived[0].position_id in frame_row.supports


def test_frames_never_store_a_support_state_or_receipt(tmp_path):
    """§1.1/§1.5 — the receipt law, enforced by the index's own shape.

    A forged supported frame must not be launderable into the database. The
    column is refused at the door and the CHECK constraint refuses it again.
    """
    engine, _a, _b, frame = _laddered_engine()
    snapshot = engine.state()
    # Forge directly on the wire format, the way a hand-edited snapshot would —
    # and relabel the frame as a claim, so the label itself cannot be trusted.
    for row in snapshot["positions"]:
        if row["id"] == frame.id:
            row["floor_kind"] = "claim"
            row["occupant"]["state"] = "supported"
            row["occupant"]["receipt"] = "https://example.test/forged"

    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_positions(ws.id, snapshot)
        row = store.get_position(ws.id, frame.id)
        # Frame-ness is derived from the support edges, not read off the label,
        # so the relabelling buys the forgery nothing.
        assert row.floor_kind == "frame"
        assert row.support_state is None
        assert row.receipt is None

        # And the constraint holds even against raw SQL.
        with pytest.raises(sqlite3.IntegrityError):
            with store.transaction() as db:
                db.execute(
                    """UPDATE positions SET support_state='supported'
                       WHERE workspace_id=? AND position_id=?""",
                    (ws.id, frame.id),
                )


def test_frame_support_is_recomputed_from_the_floor_after_a_round_trip(tmp_path):
    """§1.5 — nothing at any layer may drift from the layer below."""
    engine, a, _b, frame = _laddered_engine()
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_positions(ws.id, engine.state())
        assert engine.frame_support(frame.id)["supported"] == 0

        # A receipt lands on the floor and climbs.
        engine.resolve(a.id, "supported", receipt="https://example.test/trace")
        store.sync_positions(ws.id, engine.state())

        assert store.get_position(ws.id, a.id).support_state == "supported"
        assert store.get_position(ws.id, a.id).receipt is not None
        # The frame stores nothing; its score is computed on read, and the
        # persisted last_grounded_at is the floor's max, not an assertion.
        assert store.get_position(ws.id, frame.id).support_state is None
        support = engine.frame_support(frame.id)
        assert support["supported"] == 1
        assert store.get_position(ws.id, frame.id).last_grounded_at == (
            support["last_grounded_at"]
        )


def test_rephrasing_keeps_the_position_and_records_the_revision(tmp_path):
    """§1.2 — rephrasing an occupant never touches the structure."""
    engine = Engine(now=iter(range(1000, 2000)).__next__)
    card = engine.propose("Checkout stalls on mobile")
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_positions(ws.id, engine.state())
        before = store.get_position(ws.id, card.id)
        engine.record_occurrence(
            card.id,
            "Checkout stalls on constrained mobile networks",
            relation="refinement",
        )
        store.sync_positions(ws.id, engine.state())
        # The receipt lands on the position AFTER the rewording, and the
        # position it lands on is the same one.
        engine.resolve(card.id, "supported", receipt="https://example.test/a")
        store.sync_positions(ws.id, engine.state())
        after = store.get_position(ws.id, card.id)

        assert after.position_id == before.position_id
        assert after.created_at == before.created_at
        assert after.receipt == "https://example.test/a"
        assert after.last_grounded_at is not None
        assert after.occupant_text == (
            "Checkout stalls on constrained mobile networks"
        )
        history = [rev.text for rev in
                   store.list_occupant_revisions(ws.id, card.id)]
        assert "Checkout stalls on mobile" in history
        assert "Checkout stalls on constrained mobile networks" in history


def test_position_rows_survive_vacating_and_are_never_deleted(tmp_path):
    """§1.6 — unfold vacates the frame position; history is retained."""
    engine, a, _b, frame = _laddered_engine()
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_positions(ws.id, engine.state())
        engine.unfold(frame.id)
        store.sync_positions(ws.id, engine.state())

        vacated = store.get_position(ws.id, frame.id)
        assert vacated is not None
        assert vacated.status == "vacated"
        assert vacated.confirmed_by == "andrew"      # provenance retained
        assert store.get_position(ws.id, a.id).status == "live"
        assert store.get_position(ws.id, a.id).folded_under is None


def test_sync_positions_is_idempotent(tmp_path):
    engine, _a, _b, _frame = _laddered_engine()
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        first = store.sync_positions(ws.id, engine.state())
        second = store.sync_positions(ws.id, engine.state())
        assert [row.position_id for row in first] == [
            row.position_id for row in second
        ]
        assert [row.created_at for row in first] == [
            row.created_at for row in second
        ]
        revisions = store.list_occupant_revisions(ws.id, first[0].position_id)
        assert len(revisions) == 1


def test_stale_positions_surface_never_grounded_structure_first(tmp_path):
    engine, a, _b, _frame = _laddered_engine()
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        engine.resolve(a.id, "supported", receipt="https://example.test/a")
        store.sync_positions(ws.id, engine.state())

        stale = store.stale_positions(ws.id, older_than=10_000.0)
        # An ungrounded slot has no last_grounded_at at all and must still
        # surface (§7.4): a frame that cannot state its own evidence is a
        # hypothesis and the display has to say so.
        assert stale[0].last_grounded_at is None
        assert a.id not in [
            row.position_id for row in
            store.stale_positions(ws.id, older_than=0.0)
        ]


def test_position_fingerprint_dedupe_finds_the_existing_position(tmp_path):
    """§4 / Appendix B #2 — the one job position ids cannot do alone."""
    engine = Engine(now=iter(range(1000, 2000)).__next__)
    card = engine.propose("Checkout stalls on mobile networks")
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_positions(ws.id, engine.state())
        found = store.find_position_by_fingerprint(
            ws.id, "  CHECKOUT stalls on   mobile networks! "
        )
        assert found is not None and found.position_id == card.id
        assert store.find_position_by_fingerprint(ws.id, "Unrelated") is None


# --------------------------------------------------------------------------
# §2.3 / §2.4 — click ledger and inbox indices
# --------------------------------------------------------------------------


def test_click_attempts_are_keyed_on_position_pairs_order_independently(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.record_click_attempt(ws.id, "c9", "c2", "no_click")
        # The same pair in the other order is the SAME row, not a second one.
        store.record_click_attempt(ws.id, "c2", "c9", "declined")
        rows = store.list_click_attempts(ws.id, position_a="c9", position_b="c2")
        assert len(rows) == 1
        assert rows[0]["position_a"] == "c2" and rows[0]["position_b"] == "c9"
        assert rows[0]["outcome"] == "declined"

        # Reconsideration is a new operation_version, not an overwrite: retry
        # is a paper-trailed door (§2.3).
        store.record_click_attempt(
            ws.id, "c2", "c9", "no_click", operation_version=2
        )
        assert len(
            store.list_click_attempts(ws.id, position_a="c2", position_b="c9")
        ) == 2


def test_click_ledger_and_inbox_sync_from_the_snapshot(tmp_path):
    engine, a, b, _frame = _laddered_engine()
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Ladder")
        store.sync_click_ledger(ws.id, engine.state())
        attempts = store.list_click_attempts(ws.id)
        assert any(row["outcome"] == "clicked" for row in attempts)
        # The accepted candidate is no longer open in the inbox.
        assert store.list_click_candidates(ws.id, status="open") == []
        accepted = store.list_click_candidates(ws.id, status="accepted")
        assert len(accepted) == 1
        assert accepted[0]["scope_boundary"]
        assert {accepted[0]["position_a"], accepted[0]["position_b"]} == {
            a.id, b.id
        }


# --------------------------------------------------------------------------
# §4 — export classes, suppression, dismissal
# --------------------------------------------------------------------------


def test_outbox_refuses_a_write_with_no_export_class(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        for bad in ("", "machine_fusion", "click_candidate"):
            with pytest.raises(ValueError, match="export_class"):
                store.enqueue_raven_remember(
                    "A machine fusion", export_class=bad, workspace_id=ws.id
                )
        item = store.enqueue_raven_remember(
            "A human root",
            export_class="human_root",
            workspace_id=ws.id,
        )
        assert item.export_class == "human_root"
        assert store.claim_raven_remembers()[0].export_class == "human_root"


def test_suppression_registry_is_reversible_and_never_deletes(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("A")
        store.suppress_memory("mem-stress", reason="stress-test residue")
        assert store.is_suppressed("mem-stress", workspace_id=ws.id) is True
        assert store.recall_is_blocked(ws.id, "mem-stress") is True
        assert store.unsuppress_memory("mem-stress") is True
        assert store.is_suppressed("mem-stress", workspace_id=ws.id) is False


def test_suppression_can_be_scoped_to_one_workspace(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        a = store.create_workspace("A")
        b = store.create_workspace("B")
        store.suppress_memory("mem-1", workspace_id=a.id, reason="local noise")
        assert store.is_suppressed("mem-1", workspace_id=a.id) is True
        assert store.is_suppressed("mem-1", workspace_id=b.id) is False


def test_dismissal_is_durable_by_memory_id(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        a = store.create_workspace("A")
        b = store.create_workspace("B")
        store.dismiss_memory(a.id, "mem-1", reason="not relevant here")
        assert store.is_dismissed(a.id, "mem-1") is True
        assert store.recall_is_blocked(a.id, "mem-1") is True
        # Dismissal is workspace-local: another workspace still sees it.
        assert store.is_dismissed(b.id, "mem-1") is False
        assert [d.raven_memory_id for d in store.list_dismissals(a.id)] == [
            "mem-1"
        ]


# --------------------------------------------------------------------------
# §5 — the conservative migration path
# --------------------------------------------------------------------------


def _legacy_snapshot():
    """A Magpie-era snapshot: a receipted synthesis over two archived parents."""
    return {
        "question": "Why is checkout slow?",
        "cards": [
            {"id": "c1", "text": "Checkout stalls on mobile", "kind": "claim",
             "state": "open", "archived": True, "parents": []},
            {"id": "c2", "text": "Signup stalls on mobile", "kind": "claim",
             "state": "open", "archived": True, "parents": []},
            {"id": "c3", "text": "Mobile flows stall", "kind": "synthesis",
             "state": "supported", "receipt": "https://example.test/trace",
             "archived": False, "parents": ["c1", "c2"]},
            {"id": "c4", "text": "Something unverified", "kind": "synthesis",
             "state": "supported", "receipt": "", "archived": False,
             "parents": ["c1", "c2"]},
        ],
    }


def test_backfill_never_converts_a_legacy_synthesis_into_a_frame(tmp_path):
    """§5 — the single most important migration rule.

    Combination provenance is not identity recognition. Fabricating `supports`
    edges from `parents` would seed the ladder with exactly the structure the
    click gates exist to prevent.
    """
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Legacy", snapshot=_legacy_snapshot())
        result = store.backfill_positions(ws.id)

        row = store.get_position(ws.id, "c3")
        assert row.floor_kind == "claim"        # NOT 'frame'
        assert row.supports == []               # NOT fabricated from parents
        assert row.provenance == ["c1", "c2"]   # provenance edges, preserved
        assert row.support_state == "needs_human"
        assert row.receipt is None

        # It surfaces in the one-time migration inbox for human confirmation.
        assert [entry["position_id"] for entry in result["migration_inbox"]] == [
            "c3"
        ]
        assert result["migration_inbox"][0]["provenance"] == ["c1", "c2"]


def test_backfill_unarchives_parents_the_old_law_destroyed(tmp_path):
    """§5 — recovering material `resolve()`'s archiving branch consumed."""
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Legacy", snapshot=_legacy_snapshot())
        result = store.backfill_positions(ws.id)

        assert result["recovered_parents"] == ["c1", "c2"]
        cards = {
            card["id"]: card
            for card in store.load_workspace(ws.id).snapshot["cards"]
        }
        assert cards["c1"]["archived"] is False
        assert cards["c2"]["archived"] is False
        # Recovered, but left UNFOLDED pending human confirmation.
        assert cards["c1"].get("folded_under") is None


def test_backfill_reverts_a_receiptless_synthesis_to_needs_human(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Legacy", snapshot=_legacy_snapshot())
        result = store.backfill_positions(ws.id)

        assert result["reverted_syntheses"] == ["c4"]
        cards = {
            card["id"]: card
            for card in store.load_workspace(ws.id).snapshot["cards"]
        }
        assert cards["c4"]["state"] == "needs_human"
        assert not cards["c4"]["receipt"]
        # It is not in the migration inbox: there is nothing to confirm.
        assert "c4" not in [e["position_id"] for e in result["migration_inbox"]]


def test_backfill_seeds_the_never_retry_ledger_from_history(tmp_path):
    """§5 — migration itself seeds the never-retry memory (§2.3)."""
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Legacy", snapshot=_legacy_snapshot())
        result = store.backfill_positions(ws.id)

        # c3 and c4 are historical retries of the SAME pair; they collapse to
        # one row, because memory must be separate from output.
        assert result["seeded_attempts"] == 1
        rows = store.list_click_attempts(ws.id)
        assert len(rows) == 1
        assert (rows[0]["position_a"], rows[0]["position_b"]) == ("c1", "c2")
        assert rows[0]["outcome"] == "no_click"
        assert rows[0]["operation_version"] == 1


def test_backfill_is_idempotent(tmp_path):
    with Storage(tmp_path / "db.sqlite3") as store:
        ws = store.create_workspace("Legacy", snapshot=_legacy_snapshot())
        first = store.backfill_positions(ws.id)
        second = store.backfill_positions(ws.id)
        assert first["migration_inbox"] == second["migration_inbox"]
        assert first["seeded_attempts"] == second["seeded_attempts"]
        assert len(store.list_click_attempts(ws.id)) == 1
        # The second pass finds nothing left to recover or revert.
        assert second["recovered_parents"] == []
        assert second["reverted_syntheses"] == []


def test_v3_database_migrates_additively_to_the_position_schema(tmp_path):
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
            CREATE TABLE raven_remember_outbox (
                id TEXT PRIMARY KEY,
                workspace_id TEXT,
                dedupe_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                raven_memory_id TEXT,
                created_at REAL NOT NULL,
                available_at REAL NOT NULL,
                claimed_at REAL,
                completed_at REAL,
                error TEXT
            );
            INSERT INTO workspaces(
                id,name,question,snapshot_json,metadata_json,context_version,
                created_at,updated_at
            ) VALUES ('ws_old','Old','','{}','{}',0,1,1);
            INSERT INTO raven_remember_outbox(
                id,workspace_id,dedupe_key,payload_json,created_at,available_at
            ) VALUES ('rout_1','ws_old','k','{"content":"legacy"}',1,1);
            PRAGMA user_version = 3;
            """
        )

    with Storage(path) as store:
        assert store.load_workspace("ws_old").name == "Old"
        # A pre-existing outbox row is backfilled to `human_root`, never
        # invented into a curated frame: calling it curated would fabricate
        # the human confirmation §5 requires.
        [legacy] = store.claim_raven_remembers()
        assert legacy.export_class == "human_root"

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "positions",
            "position_supports",
            "occupant_revisions",
            "click_attempts",
            "click_candidates",
            "suppression_registry",
            "dismissals",
        } <= tables
