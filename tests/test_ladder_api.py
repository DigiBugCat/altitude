"""The ladder at the server and MCP boundaries (SPEC §1.4, §2.2–2.4, §3.2, §3.3).

Stage 3 wires the Position engine through the JSON API and the agent surface.
What these tests hold fixed is the shape of that seam:

  * the background loop is an inspector — it can expire, vacate, and *ask*,
    and it has no branch that creates field material;
  * the emergence inbox is the only door from inference to the field, and a
    human verdict is the only key;
  * a confirmed click still settles nothing, all the way out to the wire;
  * MCP can propose structure and can never settle it.

No network: the provider chain is a fake throughout.
"""

from __future__ import annotations

import pytest

from magpie import engine as engine_mod
from magpie import mcp_surface
from magpie import providers
from magpie import server
from magpie import workers
from magpie.providers import Chain


GOOD_RECOGNITION = {
    "click": True,
    "abstraction": "Every retry pathway silently duplicates downstream effects",
    "specializer_a": "when the duplicate arrives over the payment channel",
    "specializer_b": "when the duplicate arrives through the email sender",
    "scope_boundary": "does not cover read-only queries, which are idempotent",
}


class FakeProvider:
    """Returns a scripted payload; records what it was asked."""

    name = "fake"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def available(self):
        return True

    def complete(self, prompt, schema=None, timeout=30):
        self.calls += 1
        return self.payload


@pytest.fixture(autouse=True)
def restore_chain():
    yield
    providers.reset_chain(None)


@pytest.fixture
def srv(monkeypatch, tmp_path):
    """The server module on isolated storage, with no background threads."""
    monkeypatch.setattr(server, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "magpie.sqlite3")
    monkeypatch.setattr(server, "STORE", None)
    monkeypatch.setattr(server, "ENGINE", None)
    monkeypatch.setattr(server, "WORKSPACE_ID", None)
    monkeypatch.setattr(server, "_STARVATION_COUNTS", {})
    monkeypatch.setattr(server, "_FLEET_BUDGET_SPENT", 0)
    monkeypatch.setattr(server, "_FLEET_BUDGET_DAY", -1)

    store, workspace, engine = server._initialize_storage()
    server.STORE = store
    server.WORKSPACE_ID = workspace.id
    server.ENGINE = engine
    yield server
    store.close()
    server.STORE = None
    server.ENGINE = None
    server.WORKSPACE_ID = None


def _seed_pair(srv, workspace_id):
    """Two human instances plus the §7.1 budget a confirmed click costs."""
    engine, _ = server._engine_for_workspace(workspace_id)
    for index in range(5):
        engine.propose(f"funding contribution {index}", section="field")
    a = engine.propose("payment retries duplicate charges", section="field")
    b = engine.propose("email retries duplicate sends", section="field")
    with server.LOCK:
        server._persist_engine(workspace_id, engine)
    return a.id, b.id


def _confirm_a_frame(srv, workspace_id, confirmed_by="andrew"):
    """Drive one pair all the way to a confirmed frame through the API."""
    a_id, b_id = _seed_pair(srv, workspace_id)
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))
    proposed = srv._api_propose_click(
        {"workspace_id": workspace_id, "a": a_id, "b": b_id}
    )
    candidate_id = proposed["candidate"]["id"]
    resolved = srv._api_resolve_click({
        "workspace_id": workspace_id,
        "candidate_id": candidate_id,
        "verdict": "accept",
        "confirmed_by": confirmed_by,
    })
    return a_id, b_id, resolved


# ---------------- §2.1: the deleted operator stays deleted ----------------


def test_the_collide_route_and_helpers_are_gone(srv):
    """§2.1 — deleted outright, not feature-flagged, not aliased."""
    assert "/api/collide" not in srv.ROUTES
    assert not hasattr(srv, "_collide_and_fuse")
    assert not hasattr(srv, "_run_fuse")
    assert not hasattr(workers, "fuse")
    assert not hasattr(mcp_surface, "_collide")


# ---------------- §2.2: the loop is an inspector ----------------


def test_metabolism_cadence_and_gates_are_the_spec_values(srv):
    assert server.METABOLISM_PERIOD == 30.0
    assert server.QUIESCENCE_WINDOW == 60.0
    assert server.STARVATION_CLAIMS == 10
    assert engine_mod.CLICK_FLOOR == 0.62
    assert engine_mod.INBOX_CAP == 3


def test_inspection_never_generates_a_card(srv, monkeypatch):
    """§2.2 — the background loop materializes NOTHING.

    The strongest available statement of this: run a tick against a field of
    fresh claims with a provider standing by that would happily click, and
    assert the live set is byte-identical afterwards.
    """
    workspace_id = srv.WORKSPACE_ID
    _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    before = [card.to_dict() for card in engine.live()]
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))

    report = server._inspect_workspace(workspace_id)

    engine, _ = server._engine_for_workspace(workspace_id)
    assert [card.to_dict() for card in engine.live()] == before
    # And it did not even ask: the field is not quiescent.
    assert report["pair"] is None
    assert report["reason"] == "field not quiescent"


def test_quiescence_gate_blocks_a_scan_right_after_a_contribution(srv, monkeypatch):
    workspace_id = srv.WORKSPACE_ID
    _seed_pair(srv, workspace_id)

    report = server._inspect_workspace(workspace_id)

    assert report["reason"] == "field not quiescent"


def test_a_quiescent_field_still_does_nothing_without_embeddings(srv, monkeypatch):
    """§2.2 — with no ranking index every cosine reads 0.0, below CLICK_FLOOR.

    Ranking absence must degrade to silence, never to a different selection
    rule. This is what stops the scanner from inheriting `best_pair`'s habit of
    clearing its own threshold on every same-section pair.
    """
    workspace_id = srv.WORKSPACE_ID
    _seed_pair(srv, workspace_id)
    monkeypatch.setattr(server, "QUIESCENCE_WINDOW", 0.0)

    report = server._inspect_workspace(workspace_id)

    assert report["pair"] is None
    assert report["reason"] == "no eligible pair above CLICK_FLOOR"


def test_embeddings_rank_the_one_pair_and_the_floor_silences_the_rest(
    srv, monkeypatch
):
    """§2.2 — `idea_embeddings` (schema v1) finally gets its consumer.

    Embeddings are a bounded RANKING index: the close pair is selected, the
    orthogonal pair falls below CLICK_FLOOR and is not asked about at all, and
    at most one pair comes back per tick. A high cosine still triggers nothing
    on its own — it only decides which single question gets asked.
    """
    import struct

    workspace_id = srv.WORKSPACE_ID
    engine, _ = server._engine_for_workspace(workspace_id)
    for index in range(5):
        card = engine.propose(f"funding contribution {index}", section="field")
        server._bank_card(workspace_id, card, source="atomize", engine=engine)
    a = engine.propose("payment retries duplicate charges", section="field")
    b = engine.propose("email retries duplicate sends", section="field")
    far = engine.propose("the office plants need water", section="field")
    for card in (a, b, far):
        server._bank_card(workspace_id, card, source="atomize", engine=engine)
    with server.LOCK:
        server._persist_engine(workspace_id, engine)

    ideas = {
        item["local_ref"]: item["idea"].id
        for item in srv.STORE.list_workspace_ideas(workspace_id)
    }
    for card, vector in (
        (a, (1.0, 0.1, 0.0)),
        (b, (0.98, 0.2, 0.0)),
        (far, (0.0, 0.0, 1.0)),
    ):
        srv.STORE.put_embedding(
            ideas[card.id],
            model=server.EMBEDDING_MODEL,
            version=server.EMBEDDING_VERSION,
            dimensions=3,
            vector=struct.pack("<3f", *vector),
        )

    ranker = server._embedding_ranker(workspace_id, engine)
    assert ranker(a.id, b.id) > engine_mod.CLICK_FLOOR
    assert ranker(a.id, far.id) < engine_mod.CLICK_FLOOR

    monkeypatch.setattr(server, "QUIESCENCE_WINDOW", 0.0)
    report = server._inspect_workspace(workspace_id)

    assert sorted(report["pair"]) == sorted([a.id, b.id])


def test_anti_starvation_forces_one_wave_through_a_never_quiet_field(
    srv, monkeypatch
):
    """§2.2 — budget exhaustion or a busy field can never disable recognition."""
    workspace_id = srv.WORKSPACE_ID
    engine, _ = server._engine_for_workspace(workspace_id)
    for index in range(server.STARVATION_CLAIMS):
        engine.propose(f"human claim {index}", section="field")
    with server.LOCK:
        server._persist_engine(workspace_id, engine)

    # The field is emphatically NOT quiescent — everything just arrived.
    report = server._inspect_workspace(workspace_id)

    assert report["reason"] != "field not quiescent"
    # The wave is consumed: the next tick falls back to the quiescence gate.
    assert server._inspect_workspace(workspace_id)["reason"] == (
        "field not quiescent"
    )


def test_a_full_inbox_stops_emission_regardless_of_timer(srv, monkeypatch):
    """§2.2/§2.4 — emission is governed by inbox capacity, not tick frequency."""
    workspace_id = srv.WORKSPACE_ID
    _seed_pair(srv, workspace_id)
    monkeypatch.setattr(server, "QUIESCENCE_WINDOW", 0.0)
    engine, _ = server._engine_for_workspace(workspace_id)
    for index in range(engine_mod.INBOX_CAP):
        left = engine.propose(f"left instance {index}", section="field")
        right = engine.propose(f"right instance {index}", section="field")
        engine.propose_click(
            left.id,
            right.id,
            engine_mod.ClickProposal(
                abstraction=(
                    f"Concurrency invariants erode under sustained pressure {index}"
                ),
                specializer_a="observed on the ingest path",
                specializer_b="observed on the egress path",
                scope_boundary="excludes single-writer batch jobs",
            ),
        )
    with server.LOCK:
        server._persist_engine(workspace_id, engine)

    report = server._inspect_workspace(workspace_id)

    assert report["reason"] == "inbox full"
    assert report["pair"] is None


def test_the_fleet_budget_caps_recognition_spend(srv, monkeypatch):
    workspace_id = srv.WORKSPACE_ID
    _seed_pair(srv, workspace_id)
    monkeypatch.setattr(server, "QUIESCENCE_WINDOW", 0.0)
    monkeypatch.setattr(server, "FLEET_CLICK_BUDGET_PER_DAY", 0)

    report = server._inspect_workspace(workspace_id)

    assert report["reason"] == "fleet budget exhausted"


def test_inspection_expires_stale_candidates_and_vacates_empty_frames(srv):
    """The two structural chores are the only field changes a tick may make."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]
    engine, _ = server._engine_for_workspace(workspace_id)
    # Retire both instances: the frame is now a semantic shell.
    engine.positions[a_id].status = "retired"
    engine.positions[b_id].status = "retired"
    with server.LOCK:
        server._persist_engine(workspace_id, engine)

    report = server._inspect_workspace(workspace_id)

    assert report["vacated"] == 1
    engine, _ = server._engine_for_workspace(workspace_id)
    assert engine.positions[frame_id].status == "vacated"


# ---------------- §2.2/§2.3: the recognition path ----------------


def test_a_no_click_writes_a_consuming_attempt_and_no_card(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    before = {card.id for card in engine.live()}
    providers.reset_chain(Chain([FakeProvider({
        "click": False, "abstraction": "", "specializer_a": "",
        "specializer_b": "", "scope_boundary": "",
    })]))

    result = server._run_recognize(a_id, b_id, workspace_id)

    engine, _ = server._engine_for_workspace(workspace_id)
    assert result["click"] is False
    assert {card.id for card in engine.live()} == before
    assert engine.pair_consumed(a_id, b_id)


def test_a_provider_outage_records_failed_and_does_not_consume_the_pair(srv):
    """§2.2/§2.3 — an outage must never permanently suppress a recognition."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)

    class Exploding:
        name = "exploding"

        def available(self):
            return True

        def complete(self, prompt, schema=None, timeout=30):
            raise RuntimeError("boom")

    providers.reset_chain(Chain([Exploding()]))

    result = server._run_recognize(a_id, b_id, workspace_id)

    engine, _ = server._engine_for_workspace(workspace_id)
    assert result["failed"] is True
    assert not engine.pair_consumed(a_id, b_id)


def test_a_failed_gate_emits_nothing_and_consumes_the_pair(srv):
    """§1.3 — a failed gate is not a card. It writes a row and nothing else."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    before = {card.id for card in engine.live()}
    # A lexical merge of both instances: fails generativity.
    providers.reset_chain(Chain([FakeProvider({
        "click": True,
        "abstraction": "retries duplicate charges and sends",
        "specializer_a": "payment",
        "specializer_b": "email",
        "scope_boundary": "not reads",
    })]))

    result = server._run_recognize(a_id, b_id, workspace_id)

    engine, _ = server._engine_for_workspace(workspace_id)
    assert result["gate_failed"] == "generativity"
    assert {card.id for card in engine.live()} == before
    assert engine.open_candidates() == []
    assert engine.pair_consumed(a_id, b_id)


def test_propose_click_lands_in_the_inbox_never_in_the_field(srv):
    """§2.1/§2.4 — what `/api/collide` became."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    before = {card.id for card in engine.live()}
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))

    result = srv._api_propose_click(
        {"workspace_id": workspace_id, "a": a_id, "b": b_id}
    )

    engine, _ = server._engine_for_workspace(workspace_id)
    assert result["click"] is True
    assert {card.id for card in engine.live()} == before
    pending = srv._api_pending_clicks({"workspace_id": workspace_id})
    assert [c["id"] for c in pending["candidates"]] == [result["candidate"]["id"]]
    # The inbox shows both instances verbatim (§2.4).
    assert {item["id"] for item in pending["candidates"][0]["instances"]} == {
        a_id, b_id
    }


def test_propose_click_refuses_a_pair_the_ledger_already_settled(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    engine.record_attempt(a_id, b_id, "no_click")
    with server.LOCK:
        server._persist_engine(workspace_id, engine)

    with pytest.raises(ValueError, match="already attempted"):
        srv._api_propose_click(
            {"workspace_id": workspace_id, "a": a_id, "b": b_id}
        )


def test_reconsider_pair_reopens_the_door_with_a_paper_trail(srv):
    """§2.3 — retry is a deliberate act at operation_version + 1."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    engine.record_attempt(a_id, b_id, "declined")
    with server.LOCK:
        server._persist_engine(workspace_id, engine)

    result = srv._api_reconsider_pair(
        {"workspace_id": workspace_id, "a": a_id, "b": b_id}
    )

    engine, _ = server._engine_for_workspace(workspace_id)
    assert result["attempt"]["operation_version"] == 2
    assert not engine.pair_consumed(a_id, b_id)
    # The prior verdict survives; reconsideration adds, it does not erase.
    rows = srv.STORE.list_click_attempts(
        workspace_id, position_a=a_id, position_b=b_id
    )
    assert {row["operation_version"]: row["outcome"] for row in rows} == {
        1: "declined", 2: "reconsidered",
    }


# ---------------- §2.4/§1.6: the human verdict ----------------


def test_accepting_a_click_creates_an_open_frame_and_folds_the_instances(srv):
    """§1.6 — the receipt law at the wire boundary.

    A confirmed click means "organize these together", never "this is true":
    the frame comes back OPEN with no receipt, its support reads entirely from
    the floor, and both instances survive folded rather than archived.
    """
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id, resolved = _confirm_a_frame(srv, workspace_id)

    frame = resolved["frame"]
    assert frame["floor_kind"] == "frame"
    assert frame["occupant"]["state"] == "open"
    assert frame["occupant"]["receipt"] is None
    assert frame["confirmed_by"] == "andrew"
    assert frame["support"]["summary"] == "0✓ 0✗ 2○"
    assert frame["altitude"] == 1
    assert sorted(resolved["folded"]) == sorted([a_id, b_id])

    engine, _ = server._engine_for_workspace(workspace_id)
    for instance_id in (a_id, b_id):
        position = engine.positions[instance_id]
        assert position.status == "folded"
        assert position.folded_under == frame["id"]
        assert not position.occupant.archived


def test_accepting_requires_a_named_human(srv):
    """§1.6 — the confirmation is provenance; unattributed provenance is none."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))
    proposed = srv._api_propose_click(
        {"workspace_id": workspace_id, "a": a_id, "b": b_id}
    )

    with pytest.raises(ValueError, match="confirmed_by required"):
        srv._api_resolve_click({
            "workspace_id": workspace_id,
            "candidate_id": proposed["candidate"]["id"],
            "verdict": "accept",
        })


def test_accept_with_edit_makes_the_human_wording_the_occupant(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))
    proposed = srv._api_propose_click(
        {"workspace_id": workspace_id, "a": a_id, "b": b_id}
    )

    resolved = srv._api_resolve_click({
        "workspace_id": workspace_id,
        "candidate_id": proposed["candidate"]["id"],
        "verdict": "accept",
        "confirmed_by": "andrew",
        "text": "At-least-once delivery leaks into user-visible side effects",
    })

    assert resolved["frame"]["occupant"]["text"] == (
        "At-least-once delivery leaks into user-visible side effects"
    )
    assert resolved["frame"]["occupant"]["state"] == "open"


def test_declining_writes_the_attempt_and_leaves_the_field_alone(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))
    proposed = srv._api_propose_click(
        {"workspace_id": workspace_id, "a": a_id, "b": b_id}
    )
    engine, _ = server._engine_for_workspace(workspace_id)
    before = {card.id for card in engine.live()}

    srv._api_resolve_click({
        "workspace_id": workspace_id,
        "candidate_id": proposed["candidate"]["id"],
        "verdict": "decline",
    })

    engine, _ = server._engine_for_workspace(workspace_id)
    assert {card.id for card in engine.live()} == before
    assert engine.pair_consumed(a_id, b_id)


def test_a_receipt_on_the_floor_rescores_the_frame_through_the_api(srv):
    """§1.5 — support is computed from below on every read, at the wire too."""
    workspace_id = srv.WORKSPACE_ID
    a_id, _b_id, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]
    assert resolved["frame"]["support"]["summary"] == "0✓ 0✗ 2○"

    srv._api_judge({"workspace_id": workspace_id, "id": a_id, "verdict": "yes"})

    descended = srv._api_descend(
        {"workspace_id": workspace_id, "position_id": frame_id}
    )
    assert descended["frame"]["support"]["summary"] == "1✓ 0✗ 1○"
    # The frame itself still carries no receipt of its own.
    assert descended["frame"]["occupant"]["receipt"] is None
    assert "support_state" not in descended["frame"]


def test_unfold_releases_the_instances_and_vacates_the_frame(srv):
    """§1.6 — always cheap; the position record persists with its history."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]

    released = srv._api_unfold({"workspace_id": workspace_id, "frame_id": frame_id})

    assert sorted(item["id"] for item in released["released"]) == sorted([a_id, b_id])
    engine, _ = server._engine_for_workspace(workspace_id)
    assert engine.positions[frame_id].status == "vacated"
    assert frame_id in engine.positions          # never deleted
    for instance_id in (a_id, b_id):
        assert engine.positions[instance_id].status == "live"
        assert engine.positions[instance_id].folded_under is None


# ---------------- §1.4: derivation, downward ----------------


def test_derive_creates_ungrounded_slots_beneath_the_frame(srv):
    workspace_id = srv.WORKSPACE_ID
    _a_id, _b_id, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]
    providers.reset_chain(Chain([FakeProvider({
        "claims": [
            {"text": "Retry handlers lack idempotency keys",
             "falsification": "a handler reading a dedupe key"},
            {"text": "Downstream effects are not transactional",
             "falsification": "a single-commit trace"},
        ]
    })]))

    result = srv._api_derive({"workspace_id": workspace_id, "frame_id": frame_id})

    assert len(result["created"]) == 2
    assert len(result["ungrounded_slots"]) == 2
    engine, _ = server._engine_for_workspace(workspace_id)
    for card in result["created"]:
        position = engine.positions[card["id"]]
        assert position.origin == "derivation"
        assert position.occupant.state == "open"
        assert position.last_grounded_at is None      # a visibly empty socket
        assert card["id"] in engine.positions[frame_id].supports
    # Derivation asserts nothing: the frame is still speculative.
    assert engine.frame_support(frame_id)["speculative"] is True


def test_derived_claims_are_scanner_ineligible_until_grounded(srv):
    """§1.4/§7.4 — derivation fills structure; it does not assert."""
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]
    providers.reset_chain(Chain([FakeProvider({
        "claims": [{"text": "Retry handlers lack idempotency keys",
                    "falsification": "a handler reading a dedupe key"}]
    })]))

    result = srv._api_derive({"workspace_id": workspace_id, "frame_id": frame_id})

    engine, _ = server._engine_for_workspace(workspace_id)
    slot_id = result["created"][0]["id"]
    assert engine.scan_eligible(slot_id) is False


def test_derived_claims_are_never_queued_to_raven(srv):
    """§1.4/§4 — derived-ungrounded claims have no export class."""
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]
    providers.reset_chain(Chain([FakeProvider({
        "claims": [{"text": "Retry handlers lack idempotency keys",
                    "falsification": "a handler reading a dedupe key"}]
    })]))

    result = srv._api_derive({"workspace_id": workspace_id, "frame_id": frame_id})

    slot_id = result["created"][0]["id"]
    queued = srv.STORE.claim_raven_remembers(limit=50)
    assert all(
        f":{slot_id}" not in item.dedupe_key for item in queued
    )


def test_derive_refuses_a_claim_position(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, _b_id = _seed_pair(srv, workspace_id)

    with pytest.raises(ValueError, match="frames only"):
        srv._api_derive({"workspace_id": workspace_id, "frame_id": a_id})


def test_the_background_loop_can_neither_derive_nor_propose(srv):
    """§1.4/§2.2 — background-initiated derivation is excluded by design.

    Stated as a bytecode invariant rather than a behavioural one: the assertion
    is that the inspector has no branch *reaching* generation, not merely that
    no such branch happened to fire under this fixture. Names loaded by the two
    metabolism functions are exactly the operations they may perform.
    """
    forbidden = {
        "derive", "_run_derive", "propose", "_bank_card", "confirm_click",
        "resolve", "judge", "record_occurrence", "update_proposal",
    }
    for function in (server._inspect_workspace, server._metabolism_loop):
        names = set(function.__code__.co_names)
        assert not (names & forbidden), (function.__name__, names & forbidden)
    # And what it MAY do, positively stated.
    inspector = set(server._inspect_workspace.__code__.co_names)
    assert {"expire_candidates", "vacate_empty_frames", "scan_candidates"} <= (
        inspector
    )


# ---------------- §3.1/§3.2: navigation ----------------


def test_get_field_defaults_to_the_top_and_filters_by_altitude(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]

    top = srv._api_field({"workspace_id": workspace_id})
    ground = srv._api_field({"workspace_id": workspace_id, "altitude": 0})

    assert top["altitude"] == top["max_altitude"] == 1
    assert [p["id"] for p in top["positions"]] == [frame_id]
    ground_ids = {p["id"] for p in ground["positions"]}
    assert frame_id not in ground_ids
    # Folded instances are hidden at every altitude, but never archived (§1.6).
    assert a_id not in ground_ids and b_id not in ground_ids


def test_every_returned_position_carries_supports_and_grounding(srv):
    """§3.2 — supports, support summary, and last_grounded_at arrive intact."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id, resolved = _confirm_a_frame(srv, workspace_id)

    top = srv._api_field({"workspace_id": workspace_id})
    frame = top["positions"][0]

    assert sorted(frame["supports"]) == sorted([a_id, b_id])
    assert frame["support"]["summary"] == "0✓ 0✗ 2○"
    assert "last_grounded_at" in frame
    assert frame["instance_count"] == 2


def test_descend_returns_the_floor_with_folded_instances_present(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id, resolved = _confirm_a_frame(srv, workspace_id)

    descended = srv._api_descend(
        {"workspace_id": workspace_id, "position_id": resolved["frame"]["id"]}
    )

    assert sorted(item["id"] for item in descended["floor"]) == sorted([a_id, b_id])
    assert all(item["status"] == "folded" for item in descended["floor"])
    assert all(item["altitude"] == 0 for item in descended["floor"])


# ---------------- §3.3: the brief ----------------


def test_harvest_returns_a_capped_brief_not_a_dump(srv):
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)

    result = srv._api_harvest({"workspace_id": workspace_id, "max_items": 2})
    brief = result["brief"]

    assert set(brief) >= {
        "spine", "cracks", "stale", "cruxes", "decisions", "constraints",
        "experiments", "unresolved", "changed",
    }
    # §3.3 — full state is `state()`, a debugging surface. The brief is not it.
    assert "cards" not in brief
    for section in ("spine", "cracks", "stale", "cruxes", "unresolved"):
        assert len(brief[section]) <= 2, section


def test_harvest_at_an_altitude_selects_the_spine(srv):
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]

    result = srv._api_harvest({"workspace_id": workspace_id, "altitude": 1})
    brief = result["brief"]

    assert brief["altitude"] == 1
    assert [item["id"] for item in brief["spine"]] == [frame_id]
    assert brief["max_altitude"] == 1


def test_harvest_surfaces_ungrounded_slots_as_cruxes(srv):
    """§3.3 — the receipts most worth going and getting."""
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]
    providers.reset_chain(Chain([FakeProvider({
        "claims": [{"text": "Retry handlers lack idempotency keys",
                    "falsification": "a handler reading a dedupe key"}]
    })]))
    derived = srv._api_derive({"workspace_id": workspace_id, "frame_id": frame_id})
    slot_id = derived["created"][0]["id"]

    brief = srv._api_harvest({"workspace_id": workspace_id})["brief"]

    crux = next(item for item in brief["cruxes"] if item["id"] == slot_id)
    assert crux["ungrounded"] is True
    assert crux["under_frame"] == frame_id


def test_harvest_flags_a_cracked_frame(srv):
    """§3.3 — the ladder's honesty section."""
    workspace_id = srv.WORKSPACE_ID
    a_id, _b_id, resolved = _confirm_a_frame(srv, workspace_id)
    frame_id = resolved["frame"]["id"]

    srv._api_judge({"workspace_id": workspace_id, "id": a_id, "verdict": "no"})

    brief = srv._api_harvest({"workspace_id": workspace_id})["brief"]
    cracked = next(item for item in brief["cracks"] if item["id"] == frame_id)
    assert cracked["support_summary"] == "0✓ 1✗ 1○"


# ---------------- §3.2: the MCP surface ----------------


def test_mcp_propose_click_cannot_reach_the_field(srv):
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    engine, _ = server._engine_for_workspace(workspace_id)
    before = {card.id for card in engine.live()}
    providers.reset_chain(Chain([FakeProvider(GOOD_RECOGNITION)]))

    result = mcp_surface._propose_click(workspace_id, a_id, b_id)

    engine, _ = server._engine_for_workspace(workspace_id)
    assert result["click"] is True
    assert {card.id for card in engine.live()} == before


def test_mcp_has_no_way_to_confirm_a_click_or_settle_a_claim(srv):
    """§2.4/§3.2 — the verdicts are human-only, absent from MCP entirely.

    Asserted against the registered tool catalog itself, because that is the
    surface an agent actually sees. `guide()` claims these are human-only; this
    checks the claim is true rather than aspirational.
    """
    registered: list[str] = []

    class CatalogRecorder:
        def tool(self, **_kwargs):
            def decorate(fn):
                registered.append(fn.__name__)
                return fn
            return decorate

    mcp_surface.register_tools(CatalogRecorder())

    assert "propose_click" in registered
    for human_only in (
        "resolve_click", "confirm_click", "judge", "resolve", "verify", "kill",
        "collide",
    ):
        assert human_only not in registered
    assert not hasattr(mcp_surface, "_resolve_click")
    assert not hasattr(mcp_surface, "_confirm_click")


def test_mcp_get_field_routes_by_workspace_and_carries_altitude(srv):
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)
    other = srv._api_workspace_create({"name": "Elsewhere"})["workspace"]["id"]

    field = mcp_surface._get_field(workspace_id)

    assert field["workspace"]["id"] == workspace_id
    assert field["altitude"] == 1
    assert [p["id"] for p in field["positions"]] == [resolved["frame"]["id"]]
    # Creating a workspace moved the browser; the MCP read did not follow it.
    assert server.WORKSPACE_ID == other


def test_mcp_pending_clicks_can_show_near_misses_on_request(srv):
    """§7.2/Appendix B #4 — private metrics WITH an inspection surface."""
    workspace_id = srv.WORKSPACE_ID
    a_id, b_id = _seed_pair(srv, workspace_id)
    providers.reset_chain(Chain([FakeProvider({
        "click": True,
        "abstraction": "retries duplicate charges and sends",
        "specializer_a": "payment",
        "specializer_b": "email",
        "scope_boundary": "not reads",
    })]))
    server._run_recognize(a_id, b_id, workspace_id)

    quiet = mcp_surface._pending_clicks(workspace_id)
    loud = mcp_surface._pending_clicks(workspace_id, include_rejected=True)

    assert quiet["candidates"] == []
    assert "rejected" not in quiet
    assert [row["outcome"] for row in loud["rejected"]] == ["gate_failed"]


def test_mcp_harvest_accepts_an_altitude(srv):
    workspace_id = srv.WORKSPACE_ID
    _a, _b, resolved = _confirm_a_frame(srv, workspace_id)

    result = mcp_surface._harvest(workspace_id, altitude=1, max_items=3)

    assert result["brief"]["altitude"] == 1
    assert [item["id"] for item in result["brief"]["spine"]] == [
        resolved["frame"]["id"]
    ]
