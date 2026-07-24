from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from magpie import server
from magpie.engine import ClickProposal
from magpie.raven_client import RavenResult


@dataclass
class FakeRaven:
    enabled: bool = True

    def __post_init__(self):
        self.remember_calls: list[dict] = []
        self.recall_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def remember(self, content: str, **kwargs) -> RavenResult:
        self.remember_calls.append({"content": content, **kwargs})
        return RavenResult(
            ok=True,
            value={"ok": True, "id": f"mem-{len(self.remember_calls)}"},
        )

    def recall(self, query: str, **kwargs) -> RavenResult:
        self.recall_calls.append({"query": query, **kwargs})
        return RavenResult(
            ok=True,
            value={
                "ok": True,
                "query": query,
                "results": [
                    {
                        "id": "mem-recalled",
                        "content": "A remembered constraint",
                        "kind": "thought",
                        "state": "active",
                        "effective_confidence": 0.81,
                    }
                ],
            },
        )

    def get(self, memory_id: str, **kwargs) -> RavenResult:
        self.get_calls.append({"memory_id": memory_id, **kwargs})
        return RavenResult(ok=False, error="not expected")


@pytest.fixture
def raven_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "magpie.sqlite3")
    store, workspace, engine = server._initialize_storage()
    fake = FakeRaven()
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "WORKSPACE_ID", workspace.id)
    monkeypatch.setattr(server, "ENGINE", engine)
    monkeypatch.setattr(server, "RAVEN", fake)
    monkeypatch.setattr(server, "RAVEN_LAST_ERROR", None)
    try:
        yield fake, store, workspace, engine
    finally:
        store.close()
        server.STORE = None
        server.ENGINE = None
        server.WORKSPACE_ID = None


def test_card_remember_is_queued_then_bound_to_a_projection(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    card = engine.propose("Human observation")
    server._bank_card(workspace.id, card, source="atomize", engine=engine)

    [item] = store.claim_raven_remembers()
    server._deliver_raven_remember(item)

    assert fake.remember_calls == [
        {
            "content": "Human observation",
            "source": "human",
            "tags": ["magpie", f"workspace:{workspace.id}"],
            "episode_id": workspace.id,
        }
    ]
    projection = store.get_raven_projection(workspace.id, "mem-1")
    assert projection is not None
    assert projection.local_ref == card.id
    assert projection.local_status == "adopted"


def test_recall_adopt_and_dismiss_remain_workspace_local(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    engine.seed("What constraint matters?")
    store.save_workspace(
        workspace.id,
        engine.state(),
        question=engine.question,
        increment_context=True,
    )

    recalled = server._recall_workspace(workspace.id)
    assert recalled["suggestions"][0]["memory_id"] == "mem-recalled"
    assert fake.recall_calls[0]["query"] == "What constraint matters?"

    adopted = server._adopt_raven_memory(workspace.id, "mem-recalled")
    card = adopted["card"]
    # §4 — adopted material arrives QUARANTINED, not as ordinary open field
    # material: `needs_human` plus `external=True` keeps it out of the click
    # scanner until a human pins it. Recall can inform without breeding.
    assert card["state"] == "needs_human"
    position = server.ENGINE.position(card["id"])
    assert position.external is True
    assert position.origin == "recall"
    assert position.pinned_by_human is False
    assert server.ENGINE.scan_eligible(card["id"]) is False
    assert "Raven" in card["foot"]
    assert fake.remember_calls == []
    assert store.get_raven_projection(workspace.id, "mem-recalled").local_ref == card["id"]
    duplicate = server._adopt_raven_memory(workspace.id, "mem-recalled")
    assert duplicate["already_adopted"] is True
    assert duplicate["card"]["id"] == card["id"]

    # Dismissal changes only local state. It does not call Raven.
    dismissed = server._dismiss_raven_memory(workspace.id, "mem-recalled")
    assert dismissed["exposure"]["status"] == "dismissed"
    assert fake.remember_calls == []
    # §4 — and it is durable by memory id, so recall scoring cannot resurface
    # it later.
    assert store.is_dismissed(workspace.id, "mem-recalled") is True
    assert server._recall_workspace(workspace.id)["suggestions"] == []


def test_remote_epistemic_state_never_settles_an_adopted_card(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    fake.recall = lambda *_args, **_kwargs: RavenResult(
        ok=True,
        value={
            "ok": True,
            "results": [
                {
                    "id": "mem-resolved",
                    "content": "Remote says resolved",
                    "kind": "prediction",
                    "state": "resolved",
                    "effective_confidence": 0.99,
                }
            ],
        },
    )
    server._recall_workspace(workspace.id, "resolved")
    adopted = server._adopt_raven_memory(workspace.id, "mem-resolved")
    # Remote confidence 0.99 and remote state "resolved" buy nothing here: the
    # receipt law does not accept another system's word for evidence. The card
    # lands quarantined and unreceipted.
    assert adopted["card"]["state"] == "needs_human"
    assert not adopted["card"].get("receipt")


def test_keep_move_and_kill_update_only_projection_layout(raven_workspace):
    fake, store, workspace, _engine = raven_workspace
    server._recall_workspace(workspace.id, "constraint")
    adopted = server._adopt_raven_memory(workspace.id, "mem-recalled")
    card_id = adopted["card"]["id"]

    server._api_keep({"workspace_id": workspace.id, "id": card_id})
    kept = store.get_raven_projection(workspace.id, "mem-recalled")
    assert kept.pinned is True

    server._api_move(
        {
            "workspace_id": workspace.id,
            "id": card_id,
            "section": "evidence",
        }
    )
    moved = store.get_raven_projection(workspace.id, "mem-recalled")
    assert moved.section == "evidence"

    server._api_kill({"workspace_id": workspace.id, "id": card_id})
    hidden = store.get_raven_projection(workspace.id, "mem-recalled")
    assert hidden.hidden is True
    assert fake.remember_calls == []


def _confirm_a_frame(engine, a_id: str, b_id: str, *, text: str):
    """Drive the §2.4 inbox to a confirmed click, budget and gates honoured."""
    # §7.1 — one confirmed click per 5 human contributions, engine-enforced.
    while engine.click_budget_remaining() <= 0:
        engine.propose("An unrelated human contribution")
    candidate = engine.propose_click(
        a_id,
        b_id,
        ClickProposal(
            abstraction=text,
            specializer_a="in the case of the first instance",
            specializer_b="in the case of the second instance",
            scope_boundary="excludes questions about desktop throughput",
        ),
    )
    return engine.confirm_click(candidate.id, confirmed_by="andrew")


def test_confirmed_frame_is_banked_with_its_floor_raven_ids(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    a = engine.propose("Checkout stalls on mobile networks")
    b = engine.propose("Signup stalls on mobile networks")
    server._bank_card(workspace.id, a, source="atomize", engine=engine)
    server._bank_card(workspace.id, b, source="atomize", engine=engine)
    for item in store.claim_raven_remembers(limit=2):
        server._deliver_raven_remember(item)

    # §4 — a human-confirmed frame banks as `human_curated_frame`, and the
    # existing outbox parent-waiting makes the Raven DAG a ladder for free.
    # The frame's floor is its `supports`, which is what the export carries.
    frame = _confirm_a_frame(
        engine, a.id, b.id, text="Radio handover interrupts long requests"
    )
    frame.occupant.parents = list(frame.supports)
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    server._bank_card(
        workspace.id, frame.occupant, source="click", engine=engine
    )
    [item] = store.claim_raven_remembers()
    assert item.export_class == "human_curated_frame"
    server._deliver_raven_remember(item)

    assert (
        fake.remember_calls[-1]["content"]
        == "Radio handover interrupts long requests"
    )
    assert fake.remember_calls[-1]["hints"] == {
        "derived_from": ["mem-1", "mem-2"]
    }
    assert "source" not in fake.remember_calls[-1]


def test_frame_keeps_both_floor_bindings_after_raven_dedupes_content(
    raven_workspace,
    monkeypatch,
):
    fake, store, workspace, engine = raven_workspace

    def canonical_remember(content: str, **kwargs) -> RavenResult:
        fake.remember_calls.append({"content": content, **kwargs})
        memory_id = (
            "mem-canonical"
            if content == "Requests stall on mobile networks"
            else "mem-derived"
        )
        return RavenResult(ok=True, value={"ok": True, "id": memory_id})

    monkeypatch.setattr(fake, "remember", canonical_remember)
    a = engine.propose("Requests stall on mobile networks")
    b = engine.propose("Requests stall on mobile networks")
    server._bank_card(workspace.id, a, source="atomize", engine=engine)
    server._bank_card(workspace.id, b, source="atomize", engine=engine)
    for item in store.claim_raven_remembers(limit=2):
        server._deliver_raven_remember(item)

    # The projection is canonical by remote id and can retain only one local
    # ref, while completed outbox rows retain both local ancestry bindings.
    projection = store.get_raven_projection(workspace.id, "mem-canonical")
    assert projection.local_ref in {a.id, b.id}
    assert (
        store.completed_raven_memory_id_for_local_ref(workspace.id, a.id)
        == "mem-canonical"
    )
    assert (
        store.completed_raven_memory_id_for_local_ref(workspace.id, b.id)
        == "mem-canonical"
    )

    frame = _confirm_a_frame(
        engine, a.id, b.id, text="Radio handover interrupts long requests"
    )
    frame.occupant.parents = list(frame.supports)
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    server._bank_card(
        workspace.id, frame.occupant, source="click", engine=engine
    )
    [item] = store.claim_raven_remembers()
    server._deliver_raven_remember(item)

    assert fake.remember_calls[-1]["hints"] == {
        "derived_from": ["mem-canonical", "mem-canonical"]
    }
    assert store.claim_raven_remembers() == []


def test_machine_material_is_never_queued_for_raven(raven_workspace):
    """§4 — the contamination fix at its chokepoint.

    The old `_bank_card` enqueued a Raven write for every card, so runaway
    machine cards reached shared memory and were recalled back later. An
    unconfirmed frame and a derived-ungrounded claim now bank locally and
    queue nothing.
    """
    fake, store, workspace, engine = raven_workspace
    a = engine.propose("Checkout stalls on mobile networks")
    b = engine.propose("Signup stalls on mobile networks")
    frame = _confirm_a_frame(
        engine, a.id, b.id, text="Radio handover interrupts long requests"
    )
    [derived] = engine.derive(
        frame.id,
        [{"text": "p95 mobile latency exceeds 800ms",
          "falsification": "a week of p95 under 400ms"}],
    )
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    store.claim_raven_remembers(limit=20)  # drain anything already queued

    # An unconfirmed frame has no export class at all.
    engine.position(frame.id).confirmed_by = None
    server._bank_card(
        workspace.id, frame.occupant, source="click", engine=engine
    )
    # Derivation fills structure; it does not assert, so it never banks.
    server._bank_card(workspace.id, derived, source="derive", engine=engine)

    assert store.claim_raven_remembers() == []
    assert fake.remember_calls == []
    # The local idea bank still records all of it: workspace-local provenance
    # is not what §4 restricts.
    local = {
        item["idea"].text for item in store.list_workspace_ideas(workspace.id)
    }
    assert "p95 mobile latency exceeds 800ms" in local


def test_parent_wait_stops_at_retry_ceiling(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    orphan = engine.propose("Orphaned root")
    orphan.parents = ["missing-parent"]
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    server._bank_card(workspace.id, orphan, source="atomize", engine=engine)
    item = store.claim_raven_remembers()[0]

    # Delivery sees the attempt count after the claim increment.
    item = replace(item, attempts=server.RAVEN_MAX_ATTEMPTS)
    server._deliver_raven_remember(item)

    assert fake.remember_calls == []
    assert store.claim_raven_remembers() == []
    with store._lock:
        row = store._conn.execute(
            "SELECT status,error FROM raven_remember_outbox WHERE id=?",
            (item.id,),
        ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "waiting for parent Raven memories"


# --------------------------------------------------------------------------
# §4 — recall arrives AT an altitude (Delta 2, binding)
# --------------------------------------------------------------------------


def test_recall_reconstructs_a_quarantined_sub_ladder(raven_workspace, monkeypatch):
    """§4 — a banked frame comes back as a ladder, not as loose floor-0 cards.

    The frame was banked with `hints={"derived_from": [floor memory ids]}`, so
    the structure is recoverable: adoption rebuilds the floor first, then
    stands the frame on it with support edges and grounding dates intact.
    """
    fake, store, workspace, engine = raven_workspace
    floors = {
        "mem-floor-a": {
            "id": "mem-floor-a",
            "content": "Checkout stalls on mobile networks",
            "kind": "thought",
            "last_grounded_at": 1500.0,
        },
        "mem-floor-b": {
            "id": "mem-floor-b",
            "content": "Signup stalls on mobile networks",
            "kind": "thought",
            "last_grounded_at": 1600.0,
        },
    }
    frame_memory = {
        "id": "mem-frame",
        "content": "Radio handover interrupts long requests",
        "kind": "thought",
        "state": "active",
        "effective_confidence": 0.7,
        "hints": {"derived_from": ["mem-floor-a", "mem-floor-b"]},
    }

    monkeypatch.setattr(
        fake,
        "recall",
        lambda *_a, **_k: RavenResult(
            ok=True, value={"ok": True, "results": [frame_memory]}
        ),
    )
    monkeypatch.setattr(
        fake,
        "get",
        lambda memory_id, **_k: RavenResult(
            ok=True, value={"ok": True, "node": floors[memory_id]}
        )
        if memory_id in floors
        else RavenResult(ok=False, error="not found"),
    )

    server._recall_workspace(workspace.id, "mobile")
    adopted = server._adopt_raven_memory(workspace.id, "mem-frame")

    frame_position = server.ENGINE.position(adopted["card"]["id"])
    # It arrives AT an altitude: a frame standing on its reconstructed floor.
    assert frame_position.floor_kind == "frame"
    assert len(frame_position.supports) == 2
    assert server.ENGINE.altitude(frame_position.id) == 1

    floor_positions = [
        server.ENGINE.position(sid) for sid in frame_position.supports
    ]
    assert {p.occupant.text for p in floor_positions} == {
        "Checkout stalls on mobile networks",
        "Signup stalls on mobile networks",
    }
    # Grounding dates arrive intact — the floor's history is what makes the
    # reconstructed ladder honest rather than freshly minted.
    assert sorted(p.last_grounded_at for p in floor_positions) == [1500.0, 1600.0]

    # The WHOLE unit is quarantined: excluded from the scanner until pinned.
    for position in [frame_position, *floor_positions]:
        assert position.external is True
        assert position.origin == "recall"
        assert server.ENGINE.scan_eligible(position.id) is False

    # And the frame asserts nothing: support is computed from the floor, which
    # has no receipts, so it reads as visibly speculative.
    support = server.ENGINE.frame_support(frame_position.id)
    assert support["supported"] == 0
    assert support["open"] == 2
    assert support["speculative"] is True
    assert support["summary"] == "0✓ 0✗ 2○"
    assert fake.remember_calls == []


def test_readopting_a_memory_dedupes_onto_the_existing_position(
    raven_workspace, monkeypatch
):
    """§4 / Appendix B #2 — adoption dedupes by fingerprint.

    Without this, a re-adopted memory could mint a fresh position and walk
    around the never-retry ledger, which keys on position ids.
    """
    fake, store, workspace, engine = raven_workspace
    server._recall_workspace(workspace.id, "constraint")
    first = server._adopt_raven_memory(workspace.id, "mem-recalled")
    before = len(server.ENGINE.positions)

    # A second memory id carrying the SAME content must not mint a position.
    monkeypatch.setattr(
        fake,
        "recall",
        lambda *_a, **_k: RavenResult(
            ok=True,
            value={
                "ok": True,
                "results": [
                    {
                        "id": "mem-twin",
                        "content": "A remembered constraint",
                        "kind": "thought",
                        "state": "active",
                        "effective_confidence": 0.4,
                    }
                ],
            },
        ),
    )
    server._recall_workspace(workspace.id, "constraint again")
    twin = server._adopt_raven_memory(workspace.id, "mem-twin")

    assert twin["card"]["id"] == first["card"]["id"]
    assert len(server.ENGINE.positions) == before
