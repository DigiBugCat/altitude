from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from magpie import server
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
    server._bank_card(workspace.id, card, source="atomize")

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
    assert card["state"] == "open"
    assert "Raven" in card["foot"]
    assert fake.remember_calls == []
    assert store.get_raven_projection(workspace.id, "mem-recalled").local_ref == card["id"]
    duplicate = server._adopt_raven_memory(workspace.id, "mem-recalled")
    assert duplicate["already_adopted"] is True
    assert duplicate["card"]["id"] == card["id"]

    # Dismissal changes only the local exposure. It does not call Raven.
    dismissed = server._dismiss_raven_memory(workspace.id, "mem-recalled")
    assert dismissed["exposure"]["status"] == "dismissed"
    assert fake.remember_calls == []


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
    assert adopted["card"]["state"] == "open"
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


def test_collision_is_remembered_with_parent_raven_ids(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    a = engine.propose("Parent A")
    b = engine.propose("Parent B")
    server._bank_card(workspace.id, a, source="atomize")
    server._bank_card(workspace.id, b, source="atomize")
    for item in store.claim_raven_remembers(limit=2):
        server._deliver_raven_remember(item)

    child = engine.collide(a.id, b.id)
    child = engine.update_proposal(child.id, "Derived child", kind="synthesis")
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    server._bank_card(workspace.id, child, source="fusion")
    [item] = store.claim_raven_remembers()
    server._deliver_raven_remember(item)

    assert fake.remember_calls[-1]["content"] == "Derived child"
    assert fake.remember_calls[-1]["hints"] == {
        "derived_from": ["mem-1", "mem-2"]
    }
    assert "source" not in fake.remember_calls[-1]


def test_collision_keeps_both_parent_bindings_after_raven_dedupes_content(
    raven_workspace,
    monkeypatch,
):
    fake, store, workspace, engine = raven_workspace

    def canonical_remember(content: str, **kwargs) -> RavenResult:
        fake.remember_calls.append({"content": content, **kwargs})
        memory_id = (
            "mem-canonical" if content == "Same parent" else "mem-derived"
        )
        return RavenResult(ok=True, value={"ok": True, "id": memory_id})

    monkeypatch.setattr(fake, "remember", canonical_remember)
    a = engine.propose("Same parent")
    b = engine.propose("Same parent")
    server._bank_card(workspace.id, a, source="atomize")
    server._bank_card(workspace.id, b, source="atomize")
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

    child = engine.collide(a.id, b.id)
    child = engine.update_proposal(child.id, "Derived child", kind="synthesis")
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    server._bank_card(workspace.id, child, source="fusion")
    [item] = store.claim_raven_remembers()
    server._deliver_raven_remember(item)

    assert fake.remember_calls[-1]["hints"] == {
        "derived_from": ["mem-canonical", "mem-canonical"]
    }
    assert store.claim_raven_remembers() == []


def test_parent_wait_stops_at_retry_ceiling(raven_workspace):
    fake, store, workspace, engine = raven_workspace
    orphan = engine.propose("Orphaned child")
    orphan.parents = ["missing-parent"]
    store.save_workspace(workspace.id, engine.state(), question=engine.question)
    server._bank_card(workspace.id, orphan, source="fusion")
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
