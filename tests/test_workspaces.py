import json
import threading

import pytest

from magpie import server, workers


@pytest.fixture
def workspace_server(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "magpie.sqlite3")
    monkeypatch.setattr(server, "STORE", None)
    monkeypatch.setattr(server, "ENGINE", None)
    monkeypatch.setattr(server, "WORKSPACE_ID", None)

    store, workspace, engine = server._initialize_storage()
    server.STORE = store
    server.WORKSPACE_ID = workspace.id
    server.ENGINE = engine
    yield server

    store.close()
    server.STORE = None
    server.ENGINE = None
    server.WORKSPACE_ID = None


def test_empty_database_creates_resumable_default_workspace(workspace_server):
    srv = workspace_server
    current = srv._api_workspace_current({})["workspace"]
    assert current["name"] == "Untitled workspace"
    assert current["current"] is True
    assert srv._api_state({})["workspace"]["id"] == current["id"]


def test_create_open_and_resume_workspace_snapshot(workspace_server):
    srv = workspace_server
    original_id = srv.WORKSPACE_ID
    srv._api_seed({"workspace_id": original_id, "question": "Original question"})
    original_card = srv.ENGINE.propose("Original idea")
    with srv.LOCK:
        srv._bank_card(original_id, original_card, source="manual-test")
        srv._persist(increment_context=True)

    created = srv._api_workspace_create(
        {"name": "Fresh workspace", "question": "Fresh question"}
    )
    fresh_id = created["workspace"]["id"]
    assert fresh_id != original_id
    assert srv.ENGINE.question == "Fresh question"
    assert srv.ENGINE.live() == []

    srv.ENGINE.propose("Fresh idea")
    with srv.LOCK:
        srv._persist(increment_context=True)
    srv._api_workspace_open({"id": original_id})

    assert srv.ENGINE.question == "Original question"
    assert [card.text for card in srv.ENGINE.live()] == ["Original idea"]
    assert srv.STORE.current_workspace_id() == original_id

    srv._api_workspace_open({"id": fresh_id})
    assert [card.text for card in srv.ENGINE.live()] == ["Fresh idea"]


def test_background_atomize_writes_to_launch_workspace_after_switch(
    workspace_server, monkeypatch
):
    srv = workspace_server
    workspace_a = srv.WORKSPACE_ID
    monkeypatch.setattr(
        workers,
        "atomize",
        lambda text, sections: [
            {"text": "Result for workspace A", "section": sections[0], "foot": "test"}
        ],
    )
    workspace_b = srv._api_workspace_create({"name": "B"})["workspace"]["id"]

    # Simulate an inference request launched in A finishing after B was opened.
    srv._run_atomize("input from A", workspace_a)

    assert srv.WORKSPACE_ID == workspace_b
    assert srv.ENGINE.live() == []
    stored_a = srv.STORE.load_workspace(workspace_a)
    assert [card["text"] for card in stored_a.snapshot["cards"]] == [
        "Result for workspace A"
    ]


def test_explicit_http_mutations_cannot_cross_the_global_current_workspace(
    workspace_server,
):
    srv = workspace_server
    workspace_a = srv.WORKSPACE_ID
    workspace_b = srv._api_workspace_create({"name": "B"})["workspace"]["id"]
    errors = []

    def seed(workspace_id, question):
        try:
            srv._api_seed({"workspace_id": workspace_id, "question": question})
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=seed, args=(workspace_a, "Question A")),
        threading.Thread(target=seed, args=(workspace_b, "Question B")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert srv.WORKSPACE_ID == workspace_b
    assert srv._api_state({"workspace_id": workspace_a})["question"] == "Question A"
    assert srv._api_state({"workspace_id": workspace_b})["question"] == "Question B"
    with pytest.raises(ValueError, match="workspace_id required"):
        srv._api_seed({"question": "ambiguous"})


def test_same_idea_across_workspaces_shares_bank_entry(
    workspace_server, monkeypatch
):
    srv = workspace_server
    workspace_a = srv.WORKSPACE_ID
    monkeypatch.setattr(
        workers,
        "atomize",
        lambda text, sections: [
            {"text": "Mobile validation errors", "section": sections[0], "foot": "test"}
        ],
    )
    srv._run_atomize("first wording", workspace_a)
    workspace_b = srv._api_workspace_create({"name": "B"})["workspace"]["id"]
    srv._run_atomize("second wording", workspace_b)

    a_idea = srv.STORE.list_workspace_ideas(workspace_a)[0]["idea"]
    b_idea = srv.STORE.list_workspace_ideas(workspace_b)[0]["idea"]
    assert a_idea.id == b_idea.id
    assert srv.STORE.bank_revision() == 1


def test_atomize_consolidates_repeat_into_typed_thematic_occurrence(
    workspace_server, monkeypatch
):
    srv = workspace_server
    workspace_id = srv.WORKSPACE_ID
    calls = 0

    def thematic_atomize(text, sections, existing_cards=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{
                "text": "Workspaces are views into memory",
                "section": "Memory boundaries",
                "artifact_type": "preference",
                "relation": "new",
                "foot": "test",
            }]
        assert existing_cards and existing_cards[0]["id"] == "c1"
        return [{
            "text": "Workspaces are views into memory",
            "section": "Memory boundaries",
            "artifact_type": "preference",
            "relation": "repeat",
            "canonical_id": "c1",
            "foot": "test",
        }]

    monkeypatch.setattr(workers, "atomize", thematic_atomize)
    srv._run_atomize("first wording", workspace_id)
    srv._run_atomize("same idea again", workspace_id)

    state = srv._api_state({"workspace_id": workspace_id})
    live = [card for card in state["cards"] if not card["archived"]]
    assert len(live) == 1
    assert live[0]["artifact_type"] == "preference"
    assert live[0]["occurrence_count"] == 2
    assert live[0]["section"] == "memory-boundaries"
    assert state["digest"]["recurring_ideas"][0]["id"] == "c1"
    assert state["digest"]["themes"][0]["name"] == "Memory boundaries"

    occurrences = srv.STORE.list_workspace_ideas(workspace_id)
    assert len(occurrences) == 2
    assert {item["local_ref"] for item in occurrences} == {"c1"}


def test_verification_submission_failure_reopens_launch_workspace(
    workspace_server, monkeypatch
):
    srv = workspace_server
    workspace_a = srv.WORKSPACE_ID
    card_a = srv.ENGINE.propose("A needs verification")
    with srv.LOCK:
        srv._persist(increment_context=True)

    class SwitchingHook:
        def submit(self, request):
            assert request.workspace_id == workspace_a
            srv._api_workspace_create({"name": "B"})
            srv.ENGINE.propose("B card with the same local id")
            with srv.LOCK:
                srv._persist(increment_context=True)
            raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(srv, "VERIFICATION_HOOK", SwitchingHook())
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        srv._api_workspace_open({"id": workspace_a})
        srv._api_verify({"workspace_id": workspace_a, "id": card_a.id})

    workspace_b = srv.WORKSPACE_ID
    assert workspace_b != workspace_a
    assert srv.ENGINE.cards[card_a.id].text == "B card with the same local id"
    assert srv.ENGINE.cards[card_a.id].state == "open"
    saved_a = srv.STORE.load_workspace(workspace_a)
    restored_a = server.engine_mod.Engine.from_state(saved_a.snapshot)
    assert restored_a.cards[card_a.id].state == "open"
    assert restored_a.cards[card_a.id].foot == "verification submission failed"


def test_legacy_state_imports_once_and_is_left_untouched(monkeypatch, tmp_path):
    legacy = tmp_path / "state.json"
    snapshot = {
        "question": "Legacy question",
        "cap": 12,
        "seq": 1,
        "sections": [{"key": "field", "name": "FIELD", "color": "#fff"}],
        "cards": [
            {
                "id": "c1",
                "kind": "claim",
                "text": "Legacy idea",
                "section": "field",
                "mass": 0.42,
                "state": "open",
                "receipt": None,
                "foot": "",
                "pinned": False,
                "born": 1.0,
                "parents": [],
                "archived": False,
            }
        ],
        "weights": {},
        "ledger": [],
    }
    legacy.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(server, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_PATH", legacy)
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "magpie.sqlite3")

    store, workspace, engine = server._initialize_storage()
    try:
        assert workspace.name == "Imported workspace"
        assert engine.question == "Legacy question"
        assert len(store.list_workspaces()) == 1
        assert legacy.exists()
    finally:
        store.close()

    reopened, workspace2, _engine2 = server._initialize_storage()
    try:
        assert workspace2.id == workspace.id
        assert len(reopened.list_workspaces()) == 1
    finally:
        reopened.close()
