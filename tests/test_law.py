"""The law, end to end: inference proposes; receipts settle.

These regression tests prevent model output — successful or otherwise — from
being laundered into a permanent verdict that archives its parent cards.

No network. The provider chain is replaced with fakes throughout.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magpie import providers, workers  # noqa: E402
from magpie.engine import Engine  # noqa: E402
from magpie.providers import Chain, ProviderUnavailable  # noqa: E402


class DeadProvider:
    """Every backend down."""

    name = "dead"

    def available(self):
        return True

    def complete(self, prompt, schema=None, timeout=30):
        raise ProviderUnavailable("down")


class EmptyProvider:
    """Answers, but with nothing usable in the text field."""

    name = "empty"

    def __init__(self, payload):
        self.payload = payload

    def available(self):
        return True

    def complete(self, prompt, schema=None, timeout=30):
        return self.payload


class GoodProvider:
    name = "good"

    def available(self):
        return True

    def complete(self, prompt, schema=None, timeout=30):
        return {"kind": "SYNTHESIS", "text": "a real fused claim"}


@pytest.fixture
def restore_chain():
    yield
    providers.reset_chain(None)


# ---------------- workers.fuse fails closed ----------------


def test_fuse_reports_not_ok_when_no_provider_answers(restore_chain):
    providers.reset_chain(Chain([DeadProvider()]))
    out = workers.fuse({"text": "a"}, {"text": "b"}, "q")
    assert out["ok"] is False
    assert "unavailable" in out["provenance"]


@pytest.mark.parametrize("payload", [
    {"kind": "SYNTHESIS", "text": ""},
    {"kind": "SYNTHESIS", "text": "   "},
    {"kind": "SYNTHESIS"},
    {},
])
def test_fuse_reports_not_ok_on_empty_text(payload, restore_chain):
    providers.reset_chain(Chain([EmptyProvider(payload)]))
    out = workers.fuse({"text": "a"}, {"text": "b"}, "q")
    assert out["ok"] is False


def test_fuse_reports_ok_with_a_real_answer(restore_chain):
    providers.reset_chain(Chain([GoodProvider()]))
    out = workers.fuse({"text": "a"}, {"text": "b"}, "q")
    assert out["ok"] is True
    assert out["text"] == "a real fused claim"
    assert out["provenance"] == "proposed by good"


def test_fuse_falls_through_a_dead_backend_to_a_live_one(restore_chain):
    providers.reset_chain(Chain([DeadProvider(), GoodProvider()]))
    out = workers.fuse({"text": "a"}, {"text": "b"}, "q")
    assert out["ok"] is True
    assert out["provenance"] == "proposed by good"


# ---------------- the server fuse path honours it ----------------


@pytest.fixture
def srv(monkeypatch, tmp_path):
    """The server module wired to a temp runtime dir and a fresh engine."""
    from magpie import server

    monkeypatch.setattr(server, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(server, "ENGINE", Engine(cap=12))
    monkeypatch.setattr(server, "STORE", None)
    monkeypatch.setattr(server, "WORKSPACE_ID", "ws_test")
    return server


def _collided(server):
    a = server.ENGINE.propose("a", section="field")
    b = server.ENGINE.propose("b", section="field")
    child = server.ENGINE.collide(a.id, b.id)
    return a, b, child


def test_provider_outage_never_settles_the_child(srv, monkeypatch):
    """THE regression: fusion unavailable must NOT produce a supported card."""
    a, b, child = _collided(srv)
    monkeypatch.setattr(workers, "fuse", lambda *_: {
        "ok": False, "text": "unresolved collision: a / b",
        "kind": "TENSION",
        "provenance": "fusion unavailable · no provider answered",
    })
    srv._run_fuse(child.id, {"text": "a"}, {"text": "b"}, "q")

    c = srv.ENGINE.cards[child.id]
    assert c.state == "open", "a failed inference job must leave no stuck job state"
    assert c.receipt is None
    assert c.kind == "claim"
    # and the parents must survive — they were never superseded by anything
    assert not srv.ENGINE.cards[a.id].archived
    assert not srv.ENGINE.cards[b.id].archived


def test_worker_exception_never_settles_the_child(srv, monkeypatch):
    a, b, child = _collided(srv)

    def boom(*_):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(workers, "fuse", boom)
    srv._run_fuse(child.id, {"text": "a"}, {"text": "b"}, "q")

    assert srv.ENGINE.cards[child.id].state == "open"
    assert not srv.ENGINE.cards[a.id].archived


def test_blank_provenance_still_cannot_settle_the_child(srv, monkeypatch):
    a, b, child = _collided(srv)
    monkeypatch.setattr(workers, "fuse", lambda *_: {
        "ok": True, "text": "something", "kind": "SYNTHESIS", "provenance": "   ",
    })
    srv._run_fuse(child.id, {"text": "a"}, {"text": "b"}, "q")
    assert srv.ENGINE.cards[child.id].state == "open"
    assert srv.ENGINE.cards[child.id].receipt is None


def test_successful_fusion_creates_open_proposal_without_receipt(srv, monkeypatch):
    a, b, child = _collided(srv)
    monkeypatch.setattr(workers, "fuse", lambda *_: {
        "ok": True, "text": "the fused claim", "kind": "SYNTHESIS",
        "provenance": "proposed by good",
    })
    srv._run_fuse(child.id, {"text": "a"}, {"text": "b"}, "q")

    c = srv.ENGINE.cards[child.id]
    assert c.state == "open"
    assert c.text == "the fused claim"
    assert c.receipt is None
    assert c.kind == "synthesis"
    assert "proposed by good" in c.foot
    assert not any(l["kind"] == "RESOLVED" for l in srv.ENGINE.ledger)
    assert not srv.ENGINE.cards[a.id].archived
    assert not srv.ENGINE.cards[b.id].archived


def test_a_full_outage_leaves_no_settled_cards_at_all(srv, monkeypatch, restore_chain):
    """Drive the real worker over a dead chain: nothing may settle."""
    providers.reset_chain(Chain([DeadProvider()]))
    for _ in range(3):
        pair = srv.ENGINE.best_pair()
        if not pair:
            break
        child = srv.ENGINE.collide(*pair)
        srv._run_fuse(child.id, {"text": "a"}, {"text": "b"}, "q")
    settled = [c for c in srv.ENGINE.cards.values()
               if c.state in ("supported", "refuted")]
    assert settled == []


# ---------------- future verifier hook is explicit and inert by default ----


def test_verify_requires_an_installed_runtime_hook(srv):
    c = srv.ENGINE.propose("needs checking")
    with pytest.raises(ValueError, match="not configured"):
        srv._api_verify({"workspace_id": "ws_test", "id": c.id})
    assert c.state == "open"


def test_verify_hook_receives_bounded_request_without_settling(srv, monkeypatch):
    c = srv.ENGINE.propose("needs checking")

    class Hook:
        def __init__(self):
            self.request = None

        def submit(self, request):
            self.request = request
            return "job-1"

    hook = Hook()
    monkeypatch.setattr(srv, "VERIFICATION_HOOK", hook)
    out = srv._api_verify({"workspace_id": "ws_test", "id": c.id})
    assert out["job_id"] == "job-1"
    assert hook.request.card["id"] == c.id
    assert srv.ENGINE.cards[c.id].state == "testing"
    assert srv.ENGINE.cards[c.id].receipt is None


# ---------------- section colour is validated at the boundary ----------------


@pytest.mark.parametrize("bad", [
    'red" onmouseover="alert(1)',
    "#fff;background:url(javascript:alert(1))",
    "</style><script>alert(1)</script>",
    "not-a-color",
])
def test_section_rejects_non_hex_colour(srv, bad):
    with pytest.raises(ValueError, match="hex literal"):
        srv._api_section(
            {"workspace_id": "ws_test", "name": "S", "color": bad}
        )
    assert all(s.get("color") != bad for s in srv.ENGINE.sections.values())


@pytest.mark.parametrize("good", ["#fff", "#c9b8a0", "#ABCDEF"])
def test_section_accepts_hex_colour(srv, good):
    out = srv._api_section(
        {"workspace_id": "ws_test", "name": "S", "color": good}
    )
    assert out["color"] == good


def test_section_rejects_duplicate_field_name(srv):
    srv._api_section({"workspace_id": "ws_test", "name": "Questions"})
    with pytest.raises(ValueError, match="already exists"):
        srv._api_section({"workspace_id": "ws_test", "name": "Questions"})


def test_section_rename_updates_workspace_field(srv):
    created = srv._api_section(
        {"workspace_id": "ws_test", "name": "Questions"}
    )
    renamed = srv._api_section_rename(
        {
            "workspace_id": "ws_test",
            "key": created["key"],
            "name": "Open questions",
        }
    )

    assert renamed["name"] == "Open questions"
    assert srv.ENGINE.sections[created["key"]]["name"] == "Open questions"
