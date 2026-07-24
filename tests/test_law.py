"""The law, end to end: recognition organizes; receipts settle.

These regression tests prevent model output — successful or otherwise — from
being laundered into a permanent verdict, and prevent a click from consuming
the ground it stands on.

No network. The provider chain is replaced with fakes throughout.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magpie import providers  # noqa: E402
from magpie.engine import ClickProposal, Engine, GateFailure  # noqa: E402
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


@pytest.fixture
def restore_chain():
    yield
    providers.reset_chain(None)


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


def _proposal(**over):
    base = dict(
        abstraction="Every retry pathway silently duplicates downstream effects",
        specializer_a="when the duplicate arrives over the payment channel",
        specializer_b="when the duplicate arrives through the email sender",
        scope_boundary="does not cover read-only queries, which are idempotent",
    )
    base.update(over)
    return ClickProposal(**base)


def _pair(engine):
    """Two instances plus the §7.1 budget a click costs."""
    for i in range(5):
        engine.propose(f"funding contribution {i}", section="field")
    a = engine.propose("payment retries duplicate charges", section="field")
    b = engine.propose("email retries duplicate sends", section="field")
    return a, b


# ---------------- inference can never settle anything ----------------


def test_a_confirmed_click_settles_nothing(restore_chain):
    """THE regression, restated for Altitude.

    The old law let a supported synthesis archive its parents. Under §1.6 a
    click creates an OPEN frame with no receipt and folds its instances: the
    ground survives, and nothing anywhere became true.
    """
    e = Engine(cap=12)
    a, b = _pair(e)
    cand = e.propose_click(a.id, b.id, _proposal())
    frame = e.confirm_click(cand.id, confirmed_by="andrew")

    assert frame.occupant.state == "open"
    assert frame.receipt is None
    assert not e.cards[a.id].archived
    assert not e.cards[b.id].archived
    settled = [c for c in e.cards.values() if c.state in ("supported", "refuted")]
    assert settled == []
    assert not any(l["kind"] == "RESOLVED" for l in e.ledger)


def test_no_gate_passing_proposal_can_reach_the_field_without_a_human(restore_chain):
    """§2.4 — the inbox is the only door, and it needs a human verdict."""
    e = Engine(cap=12)
    a, b = _pair(e)
    before = {c.id for c in e.live()}

    e.propose_click(a.id, b.id, _proposal())

    assert {c.id for c in e.live()} == before
    assert not any(p.floor_kind == "frame" for p in e.all_positions())


def test_a_failed_gate_emits_nothing_at_all(restore_chain):
    """§1.3 — a failed gate is not a card."""
    e = Engine(cap=12)
    a, b = _pair(e)
    before = set(e.cards)

    with pytest.raises(GateFailure):
        e.propose_click(a.id, b.id, _proposal(
            abstraction="both concern retries and duplicate",
        ))

    assert set(e.cards) == before
    assert e.click_candidates == {}


def test_a_provider_outage_leaves_no_settled_positions_at_all(restore_chain):
    """Drive the scanner over a dead chain: nothing may settle, ever.

    The scanner returns a pair to *ask about*, never a card, so an outage
    produces exactly nothing — and records `failed`, which under §2.3 does not
    consume the pair.
    """
    providers.reset_chain(Chain([DeadProvider()]))
    e = Engine(cap=12)
    a, b = _pair(e)

    for _ in range(3):
        pair = e.scan_candidates(lambda x, y: 0.9)
        if not pair:
            break
        # The provider is down, so the worker returns nothing to gate.
        e.record_attempt(pair[0], pair[1], "failed")

    settled = [c for c in e.cards.values() if c.state in ("supported", "refuted")]
    assert settled == []
    assert not any(p.floor_kind == "frame" for p in e.all_positions())
    assert e.pair_consumed(a.id, b.id) is False


def test_an_empty_abstraction_is_coerced_to_no_click(restore_chain):
    """§2.2 fail-closed — a positive click with empty text emits nothing."""
    e = Engine(cap=12)
    a, b = _pair(e)

    with pytest.raises(GateFailure, match="generativity"):
        e.propose_click(a.id, b.id, _proposal(abstraction="   "))

    assert e.click_candidates == {}
    assert e.pair_consumed(a.id, b.id)


def test_a_receipt_on_the_floor_is_the_only_thing_that_moves_a_frame(restore_chain):
    """§1.5 — frames re-score from below and store nothing of their own."""
    e = Engine(cap=12)
    a, b = _pair(e)
    cand = e.propose_click(a.id, b.id, _proposal())
    frame = e.confirm_click(cand.id, confirmed_by="andrew")

    assert e.frame_support(frame.id)["supported"] == 0

    with pytest.raises(ValueError, match="only claims"):
        e.resolve(frame.id, "supported", "a receipt aimed at the wrong floor")

    e.resolve(a.id, "supported", "ledger line 4412 shows the double charge")
    assert e.frame_support(frame.id)["supported"] == 1
    assert frame.receipt is None


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
