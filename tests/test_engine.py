"""Altitude engine — the law under SPEC-ALTITUDE §1 and §2.

Every assertion here is the NEW law. Where an old assertion was deleted it was
because the spec mandated the semantic change, not because the test was
inconvenient; the deletions are enumerated in the changelog.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magpie.engine import (  # noqa: E402
    CLICK_BUDGET_PER_CONTRIBUTIONS,
    CLICK_TTL,
    DERIVE_CAP,
    INBOX_CAP,
    ClickProposal,
    Engine,
    GateFailure,
    content_words,
    load,
    save,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


@pytest.fixture
def eng():
    return Engine(cap=12, now=Clock())


def _fund_clicks(engine, n=1):
    """§7.1: a click costs 5 human contributions. Buy the budget honestly."""
    for i in range(CLICK_BUDGET_PER_CONTRIBUTIONS * n):
        engine.propose(f"funding contribution {i}", section="field")


def _good_proposal():
    return ClickProposal(
        abstraction="Every retry pathway silently duplicates downstream effects",
        specializer_a="when the duplicate arrives over the payment channel",
        specializer_b="when the duplicate arrives through the email sender",
        scope_boundary="does not cover read-only queries, which are idempotent",
    )


# ---------------- the receipt law ----------------


def test_resolve_requires_receipt(eng):
    c = eng.propose("a claim")
    for bad in ("", None, "   "):
        with pytest.raises(ValueError, match="receipt required"):
            eng.resolve(c.id, "supported", bad)
    assert c.state == "open"
    assert c.receipt is None


def test_resolve_sets_state_and_receipt(eng):
    c = eng.propose("a claim")
    out = eng.resolve(c.id, "supported", "because evidence")
    assert out.state == "supported"
    assert out.receipt == "because evidence"


def test_resolve_rejects_bad_verdict(eng):
    c = eng.propose("x")
    with pytest.raises(ValueError):
        eng.resolve(c.id, "maybe", "r")


def test_propose_cannot_set_terminal_states(eng):
    for s in ("supported", "refuted"):
        with pytest.raises(ValueError):
            eng.propose("x", state=s)


def test_no_other_path_sets_terminal_state(eng):
    """Only resolve/judge(yes|no) may produce supported|refuted."""
    a = eng.propose("a")
    b = eng.propose("b")
    eng.keep(a.id)
    eng.move(b.id, "field")
    eng.request_verify(eng.propose("c").id)
    eng.judge(eng.propose("d").id, "unknown")
    eng.kill(eng.propose("e").id)
    eng.enforce_cap()
    terminal = [c for c in eng.cards.values() if c.state in ("supported", "refuted")]
    assert terminal == []


def test_resolve_writes_text_in_the_same_transaction(eng):
    """The resolved wording must land WITH the receipt, not after it."""
    c = eng.propose("a", section="field")
    out = eng.resolve(c.id, "supported", "human verified source record",
                      text="the settled claim", foot="OBSERVED")
    assert out.text == "the settled claim"
    assert out.foot == "OBSERVED"
    assert out.receipt == "human verified source record"
    resolved = [l for l in eng.ledger if l["kind"] == "RESOLVED"]
    assert resolved[-1]["text"] == "supported · the settled claim"


def test_resolve_rejects_blank_text_override(eng):
    c = eng.propose("x")
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="non-empty"):
            eng.resolve(c.id, "supported", "r", text=bad)
    assert c.state == "open"


def test_resolve_without_text_leaves_wording_alone(eng):
    c = eng.propose("original")
    out = eng.resolve(c.id, "supported", "r")
    assert out.text == "original"


def test_resolve_stamps_last_grounded_at(eng):
    """§1.2 — a receipt is when the position was last grounded."""
    c = eng.propose("x")
    assert eng.position(c.id).last_grounded_at is None
    eng.resolve(c.id, "supported", "observed in the log")
    assert eng.position(c.id).last_grounded_at is not None


def test_resolve_never_archives_anything(eng):
    """SPEC §5, the single most important deletion: no parent-archiving branch.

    Replaces the old `test_supported_child_archives_parents_and_becomes_synthesis`.
    A click adds altitude; it must never consume ground.
    """
    a = eng.propose("a", section="field")
    b = eng.propose("b", section="field")
    c = eng.propose("c", section="field")
    c.parents = [a.id, b.id]  # legacy provenance shape
    eng.resolve(c.id, "supported", "verified observation")
    assert not eng.cards[a.id].archived
    assert not eng.cards[b.id].archived
    assert {x.id for x in eng.live()} == {a.id, b.id, c.id}
    assert not any(e["kind"] == "FUSED" for e in eng.ledger)


# ---------------- three structural floors (§1.1) ----------------


def test_frames_can_never_be_resolved(eng):
    """§1.1 — a frame is never directly supported or refuted."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    with pytest.raises(ValueError, match="only claims"):
        eng.resolve(frame.id, "supported", "a perfectly good receipt")
    with pytest.raises(ValueError, match="only claims"):
        eng.judge(frame.id, "yes")


def test_frames_have_no_stored_support_state(eng):
    """§1.5 — frames carry no stored support; reading one is an error."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    with pytest.raises(AttributeError, match="frame_support"):
        frame.support_state
    assert frame.receipt is None


def test_frame_support_is_computed_from_the_floor_every_time(eng):
    """§1.5 — nothing at any layer may drift from the layer below."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    assert eng.frame_support(frame.id)["summary"] == "0✓ 0✗ 2○"
    assert eng.frame_support(frame.id)["speculative"] is True

    eng.resolve(a.id, "supported", "ledger shows the double charge")
    assert eng.frame_support(frame.id)["summary"] == "1✓ 0✗ 1○"
    assert eng.frame_support(frame.id)["speculative"] is False

    eng.resolve(b.id, "refuted", "mail log shows exactly one send")
    support = eng.frame_support(frame.id)
    assert support["summary"] == "1✓ 1✗ 0○"
    assert support["cracked"] is True


def test_receipt_on_the_floor_propagates_grounding_upward(eng):
    """§1.5 — last_grounded_at = max over the floor."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")
    assert frame.last_grounded_at is None

    eng.resolve(a.id, "supported", "ledger shows the double charge")
    assert frame.last_grounded_at == eng.position(a.id).last_grounded_at


@pytest.mark.parametrize(
    "artifact_type",
    ["observation", "question", "preference", "constraint",
     "task", "experiment", "decision"],
)
def test_non_claim_artifacts_cannot_be_truth_judged_or_resolved(eng, artifact_type):
    c = eng.propose("not a truth-apt claim", artifact_type=artifact_type)

    with pytest.raises(ValueError, match="only claims"):
        eng.judge(c.id, "yes")
    with pytest.raises(ValueError, match="only claims"):
        eng.resolve(c.id, "supported", "human judgment")

    assert c.state == "open"
    assert c.receipt is None


# ---------------- positions are the durable entity (§1.2) ----------------


def test_rephrasing_an_occupant_never_touches_the_structure(eng):
    """§1.2 — edit the text; id, supports, lineage, receipt, grounding persist."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")
    eng.resolve(a.id, "supported", "ledger shows the double charge")

    before = (
        frame.id, list(frame.supports), set(frame.lineage),
        frame.last_grounded_at,
    )
    grounded_a = eng.position(a.id).last_grounded_at

    eng.record_occurrence(
        frame.id, "Retry paths duplicate their downstream effects",
        relation="refinement",
    )

    assert frame.occupant.text == "Retry paths duplicate their downstream effects"
    assert (
        frame.id, list(frame.supports), set(frame.lineage),
        frame.last_grounded_at,
    ) == before
    assert eng.position(a.id).last_grounded_at == grounded_a
    assert eng.cards[a.id].receipt == "ledger shows the double charge"


def test_altitude_is_derived_from_supports(eng):
    """§1.2 — altitude(claim)=0; altitude(frame)=1+max(floor)."""
    _fund_clicks(eng, n=2)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    alts = eng.altitudes()
    assert alts[a.id] == 0
    assert alts[b.id] == 0
    assert alts[frame.id] == 1

    c = eng.propose("webhook retries duplicate deliveries", section="field")
    d = eng.propose("cron retries duplicate jobs", section="field")
    cand2 = eng.propose_click(c.id, d.id, ClickProposal(
        abstraction="Scheduled repetition amplifies unbounded downstream work",
        specializer_a="in the case of an inbound webhook fan-out",
        specializer_b="in the case of a periodic batch job",
        scope_boundary="excludes single-shot manual invocations",
    ))
    frame2 = eng.confirm_click(cand2.id, confirmed_by="andrew")
    assert eng.altitudes()[frame2.id] == 1


def test_lineage_mass_never_double_counts_shared_roots(eng):
    """§1.6 — mass from the UNION of lineage roots, not ladder height."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field", mass=0.4)
    b = eng.propose("email retries duplicate sends", section="field", mass=0.6)
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    assert eng.position(frame.id).lineage == {a.id, b.id}
    assert eng.lineage_mass(frame.id) == pytest.approx(1.0)


# ---------------- the click gates (§1.3) ----------------


def test_generativity_gate_rejects_a_lexical_merge(eng):
    """§1.3 gate 1 — a tautology fails: all its words are borrowed."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")

    with pytest.raises(GateFailure, match="generativity"):
        eng.propose_click(a.id, b.id, ClickProposal(
            abstraction="both concern retries and duplicate",
            specializer_a="in the case of payment charges",
            specializer_b="in the case of email sends",
            scope_boundary="excludes queries",
        ))
    assert eng.click_candidates == {}


def test_generativity_gate_does_not_demand_invented_novelty(eng):
    """§1.3 / §6 — faithfully naming shared structure must pass.

    Replaces the retired rule "a synthesis must add something absent from both
    parents". The gate rejects borrowed vacuity, not honest recognition.
    """
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    assert cand.status == "open"


@pytest.mark.parametrize("field_name", ["specializer_a", "specializer_b"])
def test_recoverability_gate_requires_a_clause_per_instance(eng, field_name):
    """§1.3 gate 2 — empty specializer, no click."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    proposal = _good_proposal()
    setattr(proposal, field_name, "   ")

    with pytest.raises(GateFailure, match="recoverability"):
        eng.propose_click(a.id, b.id, proposal)


def test_recoverability_gate_rejects_a_specializer_that_restates_the_frame(eng):
    """§1.3 gate 2 — an X-restating specializer recovers nothing."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    proposal = _good_proposal()
    proposal.specializer_a = "silently duplicates downstream effects"

    with pytest.raises(GateFailure, match="recoverability"):
        eng.propose_click(a.id, b.id, proposal)


def test_scope_boundary_gate_rejects_an_abstraction_that_excludes_nothing(eng):
    """§1.3 gate 3 — an abstraction that excludes nothing explains nothing."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    proposal = _good_proposal()
    proposal.scope_boundary = ""

    with pytest.raises(GateFailure, match="scope_boundary"):
        eng.propose_click(a.id, b.id, proposal)


def test_a_failed_gate_is_not_a_card_and_writes_an_attempt(eng):
    """§1.3 — a failed gate emits nothing and consumes the pair."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    before = set(eng.cards)

    with pytest.raises(GateFailure):
        eng.propose_click(a.id, b.id, ClickProposal(
            abstraction="both concern retries and duplicate",
            specializer_a="payment", specializer_b="email",
            scope_boundary="excludes queries",
        ))

    assert set(eng.cards) == before
    assert eng.click_candidates == {}
    assert eng.pair_consumed(a.id, b.id)


def test_content_words_ignores_stopwords_and_punctuation():
    assert content_words("The retries, in the case of payments!") == {
        "retries", "payments"
    }


# ---------------- the fold (§1.6) ----------------


def test_confirmed_click_creates_an_OPEN_frame_with_no_receipt(eng):
    """§1.6 — accepting a click means 'organize these', not 'this is true'."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    assert frame.floor_kind == "frame"
    assert frame.occupant.state == "open"
    assert frame.receipt is None
    assert frame.origin == "click"
    assert frame.confirmed_by == "andrew"
    assert frame.confirmed_at is not None
    assert frame.scope_boundary
    assert frame.specializers[a.id]


def test_instances_are_folded_not_archived(eng):
    """§1.6 — folded, never archived: individually recallable and resolvable."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    for pid in (a.id, b.id):
        assert not eng.cards[pid].archived
        assert eng.position(pid).status == "folded"
        assert eng.position(pid).folded_under == frame.id

    # hidden at the frame's altitude, fully present on descent
    assert a.id not in {c.id for c in eng.live()}
    assert {p.id for p in eng.descend(frame.id)} == {a.id, b.id}

    # and still individually resolvable
    eng.resolve(a.id, "supported", "ledger shows the double charge")
    assert eng.cards[a.id].state == "supported"


def test_accept_with_edit_makes_the_human_wording_the_occupant(eng):
    """§2.4 — accept-with-edit: same fold, human text."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(
        cand.id, confirmed_by="andrew",
        text="Retry logic lacks idempotency at every boundary",
    )
    assert frame.occupant.text == "Retry logic lacks idempotency at every boundary"
    assert eng.position(a.id).status == "folded"


def test_unfold_releases_instances_and_vacates_the_frame(eng):
    """§1.6 — unfold is cheap; the position record persists, never deleted."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")

    released = eng.unfold(frame.id)

    assert {p.id for p in released} == {a.id, b.id}
    assert eng.position(a.id).status == "live"
    assert eng.position(a.id).folded_under is None
    assert eng.position(frame.id).status == "vacated"
    assert frame.id in eng.positions          # history retained
    assert frame.id not in {c.id for c in eng.live()}


def test_folded_instances_stay_in_recurring_ideas(eng):
    """§7.3(b) — a frame must not be a hiding place for repetition."""
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    eng.record_occurrence(a.id, "the double charge came up again")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    eng.confirm_click(cand.id, confirmed_by="andrew")

    recurring = {item["id"] for item in eng.digest()["recurring_ideas"]}
    assert a.id in recurring


def test_a_fold_counts_as_one_unit_for_the_cap(eng):
    """§1.6 — recognition is the only op that reduces field pressure."""
    e = Engine(cap=12, now=Clock())
    _fund_clicks(e)
    a = e.propose("payment retries duplicate charges", section="field")
    b = e.propose("email retries duplicate sends", section="field")
    before = len(e._billable_positions())

    cand = e.propose_click(a.id, b.id, _good_proposal())
    e.confirm_click(cand.id, confirmed_by="andrew")

    # two instances folded away, one frame added: net −1 billable unit
    assert len(e._billable_positions()) == before - 1


# ---------------- derivation, downward (§1.4) ----------------


def _framed(engine):
    _fund_clicks(engine)
    a = engine.propose("payment retries duplicate charges", section="field")
    b = engine.propose("email retries duplicate sends", section="field")
    cand = engine.propose_click(a.id, b.id, _good_proposal())
    return engine.confirm_click(cand.id, confirmed_by="andrew"), a, b


def test_derive_creates_ungrounded_claim_positions(eng):
    """§1.4 — structure awaiting evidence, asserting nothing."""
    frame, _a, _b = _framed(eng)
    kids = eng.derive(frame.id, [
        {"text": "the payment gateway lacks an idempotency key",
         "falsification": "a gateway request log showing a reused key"},
        {"text": "the mailer retries without a dedupe token",
         "falsification": "a mailer config showing a dedupe token"},
    ])

    assert len(kids) == 2
    for card in kids:
        p = eng.position(card.id)
        assert p.floor_kind == "claim"
        assert p.origin == "derivation"
        assert p.occupant.state == "open"
        assert p.last_grounded_at is None
        assert p.receipt is None
        assert card.id in frame.supports

    assert {p.id for p in eng.ungrounded_slots(frame.id)} == {k.id for k in kids}


def test_derive_refuses_a_claim_with_no_falsification_hint(eng):
    """§1.4 — a claim nobody can flip is not receipt-checkable."""
    frame, _a, _b = _framed(eng)
    with pytest.raises(ValueError, match="falsification hint"):
        eng.derive(frame.id, [{"text": "something vaguely true"}])


def test_derive_is_capped(eng):
    frame, _a, _b = _framed(eng)
    kids = eng.derive(frame.id, [
        {"text": f"derived claim {i}", "falsification": f"receipt {i}"}
        for i in range(DERIVE_CAP + 4)
    ])
    assert len(kids) == DERIVE_CAP


def test_derive_asserts_nothing_and_only_a_receipt_moves_the_frame(eng):
    """§1.4 — the only path by which derivation ever changes state."""
    frame, a, b = _framed(eng)
    kids = eng.derive(frame.id, [
        {"text": "the payment gateway lacks an idempotency key",
         "falsification": "a gateway request log showing a reused key"},
    ])
    assert eng.frame_support(frame.id)["supported"] == 0

    eng.resolve(kids[0].id, "supported", "gateway log line 4412 reused the key")

    assert eng.frame_support(frame.id)["supported"] == 1
    assert eng.ungrounded_slots(frame.id) == []


def test_derive_refuses_on_a_claim(eng):
    c = eng.propose("just a claim")
    with pytest.raises(ValueError, match="frames only"):
        eng.derive(c.id, [{"text": "x", "falsification": "y"}])


def test_derived_ungrounded_claims_are_scanner_ineligible(eng):
    """§1.4/§2.2 — derivation fills structure; it does not feed the climb."""
    frame, _a, _b = _framed(eng)
    kids = eng.derive(frame.id, [
        {"text": "the payment gateway lacks an idempotency key",
         "falsification": "a gateway request log showing a reused key"},
    ])
    assert eng.scan_eligible(kids[0].id) is False

    eng.resolve(kids[0].id, "supported", "gateway log line 4412 reused the key")
    # settled claims are not open, so still not fuel — but no longer ungrounded
    assert eng.position(kids[0].id).last_grounded_at is not None


# ---------------- the scanner (§2.2) ----------------


def test_scanner_returns_nothing_without_an_embedding_index(eng):
    """§2.2 — the background loop materializes nothing. Cosine gates the tick."""
    eng.propose("payment retries duplicate charges", section="field")
    eng.propose("email retries duplicate sends", section="field")
    assert eng.scan_candidates() is None


def test_scanner_returns_nothing_below_the_click_floor(eng):
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    assert eng.scan_candidates(lambda x, y: 0.61) is None
    assert eng.scan_candidates(lambda x, y: 0.63) == (a.id, b.id)


def test_scanner_selects_at_most_one_pair(eng):
    for i in range(5):
        eng.propose(f"claim {i}", section="field")
    pair = eng.scan_candidates(lambda x, y: 0.9)
    assert pair is not None and len(pair) == 2


def test_scanner_never_reuses_a_consumed_pair(eng):
    """§2.3 — memory is separate from output; nothing resurrects the pair."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    eng.record_attempt(a.id, b.id, "no_click")
    assert eng.scan_candidates(lambda x, y: 0.9) is None


def test_cargo_types_are_never_automatic_fuel(eng):
    """§1.1/§2.2 — cargo lives at floor 0 and drives the brief, never the climb."""
    for artifact_type in ("question", "decision", "constraint", "experiment",
                          "task", "preference", "observation"):
        c = eng.propose(f"a {artifact_type}", artifact_type=artifact_type)
        assert eng.scan_eligible(c.id) is False


def test_recall_quarantined_positions_never_scan(eng):
    """§4 — adopted material is quarantined until a human pins it."""
    c = eng.propose("recalled idea", origin="recall", external=True)
    assert eng.scan_eligible(c.id) is False
    eng.keep(c.id)
    # pinning lifts quarantine, but a pinned card is also human-held material
    assert eng.position(c.id).pinned_by_human is True


def test_a_frame_scans_only_after_its_floor_has_a_supported_member(eng):
    """§2.2 anti-recursion — the ladder must not grind on its own output."""
    frame, a, _b = _framed(eng)
    assert eng.scan_eligible(frame.id) is False
    eng.resolve(a.id, "supported", "ledger shows the double charge")
    assert eng.scan_eligible(frame.id) is True


def test_a_frame_is_never_compared_against_its_own_descendants(eng):
    """§2.2 — checked via lineage intersection."""
    frame, a, _b = _framed(eng)
    eng.resolve(a.id, "supported", "ledger shows the double charge")
    alts = eng.altitudes()
    assert eng._comparable(frame.id, a.id, alts) is False


def test_scanner_prefers_same_altitude_pairs(eng):
    """§2.2 — the ladder grows level by level."""
    frame, a, _b = _framed(eng)
    eng.resolve(a.id, "supported", "ledger shows the double charge")
    other_frame, other_a, _ = _framed(eng)
    eng.resolve(other_a.id, "supported", "second ledger line")
    ground = eng.propose("an unfolded ground claim", section="field")

    pair = eng.scan_candidates(lambda x, y: 0.9)
    alts = eng.altitudes()
    assert pair is not None
    assert alts[pair[0]] == alts[pair[1]]
    assert ground.id not in pair or alts[pair[0]] == 0


# ---------------- never-retry ledger (§2.3) ----------------


def test_attempt_rows_key_on_position_pairs_regardless_of_order(eng):
    a = eng.propose("a")
    b = eng.propose("b")
    eng.record_attempt(b.id, a.id, "no_click")
    assert eng.pair_consumed(a.id, b.id)
    assert eng.pair_consumed(b.id, a.id)


def test_rewording_cannot_resurrect_a_settled_non_click(eng):
    """§1.2/§2.3 — the ledger keys on the position, not the words."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    eng.record_attempt(a.id, b.id, "no_click")

    eng.record_occurrence(a.id, "charges are duplicated on payment retry",
                          relation="refinement")

    assert eng.pair_consumed(a.id, b.id)
    with pytest.raises(ValueError, match="already attempted"):
        eng.propose_click(a.id, b.id, _good_proposal())


@pytest.mark.parametrize(
    "outcome", ["no_click", "gate_failed", "declined", "expired", "clicked"]
)
def test_semantic_outcomes_are_terminal(eng, outcome):
    a = eng.propose("a")
    b = eng.propose("b")
    eng.record_attempt(a.id, b.id, outcome)
    assert eng.pair_consumed(a.id, b.id) is True


def test_provider_failure_does_not_consume_the_pair(eng):
    """§2.2 fail-closed — an outage must never permanently suppress a click."""
    a = eng.propose("a")
    b = eng.propose("b")
    eng.record_attempt(a.id, b.id, "failed")
    assert eng.pair_consumed(a.id, b.id) is False


def test_reconsideration_is_an_explicit_versioned_human_act(eng):
    """§2.3 — retry is a paper-trailed door, never an automatic one."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    eng.record_attempt(a.id, b.id, "declined")
    assert eng.current_operation_version(a.id, b.id) == 1

    row = eng.reconsider_pair(a.id, b.id)

    assert row["operation_version"] == 2
    assert eng.pair_consumed(a.id, b.id) is False
    # the original row survives: the paper trail is the point
    key = tuple(sorted((a.id, b.id))) + (1,)
    assert eng.click_attempts[key]["outcome"] == "declined"
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    assert cand.status == "open"


def test_reconsider_refuses_an_unconsumed_pair(eng):
    a = eng.propose("a")
    b = eng.propose("b")
    with pytest.raises(ValueError, match="not consumed"):
        eng.reconsider_pair(a.id, b.id)


# ---------------- the emergence inbox (§2.4) ----------------


def test_inbox_is_capped_at_three_open_candidates(eng):
    ids = [eng.propose(f"distinct source claim number {i}", section="field").id
           for i in range(10)]
    for i in range(INBOX_CAP):
        eng.propose_click(ids[2 * i], ids[2 * i + 1], ClickProposal(
            abstraction=f"Unbounded amplification pattern variant {i} recurs",
            specializer_a=f"observed on the inbound edge {i}",
            specializer_b=f"observed on the outbound edge {i}",
            scope_boundary="excludes purely local computation",
        ))
    assert len(eng.open_candidates()) == INBOX_CAP

    with pytest.raises(ValueError, match="inbox is full"):
        eng.propose_click(ids[8], ids[9], _good_proposal())


def test_declining_a_candidate_writes_a_declined_attempt(eng):
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())

    eng.decline_click(cand.id)

    assert cand.status == "declined"
    assert eng.pair_consumed(a.id, b.id)
    assert eng.open_candidates() == []


def test_an_unacted_candidate_expires_after_the_ttl(eng):
    """§2.4 — 7 days unacted auto-expires. No retry."""
    clock = Clock()
    e = Engine(cap=12, now=clock)
    a = e.propose("payment retries duplicate charges", section="field")
    b = e.propose("email retries duplicate sends", section="field")
    cand = e.propose_click(a.id, b.id, _good_proposal())

    clock.t += CLICK_TTL + 1
    assert e.open_candidates() == []
    assert cand.status == "expired"
    assert e.pair_consumed(a.id, b.id)


def test_candidates_never_occupy_field_space(eng):
    """§2.4 — the inbox is a separate tray."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    before = {c.id for c in eng.live()}

    eng.propose_click(a.id, b.id, _good_proposal())

    assert {c.id for c in eng.live()} == before


def test_click_budget_caps_compression_at_one_per_five_contributions(eng):
    """§7.1 — compression cannot outrun input."""
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())

    assert eng.click_budget_remaining() == 0
    with pytest.raises(ValueError, match="click budget"):
        eng.confirm_click(cand.id, confirmed_by="andrew")

    for i in range(3):
        eng.propose(f"more input {i}", section="field")
    assert eng.click_budget_remaining() == 1
    frame = eng.confirm_click(cand.id, confirmed_by="andrew")
    assert eng.click_budget_remaining() == 0
    assert frame.floor_kind == "frame"


def test_confirm_click_requires_a_named_human(eng):
    _fund_clicks(eng)
    a = eng.propose("payment retries duplicate charges", section="field")
    b = eng.propose("email retries duplicate sends", section="field")
    cand = eng.propose_click(a.id, b.id, _good_proposal())
    with pytest.raises(ValueError, match="name a human"):
        eng.confirm_click(cand.id, confirmed_by="  ")


# ---------------- persistence ----------------


def test_load_rejects_terminal_state_without_receipt():
    """A snapshot cannot smuggle in a settled card that has no receipt."""
    for state in ("supported", "refuted"):
        for receipt in (None, "", "   "):
            e = Engine.from_state({"cards": [
                {"id": "c1", "state": state, "receipt": receipt, "text": "forged"}
            ]})
            c = e.cards["c1"]
            assert c.state == "needs_human", (state, receipt)
            assert c.receipt is None


def test_load_preserves_properly_receipted_terminal_state():
    e = Engine.from_state({"cards": [
        {"id": "c1", "state": "supported", "receipt": "because evidence",
         "text": "real"}
    ]})
    assert e.cards["c1"].state == "supported"
    assert e.cards["c1"].receipt == "because evidence"


def test_load_downgrades_terminal_non_claim_even_with_a_receipt():
    e = Engine.from_state({"cards": [
        {
            "id": "c1",
            "artifact_type": "decision",
            "state": "supported",
            "receipt": "legacy judgment",
            "text": "Ship Friday",
        }
    ]})
    assert e.cards["c1"].state == "needs_human"
    assert e.cards["c1"].receipt is None


def test_load_downgrades_a_terminal_frame_even_with_a_receipt():
    """§1.1 — frames cannot carry receipts at all, including on deserialization."""
    e = Engine.from_state({
        "cards": [
            {"id": "c1", "state": "open", "text": "floor"},
            {"id": "c2", "state": "supported", "receipt": "legacy fusion receipt",
             "text": "legacy synthesis"},
        ],
        "positions": [
            {"id": "c1", "floor_kind": "claim"},
            {"id": "c2", "floor_kind": "frame", "supports": ["c1"]},
        ],
    })
    assert e.cards["c2"].state == "needs_human"
    assert e.cards["c2"].receipt is None


def test_legacy_parents_migrate_to_provenance_never_to_supports():
    """§5 backfill — combination provenance is not identity recognition."""
    e = Engine.from_state({"cards": [
        {"id": "c1", "text": "a"},
        {"id": "c2", "text": "b"},
        {"id": "c3", "kind": "synthesis", "text": "legacy synthesis",
         "parents": ["c1", "c2"]},
    ]})
    p = e.position("c3")
    assert p.provenance == ["c1", "c2"]
    assert p.supports == []
    assert p.floor_kind == "claim"
    assert e.altitudes()["c3"] == 0


def test_load_legacy_card_gets_backward_compatible_semantic_defaults():
    e = Engine.from_state({"cards": [
        {"id": "c1", "kind": "synthesis", "text": "legacy", "born": 123.0}
    ]})
    c = e.cards["c1"]
    assert c.artifact_type == "claim"
    assert c.occurrence_count == 1
    assert c.occurrences == []
    assert c.evolution == []
    assert c.first_seen == 123.0
    assert c.last_seen == 123.0


def test_round_trip_of_a_resolved_card_survives_the_load_guard(tmp_path, eng):
    c = eng.propose("a", section="field")
    eng.resolve(c.id, "supported", "test run observed expected output",
                text="settled text")
    p = tmp_path / "state.json"
    save(eng, str(p))
    e2 = load(str(p))
    assert e2.cards[c.id].state == "supported"
    assert e2.cards[c.id].text == "settled text"


def test_round_trip_preserves_the_ladder_and_the_ledger(tmp_path, eng):
    """§1.2 — the structure is the durable asset; it must survive a save."""
    frame, a, b = _framed(eng)
    eng.resolve(a.id, "supported", "ledger shows the double charge")
    eng.record_attempt(a.id, b.id, "no_click", operation_version=3)

    p = tmp_path / "state.json"
    save(eng, str(p))
    e2 = load(str(p))

    f2 = e2.position(frame.id)
    assert f2.floor_kind == "frame"
    assert set(f2.supports) == {a.id, b.id}
    assert f2.lineage == {a.id, b.id}
    assert f2.confirmed_by == "andrew"
    assert e2.position(a.id).status == "folded"
    assert e2.frame_support(frame.id)["summary"] == "1✓ 0✗ 1○"
    assert e2.current_operation_version(a.id, b.id) == 3


def test_save_load_round_trip(tmp_path, eng):
    eng.seed("round trip?")
    eng.add_section("b", "B", "#123456")
    a = eng.propose("a", section="field", mass=0.4)
    eng.propose("b", section="b", mass=0.6)
    eng.keep(a.id)
    c = eng.propose("c", section="field")
    eng.resolve(c.id, "supported", "receipted")
    p = tmp_path / "nested" / "state.json"
    save(eng, str(p))
    assert p.exists()
    assert not (tmp_path / "nested" / "state.json.tmp").exists()
    e2 = load(str(p))
    assert e2.question == eng.question
    assert e2.cap == eng.cap
    assert e2.state()["cards"] == eng.state()["cards"]
    assert e2.section_order == eng.section_order
    assert e2.ledger == eng.ledger
    new = e2.propose("fresh")
    assert new.id not in eng.cards


# ---------------- judge ----------------


def test_judge_yes_resolves_with_human_receipt(eng):
    c = eng.propose("x")
    out = eng.judge(c.id, "yes")
    assert out.state == "supported"
    assert out.receipt == "human judgment · yes"


def test_judge_no_refutes(eng):
    c = eng.propose("x")
    assert eng.judge(c.id, "no").state == "refuted"


def test_judge_unknown_reopens_and_splits(eng):
    c = eng.propose("x")
    eng.request_verify(c.id)
    kids = eng.judge(c.id, "unknown")
    assert eng.cards[c.id].state == "open"
    assert isinstance(kids, list) and len(kids) == 2
    assert all(k.state == "needs_human" for k in kids)
    assert all(k.parents == [c.id] for k in kids)
    assert all(k.artifact_type == "question" for k in kids)
    # engine-generated prompts do not fund the §7.1 click budget
    assert all(eng.position(k.id).origin == "derivation" for k in kids)


def test_judge_bad_verdict(eng):
    c = eng.propose("x")
    with pytest.raises(ValueError):
        eng.judge(c.id, "sorta")


def test_unknown_clears_stale_receipt_on_unsettled_card(eng):
    c = eng.propose("x")
    c.receipt = "stale legacy receipt"

    kids = eng.judge(c.id, "unknown")

    assert c.state == "open"
    assert c.receipt is None
    assert len(kids) == 2


@pytest.mark.parametrize("first", ["yes", "no"])
@pytest.mark.parametrize("second", ["yes", "no", "unknown"])
def test_settled_cards_cannot_be_rejudged(eng, first, second):
    c = eng.propose("x")
    eng.judge(c.id, first)
    original = (c.state, c.receipt, list(eng.ledger))

    with pytest.raises(ValueError, match="settled"):
        eng.judge(c.id, second)

    assert (c.state, c.receipt, eng.ledger) == original


# ---------------- semantic identity / occurrences ----------------


def test_propose_records_semantic_type_and_origin_provenance(eng):
    c = eng.propose(
        "Run the latency experiment",
        artifact_type="experiment",
        foot="human note",
    )

    assert c.artifact_type == "experiment"
    assert c.occurrence_count == 1
    assert c.first_seen == c.last_seen == c.born
    assert c.occurrences == [{
        "text": "Run the latency experiment",
        "relation": "origin",
        "foot": "human note",
        "ts": c.born,
    }]


def test_propose_rejects_unknown_artifact_type(eng):
    with pytest.raises(ValueError, match="bad artifact type"):
        eng.propose("x", artifact_type="vibe")


def test_propose_rejects_a_frame_with_no_floor(eng):
    """§1.2 — every level is the shadow of a click or a derivation."""
    with pytest.raises(ValueError, match="at least one position"):
        eng.propose("a groundless frame", floor_kind="frame")


def test_propose_rejects_supports_on_a_claim(eng):
    a = eng.propose("a")
    with pytest.raises(ValueError, match="only frames"):
        eng.propose("x", supports=[a.id])


def test_find_canonical_is_unicode_case_and_whitespace_normalized(eng):
    canonical = eng.propose("Ｆast   retries\nneed IDs")

    assert eng.find_canonical("fast retries need ids") is canonical
    assert eng.find_canonical("fast retries need ids!") is None
    assert eng.find_canonical("   ") is None


def test_find_canonical_ignores_archived_cards(eng):
    c = eng.propose("same idea")
    eng.kill(c.id)

    assert eng.find_canonical("SAME IDEA") is None


def test_record_repeat_adds_provenance_without_creating_a_card(eng):
    canonical = eng.propose("Use stable workspace IDs")
    before_ids = list(eng.order)

    out = eng.record_occurrence(
        canonical.id,
        "Stable workspace identifiers avoid cross-target writes",
        relation="repeat",
        foot="second interview",
    )

    assert out is canonical
    assert eng.order == before_ids
    assert canonical.text == "Use stable workspace IDs"
    assert canonical.occurrence_count == 2
    assert canonical.last_seen > canonical.first_seen
    assert canonical.occurrences[-1]["relation"] == "repeat"
    assert canonical.occurrences[-1]["foot"] == "second interview"
    assert canonical.evolution == []
    assert eng.ledger[-1]["kind"] == "OCCURRED"


def test_record_refinement_evolves_unsettled_card_without_creating_one(eng):
    canonical = eng.propose("Retries need IDs")
    before_ids = list(eng.order)

    out = eng.record_occurrence(
        canonical.id,
        "Retries must preserve one idempotency key across attempts",
        relation="refinement",
        foot="reviewed wording",
    )

    assert out is canonical
    assert eng.order == before_ids
    assert canonical.text == "Retries must preserve one idempotency key across attempts"
    assert canonical.occurrence_count == 2
    assert canonical.evolution == [{
        "from": "Retries need IDs",
        "to": "Retries must preserve one idempotency key across attempts",
        "foot": "reviewed wording",
        "ts": canonical.last_seen,
    }]
    assert eng.ledger[-1]["kind"] == "EVOLVED"


def test_record_refinement_cannot_rewrite_receipted_wording(eng):
    canonical = eng.propose("Settled wording")
    eng.resolve(canonical.id, "supported", "observed")
    before = canonical.to_dict()

    with pytest.raises(ValueError, match="settled"):
        eng.record_occurrence(
            canonical.id,
            "Different wording",
            relation="refinement",
        )

    assert canonical.to_dict() == before


def test_record_occurrence_validates_relation_and_text(eng):
    c = eng.propose("x")
    with pytest.raises(ValueError, match="repeat or refinement"):
        eng.record_occurrence(c.id, "x", relation="duplicate")
    with pytest.raises(ValueError, match="text required"):
        eng.record_occurrence(c.id, "   ")


def test_occurrence_provenance_is_bounded_but_total_count_is_not(eng):
    c = eng.propose("recurring")
    for i in range(60):
        eng.record_occurrence(c.id, f"recurring mention {i}", foot=f"source {i}")

    assert c.occurrence_count == 61
    assert len(c.occurrences) == 50
    assert c.occurrences[0]["text"] == "recurring mention 10"
    assert c.occurrences[-1]["text"] == "recurring mention 59"


# ---------------- frontier / cap ----------------


def test_frontier_ranks_needs_human_first_then_mass(eng):
    eng.propose("heavy", mass=0.9)
    eng.propose("light", mass=0.1)
    nh = eng.propose("needy", mass=0.2, state="needs_human")
    testing = eng.propose("busy", mass=0.99)
    eng.request_verify(testing.id)
    f = eng.frontier()
    assert [c.text for c in f] == ["needy", "heavy", "light"]
    assert f[0].id == nh.id


def test_enforce_cap_archives_lightest_unpinned():
    e = Engine(cap=3, now=Clock())
    heavy = e.propose("heavy", mass=0.9)
    mid = e.propose("mid", mass=0.5)
    light = e.propose("light", mass=0.1)
    assert e.enforce_cap() is None
    extra = e.propose("extra", mass=0.7)
    victim = e.enforce_cap()
    assert victim.id == light.id
    assert e.cards[light.id].archived
    assert {c.id for c in e.live()} == {heavy.id, mid.id, extra.id}
    assert any(l["kind"] == "RETIRED" for l in e.ledger)


def test_enforce_cap_never_archives_pinned():
    e = Engine(cap=2, now=Clock())
    light = e.propose("light", mass=0.05)
    e.keep(light.id)
    e.propose("mid", mass=0.5)
    e.propose("heavy", mass=0.9)
    victim = e.enforce_cap()
    assert victim.text == "mid"
    assert not e.cards[light.id].archived


def test_enforce_cap_all_pinned_is_noop():
    e = Engine(cap=1, now=Clock())
    a = e.propose("a")
    b = e.propose("b")
    e.keep(a.id)
    e.keep(b.id)
    assert e.enforce_cap() is None


def test_enforce_cap_treats_cap_as_per_field_budget():
    e = Engine(cap=2, now=Clock())
    e.add_section("questions", "QUESTIONS", "#fff")
    for i in range(4):
        e.propose(
            f"card {i}",
            section="field" if i % 2 == 0 else "questions",
        )

    assert e.state()["capacity"] == 4
    assert e.enforce_cap() is None


def test_enforce_cap_retires_machine_origin_before_human_roots():
    """§1.6 ordering — machine-origin material absorbs pressure first."""
    e = Engine(cap=2, now=Clock())
    a = e.propose("human root a", mass=0.1)
    e.propose("human root b", mass=0.1)
    generated = e.propose(
        "derived claim",
        mass=0.9,
        parents=[a.id],
        origin="derivation",
    )

    victim = e.enforce_cap()

    assert victim.id == generated.id
    assert not e.cards[a.id].archived


def test_enforce_cap_never_retires_a_position_with_dependents():
    """§1.6 ordering invariant — never retire positions with dependents."""
    e = Engine(cap=1, now=Clock())
    _fund_clicks(e)
    a = e.propose("payment retries duplicate charges", section="field", mass=0.01)
    b = e.propose("email retries duplicate sends", section="field", mass=0.01)
    cand = e.propose_click(a.id, b.id, _good_proposal())
    frame = e.confirm_click(cand.id, confirmed_by="andrew")

    for _ in range(20):
        if e.enforce_cap() is None:
            break

    assert not e.cards[a.id].archived
    assert not e.cards[b.id].archived
    assert e.position(a.id).status == "folded"
    assert e.position(frame.id).status == "live"


def test_enforce_cap_never_retires_folded_instances():
    e = Engine(cap=1, now=Clock())
    _fund_clicks(e)
    a = e.propose("payment retries duplicate charges", section="field", mass=0.01)
    b = e.propose("email retries duplicate sends", section="field", mass=0.01)
    cand = e.propose_click(a.id, b.id, _good_proposal())
    e.confirm_click(cand.id, confirmed_by="andrew")

    for _ in range(20):
        e.enforce_cap()

    assert e.position(a.id).status == "folded"
    assert e.position(b.id).status == "folded"


def test_enforce_cap_protects_needs_human_and_last_root_per_field():
    e = Engine(cap=1, now=Clock())
    root = e.propose("only root", section="field")
    prompt = e.propose(
        "what would decide this?",
        section="field",
        state="needs_human",
        parents=[root.id],
    )

    assert e.enforce_cap() is None
    assert not root.archived
    assert not prompt.archived


# ---------------- mutators ----------------


def test_keep_toggles(eng):
    c = eng.propose("x")
    assert eng.keep(c.id).pinned is True
    assert eng.keep(c.id).pinned is False


def test_kill_archives(eng):
    c = eng.propose("x")
    eng.kill(c.id)
    assert eng.cards[c.id].archived
    assert eng.live() == []
    assert eng.position(c.id).status == "retired"
    assert eng.ledger[-1]["kind"] == "KILLED"


def test_move_changes_section_and_creates_missing(eng):
    c = eng.propose("x")
    eng.move(c.id, "elsewhere")
    assert eng.cards[c.id].section == "elsewhere"
    assert "elsewhere" in eng.sections
    assert any(l["kind"] == "MOVED" for l in eng.ledger)


@pytest.mark.parametrize(
    "operation",
    [
        lambda e, card_id: e.keep(card_id),
        lambda e, card_id: e.move(card_id, "elsewhere"),
        lambda e, card_id: e.kill(card_id),
        lambda e, card_id: e.judge(card_id, "yes"),
        lambda e, card_id: e.judge(card_id, "no"),
        lambda e, card_id: e.judge(card_id, "unknown"),
        lambda e, card_id: e.request_verify(card_id),
        lambda e, card_id: e.resolve(card_id, "supported", "late verifier"),
        lambda e, card_id: e.update_proposal(card_id, "late worker result"),
        lambda e, card_id: e.reopen(card_id, foot="late worker failure"),
    ],
)
def test_archived_cards_reject_human_control_mutations(eng, operation):
    c = eng.propose("x")
    eng.kill(c.id)
    ledger = list(eng.ledger)

    with pytest.raises(ValueError, match="archived"):
        operation(eng, c.id)

    assert eng.cards[c.id].archived
    assert eng.ledger == ledger


def test_archived_cards_cannot_be_click_instances(eng):
    archived = eng.propose("payment retries duplicate charges")
    live = eng.propose("email retries duplicate sends")
    eng.kill(archived.id)
    before_ids = set(eng.cards)

    with pytest.raises(ValueError, match="archived"):
        eng.propose_click(archived.id, live.id, _good_proposal())

    assert set(eng.cards) == before_ids


def test_request_verify_only_from_open(eng):
    c = eng.propose("x")
    assert eng.request_verify(c.id).state == "testing"
    with pytest.raises(ValueError):
        eng.request_verify(c.id)


def test_propose_none_section_picks_lightest(eng):
    eng.add_section("light", "LIGHT", "#fff")
    eng.propose("heavy one", section="field", mass=0.9)
    c = eng.propose("auto")
    assert c.section == "light"


def test_missing_card_raises(eng):
    for fn in (eng.keep, eng.kill, eng.request_verify):
        with pytest.raises(KeyError):
            fn("nope")


# ---------------- reporting ----------------


def test_weights_normalize_to_one(eng):
    eng.add_section("b", "B", "#fff")
    eng.propose("x", section="field", mass=0.6)
    eng.propose("y", section="b", mass=0.2)
    w = eng.weights()
    assert w["field"]["count"] == 1
    assert w["field"]["mass"] == pytest.approx(0.6)
    assert sum(v["norm"] for v in w.values()) == pytest.approx(1.0)
    assert w["field"]["norm"] == pytest.approx(0.75)


def test_weights_empty_field_norm_zero(eng):
    assert eng.weights()["field"]["norm"] == 0.0


def test_harvest_shape_is_the_brief_not_a_dump(eng):
    """§3.3 — sections replace the old wholesale card dump."""
    eng.seed("what is true?")
    frame, a, _b = _framed(eng)
    eng.resolve(a.id, "supported", "human verified result")

    h = eng.harvest()

    assert set(h) == {
        "question", "altitude", "max_altitude", "sections", "spine", "cracks",
        "stale", "cruxes", "decisions", "constraints", "experiments",
        "unresolved", "changed",
    }
    assert h["question"] == "what is true?"
    assert h["max_altitude"] == 1
    assert frame.id in {item["id"] for item in h["spine"]}
    json.dumps(h)
    assert any(l["kind"] == "HARVESTED" for l in eng.ledger)


def test_harvest_caps_every_section(eng):
    for i in range(30):
        eng.propose(f"question {i}", artifact_type="question")
    h = eng.harvest(max_items=4)
    assert len(h["cruxes"]) == 4
    assert len(h["spine"]) == 4


def test_harvest_surfaces_cracks_and_ungrounded_slots(eng):
    """§3.3 — the ladder's honesty section."""
    frame, a, _b = _framed(eng)
    eng.resolve(a.id, "refuted", "the ledger shows exactly one charge")
    slots = eng.derive(frame.id, [
        {"text": "the mailer lacks a dedupe token",
         "falsification": "a mailer config showing the token"},
    ])

    h = eng.harvest()

    assert frame.id in {item["id"] for item in h["cracks"]}
    crux_ids = {item["id"] for item in h["cruxes"]}
    assert slots[0].id in crux_ids


def test_harvest_altitude_filter_selects_the_spine(eng):
    frame, _a, _b = _framed(eng)
    ground = eng.propose("a floor-0 claim", section="field")

    h = eng.harvest(altitude=1)

    ids = {item["id"] for item in h["spine"]}
    assert frame.id in ids
    assert ground.id not in ids


def test_digest_exposes_frames_not_between_ideas(eng):
    """§3.2 — `between_ideas` becomes `frames`, truncated to 3."""
    frame, _a, _b = _framed(eng)
    d = eng.digest()
    assert "between_ideas" not in d
    assert [item["id"] for item in d["frames"]] == [frame.id]
    assert d["frames"][0]["support_summary"] == "0✓ 0✗ 2○"


def test_digest_groups_themes_recurring_and_typed_artifacts(eng):
    eng.add_section("delivery", "DELIVERY", "#fff")
    recurring = eng.propose("Use stable IDs", section="field", mass=0.8)
    eng.record_occurrence(recurring.id, "Stable IDs came up again", foot="session 2")
    question = eng.propose(
        "Which identifier is stable?",
        section="field",
        artifact_type="question",
    )
    decision = eng.propose(
        "Adopt workspace UUIDs",
        section="delivery",
        artifact_type="decision",
    )
    constraint = eng.propose(
        "Migration cannot break old links",
        section="delivery",
        artifact_type="constraint",
    )
    experiment = eng.propose(
        "Replay writes during workspace switches",
        section="delivery",
        artifact_type="experiment",
    )
    task = eng.propose(
        "Add the replay fixture",
        section="delivery",
        artifact_type="task",
    )

    d = eng.digest()

    assert [item["id"] for item in d["recurring_ideas"]] == [recurring.id]
    assert [item["id"] for item in d["open_questions"]] == [question.id]
    assert [item["id"] for item in d["decisions"]] == [decision.id]
    assert [item["id"] for item in d["constraints"]] == [constraint.id]
    assert [item["id"] for item in d["experiments"]] == [experiment.id]
    assert [item["id"] for item in d["tasks"]] == [task.id]
    assert "frames" not in d
    assert [theme["key"] for theme in d["themes"]] == ["delivery", "field"]
    field = next(theme for theme in d["themes"] if theme["key"] == "field")
    assert field["card_count"] == 2
    assert field["occurrence_count"] == 3
    assert field["artifact_types"] == {"claim": 1, "question": 1}


def test_state_is_serializable_and_complete(eng):
    eng.seed("q")
    eng.propose("x")
    s = eng.state()
    json.dumps(s)
    assert s["question"] == "q" and s["cap"] == 12
    assert "ledger" in s and "cards" in s and "sections" in s
    assert "positions" in s and "click_attempts" in s
    assert s["digest"]["themes"][0]["top_ideas"][0]["text"] == "x"


# ---------------- ledger ----------------


def test_ledger_records_the_new_vocabulary(eng):
    eng.seed("q")
    frame, a, _b = _framed(eng)
    eng.resolve(a.id, "supported", "r")
    eng.derive(frame.id, [{"text": "derived", "falsification": "a receipt"}])
    eng.unfold(frame.id)
    kinds = [l["kind"] for l in eng.ledger]
    for k in ("ARRIVED", "CLICKED", "RESOLVED", "DERIVED", "UNFOLDED"):
        assert k in kinds
    assert "FUSING" not in kinds
    assert "FUSED" not in kinds
    assert all(set(l) == {"kind", "text", "ts"} for l in eng.ledger)


def test_ledger_capped_at_50(eng):
    for i in range(80):
        eng.propose(f"c{i}")
    assert len(eng.ledger) == 50
    assert eng.ledger[-1]["text"] == "c79"
