"""magpie.engine — the law (Altitude).

Deterministic, pure in-memory state machine for the Altitude field. No I/O
(except the module-level load/save helpers), no network, no threads, no
randomness beyond a monotonic id counter.

THE RECEIPT LAW (inviolable, SPEC §1.1 / §1.6): `resolve()` is the ONLY code
path that may set a claim's state to 'supported' or 'refuted', and it refuses
to run without a receipt. Receipts are not cards — they attach to claims.
Frames are never directly supported or refuted; their support is COMPUTED from
the floor below by `frame_support()` and is never stored. Nothing at any layer
may drift from the layer below, because no layer stores what the layer below
determines.

THE STRUCTURE (SPEC §1.2): the durable first-class entity is the *position* —
an altitude, its load-bearing edges to the floor below, its support state, its
receipt, its last-grounded date. Ideas are the current *occupants* of positions
and may be rephrased, merged, or replaced without losing the structure. The
`Card` dataclass is the occupant payload; a position's id is its occupant's id
so external references survive the inversion.

Altitude and frame support are derived, never stored (SPEC §1.5).
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

__all__ = [
    "Card",
    "Position",
    "ClickProposal",
    "ClickCandidate",
    "Engine",
    "GateFailure",
    "load",
    "save",
]

KINDS = ("thesis", "claim", "synthesis")
ARTIFACT_TYPES = (
    "observation",
    "claim",
    "question",
    "preference",
    "constraint",
    "task",
    "experiment",
    "decision",
)
STATES = ("open", "testing", "supported", "refuted", "needs_human")
LIVE_STATES = ("open", "testing", "needs_human", "supported", "refuted")
LEDGER_CAP = 50
LEDGER_TEXT_CAP = 1024
PROVENANCE_CAP = 50

# --- SPEC §1.1: the three structural floors -------------------------------
FLOOR_KINDS = ("claim", "frame")

# --- SPEC §1.2: position lifecycle ----------------------------------------
POSITION_STATUSES = ("live", "folded", "vacated", "retired")
ORIGINS = ("human", "click", "derivation", "recall")

# --- SPEC §1.4: downward derivation ---------------------------------------
DERIVE_CAP = 5

# --- SPEC §2.2: the scanner -----------------------------------------------
CLICK_FLOOR = 0.62

# --- SPEC §2.3: the never-retry ledger ------------------------------------
ATTEMPT_OUTCOMES = (
    "no_click",
    "gate_failed",
    "declined",
    "clicked",
    "expired",
    "failed",
    # Not an outcome but the OPENING of a new operation_version, written only
    # by reconsider_pair(). It is a row so the retry is provenance-visible.
    "reconsidered",
)
# Two rows do not consume their pair: `failed` (a provider outage must never
# permanently suppress a legitimate future recognition) and `reconsidered` (the
# deliberate human door, which exists precisely to un-consume). Every other
# outcome is semantically terminal for that (pair, operation_version).
NON_CONSUMING_OUTCOMES = ("failed", "reconsidered")

# --- SPEC §2.4: the emergence inbox ---------------------------------------
INBOX_CAP = 3
CLICK_TTL = 7 * 24 * 3600.0

# --- SPEC §7.1: recognition inflation guardrail ---------------------------
CLICK_BUDGET_PER_CONTRIBUTIONS = 5

# Words carrying no content for the §1.3 generativity gate. Deliberately a
# closed stoplist rather than a model call: the gate must be deterministic.
STOPWORDS = frozenset("""
a an the this that these those there here
is are was were be been being am
do does did doing done
have has had having
will would shall should can could may might must
of in on at to from by for with without within into onto over under
about against between among across through during before after above below
and or but nor so yet if then than as because while when where which who whom
whose what how why not no nor only just also both each any all some more most
much many few less least very such own same other another it its it's they them
their theirs we us our ours you your yours i me my mine he him his she her hers
one ones thing things case cases concern concerns concerning regarding
""".split())


@dataclass
class Card:
    """The occupant payload: the *idea* currently sitting in a position.

    A card is not the durable entity — see :class:`Position`. Rephrasing a card
    leaves the structure (id, supports, lineage, receipt, last_grounded_at)
    untouched.
    """

    id: str
    kind: str
    text: str
    section: str
    mass: float
    state: str
    receipt: str | None
    foot: str
    pinned: bool
    born: float
    parents: list[str] = field(default_factory=list)
    archived: bool = False
    artifact_type: str = "claim"
    occurrence_count: int = 1
    occurrences: list[dict] = field(default_factory=list)
    evolution: list[dict] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Card":
        born = float(d.get("born", 0.0))
        artifact_type = str(d.get("artifact_type", "claim") or "claim").lower()
        if artifact_type not in ARTIFACT_TYPES:
            artifact_type = "claim"
        occurrences = [
            dict(item) for item in (d.get("occurrences", []) or [])
            if isinstance(item, dict)
        ][-PROVENANCE_CAP:]
        evolution = [
            dict(item) for item in (d.get("evolution", []) or [])
            if isinstance(item, dict)
        ][-PROVENANCE_CAP:]
        try:
            occurrence_count = int(d.get("occurrence_count", 1))
        except (TypeError, ValueError):
            occurrence_count = 1
        occurrence_count = max(1, occurrence_count, len(occurrences))
        return Card(
            id=d["id"],
            kind=d.get("kind", "claim"),
            text=d.get("text", ""),
            section=d.get("section", ""),
            mass=float(d.get("mass", 0.42)),
            state=d.get("state", "open"),
            receipt=d.get("receipt"),
            foot=d.get("foot", ""),
            pinned=bool(d.get("pinned", False)),
            born=born,
            parents=list(d.get("parents", []) or []),
            archived=bool(d.get("archived", False)),
            artifact_type=artifact_type,
            occurrence_count=occurrence_count,
            occurrences=occurrences,
            evolution=evolution,
            first_seen=float(d.get("first_seen", born)),
            last_seen=float(d.get("last_seen", d.get("first_seen", born))),
        )


@dataclass
class Position:
    """SPEC §1.2 — the durable, first-class entity. Never deleted.

    ``support_state`` and ``receipt`` are meaningful for claims ONLY. For a
    frame they are structurally absent: frame support is computed by
    :meth:`Engine.frame_support` from the floor below and never stored.
    """

    id: str
    floor_kind: str                      # 'claim' | 'frame'
    occupant: Card
    supports: list[str] = field(default_factory=list)   # ids one floor DOWN
    provenance: list[str] = field(default_factory=list)  # legacy `parents`
    lineage: set[str] = field(default_factory=set)      # floor-0 root ids
    origin: str = "human"
    status: str = "live"
    folded_under: str | None = None
    last_grounded_at: float | None = None
    external: bool = False
    pinned_by_human: bool = False        # lifts recall/derivation quarantine
    confirmed_by: str | None = None      # click provenance, NEVER a receipt
    confirmed_at: float | None = None
    specializers: dict[str, str] = field(default_factory=dict)
    scope_boundary: str = ""

    # ---- derived views over the occupant (never a second source of truth) --

    @property
    def support_state(self) -> str:
        """Claims carry a stored support state; frames never do (§1.5)."""
        if self.floor_kind == "frame":
            raise AttributeError(
                "frames have no stored support_state; use frame_support()"
            )
        return self.occupant.state

    @property
    def receipt(self) -> str | None:
        if self.floor_kind == "frame":
            return None
        return self.occupant.receipt

    @property
    def text(self) -> str:
        return self.occupant.text

    @property
    def mass(self) -> float:
        return self.occupant.mass

    @property
    def section(self) -> str:
        return self.occupant.section

    @property
    def artifact_type(self) -> str:
        return self.occupant.artifact_type

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "floor_kind": self.floor_kind,
            "supports": list(self.supports),
            "provenance": list(self.provenance),
            "lineage": sorted(self.lineage),
            "origin": self.origin,
            "status": self.status,
            "folded_under": self.folded_under,
            "last_grounded_at": self.last_grounded_at,
            "external": self.external,
            "pinned_by_human": self.pinned_by_human,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at,
            "specializers": dict(self.specializers),
            "scope_boundary": self.scope_boundary,
            "occupant": self.occupant.to_dict(),
            "occupant_revisions": list(self.occupant.evolution),
        }


@dataclass
class ClickProposal:
    """The worker's returned text, before the §1.3 gates run on it."""

    abstraction: str
    specializer_a: str
    specializer_b: str
    scope_boundary: str


@dataclass
class ClickCandidate:
    """A gate-passing proposal parked in the emergence inbox (§2.4)."""

    id: str
    position_a: str
    position_b: str
    proposal: ClickProposal
    created_at: float
    status: str = "open"     # open|accepted|declined|expired

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position_a": self.position_a,
            "position_b": self.position_b,
            "abstraction": self.proposal.abstraction,
            "specializer_a": self.proposal.specializer_a,
            "specializer_b": self.proposal.specializer_b,
            "scope_boundary": self.proposal.scope_boundary,
            "created_at": self.created_at,
            "status": self.status,
        }


class GateFailure(ValueError):
    """A §1.3 gate rejected a proposal. Emits nothing; writes an attempt row."""

    def __init__(self, gate: str, detail: str = ""):
        self.gate = gate
        super().__init__(f"{gate}: {detail}" if detail else gate)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalized_text(text: str) -> str:
    """Normalize only representation, never meaning, for exact deduplication."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(normalized.split())


def content_words(text: str) -> set[str]:
    """Content-word set for the §1.3 generativity gate.

    Punctuation is stripped and stopwords removed so the gate measures *what a
    phrase is about*, not how it is written. Deterministic and free — this is
    the mechanical close on the tautology hole (Appendix B #1).
    """
    out: set[str] = set()
    for raw in _normalized_text(text).split():
        word = "".join(ch for ch in raw if ch.isalnum() or ch == "-").strip("-")
        if len(word) < 2 or word in STOPWORDS:
            continue
        out.add(word)
    return out


def _pair_key(a_id: str, b_id: str) -> tuple[str, str]:
    """Ledger keys are order-independent: min(id), max(id) — §2.3."""
    return (a_id, b_id) if a_id <= b_id else (b_id, a_id)


class Engine:
    def __init__(self, cap: int = 12, now: Callable[[], float] = time.time):
        self.cap = int(cap)
        self.now = now
        self.question: str = ""
        self.positions: dict[str, Position] = {}
        self.order: list[str] = []
        self.sections: dict[str, dict] = {}
        self.section_order: list[str] = []
        self.ledger: list[dict] = []
        # §2.3 never-retry ledger, keyed on (position_a, position_b, version).
        self.click_attempts: dict[tuple[str, str, int], dict] = {}
        # §2.4 emergence inbox.
        self.click_candidates: dict[str, ClickCandidate] = {}
        # §7.1 recognition-inflation budget.
        self.human_contributions: int = 0
        self.clicks_confirmed: int = 0
        self._seq = 0
        self._cand_seq = 0
        self.add_section("field", "FIELD", "#d9cdb6")

    # ---------------- internals ----------------

    def _id(self, prefix: str = "c") -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    def _log(self, kind: str, text: str) -> None:
        clean = str(text or "")
        if len(clean) > LEDGER_TEXT_CAP:
            clean = clean[: LEDGER_TEXT_CAP - 1] + "…"
        self.ledger.append({"kind": kind, "text": clean, "ts": self.now()})
        if len(self.ledger) > LEDGER_CAP:
            del self.ledger[: len(self.ledger) - LEDGER_CAP]

    # ---- position/card addressing -----------------------------------------

    @property
    def cards(self) -> dict[str, Card]:
        """Occupant view, keyed by position id.

        Kept because a position's id *is* its occupant's id: every existing
        reference (`engine.cards[x].state`) still addresses the right idea.
        """
        return {pid: p.occupant for pid, p in self.positions.items()}

    def position(self, position_id: str) -> Position:
        if position_id not in self.positions:
            raise KeyError(position_id)
        return self.positions[position_id]

    def _card(self, card_id: str) -> Card:
        return self.position(card_id).occupant

    def _active_card(self, card_id: str) -> Card:
        """Return an occupant only while its position remains on the live field.

        Archiving/retirement is a terminal lifecycle transition. Keeping mutation
        guards here prevents a stale browser action or delayed worker result
        from silently changing material that the user can no longer see.
        """
        card = self._card(card_id)
        if card.archived:
            raise ValueError("card is archived")
        return card

    def live(self, altitude: int | None = None) -> list[Card]:
        """Occupants visible on the field.

        Folded instances are hidden at their frame's altitude but are NOT
        archived (§1.6): they remain individually recallable and resolvable.
        ``altitude`` filters to exactly one floor (§5, `live()` gains a filter).
        """
        alts = self.altitudes() if altitude is not None else {}
        out = []
        for pid in self.order:
            p = self.positions[pid]
            if p.occupant.archived or p.status in ("vacated", "retired"):
                continue
            if p.status == "folded":
                continue
            if altitude is not None and alts.get(pid) != altitude:
                continue
            out.append(p.occupant)
        return out

    def live_positions(self, altitude: int | None = None) -> list[Position]:
        ids = {c.id for c in self.live(altitude)}
        return [self.positions[pid] for pid in self.order if pid in ids]

    def all_positions(self) -> list[Position]:
        return [self.positions[pid] for pid in self.order]

    # ---------------- seeding / sections ----------------

    def seed(self, question: str) -> None:
        self.question = question or ""
        self._log("ARRIVED", f"seeded · {self.question}")

    def add_section(self, key: str, name: str, color: str = "#d9cdb6") -> dict:
        if key not in self.sections:
            self.section_order.append(key)
            self._log("SECTION", f"section · {name}")
        self.sections[key] = {"key": key, "name": name, "color": color}
        return self.sections[key]

    def rename_section(self, key: str, name: str) -> dict:
        if key not in self.sections:
            raise KeyError(key)
        self.sections[key]["name"] = name
        self._log("SECTION", f"renamed · {name}")
        return self.sections[key]

    def _lightest_section(self) -> str:
        w = self.weights()
        best, best_mass = self.section_order[0], None
        for key in self.section_order:
            m = w[key]["mass"]
            if best_mass is None or m < best_mass:
                best, best_mass = key, m
        return best

    # ---------------- positions / occupants ----------------

    def propose(
        self,
        text: str,
        section: str | None = None,
        kind: str = "claim",
        state: str = "open",
        mass: float = 0.42,
        parents: list[str] | None = None,
        foot: str = "",
        artifact_type: str = "claim",
        *,
        origin: str = "human",
        floor_kind: str = "claim",
        supports: list[str] | None = None,
        external: bool = False,
    ) -> Card:
        """Create a position at floor 0 (or a frame, via `supports`).

        A frame position may only be created here by an explicit human author
        (§1.1: "created by exactly two operations: a confirmed click or explicit
        human authorship"); the click path goes through :meth:`confirm_click`.
        """
        if kind not in KINDS:
            raise ValueError(f"bad kind {kind!r}")
        artifact_type = str(artifact_type or "").strip().lower()
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"bad artifact type {artifact_type!r}")
        if state in ("supported", "refuted"):
            raise ValueError("only resolve() may set supported/refuted")
        if state not in STATES:
            raise ValueError(f"bad state {state!r}")
        if floor_kind not in FLOOR_KINDS:
            raise ValueError(f"bad floor kind {floor_kind!r}")
        if origin not in ORIGINS:
            raise ValueError(f"bad origin {origin!r}")
        supports = list(supports or [])
        if supports and floor_kind != "frame":
            raise ValueError("only frames may have supports")
        if floor_kind == "frame" and not supports:
            raise ValueError("a frame must stand on at least one position")
        for sid in supports:
            self.position(sid)
        if section is None:
            section = self._lightest_section()
        if section not in self.sections:
            self.add_section(section, section.upper())
        born = self.now()
        occurrence = {
            "text": str(text or ""),
            "relation": "origin",
            "foot": str(foot or ""),
            "ts": born,
        }
        card = Card(
            id=self._id(),
            kind=kind,
            text=text,
            section=section,
            mass=_clamp(mass),
            state=state,
            receipt=None,
            foot=foot,
            pinned=False,
            born=born,
            parents=list(parents or []),
            artifact_type=artifact_type,
            occurrence_count=1,
            occurrences=[occurrence],
            first_seen=born,
            last_seen=born,
        )
        position = Position(
            id=card.id,
            floor_kind=floor_kind,
            occupant=card,
            supports=supports,
            provenance=list(parents or []),
            origin=origin,
            external=bool(external),
        )
        position.lineage = self._lineage_for(position)
        self.positions[card.id] = position
        self.order.append(card.id)
        if origin == "human":
            self.human_contributions += 1
        self._log("ARRIVED", text)
        return card

    def _lineage_for(self, p: Position) -> set[str]:
        """§1.6 — the union of floor-0 root position ids a position covers."""
        if p.floor_kind == "claim":
            return {p.id}
        out: set[str] = set()
        for sid in p.supports:
            child = self.positions.get(sid)
            if child is not None:
                out |= child.lineage or self._lineage_for(child)
        return out

    def find_canonical(self, text: str) -> Card | None:
        """Return the earliest live card with the same normalized wording.

        This is deliberately an exact fallback rather than semantic matching:
        Unicode presentation, case, and runs of whitespace are ignored, but
        punctuation and actual wording remain meaningful.
        """
        needle = _normalized_text(text)
        if not needle:
            return None
        for card in self.live():
            if _normalized_text(card.text) == needle:
                return card
        return None

    def record_occurrence(
        self,
        card_id: str,
        text: str,
        *,
        relation: str = "repeat",
        foot: str = "",
    ) -> Card:
        """Attach a repeat or refinement to one canonical visible position.

        §1.2: rephrasing an occupant never touches the structure. The position
        id, ``supports``, ``lineage``, receipt, and ``last_grounded_at`` all
        persist across a refinement.
        """
        card = self._active_card(card_id)
        relation = str(relation or "").strip().lower()
        if relation not in ("repeat", "refinement"):
            raise ValueError("relation must be repeat or refinement")
        clean = str(text or "").strip()
        if not clean:
            raise ValueError("occurrence text required")
        if relation == "refinement" and card.state in ("supported", "refuted"):
            raise ValueError("cannot refine a settled card")

        ts = self.now()
        previous = card.text
        card.occurrence_count = max(1, int(card.occurrence_count)) + 1
        if not card.first_seen:
            card.first_seen = card.born or ts
        card.last_seen = ts
        card.occurrences.append({
            "text": clean,
            "relation": relation,
            "foot": str(foot or ""),
            "ts": ts,
        })
        if len(card.occurrences) > PROVENANCE_CAP:
            del card.occurrences[: len(card.occurrences) - PROVENANCE_CAP]

        if relation == "refinement":
            card.evolution.append({
                "from": previous,
                "to": clean,
                "foot": str(foot or ""),
                "ts": ts,
            })
            if len(card.evolution) > PROVENANCE_CAP:
                del card.evolution[: len(card.evolution) - PROVENANCE_CAP]
            card.text = clean
            self._log("EVOLVED", f"{card.id} · {clean}")
        else:
            self._log("OCCURRED", f"{card.id} · {clean}")
        return card

    def update_proposal(
        self,
        card_id: str,
        text: str,
        *,
        kind: str = "claim",
        foot: str = "",
    ) -> Card:
        """Rewrite an unsettled occupant without pretending it is evidence."""
        card = self._active_card(card_id)
        if card.state in ("supported", "refuted"):
            raise ValueError("cannot rewrite a settled card")
        text = str(text or "").strip()
        if not text:
            raise ValueError("proposal text required")
        if kind not in KINDS:
            raise ValueError(f"bad kind {kind!r}")
        card.text = text
        card.kind = kind
        card.foot = str(foot or "")
        card.state = "open"
        card.receipt = None
        self._log("PROPOSED", card.text)
        return card

    def reopen(self, card_id: str, foot: str | None = None) -> Card:
        """Return an unsettled card to the open frontier.

        This is primarily the failure/cancellation path for inference and
        verification jobs. It cannot erase a settled verdict.
        """
        card = self._active_card(card_id)
        if card.state in ("supported", "refuted"):
            raise ValueError("cannot reopen a settled card")
        card.state = "open"
        if foot is not None:
            card.foot = str(foot)
        self._log("REOPENED", card.text)
        return card

    # ---------------- THE RECEIPT LAW ----------------

    def resolve(
        self,
        card_id: str,
        verdict: str,
        receipt: str,
        text: str | None = None,
        foot: str | None = None,
    ) -> Card:
        """The only path to supported/refuted. Receipt is mandatory.

        SPEC §1.1: only *claims* are resolvable. A frame position is refused
        outright — its support comes from the floor and nowhere else.

        SPEC §5: the parent-archiving branch is DELETED. A receipt landing on a
        claim can no longer consume other positions; evidence climbs, it does
        not destroy. The frame above is re-scored on the next read of
        :meth:`frame_support`, which recomputes from the floor every time.

        `text`/`foot` let a caller supply the resolved wording *as part of the
        same transaction*: if the text could be rewritten after resolve()
        returned, the receipt and the RESOLVED ledger entry would attest to
        wording the card no longer carries.
        """
        position = self.position(card_id)
        card = self._active_card(card_id)
        if position.floor_kind == "frame":
            raise ValueError(
                "only claims can be supported or refuted; a frame's support is "
                "computed from the floor below"
            )
        if card.artifact_type != "claim":
            raise ValueError("only claims can be supported or refuted")
        if verdict not in ("supported", "refuted"):
            raise ValueError(f"bad verdict {verdict!r}")
        if not receipt or not str(receipt).strip():
            raise ValueError("receipt required")
        if text is not None:
            text = str(text).strip()
            if not text:
                raise ValueError("resolve text, if given, must be non-empty")
            card.text = text
        if foot is not None:
            card.foot = str(foot)
        card.state = verdict
        card.receipt = str(receipt).strip()
        position.last_grounded_at = self.now()
        self._log("RESOLVED", f"{verdict} · {card.text}")
        self._propagate_grounding(position.id)
        return card

    def _propagate_grounding(self, position_id: str) -> None:
        """§1.5 — push `last_grounded_at = max` up the ladder.

        This is a *cache of a maximum*, not stored support: the support summary
        itself is recomputed from the floor on every call. Nothing here can
        assert a state the floor does not justify.
        """
        stamp = self.positions[position_id].last_grounded_at
        if stamp is None:
            return
        for parent in self.all_positions():
            if parent.floor_kind != "frame":
                continue
            if position_id not in parent.supports:
                continue
            if parent.last_grounded_at is None or parent.last_grounded_at < stamp:
                parent.last_grounded_at = stamp
                self._propagate_grounding(parent.id)

    def judge(self, card_id: str, verdict: str) -> Card | list[Card]:
        position = self.position(card_id)
        card = self._active_card(card_id)
        if position.floor_kind == "frame":
            raise ValueError("only claims can be judged true or false")
        if card.artifact_type != "claim":
            raise ValueError("only claims can be judged true or false")
        if card.state in ("supported", "refuted"):
            raise ValueError("cannot judge a settled card")
        if verdict in ("yes", "no"):
            mapped = "supported" if verdict == "yes" else "refuted"
            out = self.resolve(card_id, mapped, f"human judgment · {verdict}")
            self._log("JUDGED", f"{verdict} · {card.text}")
            return out
        if verdict != "unknown":
            raise ValueError(f"bad verdict {verdict!r}")
        card.state = "open"
        card.receipt = None
        kids = [
            self.propose(
                text=f"{card.text} — {suffix}",
                section=card.section,
                kind="claim",
                state="needs_human",
                mass=_clamp(card.mass * 0.6),
                parents=[card.id],
                artifact_type="question",
                # Engine-generated prompts are not contributions: they must not
                # inflate the §7.1 click budget they were never asked about.
                origin="derivation",
            )
            for suffix in ("what would make this true?", "what would make this false?")
        ]
        self._log("JUDGED", f"unknown · {card.text}")
        return kids

    # ---------------- derived quantities (§1.5) ----------------

    def altitudes(self) -> dict[str, int]:
        """§1.2 — altitude is derived, never stored. Memoized walk."""
        memo: dict[str, int] = {}

        def walk(pid: str, seen: frozenset[str]) -> int:
            if pid in memo:
                return memo[pid]
            p = self.positions.get(pid)
            if p is None or p.floor_kind == "claim" or not p.supports:
                memo[pid] = 0
                return 0
            if pid in seen:
                # A cycle cannot exist through the public API; if a hand-edited
                # snapshot produces one, refuse to loop rather than crash.
                return 0
            below = seen | {pid}
            height = 1 + max(
                (walk(sid, below) for sid in p.supports if sid in self.positions),
                default=-1,
            )
            memo[pid] = height
            return height

        for pid in self.order:
            walk(pid, frozenset())
        return memo

    def altitude(self, position_id: str) -> int:
        return self.altitudes().get(position_id, 0)

    def frame_support(self, position_id: str) -> dict:
        """§1.5 — COMPUTED from the floor below, never stored.

        Returns the multiset of support states over the frame's floor, plus a
        propagated ``last_grounded_at``. Recomputed on every call; there is no
        cached copy anywhere, which is why no layer can drift from the layer
        below.
        """
        p = self.position(position_id)
        if p.floor_kind != "frame":
            raise ValueError("frame_support() applies to frames only")
        counts = {"supported": 0, "refuted": 0, "open": 0, "needs_human": 0,
                  "testing": 0}
        grounded: float | None = None
        for sid in p.supports:
            child = self.positions.get(sid)
            if child is None:
                continue
            if child.floor_kind == "frame":
                sub = self.frame_support(sid)
                for key, value in sub["counts"].items():
                    counts[key] = counts.get(key, 0) + value
                stamp = sub["last_grounded_at"]
            else:
                counts[child.occupant.state] = counts.get(child.occupant.state, 0) + 1
                stamp = child.last_grounded_at
            if stamp is not None and (grounded is None or stamp > grounded):
                grounded = stamp
        return {
            "counts": counts,
            "supported": counts["supported"],
            "refuted": counts["refuted"],
            "open": counts["open"] + counts["needs_human"] + counts["testing"],
            "cracked": counts["refuted"] > 0,
            "speculative": counts["supported"] == 0,
            "summary": (
                f"{counts['supported']}✓ {counts['refuted']}✗ "
                f"{counts['open'] + counts['needs_human'] + counts['testing']}○"
            ),
            "last_grounded_at": grounded,
        }

    def lineage_mass(self, position_id: str) -> float:
        """§1.6 — display mass from the UNION of lineage roots' masses.

        Two frames sharing roots cannot double-count them; importance tracks
        unique ground covered, not ladder height.
        """
        p = self.position(position_id)
        if p.floor_kind == "claim":
            return p.occupant.mass
        return sum(
            self.positions[rid].occupant.mass
            for rid in sorted(p.lineage)
            if rid in self.positions
        )

    def descend(self, position_id: str) -> list[Position]:
        """§3.1/§3.2 — the frame's floor, folded instances included."""
        p = self.position(position_id)
        return [self.positions[sid] for sid in p.supports if sid in self.positions]

    def staleness(self, position_id: str) -> float | None:
        """§1.2 — ``now − last_grounded_at``; None when never grounded."""
        p = self.position(position_id)
        stamp = p.last_grounded_at
        if stamp is None and p.floor_kind == "frame":
            stamp = self.frame_support(position_id)["last_grounded_at"]
        if stamp is None:
            return None
        return self.now() - stamp

    # ---------------- the click, upward (§1.3) ----------------

    def check_click_gates(
        self, a_id: str, b_id: str, proposal: ClickProposal
    ) -> None:
        """Run the three deterministic §1.3 gates. Raises :class:`GateFailure`.

        These run engine-side on the worker's returned text — never in the
        prompt, never model-judged. A failed gate emits nothing; the caller
        writes a ``gate_failed`` attempt row (§2.3).
        """
        a = self.position(a_id)
        b = self.position(b_id)
        abstraction = " ".join(str(proposal.abstraction or "").split())
        if not abstraction:
            raise GateFailure("generativity", "empty abstraction")

        # Gate 1 — generativity. Reject a lexical merge: if every content word
        # of X is borrowed from A ∪ B, X names nothing the instances did not
        # already say. This does NOT demand invented content (§1.3 nuance /
        # §6): a frame may faithfully name shared structure, it just may not be
        # built entirely out of the instances' own words.
        x_words = content_words(abstraction)
        if not x_words:
            raise GateFailure("generativity", "abstraction has no content words")
        borrowed = content_words(a.occupant.text) | content_words(b.occupant.text)
        if x_words <= borrowed:
            raise GateFailure(
                "generativity",
                "abstraction is a lexical merge of its instances",
            )

        # Gate 2 — recoverability. One clause per instance, and a specializer
        # that merely restates X recovers nothing.
        for label, spec in (
            ("specializer_a", proposal.specializer_a),
            ("specializer_b", proposal.specializer_b),
        ):
            clause = " ".join(str(spec or "").split())
            if not clause:
                raise GateFailure("recoverability", f"{label} is empty")
            spec_words = content_words(clause)
            if not spec_words:
                raise GateFailure("recoverability", f"{label} has no content words")
            if spec_words <= x_words:
                raise GateFailure(
                    "recoverability", f"{label} merely restates the abstraction"
                )

        # Gate 3 — scope boundary. An abstraction that excludes nothing
        # explains nothing.
        boundary = " ".join(str(proposal.scope_boundary or "").split())
        if not boundary or not content_words(boundary):
            raise GateFailure("scope_boundary", "no excluded cases stated")

    def record_attempt(
        self,
        a_id: str,
        b_id: str,
        outcome: str,
        *,
        operation_version: int | None = None,
    ) -> dict:
        """§2.3 — write the never-retry row. Memory is separate from output."""
        if outcome not in ATTEMPT_OUTCOMES:
            raise ValueError(f"bad attempt outcome {outcome!r}")
        key_a, key_b = _pair_key(a_id, b_id)
        version = (
            operation_version
            if operation_version is not None
            else self.current_operation_version(a_id, b_id)
        )
        row = {
            "position_a": key_a,
            "position_b": key_b,
            "operation_version": version,
            "outcome": outcome,
            "attempted_at": self.now(),
        }
        self.click_attempts[(key_a, key_b, version)] = row
        return row

    def current_operation_version(self, a_id: str, b_id: str) -> int:
        key_a, key_b = _pair_key(a_id, b_id)
        versions = [
            v for (pa, pb, v) in self.click_attempts
            if pa == key_a and pb == key_b
        ]
        return max(versions) if versions else 1

    def pair_consumed(self, a_id: str, b_id: str) -> bool:
        """§2.3 — has this pair been settled at its current version?

        A ``failed`` row (provider outage) does not consume the pair: an outage
        must never permanently suppress a legitimate future recognition.
        """
        key_a, key_b = _pair_key(a_id, b_id)
        version = self.current_operation_version(a_id, b_id)
        row = self.click_attempts.get((key_a, key_b, version))
        if row is None:
            return False
        return row["outcome"] not in NON_CONSUMING_OUTCOMES

    def reconsider_pair(self, a_id: str, b_id: str) -> dict:
        """§2.3 — the explicit, provenance-visible retry door.

        A deliberate human act. It inserts a row at ``operation_version + 1``;
        there is no automatic reopening path anywhere in the engine.
        """
        self.position(a_id)
        self.position(b_id)
        key_a, key_b = _pair_key(a_id, b_id)
        if not self.pair_consumed(a_id, b_id):
            raise ValueError("pair is not consumed; nothing to reconsider")
        version = self.current_operation_version(a_id, b_id) + 1
        # Stored under its own version so `pair_consumed` reports False again,
        # while the paper trail of every prior version survives intact.
        row = self.record_attempt(
            a_id, b_id, "reconsidered", operation_version=version
        )
        self._log("RECONSIDERED", f"{key_a} × {key_b} · v{version}")
        return row

    # ---------------- the scanner (§2.2) ----------------

    def scan_eligible(self, position_id: str) -> bool:
        """§2.2 — who may be automatic fuel. Most positions may not.

        Claims only; cargo types never scan; recall-quarantined positions never
        scan; derived-ungrounded claims never scan; a frame scans only after its
        floor has a supported member or a human pin.
        """
        p = self.positions.get(position_id)
        if p is None or p.status != "live" or p.occupant.archived:
            return False
        if p.occupant.pinned:
            return False
        if p.external and not p.pinned_by_human:
            return False
        if p.floor_kind == "claim":
            if p.occupant.artifact_type != "claim":
                return False
            if p.occupant.state != "open":
                return False
            if (
                p.origin == "derivation"
                and p.last_grounded_at is None
                and not p.pinned_by_human
            ):
                return False
            return True
        support = self.frame_support(position_id)
        return support["supported"] > 0 or p.pinned_by_human

    def scan_candidates(
        self, embedding_cosine: Callable[[str, str], float] | None = None
    ) -> tuple[str, str] | None:
        """§2.2 — select AT MOST ONE unattempted pair per tick, or nothing.

        The background loop materializes nothing: this returns a pair to *ask
        about*, never a card. Embeddings are a bounded ranking index only —
        below ``CLICK_FLOOR`` the tick does nothing, and a cosine above it is
        never on its own an acceptance or a materialization trigger.
        """
        alts = self.altitudes()
        eligible = [pid for pid in self.order if self.scan_eligible(pid)]
        best: tuple[str, str] | None = None
        best_score: float | None = None
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                a_id, b_id = eligible[i], eligible[j]
                if self.pair_consumed(a_id, b_id):
                    continue
                if not self._comparable(a_id, b_id, alts):
                    continue
                score = (
                    embedding_cosine(a_id, b_id)
                    if embedding_cosine is not None
                    else 0.0
                )
                if score < CLICK_FLOOR:
                    continue
                # Prefer same-altitude pairs so the ladder grows level by level.
                rank = (0 if alts.get(a_id) == alts.get(b_id) else 1, -score)
                if best_score is None or rank < best_score:
                    best, best_score = (a_id, b_id), rank
        return best

    def _comparable(self, a_id: str, b_id: str, alts: dict[str, int]) -> bool:
        """§2.2 anti-recursion invariant. The ladder must not grind on itself."""
        a, b = self.positions[a_id], self.positions[b_id]
        if abs(alts.get(a_id, 0) - alts.get(b_id, 0)) > 1:
            return False
        # Never against its own descendants: overlapping lineage means one
        # already covers ground the other stands on.
        if a.floor_kind == "frame" or b.floor_kind == "frame":
            if a.lineage & b.lineage:
                return False
        return True

    def propose_click(
        self, a_id: str, b_id: str, proposal: ClickProposal
    ) -> ClickCandidate:
        """§2.4 — gate a proposal into the emergence inbox. Never into the field.

        This is the sole entry point for both the scanner and the human-initiated
        ``propose_click`` request. A gate failure emits nothing and writes an
        attempt row; an inbox at capacity refuses rather than growing.
        """
        a = self.position(a_id)
        b = self.position(b_id)
        if a_id == b_id:
            raise ValueError("a position cannot click with itself")
        if a.occupant.archived or b.occupant.archived:
            raise ValueError("card is archived")
        if self.pair_consumed(a_id, b_id):
            raise ValueError("pair already attempted; use reconsider_pair()")
        if len(self.open_candidates()) >= INBOX_CAP:
            raise ValueError("emergence inbox is full")
        try:
            self.check_click_gates(a_id, b_id, proposal)
        except GateFailure:
            self.record_attempt(a_id, b_id, "gate_failed")
            raise
        self._cand_seq += 1
        candidate = ClickCandidate(
            id=f"cand{self._cand_seq}",
            position_a=a_id,
            position_b=b_id,
            proposal=proposal,
            created_at=self.now(),
        )
        self.click_candidates[candidate.id] = candidate
        self._log("PROPOSED", f"click · {proposal.abstraction}")
        return candidate

    def open_candidates(self) -> list[ClickCandidate]:
        self.expire_candidates()
        return [c for c in self.click_candidates.values() if c.status == "open"]

    def expire_candidates(self) -> list[ClickCandidate]:
        """§2.4 — 7 days unacted auto-expires the candidate. No retry."""
        now = self.now()
        expired = []
        for candidate in self.click_candidates.values():
            if candidate.status != "open":
                continue
            if now - candidate.created_at < CLICK_TTL:
                continue
            candidate.status = "expired"
            self.record_attempt(
                candidate.position_a, candidate.position_b, "expired"
            )
            expired.append(candidate)
        return expired

    def decline_click(self, candidate_id: str) -> ClickCandidate:
        candidate = self.click_candidates[candidate_id]
        if candidate.status != "open":
            raise ValueError("candidate is not open")
        candidate.status = "declined"
        self.record_attempt(candidate.position_a, candidate.position_b, "declined")
        self._log("DECLINED", candidate.proposal.abstraction)
        return candidate

    def click_budget_remaining(self) -> int:
        """§7.1 — at most one confirmed click per 5 human contributions."""
        allowed = self.human_contributions // CLICK_BUDGET_PER_CONTRIBUTIONS
        return max(0, allowed - self.clicks_confirmed)

    def confirm_click(
        self,
        candidate_id: str,
        *,
        confirmed_by: str,
        text: str | None = None,
        section: str | None = None,
    ) -> Position:
        """§1.6 — accept a click: create an OPEN frame and FOLD the instances.

        The frame is created **OPEN with no receipt**. Accepting a click means
        "organize these together", not "this proposition is true". Structural
        recognition can never masquerade as epistemic settlement: the
        confirmation is recorded as provenance (``confirmed_by``,
        ``confirmed_at``, human wording if edited), and the frame's support is
        computed from the floor by :meth:`frame_support` — never stored here.
        """
        candidate = self.click_candidates[candidate_id]
        if candidate.status != "open":
            raise ValueError("candidate is not open")
        if not str(confirmed_by or "").strip():
            raise ValueError("confirmation must name a human")
        if self.click_budget_remaining() <= 0:
            raise ValueError("click budget exhausted; compression cannot outrun input")
        a = self.position(candidate.position_a)
        b = self.position(candidate.position_b)
        wording = str(text or candidate.proposal.abstraction).strip()
        if not wording:
            raise ValueError("frame text required")
        frame_card = self.propose(
            text=wording,
            section=section or a.occupant.section,
            kind="claim",
            state="open",             # never supported: no receipt exists (§1.6)
            mass=_clamp((a.occupant.mass + b.occupant.mass) / 2),
            foot=f"frame · confirmed by {confirmed_by}",
            artifact_type="claim",
            origin="click",
            floor_kind="frame",
            supports=[a.id, b.id],
        )
        frame = self.position(frame_card.id)
        frame.confirmed_by = str(confirmed_by).strip()
        frame.confirmed_at = self.now()
        frame.specializers = {
            a.id: candidate.proposal.specializer_a,
            b.id: candidate.proposal.specializer_b,
        }
        frame.scope_boundary = candidate.proposal.scope_boundary
        # §1.5: the frame inherits nothing it can assert; last_grounded_at is
        # the max over the floor, which is exactly what the floor already says.
        frame.last_grounded_at = self.frame_support(frame.id)["last_grounded_at"]
        self.fold(a.id, frame.id)
        self.fold(b.id, frame.id)
        candidate.status = "accepted"
        self.record_attempt(a.id, b.id, "clicked")
        self.clicks_confirmed += 1
        self._log("CLICKED", f"{frame.occupant.text}")
        return frame

    def fold(self, position_id: str, frame_id: str) -> Position:
        """§1.6 — folded, NOT archived. A fold is a lens, not a merge."""
        p = self.position(position_id)
        frame = self.position(frame_id)
        if frame.floor_kind != "frame":
            raise ValueError("positions fold under frames only")
        if p.occupant.archived:
            raise ValueError("card is archived")
        p.status = "folded"
        p.folded_under = frame_id
        return p

    def unfold(self, frame_id: str) -> list[Position]:
        """§1.6 — always cheap: instances return, the frame position is VACATED.

        The position record persists with its history; nothing is deleted.
        """
        frame = self.position(frame_id)
        if frame.floor_kind != "frame":
            raise ValueError("only frames unfold")
        released = []
        for sid in frame.supports:
            child = self.positions.get(sid)
            if child is not None and child.folded_under == frame_id:
                child.folded_under = None
                child.status = "live"
                released.append(child)
        frame.status = "vacated"
        self._log("UNFOLDED", frame.occupant.text)
        return released

    def vacate_empty_frames(self) -> list[Position]:
        """§1.6 — a frame that loses all instances is vacated. No semantic shells."""
        out = []
        for p in self.all_positions():
            if p.floor_kind != "frame" or p.status != "live":
                continue
            remaining = [
                sid for sid in p.supports
                if sid in self.positions
                and not self.positions[sid].occupant.archived
                and self.positions[sid].status != "retired"
            ]
            if not remaining:
                p.status = "vacated"
                out.append(p)
        return out

    # ---------------- derivation, downward (§1.4) ----------------

    def derive(
        self,
        frame_id: str,
        proposals: list[dict],
        *,
        origin: str = "derivation",
    ) -> list[Card]:
        """§1.4 — the explicit downward operator. NEVER background-initiated.

        Asks nothing and asserts nothing: it creates up to ``DERIVE_CAP`` claim
        positions beneath the frame with ``state='open'`` and
        ``last_grounded_at=None`` — visible as ungrounded slots, structure
        awaiting evidence. Each proposal must carry a falsification hint (what
        receipt would flip it); a proposal without one is not receipt-checkable
        and is refused.

        Evidence then climbs: a receipt flips a derived claim via
        :meth:`resolve`, and the flip re-scores the frame. That is the only path
        by which derivation ever changes anything's state.
        """
        frame = self.position(frame_id)
        if frame.floor_kind != "frame":
            raise ValueError("derive() applies to frames only")
        if frame.status in ("vacated", "retired"):
            raise ValueError("cannot derive beneath a vacated frame")
        accepted: list[Card] = []
        for raw in proposals[:DERIVE_CAP]:
            text = " ".join(str((raw or {}).get("text") or "").split())
            hint = " ".join(str((raw or {}).get("falsification") or "").split())
            if not text:
                continue
            if not hint:
                raise ValueError(
                    "each derived claim needs a falsification hint: "
                    "what receipt would flip this"
                )
            card = self.propose(
                text=text,
                section=frame.occupant.section,
                kind="claim",
                state="open",
                mass=_clamp(frame.occupant.mass * 0.6),
                foot=f"derived · would be flipped by: {hint}",
                artifact_type="claim",
                origin=origin,
            )
            child = self.position(card.id)
            child.last_grounded_at = None        # ungrounded slot, by construction
            frame.supports.append(card.id)
            frame.lineage = self._lineage_for(frame)
            accepted.append(card)
        if accepted:
            self._log("DERIVED", f"{frame.occupant.text} · {len(accepted)} slots")
        return accepted

    def ungrounded_slots(self, frame_id: str) -> list[Position]:
        """§1.4/§3.1 — derived positions still awaiting their first receipt."""
        frame = self.position(frame_id)
        return [
            self.positions[sid]
            for sid in frame.supports
            if sid in self.positions
            and self.positions[sid].origin == "derivation"
            and self.positions[sid].last_grounded_at is None
        ]

    # ---------------- ordinary mutators ----------------

    def keep(self, card_id: str) -> Card:
        card = self._active_card(card_id)
        card.pinned = not card.pinned
        self.position(card_id).pinned_by_human = card.pinned
        self._log("KEPT" if card.pinned else "UNKEPT", card.text)
        return card

    def kill(self, card_id: str) -> Card:
        card = self._active_card(card_id)
        card.archived = True
        self.position(card_id).status = "retired"
        self._log("KILLED", card.text)
        return card

    def move(self, card_id: str, section: str) -> Card:
        card = self._active_card(card_id)
        if section not in self.sections:
            self.add_section(section, section.upper())
        card.section = section
        self._log("MOVED", f"{card.text} → {self.sections[section]['name']}")
        return card

    def request_verify(self, card_id: str) -> Card:
        card = self._active_card(card_id)
        if card.state != "open":
            raise ValueError(f"cannot verify from state {card.state!r}")
        card.state = "testing"
        self._log("TESTING", f"verify · {card.text}")
        return card

    # ---------------- ranking ----------------

    def frontier(self) -> list[Card]:
        cs = [c for c in self.live() if c.state in ("open", "needs_human")]
        return sorted(cs, key=lambda c: (0 if c.state == "needs_human" else 1, -c.mass))

    def enforce_cap(self) -> Card | None:
        """§1.6 — ordered retirement; a fold counts as ONE unit at its altitude.

        Recognition is the only operation that structurally reduces field
        pressure, so the system is incentivized toward compression rather than
        expansion: folding two claims under a frame takes three positions down
        to one billable unit.

        Retirement order is an explicit invariant: never retire positions with
        dependents in ``supports``; never retire folded instances; never retire
        human roots or ``needs_human``; prefer unlinked, low-mass,
        machine-origin cargo — in that order.
        """
        billable = self._billable_positions()
        capacity = self.cap * max(1, len(self.sections))
        if len(billable) <= capacity:
            return None

        depended_on: set[str] = set()
        for p in self.all_positions():
            if p.status in ("vacated", "retired"):
                continue
            depended_on.update(p.supports)

        eligible = [
            p for p in billable
            if not p.occupant.pinned
            and p.occupant.state != "needs_human"
            and p.status == "live"
            and p.id not in depended_on
            and p.floor_kind == "claim"
        ]
        if not eligible:
            return None

        machine = [p for p in eligible if p.origin != "human"]
        if machine:
            candidates = machine
        else:
            root_counts: dict[str, int] = {}
            for p in billable:
                if not p.provenance:
                    root_counts[p.occupant.section] = (
                        root_counts.get(p.occupant.section, 0) + 1
                    )
            candidates = [
                p for p in eligible
                if p.provenance or root_counts.get(p.occupant.section, 0) > 1
            ]
        if not candidates:
            return None

        victim = min(
            candidates,
            key=lambda p: (
                0 if p.occupant.artifact_type != "claim" else 1,  # cargo first
                p.occupant.mass,
                p.occupant.born,
            ),
        )
        victim.occupant.archived = True
        victim.status = "retired"
        self._log("RETIRED", victim.occupant.text)
        self.vacate_empty_frames()
        return victim.occupant

    def _billable_positions(self) -> list[Position]:
        """Cap pressure counts a fold as one unit at the frame's altitude."""
        out = []
        for pid in self.order:
            p = self.positions[pid]
            if p.occupant.archived or p.status in ("folded", "vacated", "retired"):
                continue
            out.append(p)
        return out

    # ---------------- reporting ----------------

    def _digest_card(self, card: Card) -> dict:
        p = self.positions.get(card.id)
        out = {
            "id": card.id,
            "kind": card.kind,
            "artifact_type": card.artifact_type,
            "text": card.text,
            "section": card.section,
            "state": card.state,
            "mass": card.mass,
            "pinned": card.pinned,
            "occurrence_count": card.occurrence_count,
            "first_seen": card.first_seen,
            "last_seen": card.last_seen,
            "parents": list(card.parents),
        }
        if p is not None:
            out["floor_kind"] = p.floor_kind
            out["supports"] = list(p.supports)
            out["last_grounded_at"] = p.last_grounded_at
            if p.floor_kind == "frame":
                out["support_summary"] = self.frame_support(p.id)["summary"]
                out["lineage_mass"] = self.lineage_mass(p.id)
        return out

    @staticmethod
    def _digest_rank(card: Card) -> tuple:
        return (
            0 if card.pinned else 1,
            -card.occurrence_count,
            -card.mass,
            -card.last_seen,
            card.born,
            card.id,
        )

    def digest(self) -> dict:
        """Derive a compact thematic view without adding another source of truth."""
        live = self.live()
        ranked = sorted(live, key=self._digest_rank)
        by_section: dict[str, list[Card]] = {}
        for card in live:
            by_section.setdefault(card.section, []).append(card)

        themes = []
        for position, key in enumerate(self.section_order):
            cards = by_section.get(key, [])
            if not cards:
                continue
            section = self.sections[key]
            ordered = sorted(cards, key=self._digest_rank)
            artifact_types: dict[str, int] = {}
            for card in cards:
                artifact_types[card.artifact_type] = (
                    artifact_types.get(card.artifact_type, 0) + 1
                )
            themes.append({
                "key": key,
                "name": section.get("name", key),
                "color": section.get("color", "#d9cdb6"),
                "card_count": len(cards),
                "occurrence_count": sum(c.occurrence_count for c in cards),
                "mass": sum(c.mass for c in cards),
                "artifact_types": artifact_types,
                "top_ideas": [self._digest_card(c) for c in ordered[:3]],
                "_position": position,
            })
        themes.sort(key=lambda item: (
            -item["occurrence_count"],
            -item["mass"],
            item["_position"],
        ))
        for theme in themes:
            del theme["_position"]

        def cards_of(artifact_type: str) -> list[dict]:
            return [
                self._digest_card(card)
                for card in ranked
                if card.artifact_type == artifact_type
            ]

        # §7.3(b): folded instances stay in recurring_ideas regardless of
        # altitude, so repetition surfaces through the floor and a frame cannot
        # be used as a hiding place.
        recurring_sources = sorted(
            (
                p.occupant for p in self.all_positions()
                if not p.occupant.archived and p.status in ("live", "folded")
            ),
            key=self._digest_rank,
        )
        out = {
            "themes": themes,
            "recurring_ideas": [
                self._digest_card(card)
                for card in recurring_sources
                if card.occurrence_count > 1
            ],
            "open_questions": cards_of("question"),
            "decisions": cards_of("decision"),
            "constraints": cards_of("constraint"),
            "experiments": cards_of("experiment"),
            "tasks": cards_of("task"),
        }
        # §3.2: `between_ideas` becomes `frames`, ranked by altitude then
        # lineage mass, truncated to 3.
        alts = self.altitudes()
        frames = [
            p for p in self.all_positions()
            if p.floor_kind == "frame" and p.status == "live"
            and not p.occupant.archived
        ]
        frames.sort(key=lambda p: (-alts.get(p.id, 0), -self.lineage_mass(p.id)))
        if frames:
            out["frames"] = [self._digest_card(p.occupant) for p in frames[:3]]
        return out

    def weights(self) -> dict[str, dict]:
        out = {
            k: {"mass": 0.0, "count": 0, "norm": 0.0} for k in self.section_order
        }
        for c in self.live():
            if c.section not in out:
                out[c.section] = {"mass": 0.0, "count": 0, "norm": 0.0}
            out[c.section]["mass"] += c.mass
            out[c.section]["count"] += 1
        total = sum(v["mass"] for v in out.values())
        for v in out.values():
            v["norm"] = (v["mass"] / total) if total > 0 else 0.0
        return out

    def harvest(self, altitude: int | None = None, max_items: int = 12) -> dict:
        """§3.3 — the decision-ready brief. Hard cap per section.

        A brief that cannot be read in two minutes is a dump; full state
        remains as :meth:`state`, a debugging surface rather than a deliverable.
        """
        alts = self.altitudes()
        floor = 0 if altitude is None else int(altitude)
        cap = max(1, int(max_items))

        visible = [
            p for p in self.all_positions()
            if p.status == "live" and not p.occupant.archived
        ]

        # The spine: positions at or above `altitude`, ranked by lineage mass,
        # each carrying its specializers and support summary so a reader can
        # descend in prose.
        spine = [p for p in visible if alts.get(p.id, 0) >= floor]
        spine.sort(key=lambda p: (-self.lineage_mass(p.id), p.occupant.born))
        spine_out = []
        for p in spine[:cap]:
            item = self._digest_card(p.occupant)
            item["altitude"] = alts.get(p.id, 0)
            if p.floor_kind == "frame":
                item["specializers"] = dict(p.specializers)
                item["scope_boundary"] = p.scope_boundary
            spine_out.append(item)

        cracks = []
        for p in visible:
            if p.floor_kind != "frame":
                continue
            support = self.frame_support(p.id)
            if support["cracked"]:
                item = self._digest_card(p.occupant)
                item["support_summary"] = support["summary"]
                cracks.append(item)

        stale = []
        for p in visible:
            age = self.staleness(p.id)
            item = self._digest_card(p.occupant)
            item["staleness"] = age
            if age is None:
                item["never_grounded"] = True
            stale.append((age if age is not None else float("inf"), item))
        stale.sort(key=lambda pair: -pair[0])

        # Cruxes: question cargo, plus derived-ungrounded claims under
        # high-mass frames — the receipts most worth going and getting.
        cruxes = [
            self._digest_card(p.occupant)
            for p in visible
            if p.occupant.artifact_type == "question"
        ]
        frames_by_mass = sorted(
            (p for p in visible if p.floor_kind == "frame"),
            key=lambda p: -self.lineage_mass(p.id),
        )
        for frame in frames_by_mass:
            for slot in self.ungrounded_slots(frame.id):
                item = self._digest_card(slot.occupant)
                item["under_frame"] = frame.id
                item["ungrounded"] = True
                cruxes.append(item)

        def cargo(artifact_type: str) -> list[dict]:
            return [
                self._digest_card(p.occupant)
                for p in visible
                if p.occupant.artifact_type == artifact_type
            ][:cap]

        brief = {
            "question": self.question,
            "altitude": floor,
            "max_altitude": max(alts.values()) if alts else 0,
            "sections": [dict(self.sections[k]) for k in self.section_order],
            "spine": spine_out,
            "cracks": cracks[:cap],
            "stale": [item for _, item in stale[:cap]],
            "cruxes": cruxes[:cap],
            "decisions": cargo("decision"),
            "constraints": cargo("constraint"),
            "experiments": cargo("experiment"),
            "unresolved": [
                self._digest_card(p.occupant)
                for p in visible
                if p.occupant.state == "needs_human"
            ][:cap],
            "changed": [
                dict(entry) for entry in self.ledger
                if entry["kind"] in ("CLICKED", "RESOLVED", "DERIVED", "UNFOLDED")
            ][-cap:],
        }
        self._log("HARVESTED", f"{len(spine_out)} positions")
        return brief

    def state(self) -> dict:
        return {
            "question": self.question,
            "cap": self.cap,
            "capacity": self.cap * max(1, len(self.sections)),
            "seq": self._seq,
            "cand_seq": self._cand_seq,
            "sections": [dict(self.sections[k]) for k in self.section_order],
            # `cards` stays the wire format: a position's id IS its occupant's
            # id, so every existing consumer keeps addressing the right idea.
            "cards": [self.positions[i].occupant.to_dict() for i in self.order],
            "positions": [self.positions[i].to_dict() for i in self.order],
            "click_attempts": [dict(row) for row in self.click_attempts.values()],
            "click_candidates": [
                c.to_dict() for c in self.click_candidates.values()
            ],
            "human_contributions": self.human_contributions,
            "clicks_confirmed": self.clicks_confirmed,
            "weights": self.weights(),
            "digest": self.digest(),
            "ledger": list(self.ledger),
        }

    # ---------------- persistence ----------------

    @staticmethod
    def from_state(d: dict, now: Callable[[], float] = time.time) -> "Engine":
        e = Engine(cap=int(d.get("cap", 12)), now=now)
        e.question = d.get("question", "")
        e.sections = {}
        e.section_order = []
        for s in d.get("sections", []):
            e.sections[s["key"]] = {
                "key": s["key"],
                "name": s.get("name", s["key"]),
                "color": s.get("color", "#d9cdb6"),
            }
            e.section_order.append(s["key"])
        if not e.section_order:
            e.add_section("field", "FIELD", "#d9cdb6")
        e.positions = {}
        e.order = []

        structure = {
            row["id"]: row
            for row in (d.get("positions") or [])
            if isinstance(row, dict) and row.get("id")
        }
        for cd in d.get("cards", []):
            c = Card.from_dict(cd)
            row = structure.get(c.id, {})
            floor_kind = str(row.get("floor_kind") or "claim")
            if floor_kind not in FLOOR_KINDS:
                floor_kind = "claim"
            # THE LAW SURVIVES DESERIALIZATION. A terminal state is only
            # meaningful if the receipt that justified it came back with it. A
            # snapshot claiming supported/refuted with no receipt was either
            # hand-edited or written by a code path that bypassed resolve();
            # either way it is not evidence, so it reverts to needs_human. A
            # frame can never be terminal at all (§1.1) — including any legacy
            # `kind == "synthesis"` whose receipt did not migrate.
            if c.state in ("supported", "refuted") and (
                floor_kind != "claim"
                or c.artifact_type != "claim"
                or not (c.receipt or "").strip()
            ):
                c.state = "needs_human"
                c.receipt = None
            status = str(row.get("status") or "live")
            if status not in POSITION_STATUSES:
                status = "live"
            if c.archived and status == "live":
                status = "retired"
            origin = str(row.get("origin") or "human")
            if origin not in ORIGINS:
                origin = "human"
            position = Position(
                id=c.id,
                floor_kind=floor_kind,
                occupant=c,
                supports=[str(x) for x in (row.get("supports") or [])],
                # §5 backfill: legacy `parents` map to PROVENANCE, never to
                # `supports` — combination provenance is not identity
                # recognition, and fabricating support edges would seed the
                # ladder with exactly the structure the gates exist to prevent.
                provenance=[str(x) for x in (row.get("provenance") or c.parents)],
                lineage=set(row.get("lineage") or []),
                origin=origin,
                status=status,
                folded_under=row.get("folded_under"),
                last_grounded_at=row.get("last_grounded_at"),
                external=bool(row.get("external", False)),
                pinned_by_human=bool(row.get("pinned_by_human", c.pinned)),
                confirmed_by=row.get("confirmed_by"),
                confirmed_at=row.get("confirmed_at"),
                specializers=dict(row.get("specializers") or {}),
                scope_boundary=str(row.get("scope_boundary") or ""),
            )
            e.positions[c.id] = position
            e.order.append(c.id)

        # Drop dangling support edges, then recompute lineage from the floor:
        # lineage is derived structure and a snapshot's copy is never trusted
        # over what the floor actually says.
        for p in e.positions.values():
            p.supports = [sid for sid in p.supports if sid in e.positions]
            if p.floor_kind == "frame" and not p.supports:
                p.status = "vacated"
        for pid in e.order:
            e.positions[pid].lineage = e._lineage_for(e.positions[pid])

        for row in (d.get("click_attempts") or []):
            if not isinstance(row, dict):
                continue
            try:
                key = (
                    str(row["position_a"]),
                    str(row["position_b"]),
                    int(row.get("operation_version", 1)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            e.click_attempts[key] = dict(row)

        for row in (d.get("click_candidates") or []):
            if not isinstance(row, dict) or not row.get("id"):
                continue
            e.click_candidates[str(row["id"])] = ClickCandidate(
                id=str(row["id"]),
                position_a=str(row.get("position_a", "")),
                position_b=str(row.get("position_b", "")),
                proposal=ClickProposal(
                    abstraction=str(row.get("abstraction", "")),
                    specializer_a=str(row.get("specializer_a", "")),
                    specializer_b=str(row.get("specializer_b", "")),
                    scope_boundary=str(row.get("scope_boundary", "")),
                ),
                created_at=float(row.get("created_at", 0.0)),
                status=str(row.get("status", "open")),
            )

        e.human_contributions = int(d.get("human_contributions", 0))
        e.clicks_confirmed = int(d.get("clicks_confirmed", 0))
        e.ledger = list(d.get("ledger", []))[-LEDGER_CAP:]
        e._seq = int(d.get("seq", len(e.order)))
        e._cand_seq = int(d.get("cand_seq", len(e.click_candidates)))
        return e


def save(engine: Engine, path: str) -> str:
    path = os.fspath(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(engine.state(), fh, indent=2, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load(path: str, now: Callable[[], float] = time.time) -> Engine:
    with open(os.fspath(path), "r", encoding="utf-8") as fh:
        return Engine.from_state(json.load(fh), now=now)
