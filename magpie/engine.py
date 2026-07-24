"""magpie.engine — the law.

Deterministic, pure in-memory state machine for the magpie field. No I/O
(except the module-level load/save helpers), no network, no threads, no
randomness beyond a monotonic id counter.

Invariant (THE LAW): `resolve()` is the ONLY code path that may set a card's
state to 'supported' or 'refuted', and it refuses to run without a receipt.
Every other mutator either leaves state alone or moves cards among
'open' / 'testing' / 'needs_human'.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

__all__ = ["Card", "Engine", "load", "save"]

KINDS = ("thesis", "claim", "synthesis")
STATES = ("open", "testing", "supported", "refuted", "needs_human")
LIVE_STATES = ("open", "testing", "needs_human", "supported", "refuted")
LEDGER_CAP = 50
LEDGER_TEXT_CAP = 1024
AFFINITY_THRESHOLD = 0.34


@dataclass
class Card:
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

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Card":
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
            born=float(d.get("born", 0.0)),
            parents=list(d.get("parents", []) or []),
            archived=bool(d.get("archived", False)),
        )


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class Engine:
    def __init__(self, cap: int = 12, now: Callable[[], float] = time.time):
        self.cap = int(cap)
        self.now = now
        self.question: str = ""
        self.cards: dict[str, Card] = {}
        self.order: list[str] = []
        self.sections: dict[str, dict] = {}
        self.section_order: list[str] = []
        self.ledger: list[dict] = []
        self._seq = 0
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

    def _card(self, card_id: str) -> Card:
        if card_id not in self.cards:
            raise KeyError(card_id)
        return self.cards[card_id]

    def _active_card(self, card_id: str) -> Card:
        """Return a card only while it remains on the live field.

        Archiving is a terminal lifecycle transition.  Keeping mutation
        guards here prevents a stale browser action or delayed worker result
        from silently changing material that the user can no longer see.
        """
        card = self._card(card_id)
        if card.archived:
            raise ValueError("card is archived")
        return card

    def live(self) -> list[Card]:
        return [self.cards[i] for i in self.order if not self.cards[i].archived]

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

    # ---------------- cards ----------------

    def propose(
        self,
        text: str,
        section: str | None = None,
        kind: str = "claim",
        state: str = "open",
        mass: float = 0.42,
        parents: list[str] | None = None,
        foot: str = "",
    ) -> Card:
        if kind not in KINDS:
            raise ValueError(f"bad kind {kind!r}")
        if state in ("supported", "refuted"):
            raise ValueError("only resolve() may set supported/refuted")
        if state not in STATES:
            raise ValueError(f"bad state {state!r}")
        if section is None:
            section = self._lightest_section()
        if section not in self.sections:
            self.add_section(section, section.upper())
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
            born=self.now(),
            parents=list(parents or []),
        )
        self.cards[card.id] = card
        self.order.append(card.id)
        self._log("ARRIVED", text)
        return card

    def collide(self, a_id: str, b_id: str) -> Card:
        a, b = self._active_card(a_id), self._active_card(b_id)
        child = self.propose(
            # Never recursively embed generated parent text in a persisted
            # placeholder while inference is pending.
            text=f"Collision pending: {a.id} × {b.id}",
            section=a.section,
            kind="claim",
            state="testing",
            mass=_clamp((a.mass + b.mass) / 2 + 0.08),
            parents=[a.id, b.id],
        )
        self.ledger.pop()  # replace the ARRIVED entry with FUSING
        self._log("FUSING", f"{a.id} × {b.id}")
        return child

    def update_proposal(
        self,
        card_id: str,
        text: str,
        *,
        kind: str = "claim",
        foot: str = "",
    ) -> Card:
        """Apply an inference result without pretending it is evidence.

        A collision is ``testing`` only while the one-shot inference request is
        in flight.  The model's answer is a new proposal, so it returns to
        ``open`` with provider provenance in ``foot`` and no receipt.  A later
        verifier or human may settle it through :meth:`resolve`.
        """
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
        verification jobs.  It cannot erase a settled verdict.
        """
        card = self._active_card(card_id)
        if card.state in ("supported", "refuted"):
            raise ValueError("cannot reopen a settled card")
        card.state = "open"
        if foot is not None:
            card.foot = str(foot)
        self._log("REOPENED", card.text)
        return card

    def resolve(
        self,
        card_id: str,
        verdict: str,
        receipt: str,
        text: str | None = None,
        foot: str | None = None,
    ) -> Card:
        """The only path to supported/refuted. Receipt is mandatory.

        `text`/`foot` let a caller supply the resolved wording *as part of the
        same transaction*. They are parameters rather than a follow-up
        assignment on purpose: if the text could be rewritten after resolve()
        returned, the receipt and the RESOLVED ledger entry would attest to
        wording the card no longer carries.
        """
        card = self._active_card(card_id)
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
        if verdict == "supported" and card.parents:
            card.kind = "synthesis"
            for pid in card.parents:
                p = self.cards.get(pid)
                if p is not None and not p.archived:
                    p.archived = True
            self._log("FUSED", card.text)
        self._log("RESOLVED", f"{verdict} · {card.text}")
        return card

    def judge(self, card_id: str, verdict: str) -> Card | list[Card]:
        card = self._active_card(card_id)
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
            )
            for suffix in ("what would make this true?", "what would make this false?")
        ]
        self._log("JUDGED", f"unknown · {card.text}")
        return kids

    def keep(self, card_id: str) -> Card:
        card = self._active_card(card_id)
        card.pinned = not card.pinned
        self._log("KEPT" if card.pinned else "UNKEPT", card.text)
        return card

    def kill(self, card_id: str) -> Card:
        card = self._active_card(card_id)
        card.archived = True
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
        self._log("FUSING", f"verify · {card.text}")
        return card

    # ---------------- ranking ----------------

    def frontier(self) -> list[Card]:
        cs = [c for c in self.live() if c.state in ("open", "needs_human")]
        return sorted(cs, key=lambda c: (0 if c.state == "needs_human" else 1, -c.mass))

    def affinity(self, a: Card, b: Card) -> float:
        if a.section == b.section:
            return 0.36 + (a.mass + b.mass) * 0.16
        return 0.1

    def best_pair(self) -> tuple[str, str] | None:
        """Return the next safe automatic collision within one field.

        Automatic metabolism only combines root observations. Generated
        proposals and ``needs_human`` prompts remain visible for deliberate
        human action instead of recursively feeding the background loop.
        """
        cs = [
            c
            for c in self.live()
            if not c.pinned and c.state == "open" and not c.parents
        ]
        best, best_score = None, AFFINITY_THRESHOLD
        existing_pairs = {
            frozenset(c.parents)
            for c in self.live()
            if len(c.parents) == 2
        }
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if frozenset((cs[i].id, cs[j].id)) in existing_pairs:
                    continue
                s = self.affinity(cs[i], cs[j])
                if s > best_score:
                    best, best_score = (cs[i].id, cs[j].id), s
        return best

    def enforce_cap(self) -> Card | None:
        """Retire expendable material without erasing a field's source ideas.

        ``cap`` is a per-field budget. Generated cards absorb retention
        pressure before human roots, ``needs_human`` cards are never retired,
        and at least one root observation survives in every populated field.
        """
        live = self.live()
        capacity = self.cap * max(1, len(self.sections))
        if len(live) <= capacity:
            return None
        eligible = [
            c for c in live
            if not c.pinned and c.state != "needs_human"
        ]
        derived = [c for c in eligible if c.parents]
        if derived:
            candidates = derived
        else:
            root_counts: dict[str, int] = {}
            for card in live:
                if not card.parents:
                    root_counts[card.section] = root_counts.get(card.section, 0) + 1
            candidates = [
                c for c in eligible
                if c.parents or root_counts.get(c.section, 0) > 1
            ]
        if not candidates:
            return None
        victim = min(candidates, key=lambda c: (c.mass, c.born))
        victim.archived = True
        self._log("RETIRED", victim.text)
        return victim

    # ---------------- reporting ----------------

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

    def harvest(self) -> dict:
        live = self.live()
        brief = {
            "question": self.question,
            "sections": [dict(self.sections[k]) for k in self.section_order],
            "cards": [c.to_dict() for c in live],
            "syntheses": [c.to_dict() for c in live if c.kind == "synthesis"],
            "ledger": list(self.ledger),
        }
        self._log("HARVESTED", f"{len(live)} cards")
        return brief

    def state(self) -> dict:
        return {
            "question": self.question,
            "cap": self.cap,
            "capacity": self.cap * max(1, len(self.sections)),
            "seq": self._seq,
            "sections": [dict(self.sections[k]) for k in self.section_order],
            "cards": [self.cards[i].to_dict() for i in self.order],
            "weights": self.weights(),
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
        e.cards = {}
        e.order = []
        for cd in d.get("cards", []):
            c = Card.from_dict(cd)
            # The law survives deserialization: a terminal state is only
            # meaningful if the receipt that justified it came back with it.
            # A snapshot claiming supported/refuted with no receipt was either
            # hand-edited or written by a code path that bypassed resolve();
            # either way it is not evidence, so it reverts to needs_human.
            if c.state in ("supported", "refuted") and not (c.receipt or "").strip():
                c.state = "needs_human"
                c.receipt = None
            e.cards[c.id] = c
            e.order.append(c.id)
        e.ledger = list(d.get("ledger", []))[-LEDGER_CAP:]
        e._seq = int(d.get("seq", len(e.order)))
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
