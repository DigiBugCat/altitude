"""magpie.workers — cognition.

Two jobs, both pure request/response over `providers.get_chain()`:

  atomize(text, sections)      raw thought -> 1-2 sharp claims, each filed
  fuse(a, b, question)         two cards -> what their collision produces

This module never imports `engine`. It takes plain dicts and returns plain
dicts, so the law (engine.py) and the cognition (here) stay independently
testable and independently replaceable. In particular nothing here decides a
card's *state*. Workers supply proposed text and provider provenance, never
evidence or a receipt.

Every artifact carries provenance naming the provider that produced it. That
is useful attribution, but it cannot support or refute a claim.
"""

from __future__ import annotations

from . import providers

MAX_CLAIMS = 2

_ATOMIZE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "section"],
                "properties": {
                    "text": {"type": "string"},
                    "section": {"type": "string"},
                },
            },
        }
    },
}

_FUSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "text"],
    "properties": {
        "kind": {"type": "string", "enum": ["SYNTHESIS", "TENSION", "DISCRIMINATOR"]},
        "text": {"type": "string"},
    },
}

_KINDS = ("SYNTHESIS", "TENSION", "DISCRIMINATOR")


def _clean(s, limit=400):
    return " ".join(str(s or "").split())[:limit]


def atomize(text: str, sections: list[str]) -> list[dict]:
    """Split a raw thought into 1-2 falsifiable claims, each filed to a section.

    Returns ``[{text, section, foot}, ...]``. On total provider failure the raw
    text comes back as a single claim footed 'unfiled', so a thought is never
    silently swallowed.
    """
    text = _clean(text)
    sections = [s for s in (sections or []) if s] or ["field"]
    if not text:
        return []

    prompt = (
        "Split this raw thought into at most two sharp, falsifiable claims.\n"
        "Each claim must stand alone, be one sentence, and state something that "
        "could turn out to be wrong. Do not hedge, do not add preamble, do not "
        "invent content that is not implied by the thought.\n"
        f"File each claim into exactly one of these sections: {', '.join(sections)}.\n\n"
        f"THOUGHT: {text}"
    )

    try:
        result, provider = providers.get_chain().complete(
            prompt, schema=_ATOMIZE_SCHEMA, timeout=45)
    except providers.ProviderUnavailable:
        return [{"text": text, "section": sections[0], "foot": "unfiled"}]

    foot = f"atomized by {provider}"
    out = []
    claims = result.get("claims") if isinstance(result, dict) else None
    for c in (claims or []):
        if not isinstance(c, dict):
            continue
        ctext = _clean(c.get("text"))
        if not ctext:
            continue
        section = _clean(c.get("section"), 60)
        if section not in sections:
            section = sections[0]
        out.append({"text": ctext, "section": section, "foot": foot})
        if len(out) >= MAX_CLAIMS:
            break

    if not out:
        out = [{"text": text, "section": sections[0], "foot": foot}]
    return out


def fuse(a: dict, b: dict, question: str) -> dict:
    """Collide two cards. Returns ``{ok, text, kind, provenance}``.

    ``ok`` means only that inference produced a usable proposal. It does not
    mean the result was verified, adjudicated, supported, or refuted.

    `kind` is what the collision actually produced:
      SYNTHESIS      — the two claims combine into something neither said alone
      TENSION        — they conflict; the conflict itself is the finding
      DISCRIMINATOR  — a test whose outcome would settle which one holds

    `a` and `b` are card dicts (engine's shape), used read-only.
    """
    a_text = _clean((a or {}).get("text"))
    b_text = _clean((b or {}).get("text"))
    question = _clean(question, 300)

    prompt = (
        "Two claims are colliding. Say what the collision actually produces.\n\n"
        f"OPEN QUESTION: {question or '(none stated)'}\n"
        f"CLAIM A: {a_text}\n"
        f"CLAIM B: {b_text}\n\n"
        "Choose one kind:\n"
        "  SYNTHESIS — they combine into a claim neither made alone\n"
        "  TENSION — they conflict, and naming the conflict is the finding\n"
        "  DISCRIMINATOR — an observable test whose outcome settles which holds\n"
        "Then write that result as one sharp sentence. No preamble, no hedging, "
        "and do not merely restate A or B."
    )

    try:
        result, provider = providers.get_chain().complete(
            prompt, schema=_FUSE_SCHEMA, timeout=45)
    except providers.ProviderUnavailable:
        # Fail closed. `ok: False` is the signal the caller must branch on: a
        # collision nobody could adjudicate is NOT a finding, and must never be
        # resolved into a settled state on the strength of this text.
        return {
            "ok": False,
            "text": f"unresolved collision: {a_text} / {b_text}",
            "kind": "TENSION",
            "provenance": "fusion unavailable · no provider answered",
        }

    kind = _clean((result or {}).get("kind"), 20).upper()
    if kind not in _KINDS:
        kind = "SYNTHESIS"
    ftext = _clean((result or {}).get("text"))
    if not ftext:
        # The chain answered but said nothing usable — same status as silence.
        return {
            "ok": False,
            "text": f"unresolved collision: {a_text} / {b_text}",
            "kind": "TENSION",
            "provenance": f"fusion empty · {provider} returned no text",
        }

    return {"ok": True, "text": ftext, "kind": kind,
            "provenance": f"proposed by {provider}"}
