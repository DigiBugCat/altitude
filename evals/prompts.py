"""Prompt variants used by the Magpie Braintrust experiments.

The prompt-only examples below are deliberately not eval rows.  Their IDs use
the reserved ``prompt-only-`` prefix; :func:`evals.tasks.load_cases` rejects
scored rows with that prefix so a future dataset edit cannot silently leak a
few-shot example into a holdout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from . import TASK_NAMES, VARIANT_NAMES

PROMPT_ONLY_ID_PREFIX = "prompt-only-"

_TASK_DUTIES = {
    "visible": (
        "Turn the thought into one or two user-visible ideas. Preserve a rich, "
        "mechanism-bearing idea as prose; do not force it into metric grammar. "
        "Each idea has exactly `text` and `section`. Select from supplied sections "
        "when present; otherwise infer one short, stable topical section."
    ),
    "memory": (
        "Turn the user-visible idea into atomic, testable claim proposals that "
        "remain hidden in memory. Each memory must flip as a unit. Claims use "
        "`condition → observable → bound`. You may propose a concrete resolution "
        "criterion when the source is qualitative, but do not present it as "
        "observed evidence or add unsupported facts, dates, or causal mechanisms."
    ),
    "relations": (
        "Link the supplied nodes only when the source text justifies an edge. "
        "Use the strongest justified type: derived_from, tests, supports, "
        "contradicts, depends_on, refines, duplicates, or related_to. Direction "
        "is semantic, not cosmetic. Prefer no edge over a plausible-sounding one."
    ),
    "collision": (
        "Collide parents A and B. Produce something neither parent entails alone: "
        "a synthesis, a named tension, or a discriminator. Do not restate a "
        "parent or merely blend their vocabulary."
    ),
}

_OUTPUT_SHAPES = {
    "visible": (
        '{"ideas":[{"text":"string","section":"one supplied section"}]}. '
        "Emit one or two ideas and no other keys."
    ),
    "memory": (
        '{"atoms":[{"text":"string","polarity":'
        '"affirms|denies","resolver":"sandbox|logs|market|search|human",'
        '"resolves_by":null-or-"YYYY-MM-DD"}]}. Each atom has exactly these four '
        "fields; the backend attaches identity and provenance."
    ),
    "relations": (
        '{"relations":[{"source_id":"known id","target_id":"known id","relation":'
        '"derived_from|tests|supports|contradicts|depends_on|refines|duplicates|'
        'related_to"}]}. Emit zero or one edge, no self-edges, and no unknown IDs.'
    ),
    "collision": (
        '{"kind":"SYNTHESIS|TENSION|DISCRIMINATOR","text":"one thesis-register '
        'sentence"}. Emit exactly these two fields. The backend already knows '
        "the parent IDs and can derive hidden child memories separately."
    ),
}

# These examples are intentionally mundane and outside the AI-tooling eval
# corpus. They test formatting instincts without teaching answers to scored
# examples.
_POSITIVE_EXAMPLES = {
    "visible": (
        'INPUT: {"thought":"Weekly batch retries are colliding with the lease '
        'timeout","sections":["Operations","Product"]}\n'
        'OUTPUT: {"ideas":[{"text":"Weekly batch retries and the lease timeout '
        'interact, causing duplicate work.","section":"Operations"}]}'
    ),
    "memory": (
        'INPUT: {"source_id":"note-a","text":"When we disabled retries, duplicate '
        'invoices stopped."}\n'
        'OUTPUT: {"atoms":[{"text":"disable retries → duplicate invoices → 0",'
        '"polarity":"affirms","resolver":"logs","resolves_by":null}]}'
    ),
    "relations": (
        'INPUT: {"nodes":[{"id":"a","text":"Invoices duplicate only after '
        'retries."},{"id":"b","text":"disable retries → duplicate invoices → 0"}]}\n'
        'OUTPUT: {"relations":[{"source_id":"b","target_id":"a","relation":"tests"}]}'
    ),
    "collision": (
        'INPUT: {"a":{"id":"a","text":"Retries overlap the lease."},"b":{"id":"b",'
        '"text":"Duplicates begin after lease expiry."},"question":"Which timing '
        'boundary matters?"}\nOUTPUT: {"kind":"DISCRIMINATOR","text":"The '
        'duplicate path depends on whether a retry begins before or after lease '
        'expiry."}'
    ),
}

_NEGATIVE_GUIDANCE = {
    "visible": (
        'Reject prose such as "Various factors may improve things": it has no '
        "referent and discards the source mechanism. Do not turn every useful "
        "idea into a narrow metric claim."
    ),
    "memory": (
        'Reject "retries may improve reliability" (hedged and unmeasured), '
        '"A and B therefore C" (compound), and precise-looking facts presented '
        "as already observed when the source only supports a proposed test."
    ),
    "relations": (
        "Do not use related_to merely because two nodes share nouns. Do not emit "
        "supports when the source only establishes topical proximity. Do not "
        "reverse tests: the observable claim tests the broader idea."
    ),
    "collision": (
        'Reject "a holistic retry-and-lease framework" (vocabulary blend) and '
        "any output entailed by either parent alone."
    ),
}


def _payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_prompt(task: str, input_value: Mapping[str, Any], variant: str) -> list[dict[str, str]]:
    """Return OpenAI-compatible messages for one task and prompt variant.

    V0 is the deliberately sparse baseline. V1 adds the contract, V2 adds a
    prompt-only positive example, V3 replaces that example with failure-mode
    guidance, and V4 combines both while requesting strict JSON mode.
    """
    if task not in TASK_NAMES:
        raise ValueError(f"unknown eval task {task!r}")
    variant = variant.upper()
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown prompt variant {variant!r}")

    if variant == "V0":
        instruction = {
            "visible": "Extract clear user-visible ideas and assign each a topical section.",
            "memory": "Extract testable atomic memories from this source.",
            "relations": "Find justified semantic relations among these nodes.",
            "collision": "Collide these two ideas and state the useful new result.",
        }[task]
        system = (
            "You are the Magpie claim engine. Return one JSON value only, with no "
            "markdown or commentary."
        )
        user = f"{instruction}\n\nINPUT:\n{_payload(input_value)}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    blocks = [_TASK_DUTIES[task], f"OUTPUT CONTRACT: {_OUTPUT_SHAPES[task]}"]
    if variant in ("V2", "V4"):
        blocks.append("PROMPT-ONLY FORMAT EXAMPLE:\n" + _POSITIVE_EXAMPLES[task])
    if variant in ("V3", "V4"):
        blocks.append("FAILURE CHECKS:\n" + _NEGATIVE_GUIDANCE[task])
    if variant == "V4":
        blocks.append(
            "Before answering, silently check source faithfulness, atomicity, "
            "allowed identifiers, and exact schema. Return strict JSON only."
        )

    system = (
        "You are the Magpie claim engine. Follow the task-specific epistemic "
        "contract. Do not expose hidden reasoning. Return one JSON value only, "
        "with no markdown or commentary."
    )
    user = "\n\n".join(blocks) + f"\n\nINPUT:\n{_payload(input_value)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def uses_structured_output(variant: str) -> bool:
    """Only V4 receives provider-enforced JSON mode; earlier variants are prompt-only."""
    return variant.upper() == "V4"
