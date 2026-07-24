"""Pure-stdlib contracts for Magpie's four evaluation tasks.

The production UI intentionally has a small contract: visible ideas are only
``text`` plus ``section``.  Richer epistemic data lives in separate, hidden
memory atoms and relation edges.

These validators do not import Braintrust (or Magpie's runtime).  They accept
already-decoded Python values or JSON strings and return a normalized value
plus structured issues.  This makes the exact same validation usable in unit
tests, local evals, and Braintrust code scorers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

VISIBLE_IDEA_REQUIRED = frozenset({"text", "section"})
MEMORY_ATOM_REQUIRED = frozenset({"text", "polarity", "resolver", "resolves_by"})
RELATION_REQUIRED = frozenset({"source_id", "target_id", "relation"})
COLLISION_REQUIRED = frozenset({"kind", "text"})

POLARITIES = frozenset({"affirms", "denies"})
RESOLVERS = frozenset({"sandbox", "logs", "market", "search", "human"})
RELATION_TYPES = frozenset(
    {
        "derived_from",
        "tests",
        "supports",
        "contradicts",
        "depends_on",
        "refines",
        "duplicates",
        "related_to",
    }
)
COLLISION_KINDS = frozenset({"SYNTHESIS", "TENSION", "DISCRIMINATOR"})

CLAIM_MAX_CHARS = 140
VISIBLE_IDEA_MAX_CHARS = 400
COLLISION_MAX_CHARS = 400
ARROW = " → "

_TIME_BOUND_RE = re.compile(
    r"\b(?:within|before|after|by|until|next|this|per)\s+"
    r"(?:\d+|q[1-4]\b|quarter(?:ly)?|month|week|day|hour|session|year|earnings|ye\b)"
    r"|\b(?:q[1-4]\s+\d{4}|20\d{2}-\d{2}-\d{2}|year[- ]end|ye\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable contract failure or diagnostic."""

    code: str
    path: str
    message: str


@dataclass
class ValidationResult:
    """Validation outcome shared by contracts and deterministic scorers."""

    normalized: Any = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def add(self, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, path=path, message=message))

    def issue_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


def _decode(value: Any, result: ValidationResult) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            result.add("invalid_json", "$", f"invalid JSON: {exc.msg}")
            return None
    return value


def _unwrap_items(
    value: Any,
    *,
    wrapper: str,
    result: ValidationResult,
) -> list[Any] | None:
    value = _decode(value, result)
    if value is None:
        return None
    if isinstance(value, Mapping):
        if wrapper in value:
            extra = set(value) - {wrapper}
            if extra:
                result.add(
                    "unexpected_wrapper_fields",
                    "$",
                    f"wrapper may only contain {wrapper!r}; found {sorted(extra)!r}",
                )
            value = value[wrapper]
        else:
            # A single item is accepted so task implementations can evaluate
            # one idea/atom/edge without adding an artificial list wrapper.
            value = [value]
    if not isinstance(value, list):
        result.add("wrong_type", "$", f"expected a list or {{{wrapper!r}: [...]}}")
        return None
    return value


def _validate_exact_object(
    item: Any,
    required: frozenset[str],
    result: ValidationResult,
    path: str,
) -> Mapping[str, Any] | None:
    if not isinstance(item, Mapping):
        result.add("wrong_type", path, "expected an object")
        return None
    keys = set(item)
    for key in sorted(required - keys):
        result.add("missing_field", f"{path}.{key}", f"missing required field {key!r}")
    extra = keys - required
    for key in sorted(extra):
        result.add(
            "unexpected_field",
            f"{path}.{key}",
            f"field {key!r} is not part of this contract",
        )
    return item


def _clean_text(
    value: Any,
    *,
    path: str,
    result: ValidationResult,
    max_chars: int,
) -> str | None:
    if not isinstance(value, str):
        result.add("wrong_type", path, "expected a string")
        return None
    cleaned = " ".join(value.split())
    if not cleaned:
        result.add("empty_text", path, "must not be blank")
        return None
    if len(cleaned) > max_chars:
        result.add(
            "too_long",
            path,
            f"must be at most {max_chars} characters; found {len(cleaned)}",
        )
    return cleaned


def _clean_identifier(value: Any, path: str, result: ValidationResult) -> str | None:
    if not isinstance(value, str):
        result.add("wrong_type", path, "expected a string")
        return None
    cleaned = value.strip()
    if not cleaned:
        result.add("empty_identifier", path, "must not be blank")
        return None
    if any(char.isspace() for char in cleaned):
        result.add("invalid_identifier", path, "identifiers may not contain whitespace")
    return cleaned


def validate_visible_ideas(
    output: Any,
    *,
    allowed_sections: Iterable[str] | None = None,
) -> ValidationResult:
    """Validate ``[{text, section}]`` (or ``{"ideas": [...]}``)."""

    result = ValidationResult(normalized=[])
    items = _unwrap_items(output, wrapper="ideas", result=result)
    if items is None:
        return result
    if not items:
        result.add("empty_collection", "$", "at least one visible idea is required")

    sections = set(allowed_sections) if allowed_sections is not None else None
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(items):
        path = f"$[{index}]"
        item = _validate_exact_object(raw, VISIBLE_IDEA_REQUIRED, result, path)
        if item is None:
            continue
        text = _clean_text(
            item.get("text"),
            path=f"{path}.text",
            result=result,
            max_chars=VISIBLE_IDEA_MAX_CHARS,
        )
        section = _clean_text(
            item.get("section"),
            path=f"{path}.section",
            result=result,
            max_chars=60,
        )
        if section is not None and sections is not None and section not in sections:
            result.add(
                "unknown_section",
                f"{path}.section",
                f"section must be one of {sorted(sections)!r}",
            )
        if text is not None and section is not None:
            normalized.append({"text": text, "section": section})
    result.normalized = normalized
    return result


def validate_claim_text(
    text: Any,
    *,
    path: str = "$.text",
    result: ValidationResult | None = None,
) -> ValidationResult:
    """Validate the canonical ``condition → observable → bound`` grammar."""

    result = result or ValidationResult()
    cleaned = _clean_text(
        text,
        path=path,
        result=result,
        max_chars=CLAIM_MAX_CHARS,
    )
    result.normalized = cleaned
    if cleaned is None:
        return result

    if "->" in cleaned or "=>" in cleaned or "⟶" in cleaned:
        result.add(
            "wrong_arrow",
            path,
            f"use the exact delimiter {ARROW!r}, not an ASCII or alternate arrow",
        )
    parts = cleaned.split(ARROW)
    if len(parts) != 3:
        result.add(
            "arrow_count",
            path,
            f"claim must have exactly two {ARROW.strip()!r} delimiters",
        )
    elif any(not part.strip() for part in parts):
        result.add(
            "empty_claim_part",
            path,
            "condition, observable, and bound must each be non-empty",
        )
    return result


def _validate_resolves_by(
    value: Any,
    *,
    path: str,
    claim_text: str | None,
    result: ValidationResult,
) -> str | None:
    if value is None:
        if claim_text and _TIME_BOUND_RE.search(claim_text):
            result.add(
                "missing_resolves_by",
                path,
                "time-bound claims must carry an ISO resolves_by date",
            )
        return None
    if not isinstance(value, str):
        result.add("wrong_type", path, "expected an ISO date string or null")
        return None
    cleaned = value.strip()
    try:
        date.fromisoformat(cleaned)
    except ValueError:
        result.add("invalid_date", path, "expected an ISO date in YYYY-MM-DD form")
    return cleaned


def validate_memory_atoms(output: Any) -> ValidationResult:
    """Validate hidden typed memory atoms.

    Accepted top-level forms are a list, one atom object, or ``{"atoms": [...]}``.
    Every atom has exactly ``text``, ``polarity``, ``resolver``, and
    ``resolves_by``.  The nullable date remains explicit so missing model fields
    cannot be confused with claims that intentionally have no deadline.
    """

    result = ValidationResult(normalized=[])
    items = _unwrap_items(output, wrapper="atoms", result=result)
    if items is None:
        return result
    if not items:
        result.add("empty_collection", "$", "at least one memory atom is required")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        path = f"$[{index}]"
        item = _validate_exact_object(raw, MEMORY_ATOM_REQUIRED, result, path)
        if item is None:
            continue

        claim_result = validate_claim_text(item.get("text"), path=f"{path}.text")
        result.issues.extend(claim_result.issues)
        text = claim_result.normalized

        polarity = item.get("polarity")
        if not isinstance(polarity, str):
            result.add("wrong_type", f"{path}.polarity", "expected a string")
            polarity = None
        elif polarity not in POLARITIES:
            result.add(
                "invalid_polarity",
                f"{path}.polarity",
                f"must be one of {sorted(POLARITIES)!r}",
            )

        resolver = item.get("resolver")
        if not isinstance(resolver, str):
            result.add("wrong_type", f"{path}.resolver", "expected a string")
            resolver = None
        elif resolver not in RESOLVERS:
            result.add(
                "invalid_resolver",
                f"{path}.resolver",
                f"must be one of {sorted(RESOLVERS)!r}",
            )

        resolves_by = _validate_resolves_by(
            item.get("resolves_by"),
            path=f"{path}.resolves_by",
            claim_text=text,
            result=result,
        )
        if text is not None and polarity is not None and resolver is not None:
            normalized.append(
                {
                    "text": text,
                    "polarity": polarity,
                    "resolver": resolver,
                    "resolves_by": resolves_by,
                }
            )
    result.normalized = normalized
    return result


def validate_relations(
    output: Any,
    *,
    known_ids: Iterable[str] | None = None,
) -> ValidationResult:
    """Validate typed, directed backend edges."""

    result = ValidationResult(normalized=[])
    items = _unwrap_items(output, wrapper="relations", result=result)
    if items is None:
        return result
    known = set(known_ids) if known_ids is not None else None
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(items):
        path = f"$[{index}]"
        item = _validate_exact_object(raw, RELATION_REQUIRED, result, path)
        if item is None:
            continue
        source_id = _clean_identifier(item.get("source_id"), f"{path}.source_id", result)
        target_id = _clean_identifier(item.get("target_id"), f"{path}.target_id", result)
        relation = item.get("relation")
        if not isinstance(relation, str):
            result.add("wrong_type", f"{path}.relation", "expected a string")
            relation = None
        elif relation not in RELATION_TYPES:
            result.add(
                "invalid_relation",
                f"{path}.relation",
                f"must be one of {sorted(RELATION_TYPES)!r}",
            )
        if source_id is not None and target_id is not None:
            if source_id == target_id:
                result.add("self_edge", path, "a relation may not point to itself")
            if known is not None:
                if source_id not in known:
                    result.add(
                        "unknown_endpoint",
                        f"{path}.source_id",
                        f"unknown endpoint {source_id!r}",
                    )
                if target_id not in known:
                    result.add(
                        "unknown_endpoint",
                        f"{path}.target_id",
                        f"unknown endpoint {target_id!r}",
                    )
        if source_id is not None and target_id is not None and relation is not None:
            edge = (source_id, target_id, relation)
            if edge in seen:
                result.add("duplicate_edge", path, "duplicate relation edge")
            seen.add(edge)
            normalized.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation": relation,
                }
            )
    result.normalized = normalized
    return result


def validate_collision(
    output: Any,
    *,
    known_ids: Iterable[str] | None = None,
) -> ValidationResult:
    """Validate the current collision proposal ``{kind, text}``.

    ``known_ids`` is accepted for an API symmetrical with relation validation;
    it is intentionally unused because parent IDs belong to the task input, not
    the model output.  Keeping them out prevents the model from fabricating
    provenance.
    """

    del known_ids
    result = ValidationResult(normalized=None)
    value = _decode(output, result)
    item = _validate_exact_object(value, COLLISION_REQUIRED, result, "$")
    if item is None:
        return result
    kind = item.get("kind")
    if not isinstance(kind, str):
        result.add("wrong_type", "$.kind", "expected a string")
        kind = None
    elif kind not in COLLISION_KINDS:
        result.add(
            "invalid_collision_kind",
            "$.kind",
            f"must be one of {sorted(COLLISION_KINDS)!r}",
        )
    text = _clean_text(
        item.get("text"),
        path="$.text",
        result=result,
        max_chars=COLLISION_MAX_CHARS,
    )
    if kind is not None and text is not None:
        result.normalized = {"kind": kind, "text": text}
    return result
