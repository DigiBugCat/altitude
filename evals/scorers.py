"""Deterministic Braintrust-compatible scorers for Magpie eval tasks.

Each public scorer has Braintrust's ``(input, output, expected)`` signature and
returns a list of score mappings.  Braintrust accepts this ``ScoreLike`` form
directly; local tests can use it without importing Braintrust:

    {"name": "schema_conformance", "score": 0.0..1.0, "metadata": {...}}

The lexical source-faithfulness, coverage, and collision-novelty scores are
explicitly diagnostics, not semantic judges.  They catch cheap regressions
(invented numbers, dropped gold atoms, verbatim parent restatements) while the
runner may layer LLM judges on top for genuinely semantic questions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ARROW,
    CLAIM_MAX_CHARS,
    ValidationResult,
    validate_collision,
    validate_memory_atoms,
    validate_relations,
    validate_visible_ideas,
)

ScoreLike = dict[str, Any]

_SLOP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hedge_may", re.compile(r"\bmay\b", re.IGNORECASE)),
    ("hedge_might", re.compile(r"\bmight\b", re.IGNORECASE)),
    ("hedge_could", re.compile(r"\bcould(?:\s+potentially)?\b", re.IGNORECASE)),
    ("hedge_tends", re.compile(r"\btends?\s+to\b", re.IGNORECASE)),
    ("unbounded_some", re.compile(r"\bin\s+some\s+cases\b", re.IGNORECASE)),
    ("unbounded_users", re.compile(r"\bfor\s+certain\s+users\b", re.IGNORECASE)),
    ("consultant_leverage", re.compile(r"\bleverage[sd]?\b", re.IGNORECASE)),
    ("consultant_unlock", re.compile(r"\bunlock(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("consultant_streamline", re.compile(r"\bstreamlin(?:e|es|ed|ing)\b", re.IGNORECASE)),
    ("vibes_improve", re.compile(r"\bimprov(?:e|es|ed|ing)\s+(?:reliability|dx)\b", re.IGNORECASE)),
    ("vibes_enhance", re.compile(r"\benhanc(?:e|es|ed|ing)\s+(?:reliability|dx)\b", re.IGNORECASE)),
    ("blend_synergy", re.compile(r"\bsynerg(?:y|ies|istic)\b", re.IGNORECASE)),
    ("blend_holistic", re.compile(r"\bholistic\b", re.IGNORECASE)),
    ("mediation", re.compile(r"\bmediated\s+by\b", re.IGNORECASE)),
)

_WORD_RE = re.compile(r"[a-z][a-z0-9_-]*", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:[$€£]?\d+(?:\.\d+)?%?|\bq[1-4]\b|\bye\b)(?![A-Za-z])",
    re.IGNORECASE,
)
_COMPOUND_RE = re.compile(
    r"\b(?:and|but|while|therefore|additionally|as well as)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "given",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "this",
        "to",
        "was",
        "with",
    }
)


def _score(name: str, value: float | int | bool | None, **metadata: Any) -> ScoreLike:
    numeric = None if value is None else max(0.0, min(1.0, float(value)))
    out: ScoreLike = {"name": name, "score": numeric}
    if metadata:
        out["metadata"] = metadata
    return out


def _input_mapping(input_value: Any) -> Mapping[str, Any]:
    return input_value if isinstance(input_value, Mapping) else {}


def _source_text(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, Mapping):
        for key in ("source", "text", "thought", "mush", "input"):
            value = input_value.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping) and isinstance(value.get("text"), str):
                return value["text"]
        # Dataset rows use source_card for hidden-memory extraction.  This is
        # deliberately an input alias, not an output-contract alias.
        source_card = input_value.get("source_card")
        if isinstance(source_card, Mapping) and isinstance(source_card.get("text"), str):
            return source_card["text"]
    return ""


def _known_sections(input_value: Any) -> list[str] | None:
    value = _input_mapping(input_value).get("sections")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, str)]
    return None


def _known_ids(input_value: Any) -> list[str] | None:
    mapping = _input_mapping(input_value)
    raw = mapping.get("known_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [item for item in raw if isinstance(item, str)]
    nodes = mapping.get("nodes")
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
        ids = [
            item.get("id")
            for item in nodes
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ]
        return ids or None
    endpoint_ids = []
    for key in ("source", "target"):
        endpoint = mapping.get(key)
        if isinstance(endpoint, Mapping) and isinstance(endpoint.get("id"), str):
            endpoint_ids.append(endpoint["id"])
    if endpoint_ids:
        return endpoint_ids
    return None


def _texts_from_normalized(normalized: Any) -> list[str]:
    if isinstance(normalized, Mapping):
        text = normalized.get("text")
        return [text] if isinstance(text, str) else []
    if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes)):
        return [
            item["text"]
            for item in normalized
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        ]
    return []


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text or "")
        if token.lower() not in _STOPWORDS and len(token) > 1
    }


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_recall(reference: str, candidate: str) -> float:
    wanted, seen = _tokens(reference), _tokens(candidate)
    if not wanted:
        return 1.0
    return len(wanted & seen) / len(wanted)


def slop_hits(texts: Iterable[str]) -> list[dict[str, str]]:
    """Return deterministic slop-lexicon hits with their triggering text."""

    hits: list[dict[str, str]] = []
    for text in texts:
        for name, pattern in _SLOP_PATTERNS:
            if pattern.search(text):
                hits.append({"pattern": name, "text": text})
    return hits


def _schema_score(result: ValidationResult) -> ScoreLike:
    return _score(
        "schema_conformance",
        result.valid,
        issues=[
            {"code": issue.code, "path": issue.path, "message": issue.message}
            for issue in result.issues
        ],
    )


def _nonempty_score(texts: list[str], name: str = "nonempty_output") -> ScoreLike:
    return _score(name, bool(texts), count=len(texts))


def _dedup_score(texts: list[str]) -> ScoreLike:
    normalized = [" ".join(text.lower().split()) for text in texts]
    duplicate_count = sum(count - 1 for count in Counter(normalized).values() if count > 1)
    score = 1.0 if not normalized else 1.0 - duplicate_count / len(normalized)
    return _score("exact_dedup", score, duplicate_count=duplicate_count)


def _slop_score(texts: list[str]) -> ScoreLike:
    hits = slop_hits(texts)
    return _score("slop_lexicon", not hits, hits=hits)


def _expected_texts(expected: Any, wrappers: tuple[str, ...]) -> list[str]:
    if expected is None:
        return []
    value = expected
    if isinstance(value, Mapping):
        for wrapper in wrappers:
            if wrapper in value:
                value = value[wrapper]
                break
        else:
            value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [
        item["text"]
        for item in value
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    ]


def _coverage_score(texts: list[str], expected_texts: list[str]) -> ScoreLike:
    if not expected_texts:
        return _score(
            "expected_coverage",
            None,
            diagnostic="unavailable: dataset row has no structured expected texts",
        )
    recalls = [
        max((_overlap_recall(gold, candidate) for candidate in texts), default=0.0)
        for gold in expected_texts
    ]
    return _score(
        "expected_coverage",
        sum(recalls) / len(recalls),
        per_expected_recall=recalls,
        diagnostic="lexical recall only; semantic coverage requires a judge",
    )


def _faithfulness_score(source: str, texts: list[str]) -> ScoreLike:
    if not source:
        return _score(
            "source_faithfulness",
            None,
            diagnostic="unavailable: task input has no source text",
        )
    source_numbers = {match.lower() for match in _NUMBER_RE.findall(source)}
    output_numbers = {
        match.lower() for text in texts for match in _NUMBER_RE.findall(text)
    }
    invented_numbers = sorted(output_numbers - source_numbers)
    overlap = [
        _overlap_recall(text, source)
        for text in texts
    ]
    grounded_fraction = (
        sum(1 for value in overlap if value > 0.0) / len(overlap) if overlap else 0.0
    )
    return _score(
        "source_faithfulness",
        grounded_fraction,
        invented_numbers=invented_numbers,
        lexical_overlap=overlap,
        diagnostic=(
            "lexical grounding only; numeric novelty lists proposed resolution "
            "criteria absent from the source, not necessarily invented observed facts; "
            "semantic faithfulness requires a judge"
        ),
    )


def visible_idea_scores(input: Any, output: Any, expected: Any = None) -> list[ScoreLike]:
    """Score user-visible ``{text, section}`` ideas."""

    result = validate_visible_ideas(output, allowed_sections=_known_sections(input))
    texts = _texts_from_normalized(result.normalized)
    return [
        _schema_score(result),
        _nonempty_score(texts),
        _dedup_score(texts),
        _slop_score(texts),
        _faithfulness_score(_source_text(input), texts),
        _coverage_score(texts, _expected_texts(expected, ("ideas", "cards"))),
    ]


def _claim_shape_scores(result: ValidationResult, texts: list[str]) -> list[ScoreLike]:
    arrow_failures = [
        issue for issue in result.issues if issue.code in {"arrow_count", "wrong_arrow", "empty_claim_part"}
    ]
    length_failures = [issue for issue in result.issues if issue.code == "too_long"]
    compound_hits = [
        {"text": text, "terms": sorted({hit.lower() for hit in _COMPOUND_RE.findall(text)})}
        for text in texts
        if _COMPOUND_RE.search(text)
    ]
    return [
        _score(
            "claim_arrow_grammar",
            bool(texts) and not arrow_failures,
            expected_delimiter=ARROW,
            failures=[{"code": issue.code, "path": issue.path} for issue in arrow_failures],
        ),
        _score(
            "claim_length",
            bool(texts) and not length_failures,
            max_chars=CLAIM_MAX_CHARS,
            lengths=[len(text) for text in texts],
        ),
        _score(
            "compound_claim_tripwire",
            bool(texts) and not compound_hits,
            hits=compound_hits,
            diagnostic=(
                "conjunction tripwire only; whether a proposition can be half-true "
                "ultimately requires a semantic judge"
            ),
        ),
    ]


def memory_atom_scores(input: Any, output: Any, expected: Any = None) -> list[ScoreLike]:
    """Score hidden typed claims and cheap faithfulness/coverage regressions."""

    result = validate_memory_atoms(output)
    texts = _texts_from_normalized(result.normalized)
    scores = [
        _schema_score(result),
        _nonempty_score(texts),
        *_claim_shape_scores(result, texts),
        _slop_score(texts),
        _dedup_score(texts),
        _faithfulness_score(_source_text(input), texts),
        _coverage_score(texts, _expected_texts(expected, ("atoms", "claims"))),
    ]
    return scores


def relation_scores(input: Any, output: Any, expected: Any = None) -> list[ScoreLike]:
    """Score backend relation edges, endpoint validity, and fallback overuse."""

    result = validate_relations(output, known_ids=_known_ids(input))
    edges = result.normalized if isinstance(result.normalized, list) else []
    endpoint_failures = [
        issue
        for issue in result.issues
        if issue.code in {"unknown_endpoint", "self_edge", "empty_identifier", "invalid_identifier"}
    ]
    related_count = sum(
        1 for edge in edges if isinstance(edge, Mapping) and edge.get("relation") == "related_to"
    )
    specificity = 1.0 if not edges else 1.0 - related_count / len(edges)
    expected_edges: list[tuple[str, str, str]] = []
    expected_value = expected
    expected_contract_present = False
    if isinstance(expected_value, Mapping):
        if "relations" in expected_value:
            expected_contract_present = True
            expected_value = expected_value["relations"]
        elif "edge" in expected_value:
            expected_contract_present = True
            expected_value = expected_value["edge"]
    if isinstance(expected_value, Mapping):
        expected_value = [expected_value]
    if isinstance(expected_value, Sequence) and not isinstance(expected_value, (str, bytes)):
        for edge in expected_value:
            if isinstance(edge, Mapping):
                triple = (
                    edge.get("source_id", edge.get("from")),
                    edge.get("target_id", edge.get("to")),
                    edge.get("relation", edge.get("type")),
                )
                if all(isinstance(part, str) for part in triple):
                    expected_edges.append(triple)  # type: ignore[arg-type]
    actual_edges = {
        (edge["source_id"], edge["target_id"], edge["relation"])
        for edge in edges
        if isinstance(edge, Mapping)
    }
    exact_recall = (
        len(actual_edges & set(expected_edges)) / len(expected_edges)
        if expected_edges
        else (1.0 if expected_contract_present and not actual_edges else 0.0)
        if expected_contract_present
        else None
    )
    return [
        _schema_score(result),
        _score(
            "endpoint_validity",
            not endpoint_failures,
            failures=[{"code": issue.code, "path": issue.path} for issue in endpoint_failures],
        ),
        _score(
            "relation_specificity",
            specificity,
            related_to_count=related_count,
            diagnostic="penalizes related_to fallback; semantic type choice requires a judge",
        ),
        _score(
            "expected_relation_recall",
            exact_recall,
            diagnostic=(
                "exact directed-edge recall"
                if expected_edges
                else "correct no-edge behavior"
                if expected_contract_present
                else "unavailable: dataset row has no structured expected relations"
            ),
        ),
    ]


def _parent_texts(input_value: Any) -> list[str]:
    mapping = _input_mapping(input_value)
    parents = mapping.get("parents")
    if isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)):
        texts = []
        for parent in parents:
            if isinstance(parent, str):
                texts.append(parent)
            elif isinstance(parent, Mapping) and isinstance(parent.get("text"), str):
                texts.append(parent["text"])
        if texts:
            return texts
    texts = []
    for key in ("a", "b", "parent_a", "parent_b", "claim_a", "claim_b"):
        value = mapping.get(key)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, Mapping) and isinstance(value.get("text"), str):
            texts.append(value["text"])
    return texts


def collision_scores(input: Any, output: Any, expected: Any = None) -> list[ScoreLike]:
    """Score collision shape plus deterministic grounding/novelty tripwires."""

    result = validate_collision(output)
    texts = _texts_from_normalized(result.normalized)
    text = texts[0] if texts else ""
    parents = _parent_texts(input)
    similarities = [_jaccard(text, parent) for parent in parents] if text else []
    exact_or_contained = [
        parent
        for parent in parents
        if text and (
            " ".join(text.lower().split()) == " ".join(parent.lower().split())
            or " ".join(text.lower().split()) in " ".join(parent.lower().split())
        )
    ]
    novelty = (
        0.0
        if exact_or_contained
        else 1.0 - max(similarities, default=0.0)
    )
    grounding = [
        _overlap_recall(parent, text)
        for parent in parents
    ]
    both_grounded = bool(parents) and all(value > 0.0 for value in grounding)
    expected_texts = _expected_texts(expected, ("collisions", "outputs"))
    if isinstance(expected, Mapping) and isinstance(expected.get("text"), str):
        expected_texts = [expected["text"]]
    elif (
        isinstance(expected, Mapping)
        and isinstance(expected.get("thesis"), Mapping)
        and isinstance(expected["thesis"].get("text"), str)
    ):
        expected_texts = [expected["thesis"]["text"]]
    source = " ".join(parents) or _source_text(input)
    return [
        _schema_score(result),
        _nonempty_score(texts),
        _slop_score(texts),
        _faithfulness_score(source, texts),
        _score(
            "collision_parent_grounding",
            both_grounded,
            parent_overlap=grounding,
            diagnostic="lexical grounding only; invented bridges require a judge",
        ),
        _score(
            "collision_lexical_novelty",
            novelty if parents and text else None,
            parent_similarity=similarities,
            verbatim_or_contained_parent=bool(exact_or_contained),
            diagnostic="tripwire for restatement; propositional novelty requires a judge",
        ),
        _coverage_score(texts, expected_texts),
    ]


_TASK_SCORERS = {
    "visible_ideas": visible_idea_scores,
    "visible-ideas": visible_idea_scores,
    "visible": visible_idea_scores,
    "memory_atoms": memory_atom_scores,
    "memory-atoms": memory_atom_scores,
    "memory": memory_atom_scores,
    "atoms": memory_atom_scores,
    "relations": relation_scores,
    "relation_edges": relation_scores,
    "relation-edges": relation_scores,
    "collisions": collision_scores,
    "collision": collision_scores,
}


def deterministic_scorers_for(task: str) -> list[Any]:
    """Return scorer callables suitable for ``braintrust.Eval(scores=...)``.

    Each task currently uses one callable which expands into several named
    ``ScoreLike`` results.  Keeping the return value as a list lets the runner
    add independent LLM judges without knowing this implementation detail.
    """

    try:
        scorer = _TASK_SCORERS[task.strip().lower()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(("visible_ideas", "memory_atoms", "relations", "collisions"))
        raise ValueError(f"unknown eval task {task!r}; expected one of: {supported}") from exc
    return [scorer]
