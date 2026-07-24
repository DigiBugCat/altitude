import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.contracts import (  # noqa: E402
    validate_claim_text,
    validate_collision,
    validate_memory_atoms,
    validate_relations,
    validate_visible_ideas,
)
from evals.scorers import (  # noqa: E402
    collision_scores,
    deterministic_scorers_for,
    memory_atom_scores,
    relation_scores,
    visible_idea_scores,
)


def by_name(scores):
    return {score["name"]: score for score in scores}


def valid_atom(**updates):
    atom = {
        "text": "replay at 2x traffic → 500 rate → >2x baseline",
        "polarity": "affirms",
        "resolver": "sandbox",
        "resolves_by": None,
    }
    atom.update(updates)
    return atom


# ---------------- visible ideas ----------------


def test_visible_ideas_are_exactly_text_and_section():
    result = validate_visible_ideas(
        [{"text": "Transport success hides routing failures.", "section": "Reliability"}]
    )
    assert result.valid
    assert result.normalized == [
        {"text": "Transport success hides routing failures.", "section": "Reliability"}
    ]


def test_visible_idea_rejects_backend_fields_and_unknown_section():
    result = validate_visible_ideas(
        [{"text": "x", "section": "Unknown", "polarity": "affirms"}],
        allowed_sections=["Reliability"],
    )
    assert {"unexpected_field", "unknown_section"} <= set(result.issue_codes())


def test_visible_idea_accepts_json_wrapper_and_normalizes_whitespace():
    result = validate_visible_ideas(
        json.dumps({"ideas": [{"text": "  one   useful idea ", "section": " AI "}]}),
        allowed_sections=["AI"],
    )
    assert result.valid
    assert result.normalized == [{"text": "one useful idea", "section": "AI"}]


def test_visible_idea_scores_report_expected_coverage_and_faithfulness():
    scores = by_name(
        visible_idea_scores(
            {"source": "Semantic routing fails despite successful requests.", "sections": ["AI"]},
            [{"text": "Successful requests hide semantic routing failures.", "section": "AI"}],
            [{"text": "Semantic routing failures hide behind success.", "section": "AI"}],
        )
    )
    assert scores["schema_conformance"]["score"] == 1.0
    assert scores["source_faithfulness"]["score"] == 1.0
    assert scores["expected_coverage"]["score"] > 0.5


# ---------------- memory atoms ----------------


def test_claim_text_enforces_exact_arrow_and_length():
    assert validate_claim_text("do thing → measured rate → ≤1%").valid
    wrong = validate_claim_text("do thing -> measured rate -> <=1%")
    assert {"wrong_arrow", "arrow_count"} <= set(wrong.issue_codes())
    long = validate_claim_text(f"{'x' * 140} → rate → 1")
    assert "too_long" in long.issue_codes()


@pytest.mark.parametrize("field", ["text", "polarity", "resolver", "resolves_by"])
def test_memory_atom_requires_all_typed_fields(field):
    atom = valid_atom()
    del atom[field]
    result = validate_memory_atoms([atom])
    assert "missing_field" in result.issue_codes()


def test_memory_atom_validates_enum_fields_and_rejects_unknown_metadata():
    result = validate_memory_atoms(
        [
            valid_atom(
                polarity="maybe",
                resolver="browser",
                related_to=["c1"],
            )
        ]
    )
    assert {"invalid_polarity", "invalid_resolver", "unexpected_field"} <= set(
        result.issue_codes()
    )


def test_time_bound_atom_requires_iso_resolves_by():
    missing = validate_memory_atoms(
        [valid_atom(text="next quarter → design wins → ≥2")]
    )
    assert "missing_resolves_by" in missing.issue_codes()

    invalid = validate_memory_atoms(
        [valid_atom(text="next quarter → design wins → ≥2", resolves_by="earnings")]
    )
    assert "invalid_date" in invalid.issue_codes()

    valid = validate_memory_atoms(
        [valid_atom(text="next quarter → design wins → ≥2", resolves_by="2026-10-30")]
    )
    assert valid.valid


def test_next_quarterly_print_counts_as_time_bound():
    result = validate_memory_atoms(
        [valid_atom(text="next quarterly print → NAND ASP q/q → ≥5%")]
    )
    assert "missing_resolves_by" in result.issue_codes()


def test_memory_atom_scores_catch_slop_invented_numbers_arrows_and_length():
    output = [
        valid_atom(
            text=(
                "this could potentially leverage tooling to improve reliability "
                "while creating a very long and deliberately padded claim that cannot fit "
                "inside the canonical claim surface -> error rate -> 99%"
            )
        )
    ]
    scores = by_name(memory_atom_scores({"source": "tooling affects errors"}, output))
    assert scores["schema_conformance"]["score"] == 0.0
    assert scores["claim_arrow_grammar"]["score"] == 0.0
    assert scores["claim_length"]["score"] == 0.0
    assert scores["slop_lexicon"]["score"] == 0.0
    # Numeric novelty is diagnostic: a new number may be a proposed resolution
    # criterion rather than a fabricated observation. The semantic judge
    # decides whether the source supports choosing it.
    assert scores["source_faithfulness"]["score"] > 0.0
    assert scores["source_faithfulness"]["metadata"]["invented_numbers"] == ["99%"]


def test_memory_atom_scores_include_deterministic_compound_tripwire():
    scores = by_name(
        memory_atom_scores(
            {"source": "disable cache and retry requests"},
            [valid_atom(text="disable cache and retry → 500 rate → baseline")],
        )
    )
    assert scores["compound_claim_tripwire"]["score"] == 0.0


def test_memory_atom_coverage_is_null_without_gold_and_nonzero_with_gold():
    output = [valid_atom()]
    no_gold = by_name(memory_atom_scores({"source": output[0]["text"]}, output))
    assert no_gold["expected_coverage"]["score"] is None

    gold = {"claims": [{"text": "replay at 2x traffic → 500 rate → >2x baseline"}]}
    with_gold = by_name(memory_atom_scores({"source": output[0]["text"]}, output, gold))
    assert with_gold["expected_coverage"]["score"] == 1.0


def test_memory_source_card_dataset_shape_enables_faithfulness_diagnostic():
    scores = by_name(
        memory_atom_scores(
            {"source_card": {"text": "retry same key → duplicate writes → 0"}},
            [valid_atom(text="retry same key → duplicate writes → 0")],
        )
    )
    assert scores["source_faithfulness"]["score"] == 1.0


# ---------------- relations ----------------


def test_relations_validate_type_direction_endpoints_and_duplicates():
    result = validate_relations(
        [
            {"source_id": "a", "target_id": "b", "relation": "tests"},
            {"source_id": "a", "target_id": "b", "relation": "tests"},
            {"source_id": "a", "target_id": "missing", "relation": "approximately"},
            {"source_id": "b", "target_id": "b", "relation": "supports"},
        ],
        known_ids=["a", "b"],
    )
    assert {
        "duplicate_edge",
        "unknown_endpoint",
        "invalid_relation",
        "self_edge",
    } <= set(result.issue_codes())


def test_relation_scores_penalize_related_to_fallback_and_measure_exact_recall():
    output = [
        {"source_id": "a", "target_id": "b", "relation": "tests"},
        {"source_id": "b", "target_id": "c", "relation": "related_to"},
    ]
    scores = by_name(
        relation_scores(
            {"known_ids": ["a", "b", "c"]},
            output,
            [{"source_id": "a", "target_id": "b", "relation": "tests"}],
        )
    )
    assert scores["schema_conformance"]["score"] == 1.0
    assert scores["endpoint_validity"]["score"] == 1.0
    assert scores["relation_specificity"]["score"] == 0.5
    assert scores["expected_relation_recall"]["score"] == 1.0


def test_relation_scorer_understands_dataset_endpoint_and_gold_aliases():
    output = [{"source_id": "a", "target_id": "b", "relation": "tests"}]
    scores = by_name(
        relation_scores(
            {"source": {"id": "a"}, "target": {"id": "b"}},
            output,
            {"edge": {"from": "a", "to": "b", "relation": "tests"}},
        )
    )
    assert scores["endpoint_validity"]["score"] == 1.0
    assert scores["expected_relation_recall"]["score"] == 1.0


def test_relation_no_edge_is_valid_and_scores_against_null_gold():
    scores = by_name(
        relation_scores(
            {"source": {"id": "a"}, "target": {"id": "b"}},
            {"relations": []},
            {"edge": None},
        )
    )
    assert scores["schema_conformance"]["score"] == 1.0
    assert scores["endpoint_validity"]["score"] == 1.0
    assert scores["relation_specificity"]["score"] == 1.0
    assert scores["expected_relation_recall"]["score"] == 1.0


# ---------------- collisions ----------------


def test_collision_contract_is_current_kind_and_text_shape():
    result = validate_collision(
        {"kind": "DISCRIMINATOR", "text": "Replay with cache bypass separates the hypotheses."}
    )
    assert result.valid
    assert result.normalized["kind"] == "DISCRIMINATOR"

    extra = validate_collision(
        {"kind": "TENSION", "text": "x", "parents": ["a", "b"]}
    )
    assert "unexpected_field" in extra.issue_codes()


def test_collision_scores_catch_verbatim_parent_restatement():
    parent_a = "Tool calls return success while mutating the wrong target."
    parent_b = "Retries can duplicate writes after timeouts."
    scores = by_name(
        collision_scores(
            {"parents": [{"text": parent_a}, {"text": parent_b}]},
            {"kind": "SYNTHESIS", "text": parent_a},
        )
    )
    assert scores["schema_conformance"]["score"] == 1.0
    assert scores["collision_lexical_novelty"]["score"] == 0.0


def test_collision_scores_require_lexical_grounding_in_both_parents():
    parents = [
        "Structured output validates JSON syntax.",
        "Semantic fields can still name the wrong target.",
    ]
    output = {
        "kind": "TENSION",
        "text": "Valid JSON syntax can still contain semantic fields naming the wrong target.",
    }
    scores = by_name(collision_scores({"parents": parents}, output))
    assert scores["collision_parent_grounding"]["score"] == 1.0
    assert scores["collision_lexical_novelty"]["score"] > 0.0


def test_collision_scorer_reads_nested_thesis_reference():
    scores = by_name(
        collision_scores(
            {"parent_a": {"text": "A routes tools."}, "parent_b": {"text": "B verifies results."}},
            {"kind": "SYNTHESIS", "text": "Tool routing and result verification are separate."},
            {"thesis": {"text": "Tool routing and result verification are separate."}},
        )
    )
    assert scores["expected_coverage"]["score"] == 1.0


def test_scorers_return_braintrust_scorelike_mappings_without_dependency():
    scores = visible_idea_scores(
        {"source": "RAG retrieval misses relevant documents."},
        [{"text": "RAG retrieval misses relevant documents.", "section": "AI"}],
    )
    assert scores
    assert all(set(score) >= {"name", "score"} for score in scores)
    assert all(score["score"] is None or 0.0 <= score["score"] <= 1.0 for score in scores)


def test_runner_can_select_scorers_by_stable_task_name_and_call_with_keywords():
    scorer = deterministic_scorers_for("memory_atoms")[0]
    scores = scorer(
        input={"source": "replay at 2x traffic → 500 rate → >2x baseline"},
        output=[valid_atom()],
        expected=None,
    )
    assert by_name(scores)["schema_conformance"]["score"] == 1.0
    with pytest.raises(ValueError, match="unknown eval task"):
        deterministic_scorers_for("personas")
