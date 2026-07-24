"""Focused tests for typed thematic extraction and canonical matching."""

from __future__ import annotations

import pytest

from magpie import providers, workers
from magpie.providers import Chain, ProviderUnavailable


class AnsweringProvider:
    name = "test-cognition"

    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""
        self.schema = None

    def available(self):
        return True

    def complete(self, prompt, schema=None, timeout=30):
        self.prompt = prompt
        self.schema = schema
        return self.payload


class UnavailableProvider:
    name = "offline"

    def available(self):
        return True

    def complete(self, prompt, schema=None, timeout=30):
        raise ProviderUnavailable("offline")


@pytest.fixture(autouse=True)
def restore_provider_chain():
    yield
    providers.reset_chain(None)


def _answer(**updates):
    artifact = {
        "text": "Should we ship on Friday?",
        "section": "Launch",
        "artifact_type": "question",
        "relation": "new",
        "canonical_id": "",
    }
    artifact.update(updates)
    return {"artifacts": [artifact]}


def test_atomize_preserves_question_type_and_accepts_a_new_theme():
    provider = AnsweringProvider(_answer())
    providers.reset_chain(Chain([provider]))

    result = workers.atomize("Should we ship on Friday?", ["Product"])

    assert result == [{
        "text": "Should we ship on Friday?",
        "section": "Launch",
        "artifact_type": "question",
        "relation": "new",
        "foot": "atomized by test-cognition",
    }]
    assert provider.schema["properties"]["artifacts"]["items"]["properties"][
        "artifact_type"
    ]["enum"] == list(workers.ARTIFACT_TYPES)


def test_atomize_passes_bounded_typed_canonical_summaries_to_the_prompt():
    provider = AnsweringProvider(_answer(
        text="Friday remains the launch target.",
        artifact_type="decision",
        relation="refinement",
        canonical_id="c7",
    ))
    providers.reset_chain(Chain([provider]))
    cards = [
        {
            "id": "c7",
            "text": "We chose Friday for launch.",
            "artifact_type": "decision",
            "section": "Launch",
        },
        *[
            {"id": f"c{i}", "text": f"Candidate {i}", "kind": "claim"}
            for i in range(20)
        ],
    ]

    result = workers.atomize(
        "Friday remains the launch target.", ["Launch"], existing_cards=cards
    )

    assert result[0]["relation"] == "refinement"
    assert result[0]["canonical_id"] == "c7"
    assert '"canonical_candidates"' in provider.prompt
    assert '"artifact_type": "decision"' in provider.prompt
    # The model sees only a retrieval-selected, bounded candidate set.
    assert '"id": "c12"' not in provider.prompt


def test_unknown_canonical_id_fails_closed_to_new():
    provider = AnsweringProvider(_answer(
        relation="repeat",
        canonical_id="invented-id",
    ))
    providers.reset_chain(Chain([provider]))

    result = workers.atomize(
        "Should we ship on Friday?",
        ["Launch"],
        existing_cards=[{
            "id": "known",
            "text": "Should we launch Friday?",
            "artifact_type": "question",
        }],
    )

    assert result[0]["relation"] == "new"
    assert "canonical_id" not in result[0]


def test_new_relation_never_leaks_a_canonical_id():
    provider = AnsweringProvider(_answer(canonical_id="known"))
    providers.reset_chain(Chain([provider]))

    result = workers.atomize(
        "Should we ship on Friday?",
        ["Launch"],
        existing_cards=[{
            "id": "known",
            "text": "Should we launch Friday?",
            "artifact_type": "question",
        }],
    )

    assert result[0]["relation"] == "new"
    assert "canonical_id" not in result[0]


def test_repeat_requires_the_same_speech_act_type():
    provider = AnsweringProvider(_answer(
        relation="repeat",
        canonical_id="decision-1",
    ))
    providers.reset_chain(Chain([provider]))

    result = workers.atomize(
        "Should we ship on Friday?",
        ["Launch"],
        existing_cards=[{
            "id": "decision-1",
            "text": "We will ship on Friday.",
            "artifact_type": "decision",
        }],
    )

    assert result[0]["artifact_type"] == "question"
    assert result[0]["relation"] == "new"
    assert "canonical_id" not in result[0]


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("Why is the queue growing?", "question"),
        ("TODO fix the retry loop", "task"),
        ("We need to fix the retry loop", "task"),
        ("I prefer the smaller launcher", "preference"),
        ("We decided to keep the local cache", "decision"),
        ("Test cold-start latency", "experiment"),
        ("I observed duplicate writes", "observation"),
        ("Must not send credentials", "constraint"),
        ("Retries duplicate writes", "claim"),
    ],
)
def test_provider_outage_preserves_obvious_speech_act(text, expected_type):
    providers.reset_chain(Chain([UnavailableProvider()]))

    result = workers.atomize(text, ["Field"])

    assert result == [{
        "text": text,
        "section": "Field",
        "artifact_type": expected_type,
        "relation": "new",
        "foot": "unfiled",
    }]


def test_invalid_or_empty_provider_artifacts_fall_back_without_losing_input():
    provider = AnsweringProvider({
        "artifacts": [{
            "text": "Turn this into a claim",
            "section": "Ops",
            "artifact_type": "not-a-type",
            "relation": "new",
            "canonical_id": "",
        }]
    })
    providers.reset_chain(Chain([provider]))

    result = workers.atomize("What is actually failing?", ["Ops"])

    assert result[0]["text"] == "What is actually failing?"
    assert result[0]["artifact_type"] == "question"
    assert result[0]["relation"] == "new"


def test_atomize_caps_output_at_two_artifacts():
    provider = AnsweringProvider({
        "artifacts": [
            {
                "text": f"Observation {index}",
                "section": "Ops",
                "artifact_type": "observation",
                "relation": "new",
                "canonical_id": "",
            }
            for index in range(3)
        ]
    })
    providers.reset_chain(Chain([provider]))

    result = workers.atomize("Three things happened", ["Ops"])

    assert len(result) == 2
