"""Dataset loading and generation tasks for Braintrust experiments.

This module is eval-only. It talks to the OpenAI-compatible Braintrust AI
gateway by default and never imports Magpie's production provider chain.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from . import TASK_NAMES
from .prompts import PROMPT_ONLY_ID_PREFIX, render_prompt, uses_structured_output

DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_BASE_URL = "https://gateway.braintrust.dev"
SCORED_SPLITS = frozenset({"dev", "regression", "holdout"})
DATA_DIR = Path(__file__).with_name("data")

_STRICT_SCHEMAS: dict[str, dict[str, Any]] = {
    "visible": {
        "type": "object",
        "additionalProperties": False,
        "required": ["ideas"],
        "properties": {
            "ideas": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
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
    },
    "memory": {
        "type": "object",
        "additionalProperties": False,
        "required": ["atoms"],
        "properties": {
            "atoms": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "polarity", "resolver", "resolves_by"],
                    "properties": {
                        "text": {"type": "string"},
                        "polarity": {"type": "string", "enum": ["affirms", "denies"]},
                        "resolver": {
                            "type": "string",
                            "enum": ["sandbox", "logs", "market", "search", "human"],
                        },
                        "resolves_by": {"type": ["string", "null"]},
                    },
                },
            }
        },
    },
    "relations": {
        "type": "object",
        "additionalProperties": False,
        "required": ["relations"],
        "properties": {
            "relations": {
                "type": "array",
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source_id", "target_id", "relation"],
                    "properties": {
                        "source_id": {"type": "string"},
                        "target_id": {"type": "string"},
                        "relation": {
                            "type": "string",
                            "enum": [
                                "derived_from",
                                "tests",
                                "supports",
                                "contradicts",
                                "depends_on",
                                "refines",
                                "duplicates",
                                "related_to",
                            ],
                        },
                    },
                },
            }
        },
    },
    "collision": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "text"],
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["SYNTHESIS", "TENSION", "DISCRIMINATOR"],
            },
            "text": {"type": "string"},
        },
    },
}


def _truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: Iterable[str]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _row_id(row: Mapping[str, Any], path: Path, line_number: int) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("id", "case_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return f"{path.name}:{line_number}"


def load_cases(task: str) -> list[dict[str, Any]]:
    """Load one JSONL dataset, excluding prompt-only and non-scored splits.

    Expected row envelope: ``{input, expected, metadata, tags}``. Set
    ``MAGPIE_EVAL_SPLITS`` to narrow the default dev/regression/holdout set.
    ``few_shot`` is always excluded, even if requested.
    """
    if task not in TASK_NAMES:
        raise ValueError(f"unknown eval task {task!r}")
    path = DATA_DIR / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"missing eval dataset {path}; generate or restore the JSONL datasets first"
        )

    requested = set(_csv_env("MAGPIE_EVAL_SPLITS", SCORED_SPLITS))
    requested.discard("few_shot")
    invalid = requested - SCORED_SPLITS
    if invalid:
        raise ValueError(
            "MAGPIE_EVAL_SPLITS contains unscored split(s): " + ", ".join(sorted(invalid))
        )

    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            if "input" not in row or "expected" not in row:
                raise ValueError(f"{path}:{line_number}: row needs input and expected")
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: metadata must be an object")
            split = str(metadata.get("split", "dev"))
            if split == "few_shot" or split not in requested:
                continue
            case_id = _row_id(row, path, line_number)
            if case_id.startswith(PROMPT_ONLY_ID_PREFIX):
                raise ValueError(
                    f"{path}:{line_number}: {case_id!r} uses the reserved prompt-only prefix"
                )
            cases.append(
                {
                    "input": row["input"],
                    "expected": row["expected"],
                    "metadata": {**metadata, "case_id": case_id, "split": split, "task": task},
                    "tags": list(row.get("tags") or []),
                }
            )
    if not cases:
        raise ValueError(f"{path}: no rows selected for splits {sorted(requested)}")
    return cases


def _extract_json(text: str) -> Any:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
                return value
            except json.JSONDecodeError:
                continue
        return {
            "_raw": stripped,
            "_parse_error": f"{first_error.msg} at character {first_error.pos}",
        }


def _api_key(*, include_fireworks: bool = False) -> str:
    names = ["MAGPIE_EVAL_API_KEY"]
    if include_fireworks:
        names.append("FIREWORKS_API_KEY")
    names.extend(
        [
            "BRAINTRUST_API_KEY",
            "OPENAI_API_KEY",
            "TEMP_OPENAI_API_KEY",
        ]
    )
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        "No gateway credential found. Run via the authenticated `bt eval` CLI, "
        "or set BRAINTRUST_API_KEY. For offline validation set MAGPIE_EVAL_STUB=1."
    )


def _generator_endpoint() -> tuple[str, str]:
    explicit_base = os.getenv("MAGPIE_EVAL_BASE_URL")
    fireworks_key = os.getenv("FIREWORKS_API_KEY")
    if explicit_base:
        return explicit_base, _api_key(include_fireworks=True)
    if fireworks_key:
        return "https://api.fireworks.ai/inference/v1", fireworks_key
    return DEFAULT_BASE_URL, _api_key()


def _completion(task: str, messages: list[dict[str, str]], variant: str) -> Any:
    # Imported only when a live eval actually executes.
    from braintrust import wrap_openai
    from openai import OpenAI

    base_url, api_key = _generator_endpoint()
    client = wrap_openai(
        OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("MAGPIE_EVAL_TIMEOUT", "90")),
            max_retries=int(os.getenv("MAGPIE_EVAL_MAX_RETRIES", "2")),
        )
    )
    kwargs: dict[str, Any] = {
        "model": os.getenv("MAGPIE_EVAL_MODEL", DEFAULT_MODEL),
        "messages": messages,
    }
    if os.getenv("MAGPIE_EVAL_TEMPERATURE"):
        kwargs["temperature"] = float(os.environ["MAGPIE_EVAL_TEMPERATURE"])
    if uses_structured_output(variant):
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"magpie_{task}",
                "strict": True,
                "schema": _STRICT_SCHEMAS[task],
            },
        }
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return _extract_json(content or "")


def _stub_visible(input_value: Mapping[str, Any]) -> dict[str, Any]:
    text = str(input_value.get("thought") or input_value.get("text") or "Untitled idea")
    sections = input_value.get("sections") or ["field"]
    section = str(sections[0]) if isinstance(sections, list) and sections else "field"
    return {"ideas": [{"text": " ".join(text.split())[:400], "section": section}]}


def _stub_memory(input_value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "atoms": [
            {
                "text": "run stated check → observed result → matches source-stated bound",
                "polarity": "affirms",
                "resolver": "human",
                "resolves_by": None,
            }
        ]
    }


def _node_ids(input_value: Mapping[str, Any]) -> list[str]:
    nodes = input_value.get("nodes") or input_value.get("candidates") or []
    ids = [
        str(node.get("id"))
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") is not None
    ]
    for key in ("source", "target"):
        node = input_value.get(key)
        if isinstance(node, Mapping) and node.get("id") is not None:
            ids.append(str(node["id"]))
    return list(dict.fromkeys(ids))


def _stub_relations(input_value: Mapping[str, Any]) -> dict[str, Any]:
    ids = _node_ids(input_value)
    if len(ids) < 2:
        return {"relations": []}
    return {
        "relations": [
            {"source_id": ids[0], "target_id": ids[1], "relation": "related_to"}
        ]
    }


def _stub_collision(input_value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "TENSION",
        "text": "The parents imply different outcomes under the same stated boundary.",
    }


_STUBS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "visible": _stub_visible,
    "memory": _stub_memory,
    "relations": _stub_relations,
    "collision": _stub_collision,
}


def make_task(task: str, variant: str) -> Callable[[Mapping[str, Any]], Any]:
    """Build a Braintrust task callable for a task/variant pair."""
    if task not in TASK_NAMES:
        raise ValueError(f"unknown eval task {task!r}")

    def run(input_value: Mapping[str, Any]) -> Any:
        if not isinstance(input_value, Mapping):
            raise TypeError(f"{task} input must be an object")
        if _truthy("MAGPIE_EVAL_STUB"):
            return _STUBS[task](input_value)
        return _completion(task, render_prompt(task, input_value, variant), variant)

    run.__name__ = f"{task.lower()}_{variant.lower()}_task"
    run.__doc__ = f"Run the {task} task with prompt variant {variant}."
    return run
