"""Braintrust experiment matrix for the Magpie claim engine.

The Braintrust CLI discovers each ``Eval`` call in this file. By default the
matrix is four tasks x five prompt variants with three trials per case.
Environment filters can register a smaller slice for smoke runs.
"""

from __future__ import annotations

import os
import re

from autoevals import LLMClassifier
from braintrust import Eval, wrap_openai
from openai import OpenAI

from evals import PROJECT_NAME, TASK_NAMES, VARIANT_NAMES
from evals.scorers import deterministic_scorers_for
from evals.tasks import DEFAULT_BASE_URL, DEFAULT_MODEL, load_cases, make_task

def _truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _selected(name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return allowed
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    lookup = {value.lower(): value for value in allowed}
    unknown = [value for value in requested if value.lower() not in lookup]
    if unknown:
        raise ValueError(f"{name} contains unknown values: {', '.join(unknown)}")
    return tuple(lookup[value.lower()] for value in requested)


def _gateway_key() -> str:
    for name in (
        "MAGPIE_EVAL_JUDGE_API_KEY",
        "BRAINTRUST_API_KEY",
        "OPENAI_API_KEY",
        "TEMP_OPENAI_API_KEY",
    ):
        if os.getenv(name):
            return os.environ[name]
    # The authenticated `bt eval` runner injects profile credentials before
    # Python execution. This branch mainly improves direct-run diagnostics.
    raise RuntimeError(
        "No judge credential found. Run with authenticated `bt eval`, set "
        "BRAINTRUST_API_KEY, or disable judges with MAGPIE_EVAL_JUDGE=0."
    )


_JUDGE_RUBRICS = {
    "visible": (
        "Judge whether the candidate preserves the source intent as one or two "
        "clear user-visible ideas, assigns only supplied sections, and invents "
        "nothing. Rich theses should remain prose rather than being flattened "
        "into arbitrary metrics."
    ),
    "memory": (
        "Judge the candidate atoms for source grounding, coverage, atomicity, "
        "and falsifiability. A concrete bound may be a proposed resolution "
        "criterion even when the source is qualitative. Penalize it only when "
        "it is unsuitable, or when the candidate presents invented observations, "
        "dates, causal claims, or unsupported scope as source facts."
    ),
    "relations": (
        "Judge endpoint validity, direction, and relation type. Require the "
        "strongest relation justified by the input. Penalize hallucinated edges "
        "and related_to used as a topical-similarity escape hatch."
    ),
    "collision": (
        "Judge whether the result is grounded in both parents and introduces a "
        "synthesis, tension, or discriminator that neither parent entails alone. "
        "Penalize restatement, vocabulary blending, and invented bridges."
    ),
}


def _judge(task: str) -> LLMClassifier:
    client = wrap_openai(
        OpenAI(
            api_key=_gateway_key(),
            base_url=os.getenv("MAGPIE_EVAL_JUDGE_BASE_URL", DEFAULT_BASE_URL),
            timeout=float(os.getenv("MAGPIE_EVAL_JUDGE_TIMEOUT", "90")),
            max_retries=int(os.getenv("MAGPIE_EVAL_MAX_RETRIES", "2")),
        )
    )
    prompt = (
        _JUDGE_RUBRICS[task]
        + "\n\nINPUT:\n{{input}}\n\nCANDIDATE:\n{{output}}\n\nREFERENCE "
        "BEHAVIOR:\n{{expected}}\n\nReturn exactly PASS, MIXED, or FAIL. PASS "
        "means fully grounded and useful; MIXED means a material but repairable "
        "defect; FAIL means the contract or source meaning is substantially wrong."
    )
    kwargs = {
        "name": f"{task}_quality",
        "prompt_template": prompt,
        "choice_scores": {"PASS": 1.0, "MIXED": 0.5, "FAIL": 0.0},
        "client": client,
        "use_cot": True,
    }
    # model=None deliberately delegates to the installed Autoevals default.
    # An explicit override is useful for controlled judge-model comparisons.
    if os.getenv("MAGPIE_EVAL_JUDGE_MODEL"):
        kwargs["model"] = os.environ["MAGPIE_EVAL_JUDGE_MODEL"]
    return LLMClassifier(
        **kwargs,
    )

tasks = _selected("MAGPIE_EVAL_TASKS", TASK_NAMES)
variants = _selected("MAGPIE_EVAL_VARIANTS", VARIANT_NAMES)
trial_count = int(os.getenv("MAGPIE_EVAL_TRIALS", "3"))
if trial_count < 1:
    raise ValueError("MAGPIE_EVAL_TRIALS must be at least 1")

stub_mode = _truthy("MAGPIE_EVAL_STUB")
judge_enabled = _truthy("MAGPIE_EVAL_JUDGE", default=not stub_mode)
generator_model = os.getenv("MAGPIE_EVAL_MODEL", DEFAULT_MODEL)
model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", generator_model).strip("-")

for task_name in tasks:
    for variant_name in variants:
        scorers = deterministic_scorers_for(task_name)
        if judge_enabled:
            scorers.append(_judge(task_name))
        Eval(
            PROJECT_NAME,
            data=lambda task_name=task_name: load_cases(task_name),
            task=make_task(task_name, variant_name),
            scores=scorers,
            experiment_name=f"{task_name}-{variant_name.lower()}-{model_slug}",
            trial_count=trial_count,
            metadata={
                "eval_task": task_name,
                "prompt_variant": variant_name,
                "generator_model": generator_model,
                "judge_model": (
                    os.getenv("MAGPIE_EVAL_JUDGE_MODEL", "autoevals-default")
                    if judge_enabled
                    else "disabled"
                ),
                "stub_mode": stub_mode,
                "data_policy": "scored-splits-only-no-few-shot",
            },
            tags=["claim-engine", task_name, variant_name.lower()],
            max_concurrency=int(os.getenv("MAGPIE_EVAL_CONCURRENCY", "4")),
            timeout=float(os.getenv("MAGPIE_EVAL_CASE_TIMEOUT", "180")),
        )
