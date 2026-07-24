# Magpie Claim Engine eval datasets

These JSONL files are version-controlled Braintrust dataset sources for four separate tasks:

| File | Task | Rows |
|---|---|---:|
| `visible.jsonl` | Turn messy user language into compact user-visible cards | 20 |
| `memory.jsonl` | Derive strict, hidden claim proposals from a visible card | 20 |
| `relations.jsonl` | Select the strongest justified edge, its direction, or no edge | 20 |
| `collision.jsonl` | Produce a novel synthesis, tension, or discriminator | 20 |
| **Total** |  | **80** |

Each task has two cases in each of ten domains: agents/tools, MCP/connectors, RAG/memory, prompts/evals, coding agents, structured outputs, inference cost/latency, observability, safety/privacy, and deployment/runtime proof. Across each file, the four splits contain five rows each.

## Braintrust envelope

Every line is one independent Braintrust dataset record:

```json
{
  "input": {},
  "expected": {},
  "metadata": {
    "case_id": "task-domain-NNN",
    "task": "visible|memory|relations|collision",
    "schema_version": "1.0.0",
    "domain": "stable_domain_slug",
    "challenge_labels": ["one_or_more_labels"],
    "semantic_template_id": "globally_unique_template_id",
    "split": "few_shot|dev|regression|holdout",
    "sensitivity": false
  },
  "tags": ["task", "domain", "split"]
}
```

The task contracts intentionally differ, and model-owned output stays minimal:

- `visible` returns `{"ideas":[{"text","section"}]}`. Identity and epistemic type remain backend-owned.
- `memory` returns `{"atoms":[{"text","polarity","resolver","resolves_by"}]}` in strict `condition → observable → bound` grammar.
- `relations` returns `{"relations":[{"source_id","target_id","relation"}]}`. IDs must come from input; an empty array is the explicit no-edge answer.
- `collision` returns only `{"kind","text"}`, where `kind` is `SYNTHESIS`, `TENSION`, or `DISCRIMINATOR`.

Reference child claims and invariants used by collision or relation judges live under `metadata.reference_claims` and `metadata.reference_invariants`. They are context for scoring, not fields the model is allowed to emit.

## Split and leakage policy

`few_shot` is prompt-development material only. `dev`, `regression`, and `holdout` are scored splits. Do not score `few_shot` rows, and do not place scored rows into model prompts.

Every row has a unique `semantic_template_id`. More importantly, the concepts in `few_shot` are not paraphrase templates for scored rows: each scored row changes the intervention, observable, failure boundary, or relation judgment. A shared domain or broad topic is not considered leakage; a shared causal/test template is.

- `dev`: iterative prompt and scorer development.
- `regression`: frozen cases that protect known distinctions and failure modes.
- `holdout`: sealed comparison set. Do not inspect its expected outputs while changing prompts.

When adding data, mint a new `case_id` and `semantic_template_id`; never recycle an identifier after changing the meaning of a row. If a correction materially changes the semantic test, deprecate the old row and add a new one.

## Safety and hard negatives

All rows are synthetic or sanitized, contain no credentials or personal identifiers, and set `sensitivity=false`. Privacy rows use synthetic identities and abstract project vocabulary. Prompt-injection rows describe the test but do not include executable attack payloads.

Hard negatives are represented by `challenge_labels` and, where needed, explicit `invariants`. They cover:

- transport success versus correct external mutation;
- valid syntax versus semantic consistency;
- direct-identifier removal versus contextual re-identification;
- build/deploy success versus live runtime identity;
- no-edge decisions and `related_to` overuse;
- source grounding, atomicity, polarity, resolver selection, time bounds, and refusal to present proposed criteria as observed facts.
