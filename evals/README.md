# Magpie Braintrust evaluations

This directory evaluates prompt behavior without changing Magpie's production
runtime. `magpie/` remains standard-library-only; Braintrust, Autoevals, and the
OpenAI client are development dependencies in `requirements-eval.txt`.

The Braintrust project is `magpie-claim-engine`. The default generator is
`accounts/fireworks/models/gpt-oss-120b`. The judge delegates to the installed
Autoevals default (currently `gpt-5-mini`); set an explicit override when a
comparison needs to freeze the judge model.

Braintrust CLI login and model-provider credentials are separate. OAuth login
authorizes dataset and experiment access; live inference also needs Fireworks
and OpenAI credentials under **Braintrust Settings → AI Providers**. Fireworks
serves the 120B generator, and OpenAI serves the current Autoevals-default
judge. A local `FIREWORKS_API_KEY` can bypass the gateway for generation, but
the default judge still needs a configured OpenAI route or an explicit judge
endpoint/key override.

## Experiment matrix

There are four independent tasks:

- `visible`: the current user-facing `{text, section}` idea contract
- `memory`: hidden, atomic test proposals grounded in the visible idea
- `relations`: typed edges among known ideas and atoms
- `collision`: a synthesis, tension, or discriminator grounded in both parents

Each task runs against V0–V4:

- V0: sparse baseline instruction
- V1: explicit output and epistemic contract
- V2: V1 plus a positive prompt-only example
- V3: V1 plus task-specific negative guidance
- V4: positive and negative guidance plus a strict provider-enforced JSON Schema

Prompt-only examples are deliberately separate from scored data and use the
reserved `prompt-only-` identifier prefix. Dataset rows on the `few_shot` split
are always excluded. The loader runs `dev`, `regression`, and `holdout` by
default, so a gold answer is visible to scorers but never to the generation
task.

## Run it

The local Braintrust CLI is already authenticated. From this bird directory:

```sh
bt eval --runner .venv-evals/bin/python evals/claim_engine.eval.py
```

That runs the full 4 × 5 matrix at three trials per case. Start with a cheap,
offline contract check:

```sh
MAGPIE_EVAL_STUB=1 MAGPIE_EVAL_TRIALS=1 \
  bt eval --no-send-logs --runner .venv-evals/bin/python \
  evals/claim_engine.eval.py
```

Run a small live comparison before the full matrix:

```sh
MAGPIE_EVAL_TASKS=visible,memory \
MAGPIE_EVAL_VARIANTS=V0,V4 \
MAGPIE_EVAL_TRIALS=1 \
  bt eval --sample 5 --runner .venv-evals/bin/python \
  evals/claim_engine.eval.py
```

The `bt` runner uses the saved OAuth profile. If the Python subprocess does not
receive that credential in a particular shell, set `BRAINTRUST_API_KEY` rather
than committing a key. Never add credentials to `.bt/config.json`; that file
contains only project selection.

The four uploaded annotation/reference datasets are:

- `magpie-visible-v1`
- `magpie-memory-v1`
- `magpie-relations-v1`
- `magpie-collision-v1`

Each has 20 rows and an `initial-80-row-corpus` snapshot. Local JSONL remains
the version-controlled source of truth used by the runner. After changing a
file intentionally, sync it with:

```sh
bt datasets update magpie-visible-v1 \
  --file evals/data/visible.jsonl \
  --id-field metadata.case_id
```

Use the corresponding dataset name/file for the other three tasks, then create
a new named snapshot.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MAGPIE_EVAL_MODEL` | `accounts/fireworks/models/gpt-oss-120b` | generator model |
| `MAGPIE_EVAL_JUDGE_MODEL` | Autoevals default | optional judge override |
| `MAGPIE_EVAL_TASKS` | all four | comma-separated task subset |
| `MAGPIE_EVAL_VARIANTS` | `V0,V1,V2,V3,V4` | prompt subset |
| `MAGPIE_EVAL_TRIALS` | `3` | repetitions per row |
| `MAGPIE_EVAL_SPLITS` | `dev,regression,holdout` | scored data subset |
| `MAGPIE_EVAL_STUB` | `0` | deterministic, network-free task outputs |
| `MAGPIE_EVAL_JUDGE` | on live, off in stub mode | enable LLM judge |
| `MAGPIE_EVAL_BASE_URL` | Braintrust gateway | explicit OpenAI-compatible generator endpoint |
| `MAGPIE_EVAL_API_KEY` | unset | explicit gateway credential override |
| `MAGPIE_EVAL_JUDGE_BASE_URL` | Braintrust gateway | optional judge endpoint override |
| `MAGPIE_EVAL_JUDGE_API_KEY` | unset | optional judge credential override |
| `MAGPIE_EVAL_CONCURRENCY` | `4` | cases evaluated concurrently |
| `MAGPIE_EVAL_TEMPERATURE` | provider default | optional generator temperature |

The default route is `https://gateway.braintrust.dev`. The selected
`gpt-oss-120b` model must be proven with a live smoke run because gateway model
availability can change. If it is not routed by the gateway, set
`FIREWORKS_API_KEY`; the runner will then use Fireworks directly at
`https://api.fireworks.ai/inference/v1`. An explicit `MAGPIE_EVAL_BASE_URL`
takes precedence over both routes.

Dataset files live at `evals/data/{visible,memory,relations,collision}.jsonl`.
Every line has this envelope:

```json
{
  "input": {},
  "expected": {},
  "metadata": {"id": "ai-tools-001", "split": "dev"},
  "tags": ["agents", "tool-use"]
}
```

Use stable case IDs and keep true holdouts out of prompt review. V4's positive
examples live only in `evals/prompts.py`; do not copy them into scored JSONL.

## Reading results

Compare variants within a task rather than forcing one prompt to win globally.
The deterministic scorers catch schema, source-support, edge, and lexical
failures cheaply. The Autoevals judge grades the semantic properties that code
cannot reliably decide: intent preservation, atom quality, edge strength, and
collision novelty. Inspect low-scoring examples as well as means; three trials
make instability visible instead of averaging it away.
