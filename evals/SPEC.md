# Magpie language and memory eval contract

Version: `magpie-claim-eval/1.0`

This is the executable language contract for Magpie's Braintrust experiments.
It separates what a person sees from what the background memory system stores.
The separation is intentional: readable ideas should not be forced to expose
machine-oriented verification metadata, while hidden claims must be precise
enough to retrieve, compare, test, and eventually resolve.

## 1. Two product layers

### Visible ideas

A visible idea is the object a person reads, moves, keeps, collides, or judges.
The model-facing output stays deliberately small:

```json
{
  "ideas": [
    {
      "text": "Transport success is hiding semantic routing failures.",
      "section": "Reliability"
    }
  ]
}
```

Visible prose may be a thesis, claim, question, tension, or synthesis. It must:

- preserve the source's useful meaning;
- fit the selected section;
- be concise and legible;
- avoid unsupported facts or precision;
- avoid generic consultant language.

Visible prose is not required to use the arrow grammar.

### Hidden memory atoms

A memory atom is a background proposition derived from a visible idea, trace,
receipt, or other bounded source:

```json
{
  "atoms": [
    {
      "text": "100 connector writes → intended target mutated once → ≥99",
      "polarity": "affirms",
      "resolver": "sandbox",
      "resolves_by": null
    }
  ]
}
```

Hidden claim atoms use this grammar:

```text
condition → observable → bound
```

- `condition`: the intervention or context.
- `observable`: the quantity or state read from the world.
- `bound`: the threshold, direction, categorical result, or baseline comparison
  that determines pass or fail.
- `polarity`: `affirms` or `denies`.
- `resolver`: `sandbox`, `logs`, `market`, `search`, or `human`.
- `resolves_by`: an ISO date when the claim carries a deadline.

A claim must flip as a unit. If it can be materially half-true, it is more than
one atom. Mechanisms and dependencies belong in theses or relation edges, not
inside a claim string.

These atoms are unresolved test proposals, not receipts. When a visible idea is
qualitative, the engine may propose a concrete resolution criterion so the idea
can eventually flip. It must not present that proposed threshold as an observed
fact, or invent an observation, date, causal relationship, system component, or
scope. Only receipts change epistemic state.

## 2. Relations

Relations are stored independently from visible ideas and memory atoms:

```json
{
  "relations": [
    {
      "source_id": "atom_0",
      "target_id": "idea_0",
      "relation": "tests"
    }
  ]
}
```

Allowed relation types:

| Type | Meaning |
| --- | --- |
| `derived_from` | The source artifact was extracted or synthesized from the target. |
| `tests` | The source names an observation capable of flipping the target. |
| `supports` | If the source is true, the target becomes more credible. |
| `contradicts` | Both cannot hold under the same scope. |
| `depends_on` | The source requires the target to remain true. |
| `refines` | The source states the same proposition with tighter scope or bounds. |
| `duplicates` | Source and target are propositionally equivalent. |
| `related_to` | They share a subject, but no stronger relation is justified. |

`related_to` is a safe fallback, not a default. The linker should select the
strongest relation justified by the provided artifacts and must not infer a
causal edge from topical similarity.

## 3. Collisions

A collision receives two visible ideas plus an open question and emits:

```json
{
  "kind": "SYNTHESIS",
  "text": "Transport reliability is masking target-selection failure."
}
```

Allowed kinds:

- `SYNTHESIS`: a proposition neither parent entails alone.
- `TENSION`: an incompatibility with named competing predictions.
- `DISCRIMINATOR`: an observable test that separates the parents.

A valid collision must:

- materially depend on both parents;
- add propositional content, not merely combine vocabulary;
- remain faithful to the parents and open question;
- avoid an invented causal bridge;
- enable a new check, decision, or explicit tension.

Novelty without two-parent grounding is hallucination, not synthesis.

The model does not repeat parent IDs or generate child atoms in this call.
Magpie already owns the two selected parents, attaches their lineage
deterministically, then passes the collision text through the hidden-memory
atomizer. This preserves the richer overall collision behavior without asking
the model to fabricate backend identity or provenance.

## 4. Narration register

If an experiment evaluates narration, use terse past-tense statements grounded
in recorded events. Do not add cheerleading, inferred success, or activity that
cannot be traced to an event.

## 5. Banned failure patterns

| Pattern | Failure |
| --- | --- |
| Hedged polarity | `may`, `might`, or `could potentially` absorbs refutation. |
| Vibes without an observable | `improves reliability`, `enhances DX`. |
| Compound claim | Independent propositions joined into one atom. |
| Mediation as a patch | A mechanism is inserted so every outcome can be explained. |
| Unbounded scope | `in some cases`, `for certain users`. |
| Consultant abstraction | `leverage`, `unlock`, `streamline`, `synergy`. |
| Proxy substitution | HTTP success is treated as correct mutation; citation presence as support. |
| False precision | A proposed threshold is presented as an observed source fact. |
| Thesis confetti | A generative thesis is discarded and replaced by disconnected metrics. |
| Vocabulary collision | Parent terms are blended without a new proposition. |

## 6. Evaluation principles

Structural failures are non-compensable. A malformed or invented claim cannot
be rescued by a high style or novelty score.

Deterministic checks run first:

- output shape and enum validity;
- exact arrow-segment count for hidden claims;
- nonempty condition, observable, and bound;
- character limit;
- banned lexicon;
- relation endpoint, self-edge, and duplicate-edge checks;
- deadline syntax.

Semantic judges run second:

- source faithfulness;
- proposition coverage;
- visible idea quality and section fit;
- atomicity and falsifiability;
- proxy validity;
- collision novelty and two-parent grounding;
- relation type and direction.

Results are grouped by task, domain, challenge label, and semantic template.
Overall averages never replace slice-level review.

## 7. Dataset isolation

Rows use one of four splits:

- `few_shot`: available to prompt variants and never scored.
- `dev`: used while developing prompts and scorers.
- `regression`: frozen cases representing known failure modes.
- `holdout`: excluded from prompt material and used for release comparisons.

A semantic template must not cross from `few_shot` into a scored split. Simple
paraphrases count as the same semantic template.

Historical examples are sanitized before being committed. Dataset rows contain
no credentials, personal identifiers, customer payloads, or private source
text. Provenance metadata describes the source class without embedding secrets.
