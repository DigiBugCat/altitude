# Altitude — Spec v0 (design-panel synthesis)

*Product formerly known as Magpie. Synthesized from the winning panel design (B, mean 92.8/100), reworked under two binding owner deltas, with grafts from designs C and E where they strengthen the winner. Reworks forced by the deltas are logged in Appendix A; rejected grafts in Appendix B; the judge ranking in Appendix C.*

---

## 0. The thesis

Magpie's failure was not tuning. It was that **the engine's only compositional operator destroyed its operands.** `Engine.resolve()` (engine.py:414–420) archived both parents when a synthesis was supported. The pivot — "keep ideas separate until they click, like: wait, these things are actually ONE idea" — is incompatible with that line. A click must *add altitude*, not consume ground.

Altitude's core move is therefore: **replace fusion-as-combination with subsumption-as-recognition**, and — per the owner deltas — make the field vertical in *both* directions:

- **Clicks build the ladder upward.** A click is a recognition that two claims are instances of one frame. It is proposed rarely, confirmed by a human, and leaves both instances alive one floor down.
- **Derivation fills the ladder downward.** A frame decomposes into the atomic claims that would make it true. Agents derive those claims; they enter as ungrounded positions beneath the frame.
- **Receipts are the only thing allowed to change state in either direction.** Evidence arrives at the ground and climbs: a receipt flips a claim; the flipped claim re-scores its frame. A click never settles anything; a derivation never asserts anything. Nothing at any layer may drift from the layer below.

And the second inversion: **the ladder is the tracked object, not the ideas.** Positions — an altitude, its load-bearing connections to the floor below, its support state, its receipt state, its last-grounded date — are the durable entities. Ideas are the current occupants of positions and may be rephrased, merged, or replaced without losing the structure.

> Activity is no longer a success metric. A quiet field with three well-folded levels and a short altitude-0 crux list is a healthy Altitude.

---

## 1. Core model

### 1.1 Three floors, structurally different

Frames, claims, and receipts are different kinds of things, not tags on identical cards:

- **Receipts** (ground floor) are evidence events, not ideas. They are the only objects that change epistemic state, in either direction. A receipt attaches to a claim via `resolve()` — the law survives verbatim: nothing settles without a receipt.
- **Claims** (floor 0 of the idea ladder) are atomic, receipt-checkable propositions. They are what `atomize()` extracts from human contributions and what derivation produces beneath frames. Only claims can be `supported` / `refuted`.
- **Frames** (floors 1+) are abstractions over claims or over lower frames. A frame is never directly supported or refuted. Its **support score is computed** from the states of the claims beneath it (§1.5) and is re-derived on every claim state change. Frames are created by exactly two operations: a confirmed click (upward recognition) or explicit human authorship.

Other artifact types — `decision`, `constraint`, `question`, `experiment` — remain first-class **cargo**: they live at floor 0, drive the brief (§3.3), and are never fuel for the automatic climb (§2.2). *(Graft: E — claims-only scan eligibility.)*

### 1.2 Positions: the ladder is the tracked object

The data model inverts. The first-class durable entity is the **position**:

```python
@dataclass
class Position:
    id: str                       # durable, never reused
    workspace_id: str
    floor_kind: str               # 'claim' | 'frame'
    supports: list[str]           # position ids one floor DOWN (load-bearing edges)
    occupant: Occupant            # current idea: text, artifact_type, revision history
    support_state: str            # claims: open|supported|refuted|needs_human
                                  # frames: derived, never stored (see 1.5)
    receipt: Receipt | None       # claims only
    last_grounded_at: float|None  # when a receipt last touched this position
                                  #   or (frames) last propagated up from below
    lineage: set[str]             # union of floor-0 root position ids covered (see 1.6)
    origin: str                   # 'human' | 'click' | 'derivation' | 'recall'
    status: str                   # 'live' | 'folded' | 'vacated' | 'retired'
    folded_under: str | None
    external: bool                # recall-quarantine flag (§4)
```

Consequences, all binding:

- **Rephrasing an occupant never touches the structure.** Edit the text, the position id, `supports`, `lineage`, receipt, and `last_grounded_at` persist. What is never lost is: *this altitude, supported by that floor, last grounded on this date.*
- **Merging or replacing ideas moves occupants between positions**; the vacated position is marked `vacated`, never deleted — history is retained.
- **The never-retry ledger keys on position pairs** (§2.3), so rewording cannot resurrect a settled non-click and cannot silently reopen one either.
- **Staleness is a first-class property**: `now − last_grounded_at`, surfaced per-position in the browser and the harvest.
- **Recall must arrive at an altitude** with support relationships intact (§4) — never as loose extra ideas dumped on one floor.

Altitude is still **derived, never stored**: `altitude(p) = 0` if `floor_kind == 'claim'`, else `1 + max(altitude(c) for c in p.supports)`. Memoized walk in `Engine.altitudes()`. There is no "create a level" operation; every level is the shadow of a confirmed click or an explicit derivation.

### 1.3 The click (upward)

A **click** is a proposal: *positions A and B are occupied by instances of one frame X*. It is legitimate only if X passes the **instance test** — X can be stated without the distinguishing content of either instance, and both read as obvious specializations — enforced as three deterministic, engine-side gates on the worker's returned text:

1. **Generativity** — X must not be a lexical merge. Reject if X's content-word set is a subset of `A ∪ B`'s content words. A tautology ("both concern latency") fails: its words are all borrowed.
2. **Recoverability** — the proposer must emit a one-clause *specializer* per instance (`A = X, in the case of <clause>`). Empty or X-restating specializers → no click.
3. **Scope boundary** *(graft: C)* — the proposer must state what superficially similar cases X does **not** cover. An abstraction that excludes nothing explains nothing. Empty scope boundary → no click.

Important nuance *(graft: E)*: the accepted frame should **faithfully name shared structure, not manufacture novelty**. The generativity gate rejects borrowed vacuity; it does not demand invented content. The old rule "a synthesis must add something absent from both parents" is wrong under identity recognition and is retired (§6).

A failed gate is not a card. It emits nothing and writes an attempt row (§2.3).

### 1.4 Derivation (downward)

The mirror operation, new under Delta 1. `derive(frame_id)` — invoked explicitly by a human or agent tool call, **never by the background loop** — asks a worker: *what atomic, receipt-checkable claims would make this frame true?*

- Output: up to `DERIVE_CAP = 5` proposed claims, each with a falsification hint (`what receipt would flip this`).
- Each accepted proposal creates a **claim position** beneath the frame with `origin='derivation'`, `support_state='open'`, `last_grounded_at=None` — visible as an **ungrounded slot**: structure awaiting evidence. Positions are the scaffolding the deltas make first-class; creating them is cheap and honest because they display as ungrounded.
- Derived claims are excluded from Raven banking (§4) and from the click scanner until a receipt lands or a human pins them — derivation fills structure, it does not assert.
- Evidence then climbs: a receipt flips a derived claim via `resolve()`, and the flip **re-scores the frame** (§1.5). This is the only path by which derivation ever changes anything's state.

### 1.5 Altitude and support are derived, never stored

Two derived quantities, same principle:

- `altitude()` as in §1.2.
- `frame_support(p)`: computed from the floor — the multiset of `support_state` over `p.supports` (e.g. `3 supported / 1 refuted / 2 open`), plus a propagated `last_grounded_at = max` over the floor. Recomputed on every claim state change; never stored authoritatively. A frame with a refuted floor member is visibly cracked; a frame whose floor is all-open is visibly speculative. **Nothing at any layer may drift from the layer below**, because no layer stores what the layer below determines.

### 1.6 What happens to the instances: the fold

On click confirmation, instances are **folded, not archived**. `folded_under` points at the frame. Folded positions:

- **State**: the new frame position is created in state **OPEN with no receipt** *(graft: C — reworks the winner's receipt-bearing click)*. Accepting a click means "organize these together," not "this proposition is true." Structural recognition can never masquerade as epistemic settlement; `resolve()` + receipt remains the sole path to supported/refuted, and frame support only ever comes from the floor (§1.5). The confirmation is recorded as provenance (`confirmed_by`, timestamp, human wording if edited), not as a receipt.
- **Visibility**: hidden at the frame's altitude, fully present on descent. Never `archived`.
- **Mass** *(graft: C — lineage-set mass, replacing the winner's `max+0.1·(count−1)` formula)*: every position carries `lineage` — the union of floor-0 root position ids it covers. Display mass of a frame is computed from the **union** of its lineage roots' masses, so two frames sharing roots cannot double-count them. Importance tracks unique ground covered, not ladder height. Instances keep their own mass at their own floor; mass is never transferred.
- **Cap pressure**: `enforce_cap()` counts a fold as **one** unit at the frame's altitude. Recognition is the only operation that structurally reduces field pressure — the system is incentivized toward compression, not expansion. Retirement ordering is an explicit invariant *(graft: E)*: never retire positions with dependents in `supports`; never retire folded instances; never retire human roots or `needs_human`; prefer unlinked, low-mass, machine-origin cargo — in that order.
- **Recall**: folded instances remain individually recallable and individually resolvable. A fold is a lens, not a merge.
- **Unfold** is always cheap: clears `folded_under` on the instances and marks the frame position `vacated` (Delta 2: the position record persists with its history; no silent deletion). A frame that loses all instances is vacated or reopened automatically — no semantic shells *(graft: E)*.

---

## 2. The metabolism, redesigned

### 2.1 What is deleted

`Engine.best_pair()` (engine.py:493–518), `affinity()`, and `AFFINITY_THRESHOLD` go. The arithmetic: `affinity` returns `0.36 + (mass_a + mass_b)·0.16` for any same-section pair against a `0.34` threshold — **every same-section pair clears it, always**, and the 3-second `_metabolism_loop` (server.py:705) drains the space exhaustively. 10 contributions → 45 pairs → 45 cards was the loop working as written.

Also deleted outright — not feature-flagged: `_collide_and_fuse` (server.py:664), the `testing`-state placeholder from `Engine.collide()` (engine.py:320–334), and `workers.fuse()`. The `collide` MCP tool and `/api/collide` become `propose_click(a, b)` — a human-initiated request for the same recognition attempt, landing in the same inbox, never in the field.

### 2.2 What replaces it: a scanner that mostly returns nothing

The background loop materializes nothing. Per tick, per workspace, it:

1. Selects **at most one** candidate pair from *unattempted* pairs (§2.3). Eligibility: **claims only** as automatic fuel *(graft: E)* — cargo types never scan; recall-quarantined (`external`) positions never scan (§4); derived-ungrounded claims never scan (§1.4); and — the **anti-recursion invariant** *(graft: E, stated as a hard rule)* — a frame becomes scan-eligible only after its floor has at least one supported member (or a human pin), may be compared only against other frames or unfolded claims at **adjacent altitude**, and **never against its own descendants** (checked via `lineage` intersection walk). The ladder must not grind against its own output.
2. Ranks candidates by embedding cosine over `idea_embeddings` (schema v1, finally consumed), **preferring same-altitude pairs** *(graft: E)* so the ladder grows level-by-level. Embeddings are a bounded ranking index only — never an acceptance threshold, never a materialization trigger *(graft: B-via-sol, made explicit)*. Cosine below `CLICK_FLOOR = 0.62` → tick does nothing.
3. Calls `workers.recognize(a, b, question)`, whose contract is **`{"click": false}` by default**. Prompt states most pairs are unrelated and "no" is the expected answer. Returns `{click, abstraction, specializer_a, specializer_b, scope_boundary}`.
4. Runs the §1.3 gates. Pass → row in `click_candidates`. Fail → attempt record, nothing emitted.

**Fail-closed provider semantics** *(graft: E)*: if the provider errors or times out, the attempt is recorded as `outcome='failed'` — which does **not** consume the pair (§2.3); a positive click with empty abstraction text is coerced to `no_click`. An outage must never permanently suppress a legitimate future recognition.

**Cadence**: `METABOLISM_PERIOD` = 30.0s (from 3.0s), gated on **field quiescence** — no scan within 60s of the last human contribution. Emission rate is governed by inbox capacity (§2.4), not timer frequency. A **fleet-wide LLM call budget** across workspaces caps total recognition spend per day *(graft: E)*.

**Anti-starvation trigger** *(graft: E — sharper than near-miss inspection alone)*: after `N = 10` new claims in a workspace with zero scans (budget exhaustion, quiescence never reached), force exactly one scan wave. Near-misses are recorded as **private metrics**, inspectable on request via `pending_clicks(include_rejected=True)` — never field spam.

### 2.3 Never-retry, correctly

The old guard, `existing_pairs` from live cards (engine.py:508–514), stored the memory of a failure *as* the failure's output — archive the child and the pair resurrects. Memory must be separate from output:

```sql
CREATE TABLE click_attempts (
  workspace_id      TEXT NOT NULL,
  position_a        TEXT NOT NULL,   -- min(position ids): durable, survives rephrasing
  position_b        TEXT NOT NULL,   -- max(position ids)
  operation_version INTEGER NOT NULL DEFAULT 1,
  outcome           TEXT NOT NULL,   -- 'no_click'|'gate_failed'|'declined'|'clicked'|'expired'|'failed'
  attempted_at      REAL NOT NULL,
  PRIMARY KEY (workspace_id, position_a, position_b, operation_version)
);
```

- Keys are **position ids** (Delta 2): rewording an occupant cannot resurrect a settled non-click. Re-adopted external material is deduplicated onto its existing position via `idea_fingerprint` at adoption time (§4), so a re-adopted memory cannot mint a fresh position to dodge the ledger.
- Semantic outcomes (`no_click`, `gate_failed`, `declined`, `expired`) are **terminal by default**. `failed` (provider outage) does not consume the pair.
- **Reconsideration is an explicit operation** *(graft: C/E — replaces the winner's "declined + refined re-offer" exception)*: `reconsider_pair(a, b)` is a deliberate human act that inserts a new row at `operation_version + 1`, provenance-visible. Retry is a paper-trailed door, never an automatic one.

Migration seeds this ledger from history (§5).

### 2.4 The emergence inbox

Candidates accumulate in `click_candidates`, capped at **3 open per workspace**; a fourth is not generated until one resolves. Each shows: proposed frame text, both specializers, the scope boundary, and both instances verbatim. Verdicts:

- **Accept** → fold executed (§1.6), frame position created OPEN, provenance recorded.
- **Accept with edit** → human wording becomes the occupant; same fold.
- **Decline** → `declined` attempt row.
- **Ignore** → **click TTL** *(graft: E)*: 7 days unacted auto-expires the candidate (`expired` row, no retry).

This inbox — plus explicit `derive()` acceptance (§1.4) — is the *only* path from inference to field material.

---

## 3. Navigation

### 3.1 Browser

The field renders one altitude at a time. Positions at altitude *n*; folded instances hidden. A frame shows an instance-count badge, its computed support summary (`3✓ 1✗ 2○`), and a **staleness tint** from `last_grounded_at`. Clicking **descends**: the view swaps to the frame's floor, frame pinned as header, its derived-ungrounded slots shown as visibly empty sockets awaiting receipts. `Esc` ascends. Section columns persist across altitudes.

A single altitude slider from `0` to `max(altitudes)` sets the ceiling. At the top, only the highest frames; at 0, raw claims and cargo. This is the whole navigator, and it is cheap because altitude and support are both derived.

The emergence inbox is a separate tray. Candidates never occupy field space.

### 3.2 MCP

`get_field(workspace_id, altitude=None)` — defaults to top; every returned position carries `supports`, support summary, and `last_grounded_at` intact. `descend(position_id)`. `derive(frame_id)`. `propose_click(a, b)`. `pending_clicks(workspace_id, include_rejected=False)`. `resolve_click(candidate_id, verdict, text=None)`. `reconsider_pair(a, b)`. `unfold(frame_id)`.

`get_conversation_map` (mcp_surface.py:129) survives; its `between_ideas` key becomes `frames`, ranked by altitude then lineage mass, truncated to 3 — the truncation was already the right instinct.

### 3.3 `harvest(altitude)` and the decision-ready brief

`Engine.harvest()` (engine.py:666) currently dumps everything. Replaced by `harvest(altitude=None, max_items=12)`:

- **The spine**: positions at or above `altitude`, ranked by lineage mass, each with specializers and support summary inline so a reader can descend in prose.
- **Cracks and stale ground**: frames with a refuted floor member, and positions whose `last_grounded_at` exceeds the staleness window — the ladder's honesty section.
- **Cruxes**: `question` cargo whose resolution would change a position above it; plus derived-ungrounded claims under high-mass frames (the receipts most worth going and getting).
- **Decisions / constraints**: `artifact_type` filtered, unchanged — the semantic types earn their keep here.
- **Experiments**: `experiment` cargo plus migrated `DISCRIMINATOR` text.
- **Unresolved**: `needs_human` positions.
- **What changed**: clicks confirmed, receipts landed, frames re-scored since last harvest, from the events table.

Hard cap `max_items` per section. A brief that cannot be read in two minutes is a dump. Full state remains as `state()` — a debugging surface, not a deliverable.

---

## 4. Recall integration

The contamination had a precise cause: `_bank_card` (server.py:187) enqueued a Raven write for **every** card, and `_run_fuse` (server.py:653) called it on machine fusions — every runaway auto-card was written to shared memory and recalled back later. Fixes, going forward and backward:

**Typed export classes on outbox rows** *(graft: C — replaces the winner's inline three-way guard)*. Every queued Raven write carries an `export_class`: `human_root` (atomized human contribution), `human_curated_frame` (confirmed click, banked with `hints={"derived_from": [floor memory ids]}` — the existing outbox parent-waiting at server.py:764–784 makes the Raven DAG a ladder for free), or `settled_claim` (receipt-resolved). Eligibility is an inspectable property of each queued write, not a condition buried in a function. Machine candidates, derived-ungrounded claims, and unpromoted folds have no class and are never queued; accepted folds stay **workspace-local unless promoted** by a human.

**Recall arrives AT an altitude** (Delta 2 — binding). When Raven returns a `human_curated_frame`, `_adopt_raven_memory` (server.py:917) reconstructs the sub-ladder — frame position plus floor positions from `derived_from` — with support edges and last-grounded dates intact, never as loose ideas on one floor. All adopted positions arrive `external=True`, `state="needs_human"`, quarantined: **excluded from the click scanner until a human pins them**. Adoption dedupes via `idea_fingerprint` onto existing positions rather than minting duplicates. Recall can inform without breeding.

**Cleanup of existing damage** *(grafts: C, E)*: a **local suppression registry** tags known Magpie-generated and stress-test memory ids so they stop resurfacing immediately — without deleting them or mutating Raven's global epistemic state; reversible. Plus **durable workspace-local dismissal by memory id**: a human-dismissed exposure never resurfaces in that workspace, regardless of future recall scoring.

**Recall is scoped and never self-referential.** `_recall_workspace` passes `exclude_tags=["magpie", "altitude"]` unless overridden, and filters any memory whose `derived_from` chain reaches a position live in this workspace. You cannot recall your own echo.

---

## 5. Migration path (from Magpie)

Altitude inherits the Magpie codebase (`birds/magpie`); this section is the surgical inventory.

**engine.py**

- Delete `collide()` (320–334), `affinity()` (488), `best_pair()` (493–518), `AFFINITY_THRESHOLD`.
- Delete lines 414–420 in `resolve()` — the parent-archiving branch. The single most important deletion in the design.
- Introduce `Position`/`Occupant` per §1.2; `Card` becomes the occupant payload. Legacy `parents` maps to a `provenance` edge list (see backfill — **not** to `supports`); legacy `kind == "synthesis"` maps to a provisional annotation, not to `floor_kind='frame'`.
- Add `altitudes()`, `frame_support()`, `descend()`, `confirm_click()`, `fold()`, `unfold()`, `derive()`, `reconsider_pair()`.
- `live()` gains an `altitude` filter; `enforce_cap()` implements the §1.6 ordered retirement invariant and counts folds as one unit.
- `harvest()` rewritten per §3.3; `digest()` keeps its artifact-type sections verbatim; folded instances remain in `recurring_ideas` regardless of altitude.
- `from_state`'s receipt-revalidation (engine.py:718–722) kept untouched — the law surviving deserialization, now also catching any legacy synthesis whose receipt didn't migrate.

**workers.py** — `fuse()` deleted; `recognize()` added (negative default, specializer + scope-boundary contract); `derive()` added (falsification-hint contract). `atomize()` kept entirely as-is: its typed-artifact extraction and canonical-relation matching are the parts that already work.

**server.py** — `_collide_and_fuse` deleted; `_bank_card` replaced by export-class enqueueing (§4); `_metabolism_loop` re-cadenced with quiescence gate, inbox-capacity governor, fleet budget, anti-starvation counter; `_adopt_raven_memory` rewritten for sub-ladder reconstruction + quarantine.

**storage.py** — schema v4: `positions`, `occupant_revisions`, `click_attempts`, `click_candidates`, `suppression_registry`, `dismissals`, `outbox.export_class`. Snapshot remains authoritative; tables are indices. `idea_embeddings` gets its first consumer.

**Backfill — conservative, one pass per workspace** *(graft: C — replaces the winner's auto-converting backfill)*:

- **Never auto-convert receipted syntheses into folds.** Combination provenance is not identity recognition; fabricating `supports` edges from `parents` would seed the ladder with exactly the structure the click gates exist to prevent. Instead: legacy syntheses keep `parents` as **provenance edges**, are surfaced in a one-time migration inbox, and become frames only on human confirmation through the standard gates.
- Archived parents of receipted syntheses are **un-archived** as floor-0 positions — recovering material the old law destroyed — but left unfolded pending that confirmation.
- Syntheses without receipts revert to `needs_human` claims at floor 0.
- Every historical pair, from live *and* archived cards, is written to `click_attempts` with `outcome='no_click'`, `operation_version=1` — migration itself seeds the never-retry memory. Historical retries are collapsed to one row.
- The suppression registry (§4) is seeded with all Raven memory ids traceable to machine fusions and stress tests.

**Unchanged:** workspace isolation, the events queue, the Raven outbox and its parent-waiting backoff, the MCP transport, the CLI, `_persist_engine`'s transaction discipline.

---

## 6. Evaluating the prior recommendations

**Falls out naturally** — *only claims are supportable* (now structural: frames cannot carry receipts at all, §1.1); *classify the relationship first* (the only relationship that materializes is instance-of, so classification collapses into the click gates); *materialize only on acceptance* (the inbox and the derive-confirmation flow are the only doors).

**Still needed as separate rules** — *never retry an attempted pair* (needs its own ledger; nothing about ladders provides it); *reject invented evidence* (provider-level, orthogonal); *semantic types* (they do work no ladder does — they make the brief sortable).

**Wrong under the new frame** — *"every synthesis must add something absent from both parents"* is wrong under identity recognition *(graft: E, superseding the winner's own framing)*: a frame should faithfully **name shared structure, not manufacture novelty**; the generativity gate rejects borrowed vacuity, which is a different and correct demand. *"1–3 candidate connections"* was right in number, wrong in framing: three *open at a time*, refilled only on resolution — never per tick. And *tension/dependency/analogy as materializable outputs* is wrong: under identity recognition a tension is evidence these are **not** one idea — a `no_click`. The old `TENSION` kind is exactly how compatible claims got mislabeled as irreconcilable: the schema offered it as an output, so the model produced it. Removing the option removes the failure.

---

## 7. Failure analysis

**7.1 Recognition inflation — the ladder grows because the model wants to please.** Gates catch lexical merges and boundary-less abstractions, but not all fluent vacuity. *Guardrails:* per-workspace **click budget** — at most one confirmed click per 5 human contributions, engine-enforced; compression cannot outrun input. Audit metric: fraction of frames with ≥3 instances — a ladder of exclusively binary folds is a rebranded fusion tree, and the ratio makes it visible.

**7.2 The empty field — the inverse.** Gates and budgets this conservative may produce a scanner that never clicks. This is the *acceptable* failure — plain notes were the benchmark — but still failure. *Guardrails:* the anti-starvation wave (§2.2); near-miss metrics inspectable via `pending_clicks(include_rejected=True)` showing which gate failed. **The success metric is fixed in advance** *(graft: E)*: accepted folds that **later receive receipts on their floor and appear in harvest briefs** — never click volume. If the weekly accept rate is ~0, loosen `CLICK_FLOOR` and gate thresholds with the near-miss evidence in hand — **never** re-enable auto-materialization. That metric definition is the strongest available defense against activity re-becoming the proxy for insight.

**7.3 Altitude as a hiding place.** A wrong frame buries good claims one floor down where nobody descends; the field looks clean because material was concealed. Most dangerous because it *presents as success*. *Guardrails:* (a) unfold is always cheap, vacating the position with history intact; (b) folded instances stay in `recurring_ideas` regardless of altitude, so repetition surfaces through the floor; (c) a frame never descended into, referenced, or harvested within 14 days is auto-flagged — an abstraction nobody uses is not a level, it is a lid; (d) the staleness tint (§3.1) makes an ungrounded ceiling visually distinct from a supported one.

**7.4 Derivation sprawl — the new failure the deltas introduce.** Downward derivation can mint ungrounded scaffolding faster than receipts arrive, turning the field into a wishlist. *Guardrails:* `DERIVE_CAP = 5` claims per frame per derivation; derived-ungrounded positions are scanner-ineligible and bank-ineligible; they render as empty sockets, not content; a frame whose derived floor stays fully ungrounded for 14 days is flagged alongside 7.3(c) — a frame that cannot state its own evidence is a hypothesis, and the display must say so.

---

**Excluded by design:** parallel "quick fusion" modes, a tension inbox, auto-materialization behind a feature flag, background-initiated derivation. Each is a second channel that would restore the old behavior beside the new law, and Magpie's failure was precisely what happens when generation has a path that judgment does not gate.

---

## Appendix A — Delta reconciliation

Reworks of the winning skeleton forced by the binding owner deltas:

1. **Three structural floors (Delta 1)** — The winner had one `Card` class with `kind` tags and derived altitude. Reworked into structurally distinct frames / claims / receipts (§1.1): receipts are no longer cards at all, frames cannot carry receipts, and only claims are resolvable.
2. **Downward derivation (Delta 1)** — The winner was upward-only. Added `derive()` (§1.4) as the mirror operator, explicitly invoked, producing ungrounded claim positions beneath frames. To preserve the winner's single-channel law ("nothing enters the field automatically"), derivation is never background-initiated and its output is scanner- and bank-ineligible until grounded or pinned.
3. **Receipts as sole state-changer in both directions (Delta 1)** — The winner recorded the click itself as a receipt-bearing act. Reworked (with graft C, §1.6): a confirmed click creates an OPEN frame with provenance, no receipt; frame support is computed from the floor (§1.5) and only receipts move it. "Nothing at any layer may drift from the layer below" is now enforced by construction — no layer stores what the layer below determines.
4. **Position-first data model (Delta 2)** — The winner tracked cards with `subsumes` edges. Inverted (§1.2): positions are the durable entities; ideas are occupants with revision history. Consequences propagated everywhere: never-retry keys on position pairs (not the winner's idea fingerprints — see Appendix B #2); unfold vacates positions instead of deleting frames; staleness (`last_grounded_at`) is a tracked property surfaced in browser and harvest; lineage/mass attach to positions.
5. **Recall arrives at an altitude (Delta 2)** — The winner adopted memories as single floor-0 cards. Reworked (§4): banked frames export with `derived_from` floor ids, and adoption reconstructs the sub-ladder — position, floor, support edges, grounding dates — quarantined as a unit.
6. **Backfill (Delta 1 + graft)** — The winner auto-converted receipted legacy syntheses into folds. Under Delta 1's floor separation (combination provenance ≠ identity recognition) and graft C, migration is conservative: provenance edges preserved, human confirmation required before any historical relationship becomes structure (§5).

## Appendix B — Rejected grafts

1. **[terra] Replace B's lexical-novelty (generativity) gate with C's broader click-validity model.** Rejected as a *replacement*; adopted C's scope boundary *additively* (§1.3). The content-word subset test is deterministic, free, and mechanically closes the tautology hole; removing it in favor of a broader (and partly model-judged) validity notion reopens exactly the failure it was built to prevent. Two other judges framed the scope boundary as additive, which is the coherent reading.
2. **[grok] Idea-fingerprint pair keys as the primary never-retry key.** Superseded, not adopted: Delta 2 makes position ids the durable identity, which handles rewording strictly better (rephrasing preserves the position, so the ledger row still binds). The fingerprint survives only as the adoption-time dedupe (§4) that prevents re-adopted memories from minting fresh positions to dodge the ledger — the one job position ids cannot do alone.
3. **[winner's own §2.3 exception] "Declined pairs re-offered once if a card was refined."** Removed in favor of the explicit `operation_version` reconsideration operation (graft C/E, §2.3). An automatic reopening door on edit churn is exactly the silent-retry pattern the ledger exists to kill; retry must be a deliberate, provenance-visible human act.
4. **[E] "Surface near-misses on request" vs "private metrics only" — no rejection, but the merge is deliberate:** near-misses are private metrics *with* an operator inspection surface (`include_rejected=True`). Pure metrics with no drill-down would force blind tuning, which the winner correctly forbade.

All other listed grafts were adopted (many were duplicates across judges and are consolidated; adoption sites are marked *(graft: …)* inline).

## Appendix C — Judge ranking

| Design | Model | Score | Notes |
|--------|-------|-------|-------|
| B | opus | **92.8** over 4 scores | Winner; skeleton of this spec |
| C | sol | 88.0 over 4 scores | Primary graft source (lineage mass, OPEN folds, operation_version, export classes, suppression registry, conservative migration, scope boundary) |
| E | grok | 81.5 over 4 scores | Graft source (fail-closed provider, anti-starvation + success metric, anti-recursion, cap ordering, TTL, fleet budget, closing invariant) |
| D | terra | 34.1 over 9 scores | No grafts adopted from D itself |
| A | fable | 13.0 over 4 scores | Blocker report; yielded the process finding in Appendix D |

## Appendix D — Process note (carried from A's blocker, independently confirmed)

The workflow harness interpolated a JavaScript `undefined` into the brief path handed to designers (and to this synthesis step: "Read the brief at undefined"). Fix the calling harness to **fail loudly on an unset variable** rather than passing the literal string through — otherwise future runs will keep silently asking agents to read a nonexistent file, and the failure will keep surfacing as agent-side judgment calls about whether to guess.