# magpie — a conversation map over memory

Magpie is a thematic lens over Raven, not another memory store and not an
autonomous idea blender. You seed a question or contribute a conversation;
Magpie preserves the speech act (question, decision, constraint, experiment,
and so on), groups ideas into themes, and consolidates repetitions into one
canonical card with occurrence history. The conversation map then shows what
keeps returning, what remains open, and what the conversation produced.

Connections between ideas remain available as an explicit, secondary action.
Background pair generation is off by default. Nothing becomes *settled*
without a receipt — that is still the law, enforced in `magpie/engine.py`, the
single deterministic state machine everything else talks to.

- `magpie/engine.py` — the law: cards, sections, collide/resolve/judge,
  frontier ranking, cap enforcement, ledger, harvest. Pure, no I/O.
- `magpie/providers.py` — one configurable OpenAI-compatible inference client.
- `magpie/workers.py` — typed thematic extraction, canonical relation
  classification, and optional explicit fusion.
- `magpie/storage.py` — SQLite workspaces, the shared idea bank, durable
  Raven projections/exposures, and the durable outbound memory outbox.
- `magpie/raven_client.py` — private Streamable HTTP MCP client used only by
  the Magpie server.
- `magpie/server.py` — shared storage/engine lifecycle, JSON handlers, Raven
  outbox delivery, and optional connection metabolism.
- `magpie/http_runtime.py` — one-port AviaryMCP + browser/JSON HTTP runtime.
- `magpie/mcp_surface.py` — bounded agent tools with explicit workspace routing.
- `app/index.html` — the field UI.

Inference proposes structure; it does not verify it. A generated fusion returns
to the field as an open proposal with provider provenance. Only a human
judgment or a future verification runtime may supply the receipt needed to
settle it. `magpie.server.VerificationHook` is the intentionally dormant
boundary for that future runtime.

Runtime state (SQLite database, legacy snapshot, harvests) lives outside the code directory.
Magpie uses `MAGPIE_RUNTIME_DIR` when set, then `$XDG_STATE_HOME/magpie`, and
otherwise `~/.local/state/magpie`.

## Workspaces and memory

A workspace is a durable, resumable view into one Raven memory graph, not a
separate memory container. Creating one starts with a fresh field; opening one
restores its question, selected memories, local cards, sections, pins, and
layout. The same Raven memory may appear in several workspaces with different
workspace-local presentation state.

SQLite is the source of truth at `MAGPIE_RUNTIME_DIR/magpie.sqlite3`. On first
boot, an existing `state.json` is imported into one workspace and left
untouched as a recovery artifact.

New canonical cards are visible immediately and are also placed into a leased
SQLite outbox. A repeated or refined expression updates the existing card's
occurrence/evolution history instead of creating another visible card; its
provenance is retained in the workspace idea bank. A background Magpie worker
privately calls Raven `remember`; successful writes bind the stable local card
to its canonical Raven memory. Explicit collision children wait for both
parent bindings and are remembered with Raven `derived_from` edges. An outage
never rolls back the visible workspace.

Recall is explicitly proxied through Magpie. Raven results land on the
workspace's **From memory** shelf; importing one creates an ordinary open card,
while dismissal affects only that workspace. Raven confidence and resolution
state are displayed as remote metadata and never turn into Magpie mass,
supported/refuted state, or a receipt. Local keep, move, kill, and judgment do
not send global Raven feedback.

The browser exposes a compact workspace picker. The same lifecycle is
available through the CLI:

```sh
python3 -m magpie.cli workspace list
python3 -m magpie.cli workspace create "Mobile checkout" \
  --question "Why did conversion fall?"
python3 -m magpie.cli workspace open ws_...
python3 -m magpie.cli workspace current
```

## Run

```sh
cd <path-to-aviary>/birds/magpie
uv sync
uv run python -m magpie.server            # 127.0.0.1:7351
uv run python -m magpie.server --port 7351
```

Python 3.11+. The application pins AviaryMCP in `pyproject.toml` and `uv.lock`.

The one listener exposes:

- `/` — the local browser field
- `/api/*` — the existing JSON control plane
- `/mcp` — stateless Streamable HTTP MCP
- `/birdz` — process liveness
- `/mcp-api/v1/*` — AviaryMCP's generated tool catalog/REST/OpenAPI face

MCP tools never use or change the browser's process-wide active workspace.
Every field operation requires a stable `workspace_id`. The safe catalog is
`guide`, `list_workspaces`, `create_workspace`, `get_field`,
`recall_workspace`, `adopt_memory`, `seed_field`, `contribute`, `collide`,
`organize`, `get_conversation_map`, and `harvest`. Human judgment, dismissal,
verification, Raven feedback, and receipt ingestion are intentionally
unavailable over MCP.

Set `MAGPIE_AUTO_CONNECTIONS=1` only when intentionally testing legacy
background pair generation. The default is disabled so themes and recurrence
remain the primary workspace behavior.

## Private Raven upstream

Every client calls Magpie. Browsers, ElevenLabs, Codex, and other MCP clients
never receive Raven's URL or credentials:

```text
clients -> Magpie browser/API/MCP -> private Raven MCP -> Raven Postgres
```

Configure the Magpie process with:

```sh
export MAGPIE_RAVEN_URL=http://127.0.0.1:7326/mcp
# Optional only when the private Raven endpoint requires one:
export MAGPIE_RAVEN_KEY=...
export MAGPIE_RAVEN_AGENT_ID=magpie
```

The independent Apple Silicon-compatible Raven stack lives at
`../raven/compose.magpie.yml`; setup and sealed Codex-worker instructions are
in [`../raven/LOCAL_MAGPIE_STACK.md`](../raven/LOCAL_MAGPIE_STACK.md). It uses a
new pgvector volume and applies Raven's migrations at API startup—there is no
`raven-migrate` sidecar and no import of an existing Raven corpus.

After placing the Fireworks key in Raven's private `.env.magpie`, one command
starts the independent containers and runs Magpie against them:

```sh
./scripts/run-local-raven
# or choose another Magpie listener:
./scripts/run-local-raven --port 7352
```

## ElevenLabs voice agent

The microphone button starts a private ElevenLabs Agent conversation. Speech is
handled by ElevenLabs; the agent operates the current Magpie field through the
MCP tools. The browser sends the selected `workspace_id` as a conversation
dynamic variable and ends the conversation before switching workspaces.

Configure the server, never browser JavaScript:

```sh
export ELEVENLABS_API_KEY=...
export ELEVENLABS_AGENT_ID=agent_...
```

The browser requests `POST /api/voice/session`; Magpie exchanges the server-side
key for a short-lived signed WebSocket URL. The long-lived key is never returned
to the browser.

Configure the ElevenLabs agent with:

- MCP transport: Streamable HTTP
- URL: `https://aviary.finchmcp.com/magpie/mcp`
- Authorization: a Finch key scoped only to Magpie
- Dynamic variable: `workspace_id`
- Prompt: always pass `{{workspace_id}}` to workspace-scoped Magpie tools
- Auto-approved: `guide`, `list_workspaces`, `get_field`, `contribute`
- Approval required initially: `create_workspace`, `seed_field`, `collide`,
  `organize`, `adopt_memory`, `harvest`

The UI handles MCP approval requests with visible Allow/Reject controls. Users
must be informed before starting that they are speaking with an AI and that the
conversation may be recorded or shared according to the configured ElevenLabs
privacy settings.

## Pelican and Finch

Install `aviary-magpie.service` as a systemd user unit, put provider and
ElevenLabs secrets in `~/.config/aviary/magpie.env`, then:

```sh
systemctl --user enable --now aviary-magpie.service
curl -sf http://127.0.0.1:7351/birdz
finch add magpie --service http://127.0.0.1:7351 --json
systemctl --user restart finch-tunnel.service
```

Finch's bare-service registration publishes only `/mcp`; the browser and JSON
API remain loopback-local. An unauthenticated request to the public MCP URL
should return `401`, proving the edge route is live and gated.

Prompt experiments and the hidden-memory claim contract are developed
separately from the production runtime. See
[`evals/README.md`](evals/README.md) for the Braintrust matrix, datasets, and
offline smoke command.

## Inference

The default endpoint is Fireworks' OpenAI-compatible API. Configure the model
and key at runtime:

```sh
export FIREWORKS_API_KEY=...
export MAGPIE_LLM_MODEL=accounts/fireworks/models/gpt-oss-120b
python3 -m magpie.server
```

For another OpenAI-compatible endpoint, also set
`MAGPIE_LLM_BASE_URL`; optionally set `MAGPIE_LLM_API_KEY` and
`MAGPIE_LLM_NAME`. The deterministic stub is never an automatic fallback:
enable it explicitly for offline demos with `MAGPIE_LLM_STUB=1`.

Persistent state defaults to `~/.local/state/magpie`. Override it with
`MAGPIE_RUNTIME_DIR`.

For a deterministic offline test:

```sh
MAGPIE_PROVIDERS=stub uv run python -m magpie.server
```

Then open <http://127.0.0.1:7351>. `MAGPIE_PROVIDERS=stub` keeps the local
test session deterministic and offline; omit it to use the configured
inference backend.

## Drive the runtime from the CLI

`magpie.cli` talks to the public HTTP API used by the browser. It does not
import or mutate the engine directly, so CLI automation exercises the real
server boundary.

Start a server in one terminal:

```sh
MAGPIE_PROVIDERS=stub \
MAGPIE_RUNTIME_DIR=/tmp/magpie-dev \
uv run python -m magpie.server
```

Drive it from another:

```sh
python3 -m magpie.cli state
python3 -m magpie.cli seed "What makes this idea work?"
python3 -m magpie.cli propose "The feedback loop must be repeatable."
python3 -m magpie.cli keep c1
python3 -m magpie.cli move c1 evidence
python3 -m magpie.cli judge c1 yes
python3 -m magpie.cli recall "What prior constraint matters here?"
python3 -m magpie.cli adopt-memory mem_...
python3 -m magpie.cli harvest
```

Every command writes JSON to stdout and returns a nonzero exit status when the
server cannot be reached or rejects the request. Use `--url` to drive a server
on another port or host:

```sh
python3 -m magpie.cli --url http://127.0.0.1:8000 state
```

For an endpoint added before the typed CLI catches up, `call` provides a
generic JSON escape hatch:

```sh
python3 -m magpie.cli call seed --data '{"question":"Does it compose?"}'
```

### Repeatable scenarios

A scenario is a JSON document containing actions, waits, and assertions. Run
one against an existing server:

```sh
python3 -m magpie.cli scenario scenarios/smoke.json
```

Or let the CLI create and stop an isolated server automatically:

```sh
python3 -m magpie.cli run scenarios/smoke.json
```

The isolated form uses a temporary state directory and the offline stub. Pass
`--runtime-dir PATH` to keep the resulting state. Use `--providers configured`
to opt into the inference backend already configured through the
`MAGPIE_LLM_*` environment variables.

Scenario steps use these shapes:

```json
{
  "steps": [
    {"action": "seed", "question": "What are we testing?"},
    {"action": "propose", "text": "A claim generated over HTTP."},
    {
      "wait": {"path": "cards.count", "equals": 1},
      "timeout": 5
    },
    {"action": "keep", "id": "@card:0"},
    {"assert": {"path": "question", "equals": "What are we testing?"}},
    {"state": true}
  ]
}
```

`@card:0` resolves to the first live card's ID immediately before the action.
Available stable metrics are `cards.count`, `cards.open`, `cards.testing`,
`cards.settled`, `ledger.count`, and `providers.count`; ordinary dotted state
paths such as `question` and `cap` are also supported.

## Test

```sh
cd <path-to-aviary>/birds/magpie
pytest tests/
```

`tests/test_engine.py` is engine-only: no server, no network.
