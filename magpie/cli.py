"""Command-line control plane for a running Magpie HTTP server.

The CLI deliberately talks to the public JSON API instead of importing the
engine.  A CLI-driven scenario therefore exercises the same boundary as the
browser and can also target a remote or deployed Magpie instance.

The ladder (SPEC §1–§3) is reachable the same way. Note what is NOT here:
there is no command that mints a frame directly. The only routes upward are
the emergence inbox (`propose-click` → `inbox` → `confirm-click`) and
recall adoption, exactly as over MCP.

Examples:

    python3 -m magpie.cli state
    python3 -m magpie.cli seed "What would make this idea work?"
    python3 -m magpie.cli propose "The smallest useful loop is deterministic."
    python3 -m magpie.cli positions --altitude 0
    python3 -m magpie.cli propose-click c1 c2
    python3 -m magpie.cli inbox --include-rejected
    python3 -m magpie.cli confirm-click cand1 andrew
    python3 -m magpie.cli derive c7
    python3 -m magpie.cli harvest --altitude 1
    python3 -m magpie.cli scenario scenarios/smoke.json
    python3 -m magpie.cli run scenarios/smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, TextIO

DEFAULT_URL = "http://127.0.0.1:7351"
ACTIONS = (
    "seed",
    "propose",
    # §2.1: `collide` is gone, not aliased. `click/propose` asks the same
    # question and can only reach the emergence inbox.
    "click/propose",
    "click/pending",
    "click/resolve",
    "click/reconsider",
    "derive",
    "field",
    "descend",
    "unfold",
    "verify",
    "judge",
    "keep",
    "kill",
    "move",
    "section",
    "harvest",
    "recall",
    "recall/adopt",
    "recall/dismiss",
)


# Subcommands whose readable name differs from their API path.
#
# `confirm-click` / `decline-click` are readable spellings of one verdict on
# `resolve-click`, not a second door: they post to the same endpoint, which
# still refuses an acceptance without `confirmed_by` (§1.6). There is no CLI
# path to a frame that does not pass through the emergence inbox.
_COMMAND_ACTIONS = {
    "adopt-memory": "recall/adopt",
    "dismiss-memory": "recall/dismiss",
    "propose-click": "click/propose",
    "pending-clicks": "click/pending",
    "inbox": "click/pending",
    "resolve-click": "click/resolve",
    "confirm-click": "click/resolve",
    "decline-click": "click/resolve",
    "reconsider-pair": "click/reconsider",
}


class CliError(RuntimeError):
    """A user-facing API, scenario, or runtime error."""


class Client:
    """Small stdlib-only client for Magpie's public HTTP API."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._workspace_id: str | None = None

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw.decode("utf-8")).get("error")
            except Exception:
                detail = raw.decode("utf-8", errors="replace") or str(exc)
            raise CliError(f"Magpie returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CliError(f"cannot reach Magpie at {self.base_url}: {exc}") from exc

        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CliError("Magpie returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise CliError("Magpie returned a non-object JSON response")
        if result.get("ok") is not True:
            raise CliError(str(result.get("error") or "Magpie rejected the request"))
        result = dict(result)
        result.pop("ok", None)
        return result

    def get_state(self) -> dict[str, Any]:
        path = "/api/state"
        if self._workspace_id:
            path += "?" + urllib.parse.urlencode({"workspace_id": self._workspace_id})
        return self._request("GET", path)

    def list_workspaces(self) -> dict[str, Any]:
        return self._request("GET", "/api/workspaces")

    def get_current_workspace(self) -> dict[str, Any]:
        result = self._request("GET", "/api/workspaces/current")
        workspace = result.get("workspace") or {}
        if workspace.get("id"):
            self._workspace_id = str(workspace["id"])
        return result

    def bind_current_workspace(self) -> str:
        """Pin later actions and reads to one explicit workspace."""
        if not self._workspace_id:
            self.get_current_workspace()
        if not self._workspace_id:
            raise CliError("Magpie returned no current workspace id")
        return self._workspace_id

    def create_workspace(
        self, name: str, question: str | None = None
    ) -> dict[str, Any]:
        payload = {"name": name}
        if question is not None:
            payload["question"] = question
        result = self._request("POST", "/api/workspaces", payload)
        workspace = result.get("workspace") or {}
        if workspace.get("id"):
            self._workspace_id = str(workspace["id"])
        return result

    def open_workspace(self, workspace_id: str) -> dict[str, Any]:
        result = self._request("POST", "/api/workspaces/open", {"id": workspace_id})
        self._workspace_id = workspace_id
        return result

    def post(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if action not in ACTIONS:
            raise CliError(f"unknown action {action!r}")
        routed = dict(payload or {})
        routed["workspace_id"] = str(
            routed.get("workspace_id") or self.bind_current_workspace()
        )
        return self._request("POST", f"/api/{action}", routed)


def _live_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("cards") or {}
    cards = list(raw.values()) if isinstance(raw, dict) else list(raw)
    return [
        card
        for card in cards
        if isinstance(card, dict)
        and not card.get("archived")
        and card.get("state") != "archived"
    ]


# §1.6 — a folded instance is hidden at the frame's altitude, NOT gone:
# "fully present on descent. Never archived." Treating `folded` as absent
# would make a fold look like the deleted fuse()'s consumption, and would
# also empty out the floor a frame's support is computed from. `vacated` and
# `retired` rows are history: §1.2 keeps them forever, but they do not stand.
_STANDING_STATUSES = ("live", "folded")


def _live_positions(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Positions still standing — §1.2 keeps vacated/retired rows forever."""
    raw = state.get("positions") or []
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") in _STANDING_STATUSES
    ]


def _altitudes(positions: list[dict[str, Any]]) -> dict[str, int]:
    """§1.2/§1.5 — altitude is DERIVED here too, never read off the wire.

    ``Position.to_dict()`` deliberately ships no ``altitude`` key, so a
    scenario metric cannot accidentally assert a stored figure that has
    drifted from the floor. The walk is the same one the engine memoizes:
    a claim is 0, a frame is ``1 + max(floor)``. Nothing at any layer may
    drift from the layer below, and that includes this layer.
    """
    by_id = {str(row.get("id")): row for row in positions if row.get("id")}
    memo: dict[str, int] = {}

    def walk(pid: str, seen: frozenset[str]) -> int:
        if pid in memo:
            return memo[pid]
        row = by_id.get(pid)
        if row is None or row.get("floor_kind") != "frame":
            memo[pid] = 0
            return 0
        supports = [str(s) for s in (row.get("supports") or [])]
        if not supports or pid in seen:
            memo[pid] = 0
            return 0
        below = seen | {pid}
        height = 1 + max((walk(sid, below) for sid in supports), default=-1)
        memo[pid] = height
        return height

    return {pid: walk(pid, frozenset()) for pid in by_id}


def state_value(state: dict[str, Any], path: str) -> Any:
    """Resolve stable scenario metrics and simple dotted state paths."""

    cards = _live_cards(state)
    positions = _live_positions(state)
    altitudes = _altitudes(positions)
    candidates = [
        row for row in (state.get("click_candidates") or []) if isinstance(row, dict)
    ]
    attempts = [
        row for row in (state.get("click_attempts") or []) if isinstance(row, dict)
    ]
    metrics = {
        "cards.count": len(cards),
        "cards.testing": sum(card.get("state") == "testing" for card in cards),
        "cards.settled": sum(
            card.get("state") in ("supported", "refuted") for card in cards
        ),
        "cards.open": sum(
            card.get("state") in ("open", "needs_human") for card in cards
        ),
        # §1.2 — the ladder is the tracked object, so the scenario vocabulary
        # counts positions, not just their current occupants.
        "positions.count": len(positions),
        "positions.claims": sum(
            row.get("floor_kind") == "claim" for row in positions
        ),
        "positions.frames": sum(
            row.get("floor_kind") == "frame" for row in positions
        ),
        "positions.folded": sum(bool(row.get("folded_under")) for row in positions),
        "positions.external": sum(bool(row.get("external")) for row in positions),
        "positions.grounded": sum(
            row.get("last_grounded_at") is not None for row in positions
        ),
        # `floors` is the height of the ladder: one more than the top
        # altitude, so a field of bare claims has exactly one floor.
        "floors.count": (max(altitudes.values()) + 1) if altitudes else 0,
        "floors.max_altitude": max(altitudes.values()) if altitudes else 0,
        # §2.4 — the emergence inbox, capped at 3 open per workspace.
        "inbox.count": sum(row.get("status") == "open" for row in candidates),
        "inbox.accepted": sum(row.get("status") == "accepted" for row in candidates),
        "inbox.declined": sum(row.get("status") == "declined" for row in candidates),
        # §2.3 — the never-retry ledger. `attempts.consumed` excludes the
        # non-consuming `failed` row a provider outage writes.
        "attempts.count": len(attempts),
        "attempts.consumed": sum(
            row.get("outcome") not in ("failed", "reconsidered") for row in attempts
        ),
        "clicks.confirmed": int(state.get("clicks_confirmed") or 0),
        "contributions.count": int(state.get("human_contributions") or 0),
        "ledger.count": len(state.get("ledger") or []),
        "providers.count": len(state.get("providers") or []),
    }
    if path in metrics:
        return metrics[path]

    value: Any = state
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise CliError(f"state path {path!r} does not exist")
        value = value[part]
    return value


def _resolve_card_refs(value: Any, client: Client) -> Any:
    """Resolve ``@card:N`` and ``@position:N`` to the Nth live id.

    ``@frame:N`` indexes only frame positions, which is what the ladder
    operations (`derive`, `descend`, `unfold`) actually take.
    """

    if isinstance(value, str) and value.startswith("@card:"):
        try:
            index = int(value.split(":", 1)[1])
            card = _live_cards(client.get_state())[index]
            return card["id"]
        except (ValueError, IndexError, KeyError) as exc:
            raise CliError(f"cannot resolve card reference {value!r}") from exc
    if isinstance(value, str) and value.startswith(("@position:", "@frame:")):
        prefix, _, raw_index = value.partition(":")
        try:
            index = int(raw_index)
            rows = _live_positions(client.get_state())
            if prefix == "@frame":
                rows = [row for row in rows if row.get("floor_kind") == "frame"]
            return rows[index]["id"]
        except (ValueError, IndexError, KeyError) as exc:
            raise CliError(f"cannot resolve position reference {value!r}") from exc
    if isinstance(value, dict):
        return {key: _resolve_card_refs(item, client) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_card_refs(item, client) for item in value]
    return value


def _condition(document: Any) -> tuple[str, Any]:
    if not isinstance(document, dict) or set(document) != {"path", "equals"}:
        raise CliError("condition must contain exactly 'path' and 'equals'")
    path = document["path"]
    if not isinstance(path, str) or not path:
        raise CliError("condition path must be a non-empty string")
    return path, document["equals"]


def run_scenario(
    client: Client, document: dict[str, Any], output: TextIO | None = None
) -> dict[str, Any]:
    """Execute a declarative scenario and return its final public state."""

    steps = document.get("steps")
    if not isinstance(steps, list):
        raise CliError("scenario must contain a 'steps' array")
    stream = output
    # Real HTTP clients pin one workspace for the entire scenario. Lightweight
    # protocol fakes used by callers/tests may not expose the binding method.
    bind = getattr(client, "bind_current_workspace", None)
    if callable(bind):
        bind()

    for number, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise CliError(f"scenario step {number} must be an object")

        if "action" in step:
            action = step["action"]
            if action not in ACTIONS:
                raise CliError(f"scenario step {number}: unknown action {action!r}")
            payload = {
                key: value
                for key, value in step.items()
                if key not in ("action", "note")
            }
            payload = _resolve_card_refs(payload, client)
            result = client.post(action, payload)
            if stream is not None:
                _write_json(stream, {"step": number, "action": action, "result": result})
            continue

        if "wait" in step:
            path, expected = _condition(step["wait"])
            timeout = float(step.get("timeout", 10.0))
            interval = float(step.get("interval", 0.1))
            deadline = time.monotonic() + timeout
            while True:
                state = client.get_state()
                actual = state_value(state, path)
                if actual == expected:
                    break
                if time.monotonic() >= deadline:
                    raise CliError(
                        f"scenario step {number} timed out: "
                        f"{path} was {actual!r}, expected {expected!r}"
                    )
                time.sleep(max(0.01, interval))
            if stream is not None:
                _write_json(
                    stream,
                    {"step": number, "wait": path, "equals": expected, "matched": True},
                )
            continue

        if "assert" in step:
            path, expected = _condition(step["assert"])
            actual = state_value(client.get_state(), path)
            if actual != expected:
                raise CliError(
                    f"scenario step {number} failed: "
                    f"{path} was {actual!r}, expected {expected!r}"
                )
            if stream is not None:
                _write_json(
                    stream,
                    {"step": number, "assert": path, "equals": expected, "passed": True},
                )
            continue

        if step.get("state") is True:
            if stream is not None:
                _write_json(stream, {"step": number, "state": client.get_state()})
            continue

        raise CliError(
            f"scenario step {number} must contain action, wait, assert, or state"
        )

    return client.get_state()


def load_scenario(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read scenario {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"scenario {path} is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CliError("scenario root must be a JSON object")
    return document


def positions_view(
    state: dict[str, Any],
    *,
    altitude: int | None = None,
    include_folded: bool = False,
) -> dict[str, Any]:
    """§1.2/§3.1 — the ladder, projected from state, nothing stored read back.

    Altitude and a frame's support summary are both recomputed here from the
    floor (§1.5). A frame's support is a tally of its floor's states — this
    view will never print a support figure the floor does not currently
    justify, because it has no other place to get one from.
    """
    rows = _live_positions(state)
    altitudes = _altitudes(rows)
    by_id = {str(row.get("id")): row for row in rows if row.get("id")}

    def support_summary(row: dict[str, Any]) -> dict[str, int]:
        tally = {"supported": 0, "refuted": 0, "open": 0}
        for sid in row.get("supports") or []:
            child = by_id.get(str(sid))
            if child is None:
                continue
            state_name = ((child.get("occupant") or {}).get("state")) or "open"
            if state_name in tally:
                tally[state_name] += 1
            else:
                tally["open"] += 1
        return tally

    out = []
    for row in rows:
        pid = str(row.get("id"))
        if not include_folded and row.get("folded_under"):
            continue
        height = altitudes.get(pid, 0)
        if altitude is not None and height != altitude:
            continue
        occupant = row.get("occupant") or {}
        entry: dict[str, Any] = {
            "id": pid,
            "altitude": height,
            "floor_kind": row.get("floor_kind"),
            "origin": row.get("origin"),
            "text": occupant.get("text", ""),
            "artifact_type": occupant.get("artifact_type"),
            "supports": list(row.get("supports") or []),
            "folded_under": row.get("folded_under"),
            "last_grounded_at": row.get("last_grounded_at"),
            "external": bool(row.get("external")),
        }
        if row.get("floor_kind") == "frame":
            # A frame is never directly supported or refuted (§1.1); it has
            # no receipt to report, only the floor's tally.
            entry["support"] = support_summary(row)
        else:
            entry["support_state"] = occupant.get("state")
            entry["receipt"] = occupant.get("receipt")
        out.append(entry)

    out.sort(key=lambda entry: (-entry["altitude"], entry["id"]))
    return {
        "altitude": altitude,
        "max_altitude": max(altitudes.values()) if altitudes else 0,
        "floors": (max(altitudes.values()) + 1) if altitudes else 0,
        "positions": out,
    }


def _write_json(stream: TextIO, value: Any) -> None:
    json.dump(value, stream, indent=2, sort_keys=True, default=str)
    stream.write("\n")
    stream.flush()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(client: Client, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CliError(f"local Magpie server exited with code {process.returncode}")
        try:
            client.get_state()
            return
        except CliError as exc:
            last_error = str(exc)
            time.sleep(0.05)
    raise CliError(f"local Magpie server did not become ready: {last_error}")


def run_isolated(
    document: dict[str, Any],
    *,
    port: int | None = None,
    runtime_dir: str | None = None,
    providers: str = "stub",
    timeout: float = 10.0,
    output: TextIO | None = None,
) -> dict[str, Any]:
    """Start a disposable server, run a scenario, and always stop the server."""

    chosen_port = int(port or _free_port())
    temporary = None
    if runtime_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="magpie-cli-")
        runtime_dir = temporary.name
    Path(runtime_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MAGPIE_RUNTIME_DIR"] = runtime_dir
    env["MAGPIE_PROVIDERS"] = providers
    env["MAGPIE_LLM_STUB"] = "1" if providers.strip().lower() == "stub" else "0"
    command = [
        sys.executable,
        "-m",
        "magpie.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(chosen_port),
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )
    client = Client(f"http://127.0.0.1:{chosen_port}", timeout=min(timeout, 5.0))
    try:
        _wait_ready(client, process, timeout)
        final_state = run_scenario(client, document, output=output)
        return {
            "runtime_dir": runtime_dir if temporary is None else None,
            "providers": providers,
            "state": final_state,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        if temporary is not None:
            temporary.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magpie",
        description="Drive a Magpie server through its public HTTP API.",
    )
    parser.add_argument(
        "--url", default=os.environ.get("MAGPIE_URL", DEFAULT_URL), help="server URL"
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout")
    sub = parser.add_subparsers(dest="command", required=True)

    workspace = sub.add_parser("workspace", help="manage persistent workspaces")
    workspace_sub = workspace.add_subparsers(
        dest="workspace_command", required=True
    )
    workspace_sub.add_parser("list", help="list workspaces")
    workspace_sub.add_parser("current", help="print the current workspace")
    workspace_create = workspace_sub.add_parser("create", help="create a workspace")
    workspace_create.add_argument("name")
    workspace_create.add_argument("--question")
    workspace_open = workspace_sub.add_parser("open", help="open a workspace")
    workspace_open.add_argument("id")

    sub.add_parser("state", help="print the public server state")
    seed = sub.add_parser("seed", help="set the field question")
    seed.add_argument("question")
    propose = sub.add_parser("propose", help="queue a thought for atomization")
    propose.add_argument("text")
    propose_click = sub.add_parser(
        "propose-click", help="ask whether two positions are one frame"
    )
    propose_click.add_argument("a")
    propose_click.add_argument("b")
    for inbox_name, inbox_help in (
        ("pending-clicks", "list open emergence-inbox candidates"),
        ("inbox", "the emergence inbox (alias of pending-clicks)"),
    ):
        pending = sub.add_parser(inbox_name, help=inbox_help)
        pending.add_argument(
            "--include-rejected",
            action="store_true",
            help="also show near-misses: which gate each rejected pair failed",
        )
    resolve_click = sub.add_parser(
        "resolve-click", help="accept or decline one inbox candidate"
    )
    resolve_click.add_argument("candidate_id")
    resolve_click.add_argument("verdict", choices=("accept", "decline"))
    resolve_click.add_argument("--confirmed-by")
    resolve_click.add_argument("--text")
    confirm_click = sub.add_parser(
        "confirm-click", help="accept one candidate and execute the fold"
    )
    confirm_click.add_argument("candidate_id")
    # Not optional: a click is confirmed by a human or not at all (§1.6), and
    # the server rejects an acceptance without it either way.
    confirm_click.add_argument("confirmed_by")
    confirm_click.add_argument(
        "--text", help="accept with edit: this wording becomes the occupant"
    )
    decline_click = sub.add_parser(
        "decline-click", help="decline one candidate, writing a `declined` row"
    )
    decline_click.add_argument("candidate_id")
    reconsider = sub.add_parser(
        "reconsider-pair", help="deliberately reopen a settled non-click"
    )
    reconsider.add_argument("a")
    reconsider.add_argument("b")
    derive = sub.add_parser("derive", help="propose grounding claims for a frame")
    derive.add_argument("frame_id")
    field = sub.add_parser("field", help="read one altitude of the ladder")
    field.add_argument("--altitude", type=int)
    positions = sub.add_parser(
        "positions", help="the whole ladder as durable positions, by altitude"
    )
    positions.add_argument("--altitude", type=int, help="only this floor")
    positions.add_argument(
        "--include-folded",
        action="store_true",
        help="show instances hidden under a frame (§1.6)",
    )
    descend = sub.add_parser("descend", help="read the floor beneath a frame")
    descend.add_argument("position_id")
    unfold = sub.add_parser("unfold", help="release a frame's instances")
    unfold.add_argument("frame_id")
    verify = sub.add_parser("verify", help="submit a card to the verification hook")
    verify.add_argument("id")
    judge = sub.add_parser("judge", help="record a human judgment")
    judge.add_argument("id")
    judge.add_argument("verdict", choices=("yes", "no", "unknown"))
    for name, help_text in (
        ("keep", "toggle a card's kept state"),
        ("kill", "archive a card"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("id")
    move = sub.add_parser("move", help="move a card to a section")
    move.add_argument("id")
    move.add_argument("section")
    section = sub.add_parser("section", help="create a section")
    section.add_argument("name")
    section.add_argument("--key")
    section.add_argument("--color", default="#c9b8a0")
    harvest = sub.add_parser("harvest", help="write and print a harvest brief")
    harvest.add_argument("--altitude", type=int)
    harvest.add_argument("--max-items", type=int, default=12)
    recall = sub.add_parser("recall", help="recall Raven memories through Magpie")
    recall.add_argument("query", nargs="?")
    recall.add_argument("--workspace-id")
    recall.add_argument("--limit", type=int, default=10)
    adopt = sub.add_parser("adopt-memory", help="bring a Raven memory into the field")
    adopt.add_argument("memory_id")
    adopt.add_argument("--workspace-id")
    adopt.add_argument("--section")
    dismiss = sub.add_parser(
        "dismiss-memory", help="hide a Raven suggestion in this workspace"
    )
    dismiss.add_argument("memory_id")
    dismiss.add_argument("--workspace-id")

    call = sub.add_parser("call", help="call an API action with a JSON object")
    call.add_argument("action", choices=ACTIONS)
    call.add_argument("--data", default="{}")

    scenario = sub.add_parser("scenario", help="run a scenario against --url")
    scenario.add_argument("path")
    run = sub.add_parser("run", help="run a scenario on an isolated local server")
    run.add_argument("path")
    run.add_argument("--port", type=int)
    run.add_argument("--runtime-dir")
    run.add_argument("--providers", default="stub")
    return parser


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "seed":
        return {"question": args.question}
    if args.command == "propose":
        return {"text": args.text}
    if args.command in ("propose-click", "reconsider-pair"):
        return {"a": args.a, "b": args.b}
    if args.command in ("pending-clicks", "inbox"):
        return {"include_rejected": bool(args.include_rejected)}
    if args.command == "resolve-click":
        payload = {"candidate_id": args.candidate_id, "verdict": args.verdict}
        if args.confirmed_by:
            payload["confirmed_by"] = args.confirmed_by
        if args.text:
            payload["text"] = args.text
        return payload
    if args.command == "confirm-click":
        payload = {
            "candidate_id": args.candidate_id,
            "verdict": "accept",
            "confirmed_by": args.confirmed_by,
        }
        if args.text:
            payload["text"] = args.text
        return payload
    if args.command == "decline-click":
        return {"candidate_id": args.candidate_id, "verdict": "decline"}
    if args.command in ("derive", "unfold"):
        return {"frame_id": args.frame_id}
    if args.command == "descend":
        return {"position_id": args.position_id}
    if args.command == "field":
        return {"altitude": args.altitude}
    if args.command == "harvest":
        return {"altitude": args.altitude, "max_items": args.max_items}
    if args.command in ("verify", "keep", "kill"):
        return {"id": args.id}
    if args.command == "judge":
        return {"id": args.id, "verdict": args.verdict}
    if args.command == "move":
        return {"id": args.id, "section": args.section}
    if args.command == "section":
        payload = {"name": args.name, "color": args.color}
        if args.key:
            payload["key"] = args.key
        return payload
    if args.command == "recall":
        return {
            "query": args.query or "",
            "workspace_id": args.workspace_id,
            "limit": args.limit,
        }
    if args.command == "adopt-memory":
        return {
            "memory_id": args.memory_id,
            "workspace_id": args.workspace_id,
            "section": args.section,
        }
    if args.command == "dismiss-memory":
        return {
            "memory_id": args.memory_id,
            "workspace_id": args.workspace_id,
        }
    return {}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_isolated(
                load_scenario(args.path),
                port=args.port,
                runtime_dir=args.runtime_dir,
                providers=args.providers,
                timeout=args.timeout,
                output=sys.stderr,
            )
        else:
            client = Client(args.url, timeout=args.timeout)
            if args.command == "state":
                result = client.get_state()
            elif args.command == "positions":
                result = positions_view(
                    client.get_state(),
                    altitude=args.altitude,
                    include_folded=args.include_folded,
                )
            elif args.command == "workspace":
                if args.workspace_command == "list":
                    result = client.list_workspaces()
                elif args.workspace_command == "current":
                    result = client.get_current_workspace()
                elif args.workspace_command == "create":
                    result = client.create_workspace(args.name, args.question)
                else:
                    result = client.open_workspace(args.id)
            elif args.command == "scenario":
                result = run_scenario(
                    client, load_scenario(args.path), output=sys.stderr
                )
            elif args.command == "call":
                try:
                    payload = json.loads(args.data)
                except json.JSONDecodeError as exc:
                    raise CliError(f"--data is invalid JSON: {exc}") from exc
                if not isinstance(payload, dict):
                    raise CliError("--data must be a JSON object")
                result = client.post(args.action, payload)
            elif args.command in _COMMAND_ACTIONS:
                result = client.post(_COMMAND_ACTIONS[args.command], _payload(args))
            else:
                result = client.post(args.command, _payload(args))
        _write_json(sys.stdout, result)
        return 0
    except CliError as exc:
        print(f"magpie: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
