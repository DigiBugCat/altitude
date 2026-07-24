"""Command-line control plane for a running Magpie HTTP server.

The CLI deliberately talks to the public JSON API instead of importing the
engine.  A CLI-driven scenario therefore exercises the same boundary as the
browser and can also target a remote or deployed Magpie instance.

Examples:

    python3 -m magpie.cli state
    python3 -m magpie.cli seed "What would make this idea work?"
    python3 -m magpie.cli propose "The smallest useful loop is deterministic."
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
    "collide",
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


def state_value(state: dict[str, Any], path: str) -> Any:
    """Resolve stable scenario metrics and simple dotted state paths."""

    cards = _live_cards(state)
    metrics = {
        "cards.count": len(cards),
        "cards.testing": sum(card.get("state") == "testing" for card in cards),
        "cards.settled": sum(
            card.get("state") in ("supported", "refuted") for card in cards
        ),
        "cards.open": sum(
            card.get("state") in ("open", "needs_human") for card in cards
        ),
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
    """Resolve ``@card:N`` values to the Nth live card id."""

    if isinstance(value, str) and value.startswith("@card:"):
        try:
            index = int(value.split(":", 1)[1])
            card = _live_cards(client.get_state())[index]
            return card["id"]
        except (ValueError, IndexError, KeyError) as exc:
            raise CliError(f"cannot resolve card reference {value!r}") from exc
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
    collide = sub.add_parser("collide", help="collide two card ids")
    collide.add_argument("a")
    collide.add_argument("b")
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
    sub.add_parser("harvest", help="write and print a harvest")
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
    if args.command == "collide":
        return {"a": args.a, "b": args.b}
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
            elif args.command == "adopt-memory":
                result = client.post("recall/adopt", _payload(args))
            elif args.command == "dismiss-memory":
                result = client.post("recall/dismiss", _payload(args))
            else:
                result = client.post(args.command, _payload(args))
        _write_json(sys.stdout, result)
        return 0
    except CliError as exc:
        print(f"magpie: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
