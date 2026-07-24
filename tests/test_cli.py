"""Contract tests for Magpie's local HTTP CLI and scenario runner.

The client transport is replaced at the urllib boundary, and scenarios use a
small recording client.  Nothing in this module opens a socket or touches a
real Magpie runtime.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from magpie import cli


class FakeResponse(io.BytesIO):
    """The subset of an urllib response that ``Client`` needs."""

    def __init__(self, payload, status=200):
        super().__init__(json.dumps(payload).encode("utf-8"))
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def install_transport(monkeypatch, payload):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_get_state_uses_the_state_endpoint_and_timeout(monkeypatch):
    seen = install_transport(
        monkeypatch,
        {"ok": True, "question": "What changed?", "cards": []},
    )

    result = cli.Client("http://127.0.0.1:7351/", timeout=2.5).get_state()

    request = seen["request"]
    assert request.full_url == "http://127.0.0.1:7351/api/state"
    assert request.get_method() == "GET"
    assert seen["timeout"] == 2.5
    assert result["question"] == "What changed?"


def test_post_builds_action_path_and_json_body(monkeypatch):
    seen = install_transport(
        monkeypatch,
        {"ok": True, "question": "Does the loop work?"},
    )

    result = cli.Client("http://magpie.test").post(
        "seed", {
            "workspace_id": "ws-test",
            "question": "Does the loop work?",
        }
    )

    request = seen["request"]
    assert request.full_url == "http://magpie.test/api/seed"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "workspace_id": "ws-test",
        "question": "Does the loop work?",
    }
    assert result["question"] == "Does the loop work?"


@pytest.mark.parametrize(
    ("method_name", "path"),
    [
        ("list_workspaces", "/api/workspaces"),
        ("get_current_workspace", "/api/workspaces/current"),
    ],
)
def test_workspace_get_methods_use_the_workspace_endpoints(
    monkeypatch, method_name, path
):
    seen = install_transport(monkeypatch, {"ok": True, "workspaces": []})

    result = getattr(cli.Client("http://magpie.test"), method_name)()

    request = seen["request"]
    assert request.full_url == f"http://magpie.test{path}"
    assert request.get_method() == "GET"
    assert result == {"workspaces": []}


@pytest.mark.parametrize(
    ("method_name", "args", "path", "payload"),
    [
        (
            "create_workspace",
            ("Checkout", "Why did conversion fall?"),
            "/api/workspaces",
            {"name": "Checkout", "question": "Why did conversion fall?"},
        ),
        (
            "create_workspace",
            ("Scratch",),
            "/api/workspaces",
            {"name": "Scratch"},
        ),
        (
            "open_workspace",
            ("ws-123",),
            "/api/workspaces/open",
            {"id": "ws-123"},
        ),
    ],
)
def test_workspace_post_methods_use_the_workspace_endpoints(
    monkeypatch, method_name, args, path, payload
):
    seen = install_transport(
        monkeypatch, {"ok": True, "workspace": {"id": "ws-123"}}
    )

    getattr(cli.Client("http://magpie.test"), method_name)(*args)

    request = seen["request"]
    assert request.full_url == f"http://magpie.test{path}"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == payload


def test_client_surfaces_an_api_error_message(monkeypatch):
    body = io.BytesIO(json.dumps({"ok": False, "error": "question required"}).encode())

    def reject(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {"Content-Type": "application/json"},
            body,
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", reject)

    with pytest.raises(Exception, match="question required"):
        cli.Client("http://magpie.test").post("seed", {})


class RecordingClient:
    def __init__(self, states=None):
        self.states = list(states or [{"question": "", "cards": []}])
        self.posts = []
        self.state_calls = 0

    def post(self, action, payload):
        self.posts.append((action, payload))
        return {"ok": True, "action": action}

    def get_state(self):
        index = min(self.state_calls, len(self.states) - 1)
        self.state_calls += 1
        return self.states[index]


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("seed", {"question": "q"}),
        ("propose", {"text": "claim"}),
        # §2.1 — `collide` is deleted outright, not feature-flagged. The
        # human-initiated recognition request is `click/propose`, and it can
        # only reach the emergence inbox.
        ("click/propose", {"a": "c1", "b": "c2"}),
        ("click/pending", {"include_rejected": True}),
        ("click/resolve", {"candidate_id": "cand1", "verdict": "decline"}),
        ("click/reconsider", {"a": "c1", "b": "c2"}),
        ("derive", {"frame_id": "c3"}),
        ("field", {"altitude": 1}),
        ("descend", {"position_id": "c3"}),
        ("unfold", {"frame_id": "c3"}),
        ("judge", {"id": "c1", "verdict": "yes"}),
        ("verify", {"id": "c1"}),
        ("keep", {"id": "c1"}),
        ("kill", {"id": "c1"}),
        ("move", {"id": "c1", "section": "evidence"}),
        ("section", {"name": "Evidence", "color": "#abc"}),
        ("harvest", {}),
    ],
)
def test_scenario_action_posts_only_the_payload(action, payload):
    client = RecordingClient()

    cli.run_scenario(client, {"steps": [{"action": action, **payload}]})

    assert client.posts == [(action, payload)]


def test_scenario_orchestrates_actions_wait_assert_and_state(monkeypatch):
    client = RecordingClient(
        [
            {"question": "q", "cards": []},
            {"question": "q", "cards": [{"id": "c1", "state": "testing"}]},
            {"question": "q", "cards": [{"id": "c1", "state": "testing"}]},
        ]
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli.run_scenario(
        client,
        {
            "steps": [
                {"action": "seed", "question": "q"},
                {"action": "propose", "text": "claim"},
                {"wait": {"path": "cards.testing", "equals": 1}, "timeout": 1},
                {"assert": {"path": "question", "equals": "q"}},
                {"state": True},
            ]
        },
    )

    assert client.posts == [
        ("seed", {"question": "q"}),
        ("propose", {"text": "claim"}),
    ]
    assert client.state_calls >= 3


def test_wait_supports_count_and_settled_paths(monkeypatch):
    client = RecordingClient(
        [
            {"cards": []},
            {"cards": [{"state": "supported"}, {"state": "refuted"}]},
            {"cards": [{"state": "supported"}, {"state": "refuted"}]},
        ]
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli.run_scenario(
        client,
        {
            "steps": [
                {"wait": {"path": "cards.count", "equals": 2}, "timeout": 1},
                {"assert": {"path": "cards.settled", "equals": 2}},
            ]
        },
    )


def test_wait_timeout_is_an_error(monkeypatch):
    client = RecordingClient([{"cards": []}])
    ticks = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(ticks, 1.0))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(Exception, match="(?i)timed? ?out|timeout"):
        cli.run_scenario(
            client,
            {
                "steps": [
                    {
                        "wait": {"path": "cards.count", "equals": 1},
                        "timeout": 0.5,
                    }
                ]
            },
        )


def test_assert_mismatch_is_an_error():
    client = RecordingClient([{"question": "actual", "cards": []}])

    with pytest.raises(Exception, match="question|expected"):
        cli.run_scenario(
            client,
            {
                "steps": [
                    {"assert": {"path": "question", "equals": "expected"}},
                ]
            },
        )


# ---------------------------------------------------------------- the ladder
#
# SPEC §1.2/§1.5. These exercise the CLI's projection of the ladder, which is
# the only place the CLI is allowed to compute anything: altitude and frame
# support are DERIVED here, exactly as they are engine-side, because
# `Position.to_dict()` deliberately ships neither. A scenario metric that read
# a stored figure off the wire would be the drift §1.5 forbids.


def _position(
    pid,
    floor_kind="claim",
    supports=(),
    status="live",
    folded_under=None,
    state="open",
    receipt=None,
    origin="human",
    external=False,
    last_grounded_at=None,
):
    return {
        "id": pid,
        "floor_kind": floor_kind,
        "supports": list(supports),
        "status": status,
        "folded_under": folded_under,
        "origin": origin,
        "external": external,
        "last_grounded_at": last_grounded_at,
        "occupant": {
            "id": pid,
            "text": f"text of {pid}",
            "state": state,
            "receipt": receipt,
            "artifact_type": "claim",
        },
    }


def _ladder_state():
    """One confirmed fold: a frame over two folded instances, plus cargo.

    `c1` carries a receipt, `c2` does not — so the frame's floor reads
    `1 supported / 0 refuted / 1 open` and nothing anywhere stores that.
    """
    return {
        "cards": [],
        "positions": [
            _position(
                "c1",
                folded_under="f1",
                status="folded",
                state="supported",
                receipt="log shows the duplicate charge",
                last_grounded_at=1000.0,
            ),
            _position("c2", folded_under="f1", status="folded"),
            _position("c3"),
            _position("c4", origin="recall", external=True, state="needs_human"),
            _position("f1", floor_kind="frame", supports=("c1", "c2")),
            # History: §1.2 keeps vacated rows forever, but they do not stand.
            _position("f0", floor_kind="frame", supports=("c3",), status="vacated"),
            _position("c9", status="retired"),
        ],
        "click_candidates": [
            {"id": "k1", "status": "open"},
            {"id": "k2", "status": "open"},
            {"id": "k3", "status": "accepted"},
            {"id": "k4", "status": "declined"},
        ],
        "click_attempts": [
            {"position_a": "c1", "position_b": "c2", "outcome": "clicked"},
            {"position_a": "c1", "position_b": "c3", "outcome": "no_click"},
            {"position_a": "c2", "position_b": "c3", "outcome": "gate_failed"},
            # §2.2 — a provider outage does NOT consume the pair.
            {"position_a": "c3", "position_b": "c4", "outcome": "failed"},
        ],
        "clicks_confirmed": 1,
        "human_contributions": 7,
    }


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # `c9` is retired and `f0` vacated: history, not standing structure.
        ("positions.count", 5),
        ("positions.claims", 4),
        ("positions.frames", 1),
        # §1.6 — a folded instance is hidden, never consumed. The count is
        # what makes the difference from the deleted fuse() observable.
        ("positions.folded", 2),
        ("positions.external", 1),
        ("positions.grounded", 1),
        ("floors.count", 2),
        ("floors.max_altitude", 1),
        ("inbox.count", 2),
        ("inbox.accepted", 1),
        ("inbox.declined", 1),
        ("attempts.count", 4),
        # `failed` is excluded: it is the one outcome that leaves the pair
        # available, so counting it as consumed would misreport the ledger.
        ("attempts.consumed", 3),
        ("clicks.confirmed", 1),
        ("contributions.count", 7),
    ],
)
def test_scenario_metrics_read_the_ladder(path, expected):
    assert cli.state_value(_ladder_state(), path) == expected


def test_altitude_is_derived_from_supports_not_read_off_the_wire():
    state = _ladder_state()
    # Nothing in the snapshot says how high anything is; the wire format
    # carries no `altitude` key at all (§1.2).
    assert all("altitude" not in row for row in state["positions"])

    view = cli.positions_view(state, include_folded=True)
    heights = {entry["id"]: entry["altitude"] for entry in view["positions"]}

    assert heights == {"c1": 0, "c2": 0, "c3": 0, "c4": 0, "f1": 1}


def test_a_taller_ladder_stacks_frame_on_frame():
    state = {
        "cards": [],
        "positions": [
            _position("c1"),
            _position("c2"),
            _position("f1", floor_kind="frame", supports=("c1",)),
            _position("f2", floor_kind="frame", supports=("f1", "c2")),
        ],
    }

    assert cli.state_value(state, "floors.max_altitude") == 2
    assert cli.state_value(state, "floors.count") == 3


def test_frame_support_is_computed_from_the_floor_and_carries_no_receipt():
    view = cli.positions_view(_ladder_state())
    frame = next(e for e in view["positions"] if e["floor_kind"] == "frame")

    # §1.5 — the tally is the multiset of the floor's states, recomputed.
    assert frame["support"] == {"supported": 1, "refuted": 0, "open": 1}
    # §1.1 — a frame is never directly supported or refuted, so the view has
    # no receipt field to print for one. It cannot report what cannot exist.
    assert "receipt" not in frame
    assert "support_state" not in frame


def test_positions_view_hides_folded_instances_until_asked():
    hidden = cli.positions_view(_ladder_state())
    shown = cli.positions_view(_ladder_state(), include_folded=True)

    assert [e["id"] for e in hidden["positions"]] == ["f1", "c3", "c4"]
    # §1.6 — "hidden at the frame's altitude, fully present on descent."
    # Present, and still pointing at the frame that folded them.
    assert [e["id"] for e in shown["positions"]] == ["f1", "c1", "c2", "c3", "c4"]
    assert [e["folded_under"] for e in shown["positions"] if e["id"] in ("c1", "c2")] == [
        "f1",
        "f1",
    ]


def test_positions_view_filters_to_one_altitude():
    view = cli.positions_view(_ladder_state(), altitude=1)

    assert [entry["id"] for entry in view["positions"]] == ["f1"]
    # The ceiling still reports the whole ladder, so a slider knows its range.
    assert view["max_altitude"] == 1


def test_vacated_and_retired_positions_are_history_not_structure():
    view = cli.positions_view(_ladder_state(), include_folded=True)
    ids = {entry["id"] for entry in view["positions"]}

    assert "f0" not in ids and "c9" not in ids


@pytest.mark.parametrize("reference", ["@position:1", "@frame:0"])
def test_position_references_resolve_against_standing_positions(reference):
    client = RecordingClient([_ladder_state(), _ladder_state()])

    cli.run_scenario(
        client, {"steps": [{"action": "descend", "position_id": reference}]}
    )

    # `@position:N` indexes standing positions in snapshot order (c1, c2, c3,
    # c4, f1); `@frame:N` indexes only the frames among them.
    expected = "c2" if reference == "@position:1" else "f1"
    assert client.posts == [("descend", {"position_id": expected})]


def test_main_prints_state_as_json_and_returns_zero(monkeypatch, capsys):
    seen = {}

    class MainClient:
        def __init__(self, base_url, timeout):
            seen["base_url"] = base_url
            seen["timeout"] = timeout

        def get_state(self):
            return {"question": "local loop", "cards": []}

    monkeypatch.setattr(cli, "Client", MainClient)

    status = cli.main(
        ["--url", "http://magpie.test/", "--timeout", "1.25", "state"]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert json.loads(captured.out) == {"question": "local loop", "cards": []}
    assert captured.err == ""
    assert seen == {"base_url": "http://magpie.test/", "timeout": 1.25}


@pytest.mark.parametrize(
    ("argv", "method", "expected_args"),
    [
        (["workspace", "list"], "list_workspaces", ()),
        (["workspace", "current"], "get_current_workspace", ()),
        (
            [
                "workspace",
                "create",
                "Checkout",
                "--question",
                "Why did conversion fall?",
            ],
            "create_workspace",
            ("Checkout", "Why did conversion fall?"),
        ),
        (["workspace", "create", "Scratch"], "create_workspace", ("Scratch", None)),
        (["workspace", "open", "ws-123"], "open_workspace", ("ws-123",)),
    ],
)
def test_main_routes_nested_workspace_commands(
    monkeypatch, capsys, argv, method, expected_args
):
    calls = []

    class MainClient:
        def __init__(self, base_url, timeout):
            pass

        def __getattr__(self, name):
            assert name == method

            def call(*args):
                calls.append((name, args))
                return {"workspace": {"id": "ws-123"}}

            return call

    monkeypatch.setattr(cli, "Client", MainClient)

    status = cli.main(argv)

    captured = capsys.readouterr()
    assert status == 0
    assert json.loads(captured.out) == {"workspace": {"id": "ws-123"}}
    assert captured.err == ""
    assert calls == [(method, expected_args)]


@pytest.mark.parametrize(
    ("argv", "action", "payload"),
    [
        (
            ["propose-click", "c1", "c2"],
            "click/propose",
            {"a": "c1", "b": "c2"},
        ),
        (["inbox"], "click/pending", {"include_rejected": False}),
        (
            ["inbox", "--include-rejected"],
            "click/pending",
            {"include_rejected": True},
        ),
        (
            ["pending-clicks", "--include-rejected"],
            "click/pending",
            {"include_rejected": True},
        ),
        # §1.6 — `confirm-click` is one verdict on `resolve-click`, and it
        # cannot be spelled without an attributed confirmer.
        (
            ["confirm-click", "k1", "andrew"],
            "click/resolve",
            {"candidate_id": "k1", "verdict": "accept", "confirmed_by": "andrew"},
        ),
        (
            ["confirm-click", "k1", "andrew", "--text", "human wording"],
            "click/resolve",
            {
                "candidate_id": "k1",
                "verdict": "accept",
                "confirmed_by": "andrew",
                "text": "human wording",
            },
        ),
        (
            ["decline-click", "k1"],
            "click/resolve",
            {"candidate_id": "k1", "verdict": "decline"},
        ),
        (
            ["reconsider-pair", "c1", "c2"],
            "click/reconsider",
            {"a": "c1", "b": "c2"},
        ),
        (["derive", "f1"], "derive", {"frame_id": "f1"}),
        (["unfold", "f1"], "unfold", {"frame_id": "f1"}),
        (["descend", "f1"], "descend", {"position_id": "f1"}),
        (["field"], "field", {"altitude": None}),
        (["field", "--altitude", "2"], "field", {"altitude": 2}),
        (
            ["harvest", "--altitude", "1", "--max-items", "5"],
            "harvest",
            {"altitude": 1, "max_items": 5},
        ),
    ],
)
def test_main_routes_ladder_commands_to_their_api_actions(
    monkeypatch, capsys, argv, action, payload
):
    posts = []

    class MainClient:
        def __init__(self, base_url, timeout):
            pass

        def post(self, posted_action, posted_payload):
            posts.append((posted_action, posted_payload))
            return {"routed": posted_action}

    monkeypatch.setattr(cli, "Client", MainClient)

    assert cli.main(argv) == 0
    capsys.readouterr()
    assert posts == [(action, payload)]


def test_confirm_click_cannot_be_spelled_without_a_confirmer(capsys):
    # §1.6 — "a click is confirmed by a human or not at all". argparse refuses
    # the command before a request can even be built; the server refuses it
    # again on arrival. There is no CLI spelling of an anonymous acceptance.
    with pytest.raises(SystemExit):
        cli.main(["confirm-click", "k1"])


def test_positions_is_a_projection_of_state_and_posts_nothing(monkeypatch, capsys):
    """The ladder view is read-only: it must never write to the field."""
    posts = []

    class MainClient:
        def __init__(self, base_url, timeout):
            pass

        def get_state(self):
            return _ladder_state()

        def post(self, action, payload):  # pragma: no cover - must not run
            posts.append((action, payload))
            raise AssertionError("positions must not post")

    monkeypatch.setattr(cli, "Client", MainClient)

    assert cli.main(["positions", "--altitude", "1"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert posts == []
    assert [entry["id"] for entry in printed["positions"]] == ["f1"]
    assert printed["floors"] == 2


def test_the_cli_has_no_route_that_mints_a_frame_directly():
    """§2.1/§2.4 — the inbox is the only door from inference to the field.

    Stated against the routing table rather than the help text: a readable
    alias that posted somewhere else would pass a docstring check.
    """
    assert "collide" not in cli.ACTIONS
    assert not any("collide" in action for action in cli.ACTIONS)
    # Everything that can create a frame goes through click/resolve, and the
    # server requires `confirmed_by` there.
    frame_making = {
        action
        for action in cli._COMMAND_ACTIONS.values()
        if action.startswith("click/")
    }
    assert frame_making <= {
        "click/propose",
        "click/pending",
        "click/resolve",
        "click/reconsider",
    }


# --------------------------------------------------------------- the scenario


SMOKE = Path(__file__).resolve().parent.parent / "scenarios" / "smoke.json"


def test_shipped_smoke_scenario_is_valid_for_the_runner():
    document = cli.load_scenario(SMOKE)
    steps = document["steps"]

    assert steps, "the smoke scenario must do something"
    for number, step in enumerate(steps, start=1):
        if "action" in step:
            assert step["action"] in cli.ACTIONS, f"step {number}"
        elif "wait" in step:
            cli._condition(step["wait"])
        elif "assert" in step:
            cli._condition(step["assert"])
        else:
            assert step.get("state") is True, f"step {number} does nothing"


def test_shipped_smoke_scenario_asserts_only_known_metrics():
    """Every asserted path must be a real metric, not a typo that reads a dict.

    A misspelled metric would fall through to the dotted-path walk and raise
    at runtime instead of failing here, so this pins the vocabulary.
    """
    document = cli.load_scenario(SMOKE)
    empty = {
        "cards": [],
        "positions": [],
        "click_candidates": [],
        "click_attempts": [],
        "question": "",
    }

    for step in document["steps"]:
        condition = step.get("assert") or step.get("wait")
        if not condition:
            continue
        # Resolves against an empty field without raising => it is a metric
        # (or a top-level state key), never an accidental deep path.
        cli.state_value(empty, condition["path"])


def test_shipped_smoke_scenario_exercises_the_ladder_law():
    """The scenario is the deliverable, so pin what it must still prove."""
    document = cli.load_scenario(SMOKE)
    actions = [step["action"] for step in document["steps"] if "action" in step]
    asserted = {
        (step.get("assert") or step.get("wait"))["path"]
        for step in document["steps"]
        if step.get("assert") or step.get("wait")
    }

    # §2.1/§2.3 — a recognition request, and the retry door.
    assert "click/propose" in actions
    assert "click/reconsider" in actions
    assert actions.count("click/propose") >= 2, "the ledger must be re-tested"
    # §3.1/§3.3 — the navigator and the brief.
    assert {"field", "descend", "harvest"} <= set(actions)
    # §1.6/§2.4 — nothing folded, nothing settled, nothing queued.
    assert {
        "positions.folded",
        "inbox.count",
        "clicks.confirmed",
        "attempts.count",
        "floors.count",
    } <= asserted


def test_main_returns_nonzero_and_writes_api_error_to_stderr(monkeypatch, capsys):
    class RejectingClient:
        def __init__(self, base_url, timeout):
            pass

        def post(self, action, payload):
            raise cli.CliError("question required")

    monkeypatch.setattr(cli, "Client", RejectingClient)

    status = cli.main(["seed", ""])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "question required" in captured.err
