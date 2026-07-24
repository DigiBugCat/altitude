"""Contract tests for Magpie's local HTTP CLI and scenario runner.

The client transport is replaced at the urllib boundary, and scenarios use a
small recording client.  Nothing in this module opens a socket or touches a
real Magpie runtime.
"""

from __future__ import annotations

import io
import json
import urllib.error

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
        ("collide", {"a": "c1", "b": "c2"}),
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
