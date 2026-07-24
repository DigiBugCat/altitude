"""HTTP contract tests for creating private ElevenLabs Agent sessions."""

from __future__ import annotations

import json

import pytest

from magpie import server


class _UpstreamResponse:
    def __init__(self, payload, status: int = 200):
        self.payload = payload
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture
def voice_server(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(server, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "magpie.sqlite3")
    monkeypatch.setattr(server, "STORE", None)
    monkeypatch.setattr(server, "ENGINE", None)
    monkeypatch.setattr(server, "WORKSPACE_ID", None)

    store, workspace, engine = server._initialize_storage()
    server.STORE = store
    server.WORKSPACE_ID = workspace.id
    server.ENGINE = engine
    from starlette.testclient import TestClient

    # TestClient is in-process and does not present a loopback client address.
    # Production serve() still uses local access by default.
    runtime = server.create_runtime(access="public")
    app = runtime.http_app(path="/mcp", stateless_http=True)
    try:
        with TestClient(app) as client:
            yield client, workspace.id
    finally:
        store.close()
        server.STORE = None
        server.ENGINE = None
        server.WORKSPACE_ID = None


def _post_session(client, workspace_id: str | None = None):
    body = {} if workspace_id is None else {"workspace_id": workspace_id}
    response = client.post(
        "/api/voice/session",
        content=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return response.status_code, response.json()


@pytest.mark.parametrize("missing", ["ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID"])
def test_voice_session_returns_503_when_server_configuration_is_missing(
    voice_server, monkeypatch, missing
):
    client, workspace_id = voice_server
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-secret")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent-test")
    monkeypatch.delenv(missing, raising=False)

    status, payload = _post_session(client, workspace_id)

    assert status == 503
    assert payload["ok"] is False
    assert "config" in payload["error"].lower()
    assert "test-secret" not in json.dumps(payload)


def test_voice_session_fetches_signed_url_server_side_without_exposing_key(
    voice_server, monkeypatch
):
    client, workspace_id = voice_server
    monkeypatch.setenv("ELEVENLABS_API_KEY", "private-eleven-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent with/slash")
    observed = {}

    def fake_urlopen(request, timeout=None):
        observed["url"] = request.full_url
        observed["headers"] = {k.lower(): v for k, v in request.header_items()}
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        return _UpstreamResponse(
            {"signed_url": "wss://api.elevenlabs.test/v1/convai/conversation"}
        )

    # The server module owns the upstream call; patch its imported urlopen,
    # rather than this test module's client transport.
    monkeypatch.setattr(server, "urlopen", fake_urlopen)

    status, payload = _post_session(client, workspace_id)

    assert status == 200
    assert payload == {
        "ok": True,
        "signed_url": "wss://api.elevenlabs.test/v1/convai/conversation",
        "agent_id": "agent with/slash",
    }
    assert observed["method"] == "GET"
    assert (
        observed["url"]
        == "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
        "?agent_id=agent+with%2Fslash"
    )
    assert observed["headers"]["xi-api-key"] == "private-eleven-key"
    assert "private-eleven-key" not in json.dumps(payload)
    assert observed["timeout"] is not None


def test_voice_session_requires_explicit_workspace_before_upstream_call(
    voice_server, monkeypatch
):
    client, _workspace_id = voice_server
    monkeypatch.setenv("ELEVENLABS_API_KEY", "private-eleven-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent-test")

    def unexpected_upstream(*_args, **_kwargs):
        raise AssertionError("missing workspace must fail before upstream")

    monkeypatch.setattr(server, "urlopen", unexpected_upstream)
    status, payload = _post_session(client)

    assert status == 400
    assert payload["ok"] is False
    assert "workspace_id" in payload["error"]


@pytest.mark.parametrize(
    "upstream_payload",
    [
        [],
        {},
        {"signed_url": "https://api.elevenlabs.io/not-a-websocket"},
        {"signed_url": "wss:///missing-host"},
    ],
)
def test_voice_session_rejects_malformed_upstream_payload(
    voice_server, monkeypatch, upstream_payload
):
    client, workspace_id = voice_server
    monkeypatch.setenv("ELEVENLABS_API_KEY", "private-eleven-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent-test")
    monkeypatch.setattr(
        server,
        "urlopen",
        lambda *_args, **_kwargs: _UpstreamResponse(upstream_payload),
    )

    status, payload = _post_session(client, workspace_id)

    assert status == 503
    assert payload["ok"] is False
    assert "invalid" in payload["error"].lower()
    assert "private-eleven-key" not in json.dumps(payload)
