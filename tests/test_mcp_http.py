"""End-to-end contract tests for Magpie's ElevenAgents-facing MCP surface.

These tests deliberately exercise the HTTP boundary rather than importing an
MCP implementation detail.  ElevenLabs speaks standard Streamable HTTP MCP,
and keeping the assertions at that seam leaves Magpie free to use AviaryMCP
or another conforming MCP implementation internally.
"""

from __future__ import annotations

import json

import pytest

from magpie import engine as engine_mod
from magpie import mcp_surface
from magpie import server


SAFE_TOOLS = {
    "guide",
    "list_workspaces",
    "create_workspace",
    "get_field",
    "get_conversation_map",
    "recall_workspace",
    "adopt_memory",
    "seed_field",
    "contribute",
    "collide",
    "organize",
    "harvest",
}
WORKSPACE_SCOPED_TOOLS = {
    "get_field",
    "get_conversation_map",
    "recall_workspace",
    "adopt_memory",
    "seed_field",
    "contribute",
    "collide",
    "organize",
    "harvest",
}
PROVENANCE_OR_DESTRUCTIVE_TOOLS = {"judge", "kill", "verify"}


@pytest.fixture
def mcp_server(monkeypatch, tmp_path):
    """Run the real Handler with isolated storage and no background metabolism."""

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

    # The production listener is AviaryMCP/ASGI. TestClient runs the actual
    # ASGI lifespan and Streamable HTTP session manager without opening a
    # loopback socket or testing the retired stdlib Handler by accident.
    from starlette.testclient import TestClient

    # TestClient identifies itself as an in-process test host rather than a
    # loopback peer. Public access here only disables that transport check;
    # production serve() retains LocalAuth by default.
    runtime = server.create_runtime(access="public")
    app = runtime.http_app(path="/mcp", stateless_http=True)
    try:
        with TestClient(app) as client:
            yield client, server
    finally:
        store.close()
        server.STORE = None
        server.ENGINE = None
        server.WORKSPACE_ID = None


def _decode_mcp_response(raw: bytes, content_type: str) -> dict:
    """Accept both plain JSON and Streamable HTTP's SSE representation."""

    text = raw.decode("utf-8")
    if "text/event-stream" not in content_type:
        return json.loads(text)
    data = "\n".join(
        line.removeprefix("data:").strip()
        for line in text.splitlines()
        if line.startswith("data:")
    )
    return json.loads(data)


def _mcp(client, method: str, params: dict | None, request_id: int) -> dict:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    response = client.post(
        "/mcp",
        content=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
        },
    )
    assert response.status_code == 200, response.text
    result = _decode_mcp_response(
        response.content, response.headers.get("Content-Type", "")
    )
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == request_id
    return result


def _call_tool(client, name: str, arguments: dict | None = None, request_id: int = 10):
    response = _mcp(
        client,
        "tools/call",
        {"name": name, "arguments": arguments or {}},
        request_id,
    )
    result = response["result"]
    assert result.get("isError") is not True, result
    text_items = [
        item["text"]
        for item in result.get("content", [])
        if item.get("type") == "text"
    ]
    assert text_items, result
    return json.loads(text_items[0])


def test_mcp_initialize_advertises_tools_capability(mcp_server):
    client, _srv = mcp_server

    response = _mcp(
        client,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "magpie-contract-test", "version": "1"},
        },
        1,
    )

    result = response["result"]
    assert result["protocolVersion"]
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"].lower() == "magpie"


def test_tools_list_is_exactly_the_safe_voice_agent_catalog(mcp_server):
    client, _srv = mcp_server

    response = _mcp(client, "tools/list", {}, 2)
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}

    assert names == SAFE_TOOLS
    assert names.isdisjoint(PROVENANCE_OR_DESTRUCTIVE_TOOLS)
    for tool in tools:
        assert tool.get("description"), tool["name"]
        assert tool["inputSchema"]["type"] == "object"


def test_every_workspace_specific_tool_requires_an_explicit_workspace_id(mcp_server):
    client, _srv = mcp_server

    tools = {
        tool["name"]: tool
        for tool in _mcp(client, "tools/list", {}, 3)["result"]["tools"]
    }

    for name in WORKSPACE_SCOPED_TOOLS:
        schema = tools[name]["inputSchema"]
        assert "workspace_id" in schema.get("properties", {}), name
        assert "workspace_id" in schema.get("required", []), name
    assert "workspace_id" not in tools["list_workspaces"]["inputSchema"].get(
        "required", []
    )
    assert "workspace_id" not in tools["create_workspace"]["inputSchema"].get(
        "required", []
    )


def test_tools_call_guide_and_list_workspaces_return_json_content(mcp_server):
    client, srv = mcp_server

    guide = _call_tool(client, "guide", request_id=4)
    listed = _call_tool(client, "list_workspaces", request_id=5)

    assert guide
    assert [item["id"] for item in listed["workspaces"]] == [srv.WORKSPACE_ID]


def test_mcp_create_workspace_does_not_switch_the_browser_workspace(mcp_server):
    client, srv = mcp_server
    browser_workspace_id = srv.WORKSPACE_ID

    created = _call_tool(
        client,
        "create_workspace",
        {"name": "Voice inbox", "question": "What did I notice?"},
        request_id=6,
    )

    assert created["workspace"]["id"] != browser_workspace_id
    assert srv.WORKSPACE_ID == browser_workspace_id
    assert srv.STORE.current_workspace_id() == browser_workspace_id
    assert srv.ENGINE.question == ""


def test_explicit_mcp_routing_survives_concurrent_browser_workspace_switch(
    mcp_server,
):
    """An agent must never mutate whichever workspace the browser last opened."""

    client, srv = mcp_server
    workspace_a = srv.WORKSPACE_ID
    workspace_b = srv._api_workspace_create({"name": "Browser B"})["workspace"]["id"]
    assert srv.WORKSPACE_ID == workspace_b

    result = _call_tool(
        client,
        "seed_field",
        {"workspace_id": workspace_a, "question": "Question for A only"},
        request_id=7,
    )

    assert result["question"] == "Question for A only"
    assert srv.WORKSPACE_ID == workspace_b
    assert srv.STORE.current_workspace_id() == workspace_b
    assert srv.ENGINE.question == ""
    stored_a = srv.STORE.load_workspace(workspace_a)
    restored_a = engine_mod.Engine.from_state(stored_a.snapshot)
    assert restored_a.question == "Question for A only"


def test_get_field_reads_named_workspace_without_changing_active_browser_state(
    mcp_server,
):
    client, srv = mcp_server
    workspace_a = srv.WORKSPACE_ID
    workspace_b = srv._api_workspace_create(
        {"name": "Browser B", "question": "B question"}
    )["workspace"]["id"]

    field_a = _call_tool(
        client, "get_field", {"workspace_id": workspace_a}, request_id=8
    )

    assert field_a["workspace"]["id"] == workspace_a
    assert field_a["question"] == ""
    assert srv.WORKSPACE_ID == workspace_b
    assert srv.ENGINE.question == "B question"


def test_get_conversation_map_returns_a_normalized_sparse_digest(mcp_server):
    client, srv = mcp_server
    workspace_id = srv.WORKSPACE_ID
    browser_workspace_id = srv._api_workspace_create(
        {"name": "Browser B", "question": "B question"}
    )["workspace"]["id"]

    result = _call_tool(
        client,
        "get_conversation_map",
        {"workspace_id": workspace_id},
        request_id=9,
    )

    assert result["workspace_id"] == workspace_id
    assert set(result["digest"]) >= {
        "themes",
        "recurring_ideas",
        "open_questions",
        "decisions",
        "constraints",
        "experiments",
        "tasks",
        "between_ideas",
    }
    assert all(
        isinstance(result["digest"][key], list)
        for key in (
            "themes",
            "recurring_ideas",
            "open_questions",
            "decisions",
            "constraints",
            "experiments",
            "tasks",
            "between_ideas",
        )
    )
    assert len(result["digest"]["between_ideas"]) <= 3
    assert srv.WORKSPACE_ID == browser_workspace_id
    assert srv.STORE.current_workspace_id() == browser_workspace_id
    assert srv.ENGINE.question == "B question"


def test_conversation_map_preserves_digest_and_caps_secondary_connections(
    monkeypatch,
):
    digest = {
        "themes": [{"name": "Reliability"}],
        "recurring_ideas": [{"canonical_id": "idea-1", "occurrence_count": 3}],
        "open_questions": [{"text": "What fails first?"}],
        "between_ideas": [{"id": f"link-{index}"} for index in range(5)],
        "source_context_version": 7,
    }
    monkeypatch.setattr(
        mcp_surface,
        "_state_for",
        lambda workspace_id: {"workspace": {"id": workspace_id}, "digest": digest},
    )

    result = mcp_surface._conversation_map("ws-test")

    assert result["digest"]["themes"] == [{"name": "Reliability"}]
    assert result["digest"]["recurring_ideas"][0]["occurrence_count"] == 3
    assert result["digest"]["source_context_version"] == 7
    assert [item["id"] for item in result["digest"]["between_ideas"]] == [
        "link-0",
        "link-1",
        "link-2",
    ]
    for key in ("decisions", "constraints", "experiments", "tasks"):
        assert result["digest"][key] == []


def test_birdz_reports_magpie_ready_on_the_same_http_server(mcp_server):
    client, _srv = mcp_server

    response = client.get("/birdz")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["service"] == "magpie"
