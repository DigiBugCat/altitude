import json
import urllib.error
from email.message import Message

from magpie.raven_client import RavenClient


class FakeResponse:
    def __init__(self, payload=b"", *, content_type="application/json", session=None):
        self._payload = payload
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if session:
            self.headers["Mcp-Session-Id"] = session

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


class RavenServer:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout):
        message = json.loads(request.data)
        self.requests.append((request, timeout, message))
        if message["method"] == "initialize":
            return FakeResponse(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "raven", "version": "1"},
                        },
                    }
                ).encode(),
                session="session-1",
            )
        if message["method"] == "notifications/initialized":
            return FakeResponse()
        name = message["params"]["name"]
        arguments = message["params"]["arguments"]
        if name == "remember":
            value = {"ok": True, "id": "mem_new", "status": "enqueued"}
        elif name == "recall":
            value = {"ok": True, "query": arguments["query"], "results": []}
        else:
            value = {"ok": True, "node": {"id": arguments["memory_id"]}}
        return FakeResponse(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(value)}],
                        "structuredContent": value,
                        "isError": False,
                    },
                }
            ).encode()
        )


def test_client_is_disabled_without_url_and_does_not_touch_transport():
    touched = []
    client = RavenClient(None, opener=lambda *_a, **_k: touched.append(True))

    result = client.recall("launch plan")

    assert result.ok is False
    assert result.disabled is True
    assert result.unavailable is True
    assert touched == []


def test_client_reads_url_and_key_from_environment():
    client = RavenClient.from_env(
        {
            "MAGPIE_RAVEN_URL": "https://raven.internal",
            "MAGPIE_RAVEN_API_KEY": "api-key-alias",
            "MAGPIE_RAVEN_AGENT_ID": "magpie-test",
            "MAGPIE_RAVEN_TIMEOUT": "3.5",
        }
    )

    assert client.url == "https://raven.internal/mcp"
    assert client.api_key == "api-key-alias"
    assert client.agent_id == "magpie-test"
    assert client.timeout == 3.5


def test_client_initializes_once_and_calls_only_requested_raven_tools():
    server = RavenServer()
    client = RavenClient(
        "http://raven.internal:3005",
        api_key="secret-key",
        agent_id="magpie-local",
        opener=server.open,
    )

    remembered = client.remember(
        "Keep this",
        source="human",
        tags=["workspace"],
        episode_id="ws_1",
    )
    recalled = client.recall("keep", limit=4, expand=0)
    fetched = client.get("mem_new", depth=2)

    assert remembered.ok and remembered.value["id"] == "mem_new"
    assert recalled.ok and recalled.value["query"] == "keep"
    assert fetched.ok and fetched.value["node"]["id"] == "mem_new"
    assert [item[2]["method"] for item in server.requests] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",
        "tools/call",
    ]
    first_tool_request = server.requests[2][0]
    assert first_tool_request.full_url == "http://raven.internal:3005/mcp"
    assert first_tool_request.get_header("Authorization") == "Bearer secret-key"
    assert first_tool_request.get_header("X-agent-id") == "magpie-local"
    assert first_tool_request.get_header("Mcp-session-id") == "session-1"
    assert first_tool_request.get_header("Mcp-protocol-version") == "2025-06-18"
    assert server.requests[2][2]["params"]["name"] == "remember"


def test_client_degrades_transport_failure_without_leaking_key():
    def unavailable(_request, timeout):
        raise urllib.error.URLError("connection refused")

    client = RavenClient(
        "http://127.0.0.1:9/mcp", api_key="do-not-leak", opener=unavailable
    )
    result = client.get("mem_1")

    assert result.ok is False
    assert result.unavailable is True
    assert result.disabled is False
    assert "connection refused" in result.error
    assert "do-not-leak" not in result.error


def test_client_parses_streamable_http_sse_result():
    server = RavenServer()

    def open_sse(request, timeout):
        message = json.loads(request.data)
        if message["method"] != "tools/call":
            return server.open(request, timeout)
        value = {"ok": True, "results": [{"id": "mem_1"}]}
        event = (
            "event: message\n"
            f"data: {json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': {'structuredContent': value}})}\n\n"
        )
        return FakeResponse(
            event.encode(), content_type="text/event-stream; charset=utf-8"
        )

    result = RavenClient("http://raven/mcp", opener=open_sse).recall("idea")

    assert result.ok is True
    assert result.value["results"][0]["id"] == "mem_1"


def test_client_maps_raven_validation_failure_to_non_outage_result():
    server = RavenServer()

    def validation_error(request, timeout):
        message = json.loads(request.data)
        if message["method"] != "tools/call":
            return server.open(request, timeout)
        value = {"ok": False, "error": "content is empty"}
        return FakeResponse(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"structuredContent": value},
                }
            ).encode()
        )

    result = RavenClient("http://raven/mcp", opener=validation_error).remember("")

    assert result.ok is False
    assert result.unavailable is False
    assert result.error == "content is empty"
