"""Small, dependency-free client for Raven's MCP tools.

Magpie deliberately exposes only Raven's ingest and read operations here.
There is no feedback/verdict method: merely showing or manipulating a local
projection must never mutate Raven's global epistemic state.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import Message
from typing import Any, Callable

_PROTOCOL_VERSION = "2025-11-25"


@dataclass(frozen=True)
class RavenResult:
    """A Raven operation result that makes disabled/outage states explicit."""

    ok: bool
    value: dict[str, Any] | None = None
    error: str | None = None
    unavailable: bool = False
    disabled: bool = False


class RavenClient:
    """Synchronous streamable-HTTP MCP client for Raven.

    The client lazily initializes an MCP session and can be shared between
    Magpie request threads. Transport errors are returned as ``RavenResult``
    values so Raven being down does not take the local workspace down with it.
    """

    def __init__(
        self,
        url: str | None,
        *,
        api_key: str | None = None,
        agent_id: str = "magpie",
        timeout: float = 10.0,
        opener: Callable[..., Any] | None = None,
    ):
        self.url = self._normalize_url(url)
        self.api_key = str(api_key or "")
        self.agent_id = str(agent_id or "magpie")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("Raven timeout must be positive")
        self._open = opener or urllib.request.urlopen
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._protocol_version = _PROTOCOL_VERSION
        self._next_id = 1

    @classmethod
    def from_env(
        cls, environ: dict[str, str] | os._Environ[str] | None = None
    ) -> "RavenClient":
        env = os.environ if environ is None else environ
        try:
            timeout = float(env.get("MAGPIE_RAVEN_TIMEOUT", "10"))
        except ValueError:
            timeout = 10.0
        return cls(
            env.get("MAGPIE_RAVEN_URL"),
            api_key=(
                env.get("MAGPIE_RAVEN_KEY")
                or env.get("MAGPIE_RAVEN_API_KEY")
            ),
            agent_id=env.get("MAGPIE_RAVEN_AGENT_ID", "magpie"),
            timeout=timeout,
        )

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def remember(
        self,
        content: str,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        episode_id: str | None = None,
        hints: dict[str, Any] | None = None,
    ) -> RavenResult:
        arguments: dict[str, Any] = {"content": str(content)}
        if source is not None:
            arguments["source"] = str(source)
        if tags is not None:
            arguments["tags"] = [str(tag) for tag in tags]
        if episode_id is not None:
            arguments["episode_id"] = str(episode_id)
        if hints is not None:
            arguments["hints"] = dict(hints)
        return self._call_tool("remember", arguments)

    def recall(
        self, query: str, *, limit: int = 10, expand: int = 1
    ) -> RavenResult:
        return self._call_tool(
            "recall",
            {"query": str(query), "limit": int(limit), "expand": int(expand)},
        )

    def get(self, memory_id: str, *, depth: int = 1) -> RavenResult:
        return self._call_tool(
            "get", {"memory_id": str(memory_id), "depth": int(depth)}
        )

    def _call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> RavenResult:
        if not self.enabled:
            return RavenResult(
                ok=False,
                error="Raven integration is disabled",
                unavailable=True,
                disabled=True,
            )
        with self._lock:
            try:
                self._ensure_session()
                response, _headers = self._jsonrpc(
                    "tools/call",
                    {"name": str(name), "arguments": dict(arguments or {})},
                )
                return self._tool_result(response)
            except (
                OSError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                # A dead/stale streamable-HTTP session should be retried with a
                # fresh initialize handshake on the next request.
                self._session_id = None
                return RavenResult(
                    ok=False,
                    error=f"Raven unavailable: {self._safe_error(exc)}",
                    unavailable=True,
                )

    def _ensure_session(self) -> None:
        if self._session_id is not None:
            return
        response, headers = self._jsonrpc(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "magpie", "version": "0.1.0"},
            },
            include_session=False,
        )
        if "error" in response:
            raise ValueError(self._rpc_error(response["error"]))
        result = response.get("result")
        if isinstance(result, dict) and result.get("protocolVersion"):
            self._protocol_version = str(result["protocolVersion"])
        self._session_id = headers.get("Mcp-Session-Id")
        self._jsonrpc(
            "notifications/initialized",
            notification=True,
            allow_empty=True,
        )

    def _jsonrpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        notification: bool = False,
        include_session: bool = True,
        allow_empty: bool = False,
    ) -> tuple[dict[str, Any], Message]:
        request_id: int | None = None
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            request_id = self._next_id
            self._next_id += 1
            message["id"] = request_id
        if params is not None:
            message["params"] = params
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self._protocol_version,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.agent_id:
            headers["X-Agent-Id"] = self.agent_id
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(
            self.url,
            data=json.dumps(message, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            response = self._open(request, timeout=self.timeout)
            with response:
                raw = response.read()
                response_headers = response.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read(1024).decode("utf-8", errors="replace").strip()
            raise OSError(
                f"HTTP {exc.code}" + (f": {detail}" if detail else "")
            ) from exc
        if not raw:
            if allow_empty:
                return {}, response_headers
            raise ValueError("empty Raven response")
        payload = self._decode_response(raw, response_headers, request_id)
        return payload, response_headers

    @staticmethod
    def _decode_response(
        raw: bytes, headers: Message, request_id: int | None
    ) -> dict[str, Any]:
        text = raw.decode("utf-8")
        content_type = headers.get_content_type()
        if content_type == "text/event-stream" or text.lstrip().startswith("event:"):
            candidates: list[dict[str, Any]] = []
            data_lines: list[str] = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    parsed = json.loads("\n".join(data_lines))
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                    data_lines = []
            if data_lines:
                parsed = json.loads("\n".join(data_lines))
                if isinstance(parsed, dict):
                    candidates.append(parsed)
            for candidate in candidates:
                if request_id is None or candidate.get("id") == request_id:
                    return candidate
            raise ValueError("Raven SSE response did not contain the request result")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Raven response must be a JSON object")
        return parsed

    @classmethod
    def _tool_result(cls, response: dict[str, Any]) -> RavenResult:
        if "error" in response:
            return RavenResult(ok=False, error=cls._rpc_error(response["error"]))
        result = response.get("result")
        if not isinstance(result, dict):
            return RavenResult(ok=False, error="Raven returned no tool result")
        if result.get("isError"):
            return RavenResult(ok=False, error=cls._content_error(result))
        value = result.get("structuredContent")
        if not isinstance(value, dict):
            value = cls._text_content(result)
        if not isinstance(value, dict):
            return RavenResult(ok=False, error="Raven returned an unreadable tool result")
        if value.get("ok") is False:
            return RavenResult(
                ok=False, value=value, error=str(value.get("error", "Raven call failed"))
            )
        return RavenResult(ok=True, value=value)

    @staticmethod
    def _text_content(result: dict[str, Any]) -> dict[str, Any] | None:
        for item in result.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            try:
                value = json.loads(str(item.get("text", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _content_error(result: dict[str, Any]) -> str:
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text", "Raven tool error"))
        return "Raven tool error"

    @staticmethod
    def _rpc_error(error: Any) -> str:
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message", "Raven RPC error")
            return f"{message} ({code})" if code is not None else str(message)
        return str(error)

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        text = str(exc).strip() or exc.__class__.__name__
        return text[:500]

    @staticmethod
    def _normalize_url(url: str | None) -> str | None:
        value = str(url or "").strip()
        if not value:
            return None
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("MAGPIE_RAVEN_URL must be an http(s) URL")
        path = parsed.path.rstrip("/")
        if not path:
            path = "/mcp"
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, "")
        )
