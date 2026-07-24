"""One-port AviaryMCP HTTP runtime for Magpie."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

from aviary_mcp import AviaryMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import server
from .mcp_surface import GUIDE, register_tools


def _json_ok(payload: dict[str, Any]) -> JSONResponse:
    body = {"ok": True}
    body.update(payload or {})
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


def _json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": message},
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


async def _json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        return {}
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("body must be a JSON object")
    return value


def create_runtime(*, access: str = "local") -> AviaryMCP:
    """Build the production MCP server and the existing local browser routes."""
    mcp = AviaryMCP(
        "magpie",
        api_base="/mcp-api/v1",
        access=access,
        instructions=(
            GUIDE
            + "\nThe browser and MCP share storage, but MCP tools must always use "
            "the explicit workspace_id supplied by the conversation."
        ),
    )
    register_tools(mcp)

    @mcp.custom_route("/api/state", methods=["GET"])
    async def api_state(request: Request) -> Response:
        try:
            return _json_ok(
                server._api_state(
                    {"workspace_id": request.query_params.get("workspace_id", "")}
                )
            )
        except Exception as exc:
            server.traceback.print_exc()
            return _json_error(500, str(exc))

    @mcp.custom_route("/api/workspaces/current", methods=["GET"])
    async def api_workspace_current(_request: Request) -> Response:
        try:
            return _json_ok(server._api_workspace_current({}))
        except Exception as exc:
            server.traceback.print_exc()
            return _json_error(500, str(exc))

    @mcp.custom_route("/api/workspaces", methods=["GET", "POST"])
    async def api_workspaces(request: Request) -> Response:
        if request.method == "GET":
            try:
                return _json_ok(server._api_workspaces({}))
            except Exception as exc:
                server.traceback.print_exc()
                return _json_error(500, str(exc))
        return await _dispatch_post(request)

    async def _dispatch_post(request: Request) -> Response:
        handler = server.ROUTES.get(request.url.path)
        if handler is None:
            return _json_error(404, "no such endpoint")
        try:
            body = await _json_body(request)
        except Exception as exc:
            return _json_error(400, f"bad JSON body: {exc}")
        try:
            # Some adapters perform synchronous upstream I/O. Offload the
            # complete call so one slow request cannot stall the ASGI loop.
            return _json_ok(await asyncio.to_thread(handler, body))
        except KeyError as exc:
            return _json_error(404, f"unknown id: {exc}")
        except ValueError as exc:
            return _json_error(400, str(exc))
        except server.ServiceUnavailable as exc:
            return _json_error(503, str(exc))
        except Exception as exc:
            server.traceback.print_exc()
            return _json_error(500, str(exc))

    @mcp.custom_route("/api/{action:path}", methods=["POST"])
    async def api_post(request: Request) -> Response:
        return await _dispatch_post(request)

    def static_response(path: Path, *, head: bool = False) -> Response:
        if not path.is_file():
            return _json_error(404, "not found")
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        return Response(
            b"" if head else path.read_bytes(),
            media_type=content_type,
            headers={"Cache-Control": "no-store"},
        )

    @mcp.custom_route("/", methods=["GET", "HEAD"])
    async def index(request: Request) -> Response:
        return static_response(server.APP_DIR / "index.html", head=request.method == "HEAD")

    @mcp.custom_route("/index.html", methods=["GET", "HEAD"])
    async def index_file(request: Request) -> Response:
        return static_response(server.APP_DIR / "index.html", head=request.method == "HEAD")

    return mcp
