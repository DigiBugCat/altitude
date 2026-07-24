"""A small, provider-neutral inference boundary for Magpie.

Magpie's current cognition is request/response inference, not an agent
runtime.  This module therefore exposes one OpenAI-compatible Chat
Completions client and a tiny ``Chain`` wrapper used by ``workers.py``.

The production client defaults to Fireworks' OpenAI-compatible API.  It is
configured entirely through environment variables:

``MAGPIE_LLM_BASE_URL``
    OpenAI-compatible API base URL.  Defaults to Fireworks.
``MAGPIE_LLM_API_KEY``
    API key.  Falls back to ``FIREWORKS_API_KEY``.
``MAGPIE_LLM_MODEL``
    Model identifier.  Required; there is deliberately no drift-prone default.
``MAGPIE_LLM_NAME``
    Provider label attached to generated artifacts.  Defaults to ``fireworks``.
``MAGPIE_LLM_STUB``
    Set to a truthy value to explicitly use the deterministic offline stub.

No subscription tokens, CLI auth files, or private provider protocols are
read or mutated.  Stdlib only; tests replace ``_post`` and never use network.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_TRUTHY = {"1", "true", "yes", "on"}


class ProviderUnavailable(Exception):
    """A transient inference failure which may succeed on a later attempt."""


class ProviderConfigurationError(ProviderUnavailable):
    """A permanent local/request configuration error requiring intervention."""


class ProviderResponseError(ProviderUnavailable):
    """The remote endpoint answered, but not with a usable response."""


def _post(url: str, body: bytes, headers: dict[str, str], timeout: int):
    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST")
    return urllib.request.urlopen(request, timeout=timeout)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _extract_json(text: str) -> Any:
    """Parse a JSON response, tolerating a Markdown fence or prose wrapper."""
    raw = (text or "").strip()
    if not raw:
        raise ProviderResponseError("empty structured response")
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().lower() in ("```", "```json"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ProviderResponseError(
                "structured response was not JSON") from None
        try:
            return json.loads(raw[start:end + 1])
        except (TypeError, ValueError):
            raise ProviderResponseError(
                "structured response was not JSON") from None


def _validate_schema(value: Any, schema: dict, path: str = "$") -> None:
    """Validate the JSON Schema subset used by Magpie's worker contracts."""
    expected = schema.get("type")
    type_ok = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float))
        and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if expected in type_ok and not type_ok[expected](value):
        raise ProviderResponseError(f"{path} must be {expected}")

    if "enum" in schema and value not in schema["enum"]:
        raise ProviderResponseError(f"{path} is not an allowed value")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                raise ProviderResponseError(f"{path} missing {key!r}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ProviderResponseError(
                    f"{path} has unexpected field {sorted(extras)[0]!r}")
        for key, child in value.items():
            if key in properties:
                _validate_schema(child, properties[key], f"{path}.{key}")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, child in enumerate(value):
            _validate_schema(child, schema["items"], f"{path}[{index}]")


def _coerce(text: str, schema: dict | None):
    """Return plain text or locally validate a structured JSON response."""
    if schema is None:
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("empty response")
        return text
    data = _extract_json(text)
    _validate_schema(data, schema)
    return data


class OpenAICompatibleProvider:
    """One configurable OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        name: str | None = None,
        max_tokens: int = 1024,
    ):
        self.base_url = (
            base_url
            if base_url is not None
            else os.environ.get("MAGPIE_LLM_BASE_URL", FIREWORKS_BASE_URL)
        ).strip()
        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.environ.get("MAGPIE_LLM_API_KEY")
                or os.environ.get("FIREWORKS_API_KEY")
                or ""
            )
        ).strip()
        self.model = (
            model
            if model is not None
            else os.environ.get("MAGPIE_LLM_MODEL", "")
        ).strip()
        self.name = (
            name
            if name is not None
            else os.environ.get("MAGPIE_LLM_NAME", "fireworks")
        ).strip() or "inference"
        self.max_tokens = max_tokens

    def _validate_config(self) -> None:
        if not self.base_url:
            raise ProviderConfigurationError("MAGPIE_LLM_BASE_URL is empty")
        if not self.base_url.startswith(("https://", "http://")):
            raise ProviderConfigurationError(
                "MAGPIE_LLM_BASE_URL must be an HTTP(S) URL")
        if not self.api_key:
            raise ProviderConfigurationError(
                "MAGPIE_LLM_API_KEY or FIREWORKS_API_KEY is unset")
        if not self.model:
            raise ProviderConfigurationError("MAGPIE_LLM_MODEL is unset")

    def available(self) -> bool:
        try:
            self._validate_config()
            return True
        except ProviderConfigurationError:
            return False

    def complete(
        self, prompt: str, schema: dict | None = None, timeout: int = 30
    ):
        self._validate_config()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "magpie_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = _chat_completions_url(self.base_url)
        try:
            response = _post(
                url, json.dumps(payload).encode("utf-8"), headers, timeout)
            data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404, 405, 422):
                raise ProviderConfigurationError(
                    f"{self.name} HTTP {exc.code}") from None
            raise ProviderUnavailable(
                f"{self.name} HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise ProviderUnavailable(
                f"{self.name} unreachable: {exc}") from None
        except (UnicodeError, ValueError, TypeError) as exc:
            raise ProviderResponseError(
                f"{self.name} returned invalid JSON: {exc}") from None

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderResponseError(
                f"{self.name} response had no message content") from None
        if not isinstance(content, str):
            raise ProviderResponseError(
                f"{self.name} response content was not text")
        return _coerce(content, schema)


class StubProvider:
    """Deterministic offline provider, enabled only by explicit configuration."""

    name = "stub"
    FOOT = "stub · offline"
    _PAYLOAD_LABELS = (
        "THOUGHT:", "CLAIM A:", "CLAIM B:", "FRAME:", "OPEN QUESTION:",
    )
    # `atomize()` does not use a `LABEL:` line — it sends a JSON object on the
    # line after `INPUT:`. Without this the stub echoed the prompt preamble,
    # which is identical for every thought, so every offline contribution
    # collapsed onto one card and the stub stopped being a *deterministic*
    # double and became a constant one.
    _PAYLOAD_JSON_KEYS = ("thought", "text")

    def available(self) -> bool:
        return True

    def _payload(self, prompt: str) -> str:
        found = []
        for line in (prompt or "").splitlines():
            line = line.strip()
            for label in self._PAYLOAD_LABELS:
                if line.startswith(label):
                    rest = line[len(label):].strip()
                    if rest and rest != "(none stated)":
                        found.append(rest)
                    break
            else:
                extracted = self._from_json_line(line)
                if extracted:
                    found.append(extracted)
        return " / ".join(found) if found else " ".join((prompt or "").split())

    def _from_json_line(self, line: str) -> str:
        """Pull the payload out of a JSON prompt body, if the line is one."""
        if not line.startswith("{"):
            return ""
        try:
            document = json.loads(line)
        except (ValueError, TypeError):
            return ""
        if not isinstance(document, dict):
            return ""
        for key in self._PAYLOAD_JSON_KEYS:
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def complete(
        self, prompt: str, schema: dict | None = None, timeout: int = 30
    ):
        payload = self._payload(prompt)
        if schema is None:
            return f"[{self.FOOT}] {payload[:280]}"
        return self._shape(schema, payload)

    def _shape(self, schema: dict, prompt: str):
        kind = schema.get("type", "object")
        if kind == "object":
            properties = schema.get("properties") or {}
            keys = schema.get("required") or list(properties)
            return {
                key: self._shape(
                    properties.get(key, {"type": "string"}), prompt)
                for key in keys
            }
        if kind == "array":
            return [self._shape(
                schema.get("items", {"type": "string"}), prompt)]
        if kind in ("number", "integer"):
            return 0
        if kind == "boolean":
            return False
        if kind == "null":
            return None
        if schema.get("enum"):
            return schema["enum"][0]
        return f"[{self.FOOT}] {prompt.strip()[:180]}"


class Chain:
    """Compatibility wrapper returning ``(result, provider_name)``.

    Production configuration contains exactly one inference provider.  A list
    may still be injected in tests, but a configuration error is never hidden
    by falling through to another provider.
    """

    def __init__(self, providers=None):
        self.providers = list(
            providers if providers is not None else self._from_env())
        self.last_provider: str | None = None

    @staticmethod
    def _from_env():
        explicit_stub = (
            os.environ.get("MAGPIE_LLM_STUB", "").strip().lower() in _TRUTHY
            or os.environ.get("MAGPIE_PROVIDERS", "").strip().lower() == "stub"
        )
        return [StubProvider() if explicit_stub
                else OpenAICompatibleProvider()]

    def complete(
        self, prompt: str, schema: dict | None = None, timeout: int = 30
    ):
        errors = []
        for provider in self.providers:
            try:
                output = provider.complete(
                    prompt, schema=schema, timeout=timeout)
                self.last_provider = provider.name
                return output, provider.name
            except ProviderConfigurationError:
                raise
            except ProviderUnavailable as exc:
                errors.append(f"{provider.name}: {exc}")
            except Exception as exc:
                errors.append(
                    f"{provider.name}: {type(exc).__name__}: {exc}")
        detail = "; ".join(errors) or "no inference provider configured"
        raise ProviderUnavailable(f"no provider answered — {detail}")

    def status(self) -> list[dict]:
        status = []
        for provider in self.providers:
            try:
                available = bool(provider.available())
            except Exception:
                available = False
            status.append({
                "name": provider.name,
                "available": available,
                "last": provider.name == self.last_provider,
            })
        return status


_CHAIN: Chain | None = None


def get_chain() -> Chain:
    """Return the process-wide inference client."""
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = Chain()
    return _CHAIN


def reset_chain(chain: Chain | None = None) -> None:
    """Swap or clear the singleton for tests or runtime reconfiguration."""
    global _CHAIN
    _CHAIN = chain
