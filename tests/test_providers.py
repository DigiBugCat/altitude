"""Offline tests for Magpie's OpenAI-compatible inference boundary."""

import io
import json
import os
import socket
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magpie import providers  # noqa: E402
from magpie.providers import (  # noqa: E402
    Chain,
    OpenAICompatibleProvider,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderUnavailable,
    StubProvider,
)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}


def _response(content):
    body = {"choices": [{"message": {"content": content}}]}
    return io.BytesIO(json.dumps(body).encode())


class Fake:
    def __init__(self, name, answer=None, fail=None, up=True):
        self.name = name
        self.answer = answer
        self.fail = fail
        self.up = up
        self.calls = 0

    def available(self):
        return self.up

    def complete(self, prompt, schema=None, timeout=30):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.answer


def configured(**overrides):
    values = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "model": "accounts/acme/models/test-model",
        "name": "test-provider",
    }
    values.update(overrides)
    return OpenAICompatibleProvider(**values)


# ------------------------------------------------------------ configuration


def test_defaults_to_fireworks_but_requires_model_and_key(monkeypatch):
    for name in (
        "MAGPIE_LLM_BASE_URL",
        "MAGPIE_LLM_API_KEY",
        "FIREWORKS_API_KEY",
        "MAGPIE_LLM_MODEL",
        "MAGPIE_LLM_NAME",
        "MAGPIE_LLM_STUB",
        "MAGPIE_PROVIDERS",
    ):
        monkeypatch.delenv(name, raising=False)

    provider = OpenAICompatibleProvider()
    assert provider.base_url == providers.FIREWORKS_BASE_URL
    assert provider.name == "fireworks"
    assert provider.available() is False
    with pytest.raises(ProviderConfigurationError) as exc:
        provider.complete("hello")
    assert "API_KEY" in str(exc.value)


def test_environment_config_and_fireworks_key_fallback(monkeypatch):
    monkeypatch.setenv("MAGPIE_LLM_BASE_URL", "https://gateway.test/openai/v1/")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    monkeypatch.setenv("MAGPIE_LLM_MODEL", "model-id")
    monkeypatch.setenv("MAGPIE_LLM_NAME", "gateway")
    monkeypatch.delenv("MAGPIE_LLM_API_KEY", raising=False)

    provider = OpenAICompatibleProvider()
    assert provider.base_url == "https://gateway.test/openai/v1/"
    assert provider.api_key == "fw-key"
    assert provider.model == "model-id"
    assert provider.name == "gateway"
    assert provider.available() is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"base_url": ""}, "BASE_URL"),
        ({"base_url": "file:///tmp/llm"}, "HTTP"),
        ({"api_key": ""}, "API_KEY"),
        ({"model": ""}, "MODEL"),
    ],
)
def test_invalid_configuration_is_permanent(override, message):
    provider = configured(**override)
    assert provider.available() is False
    with pytest.raises(ProviderConfigurationError) as exc:
        provider.complete("hello")
    assert message in str(exc.value)


def test_default_chain_does_not_silently_include_stub(monkeypatch):
    monkeypatch.delenv("MAGPIE_LLM_STUB", raising=False)
    monkeypatch.delenv("MAGPIE_PROVIDERS", raising=False)
    chain = Chain()
    assert len(chain.providers) == 1
    assert isinstance(chain.providers[0], OpenAICompatibleProvider)


def test_stub_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("MAGPIE_LLM_STUB", "true")
    assert isinstance(Chain().providers[0], StubProvider)


def test_legacy_stub_setting_remains_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("MAGPIE_LLM_STUB", raising=False)
    monkeypatch.setenv("MAGPIE_PROVIDERS", "stub")
    assert isinstance(Chain().providers[0], StubProvider)


# ------------------------------------------------------------ wire protocol


def test_text_completion_uses_openai_chat_completions(monkeypatch):
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen.update(
            url=url,
            payload=json.loads(body),
            headers=headers,
            timeout=timeout,
        )
        return _response("hello")

    monkeypatch.setattr(providers, "_post", fake_post)
    assert configured().complete("say hello", timeout=7) == "hello"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["payload"] == {
        "model": "accounts/acme/models/test-model",
        "messages": [{"role": "user", "content": "say hello"}],
        "max_tokens": 1024,
    }
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert seen["timeout"] == 7


def test_full_chat_completions_url_is_not_duplicated(monkeypatch):
    seen = {}

    def fake_post(url, *args, **kwargs):
        seen["url"] = url
        return _response("hello")

    monkeypatch.setattr(providers, "_post", fake_post)
    provider = configured(
        base_url="https://example.test/v1/chat/completions")
    assert provider.complete("hello") == "hello"
    assert seen["url"] == "https://example.test/v1/chat/completions"


def test_structured_completion_sends_json_schema_and_validates(monkeypatch):
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen["payload"] = json.loads(body)
        return _response('{"text": "ok"}')

    monkeypatch.setattr(providers, "_post", fake_post)
    assert configured().complete("p", schema=SCHEMA) == {"text": "ok"}
    response_format = seen["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"] == {
        "name": "magpie_response",
        "strict": True,
        "schema": SCHEMA,
    }


def test_structured_completion_accepts_fenced_json(monkeypatch):
    monkeypatch.setattr(
        providers, "_post",
        lambda *args, **kwargs: _response('```json\n{"text":"ok"}\n```'))
    assert configured().complete("p", schema=SCHEMA) == {"text": "ok"}


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[]",
        '{"other":"missing required"}',
        '{"text":7}',
        '{"text":"ok","surprise":true}',
    ],
)
def test_bad_structured_content_is_a_response_error(monkeypatch, content):
    monkeypatch.setattr(
        providers, "_post", lambda *args, **kwargs: _response(content))
    with pytest.raises(ProviderResponseError):
        configured().complete("p", schema=SCHEMA)


def test_missing_chat_content_is_a_response_error(monkeypatch):
    monkeypatch.setattr(
        providers,
        "_post",
        lambda *args, **kwargs: io.BytesIO(b'{"choices":[]}'),
    )
    with pytest.raises(ProviderResponseError):
        configured().complete("p")


# ------------------------------------------------------------ error classes


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 422])
def test_permanent_http_errors_are_configuration_errors(monkeypatch, status):
    def fail(url, body, headers, timeout):
        raise urllib.error.HTTPError(url, status, "bad request", {}, None)

    monkeypatch.setattr(providers, "_post", fail)
    with pytest.raises(ProviderConfigurationError):
        configured().complete("p")


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 503])
def test_retryable_http_errors_are_unavailable(monkeypatch, status):
    def fail(url, body, headers, timeout):
        raise urllib.error.HTTPError(url, status, "temporary", {}, None)

    monkeypatch.setattr(providers, "_post", fail)
    with pytest.raises(ProviderUnavailable) as exc:
        configured().complete("p")
    assert not isinstance(exc.value, ProviderConfigurationError)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        TimeoutError("late"),
        socket.timeout("late"),
    ],
)
def test_network_errors_are_retryable_unavailable(monkeypatch, error):
    monkeypatch.setattr(
        providers, "_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(ProviderUnavailable) as exc:
        configured().complete("p")
    assert not isinstance(exc.value, ProviderConfigurationError)


# ------------------------------------------------------------ chain contract


def test_chain_returns_result_and_provider_name():
    result, who = Chain([Fake("one", answer="answer")]).complete("p")
    assert (result, who) == ("answer", "one")


def test_chain_can_fall_through_transient_injected_fakes():
    first = Fake("one", fail=ProviderUnavailable("down"))
    second = Fake("two", answer="answer")
    assert Chain([first, second]).complete("p") == ("answer", "two")
    assert first.calls == second.calls == 1


def test_chain_does_not_hide_configuration_error():
    bad = Fake("bad", fail=ProviderConfigurationError("wrong model"))
    unused = Fake("unused", answer="must not run")
    with pytest.raises(ProviderConfigurationError):
        Chain([bad, unused]).complete("p")
    assert unused.calls == 0


def test_chain_passes_schema_and_timeout():
    seen = {}

    class Spy(Fake):
        def complete(self, prompt, schema=None, timeout=30):
            seen.update(schema=schema, timeout=timeout)
            return {"text": "ok"}

    assert Chain([Spy("spy")]).complete(
        "p", schema=SCHEMA, timeout=9) == ({"text": "ok"}, "spy")
    assert seen == {"schema": SCHEMA, "timeout": 9}


def test_chain_status_and_last_provider():
    chain = Chain([Fake("ready", answer="ok", up=True)])
    assert chain.status() == [
        {"name": "ready", "available": True, "last": False}]
    chain.complete("p")
    assert chain.status() == [
        {"name": "ready", "available": True, "last": True}]


def test_chain_reports_total_transient_failure():
    with pytest.raises(ProviderUnavailable) as exc:
        Chain([
            Fake("one", fail=ProviderUnavailable("down")),
            Fake("two", fail=ProviderUnavailable("busy")),
        ]).complete("p")
    assert "one" in str(exc.value) and "two" in str(exc.value)


def test_get_chain_is_a_singleton(monkeypatch):
    providers.reset_chain(None)
    monkeypatch.setenv("MAGPIE_LLM_STUB", "1")
    try:
        assert providers.get_chain() is providers.get_chain()
    finally:
        providers.reset_chain(None)


# ------------------------------------------------------------ explicit stub


def test_stub_text_is_marked_offline_and_deterministic():
    first = StubProvider().complete("rates will fall")
    second = StubProvider().complete("rates will fall")
    assert first == second
    assert "stub · offline" in first


def test_stub_shapes_worker_schema_without_echoing_instructions():
    schema = {
        "type": "object",
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "section"],
                    "properties": {
                        "text": {"type": "string"},
                        "section": {"type": "string"},
                    },
                },
            },
        },
    }
    output = StubProvider().complete(
        "Split this thought.\nTHOUGHT: coffee cools quickly", schema=schema)
    assert "coffee cools quickly" in output["claims"][0]["text"]
    assert "Split this thought" not in output["claims"][0]["text"]


def test_stub_echoes_the_json_prompt_body_atomize_actually_sends():
    """`atomize()` sends INPUT: {json}, not a `LABEL:` line.

    The stub is a *deterministic* double, which means distinct inputs must
    produce distinct outputs. When it could not find the payload it fell back
    to echoing the prompt preamble — identical for every thought — so every
    offline contribution atomized to the same text and collapsed onto one
    card. That made the offline scenario unable to hold two positions at once,
    and therefore unable to exercise anything about the ladder.
    """
    schema = {
        "type": "object",
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                },
            },
        },
    }
    stub = StubProvider()

    def atomize_prompt(thought):
        return (
            "Extract one or at most two useful artifacts.\n"
            "INPUT:\n"
            + json.dumps({"thought": thought, "sections": [],
                          "canonical_candidates": []})
        )

    first = stub.complete(atomize_prompt("coffee cools quickly"), schema=schema)
    second = stub.complete(atomize_prompt("kettles boil slowly"), schema=schema)

    assert "coffee cools quickly" in first["claims"][0]["text"]
    assert "kettles boil slowly" in second["claims"][0]["text"]
    assert first != second
    assert "Extract one or at most two" not in first["claims"][0]["text"]


def test_stub_still_answers_no_click_for_every_recognition():
    """§2.2 — the stub must not become a source of fabricated recognitions.

    Making the stub echo its input is a fixture repair; making it *click*
    would be a second channel into the field. `click` is a boolean, and the
    stub answers every boolean False, so no offline run can ever mint a frame.
    """
    from magpie.workers import _RECOGNIZE_SCHEMA

    result = StubProvider().complete(
        "OPEN QUESTION: q\nCLAIM A: alpha\nCLAIM B: beta",
        schema=_RECOGNIZE_SCHEMA,
    )

    assert result["click"] is False


def test_stub_honors_enum_number_boolean_and_null():
    schema = {
        "type": "object",
        "required": ["kind", "number", "enabled", "nothing"],
        "properties": {
            "kind": {"type": "string", "enum": ["A", "B"]},
            "number": {"type": "number"},
            "enabled": {"type": "boolean"},
            "nothing": {"type": "null"},
        },
    }
    assert StubProvider().complete("p", schema=schema) == {
        "kind": "A",
        "number": 0,
        "enabled": False,
        "nothing": None,
    }


def test_no_test_uses_live_network(monkeypatch):
    monkeypatch.setattr(
        providers.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live network attempted")),
    )
    assert Chain([StubProvider()]).complete("p")[1] == "stub"
