"""Portable server configuration."""

from pathlib import Path

from magpie import server


def test_runtime_dir_prefers_explicit_configuration(monkeypatch):
    monkeypatch.setenv("MAGPIE_RUNTIME_DIR", "~/custom-magpie-state")
    monkeypatch.setenv("XDG_STATE_HOME", "/ignored")

    assert server._runtime_dir() == Path.home() / "custom-magpie-state"


def test_runtime_dir_uses_xdg_state_home(monkeypatch):
    monkeypatch.delenv("MAGPIE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", "/portable/state")

    assert server._runtime_dir() == Path("/portable/state/magpie")


def test_runtime_dir_has_a_user_relative_fallback(monkeypatch):
    monkeypatch.delenv("MAGPIE_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    assert server._runtime_dir() == Path.home() / ".local" / "state" / "magpie"
