"""Smoke tests for the TUI.

These verify that the dashboard screen mounts without crashing — catching
the kind of bug that pytest-on-pure-modules would miss (NameErrors in
on_mount, broken CSS selectors, subprocess paths that throw without a
catch). Uses Textual's App.run_test() Pilot harness — no real terminal.

Scope is deliberately tight: import + mount + a few invariant checks on
methods that were the most recent source of bugs. Not a full UI driver.
"""
from __future__ import annotations

import subprocess

import pytest

from inferhost.core.registry import Model, Registry
from inferhost.settings import reload_settings


@pytest.fixture
def hermetic_tmp(tmp_path, monkeypatch):
    """Point inferhost dirs at tmp so the TUI doesn't read the user's real
    registry / logs / models. Without this the dashboard load_registry call
    sees the real on-disk state and the test isn't reproducible."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_HF_CACHE", str(tmp_path / "hf"))
    # Ensure no tmux env bleeds in from the test runner — _maybe_warn_tmux_mouse
    # would then shell out to a real tmux binary which is non-hermetic.
    monkeypatch.delenv("TMUX", raising=False)
    reload_settings()
    return tmp_path


@pytest.mark.asyncio
async def test_dashboard_mounts_without_crashing(hermetic_tmp):
    """Catch import / on_mount / CSS errors that ruff and unit tests miss."""
    import inferhost.tui.app as app_mod
    from inferhost.tui.app import InferhostApp

    # Patch _binaries_present so the app skips InstallScreen and lands
    # straight on the dashboard. Without this we'd need to fake the
    # llama-server binary on disk.
    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            # Let on_mount and the first tick run.
            await pilot.pause(0.05)
            # Splash is pushed on top — it auto-dismisses after 1s but we
            # don't need to wait; reaching here means compose + on_mount
            # for both Splash and Dashboard didn't raise.
            assert app.screen is not None
    finally:
        app_mod._binaries_present = orig


@pytest.mark.asyncio
async def test_tmux_warn_path_handles_missing_tmux_binary(hermetic_tmp, monkeypatch):
    """If $TMUX is set but no tmux binary is on PATH, we must not crash."""
    monkeypatch.setenv("TMUX", "/tmp/fake-tmux-socket,12345,0")

    # Replace subprocess.run with one that raises FileNotFoundError to simulate
    # 'tmux' not being installed even though $TMUX is set (rare but possible
    # in stripped-down containers or weird shells).
    real_run = subprocess.run

    def fake_run(*args, **kwargs):
        if args and isinstance(args[0], list) and args[0] and args[0][0] == "tmux":
            raise FileNotFoundError("tmux not on PATH")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    import inferhost.tui.app as app_mod
    from inferhost.tui.app import InferhostApp

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            # If _maybe_warn_tmux_mouse didn't swallow FileNotFoundError, the
            # event loop would have crashed and run_test would have raised.
    finally:
        app_mod._binaries_present = orig


def test_pinned_overflow_silent_for_single_pinned_model(hermetic_tmp):
    """The bug the user hit: a single pinned model should never trigger
    the 'pinned overflow' advisory — there's no second pinned model to
    coexist with, so the message is meaningless."""
    from inferhost.core import registry
    from inferhost.tui.screens.dashboard import DashboardScreen

    reg = Registry(models=[
        # Deliberately oversize for any GPU — proves the gate is "count of
        # pinned models", not "weight total > VRAM".
        Model(name="huge", repo_id="x/y", filename="y.gguf", port=8081,
              size_gib=200.0, local_path="/tmp/y.gguf", pin=True),
    ])
    registry.save(reg)

    # Stand up a bare DashboardScreen to call the method directly — full
    # mount isn't needed for a pure-logic check, and dodging the mount avoids
    # nvidia-smi being called on a CI box without a GPU.
    d = DashboardScreen()
    d._gpus = [type("G", (), {"mem_total_mib": 24576, "mem_used_mib": 0,
                              "util_pct": 0, "index": 0})()]
    assert d._pinned_overflow_warning() == ""


def test_pinned_overflow_fires_for_real_overflow(hermetic_tmp):
    """The advisory must still fire when 2+ pinned models actually overflow."""
    from inferhost.core import registry
    from inferhost.tui.screens.dashboard import DashboardScreen

    reg = Registry(models=[
        Model(name="a", repo_id="x/a", filename="a.gguf", port=8081,
              size_gib=20.0, local_path="/tmp/a.gguf", pin=True),
        Model(name="b", repo_id="x/b", filename="b.gguf", port=8082,
              size_gib=20.0, local_path="/tmp/b.gguf", pin=True),
    ])
    registry.save(reg)

    d = DashboardScreen()
    d._gpus = [type("G", (), {"mem_total_mib": 24576, "mem_used_mib": 0,
                              "util_pct": 0, "index": 0})()]
    msg = d._pinned_overflow_warning()
    assert "pinned weights total" in msg
    # No more red-flag wording — must read as advisory.
    assert "OVERFLOW" not in msg
