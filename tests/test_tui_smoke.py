"""Smoke tests for the TUI.

These verify that the dashboard screen mounts without crashing — catching
the kind of bug that pytest-on-pure-modules would miss (NameErrors in
on_mount, broken CSS selectors, subprocess paths that throw without a
catch). Uses Textual's App.run_test() Pilot harness — no real terminal.

Scope is deliberately tight: import + mount + a few invariant checks on
methods that were the most recent source of bugs. Not a full UI driver.
"""
from __future__ import annotations

import inspect
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


# ---- A1: delete confirmation ----

@pytest.mark.asyncio
async def test_delete_confirm_cancel_keeps_model(hermetic_tmp):
    """Pressing 'd' must gate on a WarningScreen confirm — Escape (cancel)
    leaves the model registered."""
    import inferhost.tui.app as app_mod
    from inferhost.core import registry
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.warning import WarningScreen

    reg = Registry(models=[
        Model(name="keep-me", repo_id="x/y", filename="y.gguf", port=8081,
              size_gib=1.0, local_path="/tmp/y.gguf"),
    ])
    registry.save(reg)

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            # SplashScreen sits on top for ~1s before auto-dismissing into the
            # dashboard — wait it out so 'd' reaches DashboardScreen.
            await pilot.pause(1.1)
            await pilot.press("d")
            await pilot.pause(0.05)
            assert isinstance(app.screen, WarningScreen)
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert registry.load().get("keep-me") is not None
    finally:
        app_mod._binaries_present = orig


@pytest.mark.asyncio
async def test_delete_confirm_confirm_removes_model(hermetic_tmp):
    """Confirming the WarningScreen (clicking its Delete button) actually
    removes the model from the registry — the same action key 'd', the
    Delete key, and the Delete button all funnel through."""
    import inferhost.tui.app as app_mod
    from inferhost.core import registry
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.warning import WarningScreen

    reg = Registry(models=[
        Model(name="remove-me", repo_id="x/y", filename="y.gguf", port=8081,
              size_gib=1.0, local_path="/tmp/y.gguf"),
    ])
    registry.save(reg)

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(1.1)  # let SplashScreen auto-dismiss first
            await pilot.press("d")
            await pilot.pause(0.05)
            assert isinstance(app.screen, WarningScreen)
            await pilot.click("#confirm")
            await pilot.pause(0.1)
            assert registry.load().get("remove-me") is None
    finally:
        app_mod._binaries_present = orig


# ---- A2: add-model modal never blocks on the daemon reload itself ----

def test_add_model_module_never_calls_reload_and_warm_pinned():
    """Root-cause regression guard for the modal-freeze bug: AddModelScreen's
    job ends at file-on-disk + registry saved. The (tens-of-seconds) daemon
    reload must live ONLY in the dashboard's _after_add callback."""
    import inferhost.tui.screens.add_model as add_model_mod

    assert "reload_and_warm_pinned" not in inspect.getsource(add_model_mod)


def test_after_add_schedules_reload_worker(hermetic_tmp, monkeypatch):
    from inferhost.tui.screens.dashboard import DashboardScreen

    d = DashboardScreen()
    notified = []
    monkeypatch.setattr(d, "notify", lambda *a, **k: notified.append(a))
    refreshed = []
    monkeypatch.setattr(d, "refresh_models", lambda: refreshed.append(True))
    scheduled = []
    monkeypatch.setattr(d, "run_worker", lambda fn, **kw: scheduled.append((fn, kw)))

    d._after_add(True)

    assert notified, "must notify the user a reload is happening"
    assert refreshed, "must refresh the sidebar immediately"
    assert len(scheduled) == 1
    fn, kw = scheduled[0]
    assert kw == {"thread": True, "exclusive": False}
    assert callable(fn)


def test_after_add_noop_when_dismissed_without_adding(hermetic_tmp, monkeypatch):
    from inferhost.tui.screens.dashboard import DashboardScreen

    d = DashboardScreen()
    calls = []
    monkeypatch.setattr(d, "notify", lambda *a, **k: calls.append("notify"))
    monkeypatch.setattr(d, "refresh_models", lambda: calls.append("refresh"))
    monkeypatch.setattr(d, "run_worker", lambda *a, **k: calls.append("worker"))

    d._after_add(False)
    d._after_add(None)

    assert calls == []


# ---- A5: error surfacing ----

def test_load_failed_toast_includes_err_log_tail(hermetic_tmp, monkeypatch):
    """force_load_model failure must enrich the toast with the model's
    err.log tail, not just a generic 'check the log panel' message."""
    from inferhost.core import paths, processes
    from inferhost.tui.screens.dashboard import DashboardScreen

    paths.ensure_dirs()
    (paths.logs_dir() / "flaky.err.log").write_text(
        "CUDA error: out of memory\n", encoding="utf-8"
    )
    monkeypatch.setattr(processes, "force_load_model", lambda name, timeout=30.0: False)

    d = DashboardScreen()

    class _FakeApp:
        def call_from_thread(self, fn, *a, **kw):
            fn(*a, **kw)

    monkeypatch.setattr(type(d), "app", property(lambda self: _FakeApp()))
    notified = []
    monkeypatch.setattr(d, "notify", lambda msg, **kw: notified.append((msg, kw)))
    monkeypatch.setattr(d, "run_worker", lambda *a, **k: None)

    d._do_load_and_refresh("flaky")

    assert notified
    msg, kw = notified[0]
    assert "CUDA error: out of memory" in msg
    assert kw.get("severity") == "error"


def test_load_failed_toast_falls_back_when_no_err_log(hermetic_tmp, monkeypatch):
    """No err.log content at all — must still give a helpful (non-crashing)
    fallback message rather than an empty enrichment."""
    from inferhost.core import processes
    from inferhost.tui.screens.dashboard import DashboardScreen

    monkeypatch.setattr(processes, "force_load_model", lambda name, timeout=30.0: False)

    d = DashboardScreen()

    class _FakeApp:
        def call_from_thread(self, fn, *a, **kw):
            fn(*a, **kw)

    monkeypatch.setattr(type(d), "app", property(lambda self: _FakeApp()))
    notified = []
    monkeypatch.setattr(d, "notify", lambda msg, **kw: notified.append((msg, kw)))
    monkeypatch.setattr(d, "run_worker", lambda *a, **k: None)

    d._do_load_and_refresh("no-log-model")

    assert notified
    msg, kw = notified[0]
    assert "no-log-model" in msg
    assert kw.get("severity") == "error"


# ---- A6: settings auto-restart ----

def test_after_settings_auto_restarts_when_swap_running(hermetic_tmp, monkeypatch):
    from inferhost.core import processes
    from inferhost.tui.screens.dashboard import DashboardScreen

    monkeypatch.setattr(
        processes, "swap_status", lambda: type("S", (), {"running": True})()
    )
    d = DashboardScreen()
    notified = []
    monkeypatch.setattr(d, "notify", lambda *a, **k: notified.append(a[0] if a else ""))
    scheduled = []
    monkeypatch.setattr(d, "run_worker", lambda fn, **kw: scheduled.append((fn, kw)))
    monkeypatch.setattr(d, "_refresh_bars", lambda: None)

    d._after_settings(True)

    assert any("restarting daemons" in n for n in notified)
    assert len(scheduled) == 1
    assert scheduled[0][1] == {"thread": True, "exclusive": False}


def test_after_settings_plain_save_when_swap_not_running(hermetic_tmp, monkeypatch):
    from inferhost.core import processes
    from inferhost.tui.screens.dashboard import DashboardScreen

    monkeypatch.setattr(
        processes, "swap_status", lambda: type("S", (), {"running": False})()
    )
    d = DashboardScreen()
    notified = []
    monkeypatch.setattr(d, "notify", lambda *a, **k: notified.append(a[0] if a else ""))
    scheduled = []
    monkeypatch.setattr(d, "run_worker", lambda fn, **kw: scheduled.append((fn, kw)))
    monkeypatch.setattr(d, "_refresh_bars", lambda: None)

    d._after_settings(True)

    assert notified == ["Settings saved."]
    assert scheduled == []  # manual 'r' binding is still the only restart path


def test_after_settings_noop_when_not_saved(hermetic_tmp, monkeypatch):
    from inferhost.tui.screens.dashboard import DashboardScreen

    d = DashboardScreen()
    calls = []
    monkeypatch.setattr(d, "notify", lambda *a, **k: calls.append("notify"))
    monkeypatch.setattr(d, "run_worker", lambda *a, **k: calls.append("worker"))

    d._after_settings(False)
    d._after_settings(None)

    assert calls == []


# ---- B8: DFlash TUI wiring ----

def test_model_row_shows_bolt_when_draft_attached(hermetic_tmp):
    """A chat model with a draft attached gets a ⚡ tag in the sidebar row."""
    from inferhost.tui.screens.dashboard import DashboardScreen

    d = DashboardScreen()
    d._model_states = {}
    plain = Model(name="a", repo_id="x/y", filename="a.gguf")
    drafted = Model(name="b", repo_id="x/y", filename="b.gguf",
                    draft_model_path="/m/draft.gguf")
    assert "⚡" not in d._model_row(plain)
    assert "⚡" in d._model_row(drafted)


def test_enable_dflash_noops_with_notice_when_no_pairing(hermetic_tmp, monkeypatch):
    """`f` on a model with no known pairing must warn, not schedule a worker."""
    from inferhost.core import registry
    from inferhost.tui.screens.dashboard import DashboardScreen

    reg = Registry(models=[
        Model(name="llama", repo_id="meta-llama/Llama-3.1-8B", filename="l.gguf",
              port=8081, local_path="/tmp/l.gguf"),
    ])
    registry.save(reg)

    d = DashboardScreen()
    d.selected_name = "llama"
    notified = []
    monkeypatch.setattr(d, "notify", lambda *a, **k: notified.append((a, k)))
    scheduled = []
    monkeypatch.setattr(d, "run_worker", lambda *a, **k: scheduled.append((a, k)))

    d.action_enable_dflash()

    assert scheduled == []  # no download worker
    assert any(kw.get("severity") == "warning" for _a, kw in notified)


def test_enable_dflash_schedules_worker_when_pairing_matches(hermetic_tmp, monkeypatch):
    """`f` on a paired model (Qwen3.6-27B) schedules the off-thread fetch worker."""
    from inferhost.core import registry
    from inferhost.tui.screens.dashboard import DashboardScreen

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/Qwen3.6-27B", filename="q.gguf",
              port=8081, local_path="/tmp/q.gguf"),
    ])
    registry.save(reg)

    d = DashboardScreen()
    d.selected_name = "qwen"
    monkeypatch.setattr(d, "notify", lambda *a, **k: None)
    scheduled = []
    monkeypatch.setattr(d, "run_worker", lambda fn, **kw: scheduled.append((fn, kw)))

    d.action_enable_dflash()

    assert len(scheduled) == 1
    _fn, kw = scheduled[0]
    assert kw == {"thread": True, "exclusive": False}


def test_enable_dflash_noops_when_draft_already_attached(hermetic_tmp, monkeypatch):
    """`f` on a model that already has a draft doesn't re-download."""
    from inferhost.core import registry
    from inferhost.tui.screens.dashboard import DashboardScreen

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/Qwen3.6-27B", filename="q.gguf",
              port=8081, local_path="/tmp/q.gguf",
              draft_model_path="/m/draft.gguf", draft_repo_id="a/b"),
    ])
    registry.save(reg)

    d = DashboardScreen()
    d.selected_name = "qwen"
    monkeypatch.setattr(d, "notify", lambda *a, **k: None)
    scheduled = []
    monkeypatch.setattr(d, "run_worker", lambda *a, **k: scheduled.append(a))

    d.action_enable_dflash()

    assert scheduled == []


def test_attach_draft_writes_fields_and_configs(hermetic_tmp):
    """draft_picker.attach_draft persists the three draft fields + re-renders."""
    from inferhost.core import hf, paths, registry
    from inferhost.tui.screens.draft_picker import attach_draft

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/Qwen3.6-27B", filename="q.gguf",
              port=8081, local_path="/tmp/q.gguf"),
    ])
    registry.save(reg)

    pick = hf.GgufFile(repo_id="a/b-DFlash-GGUF", filename="draft-Q4_K_M.gguf",
                       size_bytes=int(0.9 * 1024**3), quant="Q4_K_M")
    attach_draft("qwen", pick, "/m/draft.gguf")

    m = registry.load().get("qwen")
    assert m.draft_model_path == "/m/draft.gguf"
    assert m.draft_repo_id == "a/b-DFlash-GGUF"
    assert m.draft_size_gib == pick.size_gib
    # write_all ran -> the swap config exists on disk.
    assert paths.llama_swap_config_path().exists()


@pytest.mark.asyncio
async def test_draft_picker_redirects_raw_safetensors_repo_to_paired_gguf(hermetic_tmp, monkeypatch):
    """Pasting the official z-lab safetensors draft repo (no GGUFs) into the
    picker must auto-redirect to the known paired GGUF conversion: populate
    the table from it, update the repo Input, and explain what happened."""
    from textual.widgets import Input, Static

    import inferhost.tui.app as app_mod
    from inferhost.core import hf
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.draft_picker import DraftPickerScreen

    suggested_files = [
        hf.GgufFile(
            repo_id="AtomicChat/Qwen3.5-27B-DFlash-GGUF",
            filename="draft-Q4_K_M.gguf",
            size_bytes=int(0.9 * 1024**3),
            quant="Q4_K_M",
        ),
    ]

    def fake_list_ggufs(repo_id: str):
        if repo_id == "z-lab/Qwen3.5-27B-DFlash":
            return []
        if repo_id == "AtomicChat/Qwen3.5-27B-DFlash-GGUF":
            return suggested_files
        raise AssertionError(f"unexpected repo id fetched: {repo_id}")

    monkeypatch.setattr(hf, "list_ggufs", fake_list_ggufs)

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(1.1)  # let SplashScreen auto-dismiss first
            screen = DraftPickerScreen("qwen")
            await app.push_screen(screen)
            await pilot.pause(0.1)

            screen._fetch("z-lab/Qwen3.5-27B-DFlash")
            await app.workers.wait_for_complete()
            await pilot.pause(0.1)

            assert [f.repo_id for f in screen.files] == ["AtomicChat/Qwen3.5-27B-DFlash-GGUF"]
            assert (
                screen.query_one("#repo-input", Input).value
                == "AtomicChat/Qwen3.5-27B-DFlash-GGUF"
            )
            hint = str(screen.query_one("#hint", Static).content)
            assert "z-lab/Qwen3.5-27B-DFlash" in hint
            assert "AtomicChat/Qwen3.5-27B-DFlash-GGUF" in hint
    finally:
        app_mod._binaries_present = orig


@pytest.mark.asyncio
async def test_draft_picker_no_suggestion_shows_explanatory_hint(hermetic_tmp, monkeypatch):
    """An unrecognized repo with no GGUF files gets a prominent, explanatory
    hint (not the old one-line 'no GGUF draft files found')."""
    from textual.widgets import Static

    import inferhost.tui.app as app_mod
    from inferhost.core import hf
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.draft_picker import DraftPickerScreen

    monkeypatch.setattr(hf, "list_ggufs", lambda repo_id: [])

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(1.1)
            screen = DraftPickerScreen("qwen")
            await app.push_screen(screen)
            await pilot.pause(0.1)

            screen._fetch("someone/random-repo")
            await app.workers.wait_for_complete()
            await pilot.pause(0.1)

            assert screen.files == []
            hint = str(screen.query_one("#hint", Static).content)
            assert "No GGUF files in that repo" in hint
    finally:
        app_mod._binaries_present = orig


@pytest.mark.asyncio
async def test_model_settings_draft_section_present_for_paired_chat_model(hermetic_tmp):
    """ModelSettingsScreen shows the DFlash draft section (summary + Suggest for
    a paired model) for a chat model."""
    from textual.widgets import Button, Static

    import inferhost.tui.app as app_mod
    from inferhost.core import registry
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.model_settings import ModelSettingsScreen

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/Qwen3.6-27B", filename="q.gguf",
              port=8081, local_path="/tmp/q.gguf"),
    ])
    registry.save(reg)

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(1.1)
            screen = ModelSettingsScreen("qwen")
            await app.push_screen(screen)
            await pilot.pause(0.1)
            # Summary Static + Suggest button (pairing matched) both present.
            assert app.screen.query_one("#draft-summary", Static) is not None
            assert app.screen.query_one("#draft-suggest", Button) is not None
            assert app.screen.query_one("#draft-browse", Button) is not None
    finally:
        app_mod._binaries_present = orig


@pytest.mark.asyncio
async def test_model_settings_vision_toggle_only_for_vision_models(hermetic_tmp):
    """ModelSettingsScreen shows the Vision/image-input toggle only when the
    model has an mmproj attached, prefilled from vision_enabled."""
    from textual.widgets import Input

    import inferhost.tui.app as app_mod
    from inferhost.core import registry
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.model_settings import ModelSettingsScreen

    reg = Registry(models=[
        Model(name="vl", repo_id="x/y", filename="vl.gguf", port=8081,
              local_path="/tmp/vl.gguf", mmproj_path="/tmp/mmproj.gguf",
              vision_enabled=False),
        Model(name="plain", repo_id="x/y", filename="p.gguf", port=8082,
              local_path="/tmp/p.gguf"),
    ])
    registry.save(reg)

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(1.1)
            screen = ModelSettingsScreen("vl")
            await app.push_screen(screen)
            await pilot.pause(0.1)
            field = app.screen.query_one("#f-vision", Input)
            assert field.value == "no"  # prefilled from vision_enabled=False
            app.pop_screen()
            await pilot.pause(0.1)

            screen = ModelSettingsScreen("plain")
            await app.push_screen(screen)
            await pilot.pause(0.1)
            assert not app.screen.query("#f-vision")  # text-only: no toggle
    finally:
        app_mod._binaries_present = orig


# ---- A7: TTS add flow wiring ----

@pytest.mark.asyncio
async def test_add_model_screen_tts_radio_wiring(hermetic_tmp):
    """The add-model modal exposes a 'Text-to-speech' kind option, and
    selecting it flips AddModelScreen.kind so _fetch/_download_and_register
    route to the TTS list/registration path."""
    from textual.widgets import RadioButton

    import inferhost.tui.app as app_mod
    from inferhost.tui.app import InferhostApp
    from inferhost.tui.screens.add_model import AddModelScreen

    orig = app_mod._binaries_present
    app_mod._binaries_present = lambda: True
    try:
        app = InferhostApp()
        async with app.run_test() as pilot:
            await pilot.pause(1.1)  # let SplashScreen auto-dismiss first
            screen = AddModelScreen()
            await app.push_screen(screen)
            await pilot.pause(0.1)
            btn = app.screen.query_one("#kind-tts", RadioButton)
            assert "Text-to-speech" in str(btn.label)
            assert screen.kind == "chat"  # default before any selection
            await pilot.click("#kind-tts")
            await pilot.pause(0.1)
            assert screen.kind == "tts"
    finally:
        app_mod._binaries_present = orig
