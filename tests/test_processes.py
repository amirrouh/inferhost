"""Tests for daemon-lifecycle helpers that don't need a live llama-swap."""
from __future__ import annotations

from inferhost.core import processes
from inferhost.core.registry import Model, Registry


def test_load_pinned_models_only_loads_pinned(monkeypatch):
    """load_pinned_models force-loads exactly the pinned models, in order, and
    skips the unpinned ones — this is what restores the pin contract after a
    daemon restart (llama-swap lazy-loads, so pins go cold across a restart)."""
    reg = Registry(models=[
        Model(name="alpha", repo_id="x", filename="a.gguf", port=8081, pin=True),
        Model(name="beta", repo_id="x", filename="b.gguf", port=8082, pin=False),
        Model(name="gamma", repo_id="x", filename="c.gguf", port=8083, pin=True),
    ])
    monkeypatch.setattr(processes.registry, "load", lambda: reg)

    asked: list[str] = []

    def fake_force_load(name, timeout=120.0):
        asked.append(name)
        return True

    monkeypatch.setattr(processes, "force_load_model", fake_force_load)

    loaded = processes.load_pinned_models()
    assert asked == ["alpha", "gamma"]  # pinned only, registry order
    assert loaded == ["alpha", "gamma"]


def test_load_pinned_models_omits_failures(monkeypatch):
    """A pinned model that fails to load (OOM / bad file) is absent from the
    returned list but never raises."""
    reg = Registry(models=[
        Model(name="ok", repo_id="x", filename="a.gguf", port=8081, pin=True),
        Model(name="oom", repo_id="x", filename="b.gguf", port=8082, pin=True),
    ])
    monkeypatch.setattr(processes.registry, "load", lambda: reg)
    monkeypatch.setattr(
        processes, "force_load_model",
        lambda name, timeout=120.0: name == "ok",
    )
    assert processes.load_pinned_models() == ["ok"]


def test_load_pinned_models_empty_when_none_pinned(monkeypatch):
    reg = Registry(models=[
        Model(name="a", repo_id="x", filename="a.gguf", port=8081, pin=False),
    ])
    monkeypatch.setattr(processes.registry, "load", lambda: reg)
    monkeypatch.setattr(
        processes, "force_load_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert processes.load_pinned_models() == []


# ---- autostart (systemd user unit) ----

def test_autostart_unit_text_wires_start_and_stop():
    """The generated unit must run `inferhost start` at boot and `inferhost
    stop` on shutdown, stay 'active' after the oneshot exits (RemainAfterExit),
    and hook into the user session's default.target so it fires under linger."""
    from inferhost import _ops

    text = _ops._unit_text("/home/me/.local/bin/inferhost")
    assert "ExecStart=/home/me/.local/bin/inferhost start" in text
    assert "ExecStop=/home/me/.local/bin/inferhost stop" in text
    assert "RemainAfterExit=yes" in text
    assert "WantedBy=default.target" in text


def test_autostart_rejects_unknown_action(capsys):
    from inferhost import _ops

    assert _ops._autostart(["bogus"]) == 2
    assert "usage:" in capsys.readouterr().err
