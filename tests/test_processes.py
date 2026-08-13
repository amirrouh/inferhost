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


# ---- binary update ----

def _stub_update(monkeypatch, *, swap_running: bool, calls: list[str]):
    """Wire _update's collaborators to a call recorder — no network, no daemons."""
    from inferhost import _ops
    from inferhost.core.binaries import InstalledBinary

    monkeypatch.setattr(
        _ops.processes, "swap_status",
        lambda: processes.DaemonStatus(
            name="llama-swap", running=swap_running, pid=1 if swap_running else None,
            port=9090, log_path=None,
        ),
    )
    monkeypatch.setattr(_ops.processes, "stop_all", lambda: calls.append("stop"))
    monkeypatch.setattr(_ops, "_start", lambda: (calls.append("start"), 0)[1])
    monkeypatch.setattr(_ops.binaries, "installed_llama_server_tag", lambda: "b10068")

    def fake_server(version=None, progress_cb=None):
        calls.append(f"install-server:{version}")
        return InstalledBinary(path=None, version="b10412")

    monkeypatch.setattr(_ops.binaries, "install_llama_server", fake_server)
    monkeypatch.setattr(
        _ops.binaries, "install_llama_swap",
        lambda version=None, progress_cb=None: (
            calls.append("install-swap"), InstalledBinary(path=None, version="v249"))[1],
    )
    # sd-server absent => chat-only box, nothing to refresh.
    monkeypatch.setattr(_ops.binaries, "needs_sdcpp_refresh", lambda: True)


def test_update_stops_daemons_before_swapping_binaries(monkeypatch, capsys):
    """The stop MUST precede the install: install_llama_server purges every
    lib*.so in bin_dir before extracting the replacement set, and doing that
    under a live llama-server leaves it running against deleted libraries."""
    from inferhost import _ops

    calls: list[str] = []
    _stub_update(monkeypatch, swap_running=True, calls=calls)

    assert _ops._update([]) == 0
    assert calls == ["stop", "install-server:None", "install-swap", "start"]
    assert "b10068 -> b10412" in capsys.readouterr().out


def test_update_leaves_stopped_daemons_stopped(monkeypatch):
    """Updating on a quiet box must not silently bring the stack up."""
    from inferhost import _ops

    calls: list[str] = []
    _stub_update(monkeypatch, swap_running=False, calls=calls)

    assert _ops._update([]) == 0
    assert "stop" not in calls and "start" not in calls


def test_update_passes_an_explicit_tag_through(monkeypatch):
    from inferhost import _ops

    calls: list[str] = []
    _stub_update(monkeypatch, swap_running=False, calls=calls)

    assert _ops._update(["b10353"]) == 0
    assert "install-server:b10353" in calls


def test_update_never_overwrites_a_custom_llama_server(monkeypatch, capsys):
    """Custom-binary mode: the user's own build is theirs to update. llama-swap
    is still refreshed — it's inferhost's binary either way.

    The report must name the gap between their build and upstream. Reporting a
    bare "skipped" is how a box sits 80 builds behind while the model fails
    with "unknown model architecture" and `update` still exits 0."""
    from inferhost import _ops
    from inferhost import settings as settings_mod

    calls: list[str] = []
    _stub_update(monkeypatch, swap_running=False, calls=calls)
    monkeypatch.setattr(
        _ops.binaries, "custom_llama_server_version", lambda: "version: 1 (7ba604f)")
    monkeypatch.setattr(_ops.binaries, "latest_llama_server_tag", lambda: "b10412")
    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", "/opt/custom/llama-server")
    settings_mod.reload_settings()
    try:
        assert _ops._update([]) == 0
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()
    assert calls == ["install-swap"]
    out = capsys.readouterr().out
    assert "skipped" in out
    assert "7ba604f" in out and "b10412" in out
    assert "unknown model architecture" in out


def test_update_custom_mode_survives_an_unreachable_github(monkeypatch, capsys):
    """The version lookup is a courtesy — it must never turn `update` into a
    failure when GitHub is down or the binary won't answer --version."""
    from inferhost import _ops
    from inferhost import settings as settings_mod

    calls: list[str] = []
    _stub_update(monkeypatch, swap_running=False, calls=calls)
    monkeypatch.setattr(_ops.binaries, "custom_llama_server_version", lambda: None)
    monkeypatch.setattr(_ops.binaries, "latest_llama_server_tag", lambda: None)
    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", "/opt/custom/llama-server")
    settings_mod.reload_settings()
    try:
        assert _ops._update([]) == 0
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()
    assert "your build     : unknown" in capsys.readouterr().out


# ---- gateway health ----

def _stub_gateway_running(monkeypatch, running: bool) -> None:
    monkeypatch.setattr(
        processes, "gateway_status",
        lambda: processes.DaemonStatus(
            name="litellm", running=running, pid=42 if running else None,
            port=9001, log_path=None,
        ),
    )


def test_gateway_serving_false_when_port_refuses(monkeypatch):
    """A live PID whose port refuses connections is NOT serving. This is the
    exact shape of the fastapi/litellm skew that left swap happily answering on
    :9090 while the gateway was dead — the dashboard used to call it green."""
    import httpx

    _stub_gateway_running(monkeypatch, True)

    def boom(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", boom)
    assert processes.gateway_serving() is False


def test_gateway_serving_true_on_auth_error(monkeypatch):
    """401 means litellm is up and routing, just guarded by a master key —
    that's serving, not broken."""
    import httpx

    _stub_gateway_running(monkeypatch, True)
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: httpx.Response(401, request=httpx.Request("GET", "http://x")),
    )
    assert processes.gateway_serving() is True


def test_gateway_serving_false_when_not_running(monkeypatch):
    """No PID, no probe — never report a stopped gateway as serving."""
    _stub_gateway_running(monkeypatch, False)
    assert processes.gateway_serving() is False


def test_gateway_death_reason_surfaces_the_exception(monkeypatch, tmp_path):
    """The startup error must name the actual cause, not 'check the log'.
    Under systemd autostart nobody reads the log, which is how a broken
    gateway stays broken for days."""
    log = tmp_path / "litellm.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        '  File "/x/proxy_cli.py", line 935, in run_server\n'
        "    from .proxy_server import app\n"
        "ImportError: cannot import name 'get_flat_dependant' from 'fastapi'\n"
    )
    monkeypatch.setattr(processes.paths, "gateway_log_path", lambda: log)
    reason = processes._gateway_death_reason()
    assert "get_flat_dependant" in reason
    assert not reason.startswith("File")


def test_gateway_death_reason_handles_empty_log(monkeypatch, tmp_path):
    log = tmp_path / "litellm.log"
    log.write_text("")
    monkeypatch.setattr(processes.paths, "gateway_log_path", lambda: log)
    assert processes._gateway_death_reason() == "no output in the log"


# ---- self-heal: no binary here knows a registered model's architecture ----

def _heal_env(monkeypatch, *, arch_supported: bool, installed: str | None,
              latest: str | None, fetched: list):
    from inferhost import _ops

    monkeypatch.setattr(_ops.gguf, "architecture_cached", lambda _p: "muse-glimmer")
    monkeypatch.setattr(
        _ops.binaries, "binary_supports_arch", lambda _e, _a: arch_supported)
    monkeypatch.setattr(_ops.binaries, "managed_llama_server_tag", lambda: installed)
    monkeypatch.setattr(_ops.binaries, "latest_llama_server_tag", lambda: latest)
    monkeypatch.setattr(
        _ops.binaries, "install_managed_llama_server",
        lambda progress_cb=None: fetched.append("fetched"))


def _one_chat_model():
    return Registry(models=[
        Model(name="muse", repo_id="x", filename="m.gguf", local_path="/m.gguf"),
    ])


def test_heal_fetches_a_newer_llama_cpp_for_an_unknown_architecture(monkeypatch, capsys):
    """The whole point: a model released after the binary was installed must
    start working from an upgrade + restart, with no hand-editing of files."""
    from inferhost import _ops

    fetched: list = []
    _heal_env(monkeypatch, arch_supported=False, installed="b10331",
              latest="b10412", fetched=fetched)
    _ops._heal_unknown_architectures(_one_chat_model())
    assert fetched == ["fetched"]
    assert "b10412" in capsys.readouterr().err


def test_heal_does_nothing_when_the_binary_already_knows_the_architecture(monkeypatch):
    from inferhost import _ops

    fetched: list = []
    _heal_env(monkeypatch, arch_supported=True, installed="b10331",
              latest="b10412", fetched=fetched)
    _ops._heal_unknown_architectures(_one_chat_model())
    assert fetched == []


def test_heal_does_not_redownload_when_already_on_the_newest_build(monkeypatch, capsys):
    """Upstream simply may not support the model yet. Re-downloading the same
    build on every start would be an infinite, pointless loop."""
    from inferhost import _ops

    fetched: list = []
    _heal_env(monkeypatch, arch_supported=False, installed="b10412",
              latest="b10412", fetched=fetched)
    _ops._heal_unknown_architectures(_one_chat_model())
    assert fetched == []
    assert "doesn't support this architecture yet" in capsys.readouterr().err


def test_heal_skips_models_llama_server_never_serves(monkeypatch):
    """TTS and image models run on other engines — their files aren't even
    llama.cpp GGUFs, so they must not trigger a llama.cpp download."""
    from inferhost import _ops

    fetched: list = []
    _heal_env(monkeypatch, arch_supported=False, installed="b10331",
              latest="b10412", fetched=fetched)
    reg = Registry(models=[
        Model(name="kokoro", repo_id="x", filename="k.onnx", vocoder_path="/v.npz"),
        Model(name="flux", repo_id="x", filename="f.gguf", kind="image"),
    ])
    _ops._heal_unknown_architectures(reg)
    assert fetched == []


def test_heal_survives_an_unreachable_upstream(monkeypatch, capsys):
    from inferhost import _ops

    fetched: list = []
    _heal_env(monkeypatch, arch_supported=False, installed="b10331",
              latest=None, fetched=fetched)
    _ops._heal_unknown_architectures(_one_chat_model())
    assert fetched == []
    assert "unreachable" in capsys.readouterr().err


# ---- prune (reclaiming weights stranded by older versions) ----

def _prune_cache(monkeypatch, tmp_path, repos: dict, registered: list):
    """Build a fake HF cache and registry for _prune to walk."""
    from inferhost import _ops

    hub = tmp_path / "hub"
    for repo, size in repos.items():
        d = hub / ("models--" + repo.replace("/", "--")) / "blobs"
        d.mkdir(parents=True, exist_ok=True)
        (d / "sha").write_bytes(b"\0" * size)
    monkeypatch.setattr(_ops, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        _ops.registry, "load",
        lambda: Registry(models=[
            Model(name=f"m{i}", repo_id=r, filename="f.gguf")
            for i, r in enumerate(registered)
        ]))
    return hub


def test_prune_lists_without_deleting_by_default(monkeypatch, tmp_path, capsys):
    """The cache is shared with other tools, so the default must never delete."""
    from inferhost import _ops

    hub = _prune_cache(monkeypatch, tmp_path,
                       {"org/stale": 2048, "org/live": 1024}, ["org/live"])
    assert _ops._prune([]) == 0
    out = capsys.readouterr().out
    assert "org/stale" in out and "org/live" not in out
    assert (hub / "models--org--stale").exists()


def test_prune_deletes_only_the_named_repos(monkeypatch, tmp_path):
    """"Unused by inferhost" is not "unused" — on a box also running vLLM the
    difference is somebody else's model, so naming repos must be possible."""
    from inferhost import _ops

    hub = _prune_cache(monkeypatch, tmp_path,
                       {"org/mine": 2048, "vllm/theirs": 4096}, [])
    assert _ops._prune(["--yes", "org/mine"]) == 0
    assert not (hub / "models--org--mine").exists()
    assert (hub / "models--vllm--theirs").exists()


def test_prune_deletes_everything_listed_with_bare_yes(monkeypatch, tmp_path):
    from inferhost import _ops

    hub = _prune_cache(monkeypatch, tmp_path,
                       {"org/a": 1024, "org/b": 2048, "org/keep": 512},
                       ["org/keep"])
    assert _ops._prune(["--yes"]) == 0
    assert not (hub / "models--org--a").exists()
    assert not (hub / "models--org--b").exists()
    assert (hub / "models--org--keep").exists()


def test_prune_refuses_a_repo_that_is_in_use(monkeypatch, tmp_path, capsys):
    """Naming a registered model's repo must fail loudly, not delete it."""
    from inferhost import _ops

    hub = _prune_cache(monkeypatch, tmp_path,
                       {"org/live": 1024, "org/stale": 512}, ["org/live"])
    assert _ops._prune(["--yes", "org/live"]) == 1
    assert (hub / "models--org--live").exists()
    assert "not prunable" in capsys.readouterr().err
