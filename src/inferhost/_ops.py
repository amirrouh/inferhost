"""Headless ops for inferhost: start / stop / restart / status — no TUI required.

The user-facing entry point is the ``inferhost`` TUI, but the daemons (llama-swap
and the LiteLLM gateway) outlive the TUI process because they are spawned via
``start_new_session=True``. This module lets you control them without ever
opening the TUI — useful for servers, systemd units, and shell scripts.

Console-script: ``inferhost-ops {start|stop|restart|status|autostart}``.
"""
from __future__ import annotations

import getpass
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from inferhost.core import binaries, configs, gguf, hf, paths, processes, quant, registry
from inferhost.settings import settings


def _start() -> int:
    """Start llama-swap and the LiteLLM gateway as detached background daemons.

    Both processes survive this command and stay running until ``stop`` is
    called or you kill them by PID. Already-running daemons are left alone.
    """
    reg = registry.load()
    if not reg.models:
        print("no models registered — run `inferhost` to add one first.",
              file=sys.stderr)
        return 1
    # Upgrade-path safety: if the on-disk llama-server came from a different
    # source than what this build of inferhost expects, fetch fresh before
    # configs reference it. No-op when the source already matches.
    if binaries.needs_llama_server_refresh():
        print("inferhost: refreshing llama-server (upstream source changed) ...",
              file=sys.stderr)
        try:
            binaries.install_llama_server()
        except Exception as e:  # noqa: BLE001
            print(f"inferhost: llama-server refresh failed: {e}", file=sys.stderr)
            return 1
    # Bundle sd-server for image generation if it's missing (e.g. an install
    # that predates image support). Non-fatal — chat/TTS still work without it;
    # only image models would fail to load until it's present.
    if any(m.kind == "image" for m in reg.models) and binaries.needs_sdcpp_refresh():
        print("inferhost: fetching stable-diffusion.cpp (sd-server) for image generation ...",
              file=sys.stderr)
        try:
            binaries.install_stable_diffusion()
        except Exception as e:  # noqa: BLE001
            print(f"inferhost: sd-server fetch failed: {e}", file=sys.stderr)
    _heal_unknown_architectures(reg)
    # Regenerate configs in case the registry was edited since last launch.
    configs.write_all(reg)
    for note in configs.consume_notices():
        print(f"notice: {note}", file=sys.stderr)
    swap_started = False
    if not processes.swap_status().running:
        processes.start_swap()
        swap_started = True
    gw_started = False
    gw_error = ""
    if processes.gateway_available() and not processes.gateway_status().running:
        # Non-fatal: llama-swap is the inference path, the gateway is the front
        # door. A gateway that won't start used to abort the whole command with
        # a traceback, which under systemd autostart meant swap came up, the
        # failure went to a journal nobody reads, and the box looked healthy
        # while port 9001 refused every connection. Report it loudly instead.
        try:
            processes.start_gateway()
            gw_started = True
        except Exception as e:  # noqa: BLE001
            gw_error = str(e)
            print(f"inferhost: gateway failed to start: {e}", file=sys.stderr)
    # The TTS daemon only runs when there's a TTS model to serve.
    tts_started = False
    if processes.has_tts_models() and not processes.tts_status().running:
        processes.start_tts()
        tts_started = True
    # Re-read for accurate post-start status.
    swap = processes.swap_status()
    gw = processes.gateway_status()
    swap_note = " (just started)" if swap_started else ""
    gw_note = " (just started)" if gw_started else ""
    print(f"llama-swap : {'running' if swap.running else 'NOT running'}  "
          f"pid={swap.pid or '-'}  port={swap.port}{swap_note}")
    pw = processes.pinwatch_status()
    print(f"pinwatch   : {'running' if pw.running else 'NOT running'}  "
          f"pid={pw.pid or '-'}  (keeps pinned models in VRAM)")
    if gw.running and not processes.gateway_serving():
        gw_note = "  ⚠ not answering — see " + str(gw.log_path)
    elif not gw.running and gw_error:
        gw_note = "  ⚠ " + gw_error.splitlines()[0]
    print(f"litellm    : {'running' if gw.running else 'NOT running'}  "
          f"pid={gw.pid or '-'}  port={gw.port}{gw_note}")
    if processes.has_tts_models():
        tts = processes.tts_status()
        tts_note = " (just started)" if tts_started else ""
        print(f"inferhost-tts : {'running' if tts.running else 'NOT running'}  "
              f"pid={tts.pid or '-'}  port={tts.port}{tts_note}")
    return 0 if swap.running else 1


def _stop() -> int:
    processes.stop_all()
    print("Stopped llama-swap (and gateway, if running).")
    return 0


def _restart() -> int:
    """Stop + start. Picks up any config / registry edits made since launch."""
    processes.stop_all()
    return _start()


def _heal_unknown_architectures(reg) -> None:
    """Fetch a newer llama.cpp when no binary here can read a model.

    llama.cpp learns each new architecture — and each new weight format, like
    NVFP4 — in a specific upstream build, and inferhost otherwise fetches
    binaries only on first launch. A model released after that fetch fails with
    "unknown model architecture" (or "unknown type N"), llama-swap crash-loops
    it, and the GPU sits empty — with nothing in the UI explaining why.
    Upgrading the inferhost package doesn't help either, because the package and
    the binaries version independently.

    So: detect it from the GGUF's own architecture string and quant label, and
    self-heal by pulling the current managed binary. Deliberately model-agnostic
    — there is no list of known models here, and one released tomorrow is
    handled the same way. The download only happens when a registered model
    actually needs it AND upstream has moved on, so a genuinely unsupported
    model costs one API call per start, not a repeated download.
    """
    unknown: list[tuple[str, str]] = []
    for m in reg.models:
        if m.vocoder_path or m.kind == "image":
            continue  # not served by llama-server
        # (needle, probe) pairs: the architecture name, plus the ggml tensor
        # type for quants too new to assume every build carries them.
        wanted = [
            (arch, binaries.binary_supports_arch)
            for arch in (gguf.architecture_cached(configs._model_path(m)),) if arch
        ] + [
            (t, binaries.binary_supports_ggml_type)
            for t in (quant.RECENT_GGML_TYPES.get(quant.extract_quant(m.filename) or ""),) if t
        ]
        for needle, supported in wanted:
            if supported(paths.llama_server_path(), needle):
                continue
            if supported(paths.managed_llama_server_path(), needle):
                continue  # configs.server_binary() will route this model there
            unknown.append((m.name, needle))
    if not unknown:
        return
    listed = ", ".join(f"{name} ('{needle}')" for name, needle in unknown)
    installed = binaries.managed_llama_server_tag()
    latest = binaries.latest_llama_server_tag()
    have = installed or "the one installed"
    if latest is None:
        print(f"inferhost: {listed} needs a newer llama.cpp than {have}, and "
              "upstream is unreachable to fetch it.", file=sys.stderr)
        return
    if installed == latest:
        print(f"inferhost: {listed} — llama.cpp {latest} is the newest upstream "
              "build and it doesn't support that yet. The model will fail to "
              "load until upstream adds it.", file=sys.stderr)
        return
    print(f"inferhost: {listed} needs a newer llama.cpp than {have} — "
          f"fetching {latest} ...", file=sys.stderr)
    try:
        binaries.install_managed_llama_server(
            progress_cb=_progress_printer("llama-server"))
    except Exception as e:  # noqa: BLE001
        print(f"inferhost: llama-server refresh failed: {e}", file=sys.stderr)


def _progress_printer(label: str):
    """Coarse download progress on one line — every 10%, so logs stay readable."""
    state = {"pct": -10}

    def fn(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct >= state["pct"] + 10:
            state["pct"] = pct - pct % 10
            mib = total / (1024 * 1024)
            print(f"  {label}: {pct:3d}%  ({mib:.0f} MiB)", file=sys.stderr)

    return fn


def _update(args: list[str]) -> int:
    """Re-fetch the runtime binaries from upstream, then restore the daemons.

    llama.cpp gains model architectures continuously, so a binary installed
    months ago simply cannot load a model released last week — it fails with
    "unknown model architecture". Binaries were previously fetched only on
    first launch, which left "delete files under ~/.local/share by hand" as the
    only cure — exactly what the run.sh-only contract is meant to avoid.

    Daemons are stopped first. install_llama_server purges every lib*.so in
    bin_dir before extracting the replacement set; doing that underneath a live
    llama-server leaves it running against deleted libraries, and the next
    model load hits an ABI mismatch.

    An optional tag argument (e.g. ``update b10353``) pins this one install;
    without it the configured INFERHOST_LLAMACPP_VERSION is used ("latest").
    """
    version = args[0] if args else None
    old = binaries.installed_llama_server_tag()
    was_running = processes.swap_status().running
    if was_running:
        print("inferhost: stopping daemons before swapping binaries ...",
              file=sys.stderr)
        processes.stop_all()

    custom_path = settings().llama_server_path.strip()
    if custom_path:
        # Custom-binary mode: the user's own build is theirs to update. Saying
        # only "skipped" is how a box ends up 80 builds behind without anyone
        # noticing — the model fails with "unknown model architecture" and the
        # one command you'd reach for reports success. Name the gap instead.
        print(f"llama-server : skipped — INFERHOST_LLAMA_SERVER_PATH points at "
              f"{custom_path}")
        own = binaries.custom_llama_server_version()
        latest = binaries.latest_llama_server_tag()
        print(f"               your build     : {own or 'unknown'}")
        if latest:
            print(f"               upstream latest: {latest}")
        print("               inferhost can't rebuild this one. If a model fails "
              "with\n               \"unknown model architecture\", rebuild it "
              "from your llama.cpp\n               checkout (or unset "
              "INFERHOST_LLAMA_SERVER_PATH to use the\n               managed "
              "binary, which this command keeps current).")
    else:
        try:
            got = binaries.install_llama_server(
                version=version, progress_cb=_progress_printer("llama-server"))
        except Exception as e:  # noqa: BLE001
            print(f"inferhost: llama-server update failed: {e}", file=sys.stderr)
            return 1
        arrow = f"{old or 'unknown'} -> {got.version}"
        print(f"llama-server : {arrow}"
              f"{'  (already current)' if old == got.version else ''}")

    try:
        swap = binaries.install_llama_swap(progress_cb=_progress_printer("llama-swap"))
        print(f"llama-swap   : {swap.version}")
    except Exception as e:  # noqa: BLE001
        print(f"inferhost: llama-swap update failed: {e}", file=sys.stderr)
        return 1

    # sd-server only matters to boxes that actually generate images; refreshing
    # it on a chat-only install would download ~100 MiB nobody asked for.
    if not binaries.needs_sdcpp_refresh():
        try:
            sd = binaries.install_stable_diffusion(
                progress_cb=_progress_printer("sd-server"))
            print(f"sd-server    : {sd.version}")
        except Exception as e:  # noqa: BLE001
            print(f"inferhost: sd-server update failed (image models unaffected "
                  f"until retried): {e}", file=sys.stderr)

    if was_running:
        print("inferhost: restarting daemons ...", file=sys.stderr)
        return _start()
    print("Binaries updated. Start the stack with `inferhost start`.")
    return 0


def _prune(args: list[str]) -> int:
    """Report (and with --yes, delete) Hugging Face cache repos nothing uses.

    Deleting a model used to leave its weights behind, so existing installs
    carry tens of GiB of models that no registry entry references. Deletion now
    reclaims as it goes, but that doesn't retroactively clean what earlier
    versions stranded — this does.

    Lists by default and requires an explicit --yes, because the cache is
    shared: other tools on the box (ComfyUI, a Whisper server, a vLLM service)
    read the same directory, and inferhost cannot tell their downloads from
    stale ones. Naming every repo before touching anything lets the user veto.

    Pass repo ids to delete only those — "unused by inferhost" is not the same
    as "unused", and on a box that also runs vLLM the difference is somebody
    else's model. Without ids, --yes takes the whole list.
    """
    delete = "--yes" in args or "-y" in args
    wanted = [a for a in args if not a.startswith("-")]
    reg = registry.load()
    used: set[str] = set()
    for m in reg.models:
        for repo in (m.repo_id, m.draft_repo_id):
            if repo:
                used.add(hf.repo_dir(repo).name)
    hub = Path(HF_HUB_CACHE)
    if not hub.is_dir():
        print(f"No Hugging Face cache at {hub}.")
        return 0
    rows = []
    for d in sorted(hub.glob("models--*")):
        if d.name in used:
            continue
        size = sum(f.stat().st_size for f in d.rglob("*")
                   if f.is_file() and not f.is_symlink())
        rows.append((size, d))
    if not rows:
        print("Nothing to prune — every cached repo is in use.")
        return 0
    rows.sort(reverse=True)
    if wanted:
        by_repo = {d.name.removeprefix("models--").replace("--", "/"): (s, d)
                   for s, d in rows}
        picked, missing = [], []
        for name in wanted:
            hit = by_repo.get(name)
            if hit is None:
                missing.append(name)
            else:
                picked.append(hit)
        if missing:
            print("not prunable (in use by a registered model, or not cached): "
                  + ", ".join(missing), file=sys.stderr)
            return 1
        rows = picked
    total = sum(s for s, _ in rows)
    print(f"Cached repos no registered model uses ({hub}):\n")
    for size, d in rows:
        repo = d.name.removeprefix("models--").replace("--", "/")
        print(f"  {size / 2**30:8.1f} GiB  {repo}")
    print(f"\n  {total / 2**30:8.1f} GiB  total")
    if not delete:
        print("\nNothing deleted. This cache is shared with any other tool on "
              "this box that uses huggingface_hub (ComfyUI, vLLM, Whisper, ...), "
              "so 'unused by inferhost' does not mean unused.\n"
              "  inferhost prune --yes <repo> [<repo> ...]   delete only these\n"
              "  inferhost prune --yes                       delete everything above")
        return 0
    freed = 0
    for size, d in rows:
        try:
            shutil.rmtree(d)
        except OSError as e:
            print(f"could not remove {d.name}: {e}", file=sys.stderr)
            continue
        freed += size
    print(f"\nFreed {freed / 2**30:.1f} GiB.")
    return 0


def _status() -> int:
    swap = processes.swap_status()
    gw = processes.gateway_status()
    reg = registry.load()
    print(f"llama-swap : {'running' if swap.running else 'stopped'}  "
          f"pid={swap.pid or '-'}  port={swap.port}")
    pw = processes.pinwatch_status()
    print(f"pinwatch   : {'running' if pw.running else 'stopped'}  "
          f"pid={pw.pid or '-'}  (keeps pinned models in VRAM)")
    # A live PID is not the same as a serving gateway — probe the port, since
    # that's the difference between "the endpoint works" and "swap is serving
    # while the front door refuses connections".
    if gw.running:
        gw_health = "" if processes.gateway_serving() else \
            f"  ⚠ not answering on :{gw.port} — see {gw.log_path}"
    else:
        gw_health = ""
    print(f"litellm    : {'running' if gw.running else 'stopped'}  "
          f"pid={gw.pid or '-'}  port={gw.port}{gw_health}")
    if processes.has_tts_models():
        tts = processes.tts_status()
        print(f"inferhost-tts : {'running' if tts.running else 'stopped'}  "
              f"pid={tts.pid or '-'}  port={tts.port}")
    if reg.models:
        print(f"models     : {len(reg.models)} registered")
        for m in reg.models:
            if m.vocoder_path:
                tag = " [tts]"
            elif m.kind == "image":
                tag = " [image]"
            else:
                tag = ""
            print(f"  - {m.name}  ({m.quant or '?'}, {m.size_gib} GiB){tag}")
    else:
        print("models     : none registered")
    return 0


# ---- boot autostart (systemd user unit) ----
#
# A reboot used to leave the stack down (or half-up, if the daemons were later
# started by hand and one start step failed quietly). `inferhost autostart on`
# installs a systemd *user* unit that runs `inferhost start` at boot, plus
# lingering so the unit fires without anyone logging in.

_UNIT_NAME = "inferhost.service"


def _systemd_user_dir() -> Path:
    return Path("~/.config/systemd/user").expanduser()


def _inferhost_bin() -> str | None:
    """Absolute path baked into the unit's ExecStart.

    Prefer the PATH entry (e.g. ~/.local/bin/inferhost — a symlink that
    survives `uv tool upgrade`, unlike the tool venv it points into); fall back
    to the sibling of the running interpreter for repo/editable installs.
    """
    found = shutil.which("inferhost")
    if found:
        return found
    sibling = Path(sys.executable).parent / "inferhost"
    return str(sibling) if sibling.exists() else None


def _unit_text(bin_path: str) -> str:
    return f"""\
# Managed by `inferhost autostart on` — remove with `inferhost autostart off`.
[Unit]
Description=inferhost daemons (llama-swap + LiteLLM gateway + TTS)

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=%h
ExecStart={bin_path} start
ExecStop={bin_path} stop
# First start after an inferhost upgrade may re-download llama-server.
TimeoutStartSec=600

[Install]
WantedBy=default.target
"""


def _run_quiet(cmd: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _autostart(args: list[str]) -> int:
    action = args[0] if args else "status"
    if action not in ("on", "off", "status"):
        print("usage: inferhost autostart {on|off|status}", file=sys.stderr)
        return 2
    if shutil.which("systemctl") is None or shutil.which("loginctl") is None:
        print("autostart needs systemd (systemctl + loginctl) — Linux only.",
              file=sys.stderr)
        return 1
    unit_path = _systemd_user_dir() / _UNIT_NAME
    user = getpass.getuser()

    if action == "status":
        rc_en, enabled = _run_quiet(["systemctl", "--user", "is-enabled", _UNIT_NAME])
        _, linger = _run_quiet(["loginctl", "show-user", user, "--property=Linger"])
        print(f"unit file : {unit_path if unit_path.exists() else 'not installed'}")
        print(f"enabled   : {enabled if rc_en == 0 else 'no'}")
        print(f"linger    : {linger.removeprefix('Linger=') or 'unknown'}  "
              "(must be 'yes' for start-at-boot without login)")
        return 0

    if action == "off":
        _run_quiet(["systemctl", "--user", "disable", _UNIT_NAME])
        unit_path.unlink(missing_ok=True)
        _run_quiet(["systemctl", "--user", "daemon-reload"])
        print("Autostart disabled — the unit is removed; running daemons were "
              "left untouched. (User lingering stays on; it is harmless.)")
        return 0

    # action == "on"
    bin_path = _inferhost_bin()
    if bin_path is None:
        print("could not locate the `inferhost` executable to reference from "
              "the unit — is inferhost on your PATH?", file=sys.stderr)
        return 1
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(_unit_text(bin_path), encoding="utf-8")
    for step in (["systemctl", "--user", "daemon-reload"],
                 ["systemctl", "--user", "enable", _UNIT_NAME]):
        rc, out = _run_quiet(step)
        if rc != 0:
            print(f"autostart: `{' '.join(step)}` failed: {out}", file=sys.stderr)
            return 1
    # Lingering makes the user manager (and this unit) start at boot even with
    # nobody logged in. Usually allowed for one's own user; fall back to sudo.
    rc, out = _run_quiet(["loginctl", "enable-linger", user])
    if rc != 0:
        print(f"autostart: could not enable lingering ({out}).\n"
              f"Run this once by hand, then autostart is complete:\n"
              f"  sudo loginctl enable-linger {user}", file=sys.stderr)
    # Start through systemd now so the unit owns the daemons going forward.
    # Idempotent: `inferhost start` leaves already-running daemons alone.
    rc, out = _run_quiet(["systemctl", "--user", "start", _UNIT_NAME], timeout=600)
    if rc != 0:
        print(f"autostart: unit installed but starting it failed: {out}\n"
              f"Inspect with: systemctl --user status {_UNIT_NAME}", file=sys.stderr)
        return 1
    print(f"Autostart enabled — {_UNIT_NAME} will run `inferhost start` at boot.")
    print(f"Unit file: {unit_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    cmd = args[0] if args else "status"
    if cmd == "start":
        return _start()
    if cmd == "stop":
        return _stop()
    if cmd == "restart":
        return _restart()
    if cmd == "status":
        return _status()
    if cmd == "autostart":
        return _autostart(args[1:])
    if cmd == "update":
        return _update(args[1:])
    if cmd == "prune":
        return _prune(args[1:])
    print(f"inferhost-ops: unknown command {cmd!r}; expected one of "
          "start | stop | restart | status | update | prune | autostart.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
