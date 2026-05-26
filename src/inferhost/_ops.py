"""Headless ops for inferhost: start / stop / restart / status — no TUI required.

The user-facing entry point is the ``inferhost`` TUI, but the daemons (llama-swap
and the LiteLLM gateway) outlive the TUI process because they are spawned via
``start_new_session=True``. This module lets you control them without ever
opening the TUI — useful for servers, systemd units, and shell scripts.

Console-script: ``inferhost-ops {start|stop|restart|status}``.
"""
from __future__ import annotations

import sys

from inferhost.core import binaries, configs, processes, registry


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
    # Regenerate configs in case the registry was edited since last launch.
    configs.write_all(reg)
    for note in configs.consume_notices():
        print(f"notice: {note}", file=sys.stderr)
    swap_started = False
    if not processes.swap_status().running:
        processes.start_swap()
        swap_started = True
    gw_started = False
    if processes.gateway_available() and not processes.gateway_status().running:
        processes.start_gateway()
        gw_started = True
    # Re-read for accurate post-start status.
    swap = processes.swap_status()
    gw = processes.gateway_status()
    swap_note = " (just started)" if swap_started else ""
    gw_note = " (just started)" if gw_started else ""
    print(f"llama-swap : {'running' if swap.running else 'NOT running'}  "
          f"pid={swap.pid or '-'}  port={swap.port}{swap_note}")
    print(f"litellm    : {'running' if gw.running else 'NOT running'}  "
          f"pid={gw.pid or '-'}  port={gw.port}{gw_note}")
    return 0 if swap.running else 1


def _stop() -> int:
    processes.stop_all()
    print("Stopped llama-swap (and gateway, if running).")
    return 0


def _restart() -> int:
    """Stop + start. Picks up any config / registry edits made since launch."""
    processes.stop_all()
    return _start()


def _status() -> int:
    swap = processes.swap_status()
    gw = processes.gateway_status()
    reg = registry.load()
    print(f"llama-swap : {'running' if swap.running else 'stopped'}  "
          f"pid={swap.pid or '-'}  port={swap.port}")
    print(f"litellm    : {'running' if gw.running else 'stopped'}  "
          f"pid={gw.pid or '-'}  port={gw.port}")
    if reg.models:
        print(f"models     : {len(reg.models)} registered")
        for m in reg.models:
            print(f"  - {m.name}  ({m.quant or '?'}, {m.size_gib} GiB)")
    else:
        print("models     : none registered")
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
    print(f"inferhost-ops: unknown command {cmd!r}; expected one of "
          "start | stop | restart | status.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
