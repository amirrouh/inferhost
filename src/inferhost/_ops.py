"""Internal helper invoked by run.sh for headless ops (stop / status).

End users never touch this — the user-facing entry point is the `inferhost` TUI.
This module exists so `./run.sh stop` and `./run.sh status` can do their job
without re-introducing a CLI subcommand surface.
"""
from __future__ import annotations

import sys

from inferhost.core import processes, registry


def _stop() -> int:
    processes.stop_all()
    print("Stopped llama-swap (and gateway, if running).")
    return 0


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
    if cmd == "stop":
        return _stop()
    if cmd == "status":
        return _status()
    print(f"_ops: unknown command {cmd!r}; expected 'stop' or 'status'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
