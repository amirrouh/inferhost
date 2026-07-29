"""inferhost entry point — one binary, several modes.

Invocations:

  inferhost                 Launch the TUI dashboard (default — no args).
  inferhost start           Start daemons (llama-swap + LiteLLM) in background.
  inferhost stop            Stop daemons.
  inferhost restart         Stop + start (picks up config edits).
  inferhost status          Print daemon + endpoint status.
  inferhost autostart on|off  Start the daemons at boot (systemd user unit).
  inferhost tui             Explicit alias for the TUI.
  inferhost --help, -h      Print help.
  inferhost --version, -V   Print version.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

_HELP = """\
inferhost — run any Hugging Face GGUF model on your GPU.

Usage:
  inferhost                  Launch the TUI dashboard (default).
  inferhost tui              Same as above (explicit).
  inferhost start            Start llama-swap + LiteLLM as background daemons.
  inferhost stop             Stop both daemons.
  inferhost restart          Stop + start (picks up config edits).
  inferhost status           Print daemon + endpoint status.
  inferhost autostart on|off Start the daemons automatically at boot via a
                             systemd user unit (`autostart status` to inspect).
  inferhost --help, -h       Show this help.
  inferhost --version, -V    Print version.

Endpoints (after start):
  LiteLLM gateway    http://<host>:9001/v1   (set INFERHOST_GATEWAY_HOST)
  llama-swap direct  http://<host>:9090/v1   (set INFERHOST_SWAP_HOST)

Docs and source: https://github.com/amirrouh/inferhost
"""

_OPS_CMDS = {"start", "stop", "restart", "status", "autostart"}


def _pkg_version() -> str:
    try:
        return version("inferhost")
    except PackageNotFoundError:
        return "unknown"


def app() -> None:
    argv = sys.argv[1:]
    if not argv:
        # No subcommand — launch the TUI. This is the canonical first-run UX.
        from inferhost.tui.app import run_tui
        run_tui()
        return
    cmd = argv[0]
    if cmd in ("-h", "--help", "help"):
        print(_HELP)
        return
    if cmd in ("-V", "--version", "version"):
        print(f"inferhost {_pkg_version()}")
        return
    if cmd == "tui":
        from inferhost.tui.app import run_tui
        run_tui()
        return
    if cmd in _OPS_CMDS:
        from inferhost import _ops
        sys.exit(_ops.main(argv))
    print(f"inferhost: unknown command {cmd!r}\n", file=sys.stderr)
    print(_HELP, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    app()
