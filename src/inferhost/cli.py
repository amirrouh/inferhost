"""inferhost entry point.

Two console scripts ship together (see pyproject.toml):

  inferhost       — TUI dashboard (this entry point). Add/configure/pin models,
                    start daemons interactively, watch logs.
  inferhost-ops   — Headless control: ``start | stop | restart | status``.
                    Same daemons, no TUI. Use this in scripts and unattended
                    setups.

The TUI launches whenever ``inferhost`` is invoked with no recognized flag —
``inferhost --help`` prints the banner below and exits.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

_HELP = """\
inferhost — run any Hugging Face GGUF model on your GPU.

Usage:
  inferhost                  Launch the TUI (default).
  inferhost --help, -h       Show this help.
  inferhost --version, -V    Print version.

For headless use (no TUI), the package ships a companion command:

  inferhost-ops start        Start llama-swap + LiteLLM gateway as
                             detached background daemons.
  inferhost-ops stop         Stop them.
  inferhost-ops restart      Stop + start (picks up config edits).
  inferhost-ops status       Show daemon + endpoint status.

Endpoints (after start):
  LiteLLM gateway    http://<host>:9001/v1   (set INFERHOST_GATEWAY_HOST)
  llama-swap direct  http://<host>:9090/v1   (set INFERHOST_SWAP_HOST)

Docs and source: https://github.com/amirrouh/inferhost
"""


def _pkg_version() -> str:
    try:
        return version("inferhost")
    except PackageNotFoundError:
        return "unknown"


def app() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return
    if argv and argv[0] in ("-V", "--version", "version"):
        print(f"inferhost {_pkg_version()}")
        return
    from inferhost.tui.app import run_tui
    run_tui()


if __name__ == "__main__":
    app()
