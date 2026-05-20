"""inferhost entry point. The only public command is `inferhost`, which launches the TUI."""
from __future__ import annotations


def app() -> None:
    from inferhost.tui.app import run_tui
    run_tui()


if __name__ == "__main__":
    app()
