"""Textual TUI entry point."""
from __future__ import annotations

from pathlib import Path

from textual.app import App

from inferhost.tui.screens.dashboard import DashboardScreen


CSS_PATH = Path(__file__).parent / "styles.tcss"


class InferhostApp(App):
    CSS_PATH = str(CSS_PATH)
    TITLE = "inferhost"
    SUB_TITLE = "Local Hugging Face model server"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())


def run_tui() -> None:
    InferhostApp().run()
