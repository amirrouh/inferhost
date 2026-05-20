"""Textual TUI entry point."""
from __future__ import annotations

from pathlib import Path

from textual.app import App

from inferhost.core import paths
from inferhost.tui.screens.dashboard import DashboardScreen
from inferhost.tui.screens.install import InstallScreen

CSS_PATH = Path(__file__).parent / "styles.tcss"


def _binaries_present() -> bool:
    return paths.llama_server_path().exists() and paths.llama_swap_path().exists()


class InferhostApp(App):
    CSS_PATH = str(CSS_PATH)
    TITLE = "inferhost"
    SUB_TITLE = "Local Hugging Face model server"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        paths.ensure_dirs()
        if _binaries_present():
            self.push_screen(DashboardScreen())
        else:
            self.push_screen(InstallScreen(), self._after_install)

    def _after_install(self, ok: bool | None) -> None:
        if ok:
            self.push_screen(DashboardScreen())
        else:
            self.exit()


def run_tui() -> None:
    InferhostApp().run()
