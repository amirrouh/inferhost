"""Textual TUI entry point."""
from __future__ import annotations

from pathlib import Path

from textual.app import App

from inferhost.core import paths
from inferhost.tui.screens.dashboard import DashboardScreen
from inferhost.tui.screens.install import InstallScreen
from inferhost.tui.screens.splash import SplashScreen

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
            self._show_splash_then_dashboard()
        else:
            self.push_screen(InstallScreen(), self._after_install)

    def _after_install(self, ok: bool | None) -> None:
        if ok:
            self._show_splash_then_dashboard()
        else:
            self.exit()

    def _show_splash_then_dashboard(self) -> None:
        # Push dashboard first so it's mounted underneath the splash; the
        # splash then pops itself off after its timer, revealing the dashboard.
        # Avoids the "await dismiss from message handler" guard in Textual.
        self.push_screen(DashboardScreen())
        self.push_screen(SplashScreen())


def run_tui() -> None:
    InferhostApp().run()
