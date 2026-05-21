"""Brief startup splash showing the inferhost banner, then auto-dismisses.

We render this once on launch (after the install check) so the dashboard isn't
cluttered by a permanent banner — the dashboard itself only shows a one-line
ribbon. The splash auto-dismisses after ``_DURATION`` seconds.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

_DURATION = 1.0  # seconds


def _version() -> str:
    try:
        return version("inferhost")
    except PackageNotFoundError:
        return "dev"


class SplashScreen(Screen):
    """A 4-row colored banner shown for ~1 second on startup.

    The splash is pushed *on top of* the dashboard, not handed a dismiss
    callback. When the timer fires we ``pop_screen()`` ourselves off the stack,
    which avoids Textual's ``Can't await screen.dismiss() from the screen's
    message handler`` guard that fires when ``dismiss()`` is invoked from a
    set_timer callback.
    """

    _LOGO = (
        " ___        __         _              _   \n"
        "|_ _|_ _   / _|___ _ _| |_  ___  ___ | |_ \n"
        " | || ' \\ |  _/ -_) '_| ' \\/ _ \\(_-<|  _|\n"
        "|___|_||_||_| \\___|_| |_||_\\___/__/ \\__|"
    )

    def compose(self) -> ComposeResult:
        with Vertical(id="splash-container"):
            yield Static(self._LOGO, id="splash-logo")
            yield Static("local hugging face model server", id="splash-tagline")
            yield Static(f"v{_version()}", id="splash-version")

    def on_mount(self) -> None:
        self.set_timer(_DURATION, self._exit)

    def _exit(self) -> None:
        self.app.pop_screen()
