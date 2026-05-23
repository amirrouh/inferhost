"""One-shot warning modal used for VRAM-feasibility errors."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class WarningScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss_screen", "OK")]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="warning-dialog"):
            yield Label(f"[bold yellow]⚠ {self._title}[/bold yellow]")
            yield Static(self._body, id="warning-body")
            yield Button("OK", variant="primary", id="ok")

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#ok")
    def _on_ok(self) -> None:
        self.dismiss(None)
