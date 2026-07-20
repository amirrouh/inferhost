"""Generic confirm/cancel modal — used for destructive actions (e.g. delete).

Named ``WarningScreen`` to match the ``#warning-dialog`` / ``#warning-body``
CSS that already ships in styles.tcss (added ahead of its first caller), so
this gets styling for free with zero new CSS.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class WarningScreen(ModalScreen[bool]):
    """Yes/no confirmation dialog. Dismisses with True (confirmed) or False
    (cancelled, including Escape) — never None, so callers can treat a falsy
    result as "don't proceed" without a separate None check."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, body: str, confirm_label: str = "Confirm") -> None:
        super().__init__()
        self.title_text = title
        self.body_text = body
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="warning-dialog"):
            yield Label(f"[bold]{self.title_text}[/bold]")
            yield Static(self.body_text, id="warning-body")
            with Horizontal(id="warning-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(self.confirm_label, variant="error", id="confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        self.dismiss(True)
