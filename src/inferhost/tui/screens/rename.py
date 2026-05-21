"""Rename modal: change a model's access name (alias used by clients)."""
from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from inferhost.core import configs, registry

# Names are used as URL path components by llama-swap and as model_name aliases by
# LiteLLM, so keep them safe: lowercase, dot/dash/underscore, no whitespace.
_VALID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class RenameScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current_name: str) -> None:
        super().__init__()
        self.current_name = current_name

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-dialog"):
            yield Label("[bold]Rename model[/bold]")
            yield Static(
                f"Current name: [cyan]{self.current_name}[/cyan]\n"
                "This is the name your OpenAI client uses (the `model` field). "
                "Changing it here also updates llama-swap and LiteLLM configs.",
                id="rename-blurb",
            )
            yield Input(
                value=self.current_name,
                placeholder="new-model-name",
                id="new-name",
            )
            yield Static("", id="rename-status")
            with Horizontal(id="rename-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Rename", variant="primary", id="confirm")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        self._submit()

    @on(Input.Submitted, "#new-name")
    def _on_submit(self, _ev: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        new = self.query_one("#new-name", Input).value.strip()
        status = self.query_one("#rename-status", Static)
        if not new:
            status.update("[red]Name cannot be empty.[/red]")
            return
        if new == self.current_name:
            self.dismiss(None)
            return
        if not _VALID.match(new):
            status.update(
                "[red]Use lowercase letters, digits, and . _ - only "
                "(must start with a letter or digit).[/red]"
            )
            return
        reg = registry.load()
        if reg.get(new) is not None:
            status.update(f"[red]A model named '{new}' already exists.[/red]")
            return
        if not reg.rename(self.current_name, new):
            status.update("[red]Rename failed.[/red]")
            return
        try:
            registry.save(reg)
            configs.write_all(reg)
        except Exception as e:  # noqa: BLE001
            status.update(f"[red]Save failed: {e}[/red]")
            return
        self.dismiss(new)
