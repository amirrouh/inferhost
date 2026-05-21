"""Per-model context window editor.

Each model in the registry carries its own ``ctx`` value, which is baked into the
``llama-server -c <ctx>`` flag. This screen lets the user override that value for
a single model without touching the global ``default_ctx`` setting.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from inferhost.core import registry

# Reasonable bounds — llama.cpp itself accepts any positive int, but going outside
# this window is almost always a typo (and 0 would crash llama-server).
_MIN_CTX = 512
_MAX_CTX = 1_048_576  # 1M tokens; enough headroom for any current model.


class SetCtxScreen(ModalScreen[int | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        reg = registry.load()
        m = reg.get(model_name)
        self.current_ctx = m.ctx if m is not None else 0

    def compose(self) -> ComposeResult:
        with Vertical(id="ctx-dialog"):
            yield Label("[bold]Set context window[/bold]")
            yield Static(
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                f"Current ctx: [cyan]{self.current_ctx}[/cyan] tokens\n"
                "This is the [b]-c[/b] flag passed to llama-server. Larger values use "
                "more VRAM for the KV cache.",
                id="ctx-blurb",
            )
            yield Input(
                value=str(self.current_ctx),
                placeholder="e.g. 8192",
                id="new-ctx",
            )
            yield Static("", id="ctx-status")
            with Horizontal(id="ctx-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="confirm")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        self._submit()

    @on(Input.Submitted, "#new-ctx")
    def _on_submit(self, _ev: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        raw = self.query_one("#new-ctx", Input).value.strip()
        status = self.query_one("#ctx-status", Static)
        try:
            new_ctx = int(raw)
        except ValueError:
            status.update("[red]Must be a positive integer.[/red]")
            return
        if new_ctx < _MIN_CTX or new_ctx > _MAX_CTX:
            status.update(
                f"[red]Out of range — pick a value between {_MIN_CTX} and {_MAX_CTX}.[/red]"
            )
            return
        if new_ctx == self.current_ctx:
            self.dismiss(None)
            return
        reg = registry.load()
        m = reg.get(self.model_name)
        if m is None:
            status.update("[red]Model no longer exists.[/red]")
            return
        m.ctx = new_ctx
        try:
            registry.save(reg)
        except Exception as e:  # noqa: BLE001
            status.update(f"[red]Save failed: {e}[/red]")
            return
        self.dismiss(new_ctx)
