"""Per-model settings modal.

Edits the fields on a single ``registry.Model`` that the user is most likely to
want to tune per-model rather than globally:

* ``ctx`` — the ``-c`` flag (context window in tokens).
* ``reasoning`` / ``reasoning_budget`` — per-model thinking-mode overrides.
* ``pin`` — keep model co-resident in VRAM instead of swapping on demand.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from inferhost.core import registry, vram

_MIN_CTX = 512
_MAX_CTX = 1_048_576

# Accept common synonyms so the user isn't held to exactly "on"/"off"/"auto".
# Empty string means "inherit the global Settings value".
_REASONING_ALIASES: dict[str, str] = {
    "": "",
    "on": "on", "yes": "on", "y": "on", "true": "on", "1": "on",
    "off": "off", "no": "off", "n": "off", "false": "off", "0": "off",
    "auto": "auto",
}

# Strict bool parser for the pin field — same vocabulary as reasoning, no
# "inherit" sentinel since pinning is always a per-model decision.
_BOOL_ALIASES: dict[str, bool] = {
    "yes": True, "y": True, "true": True, "1": True, "on": True,
    "no": False, "n": False, "false": False, "0": False, "off": False,
}


class ModelSettingsScreen(ModalScreen[bool]):
    """Configure per-model overrides (ctx, reasoning, pin) for a single model."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        reg = registry.load()
        m = reg.get(model_name)
        self.current_ctx = m.ctx if m is not None else 0
        self.current_reasoning = m.reasoning if m is not None else ""
        self.current_reasoning_budget = m.reasoning_budget if m is not None else -2
        self.current_pin = m.pin if m is not None else False

    def compose(self) -> ComposeResult:
        with Vertical(id="model-settings-dialog"):
            yield Label("[bold]Model settings[/bold]")
            yield Static(
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                "These values override the global defaults for this one model. "
                "Daemons reload immediately after saving.",
                id="model-settings-blurb",
            )

            yield Label("Context window (-c)")
            yield Input(
                value=str(self.current_ctx),
                placeholder="e.g. 8192",
                id="f-ctx",
            )

            yield Label("Reasoning (--reasoning)")
            yield Input(
                value=self.current_reasoning,
                placeholder="blank=use global · on/yes · off/no · auto",
                id="f-reasoning",
            )

            yield Label("Reasoning budget (--reasoning-budget)")
            budget_str = "" if self.current_reasoning_budget == -2 else str(self.current_reasoning_budget)
            yield Input(
                value=budget_str,
                placeholder="blank=use global · -1=unlimited · 0=none · N=tokens",
                id="f-reasoning-budget",
            )

            yield Label("Pin in VRAM (co-resident with other pinned models)")
            yield Input(
                value="yes" if self.current_pin else "no",
                placeholder="yes/no — pinned models stay loaded together instead of swapping",
                id="f-pin",
            )

            yield Static("", id="model-settings-status")
            with Horizontal(id="model-settings-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        self._submit()

    @on(Input.Submitted)
    def _on_submit(self, _ev: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        status = self.query_one("#model-settings-status", Static)
        errors: list[str] = []

        raw_ctx = self.query_one("#f-ctx", Input).value.strip()
        try:
            new_ctx = int(raw_ctx)
        except ValueError:
            errors.append("ctx must be an integer")
            new_ctx = self.current_ctx
        else:
            if new_ctx < _MIN_CTX or new_ctx > _MAX_CTX:
                errors.append(f"ctx must be between {_MIN_CTX} and {_MAX_CTX}")

        raw_reasoning = self.query_one("#f-reasoning", Input).value.strip().lower()
        if raw_reasoning in _REASONING_ALIASES:
            new_reasoning = _REASONING_ALIASES[raw_reasoning]
        else:
            errors.append(
                f"reasoning: expected blank/on/off/auto (or yes/no), got '{raw_reasoning}'"
            )
            new_reasoning = self.current_reasoning

        raw_budget = self.query_one("#f-reasoning-budget", Input).value.strip()
        if raw_budget == "":
            new_budget = -2  # sentinel meaning "inherit from global"
        else:
            try:
                new_budget = int(raw_budget)
            except ValueError:
                errors.append("reasoning budget: must be blank or an integer")
                new_budget = self.current_reasoning_budget
            else:
                if new_budget < -1:
                    errors.append("reasoning budget: must be blank, -1, 0, or positive")

        raw_pin = self.query_one("#f-pin", Input).value.strip().lower()
        if raw_pin in _BOOL_ALIASES:
            new_pin = _BOOL_ALIASES[raw_pin]
        else:
            errors.append(f"pin: expected yes/no (or true/false, 1/0), got '{raw_pin}'")
            new_pin = self.current_pin

        if errors:
            status.update("[red]" + " · ".join(errors) + "[/red]")
            return

        reg = registry.load()
        m = reg.get(self.model_name)
        if m is None:
            status.update("[red]Model no longer exists.[/red]")
            return

        changed = (
            m.ctx != new_ctx
            or m.reasoning != new_reasoning
            or m.reasoning_budget != new_budget
            or m.pin != new_pin
        )
        if not changed:
            self.dismiss(False)
            return

        # VRAM feasibility is informational only — surface the estimate but
        # never block the save. The dashboard's pinned-overflow row and a real
        # llama-server OOM are the authoritative signals.
        if new_pin and not self.current_pin:
            ok, needed, free = vram.can_pin(reg, m)
            if not ok:
                status.update(
                    f"[yellow]⚠ VRAM tight: '{m.name}' needs ~{needed:.1f} GiB "
                    f"but only {free:.1f} GiB free. Saving anyway.[/yellow]"
                )

        m.ctx = new_ctx
        m.reasoning = new_reasoning
        m.reasoning_budget = new_budget
        m.pin = new_pin
        try:
            registry.save(reg)
        except Exception as e:  # noqa: BLE001
            status.update(f"[red]Save failed: {e}[/red]")
            return
        self.dismiss(True)
