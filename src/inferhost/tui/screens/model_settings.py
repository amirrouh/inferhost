"""Per-model settings modal.

Edits the fields on a single ``registry.Model`` that the user is most likely to
want to tune per-model rather than globally:

* ``ctx`` — the ``-c`` flag (context window in tokens).
* ``cache_type_k`` — the ``-ctk`` flag (KV cache K quantization, e.g. ``q8_0``).
* ``cache_type_v`` — the ``-ctv`` flag (KV cache V quantization).

KV cache quantization is the cheapest way to fit a larger ``ctx`` into the same
VRAM. ``q8_0`` is near-lossless and roughly halves KV memory; ``q4_0`` cuts it
~4× but starts to bite on long contexts.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from inferhost.core import registry

_MIN_CTX = 512
_MAX_CTX = 1_048_576

# Values llama.cpp's --cache-type-k / --cache-type-v accept. Empty string means
# "use llama.cpp default" (f16). Validation is intentionally permissive — if a
# user types something exotic that a future llama.cpp build supports, we let it
# through and surface any error from llama-server itself.
_VALID_CACHE_TYPES = {
    "", "f32", "f16", "bf16",
    "q8_0", "q5_1", "q5_0", "q4_1", "q4_0",
    "iq4_nl",
}

# Accept common synonyms so the user isn't held to exactly "on"/"off"/"auto".
# Empty string means "inherit the global Settings value".
_REASONING_ALIASES: dict[str, str] = {
    "": "",
    "on": "on", "yes": "on", "y": "on", "true": "on", "1": "on",
    "off": "off", "no": "off", "n": "off", "false": "off", "0": "off",
    "auto": "auto",
}


class ModelSettingsScreen(ModalScreen[bool]):
    """Configure ctx and KV-cache quantization for a single model."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        reg = registry.load()
        m = reg.get(model_name)
        self.current_ctx = m.ctx if m is not None else 0
        self.current_ctk = m.cache_type_k if m is not None else ""
        self.current_ctv = m.cache_type_v if m is not None else ""
        self.current_reasoning = m.reasoning if m is not None else ""
        self.current_reasoning_budget = m.reasoning_budget if m is not None else -2

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

            yield Label("KV cache type — K (-ctk)")
            yield Input(
                value=self.current_ctk,
                placeholder="blank=f16 default · q8_0 · q5_1 · q5_0 · q4_1 · q4_0",
                id="f-ctk",
            )

            yield Label("KV cache type — V (-ctv)")
            yield Input(
                value=self.current_ctv,
                placeholder="blank=f16 default · q8_0 · q5_1 · q5_0 · q4_1 · q4_0",
                id="f-ctv",
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

        new_ctk = self.query_one("#f-ctk", Input).value.strip().lower()
        new_ctv = self.query_one("#f-ctv", Input).value.strip().lower()
        if new_ctk not in _VALID_CACHE_TYPES:
            errors.append(f"-ctk: unknown type '{new_ctk}'")
        if new_ctv not in _VALID_CACHE_TYPES:
            errors.append(f"-ctv: unknown type '{new_ctv}'")

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
            or m.cache_type_k != new_ctk
            or m.cache_type_v != new_ctv
            or m.reasoning != new_reasoning
            or m.reasoning_budget != new_budget
        )
        if not changed:
            self.dismiss(False)
            return

        m.ctx = new_ctx
        m.cache_type_k = new_ctk
        m.cache_type_v = new_ctv
        m.reasoning = new_reasoning
        m.reasoning_budget = new_budget
        try:
            registry.save(reg)
        except Exception as e:  # noqa: BLE001
            status.update(f"[red]Save failed: {e}[/red]")
            return
        self.dismiss(True)
