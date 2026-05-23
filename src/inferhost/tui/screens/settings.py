"""Settings modal: edit ports, context size, GPU layers, flash attention."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from inferhost.settings import EDITABLE_FIELDS, save_overrides, settings


class SettingsScreen(ModalScreen[bool]):
    """Modal screen for editing TUI-persisted settings."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    FIELDS: tuple[tuple[str, str, str], ...] = (
        ("swap_port", "llama-swap port", "OpenAI-compatible endpoint port"),
        ("swap_host", "llama-swap bind host", "127.0.0.1 (default) / 0.0.0.0 to expose on all interfaces"),
        ("gateway_port", "Gateway port", "LiteLLM gateway port"),
        ("gateway_host", "Gateway bind host", "0.0.0.0 exposes externally; 127.0.0.1 keeps it local"),
        ("default_ctx", "Default context", "Tokens of context for new models"),
        ("gpu_layers", "GPU layers (-ngl)", "99 = offload all layers; 0 = CPU only"),
        ("flash_attention", "Flash attention", "on / off"),
        ("parallel_slots", "Parallel slots (--parallel)", "1 = serial; higher = concurrent requests on the same model"),
        ("reasoning", "Reasoning (--reasoning)", "on/off/auto (yes/no also accepted) — thinking mode for capable models"),
        ("reasoning_budget", "Reasoning budget", "Tokens of thinking allowed. -1 = unlimited, 0 = none"),
    )

    def compose(self) -> ComposeResult:
        s = settings()
        with Vertical(id="settings-dialog"):
            yield Label("[bold]Settings[/bold]")
            yield Static(
                "Saved to ~/.config/inferhost/inferhost.env. "
                "Daemons reload these values on Restart.",
                id="settings-blurb",
            )
            for field, label, hint in self.FIELDS:
                yield Label(label)
                yield Input(
                    value=str(getattr(s, field)),
                    id=f"f-{field}",
                    placeholder=hint,
                )
            yield Static("", id="settings-status")
            with Horizontal(id="settings-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="save")

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#save")
    def _on_save(self) -> None:
        self._save()

    @on(Input.Submitted)
    def _on_submit(self, _ev: Input.Submitted) -> None:
        self._save()

    def _save(self) -> None:
        updates: dict[str, object] = {}
        errors: list[str] = []
        for field, label, _hint in self.FIELDS:
            if field not in EDITABLE_FIELDS:
                continue
            raw = self.query_one(f"#f-{field}", Input).value.strip()
            if not raw:
                errors.append(f"{label}: empty")
                continue
            if field in {"swap_port", "gateway_port"}:
                try:
                    port = int(raw)
                except ValueError:
                    errors.append(f"{label}: not a number")
                    continue
                if not (1 <= port <= 65535):
                    errors.append(f"{label}: out of range")
                    continue
                updates[field] = port
            elif field in {"default_ctx", "gpu_layers"}:
                try:
                    n = int(raw)
                except ValueError:
                    errors.append(f"{label}: not a number")
                    continue
                if n < 0:
                    errors.append(f"{label}: negative")
                    continue
                updates[field] = n
            elif field == "parallel_slots":
                try:
                    n = int(raw)
                except ValueError:
                    errors.append(f"{label}: not a number")
                    continue
                if n < 1 or n > 64:
                    errors.append(f"{label}: must be between 1 and 64")
                    continue
                updates[field] = n
            elif field == "reasoning":
                v = raw.lower()
                aliases = {
                    "on": "on", "yes": "on", "y": "on", "true": "on", "1": "on",
                    "off": "off", "no": "off", "n": "off", "false": "off", "0": "off",
                    "auto": "auto",
                }
                if v not in aliases:
                    errors.append(f"{label}: expected on/off/auto (or yes/no)")
                    continue
                updates[field] = aliases[v]
            elif field == "reasoning_budget":
                try:
                    n = int(raw)
                except ValueError:
                    errors.append(f"{label}: not a number")
                    continue
                if n < -1:
                    errors.append(f"{label}: must be -1, 0, or a positive integer")
                    continue
                updates[field] = n
            elif field == "flash_attention":
                v = raw.lower()
                if v not in {"on", "off", "auto"}:
                    errors.append(f"{label}: expected on/off/auto")
                    continue
                updates[field] = v
            elif field in {"swap_host", "gateway_host"}:
                # Light validation: just an IPv4 dotted-quad or "0.0.0.0".
                # We let the kernel reject anything bogus when bind() runs.
                if not raw.replace(".", "").replace(":", "").isalnum():
                    errors.append(f"{label}: looks invalid")
                    continue
                updates[field] = raw

        swap = updates.get("swap_port")
        gw = updates.get("gateway_port")
        if swap is not None and gw is not None and swap == gw:
            errors.append("swap port and gateway port must differ")

        status = self.query_one("#settings-status", Static)
        if errors:
            status.update("[red]" + " · ".join(errors) + "[/red]")
            return

        try:
            path = save_overrides(updates)
        except Exception as e:  # noqa: BLE001
            status.update(f"[red]Failed to save: {e}[/red]")
            return

        status.update(f"[green]Saved to {path}.[/green]")
        self.dismiss(True)
