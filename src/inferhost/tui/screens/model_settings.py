"""Per-model settings modal.

Edits the fields on a single ``registry.Model`` that the user is most likely to
want to tune per-model rather than globally:

* ``ctx`` — the ``-c`` flag (context window in tokens).
* ``kv_quant_k`` / ``kv_quant_v`` — per-model ``-ctk`` / ``-ctv`` overrides
  (blank means "inherit the global Settings value").
* ``gpu_layers`` — per-model ``-ngl`` override (blank = inherit).
* ``parallel_slots`` — per-model ``--parallel`` override (blank = inherit).
* ``flash_attention`` — per-model ``-fa`` override (blank = inherit).
* ``reasoning`` / ``reasoning_budget`` — per-model thinking-mode overrides.
* ``pin`` — keep model co-resident in VRAM instead of swapping on demand.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from inferhost.core import configs, gguf, paths, registry, vram
from inferhost.settings import KV_QUANT_VALUES

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

# Flash-attention is "on" / "off" / "auto", but pass it through verbatim to
# llama-server (it's a tri-state in newer builds). Empty = inherit global.
_FA_ALIASES: dict[str, str] = {
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
    """Configure per-model overrides for one model.

    Most fields carry an "inherit from global" sentinel (blank or -1 / 0) so
    the user only has to fill in what they actually want to override for this
    particular model.
    """

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
        self.current_kv_k = m.kv_quant_k if m is not None else ""
        self.current_kv_v = m.kv_quant_v if m is not None else ""
        self.current_gpu_layers = m.gpu_layers if m is not None else -1
        self.current_parallel = m.parallel_slots if m is not None else 0
        self.current_threads = m.threads if m is not None else 0
        self.current_mlock = m.mlock if m is not None else False
        self.current_fa = m.flash_attention if m is not None else ""
        self.current_extra_args = m.extra_args if m is not None else ""
        self.current_spec_override = m.spec_draft_n_max_override if m is not None else -1
        # Read the GGUF's native trained context straight from disk so the user
        # sees the real ceiling for -c (and knows a higher value gets clamped).
        # MTP fields only matter for models that carry MTP/NextN heads.
        if m is not None:
            self.native_ctx = gguf.native_context_cached(
                m.local_path or str(paths.models_dir() / m.filename)
            )
            self.is_mtp = configs.is_mtp_capable(m)
        else:
            self.native_ctx = None
            self.is_mtp = False

    def compose(self) -> ComposeResult:
        kv_values = " · ".join(KV_QUANT_VALUES)
        with Vertical(id="model-settings-dialog"):
            yield Label("[bold]Model settings[/bold]")
            yield Static(
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                "Each field overrides the global default for this one model. "
                "Leave blank (or use the sentinel) to inherit the global "
                "Settings value. Daemons reload immediately after saving.",
                id="model-settings-blurb",
            )

            yield Label("Context window (-c)")
            yield Input(
                value=str(self.current_ctx),
                placeholder="e.g. 8192",
                id="f-ctx",
            )
            if self.native_ctx:
                yield Static(
                    f"[grey50]Model's native trained context: "
                    f"{self.native_ctx:,} tokens — the most this file supports. "
                    f"A larger -c is clamped to it on load.[/grey50]",
                    id="ctx-native-hint",
                )

            yield Label("KV cache K  (-ctk)")
            yield Input(
                value=self.current_kv_k,
                placeholder=f"blank=use global · {kv_values}",
                id="f-kv-k",
            )

            yield Label("KV cache V  (-ctv)")
            yield Input(
                value=self.current_kv_v,
                placeholder=f"blank=use global · {kv_values}",
                id="f-kv-v",
            )

            yield Label("GPU layers (-ngl)")
            gpu_str = "" if self.current_gpu_layers < 0 else str(self.current_gpu_layers)
            yield Input(
                value=gpu_str,
                placeholder="blank=use global · 0 = CPU only · 99 = full offload",
                id="f-gpu-layers",
            )

            yield Label("Parallel slots (--parallel)")
            par_str = "" if self.current_parallel <= 0 else str(self.current_parallel)
            yield Input(
                value=par_str,
                placeholder="blank=use global · concurrent requests on this model",
                id="f-parallel",
            )

            yield Label("CPU threads (--threads)")
            threads_str = "" if self.current_threads <= 0 else str(self.current_threads)
            yield Input(
                value=threads_str,
                placeholder="blank=auto (all cores) · N=generation threads (matters when partly on CPU)",
                id="f-threads",
            )

            yield Label("Flash attention (-fa)")
            yield Input(
                value=self.current_fa,
                placeholder="blank=use global · on / off / auto",
                id="f-fa",
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

            yield Label("Extra llama-server args (raw, appended to cmd)")
            yield Input(
                value=self.current_extra_args,
                placeholder='blank=none · e.g. "--embeddings --pooling last"',
                id="f-extra-args",
            )

            yield Label("MTP draft tokens (--spec-draft-n-max)")
            spec_str = "" if self.current_spec_override < 0 else str(self.current_spec_override)
            mtp_note = (
                "" if self.is_mtp
                else " · this model has no MTP heads, so it has no effect"
            )
            yield Input(
                value=spec_str,
                placeholder=(
                    f"blank=use global · 0=off · 1-5 tokens/step (2 typical){mtp_note}"
                ),
                id="f-spec-draft",
            )

            yield Label("Pin in VRAM (co-resident with other pinned models)")
            yield Input(
                value="yes" if self.current_pin else "no",
                placeholder="yes/no — pinned models stay loaded together instead of swapping",
                id="f-pin",
            )

            yield Label("Lock in system RAM (--mlock)")
            yield Input(
                value="yes" if self.current_mlock else "no",
                placeholder="yes/no — keep CPU-offloaded weights in RAM (no paging); for low-GPU-layer models",
                id="f-mlock",
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

        raw_kv_k = self.query_one("#f-kv-k", Input).value.strip().lower()
        if raw_kv_k and raw_kv_k not in KV_QUANT_VALUES:
            errors.append(
                f"kv K: expected blank or one of {', '.join(KV_QUANT_VALUES)}"
            )
            new_kv_k = self.current_kv_k
        else:
            new_kv_k = raw_kv_k

        raw_kv_v = self.query_one("#f-kv-v", Input).value.strip().lower()
        if raw_kv_v and raw_kv_v not in KV_QUANT_VALUES:
            errors.append(
                f"kv V: expected blank or one of {', '.join(KV_QUANT_VALUES)}"
            )
            new_kv_v = self.current_kv_v
        else:
            new_kv_v = raw_kv_v

        raw_gpu = self.query_one("#f-gpu-layers", Input).value.strip()
        if raw_gpu == "":
            new_gpu_layers = -1  # sentinel meaning "inherit from global"
        else:
            try:
                new_gpu_layers = int(raw_gpu)
            except ValueError:
                errors.append("gpu layers: must be blank or an integer >= 0")
                new_gpu_layers = self.current_gpu_layers
            else:
                if new_gpu_layers < 0:
                    errors.append("gpu layers: must be blank, 0, or positive")

        raw_par = self.query_one("#f-parallel", Input).value.strip()
        if raw_par == "":
            new_parallel = 0  # sentinel meaning "inherit from global"
        else:
            try:
                new_parallel = int(raw_par)
            except ValueError:
                errors.append("parallel slots: must be blank or a positive integer")
                new_parallel = self.current_parallel
            else:
                if new_parallel < 1:
                    errors.append("parallel slots: must be blank or >= 1")

        raw_threads = self.query_one("#f-threads", Input).value.strip()
        if raw_threads == "":
            new_threads = 0  # sentinel meaning "inherit from global / auto"
        else:
            try:
                new_threads = int(raw_threads)
            except ValueError:
                errors.append("CPU threads: must be blank or a positive integer")
                new_threads = self.current_threads
            else:
                if new_threads < 1:
                    errors.append("CPU threads: must be blank or >= 1")

        raw_mlock = self.query_one("#f-mlock", Input).value.strip().lower()
        if raw_mlock in _BOOL_ALIASES:
            new_mlock = _BOOL_ALIASES[raw_mlock]
        else:
            errors.append(f"mlock: expected yes/no (or true/false, 1/0), got '{raw_mlock}'")
            new_mlock = self.current_mlock

        raw_fa = self.query_one("#f-fa", Input).value.strip().lower()
        if raw_fa in _FA_ALIASES:
            new_fa = _FA_ALIASES[raw_fa]
        else:
            errors.append(
                f"flash attention: expected blank/on/off/auto (or yes/no), got '{raw_fa}'"
            )
            new_fa = self.current_fa

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

        # Free-form llama-server flags. Don't lowercase or alias — pass through
        # verbatim (only strip outer whitespace). Validation happens when
        # llama-server actually starts; a typo shows up in the model's err log.
        new_extra_args = self.query_one("#f-extra-args", Input).value.strip()

        raw_spec = self.query_one("#f-spec-draft", Input).value.strip()
        if raw_spec == "":
            new_spec_override = -1  # sentinel meaning "inherit from global"
        else:
            try:
                new_spec_override = int(raw_spec)
            except ValueError:
                errors.append("MTP draft tokens: must be blank or an integer >= 0")
                new_spec_override = self.current_spec_override
            else:
                if new_spec_override < 0:
                    errors.append("MTP draft tokens: must be blank, 0, or positive")

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
            or m.kv_quant_k != new_kv_k
            or m.kv_quant_v != new_kv_v
            or m.gpu_layers != new_gpu_layers
            or m.parallel_slots != new_parallel
            or m.threads != new_threads
            or m.mlock != new_mlock
            or m.flash_attention != new_fa
            or m.extra_args != new_extra_args
            or m.spec_draft_n_max_override != new_spec_override
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
        m.kv_quant_k = new_kv_k
        m.kv_quant_v = new_kv_v
        m.gpu_layers = new_gpu_layers
        m.parallel_slots = new_parallel
        m.threads = new_threads
        m.mlock = new_mlock
        m.flash_attention = new_fa
        m.extra_args = new_extra_args
        m.spec_draft_n_max_override = new_spec_override
        try:
            registry.save(reg)
        except Exception as e:  # noqa: BLE001
            status.update(f"[red]Save failed: {e}[/red]")
            return
        self.dismiss(True)
