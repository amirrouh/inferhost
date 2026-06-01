"""Image-model component editor + a reusable repo→list file picker.

Image models (especially Flux / SD3 / Z-Image / Qwen-Image) are assembled from
several files — a diffusion model plus a VAE and one or more text encoders — that
often live in *different* Hugging Face repos. This screen lets the user fill each
component slot with the same "paste a repo URL → pick from the list" flow used for
the main model, instead of hunting down files and typing paths.

Image models route here from the dashboard's Configure action (chat models keep
the existing ModelSettingsScreen).
"""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from inferhost.core import configs, hf, processes, registry

# Component slots: (Model field, label, sd-server flag shown to the user).
_SLOTS: list[tuple[str, str, str]] = [
    ("vae_path", "VAE", "--vae"),
    ("text_encoder_path", "Text encoder (Qwen / LLM)", "--llm"),
    ("vision_encoder_path", "Vision encoder / mmproj (Qwen-Image-Edit)", "--llm_vision"),
    ("clip_l_path", "CLIP-L", "--clip_l"),
    ("clip_g_path", "CLIP-G", "--clip_g"),
    ("t5xxl_path", "T5XXL", "--t5xxl"),
]


class RepoFilePickerScreen(ModalScreen[tuple[str, str] | None]):
    """Pick one file from a HF repo. Dismisses with (repo_id, filename) or None."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self.files: list[hf.GgufFile] = []
        self.selected_idx: int | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="add-dialog"):
            yield Label(f"[bold]Pick file — {self._title}[/bold]")
            yield Input(placeholder="Hugging Face repo, e.g. black-forest-labs/FLUX.1-schnell", id="rfp-repo")
            yield Static("Press Enter to list files.", id="rfp-hint")
            yield ListView(id="rfp-list")
            with Horizontal(id="add-buttons"):
                yield Button("Cancel", id="rfp-cancel")
                yield Button("Select", variant="primary", id="rfp-confirm")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#rfp-repo")
    def _on_repo(self, ev: Input.Submitted) -> None:
        self._fetch(ev.value.strip())

    @work(exclusive=True, thread=True)
    def _fetch(self, repo_id: str) -> None:
        if not repo_id:
            return
        self.app.call_from_thread(self._set_hint, "Fetching file list ...")
        try:
            files = hf.list_repo_files(repo_id)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._set_hint, f"[red]Error: {e}[/red]")
            return
        if not files:
            self.app.call_from_thread(self._set_hint, "[yellow]No .gguf/.safetensors files.[/yellow]")
            return
        self.app.call_from_thread(self._populate, files)

    def _set_hint(self, text: str) -> None:
        self.query_one("#rfp-hint", Static).update(text)

    def _populate(self, files: list[hf.GgufFile]) -> None:
        self.files = files
        lv = self.query_one("#rfp-list", ListView)
        lv.clear()
        for i, f in enumerate(files):
            lv.append(ListItem(Label(f"{f.size_gib:>6} GiB  {f.filename}"), name=str(i)))
        self._set_hint("Select a file and press Select.")

    @on(ListView.Highlighted, "#rfp-list")
    def _on_pick(self, ev: ListView.Highlighted) -> None:
        if ev.item is not None and ev.item.name is not None:
            self.selected_idx = int(ev.item.name)

    @on(Button.Pressed, "#rfp-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#rfp-confirm")
    def _confirm(self) -> None:
        if not self.files:
            self._set_hint("[yellow]Enter a repo and press Enter first.[/yellow]")
            return
        idx = self.selected_idx if self.selected_idx is not None else 0
        f = self.files[idx]
        self.dismiss((f.repo_id, f.filename))


class ImageComponentsScreen(ModalScreen[bool]):
    """Edit an image model's component files (VAE / encoders) + extra args."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        m = registry.load().get(model_name)
        # Working copy of slot values, mutated as the user picks files.
        self.values: dict[str, str] = {
            field: (getattr(m, field, "") if m is not None else "") for field, _, _ in _SLOTS
        }
        self.current_extra = m.extra_args if m is not None else ""
        self.busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="model-settings-dialog"):
            yield Label("[bold]Image model components[/bold]")
            yield Static(
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                "Fill the slots your model needs (Flux: VAE+CLIP-L+T5XXL · "
                "Z-Image/Qwen-Image: VAE+Text encoder). Single-file checkpoints "
                "need none. Pick fetches from any repo. Daemons reload on save.",
                id="model-settings-blurb",
            )
            for field, label, flag in _SLOTS:
                yield Label(f"{label}  [grey50]({flag})[/grey50]")
                with Horizontal(classes="slot-row"):
                    yield Static(self._slot_text(field), id=f"val-{field}", classes="slot-val")
                    yield Button("Pick", id=f"pick-{field}", classes="slot-pick")
                    yield Button("Clear", id=f"clear-{field}", classes="slot-clear")
            yield Label("Extra sd-server args (raw)")
            yield Input(
                value=self.current_extra,
                placeholder='e.g. "--steps 8 --cfg-scale 1.0 --sampling-method euler"',
                id="f-img-extra",
            )
            yield Static("", id="model-settings-status")
            with Horizontal(id="model-settings-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="confirm")

    def _slot_text(self, field: str) -> str:
        v = self.values.get(field, "")
        if not v:
            return "[grey50](not set)[/grey50]"
        # Show just the filename to keep the row short.
        return v.rsplit("/", 1)[-1]

    def _set_status(self, text: str) -> None:
        self.query_one("#model-settings-status", Static).update(text)

    def action_cancel(self) -> None:
        if not self.busy:
            self.dismiss(False)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        if not self.busy:
            self.dismiss(False)

    @on(Button.Pressed, ".slot-pick")
    def _on_pick(self, ev: Button.Pressed) -> None:
        if self.busy:
            return
        field = ev.button.id.removeprefix("pick-")
        label = next(lbl for f, lbl, _ in _SLOTS if f == field)
        self.app.push_screen(RepoFilePickerScreen(label), lambda r, fld=field: self._after_pick(fld, r))

    def _after_pick(self, field: str, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        repo_id, filename = result
        self.busy = True
        self._set_status(f"Downloading {filename} ...")
        self._download(field, repo_id, filename)

    @work(exclusive=True, thread=True)
    def _download(self, field: str, repo_id: str, filename: str) -> None:
        try:
            local = hf.download_gguf(repo_id, filename)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._set_status, f"[red]Download failed: {e}[/red]")
            self.busy = False
            return
        self.values[field] = str(local)
        self.app.call_from_thread(self._refresh_slot, field)
        self.app.call_from_thread(self._set_status, f"Set {field} → {filename}")
        self.busy = False

    def _refresh_slot(self, field: str) -> None:
        self.query_one(f"#val-{field}", Static).update(self._slot_text(field))

    @on(Button.Pressed, ".slot-clear")
    def _on_clear(self, ev: Button.Pressed) -> None:
        if self.busy:
            return
        field = ev.button.id.removeprefix("clear-")
        self.values[field] = ""
        self._refresh_slot(field)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        if self.busy:
            return
        reg = registry.load()
        m = reg.get(self.model_name)
        if m is None:
            self.dismiss(False)
            return
        for field in self.values:
            setattr(m, field, self.values[field])
        m.extra_args = self.query_one("#f-img-extra", Input).value.strip()
        registry.save(reg)
        configs.write_all(reg)
        # Re-warm pinned models so reconfiguring one model doesn't leave the
        # others cold after the reload evicts everything.
        processes.reload_and_warm_pinned()
        self.dismiss(True)
