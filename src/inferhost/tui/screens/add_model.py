"""Add-model modal screen."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from inferhost.core import configs, hf, paths, probe, quant, registry
from inferhost.settings import settings


class AddModelScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.files: list[hf.GgufFile] = []
        self.selected_idx: int | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="add-dialog"):
            yield Label("[bold]Add Hugging Face model[/bold]")
            yield Input(placeholder="e.g. Qwen/Qwen2.5-7B-Instruct-GGUF", id="repo-input")
            yield Static("Press Enter to list available GGUF files.", id="hint")
            yield ListView(id="quant-list")
            with Horizontal(id="add-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Add", variant="primary", id="confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Input.Submitted, "#repo-input")
    def _on_submit(self, ev: Input.Submitted) -> None:
        self._fetch(ev.value.strip())

    @work(exclusive=True, thread=True)
    def _fetch(self, repo_id: str) -> None:
        if not repo_id:
            return
        self.app.call_from_thread(self._set_hint, "Fetching file list ...")
        try:
            files = hf.list_ggufs(repo_id)
        except Exception as e:
            self.app.call_from_thread(self._set_hint, f"[red]Error: {e}[/red]")
            return
        if not files:
            self.app.call_from_thread(self._set_hint, "[yellow]No .gguf files found.[/yellow]")
            return
        self.app.call_from_thread(self._populate_files, files)

    def _set_hint(self, text: str) -> None:
        self.query_one("#hint", Static).update(text)

    def _populate_files(self, files: list[hf.GgufFile]) -> None:
        self.files = files
        vram = probe.probe().primary_vram_gib
        budget = vram if vram > 0 else 8.0
        best = quant.pick_best(files, budget)
        list_view = self.query_one("#quant-list", ListView)
        list_view.clear()
        for i, f in enumerate(files):
            marker = "★" if best is not None and f.filename == best.filename else " "
            fits = "✓" if f.size_gib <= max(0.0, budget - 1.5) else "·"
            label = f"{marker} {fits} {f.quant or '?':<8}  {f.size_gib:>5} GiB  {f.filename}"
            list_view.append(ListItem(Label(label), name=str(i)))
        if best is not None:
            self._set_hint(
                f"VRAM: {vram:.1f} GiB. ★ = recommended.  Select a row and press Add."
            )
        else:
            self._set_hint(f"VRAM: {vram:.1f} GiB. No file fits; smallest will be used.")

    @on(ListView.Highlighted, "#quant-list")
    def _on_pick(self, ev: ListView.Highlighted) -> None:
        if ev.item is not None and ev.item.name is not None:
            self.selected_idx = int(ev.item.name)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        if not self.files:
            self._set_hint("[yellow]Enter a repo and press Enter first.[/yellow]")
            return
        idx = self.selected_idx if self.selected_idx is not None else 0
        pick = self.files[idx]
        self._download_and_register(pick)

    @work(exclusive=True, thread=True)
    def _download_and_register(self, pick: hf.GgufFile) -> None:
        self.app.call_from_thread(self._set_hint, f"Downloading {pick.filename} ...")
        try:
            local = hf.download_gguf(pick.repo_id, pick.filename)
            reg = registry.load()
            name = hf.normalize_name(pick.repo_id)
            if pick.quant:
                name = f"{name}-{pick.quant.lower().replace('_', '-')}"
            s = settings()
            paths.ensure_dirs()
            model = registry.Model(
                name=name,
                repo_id=pick.repo_id,
                filename=pick.filename,
                quant=pick.quant,
                ctx=s.default_ctx,
                port=reg.next_port(s.swap_port),
                size_gib=pick.size_gib,
                local_path=str(local),
            )
            reg.add(model)
            registry.save(reg)
            configs.write_all(reg)
        except Exception as e:
            self.app.call_from_thread(self._set_hint, f"[red]Failed: {e}[/red]")
            return
        self.app.call_from_thread(self.dismiss, True)
