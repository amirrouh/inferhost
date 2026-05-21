"""Add-model modal screen."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, ProgressBar, Static

from inferhost.core import configs, hf, paths, probe, processes, quant, registry
from inferhost.settings import settings


class AddModelScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.files: list[hf.GgufFile] = []
        self.selected_idx: int | None = None
        self.downloading: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="add-dialog"):
            yield Label("[bold]Add Hugging Face model[/bold]")
            yield Input(placeholder="e.g. Qwen/Qwen2.5-7B-Instruct-GGUF", id="repo-input")
            yield Static("Press Enter to list available GGUF files.", id="hint")
            yield ListView(id="quant-list")
            yield Static("", id="dl-status")
            yield ProgressBar(total=100, show_eta=False, id="dl-bar")
            with Horizontal(id="add-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Add", variant="primary", id="confirm")

    def on_mount(self) -> None:
        self._set_bar_visible(False)

    def action_cancel(self) -> None:
        if self.downloading:
            return
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
        except Exception as e:  # noqa: BLE001
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
            marker = "*" if best is not None and f.filename == best.filename else " "
            fits = "+" if f.size_gib <= max(0.0, budget - 1.5) else "."
            label = f"{marker} {fits} {f.quant or '?':<8}  {f.size_gib:>5} GiB  {f.filename}"
            list_view.append(ListItem(Label(label), name=str(i)))
        if best is not None:
            self._set_hint(
                f"VRAM: {vram:.1f} GiB. * = recommended.  Select a row and press Add."
            )
        else:
            self._set_hint(f"VRAM: {vram:.1f} GiB. No file fits; smallest will be used.")

    @on(ListView.Highlighted, "#quant-list")
    def _on_pick(self, ev: ListView.Highlighted) -> None:
        if ev.item is not None and ev.item.name is not None:
            self.selected_idx = int(ev.item.name)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        if self.downloading:
            return
        self.dismiss(False)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        if self.downloading:
            return
        if not self.files:
            self._set_hint("[yellow]Enter a repo and press Enter first.[/yellow]")
            return
        idx = self.selected_idx if self.selected_idx is not None else 0
        pick = self.files[idx]
        self.downloading = True
        self._set_bar_visible(True)
        self._download_and_register(pick)

    def _set_bar_visible(self, visible: bool) -> None:
        bar = self.query_one("#dl-bar", ProgressBar)
        status = self.query_one("#dl-status", Static)
        bar.display = visible
        status.display = visible

    @work(exclusive=True, thread=True)
    def _download_and_register(self, pick: hf.GgufFile) -> None:
        self.app.call_from_thread(self._set_hint, f"Downloading {pick.filename} ...")
        self.app.call_from_thread(self._update_progress, 0, max(pick.size_bytes, 1))
        try:
            local = hf.download_gguf_with_progress(
                repo_id=pick.repo_id,
                filename=pick.filename,
                expected_bytes=max(pick.size_bytes, 1),
                progress_cb=lambda done, total: self.app.call_from_thread(
                    self._update_progress, done, total or max(pick.size_bytes, 1)
                ),
            )
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
            processes.reload_if_running()
        except Exception as e:  # noqa: BLE001
            self.downloading = False
            self.app.call_from_thread(self._set_hint, f"[red]Failed: {e}[/red]")
            return
        self.downloading = False
        self.app.call_from_thread(self.dismiss, True)

    def _update_progress(self, done: int, total: int) -> None:
        bar = self.query_one("#dl-bar", ProgressBar)
        status = self.query_one("#dl-status", Static)
        if total > 0:
            bar.update(total=total, progress=min(done, total))
            mib = done / (1024 * 1024)
            mib_total = total / (1024 * 1024)
            pct = (done / total) * 100 if total else 0.0
            status.update(f"{mib:.1f} / {mib_total:.1f} MiB  ({pct:.1f}%)")
        else:
            mib = done / (1024 * 1024)
            status.update(f"{mib:.1f} MiB")
