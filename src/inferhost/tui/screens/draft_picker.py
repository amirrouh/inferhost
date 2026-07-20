"""Draft-picker modal for DFlash speculative decoding.

Attaches a z-lab block-diffusion *draft* GGUF to an existing target model:
paste the draft repo URL (or let Suggest prefill it from the built-in pairing
table), pick the quant that fits alongside the target, download it with
progress, then write the three draft fields onto the target and re-render the
configs. A thin clone of AddModelScreen's fetch/list/download half — the target
already exists, so there's no name/port/registration step, just an attachment.
"""
from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Static,
)

from inferhost.core import configs, hf, probe, quant, registry, vram
from inferhost.core.dflash_recipes import suggest_gguf_repo

# Overhead subtracted from the pick budget so a draft doesn't claim the last
# scrap of free VRAM (the target's KV cache still needs to grow). Small — drafts
# are 0.4-2.1B, so this rarely changes the pick.
_DRAFT_OVERHEAD_GIB = 0.5


def best_draft_pick(draft_repo: str) -> hf.GgufFile:
    """List a draft repo and return the best quant that fits in free VRAM.

    Ranks with :func:`quant.pick_best` against whatever VRAM is currently free
    (falls back to primary total, then a small default when there's no GPU),
    so the chosen draft coexists with the target rather than blowing the budget.
    Raises if the repo has no GGUFs. Shared by the Browse/Suggest picker and the
    dashboard's `f`-key express lane so both pick identically.
    """
    files = hf.list_ggufs(draft_repo)
    if not files:
        raise RuntimeError(f"No GGUF files found in draft repo {draft_repo}.")
    free = vram.free_vram_gib()
    if free == float("inf"):
        free = probe.probe().primary_vram_gib
    budget = free if free > 0 else 8.0
    return quant.pick_best(files, budget, overhead_gib=_DRAFT_OVERHEAD_GIB) or files[0]


def download_draft(pick: hf.GgufFile, progress_cb) -> Path:
    """Download a draft pick (single-file or multi-part), returning shard 1's path.

    Multi-part drafts are downloaded shard-by-shard; llama-server auto-discovers
    the siblings from shard 1, so only shard 1's path is stored in
    ``draft_model_path`` (mirrors how the main model handles multi-part GGUFs).
    """
    if pick.parts:
        return hf.download_gguf_parts_with_progress(
            repo_id=pick.repo_id, parts=pick.parts, progress_cb=progress_cb
        )
    return hf.download_gguf_with_progress(
        repo_id=pick.repo_id,
        filename=pick.filename,
        expected_bytes=max(pick.size_bytes, 1),
        progress_cb=progress_cb,
    )


def attach_draft(model_name: str, pick: hf.GgufFile, local_path: str) -> None:
    """Write the draft fields onto ``model_name``, save, and re-render configs.

    Central so the picker and the dashboard express lane persist a draft the
    same way. ``draft_size_gib`` stores the summed shard size (already summed in
    ``GgufFile.size_gib`` for multi-part picks), which feeds the VRAM estimate.
    """
    reg = registry.load()
    m = reg.get(model_name)
    if m is None:
        raise RuntimeError(f"Model '{model_name}' no longer exists.")
    m.draft_model_path = local_path
    m.draft_repo_id = pick.repo_id
    m.draft_size_gib = pick.size_gib
    registry.save(reg)
    configs.write_all(reg)


class DraftPickerScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model_name: str, prefill_repo: str = "") -> None:
        super().__init__()
        self.model_name = model_name
        self.prefill_repo = prefill_repo
        self.files: list[hf.GgufFile] = []
        self.selected_idx: int | None = None
        self.downloading: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="add-dialog"):
            yield Label(f"[bold]Attach DFlash draft to {self.model_name}[/bold]")
            yield Input(
                placeholder="Paste the draft's Hugging Face URL (or owner/repo)",
                id="repo-input",
            )
            yield Static(
                "Paste a DFlash draft repo and press Enter to list its GGUF files. "
                "The draft accelerates this target via speculative decoding.",
                id="hint",
            )
            yield ListView(id="quant-list")
            yield Static("", id="dl-status")
            yield ProgressBar(total=100, show_eta=False, id="dl-bar")
            with Horizontal(id="add-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Attach", variant="primary", id="confirm")

    def on_mount(self) -> None:
        self._set_bar_visible(False)
        if self.prefill_repo:
            self.query_one("#repo-input", Input).value = self.prefill_repo
            self._fetch(hf.parse_repo_id(self.prefill_repo))

    def action_cancel(self) -> None:
        if self.downloading:
            return
        self.dismiss(False)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        if self.downloading:
            return
        self.dismiss(False)

    @on(Input.Submitted, "#repo-input")
    def _on_submit(self, ev: Input.Submitted) -> None:
        self._fetch(hf.parse_repo_id(ev.value))

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
            suggested = suggest_gguf_repo(repo_id)
            suggested_files: list[hf.GgufFile] = []
            if suggested:
                try:
                    suggested_files = hf.list_ggufs(suggested)
                except Exception:  # noqa: BLE001
                    suggested_files = []
            if suggested and suggested_files:
                # Populate first — _populate_files sets its own (VRAM) hint —
                # then overwrite the hint with the redirect explanation so it
                # sticks as the final, visible message.
                self.app.call_from_thread(self._set_repo_input, suggested)
                self.app.call_from_thread(self._populate_files, suggested_files)
                self.app.call_from_thread(
                    self._set_hint,
                    f"[yellow]{repo_id} has no GGUF files (raw safetensors draft for vLLM). "
                    f"Showing the paired GGUF conversion {suggested} instead.[/yellow]",
                )
                return
            self.app.call_from_thread(
                self._set_hint,
                "[bold yellow]No GGUF files in that repo.[/bold yellow] llama.cpp needs a "
                'GGUF draft — search Hugging Face for a "-GGUF" conversion of it.',
            )
            return
        self.app.call_from_thread(self._populate_files, files)

    def _set_hint(self, text: str) -> None:
        self.query_one("#hint", Static).update(text)

    def _set_repo_input(self, repo_id: str) -> None:
        self.query_one("#repo-input", Input).value = repo_id

    def _populate_files(self, files: list[hf.GgufFile]) -> None:
        self.files = files
        free = vram.free_vram_gib()
        if free == float("inf"):
            free = probe.probe().primary_vram_gib
        budget = free if free > 0 else 8.0
        best = quant.pick_best(files, budget, overhead_gib=_DRAFT_OVERHEAD_GIB)
        list_view = self.query_one("#quant-list", ListView)
        list_view.clear()
        for i, f in enumerate(files):
            marker = "*" if best is not None and f.filename == best.filename else " "
            fits = "+" if f.size_gib <= max(0.0, budget - _DRAFT_OVERHEAD_GIB) else "."
            parts_tag = f"  [{len(f.parts)} parts]" if f.parts else ""
            label = f"{marker} {fits} {f.quant or '?':<8}  {f.size_gib:>5} GiB  {f.filename}{parts_tag}"
            list_view.append(ListItem(Label(label), name=str(i)))
        if best is not None:
            self._set_hint(f"Free VRAM: {free:.1f} GiB. * = recommended.  Select a row and press Attach.")
        else:
            self._set_hint(f"Free VRAM: {free:.1f} GiB. No file fits; smallest will be used.")

    @on(ListView.Highlighted, "#quant-list")
    def _on_pick(self, ev: ListView.Highlighted) -> None:
        if ev.item is not None and ev.item.name is not None:
            self.selected_idx = int(ev.item.name)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        if self.downloading:
            return
        if not self.files:
            self._set_hint("[yellow]Enter a draft repo and press Enter first.[/yellow]")
            return
        idx = self.selected_idx if self.selected_idx is not None else 0
        pick = self.files[idx]
        self.downloading = True
        self._set_bar_visible(True)
        self._download_and_attach(pick)

    def _set_bar_visible(self, visible: bool) -> None:
        self.query_one("#dl-bar", ProgressBar).display = visible
        self.query_one("#dl-status", Static).display = visible

    @work(exclusive=True, thread=True)
    def _download_and_attach(self, pick: hf.GgufFile) -> None:
        try:
            self.app.call_from_thread(self._set_hint, f"Downloading draft {pick.filename} ...")
            self.app.call_from_thread(self._update_progress, 0, max(pick.size_bytes, 1))
            local = download_draft(
                pick,
                progress_cb=lambda done, total: self.app.call_from_thread(
                    self._update_progress, done, total or max(pick.size_bytes, 1)
                ),
            )
            attach_draft(self.model_name, pick, str(local))
            # NOTE: no daemon reload here — the caller (model_settings /
            # dashboard) reloads off-thread after this modal dismisses, same
            # contract as the add-model modal.
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
            status.update(f"{done / (1024 * 1024):.1f} MiB")
