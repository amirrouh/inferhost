"""First-launch screen that downloads runtime binaries with a progress bar."""
from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ProgressBar, Static

from inferhost.core import binaries, paths


class InstallScreen(Screen[bool]):
    """Downloads llama-server + llama-swap. Auto-dismisses with True on success."""

    def compose(self) -> ComposeResult:
        with Vertical(id="install-dialog"):
            yield Label("[bold]Setting up inferhost[/bold]")
            yield Static(
                "Downloading runtime binaries (llama.cpp + llama-swap) on first launch.",
                id="install-blurb",
            )
            yield Static("Starting ...", id="install-status")
            yield ProgressBar(total=100, show_eta=False, id="install-bar")

    def on_mount(self) -> None:
        paths.ensure_dirs()
        self._run_install()

    @work(exclusive=True, thread=True)
    def _run_install(self) -> None:
        try:
            self.app.call_from_thread(self._set_status, "Downloading llama-server ...")
            self.app.call_from_thread(self._reset_bar)
            binaries.install_llama_server(progress_cb=self._cb("llama-server"))
            self.app.call_from_thread(self._set_status, "Downloading llama-swap ...")
            self.app.call_from_thread(self._reset_bar)
            binaries.install_llama_swap(progress_cb=self._cb("llama-swap"))
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._set_status, f"[red]Install failed: {e}[/red]")
            return
        self.app.call_from_thread(self.dismiss, True)

    def _cb(self, label: str):
        def fn(done: int, total: int) -> None:
            self.app.call_from_thread(self._update_progress, label, done, total)
        return fn

    def _reset_bar(self) -> None:
        bar = self.query_one("#install-bar", ProgressBar)
        bar.update(total=100, progress=0)

    def _update_progress(self, label: str, done: int, total: int) -> None:
        bar = self.query_one("#install-bar", ProgressBar)
        if total > 0:
            bar.update(total=total, progress=done)
            mb = done / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._set_status(f"Downloading {label} ... {mb:.1f} / {mb_total:.1f} MiB")
        else:
            mb = done / (1024 * 1024)
            self._set_status(f"Downloading {label} ... {mb:.1f} MiB")

    def _set_status(self, text: str) -> None:
        self.query_one("#install-status", Static).update(text)
