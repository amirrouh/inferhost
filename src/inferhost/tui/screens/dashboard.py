"""Dashboard screen: model list, details, logs."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Log, Static

from inferhost.core import configs, processes, registry
from inferhost.core.logs import log_path, tail
from inferhost.settings import settings
from inferhost.tui.screens.add_model import AddModelScreen


class DashboardScreen(Screen):
    BINDINGS = [
        ("a", "add_model", "Add"),
        ("s", "start_swap", "Start"),
        ("x", "stop_swap", "Stop"),
        ("r", "restart_swap", "Restart"),
        ("d", "remove_model", "Remove"),
        ("delete", "remove_model", "Remove"),
        ("R", "refresh", "Refresh"),
    ]

    selected_name: reactive[str | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(self._endpoint_text(), id="endpoint-bar")
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Models")
                yield ListView(id="model-list")
            with Vertical(id="details-pane"):
                yield Static("Select a model", id="details")
                yield Log(id="logs", highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_models()
        self.set_interval(2.0, self._tick)

    # ---- data refresh ----

    def _endpoint_text(self) -> str:
        s = settings()
        swap = processes.swap_status()
        marker = "[green]●[/green]" if swap.running else "[red]○[/red]"
        return f"{marker} llama-swap http://localhost:{s.swap_port}/v1"

    def refresh_models(self) -> None:
        reg = registry.load()
        list_view = self.query_one("#model-list", ListView)
        list_view.clear()
        for m in reg.models:
            list_view.append(ListItem(Label(f"{m.name}  ({m.quant or '?'})"), name=m.name))
        if reg.models and self.selected_name is None:
            self.selected_name = reg.models[0].name
        self._refresh_endpoint()
        self._refresh_details()

    def _refresh_endpoint(self) -> None:
        try:
            self.query_one("#endpoint-bar", Static).update(self._endpoint_text())
        except Exception:
            pass

    def _refresh_details(self) -> None:
        details = self.query_one("#details", Static)
        log_widget = self.query_one("#logs", Log)
        if self.selected_name is None:
            details.update("No model selected.")
            return
        reg = registry.load()
        m = reg.get(self.selected_name)
        if m is None:
            details.update("Model not found.")
            return
        details.update(
            f"[bold]{m.name}[/bold]\n"
            f"repo:   {m.repo_id}\n"
            f"file:   {m.filename}\n"
            f"quant:  {m.quant or '?'}    size: {m.size_gib} GiB    ctx: {m.ctx}\n"
            f"port:   {m.port}    endpoint: http://localhost:{settings().swap_port}/v1\n"
            f"model:  {m.local_path}"
        )
        log_widget.clear()
        path = log_path("swap")
        for line in tail(path, 200):
            log_widget.write_line(line)

    def _tick(self) -> None:
        self._refresh_endpoint()
        log_widget = self.query_one("#logs", Log)
        path = log_path("swap")
        if path.exists():
            current = path.read_text(errors="replace").splitlines()
            shown = log_widget.line_count
            if len(current) > shown:
                for line in current[shown:]:
                    log_widget.write_line(line)

    # ---- handlers ----

    @on(ListView.Highlighted, "#model-list")
    def _on_highlight(self, ev: ListView.Highlighted) -> None:
        item = ev.item
        if item is not None:
            self.selected_name = item.name
            self._refresh_details()

    @on(ListView.Selected, "#model-list")
    def _on_select(self, ev: ListView.Selected) -> None:
        item = ev.item
        if item is not None:
            self.selected_name = item.name
            self._refresh_details()

    # ---- actions ----

    def action_add_model(self) -> None:
        self.app.push_screen(AddModelScreen(), self._after_add)

    def _after_add(self, added: bool | None) -> None:
        if added:
            self.refresh_models()

    def action_remove_model(self) -> None:
        if self.selected_name is None:
            return
        reg = registry.load()
        if reg.remove(self.selected_name):
            registry.save(reg)
            configs.write_all(reg)
        self.selected_name = None
        self.refresh_models()

    def action_start_swap(self) -> None:
        try:
            reg = registry.load()
            configs.write_all(reg)
            processes.start_swap()
        except Exception as e:
            self.notify(f"Start failed: {e}", severity="error")
        self._refresh_endpoint()

    def action_stop_swap(self) -> None:
        processes.stop_swap()
        self._refresh_endpoint()

    def action_restart_swap(self) -> None:
        processes.stop_swap()
        try:
            reg = registry.load()
            configs.write_all(reg)
            processes.start_swap()
        except Exception as e:
            self.notify(f"Restart failed: {e}", severity="error")
        self._refresh_endpoint()

    def action_refresh(self) -> None:
        self.refresh_models()
