"""Dashboard screen: full control surface for inferhost.

Shows the running state of every daemon, every key setting, and the model
registry — and exposes every action (add / rename / remove / start / stop /
restart / gateway toggle / settings) through single-key bindings.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Log, Static

from inferhost.core import configs, processes, registry
from inferhost.core.logs import log_path, tail
from inferhost.settings import reload_settings, settings
from inferhost.tui.screens.add_model import AddModelScreen
from inferhost.tui.screens.model_settings import ModelSettingsScreen
from inferhost.tui.screens.rename import RenameScreen
from inferhost.tui.screens.settings import SettingsScreen


class DashboardScreen(Screen):
    # `show=False` keeps the binding active but hides it from Textual's auto-built
    # Footer — we render our own two-row action bar below.
    BINDINGS = [
        ("a", "add_model", "Add"),
        ("n", "rename_model", "Rename"),
        ("c", "configure_model", "Config"),
        ("d", "remove_model", "Del"),
        ("delete", "remove_model", "Del"),
        ("s", "start_swap", "Start"),
        ("x", "stop_swap", "Stop"),
        ("r", "restart_swap", "Restart"),
        ("g", "toggle_gateway", "Gateway"),
        ("p", "open_settings", "Prefs"),
        ("R", "refresh", "Refresh"),
    ]

    # Two-row docked action bar. Keep groups balanced so neither row is too wide
    # for a narrow terminal; the row Static widgets wrap rather than clipping.
    _ROW1 = "[b]a[/b] Add   [b]n[/b] Rename   [b]c[/b] Config   [b]d[/b] Del"
    _ROW2 = (
        "[b]s[/b] Start   [b]x[/b] Stop   [b]r[/b] Restart   "
        "[b]g[/b] Gateway   [b]p[/b] Prefs   [b]R[/b] Refresh   [b]q[/b] Quit"
    )

    selected_name: reactive[str | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Models")
                yield ListView(id="model-list")
            with Vertical(id="details-pane"):
                yield Static("Select a model", id="details")
                yield Log(id="logs", highlight=False)
        with Vertical(id="actions"):
            yield Static(self._ROW1, id="action-row-1")
            yield Static(self._ROW2, id="action-row-2")

    def on_mount(self) -> None:
        self.refresh_models()
        self.set_interval(2.0, self._tick)

    # ---- status bar ----

    def _status_text(self) -> str:
        s = settings()
        swap = processes.swap_status()
        gw = processes.gateway_status()
        gw_available = processes.gateway_available()

        swap_dot = "[green]●[/green]" if swap.running else "[red]○[/red]"
        if gw_available:
            gw_dot = "[green]●[/green]" if gw.running else "[red]○[/red]"
            gw_suffix = ""
        else:
            gw_dot = "[grey50]○[/grey50]"
            gw_suffix = " (not installed)"
        # Single compact line: daemon dots + ports + a couple of key settings.
        # Anything finer-grained lives behind `p` (Settings).
        return (
            f"{swap_dot} :{s.swap_port}  "
            f"{gw_dot} :{s.gateway_port}{gw_suffix}  "
            f"│ ctx={s.default_ctx} slots={s.parallel_slots} ngl={s.gpu_layers} fa={s.flash_attention}"
        )

    def _refresh_bars(self) -> None:
        try:
            self.query_one("#status-bar", Static).update(self._status_text())
        except Exception:  # noqa: BLE001
            pass

    # ---- data refresh ----

    def refresh_models(self) -> None:
        reg = registry.load()
        list_view = self.query_one("#model-list", ListView)
        list_view.clear()
        for m in reg.models:
            list_view.append(ListItem(Label(f"{m.name}  ({m.quant or '?'})"), name=m.name))
        if self.selected_name is None and reg.models:
            self.selected_name = reg.models[0].name
        elif self.selected_name is not None and reg.get(self.selected_name) is None:
            self.selected_name = reg.models[0].name if reg.models else None
        self._refresh_bars()
        self._refresh_details()

    def _refresh_details(self) -> None:
        details = self.query_one("#details", Static)
        log_widget = self.query_one("#logs", Log)
        if self.selected_name is None:
            details.update("No models registered yet — press [bold]a[/bold] to add one.")
            log_widget.clear()
            return
        reg = registry.load()
        m = reg.get(self.selected_name)
        if m is None:
            details.update("Model not found.")
            return
        s = settings()
        kv = ""
        if m.cache_type_k or m.cache_type_v:
            kv = (
                f"    kv: K={m.cache_type_k or 'f16'} V={m.cache_type_v or 'f16'}"
            )
        details.update(
            f"[bold]{m.name}[/bold]\n"
            f"repo:     {m.repo_id}\n"
            f"file:     {m.filename}\n"
            f"quant:    {m.quant or '?'}    size: {m.size_gib} GiB    ctx: {m.ctx}{kv}\n"
            f"backend:  port {m.port}  ->  swap http://localhost:{s.swap_port}/v1\n"
            f"path:     {m.local_path}"
        )
        log_widget.clear()
        path = log_path("swap")
        for line in tail(path, 200):
            log_widget.write_line(line)

    def _tick(self) -> None:
        self._refresh_bars()
        log_widget = self.query_one("#logs", Log)
        path = log_path("swap")
        if path.exists():
            current = path.read_text(errors="replace").splitlines()
            shown = log_widget.line_count
            if len(current) > shown:
                for line in current[shown:]:
                    log_widget.write_line(line)

    # ---- list handlers ----

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

    # ---- actions: models ----

    def action_add_model(self) -> None:
        self.app.push_screen(AddModelScreen(), self._after_add)

    def _after_add(self, added: bool | None) -> None:
        if added:
            self.refresh_models()

    def action_rename_model(self) -> None:
        if self.selected_name is None:
            self.notify("Select a model first.", severity="warning")
            return
        self.app.push_screen(RenameScreen(self.selected_name), self._after_rename)

    def _after_rename(self, new_name: str | None) -> None:
        if not new_name:
            return
        self.selected_name = new_name
        # Reload every running daemon so swap AND the gateway see the new alias.
        try:
            configs.write_all(registry.load())
            swap_reloaded, gw_reloaded = processes.reload_if_running()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Renamed, but reload failed: {e}", severity="error")
        else:
            if swap_reloaded or gw_reloaded:
                self.notify(f"Renamed and reloaded as '{new_name}'.")
            else:
                self.notify(f"Renamed to '{new_name}'.")
        self.refresh_models()

    def action_configure_model(self) -> None:
        if self.selected_name is None:
            self.notify("Select a model first.", severity="warning")
            return
        self.app.push_screen(
            ModelSettingsScreen(self.selected_name), self._after_configure
        )

    def _after_configure(self, saved: bool | None) -> None:
        if not saved:
            return
        try:
            configs.write_all(registry.load())
            swap_reloaded, gw_reloaded = processes.reload_if_running()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Saved, but reload failed: {e}", severity="error")
        else:
            if swap_reloaded or gw_reloaded:
                self.notify("Model settings saved; daemons reloaded.")
            else:
                self.notify("Model settings saved.")
        self.refresh_models()

    def action_remove_model(self) -> None:
        if self.selected_name is None:
            return
        reg = registry.load()
        if reg.remove(self.selected_name):
            registry.save(reg)
            configs.write_all(reg)
            try:
                processes.reload_if_running()
            except Exception as e:  # noqa: BLE001
                self.notify(f"Removed, but reload failed: {e}", severity="error")
        self.selected_name = None
        self.refresh_models()

    # ---- actions: swap ----

    def action_start_swap(self) -> None:
        try:
            reg = registry.load()
            configs.write_all(reg)
            processes.start_swap()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Start failed: {e}", severity="error")
        self._refresh_bars()

    def action_stop_swap(self) -> None:
        processes.stop_swap()
        self._refresh_bars()

    def action_restart_swap(self) -> None:
        processes.stop_swap()
        try:
            reg = registry.load()
            configs.write_all(reg)
            processes.start_swap()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Restart failed: {e}", severity="error")
        self._refresh_bars()

    # ---- actions: gateway ----

    def action_toggle_gateway(self) -> None:
        if not processes.gateway_available():
            self.notify(
                "LiteLLM not installed. Install with: pip install 'inferhost[gateway]'",
                severity="warning",
            )
            return
        st = processes.gateway_status()
        try:
            if st.running:
                processes.stop_gateway()
                self.notify("Gateway stopped.")
            else:
                reg = registry.load()
                configs.write_all(reg)
                processes.start_gateway()
                self.notify("Gateway started.")
        except Exception as e:  # noqa: BLE001
            self.notify(f"Gateway action failed: {e}", severity="error")
        self._refresh_bars()

    # ---- actions: settings ----

    def action_open_settings(self) -> None:
        self.app.push_screen(SettingsScreen(), self._after_settings)

    def _after_settings(self, saved: bool | None) -> None:
        if not saved:
            return
        reload_settings()
        # Regenerate configs so the new ctx/gpu values are baked into llama-swap.yaml.
        try:
            configs.write_all(registry.load())
        except Exception as e:  # noqa: BLE001
            self.notify(f"Saved, but re-rendering configs failed: {e}", severity="error")
            return
        if processes.swap_status().running:
            self.notify("Saved. Press [bold]r[/bold] to restart llama-swap and pick up the changes.")
        else:
            self.notify("Settings saved.")
        self._refresh_bars()

    def action_refresh(self) -> None:
        self.refresh_models()
