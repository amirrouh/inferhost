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
from textual.widgets import Button, Label, ListItem, ListView, Log, Static

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
        ("c", "configure_model", "Configure"),
        ("d", "remove_model", "Delete"),
        ("delete", "remove_model", "Delete"),
        ("s", "start_swap", "Start"),
        ("x", "stop_swap", "Stop"),
        ("r", "restart_swap", "Restart"),
        ("g", "toggle_gateway", "Gateway"),
        ("p", "open_settings", "Settings"),
        ("R", "refresh", "Refresh"),
    ]

    # Two-row docked action bar. Each "button" is a Button widget so mouse
    # clicks land on the right action; the keyboard shortcut is shown inline
    # in the label. The dispatch table maps button id → bound action name.
    _BUTTONS: tuple[tuple[int, str, str, str], ...] = (
        # (row, btn_id, label, action)
        (1, "btn-add",     "a Add",       "add_model"),
        (1, "btn-rename",  "n Rename",    "rename_model"),
        (1, "btn-config",  "c Configure", "configure_model"),
        (1, "btn-del",     "d Delete",    "remove_model"),
        (2, "btn-start",   "s Start",     "start_swap"),
        (2, "btn-stop",    "x Stop",      "stop_swap"),
        (2, "btn-restart", "r Restart",   "restart_swap"),
        (2, "btn-gw",      "g Gateway",   "toggle_gateway"),
        (2, "btn-prefs",   "p Settings",  "open_settings"),
        (2, "btn-refresh", "R Refresh",   "refresh"),
        (2, "btn-quit",    "q Quit",      "quit"),
    )

    selected_name: reactive[str | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Static("", id="gpu-bar")
        yield Static("", id="status-bar")
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Models", id="sidebar-label")
                yield ListView(id="model-list")
            with Vertical(id="details-pane"):
                yield Static("Select a model", id="details")
                yield Log(id="logs", highlight=False)
        with Vertical(id="actions"):
            with Horizontal(id="action-row-1"):
                for row, btn_id, label, _action in self._BUTTONS:
                    if row == 1:
                        yield Button(label, id=btn_id, classes="action-btn")
            with Horizontal(id="action-row-2"):
                for row, btn_id, label, _action in self._BUTTONS:
                    if row == 2:
                        yield Button(label, id=btn_id, classes="action-btn")

    def on_mount(self) -> None:
        self.refresh_models()
        self.set_interval(2.0, self._tick)

    # ---- status bar ----

    def _status_text(self) -> str:
        s = settings()
        swap = processes.swap_status()
        gw = processes.gateway_status()
        gw_available = processes.gateway_available()
        reg = registry.load()
        n_models = len(reg.models)

        swap_dot = "[green]●[/green]" if swap.running else "[red]○[/red]"
        if gw_available:
            gw_dot = "[green]●[/green]" if gw.running else "[red]○[/red]"
            gw_suffix = ""
        else:
            gw_dot = "[grey50]○[/grey50]"
            gw_suffix = " (not installed)"

        # ctx tracks the SELECTED model's actual ctx (what llama-server runs
        # with), not the global new-model default. When no model is selected,
        # fall back to the default and label it explicitly so users don't
        # mistake it for an active context.
        sel = reg.get(self.selected_name) if self.selected_name else None
        ctx_part = f"ctx={sel.ctx}" if sel is not None else f"ctx={s.default_ctx} (default)"

        # Which model is actually resident in VRAM right now (llama-swap
        # /running). Empty when swap is down or no model has been hit yet.
        loaded = processes.currently_loaded() if swap.running else []
        loaded_part = f"  │  loaded: [cyan]{', '.join(loaded)}[/cyan]" if loaded else ""

        return (
            f"[bold]◆ inferhost[/bold]  "
            f"│ {swap_dot} :{s.swap_port}  "
            f"{gw_dot} :{s.gateway_port}{gw_suffix}  "
            f"│ {n_models} model{'s' if n_models != 1 else ''}  "
            f"│ {ctx_part} slots={s.parallel_slots} ngl={s.gpu_layers} fa={s.flash_attention}"
            f"{loaded_part}"
        )

    @staticmethod
    def _vram_bar(used_gib: float, total_gib: float, width: int = 10) -> str:
        if total_gib <= 0:
            return "─" * width
        frac = max(0.0, min(1.0, used_gib / total_gib))
        filled = int(round(frac * width))
        if frac < 0.80:
            color = "green"
        elif frac < 0.95:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]" + "█" * filled + "[/]" + "░" * (width - filled)

    def _gpu_text(self) -> str:
        gpus = processes.query_gpus()
        if not gpus:
            return ""
        parts: list[str] = []
        for g in gpus:
            used = g.mem_used_mib / 1024
            total = g.mem_total_mib / 1024
            bar = self._vram_bar(used, total)
            parts.append(
                f"[bold]GPU{g.index}[/bold] {bar} {used:.1f}/{total:.1f} GiB · util {g.util_pct}%"
            )
        return "  │  ".join(parts)

    def _refresh_bars(self) -> None:
        try:
            self.query_one("#status-bar", Static).update(self._status_text())
            self.query_one("#gpu-bar", Static).update(self._gpu_text())
        except Exception:  # noqa: BLE001
            pass

    # ---- data refresh ----

    def refresh_models(self) -> None:
        reg = registry.load()
        list_view = self.query_one("#model-list", ListView)
        list_view.clear()
        for m in reg.models:
            list_view.append(ListItem(Label(f"{m.name}  ({m.quant or '?'})"), name=m.name))
        try:
            self.query_one("#sidebar-label", Label).update(f"Models ({len(reg.models)})")
        except Exception:  # noqa: BLE001
            pass
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
        # Resolve per-model overrides against the global Settings so the panel
        # shows what llama-server will actually be invoked with — including
        # which values are inherited vs. set explicitly per-model.
        eff_reasoning = m.reasoning if m.reasoning else s.reasoning
        eff_budget = m.reasoning_budget if m.reasoning_budget != -2 else s.reasoning_budget
        inh_r = "" if m.reasoning else "  [grey50](global)[/grey50]"
        inh_b = "" if m.reasoning_budget != -2 else "  [grey50](global)[/grey50]"
        kv = f"K={m.cache_type_k or 'f16'} V={m.cache_type_v or 'f16'}"
        details.update(
            f"[bold]{m.name}[/bold]\n"
            f"repo:     {m.repo_id}\n"
            f"file:     {m.filename}\n"
            f"quant:    {m.quant or '?'}    size: {m.size_gib} GiB\n"
            f"ctx:      {m.ctx}    kv: {kv}\n"
            f"reasoning:{eff_reasoning}{inh_r}    budget: {eff_budget}{inh_b}\n"
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
            self._refresh_bars()

    @on(ListView.Selected, "#model-list")
    def _on_select(self, ev: ListView.Selected) -> None:
        item = ev.item
        if item is not None:
            self.selected_name = item.name
            self._refresh_details()
            self._refresh_bars()

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

    @on(Button.Pressed, ".action-btn")
    def _on_action_button(self, ev: Button.Pressed) -> None:
        if ev.button.id is None:
            return
        # Stop the event so it doesn't bubble back up into other Button.Pressed
        # handlers (e.g. modal save/cancel buttons that share the dashboard).
        ev.stop()
        for _row, btn_id, _label, action in self._BUTTONS:
            if btn_id != ev.button.id:
                continue
            # Call action_<name> directly — self.run_action() is async in
            # Textual, so calling it from a sync handler just builds an
            # un-awaited coroutine and silently does nothing.
            screen_method = getattr(self, f"action_{action}", None)
            if screen_method is not None:
                screen_method()
            elif action == "quit":
                self.app.exit()
            return
