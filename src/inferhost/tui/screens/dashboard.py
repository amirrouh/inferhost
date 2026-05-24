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

from inferhost.core import configs, processes, registry, vram
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
        # Lowercase p toggles pin/unpin (common action — easy reach).
        # Capital P kept as a backwards-compatible alias.
        ("p", "toggle_pin", "Pin"),
        ("P", "toggle_pin", ""),
        # l / Enter: load (or unload, if already loaded) the SELECTED model.
        # This is distinct from `s` which starts the llama-swap daemon — it
        # acts on whichever model is highlighted in the sidebar.
        ("l", "toggle_load", "Load"),
        ("enter", "toggle_load", ""),
        ("s", "start_swap", "Daemon"),
        ("x", "stop_swap", "Stop daemon"),
        ("r", "restart_swap", "Restart daemon"),
        ("g", "toggle_gateway", "Gateway"),
        # Settings moved off `p` so the lowercase letter could host pin/unpin.
        # Use the named key "comma" — Textual parses bare "," as a key-list
        # separator and crashes on launch with InvalidBinding.
        ("comma", "open_settings", "Settings"),
        ("R", "refresh", "Refresh"),
    ]

    # Two-row docked action bar. Row 1 = per-model actions (act on the
    # highlighted sidebar item). Row 2 = daemon-level actions (control the
    # whole stack). Keep the rows aligned with that mental model so users
    # don't confuse "start model" with "start daemon".
    _BUTTONS: tuple[tuple[int, str, str, str], ...] = (
        # (row, btn_id, label, action)
        (1, "btn-add",     "a Add",       "add_model"),
        (1, "btn-rename",  "n Rename",    "rename_model"),
        (1, "btn-config",  "c Configure", "configure_model"),
        (1, "btn-pin",     "p Pin",       "toggle_pin"),
        (1, "btn-load",    "l Load",      "toggle_load"),
        (1, "btn-del",     "d Delete",    "remove_model"),
        (2, "btn-start",   "s Daemon",    "start_swap"),
        (2, "btn-stop",    "x Stop",      "stop_swap"),
        (2, "btn-restart", "r Restart",   "restart_swap"),
        (2, "btn-gw",      "g Gateway",   "toggle_gateway"),
        (2, "btn-prefs",   ", Settings",  "open_settings"),
        (2, "btn-refresh", "R Refresh",   "refresh"),
        (2, "btn-quit",    "q Quit",      "quit"),
    )

    selected_name: reactive[str | None] = reactive(None)

    def watch_selected_name(self, _value: str | None) -> None:
        self._refresh_pin_button()
        self._refresh_load_button()

    def _refresh_pin_button(self) -> None:
        try:
            reg = registry.load()
            m = reg.get(self.selected_name) if self.selected_name else None
            label = "p Unpin" if (m is not None and m.pin) else "p Pin"
            from textual.css.query import NoMatches
            try:
                btn = self.query_one("#btn-pin", Button)
                btn.label = label
                # `width: auto` doesn't always re-expand when the label gets
                # longer ("p Pin" → "p Unpin"), so the new text gets clipped
                # to just "p". Force a layout-recomputing refresh.
                btn.refresh(layout=True)
            except NoMatches:
                pass
        except Exception:  # noqa: BLE001
            pass

    def _refresh_load_button(self) -> None:
        """Toggle the Load/Unload button text based on the selected model's state."""
        try:
            state = self._model_states.get(self.selected_name or "")
            label = "l Unload" if state == "ready" else "l Load"
            from textual.css.query import NoMatches
            try:
                btn = self.query_one("#btn-load", Button)
                btn.label = label
                btn.refresh(layout=True)
            except NoMatches:
                pass
        except Exception:  # noqa: BLE001
            pass

    def compose(self) -> ComposeResult:
        # Three top rows wrapped in ONE docked Vertical. Docking each Static
        # independently to `top` doesn't stack them — they all collapse onto
        # y=0 and the last one painted (the status bar) covers the others,
        # hiding the VRAM bar entirely. A single docked container with
        # vertical layout stacks its children naturally.
        with Vertical(id="header"):
            yield Static("", id="gpu-bar")
            yield Static("", id="gpu-warning")
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
        # Cached snapshots refreshed by the background tick worker. Reading
        # these on the event loop is cheap; the actual blocking I/O (nvidia-smi,
        # the llama-swap HTTP poll, log file reads) happens off-thread so the UI
        # never freezes — see _tick / _collect.
        self._gpus: list[processes.GpuStat] = []
        self._loaded_models: list[str] = []
        # llama-swap state per model: 'ready' | 'starting' | 'stopping' | None
        # (absent from the map = not loaded at all).
        self._model_states: dict[str, str] = {}
        self._swap_running: bool = False
        self._log_offset: int = 0
        self._tick_in_flight: bool = False

        # First-paint VRAM/state: do ONE synchronous read so the dashboard
        # doesn't look dead for the 1-4 seconds before the background tick
        # worker fires. This is fine: it runs once at startup, not every tick.
        # The blocking-IO concern (which is what 0.5.1 fixed) is only about
        # *repeated* polls on the event loop, not a one-shot init.
        try:
            self._gpus = processes.query_gpus()
            self._swap_running = processes.swap_status().running
            self._model_states = processes.model_states() if self._swap_running else {}
            self._loaded_models = list(self._model_states.keys())
        except Exception:  # noqa: BLE001
            pass

        self.refresh_models()
        self._refresh_pin_button()
        self._refresh_bars()  # paint VRAM bar before first frame, not after first tick
        # Initial log fill (synchronous one-shot — fast, uses seek-based tail).
        self._initial_log_fill()
        # Subsequent refreshes happen off-thread to keep the UI responsive
        # under GPU load (2 s cadence). Faster than this is wasteful; slower
        # than this and the VRAM bar feels stale during a model load.
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
        # Read from the cached snapshot — querying llama-swap here would block
        # the UI thread on a synchronous HTTP call.
        loaded = self._loaded_models
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
        # Use the cached snapshot updated by the background _collect worker —
        # calling nvidia-smi here would block the event loop (50 ms idle, up to
        # 1-2 s while the GPU is busy with inference).
        gpus = self._gpus
        if not gpus:
            # Placeholder so the bar isn't blank during the first paint while
            # the worker collects. Visible as "GPU: …" instead of an empty
            # 1-line gap that makes the TUI look broken.
            return "[dim]GPU: detecting…[/dim]"
        parts: list[str] = []
        for g in gpus:
            used = g.mem_used_mib / 1024
            total = g.mem_total_mib / 1024
            pct = (used / total * 100) if total > 0 else 0.0
            bar = self._vram_bar(used, total, width=12)
            parts.append(
                f"[bold]GPU{g.index}[/bold] {bar} "
                f"[bold]{pct:.0f}%[/bold] {used:.1f}/{total:.1f}G · util {g.util_pct}%"
            )
        return "  │  ".join(parts)

    def _warning_text(self) -> str:
        """The pinned-overflow warning, rendered on its own line if present."""
        warn = self._pinned_overflow_warning()
        if warn:
            return f"[bold white on red]{warn}[/bold white on red]"
        return ""

    def _pinned_overflow_warning(self) -> str:
        """Return a short warning if pinned WEIGHTS alone exceed primary VRAM.

        Uses weights-only (size_gib * 1.05) as the threshold — KV cache size
        depends on attention layout (GQA reduces it 4-8x on modern models) and
        a conservative KV estimate would false-alarm. Weights are a firm floor:
        if pinned weights > GPU, the models *physically* cannot coexist no
        matter how aggressively the KV cache is compressed.

        This is the case that fooled me on gpu-3090: two huge models marked
        pinned, both green in the registry, weight-total 25.2 GiB > 24 GiB GPU,
        and llama-server kept OOMing on load.
        """
        if not self._gpus:
            return ""
        try:
            reg = registry.load()
        except Exception:  # noqa: BLE001
            return ""
        pinned_weights = sum(m.size_gib * 1.05 for m in reg.models if m.pin)
        if pinned_weights <= 0:
            return ""
        total_gib = self._gpus[0].mem_total_mib / 1024
        if pinned_weights <= total_gib:
            return ""
        return (
            f"⚠ PINNED OVERFLOW: weights alone ~{pinned_weights:.1f} GiB > "
            f"{total_gib:.1f} GiB GPU — unpin one to let models load"
        )

    def _pinned_loaded_text(self) -> str:
        try:
            reg = registry.load()
            pinned_est = vram.pinned_vram_estimate(reg)
            # Cached snapshot — see _gpu_text comment.
            loaded_names = self._loaded_models
            n_loaded = len(loaded_names)
            loaded_est = sum(
                vram.estimate_model_vram_gib(m)
                for m in reg.models
                if m.name in loaded_names
            )
            return (
                f"Pinned (est): {pinned_est:.1f} GiB · "
                f"Loaded: {n_loaded} model{'s' if n_loaded != 1 else ''} "
                f"({loaded_est:.1f} GiB)"
            )
        except Exception:  # noqa: BLE001
            return ""

    def _refresh_bars(self) -> None:
        try:
            self.query_one("#status-bar", Static).update(self._status_text())
            self.query_one("#gpu-bar", Static).update(self._gpu_text())
            # Show/hide the warning row via the `display` attribute so the
            # row collapses to 0 height when there's nothing to warn about.
            warning_widget = self.query_one("#gpu-warning", Static)
            warning = self._warning_text()
            warning_widget.update(warning)
            warning_widget.display = bool(warning)
        except Exception:  # noqa: BLE001
            pass

    # ---- data refresh ----

    def _model_row(self, m) -> str:
        """Sidebar label for one model — colored filled dot + ★ if pinned.

        Always a *filled* circle (●) so the symbol stays the same and only the
        color changes — easy to scan in low-contrast / SSH terminals where an
        empty ○ vs filled ● can look identical.

          [green]●     ready (loaded, serving)
          [yellow]●    starting / stopping (transient)
          [red]●       offline (not loaded)
        """
        state = self._model_states.get(m.name)
        if state == "ready":
            dot = "[bold green]●[/bold green]"
        elif state in ("starting", "stopping"):
            dot = "[bold yellow]●[/bold yellow]"
        else:
            dot = "[bold red]●[/bold red]"
        star = "[yellow]★[/yellow]" if m.pin else " "
        return f"{dot} {star} {m.name}  ({m.quant or '?'})"

    def _refresh_model_dots(self) -> None:
        """Update only the dot/star prefix on each existing sidebar row."""
        try:
            list_view = self.query_one("#model-list", ListView)
        except Exception:  # noqa: BLE001
            return
        reg = registry.load()
        by_name = {m.name: m for m in reg.models}
        for item in list_view.query(ListItem):
            m = by_name.get(item.name or "")
            if m is None:
                continue
            label = item.query(Label).first()
            if label is not None:
                label.update(self._model_row(m))

    def refresh_models(self) -> None:
        reg = registry.load()
        list_view = self.query_one("#model-list", ListView)
        list_view.clear()
        for m in reg.models:
            list_view.append(
                ListItem(Label(self._model_row(m)), name=m.name)
            )
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
        sset = settings()
        kv = f"KV: K={getattr(sset, 'kv_quant_k', 'q8_0')} V={getattr(sset, 'kv_quant_v', 'turbo3')} (asymmetric)"
        pin_part = "[yellow]★ pinned[/yellow] (co-resident)" if m.pin else "swap on demand"
        details.update(
            f"[bold]{m.name}[/bold]\n"
            f"repo:     {m.repo_id}\n"
            f"file:     {m.filename}\n"
            f"quant:    {m.quant or '?'}    size: {m.size_gib} GiB\n"
            f"ctx:      {m.ctx}    kv: {kv}\n"
            f"reasoning:{eff_reasoning}{inh_r}    budget: {eff_budget}{inh_b}\n"
            f"loading:  {pin_part}\n"
            f"backend:  port {m.port}  ->  swap http://localhost:{s.swap_port}/v1\n"
            f"path:     {m.local_path}"
        )
        # NOTE: the log widget is owned by the tick worker (initial fill in
        # on_mount, incremental appends in _apply_tick). Don't clear+rewrite it
        # on selection change — the swap log is the same regardless of which
        # model is highlighted, and re-rendering 200 lines on every arrow-key
        # navigation was a major source of UI lag.
        _ = log_widget  # silence unused-local lint; kept for the query_one above

    def _tick(self) -> None:
        # Cheap on the event loop: just dispatch the worker. Skip if a previous
        # tick is still running (e.g. nvidia-smi is hung) so they don't pile up.
        if self._tick_in_flight:
            return
        self._tick_in_flight = True
        self.run_worker(self._collect, thread=True, exclusive=False)

    def _collect(self) -> None:
        """Off-thread: gather every blocking I/O the dashboard needs.

        nvidia-smi can take 1-2 s under GPU load; the llama-swap /running HTTP
        poll can also block for hundreds of ms. Running both here means the UI
        thread stays responsive even when the GPU is at 100 %.
        """
        try:
            gpus = processes.query_gpus()
            swap_running = processes.swap_status().running
            states = processes.model_states() if swap_running else {}
            loaded = list(states.keys())
            new_lines = self._read_new_log_lines()
        except Exception:  # noqa: BLE001
            # Failures here just mean a stale snapshot — never propagate to UI.
            self.app.call_from_thread(self._tick_done)
            return
        self.app.call_from_thread(
            self._apply_tick, gpus, loaded, states, swap_running, new_lines,
        )

    def _apply_tick(
        self,
        gpus: list[processes.GpuStat],
        loaded: list[str],
        states: dict[str, str],
        swap_running: bool,
        new_lines: list[str],
    ) -> None:
        self._gpus = gpus
        self._loaded_models = loaded
        self._model_states = states
        self._swap_running = swap_running
        self._refresh_bars()
        # Also re-render the sidebar so dots update without waiting for the
        # next add/remove. Cheap — just iterates the registry.
        try:
            self._refresh_model_dots()
            self._refresh_load_button()
        except Exception:  # noqa: BLE001
            pass
        try:
            log_widget = self.query_one("#logs", Log)
            for line in new_lines:
                log_widget.write_line(line)
        except Exception:  # noqa: BLE001
            pass
        self._tick_done()

    def _tick_done(self) -> None:
        self._tick_in_flight = False

    def _initial_log_fill(self) -> None:
        path = log_path("swap")
        if not path.exists():
            self._log_offset = 0
            return
        try:
            for line in tail(path, 200):
                self.query_one("#logs", Log).write_line(line)
            self._log_offset = path.stat().st_size
        except Exception:  # noqa: BLE001
            self._log_offset = 0

    def _read_new_log_lines(self) -> list[str]:
        """Incremental tail: seek to last-known offset, read only what's new.

        Re-reading the entire swap log every tick was the third blocking source
        in the old _tick — it grew unboundedly across a session.
        """
        path = log_path("swap")
        if not path.exists():
            self._log_offset = 0
            return []
        try:
            size = path.stat().st_size
            if size < self._log_offset:
                # Log rotated or truncated since last read — start over.
                self._log_offset = 0
            if size == self._log_offset:
                return []
            with path.open("rb") as f:
                f.seek(self._log_offset)
                chunk = f.read()
                self._log_offset = f.tell()
            return chunk.decode(errors="replace").splitlines()
        except Exception:  # noqa: BLE001
            return []

    # ---- list handlers ----

    @on(ListView.Highlighted, "#model-list")
    def _on_highlight(self, ev: ListView.Highlighted) -> None:
        item = ev.item
        if item is not None:
            self.selected_name = item.name
            self._refresh_details()
            # _refresh_bars uses the cached snapshot, so it's cheap, but the
            # status bar's ctx readout depends on selected_name — refresh it.
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

    def action_toggle_load(self) -> None:
        """Load the highlighted model into VRAM (or unload it if already there).

        Separate from action_toggle_pin: this is "serve this model RIGHT NOW",
        not "keep it pinned across restarts". Useful for ad-hoc inference.
        """
        if self.selected_name is None:
            self.notify("Select a model first.", severity="warning")
            return
        name = self.selected_name
        if not processes.swap_status().running:
            self.notify("llama-swap isn't running — press 's' to start it first.",
                        severity="warning")
            return

        state = self._model_states.get(name)
        if state == "ready":
            # Unload — quick, returns once llama-swap evicts.
            self.notify(f"Unloading {name}…")
            self.run_worker(
                lambda n=name: self._do_unload_and_refresh(n),
                thread=True, exclusive=False,
            )
        else:
            self.notify(f"Loading {name}… (first request can take 15-30 s on big models)")
            self.run_worker(
                lambda n=name: self._do_load_and_refresh(n),
                thread=True, exclusive=False,
            )

    def _do_load_and_refresh(self, name: str) -> None:
        ok = processes.force_load_model(name, timeout=120.0)
        # Post a success/failure toast and force an immediate state refresh
        # so the dot flips to green (or stays red) without waiting for tick.
        if ok:
            self.app.call_from_thread(self.notify, f"Loaded {name}.")
        else:
            self.app.call_from_thread(
                self.notify,
                f"Load failed for {name} — check the log panel. "
                "Common cause: pinned weights exceed GPU VRAM (see warning row).",
                severity="error",
            )
        self.app.call_from_thread(self.run_worker, self._collect, thread=True)

    def _do_unload_and_refresh(self, name: str) -> None:
        ok = processes.force_unload_model(name, timeout=10.0)
        if ok:
            self.app.call_from_thread(self.notify, f"Unloaded {name}.")
        else:
            self.app.call_from_thread(
                self.notify, f"Unload failed for {name}.", severity="error",
            )
        self.app.call_from_thread(self.run_worker, self._collect, thread=True)

    def action_toggle_pin(self) -> None:
        if self.selected_name is None:
            self.notify("Select a model first.", severity="warning")
            return
        reg = registry.load()
        m = reg.get(self.selected_name)
        if m is None:
            return

        if m.pin:
            # --- unpinning ---
            m.pin = False
            registry.save(reg)
            try:
                configs.write_all(reg)
                processes.reload_if_running()
            except Exception as e:  # noqa: BLE001
                self.notify(f"Unpinned, but reload failed: {e}", severity="error")
            else:
                self.notify(f"'{m.name}' unpinned (swap on demand).")
            ok = processes.force_unload_model(self.selected_name)
            if not ok:
                log = self.query_one("#logs", Log)
                log.write_line(f"[warn] force_unload_model('{m.name}') returned False")
            self.refresh_models()
            self._refresh_pin_button()
        else:
            # --- pinning ---
            # VRAM check is informational only: the pinned-overflow row and
            # llama-server's own OOM are the real signals. Don't block the user.
            ok, needed, free = vram.can_pin(reg, m)
            if not ok:
                self.notify(
                    f"VRAM tight: '{m.name}' needs ~{needed:.1f} GiB, "
                    f"only {free:.1f} GiB free. Pinning anyway — load may OOM.",
                    severity="warning",
                )
            m.pin = True
            registry.save(reg)
            try:
                configs.write_all(reg)
                processes.reload_if_running()
            except Exception as e:  # noqa: BLE001
                self.notify(f"Pinned, but reload failed: {e}", severity="error")
            else:
                self.notify(f"[yellow]★[/yellow] '{m.name}' pinned (co-resident).")
            model_name = m.name
            self.run_worker(
                lambda: processes.force_load_model(model_name), thread=True
            )
            self.refresh_models()
            self._refresh_pin_button()

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
        swap_was_running = processes.swap_status().running
        gw_was_running = processes.gateway_status().running
        try:
            reg = registry.load()
            configs.write_all(reg)
            swap_st = processes.start_swap()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Start failed: {e}", severity="error")
            return
        # Bring the gateway up alongside swap so :9001 honors the README's
        # "single endpoint, always on" promise. Gateway failures are
        # non-fatal — swap is the inference path; gateway is the front door.
        gw_st = None
        if processes.gateway_available() and not gw_was_running:
            try:
                gw_st = processes.start_gateway()
            except Exception as e:  # noqa: BLE001
                self.notify(f"Gateway start failed: {e}", severity="warning")
        # Always give user-visible feedback — silent no-op when already running
        # was making the TUI feel dead.
        if swap_was_running:
            self.notify(f"llama-swap already running (pid={swap_st.pid}, port={swap_st.port})")
        else:
            self.notify(f"llama-swap started (pid={swap_st.pid}, port={swap_st.port})", severity="information")
        if gw_st is not None and gw_st.running:
            self.notify(f"gateway started (pid={gw_st.pid}, port={gw_st.port})", severity="information")
        self.run_worker(self._collect, thread=True, exclusive=False)

    def action_stop_swap(self) -> None:
        # Stop gateway first so it doesn't briefly serve requests against a
        # vanished swap backend.
        if processes.gateway_status().running:
            processes.stop_gateway()
        processes.stop_swap()
        self.notify("stack stopped (llama-swap + gateway)")
        self.run_worker(self._collect, thread=True, exclusive=False)

    def action_restart_swap(self) -> None:
        gw_was_running = processes.gateway_status().running
        if gw_was_running:
            processes.stop_gateway()
        processes.stop_swap()
        try:
            reg = registry.load()
            configs.write_all(reg)
            swap_st = processes.start_swap()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Restart failed: {e}", severity="error")
            return
        gw_st = None
        if processes.gateway_available() and gw_was_running:
            try:
                gw_st = processes.start_gateway()
            except Exception as e:  # noqa: BLE001
                self.notify(f"Gateway restart failed: {e}", severity="warning")
        msg = f"llama-swap restarted (pid={swap_st.pid})"
        if gw_st is not None and gw_st.running:
            msg += f"; gateway pid={gw_st.pid}"
        self.notify(msg)
        self.run_worker(self._collect, thread=True, exclusive=False)

    # ---- actions: gateway ----

    def action_toggle_gateway(self) -> None:
        if not processes.gateway_available():
            self.notify(
                "litellm not found in this environment. "
                "Reinstall: uv tool install --reinstall inferhost",
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
