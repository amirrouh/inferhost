"""Add-model modal screen."""
from __future__ import annotations

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
    RadioButton,
    RadioSet,
    Static,
)

from inferhost.core import (
    binaries,
    configs,
    gguf,
    hf,
    image_recipes,
    paths,
    probe,
    processes,
    quant,
    registry,
)
from inferhost.settings import settings


class AddModelScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.files: list[hf.GgufFile] = []
        self.selected_idx: int | None = None
        self.downloading: bool = False
        self.kind: str = "chat"

    def compose(self) -> ComposeResult:
        with Vertical(id="add-dialog"):
            yield Label("[bold]Add Hugging Face model[/bold]")
            with RadioSet(id="kind-set"):
                yield RadioButton("Chat / LLM", value=True, id="kind-chat")
                yield RadioButton("Image generation", id="kind-image")
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

    @on(RadioSet.Changed, "#kind-set")
    def _on_kind(self, ev: RadioSet.Changed) -> None:
        self.kind = "image" if ev.pressed.id == "kind-image" else "chat"
        placeholder = (
            "e.g. city96/FLUX.1-dev-gguf  or  stabilityai/sdxl-turbo"
            if self.kind == "image"
            else "e.g. Qwen/Qwen2.5-7B-Instruct-GGUF"
        )
        self.query_one("#repo-input", Input).placeholder = placeholder
        files_word = ".gguf / .safetensors" if self.kind == "image" else ".gguf"
        self._set_hint(f"Press Enter to list available {files_word} files.")

    @on(Input.Submitted, "#repo-input")
    def _on_submit(self, ev: Input.Submitted) -> None:
        self._fetch(ev.value.strip())

    @work(exclusive=True, thread=True)
    def _fetch(self, repo_id: str) -> None:
        if not repo_id:
            return
        self.app.call_from_thread(self._set_hint, "Fetching file list ...")
        try:
            files = hf.list_image_files(repo_id) if self.kind == "image" else hf.list_ggufs(repo_id)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._set_hint, f"[red]Error: {e}[/red]")
            return
        if not files:
            self.app.call_from_thread(self._set_hint, "[yellow]No matching model files found.[/yellow]")
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
        if self.kind == "image":
            self._register_image(pick)
            return
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
            # If the repo ships an mmproj-*.gguf, grab it too so vision works.
            mmproj_local = ""
            mmproj_name = hf.find_mmproj(pick.repo_id)
            if mmproj_name:
                self.app.call_from_thread(
                    self._set_hint, f"Downloading vision projector {mmproj_name} ..."
                )
                mmproj_local = str(hf.download_gguf(pick.repo_id, mmproj_name))
            # If the repo ships a WavTokenizer/vocoder GGUF, grab it too — its
            # presence reclassifies this model as text-to-speech (served by the
            # inferhost-tts daemon, not llama-swap).
            vocoder_local = ""
            vocoder_name = hf.find_vocoder(pick.repo_id)
            if vocoder_name:
                self.app.call_from_thread(
                    self._set_hint, f"Downloading TTS vocoder {vocoder_name} ..."
                )
                vocoder_local = str(hf.download_gguf(pick.repo_id, vocoder_name))
            reg = registry.load()
            name = hf.normalize_name(pick.repo_id)
            if pick.quant:
                name = f"{name}-{pick.quant.lower().replace('_', '-')}"
            s = settings()
            paths.ensure_dirs()
            # Never register a window the file can't actually serve: if the
            # GGUF's native trained context is below the global default, store
            # that instead so the advertised/served window matches the file.
            native = gguf.native_context_cached(str(local))
            ctx = min(s.default_ctx, native) if native else s.default_ctx
            model = registry.Model(
                name=name,
                repo_id=pick.repo_id,
                filename=pick.filename,
                quant=pick.quant,
                ctx=ctx,
                port=reg.next_port(s.swap_port),
                size_gib=pick.size_gib,
                local_path=str(local),
                mmproj_path=mmproj_local,
                vocoder_path=vocoder_local,
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

    def _register_image(self, pick: hf.GgufFile) -> None:
        """Download + register an image model (stable-diffusion.cpp / sd-server).

        If the model matches a known family (Flux.1/.2, Z-Image, Qwen-Image) it
        auto-downloads the correct companion VAE/encoders from a built-in recipe
        and sets sane sampling defaults — so the user doesn't need to know which
        files to fetch. Otherwise it falls back to same-repo auto-detect. The
        sd-server binary is fetched on first use. Registered with kind='image';
        unmatched/cross-repo companions are completable via the Configure picker.
        """
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
            aux_paths: dict[str, str] = {}
            default_args = ""
            # 1. Known-family recipe: fetch the exact companions + defaults.
            recipe = image_recipes.match_recipe(pick.repo_id, hf.repo_tags(pick.repo_id))
            if recipe is not None:
                self.app.call_from_thread(
                    self._set_hint, f"Detected {recipe.label} — fetching companion files ..."
                )
                default_args = recipe.default_args
                for fld, (repo, fname) in recipe.companions.items():
                    self.app.call_from_thread(
                        self._set_hint, f"Downloading {recipe.label} {fld.replace('_path','')}: {fname} ..."
                    )
                    aux_paths[fld] = str(hf.download_gguf(repo, fname))
            # 2. Fill any slot the recipe didn't cover from same-repo companions.
            for fld, fname in hf.find_sd_aux(pick.repo_id).items():
                if fld in aux_paths:
                    continue
                self.app.call_from_thread(self._set_hint, f"Downloading {fld} {fname} ...")
                aux_paths[fld] = str(hf.download_gguf(pick.repo_id, fname))
            # Ensure the sd-server binary is present (first image model on this box).
            if binaries.needs_sdcpp_refresh():
                self.app.call_from_thread(self._set_hint, "Fetching sd-server (image engine) ...")
                self.app.call_from_thread(self._update_progress, 0, 1)
                binaries.install_stable_diffusion(
                    progress_cb=lambda done, total: self.app.call_from_thread(
                        self._update_progress, done, total or 1
                    )
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
                port=reg.next_port(s.swap_port),
                size_gib=pick.size_gib,
                local_path=str(local),
                kind="image",
                vae_path=aux_paths.get("vae_path", ""),
                clip_l_path=aux_paths.get("clip_l_path", ""),
                clip_g_path=aux_paths.get("clip_g_path", ""),
                t5xxl_path=aux_paths.get("t5xxl_path", ""),
                text_encoder_path=aux_paths.get("text_encoder_path", ""),
                vision_encoder_path=aux_paths.get("vision_encoder_path", ""),
                extra_args=default_args,
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
