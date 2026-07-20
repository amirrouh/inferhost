"""Add-model modal screen."""
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
                yield RadioButton("Text-to-speech", id="kind-tts")
            yield Input(
                placeholder="Paste a Hugging Face link or owner/repo, e.g. Qwen/Qwen2.5-7B-Instruct-GGUF",
                id="repo-input",
            )
            yield Static("Paste the model's Hugging Face URL (or owner/repo) and press Enter to list files.", id="hint")
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
        if ev.pressed.id == "kind-image":
            self.kind = "image"
        elif ev.pressed.id == "kind-tts":
            self.kind = "tts"
        else:
            self.kind = "chat"
        placeholder = {
            "image": "Paste a link or owner/repo, e.g. city96/FLUX.1-dev-gguf  or  stabilityai/sdxl-turbo",
            "tts": "Paste a link or owner/repo, e.g. OuteAI/OuteTTS-0.2-500M-GGUF",
        }.get(
            self.kind,
            "Paste a link or owner/repo, e.g. Qwen/Qwen2.5-7B-Instruct-GGUF",
        )
        self.query_one("#repo-input", Input).placeholder = placeholder
        files_word = ".gguf / .safetensors" if self.kind == "image" else ".gguf"
        self._set_hint(f"Press Enter to list available {files_word} files.")

    @on(Input.Submitted, "#repo-input")
    def _on_submit(self, ev: Input.Submitted) -> None:
        self._fetch(hf.parse_repo_id(ev.value))

    @work(exclusive=True, thread=True)
    def _fetch(self, repo_id: str) -> None:
        if not repo_id:
            return
        self.app.call_from_thread(self._set_hint, "Fetching file list ...")
        try:
            if self.kind == "image":
                files = hf.list_image_files(repo_id)
            elif self.kind == "tts":
                files = hf.list_tts_files(repo_id)
            else:
                files = hf.list_ggufs(repo_id)
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
            parts_tag = f"  [{len(f.parts)} parts]" if f.parts else ""
            label = f"{marker} {fits} {f.quant or '?':<8}  {f.size_gib:>5} GiB  {f.filename}{parts_tag}"
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
        if self.kind == "tts":
            self._register_tts(pick)
            return
        self._register_chat(pick)

    # ---- shared download helpers ----

    def _download_main_or_parts(self, pick: hf.GgufFile) -> Path:
        """Download the main model file — or every shard, for a multi-part GGUF.

        Shared by the chat / image / TTS registration paths so multi-part
        support doesn't need separate wiring per kind. Reports progress the
        same way either way, via ``_update_progress``.
        """
        self.app.call_from_thread(self._set_hint, f"Downloading {pick.filename} ...")
        self.app.call_from_thread(self._update_progress, 0, max(pick.size_bytes, 1))
        if pick.parts:
            return hf.download_gguf_parts_with_progress(
                repo_id=pick.repo_id,
                parts=pick.parts,
                progress_cb=lambda done, total: self.app.call_from_thread(
                    self._update_progress, done, total or max(pick.size_bytes, 1)
                ),
            )
        return hf.download_gguf_with_progress(
            repo_id=pick.repo_id,
            filename=pick.filename,
            expected_bytes=max(pick.size_bytes, 1),
            progress_cb=lambda done, total: self.app.call_from_thread(
                self._update_progress, done, total or max(pick.size_bytes, 1)
            ),
        )

    def _download_companion(self, repo_id: str, filename: str, label: str) -> str:
        """Download one companion file (mmproj / vocoder / aux) with progress.

        On failure, wraps the error as ``RuntimeError(f"{label} ({filename})
        download failed: ...")`` so the toast names the actual component that
        broke instead of a generic "Failed: ...".
        """
        size = hf.repo_file_size(repo_id, filename)
        self.app.call_from_thread(self._set_hint, f"Downloading {label}: {filename} ...")
        self.app.call_from_thread(self._update_progress, 0, max(size, 1))
        try:
            local = hf.download_gguf_with_progress(
                repo_id=repo_id,
                filename=filename,
                expected_bytes=max(size, 1),
                progress_cb=lambda done, total: self.app.call_from_thread(
                    self._update_progress, done, total or max(size, 1)
                ),
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{label} ({filename}) download failed: {e}") from e
        return str(local)

    # ---- per-kind registration ----

    def _register_chat(self, pick: hf.GgufFile) -> None:
        try:
            local = self._download_main_or_parts(pick)
            # If the repo ships an mmproj-*.gguf, grab it too so vision works.
            mmproj_local = ""
            mmproj_name = hf.find_mmproj(pick.repo_id)
            if mmproj_name:
                mmproj_local = self._download_companion(pick.repo_id, mmproj_name, "vision projector")
            # If the repo ships a WavTokenizer/vocoder GGUF, grab it too — its
            # presence reclassifies this model as text-to-speech (served by the
            # inferhost-tts daemon, not llama-swap).
            vocoder_local = ""
            vocoder_name = hf.find_vocoder(pick.repo_id)
            if vocoder_name:
                vocoder_local = self._download_companion(pick.repo_id, vocoder_name, "TTS vocoder")
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
            # NOTE: no daemon reload here — this modal's job ends at
            # file-on-disk + registry saved. The dashboard's _after_add
            # callback runs the (tens-of-seconds) daemon reload off-thread
            # AFTER this modal dismisses, so the modal never sits frozen.
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
        try:
            local = self._download_main_or_parts(pick)
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
                    aux_paths[fld] = self._download_companion(
                        repo, fname, f"{recipe.label} {fld.replace('_path', '')}"
                    )
            # 2. Fill any slot the recipe didn't cover from same-repo companions.
            for fld, fname in hf.find_sd_aux(pick.repo_id).items():
                if fld in aux_paths:
                    continue
                aux_paths[fld] = self._download_companion(pick.repo_id, fname, fld)
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
            # NOTE: no daemon reload here — see _register_chat's comment.
        except Exception as e:  # noqa: BLE001
            self.downloading = False
            self.app.call_from_thread(self._set_hint, f"[red]Failed: {e}[/red]")
            return
        self.downloading = False
        self.app.call_from_thread(self.dismiss, True)

    def _register_tts(self, pick: hf.GgufFile) -> None:
        """Download + register a text-to-speech model.

        Every TTS repo needs a vocoder companion — WavTokenizer for
        OuteTTS-style models, or the qwen3-tts-tokenizer GGUF for Qwen3-TTS —
        which is what marks the registered model as TTS
        (``registry.Model.vocoder_path != ""``); it's required here, not
        optional, since a TTS pick with no vocoder can't actually be served.
        Qwen3-TTS models additionally need the qwen3-tts.cpp engine, which
        ships no prebuilt release: it's built from source on demand, the first
        time such a model is added (mirrors how install_stable_diffusion is
        gated in _register_image, not in InstallScreen).
        """
        try:
            local = self._download_main_or_parts(pick)
            vocoder_name = hf.find_vocoder(pick.repo_id)
            if not vocoder_name:
                raise RuntimeError(
                    f"No vocoder/tokenizer GGUF found in {pick.repo_id} — a "
                    "text-to-speech model needs one in the same repo "
                    "(WavTokenizer for OuteTTS-style models, or the "
                    "qwen3-tts-tokenizer GGUF for Qwen3-TTS)."
                )
            vocoder_local = self._download_companion(pick.repo_id, vocoder_name, "TTS vocoder")

            # Qwen3-TTS models are routed to the qwen3-tts.cpp engine by
            # filename convention (mirrors tts_serve.py's _is_qwen3_tts).
            if pick.filename.lower().startswith("qwen3-tts") and binaries.needs_qwen3_tts_refresh():
                self.app.call_from_thread(
                    self._set_hint,
                    "Building the Qwen3-TTS engine (qwen3-tts.cpp) — compiling "
                    "from source, this can take a few minutes ...",
                )
                binaries.install_qwen3_tts_cpp(
                    progress_cb=lambda step, total: self.app.call_from_thread(
                        self._update_progress, step, total, True
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
                vocoder_path=vocoder_local,
            )
            reg.add(model)
            registry.save(reg)
            configs.write_all(reg)
            # NOTE: no daemon reload here — see _register_chat's comment.
        except Exception as e:  # noqa: BLE001
            self.downloading = False
            self.app.call_from_thread(self._set_hint, f"[red]Failed: {e}[/red]")
            return
        self.downloading = False
        self.app.call_from_thread(self.dismiss, True)

    def _update_progress(self, done: int, total: int, stage: bool = False) -> None:
        bar = self.query_one("#dl-bar", ProgressBar)
        status = self.query_one("#dl-status", Static)
        if stage:
            # Stage-based progress (the qwen3-tts.cpp source build): `total`
            # is a small step count, not bytes — "Step 3/5" reads sanely where
            # a MiB-based render would show nonsense like "3.0 / 5.0 MiB".
            bar.update(total=total, progress=min(done, total))
            status.update(f"Step {done}/{total}")
            return
        if total > 0:
            bar.update(total=total, progress=min(done, total))
            mib = done / (1024 * 1024)
            mib_total = total / (1024 * 1024)
            pct = (done / total) * 100 if total else 0.0
            status.update(f"{mib:.1f} / {mib_total:.1f} MiB  ({pct:.1f}%)")
        else:
            mib = done / (1024 * 1024)
            status.update(f"{mib:.1f} MiB")
