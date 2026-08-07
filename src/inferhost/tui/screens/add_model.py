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
        self._files_hint: str = ""

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
            "tts": (
                "Paste a link or owner/repo, e.g. hexgrad/Kokoro-82M  ·  "
                "unsloth/orpheus-3b-0.1-ft-GGUF  ·  OuteAI/OuteTTS-0.2-500M-GGUF"
            ),
        }.get(
            self.kind,
            "Paste a link or owner/repo, e.g. Qwen/Qwen2.5-7B-Instruct-GGUF",
        )
        self.query_one("#repo-input", Input).placeholder = placeholder
        if self.kind == "image":
            files_word = ".gguf / .safetensors"
        elif self.kind == "tts":
            files_word = ".onnx / .gguf"
        else:
            files_word = ".gguf"
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
        # Companion files (mmproj projectors, DSpark drafters) stay listed but
        # never take the recommendation star — their small size and high-rank
        # quants (BF16) would otherwise beat every real candidate.
        main_files = [f for f in files if not hf.is_companion_file(f.filename)]
        best = quant.pick_best(main_files or files, budget)
        list_view = self.query_one("#quant-list", ListView)
        list_view.clear()
        for i, f in enumerate(files):
            marker = "*" if best is not None and f.filename == best.filename else " "
            fits = "+" if f.size_gib <= max(0.0, budget - 1.5) else "."
            parts_tag = f"  [{len(f.parts)} parts]" if f.parts else ""
            label = f"{marker} {fits} {f.quant or '?':<8}  {f.size_gib:>5} GiB  {f.filename}{parts_tag}"
            list_view.append(ListItem(Label(label), name=str(i)))
        if best is not None:
            self._files_hint = (
                f"VRAM: {vram:.1f} GiB. * = recommended.  Select a row and press Add."
            )
        else:
            self._files_hint = f"VRAM: {vram:.1f} GiB. No file fits; smallest will be used."
        self._set_hint(self._files_hint)

    @on(ListView.Highlighted, "#quant-list")
    def _on_pick(self, ev: ListView.Highlighted) -> None:
        if ev.item is not None and ev.item.name is not None:
            self.selected_idx = int(ev.item.name)
            pick = self.files[self.selected_idx]
            if pick.quant in ("Q2_0", "PQ2_0"):
                # Group-128 ternary packing — readable only by PrismML's
                # llama.cpp fork, not the upstream llama-server we ship.
                self._set_hint(
                    "[yellow]This group-128 ternary file (Q2_0/PQ2_0) needs PrismML's "
                    "llama.cpp fork. Pick the *_g64 file to run on the bundled "
                    "mainline llama-server.[/yellow]"
                )
            elif self._files_hint:
                self._set_hint(self._files_hint)

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

        Three engine families, told apart by the picked file and the repo:
        - ``.onnx`` — Kokoro (kokoro-onnx, in-process): the per-voice style
          vectors from the same repo are bundled into one voices .npz, stored
          as vocoder_path.
        - ``.gguf`` in an Orpheus repo — Orpheus (llama-server + SNAC): the
          small SNAC decoder .onnx is fetched from its fixed community export
          and stored as vocoder_path; the GGUF itself is served by llama-swap.
        - ``.gguf`` otherwise — OuteTTS-style (llama-tts): needs the
          WavTokenizer / vocoder GGUF companion from the same repo.
        Either way a non-empty ``registry.Model.vocoder_path`` is what marks
        the registered model as TTS; it's required here, not optional, since a
        TTS pick without it can't actually be served. Its extension is also
        the engine discriminator at serve time (``configs.tts_engine``).
        """
        try:
            local = self._download_main_or_parts(pick)
            if pick.filename.endswith(".onnx"):
                vocoder_local = self._download_kokoro_voices(pick.repo_id)
            elif hf.is_orpheus_repo(pick.repo_id):
                vocoder_local = self._download_companion(
                    hf.SNAC_ONNX_REPO, hf.SNAC_DECODER_FILE, "SNAC audio decoder"
                )
            else:
                vocoder_name = hf.find_vocoder(pick.repo_id)
                if not vocoder_name:
                    raise RuntimeError(
                        f"No vocoder GGUF found in {pick.repo_id} — an "
                        "OuteTTS-style model needs its WavTokenizer companion "
                        "in the same repo (Orpheus repos are detected by name)."
                    )
                vocoder_local = self._download_companion(pick.repo_id, vocoder_name, "TTS vocoder")

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

    def _download_kokoro_voices(self, repo_id: str) -> str:
        """Fetch every per-voice .bin and bundle them into one voices .npz.

        kokoro-onnx loads a single .npz keyed by voice name; the Kokoro ONNX
        repo ships one raw .bin per voice instead, so the bundle is assembled
        locally. The .npz path is stored as the model's vocoder_path.
        """
        voice_files = hf.list_kokoro_voice_files(repo_id)
        if not voice_files:
            raise RuntimeError(
                f"No voices/*.bin files found in {repo_id} — a Kokoro repo "
                "must ship its per-voice style vectors."
            )
        total = sum(size for _, size in voice_files) or 1
        self.app.call_from_thread(
            self._set_hint, f"Downloading {len(voice_files)} Kokoro voices ..."
        )
        self.app.call_from_thread(self._update_progress, 0, total)
        done = 0
        local: dict[str, Path] = {}
        for fname, size in voice_files:
            local[Path(fname).stem] = hf.download_gguf(repo_id, fname)
            done += size
            self.app.call_from_thread(self._update_progress, done, total)
        paths.ensure_dirs()
        out = paths.models_dir() / f"{hf.normalize_name(repo_id)}-voices.npz"
        return str(hf.build_kokoro_voices_npz(local, out))

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
