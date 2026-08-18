"""TOML-backed registry of locally configured models."""
from __future__ import annotations

import socket
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w

from inferhost.core import paths


def _port_free(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


@dataclass
class Model:
    name: str
    repo_id: str
    filename: str
    quant: str | None = None
    ctx: int = 8192
    port: int = 0
    size_gib: float = 0.0
    local_path: str = ""
    mmproj_path: str = ""  # multimodal projector for vision-capable models; "" = text only
    # Vision toggle. Only meaningful when mmproj_path is set. False serves the
    # model text-only (no --mmproj rendered), which frees the draft-based
    # speculative lanes (DFlash / MTP) that llama-server can't combine with
    # image batches. The projector file stays attached so flipping back to
    # True re-enables image input without re-downloading anything.
    vision_enabled: bool = True
    # TTS companion file. Non-empty marks this as a text-to-speech model,
    # served by the inferhost-tts daemon, NOT by llama-server/llama-swap.
    # For OuteTTS-style models it's the WavTokenizer vocoder GGUF; for Kokoro
    # (.onnx models) it's the bundled voices .npz. "" = not TTS.
    vocoder_path: str = ""
    # Modality. "chat" (default) = served by llama-server; "image" = served by
    # stable-diffusion.cpp's sd-server (fronted by llama-swap, OpenAI
    # /v1/images/generations). TTS is still detected via vocoder_path, not here.
    kind: str = "chat"
    # Image-model companion files (stable-diffusion.cpp split loading). For a
    # single-file checkpoint these stay empty and local_path is the checkpoint;
    # for Flux/SD3 they point at the standalone encoders/VAE and local_path is the
    # diffusion model. Any non-empty value triggers sd-server split-load mode.
    vae_path: str = ""
    clip_l_path: str = ""
    clip_g_path: str = ""
    t5xxl_path: str = ""
    # LLM/Qwen text encoder for Qwen-Image / Z-Image (sd-server --llm). These
    # models use a Qwen text encoder instead of CLIP/T5. Non-empty => split load.
    text_encoder_path: str = ""
    # Vision encoder / ViT (sd-server --llm_vision), e.g. the Qwen2.5-VL mmproj
    # that Qwen-Image-Edit uses to condition on the input image. "" = not used.
    vision_encoder_path: str = ""
    # Reasoning override. "" means "use the global Settings.reasoning value".
    # Non-empty values: "on", "off", "auto".
    reasoning: str = ""
    # Reasoning budget override. -2 is a sentinel meaning "use the global
    # Settings.reasoning_budget value". Real values: -1 (unlimited), 0 (none),
    # or any positive int.
    reasoning_budget: int = -2
    # How hard the model should think, for templates that grade it rather than
    # just switching it on and off (Qwen3.8: low / medium / high, where "high"
    # is the template's own alias for its xhigh default). "" = inherit the
    # global Settings.reasoning_effort. Rendered as a chat-template kwarg, not
    # a llama-server flag — see configs._llama_server_cmd.
    reasoning_effort: str = ""
    # Pinned models are emitted into a llama-swap group with `swap: false` so
    # they stay co-resident in VRAM. Two pinned models load simultaneously
    # instead of one unloading the other on swap.
    pin: bool = False

    # Per-model llama-server overrides. Each carries an "inherit from global
    # Settings" sentinel so the registry entry stays small and the global
    # default keeps working when the user hasn't tuned the model. The
    # ModelSettingsScreen ("Configure") exposes all of these.
    #   kv_quant_k / kv_quant_v: "" means inherit settings.kv_quant_k / _v.
    #   gpu_layers:               -1 means inherit settings.gpu_layers.
    #   parallel_slots:            0 means inherit settings.parallel_slots.
    #   flash_attention:          "" means inherit settings.flash_attention.
    #   threads:                   0 means inherit settings.threads.
    kv_quant_k: str = ""
    kv_quant_v: str = ""
    gpu_layers: int = -1
    parallel_slots: int = 0
    flash_attention: str = ""
    # CPU threads for generation (--threads). 0 = inherit settings.threads.
    threads: int = 0
    # MoE expert offload (--n-cpu-moe N): keep the experts of the first N layers
    # on CPU; the rest of the experts run on GPU. -1 = don't pass the flag (let
    # -ngl decide everything). 0 = all experts on GPU (fastest if it fits).
    # Only meaningful for Mixture-of-Experts models. Pairs with gpu_layers=99.
    cpu_moe_layers: int = -1
    # Lock the model in system RAM (--mlock) so the CPU-offloaded weights aren't
    # paged out/compressed — complements pin (which keeps a model in *VRAM*).
    # Most useful for models partly on CPU (low gpu_layers / --cpu-moe).
    mlock: bool = False
    # Free-form extra llama-server flags appended verbatim to the cmd, e.g.
    # "--embeddings --pooling last" for an embedding model. shlex-parsed so
    # quoted values work. Empty = nothing appended. No validation — a typo
    # surfaces as a llama-server startup failure in the model's err log.
    extra_args: str = ""
    # Per-model MTP speculative-decode draft depth (--spec-draft-n-max), only
    # applied to MTP-capable models. -1 is a sentinel meaning "inherit the
    # global Settings.spec_draft_n_max". 0 disables the MTP lane for this model;
    # a positive N drafts N tokens per step.
    spec_draft_n_max_override: int = -1
    # DFlash speculative decoding — a per-model draft attachment (like
    # mmproj_path), NOT a separate registry entry. draft_model_path is the
    # functional switch: non-empty attaches a z-lab block-diffusion draft GGUF
    # to this target via `--model-draft ... --spec-type draft-dflash`. When set,
    # DFlash wins over auto-detected MTP (explicit user act beats auto-detect).
    #   draft_model_path: shard-1 path of the draft GGUF ("" = no draft).
    #   draft_repo_id:    the HF repo the draft came from (shown in the UI).
    #   draft_size_gib:   draft weights size, folded into the VRAM estimate.
    draft_model_path: str = ""
    draft_repo_id: str = ""
    draft_size_gib: float = 0.0

    @property
    def vision_active(self) -> bool:
        """True when this model actually serves image input (projector attached
        AND the per-model vision toggle is on)."""
        return bool(self.mmproj_path) and self.vision_enabled

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quant"] = d["quant"] or ""
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Model:
        # Silently drop unknown keys (e.g. old cache_type_k / cache_type_v fields)
        # so we remain tolerant of registry files written by older inferhost versions.
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # Coerce types for fields that need it
        if "quant" in known:
            known["quant"] = known["quant"] or None
        if "ctx" in known:
            known["ctx"] = int(known["ctx"])
        if "port" in known:
            known["port"] = int(known["port"])
        if "size_gib" in known:
            known["size_gib"] = float(known["size_gib"])
        if "reasoning_budget" in known:
            known["reasoning_budget"] = int(known["reasoning_budget"])
        if "pin" in known:
            known["pin"] = bool(known["pin"])
        if "gpu_layers" in known:
            known["gpu_layers"] = int(known["gpu_layers"])
        if "parallel_slots" in known:
            known["parallel_slots"] = int(known["parallel_slots"])
        if "threads" in known:
            known["threads"] = int(known["threads"])
        if "cpu_moe_layers" in known:
            known["cpu_moe_layers"] = int(known["cpu_moe_layers"])
        if "mlock" in known:
            known["mlock"] = bool(known["mlock"])
        if "vision_enabled" in known:
            known["vision_enabled"] = bool(known["vision_enabled"])
        if "spec_draft_n_max_override" in known:
            known["spec_draft_n_max_override"] = int(known["spec_draft_n_max_override"])
        if "draft_size_gib" in known:
            known["draft_size_gib"] = float(known["draft_size_gib"])
        return cls(**known)


@dataclass
class Registry:
    models: list[Model] = field(default_factory=list)

    def add(self, model: Model) -> None:
        self.remove(model.name)
        self.models.append(model)

    def remove(self, name: str) -> bool:
        before = len(self.models)
        self.models = [m for m in self.models if m.name != name]
        return len(self.models) < before

    def rename(self, old: str, new: str) -> bool:
        """Rename a model in-place. Returns False if ``old`` is missing or ``new`` is taken."""
        if old == new:
            return False
        if self.get(new) is not None:
            return False
        m = self.get(old)
        if m is None:
            return False
        m.name = new
        return True

    def get(self, name: str) -> Model | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def names(self) -> list[str]:
        return [m.name for m in self.models]

    def next_port(self, base: int) -> int:
        used = {m.port for m in self.models if m.port}
        candidate = base + 1
        # Skip ports used in registry OR currently held by a foreign process
        while candidate in used or not _port_free(candidate):
            candidate += 1
            if candidate > base + 200:
                raise RuntimeError(f"Could not find free port near {base}")
        return candidate


def _path() -> Path:
    return paths.registry_path()


def load() -> Registry:
    p = _path()
    if not p.exists():
        return Registry()
    with p.open("rb") as f:
        data = tomllib.load(f)
    models = [Model.from_dict(m) for m in data.get("models", [])]
    return Registry(models=models)


def save(reg: Registry) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"models": [m.to_dict() for m in reg.models]}
    with p.open("wb") as f:
        tomli_w.dump(data, f)
