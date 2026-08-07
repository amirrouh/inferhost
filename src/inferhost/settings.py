"""Runtime configuration loaded from environment / .env file.

Configuration precedence (highest wins):

1. Real process environment variables (`INFERHOST_*`).
2. A TUI-managed overrides file at ``<config_dir>/inferhost.env`` — written when the
   user edits settings in the TUI.
3. A project-local ``.env`` in the current working directory.
4. Built-in defaults.

Users who prefer to manage everything by hand can stick to step 3 (a `.env` file).
Users who change settings in the TUI get step 2 — the TUI just writes another
`.env`-style file in the user config dir.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _default_config_dir() -> Path:
    return _expand("~/.config/inferhost")


def overrides_env_path() -> Path:
    """Where the TUI persists user-edited settings."""
    return _default_config_dir() / "inferhost.env"


# pydantic-settings reads env_file entries in order; later entries override earlier ones.
# So: .env (project-local) is loaded first, then the TUI-managed file overrides it,
# then real env vars override both.
_ENV_FILES: tuple[str, ...] = (
    ".env",
    str(Path.cwd() / ".env"),
    str(overrides_env_path()),
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERHOST_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gateway_port: int = 9001
    swap_port: int = 9090
    # Port for the inferhost-tts daemon, which serves TTS models (Kokoro via
    # kokoro-onnx, OuteTTS via the standalone llama-tts binary) and exposes
    # POST /v1/audio/speech. LiteLLM routes the gateway's /v1/audio/speech
    # here; it's also reachable directly.
    tts_port: int = 9092
    # Default Kokoro voice for /v1/audio/speech requests that omit `voice` or
    # send an unknown name. Any voice in the downloaded bundle works
    # (af_heart, am_michael, bf_emma, ...); OpenAI preset names (alloy, nova,
    # ...) are mapped automatically in tts_serve.
    tts_voice: str = "af_heart"

    # Bind addresses for the two daemons. Both default to 0.0.0.0 so that on
    # any networked GPU box (Tailscale, LAN, VPC) inferhost is reachable from
    # other machines after a single `inferhost start` — no env-var editing.
    # Override to "127.0.0.1" if you need loopback-only on either daemon.
    gateway_host: str = "0.0.0.0"  # noqa: S104 — intentional, this is the public gate
    swap_host: str = "0.0.0.0"     # noqa: S104 — same; loopback-only is opt-in
    tts_host: str = "0.0.0.0"      # noqa: S104 — same; loopback-only is opt-in

    data_dir: Path = Field(default=Path("~/.local/share/inferhost"))
    config_dir: Path = Field(default=Path("~/.config/inferhost"))
    hf_cache: Path = Field(default=Path("~/.cache/huggingface"))

    gpu_layers: int = 99
    # Context window a SINGLE request gets, in tokens, for newly added models.
    # This is the per-request window, not the total KV cache: the renderer
    # multiplies it by `parallel_slots` when it emits llama-server's `-c`, so
    # raising slot count never shrinks the window a client can actually use.
    default_ctx: int = 8192
    flash_attention: str = "on"

    # Completion cap advertised to OpenAI-wire clients as `max_output_tokens`.
    # In llama.cpp, output and input share one context budget — there is no
    # separate server-side output limit — so 0 (the default) advertises the
    # full served context window, which is the honest physical maximum. Some
    # agent frameworks instead *reserve* `max_output_tokens` worth of room for
    # the reply and subtract it from the window; on those, advertising the full
    # window leaves no room for the prompt. Set this to a positive N (e.g.
    # 8192) to cap the advertised completion length without touching the real
    # context window. The advertised value is always min(N, served context).
    max_output_tokens: int = 0

    # Number of parallel request slots per llama-server instance (--parallel N).
    # 1 is the safest default — one in-flight request at a time per model, no KV
    # cache contention. Bump this if you need concurrency on the same model.
    # Costs VRAM: each slot gets its own full context window, so N slots means
    # N x the KV cache. (llama-server divides `-c` across slots; inferhost sizes
    # `-c` as ctx x slots so each request still gets the configured window.)
    parallel_slots: int = 1

    # CPU threads for generation (--threads). 0 means "don't pass the flag" so
    # llama-server auto-picks (its default is the physical core count). Matters
    # most for models partly on CPU (low -ngl, --cpu-moe); for a fully GPU-
    # offloaded model it has little effect. More threads than physical cores
    # usually hurts.
    threads: int = 0

    # Reasoning / "thinking" mode for capable models (DeepSeek, Qwen3-Thinking,
    # GPT-OSS, ...). Maps to llama-server's --reasoning flag.
    #   "auto" — let the model decide based on its chat template (default)
    #   "on"   — always emit reasoning_content
    #   "off"  — suppress thinking
    reasoning: str = "auto"
    # Token cap on thinking ( --reasoning-budget ). -1 = unlimited, 0 = no
    # thinking, positive N = cut off after N tokens.
    reasoning_budget: int = -1

    # Speculative decoding — only applied to MTP-capable models (filename contains "mtp").
    # Two lanes are stacked: --spec-type draft-mtp AND --spec-type ngram-mod.
    # Set any of these to 0 to disable that lane individually.
    spec_draft_n_max: int = 2          # MTP draft tokens per step (PR author: 2 is sweet spot)
    spec_ngram_mod_n_match: int = 24   # min matching sequence length before ngram drafts
    spec_ngram_mod_n_min: int = 48     # min context window to search back through
    spec_ngram_mod_n_max: int = 64     # max draft tokens ngram-mod proposes on strong match

    # DFlash speculative decoding — draft depth (--spec-draft-n-max) applied
    # when a z-lab block-diffusion draft GGUF is attached to a model
    # (Model.draft_model_path). 3-4 is the consumer-GPU sweet spot; big GPUs
    # can push it to 15-16. The per-model spec_draft_n_max_override (>=0) wins
    # over this global; 0 disables the DFlash lane for that model.
    spec_dflash_n_max: int = 4

    llamacpp_version: str = "latest"
    llamaswap_version: str = "latest"
    # stable-diffusion.cpp release (image generation via sd-server). Rolling
    # master-* tags upstream; "latest" pulls the newest. Bundled automatically so
    # install/update light up /v1/images/generations once an image model exists.
    sdcpp_version: str = "latest"

    # Default image-generation sampling, baked into the sd-server launch cmd for
    # image models. Per-request `size` still overrides; per-model overrides go in
    # the model's extra_args. 0 = let sd-server use its own default.
    sd_steps: int = 0
    sd_cfg_scale: float = 0.0
    sd_sampler: str = ""

    # Which prebuilt llama.cpp binary variant to download from the upstream
    # ggml-org/llama.cpp GitHub release. "auto" picks based on the hardware
    # probe (NVIDIA -> vulkan, Apple Silicon -> macOS arm64, otherwise CPU).
    # Override values: vulkan | cuda | rocm | sycl | openvino | cpu | metal.
    # Note: upstream does NOT publish a Linux CUDA build — pick "vulkan" on
    # NVIDIA Linux boxes, or set INFERHOST_LLAMA_SERVER_PATH to a custom
    # CUDA-enabled binary.
    llamacpp_backend: str = "auto"

    # KV-cache quantization, applied as `-ctk` / `-ctv` to llama-server.
    # Default is q8_0 for both K and V — ~2x compression of the f16 baseline
    # with near-lossless quality. Override per axis if you have spare VRAM
    # (f16) or want more aggressive compression on V (q5_0 / q4_0).
    # Set either to "off" to omit the flag entirely.
    kv_quant_k: str = "q8_0"
    kv_quant_v: str = "q8_0"

    # How often (seconds) the inferhost-pinwatch daemon checks llama-swap and
    # re-loads pinned models that were evicted (by an exclusive swap, a crash,
    # or a daemon restart) once the GPU is idle again. The watcher never
    # preempts a resident swappable model — it only refills freed VRAM.
    pinwatch_poll_s: int = 10

    # Textual mouse capture. Defaults ON so click-on-button works out of the box.
    # Trade-off: with capture on, terminal-native click-and-drag selection is
    # intercepted by Textual — hold Shift while selecting to bypass it in most
    # terminals (GNOME Terminal, iTerm2, Kitty, Wezterm, Alacritty). Set
    # INFERHOST_MOUSE=off to restore native selection.
    mouse: bool = True

    log_level: str = "INFO"

    def model_post_init(self, __ctx) -> None:  # type: ignore[override]
        self.data_dir = _expand(self.data_dir)
        self.config_dir = _expand(self.config_dir)
        self.hf_cache = _expand(self.hf_cache)


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings


# ---- user-editable overrides (written by the TUI) -------------------------------

# Whitelist of fields the TUI is allowed to persist. Keeps the override file tidy
# and prevents accidental writes of internal / path-y settings.
EDITABLE_FIELDS: tuple[str, ...] = (
    "swap_port",
    "swap_host",
    "gateway_port",
    "gateway_host",
    "tts_port",
    "tts_host",
    "tts_voice",
    "default_ctx",
    "max_output_tokens",
    "gpu_layers",
    "flash_attention",
    "parallel_slots",
    "threads",
    "reasoning",
    "reasoning_budget",
    "spec_dflash_n_max",
    "kv_quant_k",
    "kv_quant_v",
    "llamacpp_version",
    "llamacpp_backend",
    "sdcpp_version",
    "sd_steps",
    "sd_cfg_scale",
    "sd_sampler",
)


# Accepted llama.cpp KV cache types. Used for TUI validation so the user gets
# a clear error instead of a llama-server abort on next load. These match
# upstream ggml-org/llama.cpp's `-ctk` / `-ctv` allowed values.
KV_QUANT_VALUES: tuple[str, ...] = (
    "f32", "f16", "bf16",
    "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl",
    "off",
)


# Accepted llama.cpp backend choices for the prebuilt-asset picker. "auto"
# defers to the hardware probe.
LLAMACPP_BACKEND_VALUES: tuple[str, ...] = (
    "auto", "vulkan", "cuda", "rocm", "sycl", "openvino", "cpu", "metal",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Managed by inferhost — written when you edit Settings in the TUI.",
        "# You can also edit this file by hand; values follow KEY=VALUE format.",
        "",
    ]
    for key in sorted(values):
        lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_overrides() -> dict[str, str]:
    """Read the TUI-managed overrides file as a dict of {bare_field: value}.

    Strips the ``INFERHOST_`` prefix so callers see field names matching ``Settings``.
    """
    raw = _parse_env_file(overrides_env_path())
    prefix = "INFERHOST_"
    return {
        k[len(prefix):].lower(): v
        for k, v in raw.items()
        if k.startswith(prefix)
    }


def save_overrides(updates: dict[str, object]) -> Path:
    """Merge ``updates`` into the TUI-managed overrides file and reload settings.

    Keys must be ``Settings`` field names (e.g. ``swap_port``). Only fields listed in
    ``EDITABLE_FIELDS`` are persisted; the rest are silently ignored.

    Also re-writes the file from a dict, which collapses any duplicate KEY=...
    lines left behind by manual ``echo >>`` edits.
    """
    path = overrides_env_path()
    existing = _parse_env_file(path)
    prefix = "INFERHOST_"
    for key, value in updates.items():
        if key not in EDITABLE_FIELDS:
            continue
        existing[f"{prefix}{key.upper()}"] = str(value)
    _write_env_file(path, existing)
    reload_settings()
    return path
