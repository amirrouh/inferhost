"""Render llama-swap.yaml and litellm.yaml from the model registry."""
from __future__ import annotations

import contextlib
import shlex
from pathlib import Path

import yaml

from inferhost.core import gguf, paths
from inferhost.core.llama_caps import pick_kv_quant, supported_cache_types
from inferhost.core.registry import Model, Registry
from inferhost.settings import settings


def _model_path(m: Model) -> str:
    return m.local_path or str(paths.models_dir() / m.filename)


def effective_ctx(m: Model, notices: list[str] | None = None) -> int:
    """Context window we actually serve & advertise for ``m``.

    The user configures ``-c`` (``m.ctx``), but a GGUF can only be loaded up to
    its native trained context (``<arch>.context_length`` in the file). If the
    configured window exceeds that, llama-server silently clamps on load — so
    what agents are *told* (litellm / Hermes) would no longer match what's
    actually served. We read the native window straight from the file on disk
    and clamp to it here so the advertised and served windows always agree.
    """
    native = gguf.native_context_cached(_model_path(m))
    if native and native > 0 and m.ctx > native:
        if notices is not None:
            notices.append(
                f"{m.name}: configured context {m.ctx} exceeds the model's "
                f"native trained context {native}; serving {native}."
            )
        return native
    return m.ctx


def is_tts(m: Model) -> bool:
    """A model is text-to-speech when it carries a vocoder GGUF.

    TTS models are served by the inferhost-tts daemon (the standalone llama-tts
    binary), not by llama-server/llama-swap, so they are excluded from the
    llama-swap config and registered with LiteLLM as audio_speech endpoints.
    """
    return bool(m.vocoder_path)


def is_image(m: Model) -> bool:
    """A model is image-generation when kind == 'image'.

    Image models are served by stable-diffusion.cpp's sd-server, fronted by
    llama-swap (so they swap VRAM with LLMs), and exposed to LiteLLM as
    image_generation endpoints.
    """
    return m.kind == "image"


def _sd_server_cmd(m: Model, notices: list[str] | None = None) -> str:
    """Build the sd-server launch command for an image model (run by llama-swap).

    Single-file checkpoint -> `-m <ckpt>`. Split (Flux/SD3) -> `--diffusion-model`
    plus whichever encoder/VAE files are set. Generation defaults (steps/cfg/
    sampler) come from Settings; per-model raw flags via extra_args; per-request
    `size` is honored by sd-server itself.
    """
    s = settings()
    sd_bin = paths.sd_server_path()
    parts = [
        "env",
        f"LD_LIBRARY_PATH={paths.sd_bin_dir()}",
        str(sd_bin),
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(m.port),
        "--diffusion-fa",  # flash attention in the diffusion model — saves VRAM
    ]
    # Split load when any companion file is set; otherwise single checkpoint.
    if (m.vae_path or m.clip_l_path or m.clip_g_path or m.t5xxl_path
            or m.text_encoder_path or m.vision_encoder_path):
        parts += ["--diffusion-model", _model_path(m)]
        if m.vae_path:
            parts += ["--vae", m.vae_path]
        if m.clip_l_path:
            parts += ["--clip_l", m.clip_l_path]
        if m.clip_g_path:
            parts += ["--clip_g", m.clip_g_path]
        if m.t5xxl_path:
            parts += ["--t5xxl", m.t5xxl_path]
        if m.text_encoder_path:
            # Qwen/LLM text encoder (Qwen-Image, Z-Image). --llm is the current
            # flag; --qwen2vl is a deprecated alias in sd-server.
            parts += ["--llm", m.text_encoder_path]
        if m.vision_encoder_path:
            # Vision ViT/mmproj (Qwen-Image-Edit conditions on the input image).
            parts += ["--llm_vision", m.vision_encoder_path]
    else:
        parts += ["-m", _model_path(m)]
    # Optional generation defaults (0 / "" = let sd-server decide).
    if s.sd_steps > 0:
        parts += ["--steps", str(s.sd_steps)]
    if s.sd_cfg_scale > 0:
        parts += ["--cfg-scale", str(s.sd_cfg_scale)]
    if s.sd_sampler:
        parts += ["--sampling-method", s.sd_sampler]
    if m.extra_args.strip():
        try:
            parts += shlex.split(m.extra_args)
        except ValueError as e:
            if notices is not None:
                notices.append(f"{m.name}: extra_args parse error ({e}); flags ignored")
    inner = " ".join(shlex.quote(p) for p in parts)
    err_log = paths.logs_dir() / f"{m.name}.err.log"
    wrapped = f"exec {inner} 2>>{shlex.quote(str(err_log))}"
    return f"/bin/sh -c {shlex.quote(wrapped)}"


def is_mtp_capable(m: Model) -> bool:
    """True if the model can actually use MTP speculative decoding.

    Primary signal is the GGUF metadata: a real NextN/MTP model advertises
    ``*.nextn_predict_layers`` (or similar) — we read that straight from the file
    (:func:`gguf.has_mtp_heads_cached`). This is authoritative: forcing an MTP
    context on a model *without* those layers makes llama-server abort with
    "model doesn't contain MTP layers", so we must not enable it then.

    Filename/name containing 'mtp' is kept as a fallback signal — it lets a user
    force-enable by renaming, and covers GGUFs whose metadata predates the key —
    but the metadata check is what makes detection automatic and correct.
    """
    if gguf.has_mtp_heads_cached(_model_path(m)):
        return True
    haystack = f"{m.filename} {m.name}".lower()
    return "mtp" in haystack


def _llama_server_cmd(m: Model, notices: list[str] | None = None) -> str:
    s = settings()
    bin_path = paths.llama_server_path()
    # Resolve per-model overrides against the global Settings. Empty / sentinel
    # values fall back to the Settings default so an untuned model still works.
    eff_gpu_layers = m.gpu_layers if m.gpu_layers >= 0 else s.gpu_layers
    eff_parallel = m.parallel_slots if m.parallel_slots > 0 else s.parallel_slots
    eff_fa = m.flash_attention if m.flash_attention else s.flash_attention
    eff_threads = m.threads if m.threads > 0 else s.threads
    # Reasoning: model-level override wins ("" means inherit global), otherwise
    # fall back to the global Settings value. `--reasoning off` sets
    # enable_thinking=false, which is the mechanism that actually suppresses
    # thinking. (Do NOT also pin --reasoning-budget to 0 for "off": the budget-0
    # hard-stop force-injects the end-of-thinking tag at token 0, which makes
    # some finetuned/MTP models run away instead of answering.)
    eff_reasoning = m.reasoning if m.reasoning else s.reasoning
    eff_reasoning_budget = (
        m.reasoning_budget if m.reasoning_budget != -2 else s.reasoning_budget
    )
    parts = [
        "env",
        f"LD_LIBRARY_PATH={paths.bin_dir()}",
        str(bin_path),
        "--model", _model_path(m),
        "--host", "127.0.0.1",
        "--port", str(m.port),
        "-ngl", str(eff_gpu_layers),
        "-c", str(effective_ctx(m, notices=notices)),
        "-fa", eff_fa,
        "--parallel", str(max(1, eff_parallel)),
        # Use the model's own jinja chat template from the GGUF metadata.
        # llama-server's legacy built-in templates strip tool-call blocks and
        # OpenAI vision content parts, so without --jinja a tool-trained or
        # vision-trained GGUF silently behaves like a plain text-only model.
        # Newer llama-server defaults this on, older builds default off — pass
        # it explicitly so behavior is consistent across the prebuilt binaries.
        "--jinja",
        # Reasoning (resolved above; budget pinned to 0 when reasoning is off).
        "--reasoning", eff_reasoning,
        "--reasoning-budget", str(eff_reasoning_budget),
        # Intentionally NOT passing --log-disable: llama-server's stderr is the
        # only place that prints the actual reason for an abort (GGML_ASSERT,
        # CUDA OOM, etc.). llama-swap captures the child stderr into its own
        # log, so silencing it means crashes are indistinguishable in
        # postmortem. Verbosity cost is negligible.
    ]
    # CPU threads (--threads). 0 (global or per-model) means "don't pass it" so
    # llama-server auto-picks the physical core count.
    if eff_threads > 0:
        parts += ["--threads", str(eff_threads)]
    # MoE expert offload (--n-cpu-moe N): keep the first N layers' experts on
    # CPU, rest on GPU. -1 = omit. For a MoE model, pairing gpu_layers=99 with a
    # low N puts most experts on GPU (big speedup); N=0 = all experts on GPU.
    if m.cpu_moe_layers >= 0:
        parts += ["--n-cpu-moe", str(m.cpu_moe_layers)]
    # Lock the model into system RAM (--mlock) so CPU-offloaded weights aren't
    # paged out. Per-model opt-in; complements pin (which is about VRAM).
    if m.mlock:
        parts += ["--mlock"]
    # KV cache quantization. Default: K=q8_0, V=q8_0 — ~2x compression of the
    # f16 baseline with near-lossless quality. Per-model override wins ("" means
    # inherit). If the installed llama-server doesn't support the requested
    # value (e.g. a custom build is missing a codec), `pick_kv_quant`
    # substitutes a supported fallback and returns a notice so we don't fail
    # every model load with a cryptic 502.
    kv_quant_k = m.kv_quant_k or getattr(s, "kv_quant_k", "q8_0")
    kv_quant_v = m.kv_quant_v or getattr(s, "kv_quant_v", "q8_0")
    supported = supported_cache_types()
    if kv_quant_k and kv_quant_k != "off":
        chosen, warn = pick_kv_quant(kv_quant_k, supported)
        if warn and notices is not None:
            notices.append(f"-ctk: {warn}")
        parts += ["-ctk", chosen]
    if kv_quant_v and kv_quant_v != "off":
        chosen, warn = pick_kv_quant(kv_quant_v, supported)
        if warn and notices is not None:
            notices.append(f"-ctv: {warn}")
        parts += ["-ctv", chosen]
    if m.mmproj_path:
        # Vision (multimodal projector). llama-server emits image-tokens via OpenAI
        # vision content blocks once --mmproj is attached. Long form (not -mm) so
        # the rendered YAML and crash logs stay greppable for "mmproj".
        parts += ["--mmproj", m.mmproj_path]
    # Free-form per-model extra args (e.g. "--embeddings --pooling last" for an
    # embedding model). Appended after the structured flags so a user override
    # like "-c 16384" takes precedence over the values we already emitted. A
    # malformed string (unbalanced quote) becomes a notice instead of crashing
    # the whole config render.
    if m.extra_args.strip():
        try:
            parts += shlex.split(m.extra_args)
        except ValueError as e:
            if notices is not None:
                notices.append(
                    f"{m.name}: extra_args parse error ({e}); flags ignored"
                )
    if is_mtp_capable(m):
        # Stack two speculative-decode lanes (llama.cpp accepts multiple --spec-type):
        #   1. draft-mtp uses the MTP heads baked into the GGUF
        #   2. ngram-mod uses pattern lookup over already-generated text
        # MTP handles novel generation; ngram-mod dominates on repeated patterns
        # (code, function names, repeated constructs).
        # Per-model override wins; -1 means inherit the global default.
        eff_spec_draft = (
            m.spec_draft_n_max_override
            if m.spec_draft_n_max_override >= 0
            else s.spec_draft_n_max
        )
        if eff_spec_draft > 0:
            parts += [
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(eff_spec_draft),
            ]
        if s.spec_ngram_mod_n_max > 0:
            parts += [
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-match", str(s.spec_ngram_mod_n_match),
                "--spec-ngram-mod-n-min", str(s.spec_ngram_mod_n_min),
                "--spec-ngram-mod-n-max", str(s.spec_ngram_mod_n_max),
            ]
    # Capture llama-server's stderr to a per-model file. llama-swap parses
    # `cmd:` into argv itself (NOT via `sh -c`) and discards the child's
    # stderr to a pipe that it reads and throws away — so the actual abort
    # reason (GGML_ASSERT, CUDA OOM, kernel edge case) is lost.
    #
    # Wrapping the whole cmd in `/bin/sh -c '<inner> 2>>file'` forces an
    # explicit shell to handle the redirect BEFORE llama-server starts,
    # so the child's fd 2 points at the file instead of llama-swap's pipe.
    inner = " ".join(shlex.quote(p) for p in parts)
    err_log = paths.logs_dir() / f"{m.name}.err.log"
    wrapped = f"exec {inner} 2>>{shlex.quote(str(err_log))}"
    return f"/bin/sh -c {shlex.quote(wrapped)}"


def render_llama_swap(reg: Registry, notices: list[str] | None = None) -> dict:
    models_block: dict[str, dict] = {}
    # Models llama-swap fronts: chat/vision (llama-server) + image (sd-server).
    # TTS models are NOT here — they're served by the standalone inferhost-tts
    # daemon, not llama-swap.
    swap_models = [m for m in reg.models if not is_tts(m)]
    for m in swap_models:
        # ttl=0 disables llama-swap's idle-unload, which is what "pinned"
        # users actually expect ("keep this model in VRAM"). The group's
        # `swap: false` only prevents eviction-by-other-model, it does NOT
        # override per-model TTL — without ttl=0 here, a pinned big model
        # still dies after 10 min of inactivity and pays its full reload
        # cost on the next request. Swappable models keep the 10 min TTL
        # so VRAM is reclaimed when they're idle.
        ttl = 0 if m.pin else 600
        if is_image(m):
            # Image model: run sd-server instead of llama-server. sd-server has no
            # /health endpoint, so point llama-swap's readiness check at
            # /v1/models (200 once the model has loaded).
            models_block[m.name] = {
                "cmd": _sd_server_cmd(m, notices=notices),
                "proxy": f"http://127.0.0.1:{m.port}",
                "checkEndpoint": "/v1/models",
                "ttl": ttl,
            }
        else:
            models_block[m.name] = {
                "cmd": _llama_server_cmd(m, notices=notices),
                "proxy": f"http://127.0.0.1:{m.port}",
                "ttl": ttl,
            }
    cfg: dict = {
        "healthCheckTimeout": 300,
        "logRequests": False,
        "models": models_block,
    }
    # Two lifecycle groups:
    #   - "pinned":     swap=false, members stay co-resident in VRAM
    #   - "swappable":  swap=true + exclusive=true, only one unpinned model
    #                   resident at a time. The exclusive flag is what makes
    #                   llama-swap actually evict the previous model when a
    #                   different unpinned model is requested — without it,
    #                   models accumulate in VRAM and a large model fails to
    #                   load with cudaMalloc OOM even though it should fit.
    pinned = [m.name for m in swap_models if m.pin]
    swappable = [m.name for m in swap_models if not m.pin]
    groups: dict = {}
    if pinned:
        groups["pinned"] = {
            "swap": False,
            "exclusive": False,
            "members": pinned,
        }
    if swappable:
        groups["swappable"] = {
            "swap": True,
            "exclusive": True,
            "members": swappable,
        }
    if groups:
        cfg["groups"] = groups
    return cfg


def render_litellm(reg: Registry) -> dict:
    s = settings()
    model_list = []
    for m in reg.models:
        if is_tts(m):
            # TTS model: route the gateway's /v1/audio/speech for this model to
            # the inferhost-tts daemon (OpenAI-compatible). `mode: audio_speech`
            # tells LiteLLM this is a speech endpoint, not a chat/completions one.
            model_list.append(
                {
                    "model_name": m.name,
                    "litellm_params": {
                        "model": f"openai/{m.name}",
                        "api_base": f"http://127.0.0.1:{s.tts_port}/v1",
                        "api_key": "none",
                    },
                    "model_info": {"mode": "audio_speech"},
                }
            )
            continue
        if is_image(m):
            # Image model: route /v1/images/generations through llama-swap (which
            # lazy-starts sd-server). mode=image_generation tells LiteLLM this is
            # an images endpoint, not chat.
            model_list.append(
                {
                    "model_name": m.name,
                    "litellm_params": {
                        "model": f"openai/{m.name}",
                        "api_base": f"http://127.0.0.1:{s.swap_port}/v1",
                        "api_key": "none",
                    },
                    "model_info": {"mode": "image_generation"},
                }
            )
            continue
        # Advertise the window we actually serve — clamped to the GGUF's native
        # trained context — not the raw configured -c, so a client never sends a
        # prompt longer than llama-server can hold.
        adv_ctx = effective_ctx(m)
        # Completion cap: 0 means "no separate limit" (llama.cpp draws output
        # from the same context budget), so advertise the full window; a
        # positive setting caps it for frameworks that reserve output room.
        adv_out = min(adv_ctx, s.max_output_tokens) if s.max_output_tokens > 0 else adv_ctx
        model_list.append(
            {
                "model_name": m.name,
                "litellm_params": {
                    "model": f"openai/{m.name}",
                    "api_base": f"http://127.0.0.1:{s.swap_port}/v1",
                    "api_key": "none",
                },
                # Expose context window + capability flags so OpenAI-wire
                # clients (Hermes Agent, litellm callers, Open WebUI) auto-detect
                # context_length, tool-calling, and vision. Without these flags
                # clients can refuse to send tool/image content even when the
                # underlying llama-server supports them.
                "model_info": {
                    "max_tokens": adv_ctx,
                    "max_input_tokens": adv_ctx,
                    "max_output_tokens": adv_out,
                    "supports_function_calling": True,
                    "supports_tool_choice": True,
                    "supports_vision": bool(m.mmproj_path),
                },
            }
        )
    return {
        "model_list": model_list,
        "litellm_settings": {"drop_params": False},
    }


def _dump_yaml(data: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _hermes_base_urls() -> list[str]:
    """Loopback URLs Hermes is likely to address the litellm gateway by.

    Hermes keys its persistent context cache as ``<model>@<base_url>``. The
    same gateway is reachable under several loopback aliases (``localhost``,
    ``127.0.0.1``) and — if the user has wired Hermes to a LAN/Tailscale IP —
    under ``gateway_host`` too. Seeding all of them is cheap and saves the
    user from having to know which one Hermes will use.
    """
    s = settings()
    hosts = ["localhost", "127.0.0.1"]
    if s.gateway_host and s.gateway_host not in {"0.0.0.0", "::", ""} | set(hosts):
        hosts.append(s.gateway_host)
    return [f"http://{h}:{s.gateway_port}/v1" for h in hosts]


def seed_hermes_context_cache(reg: Registry) -> Path | None:
    """Write each model's actual loaded context into Hermes' persistent cache.

    Hermes auto-resolves context length per ``(model, base_url)`` and stores
    discovered values in ``$HERMES_HOME/context_length_cache.yaml``. LiteLLM's
    OpenAI-shape ``/v1/models`` endpoint can't carry that field, so the live
    probe falls back to a generic default and Hermes' TUI shows the wrong
    window. By seeding the cache from llama-swap's authoritative ``-c <N>``
    value, the inferhost stack stays the single source of truth.

    Other cache entries (non-inferhost providers Hermes has discovered) are
    preserved.
    """
    cache_path = paths.hermes_context_cache_path()
    base_urls = _hermes_base_urls()

    existing: dict[str, int] = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            existing = dict(data.get("context_lengths", {}) or {})
        except (OSError, yaml.YAMLError):
            existing = {}

    merged = dict(existing)
    for m in reg.models:
        if is_tts(m) or is_image(m):
            continue  # TTS/image models have no chat context window to advertise.
        # Seed the window we actually serve (clamped to the file's native
        # context), so Hermes' TUI matches what llama-server loaded.
        adv_ctx = effective_ctx(m)
        for url in base_urls:
            merged[f"{m.name}@{url}"] = int(adv_ctx)

    if merged == existing:
        return cache_path if cache_path.exists() else None

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"context_lengths": merged}, f, default_flow_style=False)
    except OSError:
        return None
    return cache_path


def write_all(reg: Registry) -> tuple[Path, Path]:
    # Collect any notices raised while rendering (e.g. unsupported KV quant
    # downgrades). We dedupe per-message because the same warning fires once
    # per model otherwise. Then persist to a small file so the CLI/TUI start
    # paths can surface them to the user instead of failing silently.
    notices: list[str] = []
    swap_cfg = render_llama_swap(reg, notices=notices)
    litellm_cfg = render_litellm(reg)
    _dump_yaml(swap_cfg, paths.llama_swap_config_path())
    _dump_yaml(litellm_cfg, paths.litellm_config_path())
    # Seed Hermes' context cache from the same registry, so any Hermes TUI
    # pointed at this gateway shows the real loaded window without manual
    # config edits. Silent best-effort — Hermes may not be installed.
    with contextlib.suppress(Exception):
        seed_hermes_context_cache(reg)
    notice_file = paths.notices_path()
    notice_file.parent.mkdir(parents=True, exist_ok=True)
    deduped = list(dict.fromkeys(notices))
    if deduped:
        notice_file.write_text("\n".join(deduped) + "\n")
    else:
        # Clear stale notices from a previous render so the user isn't
        # warned about a setting they already fixed.
        if notice_file.exists():
            notice_file.unlink()
    return paths.llama_swap_config_path(), paths.litellm_config_path()


def consume_notices() -> list[str]:
    """Return any notices from the most-recent write_all and delete the file.

    One-shot: callers print them then they're gone. If write_all is called
    again with the same notices, they'll come back; if the user fixes the
    underlying setting, the notices file isn't recreated.
    """
    p = paths.notices_path()
    if not p.exists():
        return []
    try:
        text = p.read_text()
    except OSError:
        return []
    with contextlib.suppress(OSError):
        p.unlink()
    return [line for line in text.splitlines() if line.strip()]
