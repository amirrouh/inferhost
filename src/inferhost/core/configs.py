"""Render llama-swap.yaml and litellm.yaml from the model registry."""
from __future__ import annotations

import contextlib
import shlex
from pathlib import Path

import yaml

from inferhost.core import paths
from inferhost.core.llama_caps import pick_kv_quant, supported_cache_types
from inferhost.core.registry import Model, Registry
from inferhost.settings import settings


def _is_mtp_capable(m: Model) -> bool:
    """A model is MTP-capable if 'mtp' appears in its filename or registry name.

    Convention: GGUFs that ship NextN / MTP heads (e.g. Qwen3.6 MTP variants) carry
    the 'mtp' tag in their filename. The user can also force-enable by renaming the
    registry entry to include 'mtp'.
    """
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
    parts = [
        "env",
        f"LD_LIBRARY_PATH={paths.bin_dir()}",
        str(bin_path),
        "--model", m.local_path or str(paths.models_dir() / m.filename),
        "--host", "127.0.0.1",
        "--port", str(m.port),
        "-ngl", str(eff_gpu_layers),
        "-c", str(m.ctx),
        "-fa", eff_fa,
        "--parallel", str(max(1, eff_parallel)),
        # Use the model's own jinja chat template from the GGUF metadata.
        # llama-server's legacy built-in templates strip tool-call blocks and
        # OpenAI vision content parts, so without --jinja a tool-trained or
        # vision-trained GGUF silently behaves like a plain text-only model.
        # Newer llama-server defaults this on, older builds default off — pass
        # it explicitly so behavior is consistent across the prebuilt binaries.
        "--jinja",
        # Reasoning: model-level override wins, otherwise fall back to global.
        "--reasoning", m.reasoning if m.reasoning else s.reasoning,
        "--reasoning-budget",
        str(m.reasoning_budget if m.reasoning_budget != -2 else s.reasoning_budget),
        # Intentionally NOT passing --log-disable: llama-server's stderr is the
        # only place that prints the actual reason for an abort (GGML_ASSERT,
        # CUDA OOM, etc.). llama-swap captures the child stderr into its own
        # log, so silencing it means crashes are indistinguishable in
        # postmortem. Verbosity cost is negligible.
    ]
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
    if _is_mtp_capable(m):
        # Stack two speculative-decode lanes (llama.cpp accepts multiple --spec-type):
        #   1. draft-mtp uses the MTP heads baked into the GGUF
        #   2. ngram-mod uses pattern lookup over already-generated text
        # MTP handles novel generation; ngram-mod dominates on repeated patterns
        # (code, function names, repeated constructs).
        if s.spec_draft_n_max > 0:
            parts += [
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(s.spec_draft_n_max),
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
    for m in reg.models:
        models_block[m.name] = {
            "cmd": _llama_server_cmd(m, notices=notices),
            "proxy": f"http://127.0.0.1:{m.port}",
            "ttl": 600,
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
    pinned = [m.name for m in reg.models if m.pin]
    swappable = [m.name for m in reg.models if not m.pin]
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
                    "max_tokens": m.ctx,
                    "max_input_tokens": m.ctx,
                    "max_output_tokens": m.ctx,
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
