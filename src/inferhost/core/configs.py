"""Render llama-swap.yaml and litellm.yaml from the model registry."""
from __future__ import annotations

import shlex
from pathlib import Path

import yaml

from inferhost.core import paths
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


def _llama_server_cmd(m: Model) -> str:
    s = settings()
    bin_path = paths.llama_server_path()
    parts = [
        "env",
        f"LD_LIBRARY_PATH={paths.bin_dir()}",
        str(bin_path),
        "--model", m.local_path or str(paths.models_dir() / m.filename),
        "--host", "127.0.0.1",
        "--port", str(m.port),
        "-ngl", str(s.gpu_layers),
        "-c", str(m.ctx),
        "-fa", s.flash_attention,
        "--parallel", str(max(1, s.parallel_slots)),
        # Reasoning: model-level override wins, otherwise fall back to global.
        "--reasoning", m.reasoning if m.reasoning else s.reasoning,
        "--reasoning-budget",
        str(m.reasoning_budget if m.reasoning_budget != -2 else s.reasoning_budget),
        # Intentionally NOT passing --log-disable: llama-server's stderr is the
        # only place that prints the actual reason for an abort (GGML_ASSERT,
        # CUDA OOM, TurboQuant edge cases). llama-swap captures the child
        # stderr into its own log, so silencing it means crashes are
        # indistinguishable in postmortem. Verbosity cost is negligible.
    ]
    # Asymmetric KV cache quantization. Per the TurboQuant authors:
    # "V tolerates aggressive compression, K does not." Default: K=q8_0, V=turbo3.
    # TurboQuant adds turbo2/turbo3/turbo4 as new value choices for the existing
    # -ctk / -ctv flags (it is NOT a separate --kv-quant flag).
    kv_quant_k = getattr(s, "kv_quant_k", "q8_0")
    kv_quant_v = getattr(s, "kv_quant_v", "turbo3")
    if kv_quant_k and kv_quant_k != "off":
        parts += ["-ctk", kv_quant_k]
    if kv_quant_v and kv_quant_v != "off":
        parts += ["-ctv", kv_quant_v]
    if m.mmproj_path:
        # Vision (multimodal projector). llama-server emits image-tokens via OpenAI
        # vision content blocks once -mm is attached.
        parts += ["-mm", m.mmproj_path]
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
    # Capture llama-server's stderr to a per-model file. llama-swap discards
    # the child process's stderr entirely (it does not forward to its own
    # log), so without this redirect the actual reason for a SIGABRT or
    # CUDA OOM is lost — we only see "ExitError signal: aborted". The yaml
    # `cmd:` value is exec'd via shell, so a trailing `2>>file` works.
    argv = " ".join(shlex.quote(p) for p in parts)
    err_log = paths.logs_dir() / f"{m.name}.err.log"
    return f"{argv} 2>>{shlex.quote(str(err_log))}"


def render_llama_swap(reg: Registry) -> dict:
    models_block: dict[str, dict] = {}
    for m in reg.models:
        models_block[m.name] = {
            "cmd": _llama_server_cmd(m),
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
                # Expose context window so OpenAI-wire clients (Hermes Agent,
                # litellm callers) can auto-detect model context_length.
                "model_info": {
                    "max_tokens": m.ctx,
                    "max_input_tokens": m.ctx,
                    "max_output_tokens": m.ctx,
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
    swap_cfg = render_llama_swap(reg)
    litellm_cfg = render_litellm(reg)
    _dump_yaml(swap_cfg, paths.llama_swap_config_path())
    _dump_yaml(litellm_cfg, paths.litellm_config_path())
    return paths.llama_swap_config_path(), paths.litellm_config_path()
