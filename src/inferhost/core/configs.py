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
        "--log-disable",
    ]
    if m.cache_type_k:
        parts += ["-ctk", m.cache_type_k]
    if m.cache_type_v:
        parts += ["-ctv", m.cache_type_v]
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
    return " ".join(shlex.quote(p) for p in parts)


def render_llama_swap(reg: Registry) -> dict:
    models_block: dict[str, dict] = {}
    for m in reg.models:
        models_block[m.name] = {
            "cmd": _llama_server_cmd(m),
            "proxy": f"http://127.0.0.1:{m.port}",
            "ttl": 600,
        }
    return {
        "healthCheckTimeout": 300,
        "logRequests": False,
        "models": models_block,
    }


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
