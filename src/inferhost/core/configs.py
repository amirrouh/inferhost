"""Render llama-swap.yaml and litellm.yaml from the model registry."""
from __future__ import annotations

import shlex
from pathlib import Path

import yaml

from inferhost.core import paths
from inferhost.core.registry import Model, Registry
from inferhost.settings import settings


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
        "--log-disable",
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
