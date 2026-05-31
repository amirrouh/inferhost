"""Filesystem paths used by inferhost runtime artifacts."""
from __future__ import annotations

import os
from pathlib import Path

from inferhost.settings import settings


def data_dir() -> Path:
    return settings().data_dir


def config_dir() -> Path:
    return settings().config_dir


def bin_dir() -> Path:
    return data_dir() / "bin"


def models_dir() -> Path:
    return data_dir() / "models"


def logs_dir() -> Path:
    return data_dir() / "logs"


def run_dir() -> Path:
    return data_dir() / "run"


def hf_cache() -> Path:
    return settings().hf_cache


def llama_server_path() -> Path:
    return bin_dir() / "llama-server"


def llama_swap_path() -> Path:
    return bin_dir() / "llama-swap"


def llama_tts_path() -> Path:
    # Standalone one-shot TTS tool from the llama.cpp release. Used by the
    # inferhost-tts daemon to synthesize OuteTTS+vocoder speech per request.
    return bin_dir() / "llama-tts"


def sd_bin_dir() -> Path:
    # stable-diffusion.cpp binaries live in their OWN subdir, NOT bin/. The
    # llama.cpp reinstall purges every lib*.so in bin/ (ABI hygiene) and would
    # otherwise wipe libstable-diffusion.so. Isolating them keeps both stacks
    # independently re-installable.
    return bin_dir() / "sd"


def sd_server_path() -> Path:
    return sd_bin_dir() / "sd-server"


def registry_path() -> Path:
    return config_dir() / "models.toml"


def llama_swap_config_path() -> Path:
    return config_dir() / "llama-swap.yaml"


def litellm_config_path() -> Path:
    return config_dir() / "litellm.yaml"


def swap_pid_file() -> Path:
    return run_dir() / "llama-swap.pid"


def gateway_pid_file() -> Path:
    return run_dir() / "litellm.pid"


def tts_pid_file() -> Path:
    return run_dir() / "inferhost-tts.pid"


def swap_log_path() -> Path:
    return logs_dir() / "llama-swap.log"


def gateway_log_path() -> Path:
    return logs_dir() / "litellm.log"


def tts_log_path() -> Path:
    return logs_dir() / "inferhost-tts.log"


def model_log_path(name: str) -> Path:
    return logs_dir() / f"{name}.log"


def notices_path() -> Path:
    return data_dir() / "notices.txt"


def hermes_home() -> Path:
    """Resolve Hermes' home dir the same way Hermes itself does (HERMES_HOME or ~/.hermes)."""
    return Path(os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()


def hermes_context_cache_path() -> Path:
    return hermes_home() / "context_length_cache.yaml"


def ensure_dirs() -> None:
    for d in (bin_dir(), sd_bin_dir(), models_dir(), logs_dir(), run_dir(), config_dir()):
        d.mkdir(parents=True, exist_ok=True)
