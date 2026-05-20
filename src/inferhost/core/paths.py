"""Filesystem paths used by inferhost runtime artifacts."""
from __future__ import annotations

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


def swap_log_path() -> Path:
    return logs_dir() / "llama-swap.log"


def gateway_log_path() -> Path:
    return logs_dir() / "litellm.log"


def model_log_path(name: str) -> Path:
    return logs_dir() / f"{name}.log"


def ensure_dirs() -> None:
    for d in (bin_dir(), models_dir(), logs_dir(), run_dir(), config_dir()):
        d.mkdir(parents=True, exist_ok=True)
