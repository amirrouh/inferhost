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
    """Where llama-server lives — the managed binary, or the user's own build.

    ``INFERHOST_LLAMA_SERVER_PATH`` is the escape hatch for binaries inferhost
    can't fetch: upstream publishes no Linux CUDA build, so an NVIDIA box that
    wants CUDA instead of Vulkan has to self-compile. Honouring it here (rather
    than only inside install_llama_server) is what makes the setting work on
    its own — the installer is skipped entirely in custom-binary mode, so a
    path resolved only there would never reach the generated llama-swap config.

    Point it at a statically linked build (``-DBUILD_SHARED_LIBS=OFF``). The
    generated config pins LD_LIBRARY_PATH to bin_dir(), which holds the managed
    backend's libggml*.so — a dynamic custom build would load those instead of
    its own.
    """
    custom = settings().llama_server_path.strip()
    if custom:
        return Path(custom).expanduser()
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


def pinwatch_pid_file() -> Path:
    return run_dir() / "inferhost-pinwatch.pid"


def swap_log_path() -> Path:
    return logs_dir() / "llama-swap.log"


def gateway_log_path() -> Path:
    return logs_dir() / "litellm.log"


def tts_log_path() -> Path:
    return logs_dir() / "inferhost-tts.log"


def pinwatch_log_path() -> Path:
    return logs_dir() / "inferhost-pinwatch.log"


def model_log_path(name: str) -> Path:
    return logs_dir() / f"{name}.log"


def notices_path() -> Path:
    return data_dir() / "notices.txt"


def ensure_dirs() -> None:
    for d in (bin_dir(), sd_bin_dir(), models_dir(), logs_dir(), run_dir(), config_dir()):
        d.mkdir(parents=True, exist_ok=True)
