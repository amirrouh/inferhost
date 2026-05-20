---
layout: default
title: Configuration
---

[← Back to overview](index.md)

# Configuration

inferhost reads every setting from environment variables, or from a `.env` file in the directory you run it from. No YAML, no JSON, no config CLI.

## .env example

Drop a `.env` file next to wherever you launch `inferhost` (or in your project root):

```env
# Ports
INFERHOST_SWAP_PORT=9090
INFERHOST_GATEWAY_PORT=9001

# Where binaries, logs, and configs live
INFERHOST_DATA_DIR=~/.local/share/inferhost
INFERHOST_CONFIG_DIR=~/.config/inferhost
INFERHOST_HF_CACHE=~/.cache/huggingface

# Inference defaults
INFERHOST_GPU_LAYERS=99          # offload everything to GPU
INFERHOST_DEFAULT_CTX=8192
INFERHOST_FLASH_ATTENTION=on

# Pin specific upstream releases (default: latest)
INFERHOST_LLAMACPP_VERSION=latest
INFERHOST_LLAMASWAP_VERSION=latest

# Force a GPU backend (default: auto-detect)
# INFERHOST_LLAMACPP_BACKEND=cuda
```

## Full reference

| Variable | Default | What it does |
|---|---|---|
| `INFERHOST_SWAP_PORT` | `9090` | The user-facing OpenAI-compatible endpoint port (llama-swap). |
| `INFERHOST_GATEWAY_PORT` | `9001` | LiteLLM gateway port (when the gateway extra is installed). |
| `INFERHOST_DATA_DIR` | `~/.local/share/inferhost` | Where downloaded binaries, logs, and PID files live. |
| `INFERHOST_CONFIG_DIR` | `~/.config/inferhost` | Where the generated `llama-swap.yaml` and the model registry live. |
| `INFERHOST_HF_CACHE` | `~/.cache/huggingface` | Hugging Face model cache root. |
| `INFERHOST_GPU_LAYERS` | `99` | The `-ngl` flag passed to llama-server (number of layers offloaded to GPU). `99` ≈ "everything that fits". |
| `INFERHOST_DEFAULT_CTX` | `8192` | Default context length for newly added models. |
| `INFERHOST_FLASH_ATTENTION` | `on` | Pass `-fa` to llama-server. Set to `off` if your GPU doesn't support it. |
| `INFERHOST_LLAMACPP_BACKEND` | _auto_ | Force the backend: `vulkan`, `cuda`, `rocm`, `sycl`, `openvino`, or `cpu`. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific [llama.cpp release tag](https://github.com/ggml-org/llama.cpp/releases). |
| `INFERHOST_LLAMASWAP_VERSION` | `latest` | Pin a specific [llama-swap release tag](https://github.com/mostlygeek/llama-swap/releases). |

## How auto-detection works

If you don't set `INFERHOST_LLAMACPP_BACKEND`, inferhost runs a small probe at install time:

1. **NVIDIA?** Prefer CUDA, fall back to Vulkan.
2. **AMD?** Prefer ROCm, fall back to Vulkan.
3. **Intel discrete?** Prefer SYCL / OpenVINO.
4. **Apple Silicon?** Use the universal macOS Metal build.
5. **No GPU?** CPU.

If a preferred backend has no prebuilt asset for your platform, inferhost falls back to the next best. Run with the env override above if you want to pin it.

## Changing settings

Any change to a `.env` value or env var takes effect the next time `inferhost` (or `./run.sh start`) launches the TUI / daemon. After changing `INFERHOST_SWAP_PORT`, press **`r`** in the TUI (restart) to rebind the daemon.

[Continue to Troubleshooting →](troubleshooting.md)
