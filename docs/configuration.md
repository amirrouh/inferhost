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
INFERHOST_SWAP_PORT=9090        # internal/loopback only in v0.5+
INFERHOST_GATEWAY_PORT=9001     # user-facing LiteLLM endpoint

# TurboQuant KV cache compression (default: turbo3_0, ~4.9x compression)
INFERHOST_KV_QUANT=turbo3_0

# Custom llama-server binary (Vulkan / ROCm / local source builds)
# INFERHOST_LLAMA_SERVER_PATH=/usr/local/bin/llama-server

# Where binaries, logs, and configs live
INFERHOST_DATA_DIR=~/.local/share/inferhost
INFERHOST_CONFIG_DIR=~/.config/inferhost
INFERHOST_HF_CACHE=~/.cache/huggingface

# Inference defaults
INFERHOST_GPU_LAYERS=99          # offload everything to GPU
INFERHOST_DEFAULT_CTX=8192
INFERHOST_FLASH_ATTENTION=on
INFERHOST_PARALLEL_SLOTS=1       # --parallel; 1 = serial requests per model

# Reasoning / "thinking" mode for capable models
INFERHOST_REASONING=auto         # auto | on | off
INFERHOST_REASONING_BUDGET=-1    # token cap on thinking; -1 = unlimited, 0 = none

# Pin specific upstream releases (default: latest)
INFERHOST_LLAMACPP_VERSION=latest
INFERHOST_LLAMASWAP_VERSION=latest

# Force a GPU backend (default: auto-detect)
# INFERHOST_LLAMACPP_BACKEND=cuda

# Stacked speculative decoding (only applied to MTP-capable models).
# Set any value to 0 to disable that lane.
INFERHOST_SPEC_DRAFT_N_MAX=2          # MTP draft tokens per step
INFERHOST_SPEC_NGRAM_MOD_N_MATCH=24   # min matching length before ngram drafts
INFERHOST_SPEC_NGRAM_MOD_N_MIN=48     # min context window to search back through
INFERHOST_SPEC_NGRAM_MOD_N_MAX=64     # max ngram draft tokens on a strong match
```

## Full reference

| Variable | Default | What it does |
|---|---|---|
| `INFERHOST_SWAP_PORT` | `9090` | llama-swap listen port. **Internal/loopback-only in v0.5+** — llama-swap binds `127.0.0.1` and is not reachable from the network. Use the LiteLLM gateway port for external access. |
| `INFERHOST_GATEWAY_PORT` | `9001` | LiteLLM gateway port — the single user-facing OpenAI-compatible endpoint. |
| `INFERHOST_KV_QUANT` | `turbo3_0` | TurboQuant KV cache compression level (see table below). |
| `INFERHOST_LLAMA_SERVER_PATH` | _(auto)_ | Absolute path to a custom `llama-server` binary. Use this to run inferhost with a Vulkan, ROCm, or locally compiled binary instead of a prebuilt asset. |
| `INFERHOST_DATA_DIR` | `~/.local/share/inferhost` | Where downloaded binaries, logs, and PID files live. |
| `INFERHOST_CONFIG_DIR` | `~/.config/inferhost` | Where the generated `llama-swap.yaml` and the model registry live. |
| `INFERHOST_HF_CACHE` | `~/.cache/huggingface` | Hugging Face model cache root. |
| `INFERHOST_GPU_LAYERS` | `99` | The `-ngl` flag passed to llama-server (number of layers offloaded to GPU). `99` ≈ "everything that fits". |
| `INFERHOST_DEFAULT_CTX` | `8192` | Default context length for newly added models. |
| `INFERHOST_FLASH_ATTENTION` | `on` | Pass `-fa` to llama-server. Set to `off` if your GPU doesn't support it. |
| `INFERHOST_PARALLEL_SLOTS` | `1` | Pass `--parallel <n>` to llama-server. Each slot can handle one in-flight request on the same model. Keep at `1` unless you actually need concurrency. |
| `INFERHOST_REASONING` | `auto` | `--reasoning` flag for thinking-capable models (DeepSeek, Qwen3-Thinking, GPT-OSS, ...). `auto` lets the model decide, `on` forces thinking, `off` suppresses it. |
| `INFERHOST_REASONING_BUDGET` | `-1` | `--reasoning-budget` — token cap on thinking. `-1` = unlimited, `0` = none, positive = hard cut-off. |
| `INFERHOST_LLAMACPP_BACKEND` | _auto_ | Force the backend: `vulkan`, `cuda`, `rocm`, `sycl`, `openvino`, or `cpu`. Only applies when `INFERHOST_LLAMA_SERVER_PATH` is not set. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific llama.cpp release tag. |
| `INFERHOST_LLAMASWAP_VERSION` | `latest` | Pin a specific llama-swap release tag. |
| `INFERHOST_SPEC_DRAFT_N_MAX` | `2` | MTP draft tokens per step (`--spec-draft-n-max`). Only applied to models with `mtp` in the filename. Set to `0` to disable the MTP lane. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MATCH` | `24` | Min matching sequence length before ngram-mod drafts (`--spec-ngram-mod-n-match`). |
| `INFERHOST_SPEC_NGRAM_MOD_N_MIN` | `48` | Min context window ngram-mod searches back through (`--spec-ngram-mod-n-min`). |
| `INFERHOST_SPEC_NGRAM_MOD_N_MAX` | `64` | Max draft tokens ngram-mod proposes on a strong match (`--spec-ngram-mod-n-max`). Set to `0` to disable the ngram-mod lane. |

## TurboQuant KV cache compression (`INFERHOST_KV_QUANT`)

inferhost uses a custom `llama-server` build from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) that adds sub-byte KV cache quantization. This frees significant VRAM for larger context windows without recompiling anything.

| Value | Approximate compression | Notes |
|---|---|---|
| `off` | 1× (no compression) | Uses llama.cpp's default `f16` KV cache. |
| `turbo2_0` | ~2.5× | Light compression, near-lossless. |
| `turbo3_0` | ~4.9× | **Default.** Good balance of VRAM savings and quality. |
| `turbo4_0` | ~8× | Aggressive; may affect quality on very long contexts. |

To disable compression entirely:

```env
INFERHOST_KV_QUANT=off
```

## `INFERHOST_LLAMA_SERVER_PATH` — escape hatch for Vulkan / ROCm / custom builds

If the three prebuilt targets (Linux CUDA, Linux CPU, macOS Metal) don't match your hardware, you can point inferhost at any compatible `llama-server` binary:

```bash
# Build your own (e.g. ROCm), then:
export INFERHOST_LLAMA_SERVER_PATH=/home/user/llama.cpp/build/bin/llama-server
inferhost
```

When this variable is set, inferhost skips the binary download step entirely and uses your path instead.

## How auto-detection works

If you don't set `INFERHOST_LLAMACPP_BACKEND` and don't set `INFERHOST_LLAMA_SERVER_PATH`, inferhost runs a small probe at install time:

1. **NVIDIA?** Use the CUDA prebuilt asset.
2. **Apple Silicon?** Use the macOS arm64 Metal prebuilt asset.
3. **No GPU / fallback?** Use the CPU prebuilt asset.

For Vulkan, ROCm, SYCL, or OpenVINO, use `INFERHOST_LLAMA_SERVER_PATH` to supply your own binary.

## Changing settings

Any change to a `.env` value or env var takes effect the next time `inferhost` (or `./run.sh start`) launches the TUI / daemon. After changing `INFERHOST_GATEWAY_PORT`, press **`r`** in the TUI (restart) to rebind the daemon.

[Continue to Troubleshooting →](troubleshooting.md)
