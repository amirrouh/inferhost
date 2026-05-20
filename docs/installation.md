---
layout: default
title: Installation
---

[← Back to overview](index.md)

# Installation

## System requirements

| | |
|---|---|
| **Python** | 3.11, 3.12, or 3.13 |
| **OS** | Linux or macOS |
| **GPU (optional)** | NVIDIA (CUDA / Vulkan), AMD (ROCm), Intel (SYCL / OpenVINO), Apple Silicon (Metal) |
| **RAM** | depends on the model you want to run (a 7B model in Q4 is ~5 GB) |

CPU-only is fully supported — it'll just be slower.

## Install (pip)

```bash
pip install inferhost
```

## Install (uv, recommended)

[uv](https://github.com/astral-sh/uv) is faster and isolates the install:

```bash
uv tool install inferhost
```

## Install with the LiteLLM gateway

The optional gateway adds friendly aliases, routing, and rate limits across many providers:

```bash
pip install 'inferhost[gateway]'
```

## First launch

```bash
inferhost
```

On the very first launch, inferhost downloads two runtime binaries to `~/.local/share/inferhost/bin/`:

- **llama-server** — from the upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) project, in whichever GPU backend matches your hardware.
- **llama-swap** — the lazy-loading proxy from [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap).

You'll see a progress bar for each. After that, the dashboard opens and you're ready to add a model.

## Choosing the GPU backend

inferhost auto-detects the best backend for your hardware. If you want to pin it explicitly, set an environment variable before launching:

```bash
export INFERHOST_LLAMACPP_BACKEND=cuda   # or vulkan, rocm, sycl, openvino, cpu
inferhost
```

See the [Configuration](configuration.md) page for the full list.

## Verify

After the install screen, the dashboard's top bar shows the live endpoint, e.g.:

```
● llama-swap http://localhost:9090/v1
```

The green ● means the daemon is up. Press **`a`** to add your first model.

[Continue to Usage →](usage.md)
