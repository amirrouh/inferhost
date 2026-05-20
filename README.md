# inferhost

Run any Hugging Face GGUF model on your own machine in a single command. inferhost is a small Python framework that wraps **llama.cpp**, **llama-swap**, and (optionally) **LiteLLM** behind one CLI and a Textual TUI. Point it at a Hugging Face repository and it returns an OpenAI-compatible endpoint.

```bash
pip install inferhost
inferhost install
inferhost serve Qwen/Qwen2.5-7B-Instruct-GGUF
# OpenAI-compatible endpoint: http://localhost:9090/v1
```

## Features

- One-command serving of any GGUF model published on Hugging Face.
- Automatic quantization selection based on available VRAM (`Q6 → Q5 → Q4 → IQ4` fallback).
- OpenAI-compatible API out of the box; works with the official SDKs and any compatible client.
- Multi-model support via llama-swap, which lazy-loads model backends on demand.
- Textual TUI for adding, starting, stopping, and tailing logs of registered models.
- Auto-detected hardware: NVIDIA via Vulkan, AMD via ROCm, Intel via SYCL/OpenVINO, or CPU.
- All defaults overridable through environment variables or a `.env` file.

## Installation

Requirements: Python 3.11+, Linux or macOS. NVIDIA, AMD, Intel, or Apple Silicon GPUs are auto-detected; CPU-only is supported.

```bash
# Recommended
uv tool install inferhost

# Or with pip
pip install inferhost

# With the LiteLLM gateway (unified endpoint + routing + aliases)
pip install 'inferhost[gateway]'
```

Then download the runtime binaries (llama-server from llama.cpp, plus llama-swap):

```bash
inferhost install
inferhost doctor
```

`doctor` prints a summary of detected hardware, installed binaries, and configured paths.

## Usage

### One-command serve

```bash
inferhost serve bartowski/SmolLM2-360M-Instruct-GGUF
```

inferhost will:

1. List the GGUF files in the repository.
2. Pick the highest-quality quant that fits in your VRAM.
3. Download it via the standard Hugging Face cache.
4. Register the model and render the llama-swap configuration.
5. Start llama-swap and expose the endpoint.

Test the endpoint with any OpenAI client:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-360m-instruct-f16",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Manage multiple models

```bash
inferhost add meta-llama/Llama-3.2-3B-Instruct-GGUF
inferhost add bartowski/gemma-2-9b-it-GGUF --quant Q5_K_M
inferhost ls
inferhost start
```

llama-swap loads each model on the first request and unloads it after a configurable idle period.

### TUI

```bash
inferhost tui
```

```
inferhost                              llama-swap http://localhost:9090/v1

Models                       Details
---------------------------  -------------------------------------------------
qwen2.5-7b-instruct          name:  qwen2.5-7b-instruct-q4-k-m
llama-3.2-3b-instruct        repo:  Qwen/Qwen2.5-7B-Instruct-GGUF
gemma-2-9b-it                quant: Q4_K_M    size: 4.4 GiB    ctx: 8192
                             port:  9091

                             Logs
                             llm_load_tensors: offloaded 33/33 layers to GPU
                             ...

a=add  s=start  x=stop  r=restart  d=remove  q=quit
```

The add-model modal accepts any Hugging Face repository ID, fetches available GGUF files, and highlights the recommended quantization for your hardware.

## Command reference

| Command | Description |
|---|---|
| `inferhost install` | First-time setup: download llama-server and llama-swap. |
| `inferhost serve <repo>` | Add a model and start serving it. |
| `inferhost add <repo>` | Add a model without starting llama-swap. |
| `inferhost start` | Start llama-swap. |
| `inferhost stop [--all]` | Stop llama-swap (and the gateway with `--all`). |
| `inferhost restart` | Restart llama-swap with the current configuration. |
| `inferhost ls` | List registered models. |
| `inferhost rm <name>` | Remove a model from the registry. |
| `inferhost logs <name> [-f]` | Show or follow logs. |
| `inferhost status` | Show daemon status. |
| `inferhost doctor` | Environment check. |
| `inferhost gateway start\|stop` | Manage the LiteLLM gateway (optional). |
| `inferhost tui` | Launch the dashboard. |

## Configuration

Every setting is overridable through environment variables or a `.env` file in the working directory. Copy `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `INFERHOST_SWAP_PORT` | `9090` | llama-swap listen port (user-facing OpenAI endpoint). |
| `INFERHOST_GATEWAY_PORT` | `9001` | LiteLLM gateway port when enabled. |
| `INFERHOST_DATA_DIR` | `~/.local/share/inferhost` | Binaries, logs, and PID files. |
| `INFERHOST_CONFIG_DIR` | `~/.config/inferhost` | Model registry and generated YAML. |
| `INFERHOST_HF_CACHE` | `~/.cache/huggingface` | Hugging Face model cache. |
| `INFERHOST_GPU_LAYERS` | `99` | `-ngl` value passed to llama-server. |
| `INFERHOST_DEFAULT_CTX` | `8192` | Default context length for new models. |
| `INFERHOST_FLASH_ATTENTION` | `on` | `-fa` flag for llama-server. |
| `INFERHOST_LLAMACPP_BACKEND` | auto | Force a backend: `vulkan`, `cuda`, `rocm`, `sycl`, `openvino`, or `cpu`. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific llama.cpp release tag. |
| `INFERHOST_LLAMASWAP_VERSION` | `latest` | Pin a specific llama-swap release tag. |

## Architecture

```
   Client                inferhost                       Inference
   ------                ---------                       ---------
   Your app  --HTTP-->   llama-swap        spawns/kills  llama-server
                         :9090                           (llama.cpp)
                            ^
                            |
                  (optional) LiteLLM
                         :9001
```

- **llama.cpp** runs the inference (using a prebuilt Vulkan, CUDA, ROCm, SYCL, OpenVINO, or CPU binary, whichever fits the host).
- **llama-swap** sits in front of multiple llama-server instances and lazy-loads them on demand.
- **LiteLLM** (optional) provides a unified gateway with friendly aliases, routing, rate limits, and fallbacks across local and hosted providers.

## Development

```bash
git clone git@github.com:amirrouh/inferhost.git
cd inferhost
./run.sh install            # creates venv, installs in editable mode, fetches binaries
./run.sh test               # runs pytest
./run.sh doctor             # confirms environment
./run.sh serve bartowski/SmolLM2-360M-Instruct-GGUF
```

`run.sh` is a thin wrapper around the `inferhost` CLI that activates the project venv and forwards arguments. Run `./run.sh help` for the full list of dev shortcuts.

## License

Apache 2.0
