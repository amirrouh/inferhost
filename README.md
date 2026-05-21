# inferhost

📖 **Full documentation:** <https://amirrouh.github.io/inferhost/>

Run any Hugging Face GGUF model on your own machine — **TUI only**. `inferhost` is a small Python framework that wraps **llama.cpp**, **llama-swap**, and (optionally) **LiteLLM** behind a single Textual TUI. Point it at a Hugging Face repository and it returns an OpenAI-compatible endpoint.

![inferhost TUI dashboard](https://raw.githubusercontent.com/amirrouh/inferhost/master/docs/assets/screenshot.png)

```bash
pip install inferhost
inferhost
```

That's it. The first launch downloads the runtime binaries (llama-server + llama-swap) for you with a progress bar; then the dashboard opens and you can add, start, stop, and inspect models from the keyboard.

## What it does

- One-key serving of any GGUF model published on Hugging Face.
- Automatic quantization selection based on available VRAM (`Q6 → Q5 → Q4 → IQ4` fallback).
- OpenAI-compatible API out of the box; works with the official SDKs and any compatible client.
- Multi-model support via llama-swap, which lazy-loads model backends on demand.
- Auto-detected hardware: NVIDIA via Vulkan, AMD via ROCm, Intel via SYCL/OpenVINO, or CPU.
- Live download progress for both runtime binaries and Hugging Face model files.
- **Full control from the TUI** — change ports, edit context size and GPU layers,
  rename a model's public alias, toggle the LiteLLM gateway, watch status of every
  daemon. No editor, no YAML, no extra commands.
- All defaults still overridable through environment variables or a `.env` file —
  the TUI just writes another `.env` file at `~/.config/inferhost/inferhost.env`
  so your changes survive restarts.

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

## Usage

There is exactly one command:

```bash
inferhost
```

This opens the TUI. On first launch it downloads `llama-server` and `llama-swap` with a progress bar. Afterward you land on the dashboard.

### Keys

| Key | Action |
|---|---|
| `a` | Add a Hugging Face model (downloads the GGUF with a progress bar) |
| `n` | Rename the highlighted model's public alias (regenerates llama-swap + LiteLLM configs) |
| `d` / `Delete` | Remove the highlighted model from the registry |
| `s` | Start llama-swap |
| `x` | Stop llama-swap |
| `r` | Restart llama-swap |
| `g` | Toggle the LiteLLM gateway on/off |
| `p` | Open the Settings panel (ports, context, GPU layers, flash attention) |
| `R` | Refresh |
| `q` | Quit |

The top of the dashboard always shows the running state of both the `llama-swap`
and the (optional) `litellm` daemon, plus a one-line summary of every setting
currently in effect.

### Adding a model

Press `a`, type a Hugging Face repo id (e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`), and press Enter. The TUI lists the available GGUF files, marks the recommended quant for your hardware, and shows a live progress bar while it downloads. The model is registered against llama-swap and ready to serve.

### Renaming a model

The name shown in the model list is also the value clients send as the OpenAI
`model` field. Press `n` to change it. inferhost rewrites the llama-swap and
LiteLLM configs in one shot and, if llama-swap is running, restarts it so the new
alias is reachable immediately. No need to edit any YAML by hand.

### Changing ports and other settings

Press `p` to open the Settings panel. You can edit `swap_port`, `gateway_port`,
`default_ctx`, `gpu_layers`, and `flash_attention` directly. Saving writes a
managed env file at `~/.config/inferhost/inferhost.env`, so your changes persist
across restarts. Press `r` afterwards to restart llama-swap with the new values.

### Endpoint

The dashboard shows the current OpenAI-compatible endpoint, e.g. `http://localhost:9090/v1`. Use the model `name` column in any OpenAI client:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4-k-m",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

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

The repo ships a `run.sh` wrapper for source-tree work:

```bash
git clone git@github.com:amirrouh/inferhost.git
cd inferhost
./run.sh install     # creates venv, installs in editable mode
./run.sh start       # launches the TUI (downloads binaries on first run)
./run.sh status      # headless status print
./run.sh stop        # stop daemons
./run.sh test        # run pytest
```

Run `./run.sh help` for the full list. End users do not need `run.sh` — they only ever type `inferhost`.

## License

Apache 2.0
