# inferhost

📖 **Full documentation:** <https://amirrouh.github.io/inferhost/>

Run any Hugging Face GGUF model on your own machine. `inferhost` is a small Python framework that wraps **llama.cpp** and **llama-swap** behind a single **LiteLLM gateway**, exposing one OpenAI-compatible endpoint at `http://<host>:9001/v1`.

One binary, two modes:

| Command | What it does |
|---|---|
| `inferhost` | Interactive TUI dashboard — add models, pin, watch logs. |
| `inferhost start \| stop \| restart \| status` | Headless control of the same daemons. No terminal required. |

Key features:
- **Single endpoint, always on:** LiteLLM is bundled (no extra required) and auto-starts on `:9001`.
- **TurboQuant asymmetric KV cache compression:** K=`q8_0`, V=`turbo3` by default (the TurboQuant authors' recommended pairing — "V is free, K is everything"). ~3-4× total KV reduction via a custom `llama-server` built from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant).
- **Pin = load now, with VRAM guard:** Pressing `P` immediately loads the model into VRAM. If it won't fit, inferhost warns you and asks you to unpin something else first.
- **Prebuilt binaries, nothing to compile:** inferhost ships `llama-server` as prebuilt assets for Linux x86_64 CUDA, Linux x86_64 CPU, and macOS arm64 Metal.

![inferhost TUI dashboard](https://raw.githubusercontent.com/amirrouh/inferhost/master/docs/assets/screenshot.png)

```bash
uv tool install inferhost
inferhost                  # TUI: add a model, press 's' to start
# … or, once a model is registered, run everything headlessly:
inferhost start            # spawn llama-swap + LiteLLM as background daemons
inferhost status
inferhost stop
```

That's it. The first launch downloads the runtime binaries (llama-server + llama-swap) for you with a progress bar; then the dashboard opens and you can add, start, stop, and inspect models from the keyboard. On unattended hosts, the same `inferhost start/stop/...` subcommands run everything without a terminal.

## What it does

- One-key serving of any GGUF model published on Hugging Face.
- Automatic quantization selection based on available VRAM (`Q6 → Q5 → Q4 → IQ4` fallback).
- OpenAI-compatible API out of the box, including **tool calling** and **vision**
  for any GGUF that ships an `mmproj-*.gguf` (auto-downloaded alongside the main file).
- **Stacked speculative decoding** for MTP-capable models — combines llama.cpp's
  `--spec-type draft-mtp` with `--spec-type ngram-mod` so MTP handles novel tokens
  while ngram-mod dominates on repeated patterns (code, function names, etc.).
- Multi-model support via llama-swap, which lazy-loads model backends on demand.
- Auto-detected hardware: NVIDIA CUDA, CPU, or Apple Silicon Metal (prebuilt assets);
  for Vulkan/ROCm, point `INFERHOST_LLAMA_SERVER_PATH` at your own binary.
- Live download progress for both runtime binaries and Hugging Face model files.
- **One binary, two modes** — `inferhost` opens the TUI; `inferhost
  start/stop/restart/status` controls the same daemons headlessly.
  Drop-in for servers, cron jobs, and anything without a TTY.
- **Full control from the TUI** — change ports, edit context size and GPU layers,
  watch status of every daemon. No editor, no YAML, no extra commands.
- All defaults still overridable through environment variables or a `.env` file —
  the TUI just writes another `.env` file at `~/.config/inferhost/inferhost.env`
  so your changes survive restarts.

## Installation

Requirements: Python 3.11+, Linux or macOS. NVIDIA CUDA, Linux CPU, or Apple Silicon Metal are the supported prebuilt targets.

`inferhost` is a CLI app, not a library — install it **globally** with `uv tool` (or `pipx`), not into a project's dependencies.

```bash
# Recommended — global, isolated, on your PATH
uv tool install inferhost

# Alternatives
pipx install inferhost
pip install inferhost            # only inside an existing venv
```

> **Note:** In v0.4 and earlier, LiteLLM was an optional `[gateway]` extra (`inferhost[gateway]`). From v0.5 it is bundled — a plain `uv tool install inferhost` is all you need. The `[gateway]` extra still exists as an empty alias for one release to avoid breaking existing install scripts.

> ⚠️ **Don't use `uv add inferhost`.** `uv add` pins it as a project dependency, so you can only run it via `uv run inferhost` inside that one project directory. Use `uv tool install` so `inferhost` is a normal command on your PATH.

### Upgrade

```bash
uv tool upgrade inferhost                # if installed with `uv tool`
pipx upgrade inferhost                   # if installed with pipx
pip install -U inferhost                 # if installed with pip
```

To pin a specific version:

```bash
uv tool install --force 'inferhost==0.5.0'
```

### Uninstall

```bash
uv tool uninstall inferhost              # if installed with `uv tool`
pipx uninstall inferhost                 # if installed with pipx
pip uninstall inferhost                  # if installed with pip
```

Inferhost stores runtime binaries, logs, and the model registry outside the Python install. To wipe **everything** (binaries, llama-server logs, model registry — but **not** downloaded GGUFs, which live in the Hugging Face cache):

```bash
rm -rf ~/.local/share/inferhost ~/.config/inferhost
```

To also drop downloaded models: `rm -rf ~/.cache/huggingface/hub/models--*`.

## Usage

There is exactly one command:

```bash
inferhost
```

This opens the TUI. On first launch it downloads `llama-server` and `llama-swap` with a progress bar. Afterward you land on the dashboard.

### Keys

| Key | Action |
|---|---|
| `a` | Add a Hugging Face model (downloads the GGUF + any `mmproj-*.gguf` for vision) |
| `n` | Rename the highlighted model's public alias (regenerates llama-swap + LiteLLM configs) |
| `c` | Configure the highlighted model: per-model context window (`-c`) |
| `P` | Toggle **pin** on the highlighted model — pins load the model into VRAM immediately; unpinning unloads it. inferhost checks VRAM first and warns if it won't fit. |
| `d` / `Delete` | Remove the highlighted model from the registry |
| `s` | Start llama-swap |
| `x` | Stop llama-swap |
| `r` | Restart llama-swap |
| `p` | Open the Settings panel (ports, context, GPU layers, flash attention) |
| `R` | Refresh |
| `q` | Quit |

The top of the dashboard shows two live status rows: a **GPU bar** (per-card
VRAM bar, used / total, utilization — via `nvidia-smi`, hidden on non-NVIDIA
boxes) and a **status bar** with the daemon dots, ports, the selected model's
active `ctx`, and which model llama-swap currently has resident in VRAM
(`loaded: <name>`). Both refresh every two seconds.

### Adding a model

Press `a`, type a Hugging Face repo id (e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`), and press Enter. The TUI lists the available GGUF files, marks the recommended quant for your hardware, and shows a live progress bar while it downloads. The model is registered against llama-swap and ready to serve.

### Pinning models (load into VRAM immediately)

Press **`P`** on a highlighted model to pin it. Pinning loads the model into VRAM right away — it does not wait for the next request. inferhost checks available VRAM before pinning; if the model won't fit, a modal appears asking you to unpin something else first. Press **`P`** again on a pinned model to unpin and unload it. The sidebar marks pinned models with a `★`.

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

The single OpenAI-compatible endpoint is the **LiteLLM gateway** on port `9001`:

```bash
curl http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4-k-m",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Use the model `name` column from the dashboard as the `model` field.

## Configuration

Every setting is overridable through environment variables or a `.env` file in the working directory. Copy `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `INFERHOST_SWAP_PORT` | `9090` | llama-swap listen port (internal, loopback-only in v0.5+). |
| `INFERHOST_GATEWAY_PORT` | `9001` | LiteLLM gateway port — the user-facing OpenAI endpoint. |
| `INFERHOST_KV_QUANT_K` | `q8_0` | K cache type (`-ctk`). Keep at `q8_0` or `f16`; turbo on K is discouraged. |
| `INFERHOST_KV_QUANT_V` | `turbo3` | V cache type (`-ctv`). Recommended: `turbo4` (light) / `turbo3` (default) / `turbo2` (heavy). |
| `INFERHOST_LLAMA_SERVER_PATH` | _(auto)_ | Path to a custom `llama-server` binary (Vulkan/ROCm/local builds). |
| `INFERHOST_DATA_DIR` | `~/.local/share/inferhost` | Binaries, logs, and PID files. |
| `INFERHOST_CONFIG_DIR` | `~/.config/inferhost` | Model registry and generated YAML. |
| `INFERHOST_HF_CACHE` | `~/.cache/huggingface` | Hugging Face model cache. |
| `INFERHOST_GPU_LAYERS` | `99` | `-ngl` value passed to llama-server. |
| `INFERHOST_DEFAULT_CTX` | `8192` | Default context length for new models. |
| `INFERHOST_FLASH_ATTENTION` | `on` | `-fa` flag for llama-server. |
| `INFERHOST_PARALLEL_SLOTS` | `1` | `--parallel` flag — concurrent request slots per llama-server instance. `1` = serial. |
| `INFERHOST_REASONING` | `auto` | `--reasoning` flag — thinking mode for capable models. `on`, `off`, or `auto`. |
| `INFERHOST_REASONING_BUDGET` | `-1` | `--reasoning-budget` flag — token cap on thinking. `-1` = unlimited, `0` = none. |
| `INFERHOST_LLAMACPP_BACKEND` | auto | Force a backend: `vulkan`, `cuda`, `rocm`, `sycl`, `openvino`, or `cpu`. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific llama.cpp release tag. |
| `INFERHOST_LLAMASWAP_VERSION` | `latest` | Pin a specific llama-swap release tag. |
| `INFERHOST_SPEC_DRAFT_N_MAX` | `2` | MTP draft tokens per step (only used on MTP-capable models). Set to `0` to disable the MTP lane. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MATCH` | `24` | Minimum matching sequence length before ngram-mod drafts. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MIN` | `48` | Minimum context window ngram-mod searches back through. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MAX` | `64` | Max draft tokens ngram-mod proposes on a strong match. |

## Architecture

```
   Client                inferhost
   ------                ---------
   Your app  --HTTP-->   LiteLLM gateway        llama-swap (loopback)   llama-server
                         :9001 (public)  --->   127.0.0.1:9090   --->   (llama.cpp)
```

- **llama.cpp** runs the inference via a prebuilt TurboQuant-enabled `llama-server` (Linux x86_64 CUDA, Linux x86_64 CPU, macOS arm64 Metal).
- **llama-swap** sits in front of multiple llama-server instances and lazy-loads them on demand. It binds loopback (127.0.0.1) only.
- **LiteLLM** is the single user-facing gateway — always on, always bundled, serving `:9001`.

> **Troubleshooting:** If you try `curl http://<lan-ip>:9090/...` and it fails, that is expected — llama-swap is loopback-only by design in v0.5+. Use the LiteLLM port `:9001` instead.

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
