# inferhost

📖 **Full documentation:** <https://amirrouh.github.io/inferhost/>

Run any Hugging Face GGUF model on your own machine — **TUI only**. `inferhost` is a small Python framework that wraps **llama.cpp**, **llama-swap**, and (optionally) **LiteLLM** behind a single Textual TUI. Point it at a Hugging Face repository and it returns an OpenAI-compatible endpoint.

![inferhost TUI dashboard](https://raw.githubusercontent.com/amirrouh/inferhost/master/docs/assets/screenshot.png)

```bash
uv tool install inferhost
inferhost
```

That's it. The first launch downloads the runtime binaries (llama-server + llama-swap) for you with a progress bar; then the dashboard opens and you can add, start, stop, and inspect models from the keyboard.

## What it does

- One-key serving of any GGUF model published on Hugging Face.
- Automatic quantization selection based on available VRAM (`Q6 → Q5 → Q4 → IQ4` fallback).
- OpenAI-compatible API out of the box, including **tool calling** and **vision**
  for any GGUF that ships an `mmproj-*.gguf` (auto-downloaded alongside the main file).
- **Stacked speculative decoding** for MTP-capable models — combines llama.cpp's
  `--spec-type draft-mtp` with `--spec-type ngram-mod` so MTP handles novel tokens
  while ngram-mod dominates on repeated patterns (code, function names, etc.).
- Multi-model support via llama-swap, which lazy-loads model backends on demand.
- Auto-detected hardware: NVIDIA via Vulkan, AMD via ROCm, Intel via SYCL/OpenVINO, or CPU.
- Live download progress for both runtime binaries and Hugging Face model files.
- **Full control from the TUI** — change ports, edit context size and GPU layers,
  set a per-model context window, rename a model's public alias, toggle the LiteLLM
  gateway, watch status of every daemon. No editor, no YAML, no extra commands.
- All defaults still overridable through environment variables or a `.env` file —
  the TUI just writes another `.env` file at `~/.config/inferhost/inferhost.env`
  so your changes survive restarts.

## Installation

Requirements: Python 3.11+, Linux or macOS. NVIDIA, AMD, Intel, or Apple Silicon GPUs are auto-detected; CPU-only is supported.

`inferhost` is a CLI app, not a library — install it **globally** with `uv tool` (or `pipx`), not into a project's dependencies.

```bash
# Recommended — global, isolated, on your PATH
uv tool install inferhost

# With the LiteLLM gateway (unified endpoint + routing + aliases)
uv tool install 'inferhost[gateway]'

# Alternatives
pipx install inferhost
pip install inferhost            # only inside an existing venv
```

> ⚠️ **Don't use `uv add inferhost`.** `uv add` pins it as a project dependency, so you can only run it via `uv run inferhost` inside that one project directory. Use `uv tool install` so `inferhost` is a normal command on your PATH.

### Upgrade

```bash
uv tool upgrade inferhost                # if installed with `uv tool`
pipx upgrade inferhost                   # if installed with pipx
pip install -U inferhost                 # if installed with pip
```

To pin a specific version:

```bash
uv tool install --force 'inferhost==0.4.13'
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
| `c` | Configure the highlighted model: per-model context window (`-c`) and KV cache quant (`-ctk`/`-ctv`) |
| `P` | Toggle **pin** on the highlighted model (pinned models stay co-resident in VRAM) |
| `d` / `Delete` | Remove the highlighted model from the registry |
| `s` | Start llama-swap |
| `x` | Stop llama-swap |
| `r` | Restart llama-swap |
| `g` | Toggle the LiteLLM gateway on/off |
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

### Configuring a model (per-model context + KV cache quant)

The global `default_ctx` (in Settings) applies to newly added models, but you
can override it per-model. Highlight a model and press **`c`** to open the
per-model settings panel, where you can edit:

- **Context window (`-c`)** — tokens per request for this model.
- **KV cache type K / V (`-ctk` / `-ctv`)** — quantization of the KV cache.
  Leave blank for llama.cpp's default (`f16`). `q8_0` is near-lossless and
  roughly halves KV memory; `q4_0` cuts it ~4× but starts to bite on long
  contexts. The smallest near-lossless way to fit a larger ctx into the same
  VRAM is `-ctk q8_0 -ctv q8_0`.

inferhost saves the values to the registry, re-renders `llama-swap.yaml`, and
reloads any running daemon so the new flags take effect immediately.

### Pinning models (load two at once)

By default llama-swap holds **one** model in VRAM at a time and unloads it when
a different model is requested. To keep two (or more) models loaded
simultaneously, highlight each one and press **`P`** to pin it — pinned models
share a llama-swap group with `swap: false`, so they stay co-resident instead
of swapping each other out. The sidebar marks pinned models with a `★`, and
the details panel shows `loading: ★ pinned (co-resident)`. The `c` (Configure)
panel also has a `Pin in VRAM` field for the same toggle. Unpinned models still
swap on demand as before. Make sure pinned models actually fit together in
VRAM — the GPU bar at the top is your guide.

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
| `INFERHOST_PARALLEL_SLOTS` | `1` | `--parallel` flag — concurrent request slots per llama-server instance. `1` = serial. |
| `INFERHOST_REASONING` | `auto` | `--reasoning` flag — thinking mode for capable models. `on`, `off`, or `auto`. |
| `INFERHOST_REASONING_BUDGET` | `-1` | `--reasoning-budget` flag — token cap on thinking. `-1` = unlimited, `0` = none. |
| `INFERHOST_LLAMACPP_BACKEND` | auto | Force a backend: `vulkan`, `cuda`, `rocm`, `sycl`, `openvino`, or `cpu`. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific llama.cpp release tag. |
| `INFERHOST_LLAMASWAP_VERSION` | `latest` | Pin a specific llama-swap release tag. |
| `INFERHOST_SPEC_DRAFT_N_MAX` | `2` | MTP draft tokens per step (only used on MTP-capable models). Set to `0` to disable the MTP lane. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MATCH` | `24` | Minimum matching sequence length before ngram-mod drafts. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MIN` | `48` | Minimum context window ngram-mod searches back through. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MAX` | `64` | Max draft tokens ngram-mod proposes on a strong match. Set to `0` to disable the ngram-mod lane. |

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
