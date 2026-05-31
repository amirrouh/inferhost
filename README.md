# inferhost

📖 **Full documentation:** <https://amirrouh.github.io/inferhost/>

Run any Hugging Face GGUF model on your own machine — **text/chat, vision, embeddings, text-to-speech, and image generation** — behind one OpenAI-compatible endpoint. `inferhost` is a small Python framework that wraps **llama.cpp**, **stable-diffusion.cpp**, and **llama-swap** behind a single **LiteLLM gateway** at `http://<host>:9001/v1`.

One binary, two modes:

| Command | What it does |
|---|---|
| `inferhost` | Interactive TUI dashboard — add models, pin, watch logs. |
| `inferhost start \| stop \| restart \| status` | Headless control of the same daemons. No terminal required. |

Key features:
- **Multi-modal, one endpoint:** chat/vision (`/v1/chat/completions`), **text-to-speech** (`/v1/audio/speech`, OuteTTS), and **image generation** (`/v1/images/generations`, SDXL / Flux / Z-Image / Qwen-Image) — all on `:9001`, all OpenAI-compatible. The extra engines (`llama-tts`, `sd-server`) are fetched automatically when you add a model that needs them.
- **Single endpoint, always on:** LiteLLM is bundled (no extra required) and auto-starts on `:9001`.
- **KV cache compression on by default:** K=`q8_0`, V=`q8_0` — ~2× compression of the f16 baseline with near-lossless quality. Override per axis from the TUI Settings screen or `.env`.
- **Pin = load now, with VRAM guard:** Pressing `P` immediately loads the model into VRAM. If it won't fit, inferhost warns you and asks you to unpin something else first.
- **Prebuilt binaries from upstream, nothing to compile:** inferhost pulls `llama-server` straight from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) releases (Vulkan / ROCm / SYCL / OpenVINO / CPU on Linux, Metal on macOS arm64). Pin the version or change the backend from the TUI.

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
- **Text-to-speech** via llama.cpp's `llama-tts` — add an OuteTTS model and get
  `/v1/audio/speech` (WAV) on the same gateway. See [Text-to-speech](#text-to-speech-v1audiospeech).
- **Image generation** via [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
  `sd-server` — SDXL/SD1.5 (single-file), and Flux / SD3 / Z-Image / Qwen-Image
  (multi-file, assembled with a paste-repo→pick-from-list component picker). Runs
  under llama-swap so it **swaps VRAM with your LLMs**. See [Image generation](#image-generation-v1imagesgenerations).
- **Stacked speculative decoding** for MTP-capable models — combines llama.cpp's
  `--spec-type draft-mtp` with `--spec-type ngram-mod` so MTP handles novel tokens
  while ngram-mod dominates on repeated patterns (code, function names, etc.). The
  MTP draft depth is tunable per model in **Configure** (`--spec-draft-n-max`).
- **Honest context windows** — the served and advertised window is read against the
  GGUF's native trained context on disk, so what agents see over the API (and in the
  Hermes cache) always matches what llama-server actually loaded.
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
| `c` | Configure the highlighted model: per-model `-c` (context), `-ctk` / `-ctv` (KV cache K/V quant), `-ngl` (GPU layers), `--parallel`, `-fa`, reasoning, reasoning budget, raw extra args, MTP draft tokens (`--spec-draft-n-max`), and pin. The dialog shows the model's **native trained context** read from the GGUF on disk; a `-c` above it is clamped on load and the advertised window matches what's actually served. Blank fields inherit the global Settings value. |
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
`default_ctx`, `gpu_layers`, `flash_attention`, `parallel_slots`, `reasoning`,
`reasoning_budget`, the KV cache quants (`kv_quant_k` / `kv_quant_v` — accepts
any `f16`/`q8_0`/`q5_*`/`q4_*`/`iq4_nl`/`off`), and the llama.cpp version +
backend (`llamacpp_version`, `llamacpp_backend`) directly. Saving writes a
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

### Text-to-speech (`/v1/audio/speech`)

If you add a Hugging Face repo whose GGUF ships a **WavTokenizer / vocoder**
alongside the model (e.g. an OuteTTS repo), inferhost auto-detects it, downloads
it, and serves the model as a **text-to-speech** model on the same gateway:

```bash
curl http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "outetts-0.2-500m-q4-k-m", "input": "Hello from inferhost.", "voice": "default"}' \
  --output speech.wav
```

The OpenAI Python SDK works unchanged
(`client.audio.speech.create(model=..., input=..., voice="default")`). Notes:

- `voice` is **required** when calling through the gateway (the OpenAI spec and
  LiteLLM both mandate it). Any value works — see below; the SDK always sends it.

- TTS runs via llama.cpp's standalone `llama-tts` binary (the only way to
  synthesize OuteTTS), bundled automatically on install/update. It has no
  resident-server mode, so **each request reloads the model** (a few seconds of
  overhead) — fine for occasional/scripted use.
- Output is **WAV**. The `voice` *value* is ignored unless it points at a
  llama-tts speaker file — the model's built-in speaker is used otherwise — but
  the field must still be present (see above).
- TTS models are marked `♪ [tts]` in the dashboard. They don't run under
  llama-swap and can't be pinned — they're served on demand by a small
  `inferhost-tts` daemon that `inferhost start` brings up automatically whenever
  a TTS model is registered.
- **Auto-detect only:** the vocoder must live in the *same* repo as the model.
  Repos that publish the vocoder separately aren't picked up. If you previously
  added an OuteTTS model as a plain chat model, **remove and re-add it** so the
  vocoder is detected and the model is reclassified as TTS.

### Image generation (`/v1/images/generations`)

inferhost bundles [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
`sd-server`, so adding an image model lights up an OpenAI-compatible image
endpoint on the same gateway. Add a model with the **Image generation** kind
selected in the add screen (same "paste repo → pick from a list" flow as LLMs):

```bash
curl http://localhost:9001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "stable-diffusion-v1-5-q4-0", "prompt": "a red apple on a table", "size": "512x512"}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

The OpenAI Python SDK works unchanged (`client.images.generate(model=..., prompt=..., size=...)`).
Notes:

- **Single-file (SD1.5 / SDXL):** paste the repo, pick one `.gguf`/`.safetensors`
  → done. **Multi-file (Flux / SD3):** inferhost auto-detects and downloads the
  companion VAE / CLIP / T5 files **shipped in the same repo**. The `sd-server`
  binary itself is fetched automatically the first time you add an image model.
- **Shares VRAM with your LLMs:** image models run through llama-swap in the
  swappable group, so an image model and a large LLM **evict each other** — only
  one big model is resident at a time. The model lazy-loads on the first request.
- **Parameters:** resolution via `size` per request; `steps`/`cfg`/`sampler` are
  set as per-model defaults (the model's `extra_args` in Configure) or per request
  via sd-server's `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` block in the prompt.
- **Quality:** same model weights as ComfyUI → comparable txt2img, but a subset of
  ComfyUI's ecosystem (no full node graphs) and slower on Vulkan than CUDA-PyTorch.
- **Multi-file models with components in separate repos** (Flux, **Z-Image-Turbo**,
  **Qwen-Image**): add the diffusion model first, then open **Configure** — image
  models get a **component editor** where each slot (VAE, Text encoder, CLIP-L/G,
  T5XXL) is filled with the same *paste repo → pick from list* flow. inferhost
  downloads each picked file and wires it in. Supported encoder families: CLIP +
  T5 (Flux/SD3) and **Qwen/LLM text encoder** (Qwen-Image, Z-Image, via
  `sd-server --llm`).

  Example — **Z-Image-Turbo** (3 files from 3 repos): diffusion
  `leejet/Z-Image-Turbo-GGUF`, VAE `second-state/FLUX.1-schnell-GGUF/ae.safetensors`
  (a non-gated mirror — the official `black-forest-labs/FLUX.1-schnell` is gated),
  text encoder `unsloth/Qwen3-4B-Instruct-2507-GGUF`. Set `--steps 8 --cfg-scale 1`
  in the model's extra args (it's a turbo/few-step model). Verified end-to-end.

- **Image editing (Qwen-Image-Edit, Flux Kontext):** the OpenAI edit endpoint is
  `multipart/form-data`, which the LiteLLM gateway doesn't route by model. Reach
  it directly on llama-swap, which forwards any path for a model:
  `POST http://<host>:9090/upstream/<model>/v1/images/edits` (image + prompt as
  form fields). txt2img still goes through the normal `:9001` gateway.

## Configuration

Every setting is overridable through environment variables or a `.env` file in the working directory. Copy `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `INFERHOST_SWAP_PORT` | `9090` | llama-swap listen port. Defaults to `0.0.0.0` so it's reachable from your LAN / Tailscale — override `INFERHOST_SWAP_HOST=127.0.0.1` to keep loopback-only. |
| `INFERHOST_GATEWAY_PORT` | `9001` | LiteLLM gateway port — the user-facing OpenAI endpoint. |
| `INFERHOST_TTS_PORT` | `9092` | Port for the `inferhost-tts` daemon (serves `/v1/audio/speech` for TTS models). Only runs when a TTS model is registered. Defaults to `0.0.0.0`; override `INFERHOST_TTS_HOST=127.0.0.1` for loopback-only. |
| `INFERHOST_SDCPP_VERSION` | `latest` | Pin a stable-diffusion.cpp release tag (image generation). `latest` pulls the newest; the `sd-server` binary is fetched automatically when you add your first image model. |
| `INFERHOST_SD_STEPS` | `0` | Default diffusion steps baked into the image model's launch (`0` = sd-server's own default). Override per-model via `extra_args`. |
| `INFERHOST_SD_CFG_SCALE` | `0` | Default CFG scale for image models (`0` = sd-server default). |
| `INFERHOST_SD_SAMPLER` | _(default)_ | Default sampler for image models (e.g. `euler`, `dpm++2m`); blank = sd-server default. |
| `INFERHOST_KV_QUANT_K` | `q8_0` | K cache type (`-ctk`). `q8_0` is ~2× compression, near-lossless; `f16` is lossless. |
| `INFERHOST_KV_QUANT_V` | `q8_0` | V cache type (`-ctv`). Same accepted values as K. Drop to `q5_0` / `q4_0` to save more VRAM. |
| `INFERHOST_LLAMA_SERVER_PATH` | _(auto)_ | Path to a custom `llama-server` binary (e.g. a self-built CUDA binary). |
| `INFERHOST_DATA_DIR` | `~/.local/share/inferhost` | Binaries, logs, and PID files. |
| `INFERHOST_CONFIG_DIR` | `~/.config/inferhost` | Model registry and generated YAML. |
| `INFERHOST_HF_CACHE` | `~/.cache/huggingface` | Hugging Face model cache. |
| `INFERHOST_GPU_LAYERS` | `99` | `-ngl` value passed to llama-server. |
| `INFERHOST_DEFAULT_CTX` | `8192` | Default context length for new models (clamped to the GGUF's native trained context if smaller). |
| `INFERHOST_MAX_OUTPUT_TOKENS` | `0` | Completion cap advertised to agents as `max_output_tokens`. `0` advertises the full served window (llama.cpp shares one budget for input + output); set a positive N to cap it for frameworks that reserve output room. |
| `INFERHOST_FLASH_ATTENTION` | `on` | `-fa` flag for llama-server. |
| `INFERHOST_PARALLEL_SLOTS` | `1` | `--parallel` flag — concurrent request slots per llama-server instance. `1` = serial. |
| `INFERHOST_REASONING` | `auto` | `--reasoning` flag — thinking mode for capable models. `on`, `off`, or `auto`. |
| `INFERHOST_REASONING_BUDGET` | `-1` | `--reasoning-budget` flag — token cap on thinking. `-1` = unlimited, `0` = none. |
| `INFERHOST_LLAMACPP_BACKEND` | auto | Force the prebuilt variant: `vulkan`, `rocm`, `sycl`, `openvino`, `cpu`, or `metal`. Upstream has no Linux CUDA build — pick `vulkan` on NVIDIA Linux. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific upstream llama.cpp release tag (e.g. `b9320` or `9320`). |
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

- **llama.cpp** runs the inference via the official upstream `llama-server` from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the backend (Vulkan, ROCm, SYCL, OpenVINO, CPU, Metal) is picked by hardware probe and can be overridden in the TUI.
- **llama-swap** sits in front of multiple llama-server instances and lazy-loads them on demand. It binds loopback (127.0.0.1) only.
- **LiteLLM** is the single user-facing gateway — always on, always bundled, serving `:9001`.
- **inferhost-tts** (optional, only with a TTS model registered) wraps llama.cpp's `llama-tts` behind `/v1/audio/speech`; LiteLLM routes the gateway's speech requests to it.
- **stable-diffusion.cpp `sd-server`** (optional, only with an image model registered) runs *under* llama-swap — so image models lazy-load and swap VRAM with LLMs — and serves `/v1/images/generations`. No extra daemon; LiteLLM routes image requests through llama-swap to it.

> **Troubleshooting:** Both endpoints are reachable on all interfaces by default (`0.0.0.0`). If you set `INFERHOST_SWAP_HOST=127.0.0.1` and then `curl http://<lan-ip>:9090/...` fails, that override is the reason — switch back to `0.0.0.0` or use `:9001` (the LiteLLM gateway).
>
> **Mouse clicks not working?** If you run inferhost inside `tmux`, tmux must have mouse mode on or it eats the click events before the TUI ever sees them. Fix once with: `tmux set -g mouse on` (or add `set -g mouse on` to `~/.tmux.conf`). inferhost detects this on startup and surfaces a warning toast. To disable mouse capture entirely, set `INFERHOST_MOUSE=off` in your `.env`.

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
