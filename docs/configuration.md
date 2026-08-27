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
INFERHOST_SWAP_PORT=9090        # bound on 0.0.0.0 by default — LAN/Tailscale-reachable
INFERHOST_GATEWAY_PORT=9001     # user-facing LiteLLM endpoint

# KV cache quantization (~2x compression, near-lossless at q8_0).
INFERHOST_KV_QUANT_K=q8_0
INFERHOST_KV_QUANT_V=q8_0

# Repetition guards. Upstream llama-server ships with these OFF, which lets a
# model degenerate into an endless repeat loop (common on aggressively
# quantized / "uncensored" merges) instead of ever stopping. Set either
# multiplier to 0 to go back to upstream's unguarded default.
INFERHOST_REPEAT_PENALTY=1.1
INFERHOST_REPEAT_LAST_N=256
INFERHOST_DRY_MULTIPLIER=0.8
INFERHOST_DRY_BASE=1.75
INFERHOST_DRY_ALLOWED_LENGTH=2
INFERHOST_DRY_PENALTY_LAST_N=4096

# Custom llama-server binary (self-built CUDA, ROCm, etc.)
# INFERHOST_LLAMA_SERVER_PATH=/usr/local/bin/llama-server

# Where binaries, logs, and configs live
INFERHOST_DATA_DIR=~/.local/share/inferhost
INFERHOST_CONFIG_DIR=~/.config/inferhost
INFERHOST_HF_CACHE=~/.cache/huggingface

# Inference defaults
INFERHOST_GPU_LAYERS=99          # offload everything to GPU
INFERHOST_DEFAULT_CTX=8192       # tokens ONE request may use (prompt + reply)
INFERHOST_FLASH_ATTENTION=on
INFERHOST_PARALLEL_SLOTS=1       # --parallel; 1 = serial requests per model
                                 # each slot costs a full context of KV cache

# Reasoning / "thinking" mode for capable models. NOTE: a per-model reasoning
# override (set in the model's settings screen) beats this global value — if a
# model still thinks after setting this to "off", clear or change the per-model
# override too.
INFERHOST_REASONING=auto         # auto | on | off
INFERHOST_REASONING_BUDGET=-1    # token cap on thinking; -1 = unlimited, 0 = none
INFERHOST_REASONING_EFFORT=      # blank | low | medium | high — how hard to think

# Pin specific upstream releases (default: latest). llama.cpp tags look like
# "b9320" (or just "9320"); llama-swap tags look like "v123".
INFERHOST_LLAMACPP_VERSION=latest
INFERHOST_LLAMASWAP_VERSION=latest

# Force a GPU backend (default: auto-detect)
# Accepted: vulkan | rocm | sycl | openvino | cpu | metal
# INFERHOST_LLAMACPP_BACKEND=vulkan

# Stacked speculative decoding (only applied to MTP-capable models).
# Set any value to 0 to disable that lane.
INFERHOST_SPEC_DRAFT_N_MAX=2          # MTP draft tokens per step
INFERHOST_SPEC_NGRAM_MOD_N_MATCH=24   # min matching length before ngram drafts
INFERHOST_SPEC_NGRAM_MOD_N_MIN=48     # min context window to search back through
INFERHOST_SPEC_NGRAM_MOD_N_MAX=64     # max ngram draft tokens on a strong match

# DFlash draft depth — applied when a draft model is attached to a target
# (per-model, via the TUI's `f` key or Configure → Suggest/Browse). 3-4 is the
# consumer-GPU sweet spot; big GPUs can push it to 15-16. 0 disables the lane.
INFERHOST_SPEC_DFLASH_N_MAX=4
```

## Full reference

| Variable | Default | What it does |
|---|---|---|
| `INFERHOST_SWAP_PORT` | `9090` | llama-swap listen port. Bound on `0.0.0.0` by default — reachable from your LAN / Tailscale. Set `INFERHOST_SWAP_HOST=127.0.0.1` for loopback-only. |
| `INFERHOST_GATEWAY_PORT` | `9001` | LiteLLM gateway port — the single user-facing OpenAI-compatible endpoint. |
| `INFERHOST_TTS_PORT` | `9092` | Port for the `inferhost-tts` daemon (serves `/v1/audio/speech`). Only runs when a TTS model is registered. `INFERHOST_TTS_HOST` controls the bind address (`0.0.0.0` by default). |
| `INFERHOST_PINWATCH_POLL_S` | `10` | How often the `inferhost-pinwatch` daemon checks llama-swap. It re-loads pinned models that got evicted (exclusive swap, crash, restart) once no swappable model is using the GPU. |
| `INFERHOST_SDCPP_VERSION` | `latest` | Pin a stable-diffusion.cpp release tag (image generation). The `sd-server` binary is fetched automatically when you add your first image model. |
| `INFERHOST_SD_STEPS` | `0` | Default diffusion steps for image models (`0` = sd-server default). Per-model override via the model's `extra_args`. |
| `INFERHOST_SD_CFG_SCALE` | `0` | Default CFG scale for image models (`0` = sd-server default). |
| `INFERHOST_SD_SAMPLER` | _(default)_ | Default sampler for image models (e.g. `euler`, `dpm++2m`). Blank = sd-server default. |
| `INFERHOST_MAX_OUTPUT_TOKENS` | `0` | Completion cap advertised to agents as `max_output_tokens`. `0` advertises the full served window; set a positive N for frameworks that reserve output room. |
| `INFERHOST_KV_QUANT_K` | `q8_0` | K cache type passed as `-ctk`. `q8_0` is ~2× compression and near-lossless; `f16` is the lossless baseline. |
| `INFERHOST_KV_QUANT_V` | `q8_0` | V cache type passed as `-ctv`. Same accepted values as K — drop to `q5_0` / `q4_0` to save VRAM at the cost of quality. |
| `INFERHOST_LLAMA_SERVER_PATH` | _(auto)_ | Absolute path to a custom `llama-server` binary. Use this for self-built CUDA binaries or any other custom build. |
| `INFERHOST_REPEAT_PENALTY` | `1.1` | `--repeat-penalty` — token-level repetition guard. Upstream default is `1.0` (disabled); inferhost enables a mild penalty by default so a model can't loop forever without ever hitting a stop token. `1.0` disables it, matching upstream. |
| `INFERHOST_REPEAT_LAST_N` | `256` | `--repeat-last-n` — how many recent tokens the repeat penalty looks back over. |
| `INFERHOST_DRY_MULTIPLIER` | `0.8` | `--dry-multiplier` — DRY (Don't Repeat Yourself) sampling strength. Penalizes repeated *sequences* rather than single tokens, so it catches the long garbage loops that plain repeat-penalty misses, without over-penalizing normal code/JSON. Upstream default is `0` (disabled); set to `0` to go back to upstream behavior. |
| `INFERHOST_DRY_BASE` | `1.75` | `--dry-base` — DRY penalty growth base. |
| `INFERHOST_DRY_ALLOWED_LENGTH` | `2` | `--dry-allowed-length` — shortest repeated sequence DRY will penalize. |
| `INFERHOST_DRY_PENALTY_LAST_N` | `4096` | `--dry-penalty-last-n` — how many recent tokens DRY scans for repeats. Upstream default (`64`) is too short to catch a loop that's already run for a while; `4096` covers a realistic tool-call-sized window. |
| `INFERHOST_DATA_DIR` | `~/.local/share/inferhost` | Where downloaded binaries, logs, and PID files live. |
| `INFERHOST_CONFIG_DIR` | `~/.config/inferhost` | Where the generated `llama-swap.yaml` and the model registry live. |
| `INFERHOST_HF_CACHE` | `~/.cache/huggingface` | Hugging Face model cache root. |
| `INFERHOST_GPU_LAYERS` | `99` | The `-ngl` flag passed to llama-server (number of layers offloaded to GPU). `99` ≈ "everything that fits". |
| `INFERHOST_DEFAULT_CTX` | `8192` | Default context window for newly added models, in tokens. This is the window a **single request** gets (prompt + reply), not the total KV cache — see `INFERHOST_PARALLEL_SLOTS`. |
| `INFERHOST_FLASH_ATTENTION` | `on` | Pass `-fa` to llama-server. Set to `off` if your GPU doesn't support it. |
| `INFERHOST_PARALLEL_SLOTS` | `1` | Pass `--parallel <n>` to llama-server. Each slot can handle one in-flight request on the same model. Keep at `1` unless you actually need concurrency: **every slot gets its own full context window**, so `n` slots cost `n ×` the KV cache VRAM. (llama-server divides its `-c` across the slots; inferhost sizes `-c` as *context × slots* so raising the slot count never shrinks the window a request can use.) If *context × slots* won't fit the GPU, inferhost lowers the slot count rather than your context, and says so in a notice and in the dashboard's degraded banner. |
| `INFERHOST_THREADS` | `0` | CPU threads for generation (`--threads`). `0` = auto (llama-server uses the physical core count). Matters mainly for models running partly on CPU (low GPU layers or `--cpu-moe`); negligible for a fully GPU-offloaded model. Per-model override in Configure. |
| `INFERHOST_REASONING` | `auto` | `--reasoning` flag for thinking-capable models (DeepSeek, Qwen3-Thinking, GPT-OSS, ...). `auto` lets the model decide, `on` forces thinking, `off` suppresses it. |
| `INFERHOST_REASONING_BUDGET` | `-1` | `--reasoning-budget` — token cap on thinking. `-1` = unlimited, `0` = none, positive = hard cut-off. |
| `INFERHOST_REASONING_EFFORT` | _(blank)_ | How hard the model should think, for chat templates that grade thinking rather than just switching it on and off: `low`, `medium`, `high`. Passed as `--chat-template-kwargs '{"reasoning_effort": "..."}'`, not as a llama-server flag, and skipped entirely when reasoning resolves to `off`. Blank leaves the template's own default — which on Qwen3.8 is its most expensive setting, so setting `medium` is a large latency win. Per-model override in Configure. Ignored by templates that don't read it. |
| `INFERHOST_LLAMACPP_BACKEND` | _auto_ | Force the prebuilt variant: `vulkan`, `rocm`, `sycl`, `openvino`, `cpu`, or `metal`. Only applies when `INFERHOST_LLAMA_SERVER_PATH` is not set. Note: upstream does not ship a Linux CUDA prebuilt — pick `vulkan` on NVIDIA Linux. |
| `INFERHOST_LLAMACPP_VERSION` | `latest` | Pin a specific upstream llama.cpp release tag (e.g. `b9320` or `9320`). |
| `INFERHOST_LLAMASWAP_VERSION` | `latest` | Pin a specific llama-swap release tag. |
| `INFERHOST_SPEC_DRAFT_N_MAX` | `2` | MTP draft tokens per step (`--spec-draft-n-max`). Applied to models whose GGUF declares MTP heads (`*.nextn_predict_layers`). Set to `0` to disable the MTP lane. Per-model override in Configure. Measured on a 3090 Ti with Qwen3.8-27B at 64k: 44 tok/s with no draft, 71 at 2, **74 at 3**, 68 at 4 — the optimum is model- and GPU-specific, so sweep it rather than trusting the default. |
| `INFERHOST_SPEC_NGRAM_MOD_N_MATCH` | `24` | Min matching sequence length before ngram-mod drafts (`--spec-ngram-mod-n-match`). |
| `INFERHOST_SPEC_NGRAM_MOD_N_MIN` | `48` | Min context window ngram-mod searches back through (`--spec-ngram-mod-n-min`). |
| `INFERHOST_SPEC_NGRAM_MOD_N_MAX` | `64` | Max draft tokens ngram-mod proposes on a strong match (`--spec-ngram-mod-n-max`). Set to `0` to disable the ngram-mod lane. |
| `INFERHOST_SPEC_DFLASH_N_MAX` | `4` | DFlash draft tokens per step (`--spec-draft-n-max`), applied when a DFlash draft model is attached to a target. 3–4 is the consumer-GPU sweet spot; big GPUs can go to 15–16. Set to `0` to disable the DFlash lane. Per-model **DFlash draft tokens** in Configure overrides this. |

## KV cache quantization (`INFERHOST_KV_QUANT_K` / `_V`)

inferhost passes these directly as `-ctk` / `-ctv` to upstream `llama-server`. The default is `q8_0` for both — ~2× compression of the f16 baseline with near-lossless quality.

| Value | Approx. KV bytes/element | Notes |
|---|---|---|
| `f16` / `bf16` | 2.0 | Lossless baseline. |
| `q8_0` | 1.06 | **Default.** ~2× compression, near-lossless. |
| `q5_1` / `q5_0` | 0.75 / 0.69 | Saves more VRAM; small quality hit. |
| `q4_1` / `q4_0` / `iq4_nl` | 0.63 / 0.56 / 0.50 | Aggressive; quality varies by model. |
| `off` | — | Don't pass the flag (llama-server picks its own default). |

To disable KV quant entirely:

```env
INFERHOST_KV_QUANT_K=off
INFERHOST_KV_QUANT_V=off
```

## `INFERHOST_LLAMA_SERVER_PATH` — escape hatch for custom builds

If the upstream prebuilt for your hardware doesn't exist (e.g. you want a Linux CUDA build), point inferhost at any compatible `llama-server` binary. Set it in `~/.config/inferhost/inferhost.env` like any other variable:

```bash
INFERHOST_LLAMA_SERVER_PATH=/home/user/src/llama.cpp/build/bin/llama-server
```

When this variable is set, inferhost skips the binary download step entirely and uses your path instead — including on upgrades, so keeping the binary current becomes your job.

### Building a CUDA `llama-server` on Linux

Upstream ships no Linux CUDA prebuilt, so NVIDIA boxes run Vulkan by default. Compiling CUDA yourself mainly buys faster prompt processing; token generation on a large quant is memory-bandwidth-bound and moves much less.

```bash
git clone --depth 1 --branch <tag> https://github.com/ggml-org/llama.cpp.git ~/src/llama.cpp
cd ~/src/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<your compute capability, e.g. 89 for Ada> \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j "$(nproc)"
```

Two things matter here:

- **`-DBUILD_SHARED_LIBS=OFF` is required.** The generated llama-swap config pins `LD_LIBRARY_PATH` to inferhost's own `bin/` directory, which holds the managed backend's `libggml*.so`. A dynamically linked custom build resolves those instead of its own and you get the backend you were trying to replace — or an ABI crash.
- **Match your CUDA toolkit's host compiler.** CUDA 12.0 rejects GCC 13; pass `-DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12` if `nvcc --version` predates your default GCC.

Verify before pointing inferhost at it:

```bash
./build/bin/llama-server --list-devices   # should list CUDA0: <your GPU>
```

## How auto-detection works

If you don't set `INFERHOST_LLAMACPP_BACKEND` and don't set `INFERHOST_LLAMA_SERVER_PATH`, inferhost runs a small probe at install time:

1. **Apple Silicon?** Use the macOS arm64 Metal prebuilt asset.
2. **NVIDIA GPU on Linux?** Use the Vulkan prebuilt asset (upstream does not ship a Linux CUDA build).
3. **No GPU / fallback?** Use the CPU prebuilt asset.

For ROCm (AMD), SYCL / OpenVINO (Intel), set `INFERHOST_LLAMACPP_BACKEND` explicitly.

## Changing settings

Any change to a `.env` value or env var takes effect the next time `inferhost` (or `./run.sh start`) launches the TUI / daemon.

Changes made **inside the TUI's Settings panel** (`,`) auto-apply: if llama-swap is already running, saving restarts it (and re-warms any pinned/loaded models) automatically — no extra keypress needed. **`r`** still force-restarts on demand (e.g. after editing `.env` directly, or if you just want to be sure).

[Continue to Troubleshooting →](troubleshooting.md)
