---
layout: default
title: Troubleshooting
---

[← Back to overview](index.md)

# Troubleshooting

## The TUI says "llama-swap stopped"

The daemon isn't running. Press **`s`** in the TUI to start it. If it fails to start, check the log file at `~/.local/share/inferhost/logs/llama-swap.log`.

Most common causes:

- **No models registered yet.** Press **`a`** to add one first.
- **Port in use.** Another process is on `9090`. Set `INFERHOST_SWAP_PORT=...` in `.env` to a free port and restart.
- **Binary missing.** Re-launch `inferhost` — it will redownload missing binaries on next start.

## "Port 9090 is already in use"

Either:

- Find the process: `lsof -i :9090` (Linux/macOS) — and kill it if it's another inferhost from earlier.
- Or change the port in `.env`:

```env
INFERHOST_SWAP_PORT=9099
```

Then re-launch `inferhost`.

## `curl http://<lan-ip>:9090/...` doesn't work

This is expected. In v0.5+, llama-swap binds `127.0.0.1` (loopback) only and is **not reachable from the network by design**. It is an internal component.

Use the **LiteLLM gateway on port `9001`** instead — that is the single user-facing endpoint:

```bash
curl http://<lan-ip>:9001/v1/chat/completions ...
```

If you need to change the gateway port, set `INFERHOST_GATEWAY_PORT` in `.env`.

## The model fails to start when I make a request

Open the TUI and look at the **Logs** panel — that's the live tail of `llama-swap.log`. The most common errors:

| Log message | Fix |
|---|---|
| `failed to load model` | The GGUF file may be incomplete. Remove and re-add the model. |
| `out of memory` / `CUDA error: out of memory` | Pick a smaller quant for this model, or set `INFERHOST_GPU_LAYERS` to a smaller number to offload less to the GPU. You can also try a lighter `INFERHOST_KV_QUANT` value. |
| `flash attention not supported` | Set `INFERHOST_FLASH_ATTENTION=off` in `.env`. |

## "unknown model architecture" on a model that just came out

```
error loading model: unknown model architecture: 'muse-glimmer'
```

Nothing is wrong with the download — the `llama-server` on disk is simply older than the model. llama.cpp adds each new architecture in a specific upstream build, and binaries are otherwise fetched only on first launch, so they go stale as the box ages.

inferhost handles this on its own: at `start` it reads each model's architecture from the GGUF, checks whether the binary knows it, and pulls a current llama.cpp when none does. So a restart usually clears it. You can also do it explicitly:

```bash
inferhost update          # or: ./run.sh update
```

This stops the daemons, re-downloads `llama-server` + `llama-tts`, `llama-swap` (and `sd-server` if image generation is installed), then brings back whatever was running. It prints the old and new build, e.g. `llama-server : b10068 -> b10412`.

If a specific build is what you need, pass its upstream tag:

```bash
inferhost update b10353
```

To stay on one build permanently, set `INFERHOST_LLAMACPP_VERSION=b10353` in `.env` (default: `latest`).

### If you use your own build

With `INFERHOST_LLAMA_SERVER_PATH` set, `update` never touches your binary — it's yours to rebuild — but it does report how far behind it is:

```
llama-server : skipped — INFERHOST_LLAMA_SERVER_PATH points at /home/me/src/llama.cpp/build/bin/llama-server
               your build     : version: 1 (7ba604f)
               upstream latest: b10412
```

A custom build that predates a model doesn't stop that model from running: inferhost serves *that one model* with its own managed binary and says so, while everything else keeps using your build. Rebuild your llama.cpp to get it back on your binary everywhere:

```bash
cd ~/src/llama.cpp && git fetch --tags && git checkout b10412
cmake --build build --target llama-server -j$(nproc)
```

## Prebuilt llama-server doesn't match my platform

inferhost ships prebuilt `llama-server` binaries for three targets: **Linux x86_64 CUDA 12.x**, **Linux x86_64 CPU**, and **macOS arm64 Metal**. If you run Vulkan, ROCm, an older CUDA, or a platform not in that list, the prebuilt binary may not work.

Use the `INFERHOST_LLAMA_SERVER_PATH` escape hatch to point inferhost at a compatible binary you build or obtain yourself:

```bash
# Example: ROCm build from source
cd ~/llama.cpp
cmake -B build -DGGML_HIPBLAS=ON
cmake --build build --target llama-server -j$(nproc)

# Tell inferhost to use it
export INFERHOST_LLAMA_SERVER_PATH=~/llama.cpp/build/bin/llama-server
inferhost
```

Or add it to `.env` so it persists:

```env
INFERHOST_LLAMA_SERVER_PATH=/home/user/llama.cpp/build/bin/llama-server
```

When `INFERHOST_LLAMA_SERVER_PATH` is set, inferhost skips the binary download step entirely.

## "Hugging Face repo not found"

Double-check the spelling. The repo id is the `org/name` shown at the top of the Hugging Face page, e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`. It must point to a repo containing **GGUF** files.

If the repo is gated or private, log in first:

```bash
huggingface-cli login
```

Then re-launch `inferhost`.

## Download is slow

Hugging Face throttles unauthenticated downloads. Two fixes:

1. `huggingface-cli login` — authenticated downloads are faster.
2. Install [`hf_transfer`](https://huggingface.co/docs/huggingface_hub/hf_transfer):
   ```bash
   pip install hf_transfer
   export HF_HUB_ENABLE_HF_TRANSFER=1
   inferhost
   ```

## The dashboard shows the wrong model name format

Names are derived from the repo id and the quant tag — they're lowercase, dashes only. If you don't like the auto-generated name, you can edit `~/.config/inferhost/models.toml` directly and then press **`r`** to restart llama-swap.

## Deleting a model did not free any disk

It does now — removing a model deletes its weights (blobs included), and the confirm prompt shows how much it will free. Files shared with another registered model are kept, and a model you added from your own path outside the Hugging Face cache is never deleted.

Weights stranded by older versions are still there. List them, with sizes:

```bash
inferhost prune            # or: ./run.sh prune
```

Nothing is deleted until you pass `--yes`. Read the list first: the Hugging Face cache is shared with every other tool on the box that uses `huggingface_hub` (ComfyUI, a Whisper server, ...), and inferhost cannot tell their downloads from stale ones.

```bash
inferhost prune --yes
```

## I want to reset everything and start over

From the repo (development) directory:

```bash
./run.sh reset       # stops daemons and clears the registry (keeps GGUFs in HF cache)
./run.sh uninstall   # also removes the venv and the data dir
```

If you installed via `pip install inferhost`:

```bash
# Stop any daemons
pkill -f llama-swap || true
pkill -f litellm || true
# Wipe inferhost state (keeps the Hugging Face model cache)
rm -rf ~/.local/share/inferhost ~/.config/inferhost
```

## My model isn't on Hugging Face as GGUF

inferhost only supports GGUF (the format llama.cpp uses). If you have a model in safetensors / `.bin`, convert it first with [llama.cpp's conversion scripts](https://github.com/ggml-org/llama.cpp/blob/master/examples/convert_hf_to_gguf.py), upload the GGUF to Hugging Face (or a local path), and then point inferhost at the repo.

## I think I found a bug

Please open an issue on [GitHub](https://github.com/amirrouh/inferhost/issues) with:

- The output of running `python -c "import inferhost; print(inferhost.__version__)"`
- Your OS, Python version, and GPU
- The relevant part of `~/.local/share/inferhost/logs/llama-swap.log`

[← Back to overview](index.md)
