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

## The model fails to start when I make a request

Open the TUI and look at the **Logs** panel — that's the live tail of `llama-swap.log`. The most common errors:

| Log message | Fix |
|---|---|
| `failed to load model` | The GGUF file may be incomplete. Remove and re-add the model. |
| `out of memory` / `CUDA error: out of memory` | Pick a smaller quant for this model, or set `INFERHOST_GPU_LAYERS` to a smaller number to offload less to the GPU. |
| `flash attention not supported` | Set `INFERHOST_FLASH_ATTENTION=off` in `.env`. |

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
