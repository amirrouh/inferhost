---
layout: default
title: Usage
---

[← Back to overview](index.md)

# Usage

## Launching

There is exactly one command:

```bash
inferhost
```

This opens the TUI. Everything happens inside the TUI: adding models, starting / stopping the daemon, watching logs, removing models.

## The dashboard

```
┌─ inferhost ──────────────────────────────────────────────────────────┐
│ ● swap http://localhost:9090/v1    ○ litellm http://localhost:9001/v1│
│ swap_port=9090  gateway_port=9001  ctx=8192  gpu_layers=99  fa=on    │
│                                                                      │
│ Models                       Details                                 │
│ ───────────────────────────  ────────────────────────────────────── │
│ qwen2.5-7b-instruct-q4-k-m   name:  qwen2.5-7b-instruct-q4-k-m       │
│ llama-3.2-3b-instruct-q5     repo:  Qwen/Qwen2.5-7B-Instruct-GGUF    │
│ gemma-2-9b-it-q4-k-m         quant: Q4_K_M  size: 4.4 GiB  ctx: 8192│
│                              port:  9091                             │
│                                                                      │
│                              Logs                                    │
│                              llm_load_tensors: offloaded 33/33 ...   │
│                                                                      │
│ a=add  n=rename  c=ctx  d=remove │ s/x/r=swap │ g=gateway │ p=settings│
└──────────────────────────────────────────────────────────────────────┘
```

The top two lines show, at a glance, **what's running** (green dot = up, red /
grey dot = down) and **every setting that's currently in effect**. Nothing is
hidden behind a hidden menu.

### Every key

| Key | Action |
|---|---|
| **`a`** | **A**dd a Hugging Face model (with download progress) |
| **`n`** | Re**n**ame the highlighted model's alias |
| **`c`** | **C**onfigure the highlighted model: per-model context (`-c`) and KV cache quant (`-ctk` / `-ctv`) |
| **`P`** | **P**in the highlighted model — pinned models stay co-resident in VRAM instead of swapping |
| **`d`** / **`Delete`** | **D**elete the highlighted model from the registry |
| **`s`** | **S**tart llama-swap |
| **`x`** | Stop llama-swap |
| **`r`** | **R**estart llama-swap |
| **`g`** | Toggle the LiteLLM **g**ateway on/off |
| **`p`** | Open the **P**references / Settings panel |
| **`R`** | **R**efresh the view |
| **`q`** | **Q**uit |

## Adding a model

1. Press **`a`** to open the Add Model dialog.
2. Type a Hugging Face repo id, e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`, and press **Enter**.
3. inferhost lists all GGUF files in the repo. Each row shows:
   - **★** — the recommended quant for your GPU
   - **✓** / **·** — whether the file fits in your VRAM
   - quant tag, size, and filename
4. Use the arrow keys to highlight a row (or accept the recommendation) and press **Add**.
5. A progress bar appears while the file downloads from Hugging Face. When it finishes, the dialog closes and the model is registered.

## Starting and using it

llama-swap starts the model lazily on the first request. To pre-warm it, press **`s`** (start). To restart after changing the registry, press **`r`**.

Then point any OpenAI-compatible client at the endpoint shown in the top bar.

### curl

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4-k-m",
    "messages": [{"role": "user", "content": "Tell me a joke about cats."}]
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9090/v1", api_key="none")

resp = client.chat.completions.create(
    model="qwen2.5-7b-instruct-q4-k-m",
    messages=[{"role": "user", "content": "Tell me a joke about cats."}],
)
print(resp.choices[0].message.content)
```

### Continue / Cursor / LibreChat / Open WebUI

In any tool that supports a custom OpenAI base URL:

| Setting | Value |
|---|---|
| Base URL | `http://localhost:9090/v1` |
| API key | anything non-empty (e.g. `none`) |
| Model | the `name` column from the dashboard |

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:9090/v1",
    api_key="none",
    model="qwen2.5-7b-instruct-q4-k-m",
)
```

## Renaming a model

The model name shown in the sidebar is the same name your client puts in the
OpenAI `model` field. To change it, highlight the model and press **`n`**.

```
┌── Rename model ─────────────────────────────┐
│ Current: qwen2.5-7b-instruct-q4-k-m         │
│ This is the name your OpenAI client uses... │
│ [my-fast-qwen____________________]          │
│                                             │
│              [Cancel]  [Rename]             │
└─────────────────────────────────────────────┘
```

inferhost rewrites the llama-swap and LiteLLM YAML configs in one shot — you
never need to touch them by hand. If llama-swap is already running, it restarts
automatically so the new alias is immediately reachable.

## Configuring a model (ctx + KV cache quant)

The global **Default context** (in Settings) is only used when adding a *new*
model. To change settings on an existing model, highlight it and press
**`c`**:

```
┌── Model settings ──────────────────────────────────┐
│ Model: qwen3.6-27b-heretic-mtp-q5-k-m              │
│                                                    │
│ Context window (-c)                                │
│ [32768___________________________________]         │
│                                                    │
│ KV cache type — K (-ctk)                           │
│ [q8_0__________________ blank=f16 default · q8_0…] │
│                                                    │
│ KV cache type — V (-ctv)                           │
│ [q8_0__________________________________]           │
│                                                    │
│              [Cancel]  [Save]                      │
└────────────────────────────────────────────────────┘
```

inferhost saves the values to the registry, regenerates `llama-swap.yaml`, and
reloads any running daemon so the new flags take effect immediately.

**KV cache quantization** is the cheapest way to fit a larger `ctx` into the
same VRAM:

| Value | KV memory vs. `f16` | Quality |
|---|---|---|
| (blank) / `f16` | 1.0× (default) | reference |
| `q8_0` | ~0.5× | near-lossless |
| `q5_1`, `q5_0` | ~0.4× | small loss |
| `q4_1`, `q4_0` | ~0.25× | noticeable on long contexts |

## Vision (multimodal) models

When a Hugging Face repo ships an `mmproj-*.gguf` (e.g. Qwen-VL, Gemma vision,
LLaVA), inferhost auto-downloads it alongside the main file and adds the
`-mm <path>` flag to the llama-server command. From then on the model accepts
OpenAI-style image content blocks:

```python
client.chat.completions.create(
    model="qwen3vl-8b-instruct-q8-0",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ],
    }],
)
```

No extra setup, no flags. If the repo doesn't ship an `mmproj`, the model
stays text-only and `-mm` is simply not added.

## Speculative decoding (MTP models)

Models with `mtp` in the filename (e.g. `qwen3.6-27b-heretic-mtp-q5-k-m`) get
two speculative-decode lanes stacked automatically:

- **`--spec-type draft-mtp`** uses the MTP heads baked into the GGUF.
- **`--spec-type ngram-mod`** uses pattern lookup over the already-generated
  text.

MTP wins on novel generation, ngram-mod dominates on repeated patterns (code,
function names, repeated constructs). All four knobs are tunable via
`INFERHOST_SPEC_*` env vars (see [Configuration](configuration.md)).

## Changing ports, context, or GPU layers

Press **`p`** to open the Settings panel. You can edit:

| Field | What it does |
|---|---|
| llama-swap port | The OpenAI-compatible endpoint port (default `9090`) |
| Gateway port | The LiteLLM gateway port (default `9001`) |
| Default context | Context window for newly added models (tokens) |
| GPU layers (-ngl) | `99` = offload everything, `0` = CPU only |
| Flash attention | `on`, `off`, or `auto` |
| Parallel slots (--parallel) | Concurrent request slots per llama-server instance. `1` (default) = serial. |

Saving writes a managed env file at `~/.config/inferhost/inferhost.env`, so your
changes persist across restarts of the TUI. After saving, press **`r`** to
restart llama-swap with the new values.

## Toggling the LiteLLM gateway

Press **`g`** to start (or stop) the LiteLLM gateway. The status bar at the top
shows whether it's running and on which port. The gateway is optional — install
it with `uv tool install 'inferhost[gateway]'` (or reinstall with the extra) if
you want a single OpenAI-compatible endpoint that can route across multiple
providers.

## Running more than one model

Add as many as you like. By default llama-swap loads each one on the first
request and unloads it after an idle period, so you can keep dozens registered
without burning VRAM. Only one model is resident at a time — when you call a
second model, the first gets unloaded.

Use the model `name` from the dashboard as the `model` field in your request — llama-swap routes it to the right backend.

### Keeping two (or more) models loaded together — pin

If you want two models **co-resident** instead of swapping each other out,
**pin** them. Highlight a model and press **`P`** to toggle the pin (or use
the `Pin in VRAM` field in the **`c`** Configure modal). Pinned models share a
llama-swap group with `swap: false`, so they all stay loaded together; unpinned
models still swap on demand. The sidebar marks pinned models with a yellow
`★`, and the details panel shows `loading: ★ pinned (co-resident)`.

Make sure your pinned set actually fits in VRAM — the GPU bar at the top of the
dashboard is your guide. If you pin more than the card can hold, llama-server
will OOM trying to load the second one.

## Streaming

All OpenAI streaming features (`stream=True`, server-sent events, tool calls, JSON mode where the model supports it) work out of the box, because llama.cpp's `llama-server` already implements them.

## Removing a model

Highlight a model in the sidebar and press **`d`** (or `Delete`). This removes it from the registry but keeps the GGUF file in the Hugging Face cache — adding the same repo again is instant.

## Quitting

Press **`q`** to leave the TUI. **llama-swap keeps running in the background** so your endpoint stays up. To stop it from a shell:

```bash
# If you installed via pip and you're not in the repo:
pkill -f llama-swap

# If you cloned the repo:
./run.sh stop
```

[Continue to Configuration →](configuration.md)
