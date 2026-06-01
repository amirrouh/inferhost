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
│ ● swap 127.0.0.1:9090 (internal)   ● litellm http://localhost:9001/v1│
│ gateway_port=9001  ctx=8192  gpu_layers=99  fa=on  kv=q8_0/turbo3   │
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
│ a=add  n=rename  c=ctx  d=remove │ s/x/r=swap │ p=settings           │
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
| **`c`** | **C**onfigure the highlighted model: per-model context (`-c`) |
| **`P`** | **P**in the highlighted model — loads it into VRAM immediately. Press **`P`** again to unpin and unload. inferhost checks VRAM first and shows a warning if the model won't fit. |
| **`d`** / **`Delete`** | **D**elete the highlighted model from the registry |
| **`s`** | **S**tart llama-swap |
| **`x`** | Stop llama-swap |
| **`r`** | **R**estart llama-swap |
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

Then point any OpenAI-compatible client at the **LiteLLM gateway endpoint** shown in the top bar.

### curl

```bash
curl http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4-k-m",
    "messages": [{"role": "user", "content": "Tell me a joke about cats."}]
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9001/v1", api_key="none")

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
| Base URL | `http://localhost:9001/v1` |
| API key | anything non-empty (e.g. `none`) |
| Model | the `name` column from the dashboard |

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:9001/v1",
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

## Configuring a model (context window)

The global **Default context** (in Settings) is only used when adding a *new*
model. To change settings on an existing model, highlight it and press
**`c`**:

```
┌── Model settings ────────────────────────────┐
│ Model: qwen3.6-27b-heretic-mtp-q5-k-m        │
│                                              │
│ Context window (-c)                          │
│ [32768_________________________________]     │
│                                              │
│              [Cancel]  [Save]                │
└──────────────────────────────────────────────┘
```

inferhost saves the value to the registry, regenerates `llama-swap.yaml`, and
reloads any running daemon so the new flag takes effect immediately.

**KV cache compression** is handled globally and asymmetrically via `INFERHOST_KV_QUANT_K` (default `q8_0`) and `INFERHOST_KV_QUANT_V` (default `turbo3`). The split exists because K compression breaks attention while V compression is essentially free — the TurboQuant fork lets us aggressively compress V while keeping K safe. To tune or disable, set those variables in your `.env`. See [Configuration](configuration.md) for the full table.

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

## Text-to-speech models

When a Hugging Face repo ships a **WavTokenizer / vocoder** GGUF alongside the
model (e.g. an OuteTTS repo), inferhost auto-downloads it and serves the model as
a **text-to-speech** model. It's marked `♪ [tts]` in the dashboard and exposed on
the same gateway at `/v1/audio/speech`:

```bash
curl http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "outetts-0.2-500m-q4-k-m", "input": "Hello from inferhost.", "voice": "default"}' \
  --output speech.wav
```

```python
# OpenAI Python SDK
client.audio.speech.create(model="outetts-0.2-500m-q4-k-m", input="Hello.", voice="default")
```

`voice` is **required** when calling through the gateway (OpenAI/LiteLLM mandate
it). The value is ignored unless it's a path to a `llama-tts` speaker file.

How it works and what to expect:

- Synthesis runs through llama.cpp's standalone `llama-tts` binary — the only way
  to render OuteTTS+vocoder. It's bundled automatically on install/update.
- `llama-tts` has no resident-server mode, so **the model reloads on every
  request** (a few seconds of overhead). Good for occasional/scripted use, not
  for low-latency streaming.
- A small `inferhost-tts` daemon serves the endpoint; `inferhost start` brings it
  up automatically whenever a TTS model is registered (`INFERHOST_TTS_PORT`,
  default `9092`). LiteLLM routes the gateway's `/v1/audio/speech` to it.
- Output is **WAV**. `voice` is ignored unless it's a path to a `llama-tts`
  speaker file.
- TTS models don't run under llama-swap and can't be pinned/loaded into VRAM
  ahead of time — there's nothing to keep resident.
- **Auto-detect only:** the vocoder must live in the *same* repo as the model.
  If a vocoder ships in a separate repo it won't be picked up. If you added an
  OuteTTS model before it was recognized as TTS (it ran as a plain chat model),
  **remove and re-add it** so the vocoder is detected.

## Image generation

inferhost bundles [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
`sd-server`. In the add-model screen, switch the kind selector to **Image
generation**, then add a model exactly like an LLM (paste repo → pick from the
list — now including `.safetensors`):

```bash
curl http://localhost:9001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "stable-diffusion-v1-5-q4-0", "prompt": "a watercolor fox", "size": "512x512"}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

```python
# OpenAI Python SDK
img = client.images.generate(model="stable-diffusion-v1-5-q4-0", prompt="a watercolor fox", size="512x512")
```

How it works:

- **Single-file (SD1.5/SDXL):** one pick. **Multi-file (Flux/SD3):** inferhost
  auto-detects + downloads the VAE/CLIP/T5 companions **in the same repo**. The
  `sd-server` binary is fetched automatically the first time you add an image model.
- **VRAM:** image models run under llama-swap in the swappable group, so they
  **evict and are evicted by LLMs** — only one big model resident at a time. They
  lazy-load on the first request (the first image after a swap is slower).
- **Parameters:** `size` per request; `steps`/`cfg`/`sampler` as per-model defaults
  in the model's `extra_args` (Configure), or per request by embedding
  `<sd_cpp_extra_args>{"sample_steps":8}</sd_cpp_extra_args>` in the prompt.
- **Multi-file models (Flux.1 / Flux.2 Klein / Z-Image / Qwen-Image) auto-assemble.**
  inferhost ships **recipes** for these families: add the diffusion model and it
  recognizes the family, **auto-downloads the right VAE + text encoder(s)** from
  known-good non-gated repos, and sets sane `--steps`/`--cfg-scale` — no manual
  file hunting. (Bonsai-Image is a Flux.2-Klein model, so it uses that recipe.)
- **No recipe? Use the component editor.** Add the diffusion file, open
  **Configure**, and fill each slot (VAE, Text encoder `--llm`, Vision encoder
  `--llm_vision`, CLIP-L/G, T5XXL) via the same *paste repo → pick from list* flow.
- **Image editing (Qwen-Image-Edit, Flux Kontext):** the OpenAI `/v1/images/edits`
  endpoint is multipart, which the gateway doesn't route by model — hit llama-swap
  directly: `POST http://<host>:9090/upstream/<model>/v1/images/edits`.
- **Quality:** same weights as ComfyUI → comparable txt2img; not ComfyUI's full
  feature set/speed. ComfyUI can run alongside inferhost if you need more.

## Speculative decoding (MTP models)

Models with `mtp` in the filename (e.g. `qwen3.6-27b-heretic-mtp-q5-k-m`) get
two speculative-decode lanes stacked automatically:

- **`--spec-type draft-mtp`** uses the MTP heads baked into the GGUF.
- **`--spec-type ngram-mod`** uses pattern lookup over the already-generated
  text.

MTP wins on novel generation, ngram-mod dominates on repeated patterns (code,
function names, repeated constructs). All four knobs are tunable via
`INFERHOST_SPEC_*` env vars (see [Configuration](configuration.md)).

## Pinning models (load into VRAM immediately)

Press **`P`** on a highlighted model to pin it. Pinning:

1. **Immediately loads the model into VRAM** — it does not wait for a client request.
2. **Checks VRAM first.** If the model would exceed available VRAM, inferhost shows a modal: "Not enough VRAM — unpin another model first."
3. Pinned models are co-resident: they share a llama-swap group with `swap: false` so they stay loaded together instead of unloading each other.

Press **`P`** again on a pinned model to **unpin and unload** it.

The sidebar marks pinned models with a `★`. The details panel shows `loading: ★ pinned (co-resident)`.

## Changing ports, context, or GPU layers

Press **`p`** to open the Settings panel. You can edit:

| Field | What it does |
|---|---|
| llama-swap port | Port for llama-swap (default `9090`, bound on `0.0.0.0`) |
| Gateway port | The LiteLLM user-facing endpoint port (default `9001`) |
| Default context | Context window for newly added models (tokens) |
| GPU layers (-ngl) | `99` = offload everything, `0` = CPU only |
| Flash attention | `on`, `off`, or `auto` |
| Parallel slots (--parallel) | Concurrent request slots per llama-server instance. `1` (default) = serial. |

Saving writes a managed env file at `~/.config/inferhost/inferhost.env`, so your
changes persist across restarts of the TUI. After saving, press **`r`** to
restart llama-swap with the new values.

The per-model **Configure** screen (`c`) additionally exposes **CPU threads**
(`--threads`), **MoE experts on CPU** (`--n-cpu-moe`), and **Lock in RAM**
(`--mlock`).

### Speeding up MoE models (Mixture-of-Experts)

For a MoE model (e.g. Qwen3-A3B, Mixtral) the experts are most of the weight but
only a few are active per token. The biggest speed lever is getting the *experts*
onto the GPU, not just raising `-ngl`:

- Set **GPU layers = 99** (all attention on GPU) **and** **MoE experts on CPU
  (`--n-cpu-moe`) = N**: keep only the first N layers' experts on CPU, the rest
  run on GPU. Lower N → more experts on GPU → faster, until VRAM fills.
- **`--n-cpu-moe 0`** = *all* experts on GPU (fastest, if it fits).

Tune N to your VRAM budget: a higher N keeps the model leaner so it can share the
GPU with other models. (Measured example, 35B-A3B at 100k context on a 24 GB card:
~10 tok/s with everything swapping to CPU vs ~75 tok/s with all experts on GPU.)

## Running more than one model

Add as many as you like. By default llama-swap loads each one on the first
request and unloads it after an idle period, so you can keep dozens registered
without burning VRAM. Only one model is resident at a time — when you call a
second model, the first gets unloaded.

Use the model `name` from the dashboard as the `model` field in your request — llama-swap routes it to the right backend.

## Streaming

All OpenAI streaming features (`stream=True`, server-sent events, tool calls, JSON mode where the model supports it) work out of the box, because llama.cpp's `llama-server` already implements them.

## Removing a model

Highlight a model in the sidebar and press **`d`** (or `Delete`). This removes it from the registry but keeps the GGUF file in the Hugging Face cache — adding the same repo again is instant.

## Quitting

Press **`q`** to leave the TUI. **llama-swap and LiteLLM keep running in the background** so your endpoint stays up. To stop them from a shell:

```bash
# If you installed via pip and you're not in the repo:
pkill -f llama-swap
pkill -f litellm

# If you cloned the repo:
./run.sh stop
```

[Continue to Configuration →](configuration.md)
