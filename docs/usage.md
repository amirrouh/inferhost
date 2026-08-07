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
| **`p`** / **`P`** | **P**in/unpin the highlighted model — pinning loads it into VRAM immediately. inferhost checks VRAM first and shows a warning if the model won't fit. |
| **`l`** / **`Enter`** | **L**oad (or unload) the highlighted model right now, without pinning it |
| **`f`** | Enable DF**l**ash speculative decoding: fetch + attach the paired community draft model to the highlighted chat model (no-op with a notice if there's no known pairing) |
| **`d`** / **`Delete`** | **D**elete the highlighted model from the registry — asks for confirmation first |
| **`s`** | **S**tart llama-swap |
| **`x`** | Stop llama-swap |
| **`r`** | **R**estart llama-swap |
| **`g`** | Toggle the **g**ateway (LiteLLM) on/off |
| **`,`** | Open the Settings panel — changes auto-apply (restarting llama-swap if it's running); **`r`** still force-restarts on demand |
| **`R`** | **R**efresh the view |
| **`q`** | **Q**uit |

## Adding a model

1. Press **`a`** to open the Add Model dialog.
2. Pick a kind: **Chat / LLM**, **Image generation**, or **Text-to-speech**.
3. Type a Hugging Face repo id, e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`, and press **Enter**.
4. inferhost lists the matching files in the repo. Each row shows:
   - **★** — the recommended quant for your GPU
   - **✓** / **·** — whether the file fits in your VRAM
   - quant tag, size, and filename (multi-part GGUFs show a `[N parts]` tag and download every shard)
5. Use the arrow keys to highlight a row (or accept the recommendation) and press **Add**.
6. A progress bar appears while the file (and any companions — mmproj / vocoder / VAE / encoders) downloads from Hugging Face. Once everything is saved to disk the dialog closes immediately and the dashboard shows "reloading daemons…" while llama-swap picks up the new model in the background — the dialog itself never sits frozen waiting on that restart.

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

Vision can also be turned **off** per model (Configure (`c`) → **Vision /
image input** → `no`) to serve it text-only — useful because it re-enables
DFlash/MTP speculative decoding, which can't run alongside image input (see
the [vision-model caveat](#vision-model-caveat)).

## Text-to-speech models

Select the **Text-to-speech** kind in the Add Model dialog and paste a repo.
Three model families are supported:

- **Kokoro-82M** (fast, tiny, CPU): paste `hexgrad/Kokoro-82M` (or the ONNX
  repo `onnx-community/Kokoro-82M-v1.0-ONNX` — the official repo resolves to
  it automatically) and pick a precision variant (`F32` is the reference
  quality; all are tiny). The ~50 voice style vectors are downloaded and
  bundled automatically.
- **Orpheus-3B** (most natural, emotional; GPU): paste any Orpheus-family
  GGUF repo — e.g. `unsloth/orpheus-3b-0.1-ft-GGUF` — and pick a quant like
  any LLM. Orpheus repos are detected by name and the small SNAC audio
  decoder is fetched automatically. Finetuned Orpheus voices from the Hub
  work the same way — paste the finetune's GGUF repo.
- **OuteTTS-style GGUF** repos that ship a **WavTokenizer / vocoder** GGUF
  alongside the model. The vocoder companion is auto-detected and required —
  a TTS pick with no vocoder in the repo is rejected with a clear error.
  (Pasting such a repo with the **Chat / LLM** kind selected also works; the
  vocoder marks it as TTS.)

Either way the model is marked `♪ [tts]` in the dashboard and exposed on the
same gateway at `/v1/audio/speech`:

```bash
curl http://localhost:9001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-82m-v1.0-f32", "input": "Hello from inferhost.", "voice": "af_heart"}' \
  --output speech.wav
```

```python
# OpenAI Python SDK
client.audio.speech.create(model="kokoro-82m-v1.0-f32", input="Hello.", voice="af_heart")
```

`voice` is **required** when calling through the gateway (OpenAI/LiteLLM
mandate it). For Kokoro it selects one of the bundled voices — `af_heart`,
`am_michael`, `bf_emma`, `jf_alpha`, ... (`af_*` = American female, `bm_*` =
British male, and so on; the prefix also picks the language). For Orpheus
it's one of `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`.
OpenAI preset names (`alloy`, `nova`, `echo`, ...) are mapped to the closest
equivalent on either engine, and an unknown name falls back to the default
voice (`INFERHOST_TTS_VOICE`, default `af_heart`) instead of erroring. The
optional OpenAI `speed` field (0.5–2.0) is honored for Kokoro. For OuteTTS
models, `voice` is ignored unless it's a path to a `llama-tts` speaker file.

How it works and what to expect:

- **Kokoro** is synthesized in-process via
  [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (ONNX Runtime on
  CPU — an 82M model synthesizes faster than realtime and takes no VRAM from
  your chat models). The model loads once and stays resident, so requests
  after the first are fast. **Pinning** a Kokoro model pre-loads it when the
  daemon starts, so even the first request is fast.
- **Orpheus** runs its GGUF under `llama-server` behind **llama-swap**, like
  a chat model — it swaps VRAM with your LLMs, shows a live load-state dot,
  can be loaded/unloaded from the dashboard, and **pinning works exactly
  like a chat model's** (kept in VRAM, re-warmed by pinwatch after
  evictions/reboots). The `inferhost-tts` daemon prompts it, then decodes
  the generated SNAC audio tokens on CPU. Supports inline emotion tags in
  the text: `<laugh>`, `<sigh>`, `<chuckle>`, `<gasp>`, `<yawn>`, `<cough>`.
- **OuteTTS** runs through llama.cpp’s standalone `llama-tts` binary (bundled
  automatically). It has no resident-server mode, so the model reloads on
  every request — a few seconds of overhead per call. (This is also why
  pinning is not available for OuteTTS models.)
- A small `inferhost-tts` daemon serves the endpoint; `inferhost start` brings
  it up automatically whenever a TTS model is registered
  (`INFERHOST_TTS_PORT`, default `9092`). LiteLLM routes the gateway’s
  `/v1/audio/speech` to it.
- Output is **WAV** (24 kHz mono).
- Kokoro and OuteTTS don’t run under llama-swap and don’t occupy VRAM;
  Orpheus does both.

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

## Speculative decoding (MTP / NextN models)

inferhost **auto-detects** whether a model ships MTP/NextN draft heads by reading
the GGUF metadata (`*.nextn_predict_layers`) — not by guessing from the filename.
When the heads are present it enables stacked speculative decoding automatically
(MTP draft at `--spec-draft-n-max 2` by default, plus ngram-mod); when they're
absent it stays off, so a non-MTP model is never force-fed an MTP context (which
would make llama-server abort with "model doesn't contain MTP layers").

Models with `mtp` in the filename (e.g. `qwen3.6-27b-heretic-mtp-q5-k-m`) get
two speculative-decode lanes stacked automatically:

- **`--spec-type draft-mtp`** uses the MTP heads baked into the GGUF.
- **`--spec-type ngram-mod`** uses pattern lookup over the already-generated
  text.

MTP wins on novel generation, ngram-mod dominates on repeated patterns (code,
function names, repeated constructs). All four knobs are tunable via
`INFERHOST_SPEC_*` env vars (see [Configuration](configuration.md)).

> **Vision models:** if the model also has a vision projector (`--mmproj`), the
> MTP draft lane is suppressed and only `ngram-mod` runs — draft-based
> speculation can't decode image batches. See
> [Vision-model caveat](#vision-model-caveat) below.

## DFlash speculative decoding (draft models)

**DFlash** is a different flavour of speculative decoding: instead of using
draft heads *baked into* the target GGUF (MTP), you attach a small, separate
**draft model** — a z-lab block-diffusion model trained to predict several of
the target's next tokens per step, which the big model then verifies in one
pass. When acceptance is high, you get the big model's quality at a fraction of
the wall-clock time. Nothing to compile: DFlash is served by the same
`llama-server` (upstream since build **b9831**) via
`--model-draft <draft.gguf> --spec-type draft-dflash --spec-draft-n-max N`.

A draft is a **per-model attachment** (like a vision projector), not a separate
model in the registry. The ⚡ tag in the sidebar marks a model that has one.

### Supported pairings (auto-download)

inferhost ships a table of published community draft GGUFs. When the target
matches, the draft downloads and wires itself up with one keypress:

| Target family | Draft repo | Notes |
|---|---|---|
| Qwen3.6-27B | `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` | dense |
| Qwen3.6-35B-A3B | `Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp` | MoE — smaller speedup |
| Gemma-4-31B | `Alittlehammmer/gemma-4-31B-it-DFlash-GGUF-llama.cpp` | dense |
| Gemma-4-26B-A4B | `Alittlehammmer/gemma-4-26B-A4B-it-DFlash-GGUF-llama.cpp` | MoE — smaller speedup |
| Gemma-4-12B | `williamliao/gemma-4-12B-it-DFlash-GGUF` | dense |
| Qwen3.5-27B | `AtomicChat/Qwen3.5-27B-DFlash-GGUF` | dense |
| Qwen3-Coder-30B-A3B | `AtomicChat/Qwen3-Coder-30B-A3B-DFlash-GGUF` | MoE — smaller speedup |
| Qwen3.5-9B | `Anbeeld/Qwen3.5-9B-DFlash-GGUF` | dense |

For a Mixture-of-Experts target (`…-A3B` / `…-A4B`), only a few billion params
are active per token, so it's already cheap per step — DFlash still helps but
buys a smaller speedup than on a dense model of similar total size.

### Enabling it

- **Press `f`** on a highlighted chat model. If it has a known pairing, the
  best-fitting draft quant downloads in the background and attaches; the daemons
  reload and the ⚡ tag appears. On a model with no pairing (or one that already
  has a draft), `f` just tells you so — nothing destructive happens.
- **Configure (`c`) → Suggest / Browse / Clear.** *Suggest* appears for paired
  targets and does the same as `f` but with a progress bar. *Browse* lets you
  paste **any** DFlash draft repo URL and pick the file yourself (the fallback
  for newly released drafts not yet in the table). *Clear* detaches the draft.
  If you paste an official z-lab draft repo (raw safetensors for vLLM/SGLang,
  no GGUFs — e.g. `z-lab/Qwen3.5-27B-DFlash`), Browse auto-redirects to the
  known paired GGUF conversion (e.g. `AtomicChat/Qwen3.5-27B-DFlash-GGUF`)
  instead of coming back empty.

The draft attaches over auto-detected MTP: if you attach a DFlash draft to an
MTP-capable model, DFlash is used (they're alternative drafting strategies for
the same model — you wouldn't run both). The `ngram-mod` lane still stacks on
top of either.

### Thinking-mode caveat

DFlash acceptance **drops sharply (~5–14%) with reasoning on** — the draft's
block-diffusion predictions diverge once the target starts a long chain of
thought. If you rely on DFlash speed, run the model with **reasoning off**
(Configure → Reasoning → `off`). inferhost surfaces a notice when a draft is
attached to a model whose effective reasoning is `on`.

### Vision-model caveat

**Draft-based speculative decoding cannot run on a vision model** (one with an
`--mmproj` projector attached). Once the target expands an image placeholder
into its image tokens, the draft context is asked to decode at sequence
positions it never saw, and `llama-server` aborts *every* image request with:

```
decode() failed: failed to process speculative batch
```

This is a known upstream limitation of `llama.cpp` — it applies to **both** the
external DFlash draft ([#17066](https://github.com/ggml-org/llama.cpp/issues/17066),
[#19712](https://github.com/ggml-org/llama.cpp/issues/19712)) **and** the
in-model MTP heads ([#22867](https://github.com/ggml-org/llama.cpp/issues/22867)).

So when a model has an mmproj attached, inferhost automatically **suppresses the
DFlash/MTP draft lane** and serves the model with the model-free **`ngram-mod`**
lane only (which verifies drafted tokens inline in the main context, so it is
unaffected). Text is a little slower on novel generation than with a full draft
lane, but **image requests always work**. The draft stays attached in the
registry (harmless — a future `llama.cpp` may lift the limitation) and inferhost
emits a notice plus a caveat on the model's details pane. To detach it entirely,
Configure (`c`) → Clear.

**Prefer draft speed over image input?** You can make the trade per model:
Configure (`c`) → **Vision / image input** → `no`. The model is then served
**text-only** (no `--mmproj` on the command line) and the DFlash/MTP draft lane
switches back on. The projector file stays attached and downloaded, so setting
it back to `yes` restores image input instantly — the dashboard shows a
`vision off` marker while the toggle is off, and the gateway stops advertising
`supports_vision` for the model so clients won't send it images.

### Older binaries

DFlash needs `llama-server` ≥ **b9831**. On first start, inferhost's version
gate re-fetches once if your installed build is older (unless you run a custom
`INFERHOST_LLAMA_SERVER_PATH`, which is never overwritten). If the running
binary still doesn't advertise `draft-dflash`, inferhost emits a notice and
serves the model **without** the draft rather than rendering a command that
would abort the swap entry — so nothing breaks, you just don't get the speedup
until the binary is updated.

### VRAM

An attached draft is co-resident with the target, so its weights (0.4–2.1 B,
typically well under 2 GiB) count toward the VRAM estimate and pin-feasibility
check. inferhost folds `draft_size_gib × 1.1` into the estimate automatically.

### Tuning the draft depth

`--spec-draft-n-max` controls how many tokens the draft proposes per step.
The global default is `INFERHOST_SPEC_DFLASH_N_MAX=4` (3–4 is the consumer-GPU
sweet spot; big GPUs can push it to 15–16). Per model, Configure → **DFlash
draft tokens** overrides it — `0` disables the DFlash lane for that model
without detaching the draft.

## Pinning models (load into VRAM immediately)

Press **`P`** on a highlighted model to pin it. Pinning:

1. **Immediately loads the model into VRAM** — it does not wait for a client request.
2. **Checks VRAM first.** If the model would exceed available VRAM, inferhost shows a modal: "Not enough VRAM — unpin another model first."
3. Pinned models are co-resident: they share a llama-swap group with `swap: false` so they stay loaded together instead of unloading each other.

Press **`P`** again on a pinned model to **unpin and unload** it.

The sidebar marks pinned models with a `★`. The details panel shows `loading: ★ pinned (co-resident)`.

**Pins come back on their own.** A pinned model can still leave VRAM
temporarily — a swappable model too big to co-fit evicts it while it runs, a
llama-server crash kills it, or a daemon restart/reboot brings llama-swap up
cold. The `inferhost-pinwatch` daemon (started and stopped automatically with
llama-swap) watches for exactly this: as soon as the pin is missing **and** no
swappable model is using the GPU, it loads the pin back. It never preempts a
model that's currently resident — the guest keeps the GPU until it idles out,
then the pin returns. Poll interval is `INFERHOST_PINWATCH_POLL_S` (default
10s); its log is `~/.local/share/inferhost/logs/inferhost-pinwatch.log`.

## Changing ports, context, or GPU layers

Press **`p`** to open the Settings panel. You can edit:

| Field | What it does |
|---|---|
| llama-swap port | Port for llama-swap (default `9090`, bound on `0.0.0.0`) |
| Gateway port | The LiteLLM user-facing endpoint port (default `9001`) |
| Default context | Context window for newly added models — tokens **one request** may use (prompt + reply) |
| GPU layers (-ngl) | `99` = offload everything, `0` = CPU only |
| Flash attention | `on`, `off`, or `auto` |
| Parallel slots (--parallel) | Concurrent request slots per llama-server instance. `1` (default) = serial. Each slot holds its own full context window, so `n` slots cost `n ×` the KV cache VRAM. |

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
