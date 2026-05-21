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
│ a=add  n=rename  d=remove │ s/x/r=swap │ g=gateway │ p=settings      │
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

## Changing ports, context, or GPU layers

Press **`p`** to open the Settings panel. You can edit:

| Field | What it does |
|---|---|
| llama-swap port | The OpenAI-compatible endpoint port (default `9090`) |
| Gateway port | The LiteLLM gateway port (default `9001`) |
| Default context | Context window for newly added models (tokens) |
| GPU layers (-ngl) | `99` = offload everything, `0` = CPU only |
| Flash attention | `on`, `off`, or `auto` |

Saving writes a managed env file at `~/.config/inferhost/inferhost.env`, so your
changes persist across restarts of the TUI. After saving, press **`r`** to
restart llama-swap with the new values.

## Toggling the LiteLLM gateway

Press **`g`** to start (or stop) the LiteLLM gateway. The status bar at the top
shows whether it's running and on which port. The gateway is optional — install
it with `pip install 'inferhost[gateway]'` if you want a single OpenAI-compatible
endpoint that can route across multiple providers.

## Running more than one model

Add as many as you like. llama-swap loads each one on the first request and unloads it after an idle period, so you can keep dozens registered without burning VRAM.

Use the model `name` from the dashboard as the `model` field in your request — llama-swap routes it to the right backend.

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
