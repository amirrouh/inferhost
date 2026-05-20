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
│ ● llama-swap http://localhost:9090/v1                                │
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
│ a=add  s=start  x=stop  r=restart  d=remove  R=refresh  q=quit       │
└──────────────────────────────────────────────────────────────────────┘
```

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
