<div align="center">

# 🛰️ inferhost

**Your own private, multi-modal AI server — one command, any GPU, no compiling.**

[![PyPI](https://img.shields.io/pypi/v/inferhost?color=blue)](https://pypi.org/project/inferhost/)
[![Python](https://img.shields.io/pypi/pyversions/inferhost)](https://pypi.org/project/inferhost/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-purple)](https://amirrouh.github.io/inferhost/)

**Chat · Vision · Speech · Image generation** — all behind one OpenAI-compatible endpoint.

<img src="https://raw.githubusercontent.com/amirrouh/inferhost/master/docs/assets/screenshot.png" width="820" alt="inferhost dashboard">

</div>

---

inferhost turns any GPU box into a private AI server. It wraps **llama.cpp** and **stable-diffusion.cpp** behind a single **OpenAI-compatible endpoint** — pulls the official upstream binaries for you (nothing to compile), **auto-fetches the right model files** when you paste a Hugging Face link, and **hot-swaps models in and out of VRAM** so one card can serve a big LLM *and* image generation. You only ever touch a keyboard-driven dashboard (and an optional `.env`).

## ⚡ Quick start

```bash
uv tool install inferhost      # or:  pipx install inferhost  /  pip install inferhost
inferhost                      # opens the dashboard — press 'a' to add a model
```

That's the whole setup. First launch fetches the runtime binaries automatically. To add a model, press **`a`** and **paste a Hugging Face repo** — inferhost lists the files, downloads what's needed, and serves it. Then call it like OpenAI:

<table>
<tr>
<th>🗣️ Chat / LLM</th>
<th>🔊 Text-to-speech</th>
<th>🎨 Image generation</th>
</tr>
<tr valign="top">
<td>paste<br><code>Qwen/Qwen2.5-7B-Instruct-GGUF</code></td>
<td>paste<br><code>OuteAI/OuteTTS-0.2-500M-GGUF</code></td>
<td>paste<br><code>OlegSkutte/sdxl-turbo-GGUF</code></td>
</tr>
</table>

```bash
# 🗣️  Chat  →  /v1/chat/completions
curl http://localhost:9001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<name-from-dashboard>","messages":[{"role":"user","content":"Hello!"}]}'

# 🔊  Speech  →  /v1/audio/speech   (returns WAV)
curl http://localhost:9001/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"<name>","input":"Hello from inferhost.","voice":"default"}' --output speech.wav

# 🎨  Image  →  /v1/images/generations   (returns base64 PNG)
curl http://localhost:9001/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"model":"<name>","prompt":"a red apple on a table","size":"512x512"}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

Everything lives on `http://localhost:9001/v1` — point any OpenAI client (Python SDK, Open WebUI, your app) at it. The model name is whatever shows in the dashboard.

## ✨ Why inferhost

- **One endpoint, every modality** — chat, vision, speech, and images on the same OpenAI-compatible `:9001`. No per-model servers to wire up.
- **Nothing to compile** — official `llama-server` / `sd-server` binaries are pulled from upstream for your hardware (NVIDIA Vulkan, ROCm, SYCL, CPU, Apple Metal).
- **Paste a link, it figures out the rest** — picks the best quant for your VRAM, and for multi-file models (Flux, Z-Image, Qwen-Image) **auto-downloads the right VAE + text encoders** from known-good repos.
- **One GPU, many models** — llama-swap lazy-loads and hot-swaps models in/out of VRAM on demand, so a 24 GB card serves a 27B LLM *and* Flux image generation.
- **TUI or headless** — drive everything from a keyboard dashboard, or run `inferhost start/stop/status` on a server with no terminal.
- **Tuned by default** — q8_0 KV-cache compression, stacked MTP + ngram speculative decoding, and honest context windows, all overridable from a `.env`.

## 🧩 Supported models

| Modality | Models | How |
|---|---|---|
| **Chat / Vision** | any GGUF LLM (Qwen, Llama, Gemma, DeepSeek…), vision via `mmproj` | paste repo → pick quant |
| **Speech (TTS)** | OuteTTS | paste repo (vocoder auto-detected) |
| **Image — single-file** | SD 1.5, SDXL (incl. Turbo) | paste repo → pick file |
| **Image — Flux.1** | schnell / dev | auto-fetches VAE + CLIP-L + T5XXL |
| **Image — Flux.2 Klein** | incl. **Bonsai-Image** (1-bit) | auto-fetches VAE + Qwen3-4B |
| **Image — Z-Image** | Z-Image-Turbo | auto-fetches VAE + Qwen3-4B |
| **Image — Qwen-Image** | Qwen-Image / **Qwen-Image-Edit** | auto-fetches VAE + Qwen2.5-VL + mmproj |

All image families above were verified end-to-end on a Vulkan GPU (SDXL-Turbo ~2 s, Flux-schnell ~4 s, Bonsai ~2 s, Z-Image-Turbo ~11 s, Qwen-Image-Edit via `/v1/images/edits`).

## 📚 Documentation

Full guides live in **[docs](https://amirrouh.github.io/inferhost/)** (and in the [`docs/`](docs/) folder):

- **[Installation](docs/installation.md)** — install, upgrade, uninstall, requirements
- **[Usage](docs/usage.md)** — the dashboard, keyboard keys, and chat / TTS / image / Flux / Z-Image / Qwen-Image walkthroughs
- **[Configuration](docs/configuration.md)** — every `.env` variable, KV-cache quant, custom binaries
- **[Troubleshooting](docs/troubleshooting.md)** — ports, tmux mouse, common errors

<details>
<summary><b>🏗️ Architecture</b></summary>

```
Your app ──HTTP──▶  LiteLLM gateway        llama-swap (loopback)       llama-server  (chat/vision)
                    :9001 (public)   ──▶    127.0.0.1:9090      ──┬──▶  sd-server     (images)
                                                                  └──▶  inferhost-tts (speech)
```

- **llama.cpp** (`llama-server`) runs chat/vision inference — official upstream binary, backend auto-detected.
- **llama-swap** fronts the model backends and lazy-loads / hot-swaps them on demand (loopback only). Image models (`sd-server`) ride here too, so they swap VRAM with LLMs.
- **inferhost-tts** wraps llama.cpp's `llama-tts` behind `/v1/audio/speech` (started only when a TTS model is registered).
- **LiteLLM** is the single always-on public gateway on `:9001`, routing each request to the right backend.

The extra engines (`llama-tts`, `sd-server`) are fetched automatically the first time you add a model that needs them.

</details>

<details>
<summary><b>🛠️ Development</b></summary>

The repo ships a `run.sh` wrapper for source-tree work (end users never need it — they only type `inferhost`):

```bash
git clone git@github.com:amirrouh/inferhost.git && cd inferhost
./run.sh install     # venv + editable install
./run.sh start       # launch the TUI (downloads binaries on first run)
./run.sh status      # headless status
./run.sh stop        # stop daemons
./run.sh test        # pytest
```

Run `./run.sh help` for the full list.

</details>

## License

[Apache 2.0](LICENSE)
