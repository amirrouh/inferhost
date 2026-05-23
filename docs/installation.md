---
layout: default
title: Installation
---

[← Back to overview](index.md)

# Installation

## System requirements

| | |
|---|---|
| **Python** | 3.11, 3.12, or 3.13 |
| **OS** | Linux or macOS |
| **GPU (optional)** | NVIDIA (CUDA / Vulkan), AMD (ROCm), Intel (SYCL / OpenVINO), Apple Silicon (Metal) |
| **RAM** | depends on the model you want to run (a 7B model in Q4 is ~5 GB) |

CPU-only is fully supported — it'll just be slower.

## Install (uv, recommended)

[uv](https://github.com/astral-sh/uv) installs `inferhost` into its own isolated environment and puts it on your `PATH` as a normal command:

```bash
uv tool install inferhost
```

## Install with the LiteLLM gateway

The optional gateway adds friendly aliases, routing, and rate limits across many providers. Install the `[gateway]` extra with whichever installer you prefer:

```bash
uv tool install 'inferhost[gateway]'
# or
pipx install 'inferhost[gateway]'
# or, inside an existing venv:
pip install 'inferhost[gateway]'
```

## Install (pipx)

If you already use [pipx](https://pipx.pypa.io/) for global CLI apps:

```bash
pipx install inferhost
```

## Install (pip)

`pip install inferhost` works too, but only inside an existing virtual environment — if you run it on the system Python you'll likely hit PEP 668 (`externally-managed-environment`). Prefer `uv tool` or `pipx` for a global install.

```bash
pip install inferhost
```

## ⚠️ Don't use `uv add inferhost`

`uv add` adds a package as a **project dependency**, meaning:

- It edits whatever `pyproject.toml` is in your current directory
- The `inferhost` command is only available via `uv run inferhost` from inside that project
- Upgrades go through `uv lock --upgrade-package inferhost && uv sync`

`inferhost` is a CLI app you launch, not a library you `import` from your code, so the right tool is `uv tool install` (or `pipx install`).

If you've already done `uv add inferhost`, switch over with:

```bash
uv remove inferhost              # from inside the project you ran `uv add` in
uv tool install inferhost        # then, from anywhere
```

## Upgrade

```bash
uv tool upgrade inferhost                # if installed with `uv tool`
pipx upgrade inferhost                   # if installed with pipx
pip install -U inferhost                 # if installed with pip (inside the venv)
```

Pin to a specific version:

```bash
uv tool install --force 'inferhost==0.4.13'
```

Check the installed version:

```bash
uv tool list | grep inferhost
```

## Uninstall

Remove the package:

```bash
uv tool uninstall inferhost              # if installed with `uv tool`
pipx uninstall inferhost                 # if installed with pipx
pip uninstall inferhost                  # if installed with pip
```

Inferhost keeps runtime files **outside** the Python install. To remove the runtime binaries, logs, PID files, and the model registry, also run:

```bash
rm -rf ~/.local/share/inferhost          # llama-server / llama-swap binaries, logs, PIDs
rm -rf ~/.config/inferhost               # model registry + generated llama-swap.yaml / litellm.yaml
```

Downloaded GGUFs live in the Hugging Face cache (`~/.cache/huggingface/hub/`) and are **not** removed by the steps above. They're reusable by any other Hugging Face tool, so most people leave them alone. To delete them anyway:

```bash
rm -rf ~/.cache/huggingface/hub/models--*
```

## First launch

```bash
inferhost
```

On the very first launch, inferhost downloads two runtime binaries to `~/.local/share/inferhost/bin/`:

- **llama-server** — from the upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) project, in whichever GPU backend matches your hardware.
- **llama-swap** — the lazy-loading proxy from [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap).

You'll see a progress bar for each. After that, the dashboard opens and you're ready to add a model.

## Choosing the GPU backend

inferhost auto-detects the best backend for your hardware. If you want to pin it explicitly, set an environment variable before launching:

```bash
export INFERHOST_LLAMACPP_BACKEND=cuda   # or vulkan, rocm, sycl, openvino, cpu
inferhost
```

See the [Configuration](configuration.md) page for the full list.

## Verify

After the install screen, the dashboard's top bar shows the live endpoint, e.g.:

```
● llama-swap http://localhost:9090/v1
```

The green ● means the daemon is up. Press **`a`** to add your first model.

[Continue to Usage →](usage.md)
