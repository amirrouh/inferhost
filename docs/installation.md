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
| **GPU (optional)** | see Supported platforms below |
| **RAM** | depends on the model you want to run (a 7B model in Q4 is ~5 GB) |

CPU-only is fully supported — it'll just be slower.

## Supported platforms

inferhost ships prebuilt `llama-server` binaries (from its own GitHub Actions, built from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)) for three targets:

| Target | Binary asset |
|---|---|
| **Linux x86_64 CUDA 12.x** | `llama-server-linux-x86_64-cuda` |
| **Linux x86_64 CPU** | `llama-server-linux-x86_64-cpu` |
| **macOS arm64 Metal** | `llama-server-macos-arm64-metal` |

**Vulkan / ROCm / local builds:** If you run a GPU not covered by the three targets above, set `INFERHOST_LLAMA_SERVER_PATH` to the path of your own `llama-server` binary:

```bash
export INFERHOST_LLAMA_SERVER_PATH=/usr/local/bin/llama-server
inferhost
```

inferhost will use that binary instead of downloading one. See the [Configuration](configuration.md) page for details.

## Install (uv, recommended)

[uv](https://github.com/astral-sh/uv) installs `inferhost` into its own isolated environment and puts it on your `PATH` as a normal command:

```bash
uv tool install inferhost
```

One binary, two modes:

| Invocation | What it does |
|---|---|
| `inferhost` | Launches the TUI dashboard. Add models, configure, watch logs. |
| `inferhost start \| stop \| restart \| status` | Headless control of the same daemons. No terminal required. |

**That's all.** LiteLLM is bundled — no extra needed.

> **Upgrading from v0.4?** In v0.4, LiteLLM was an optional `[gateway]` extra. From v0.5 it is included in the base package. Running `uv tool upgrade inferhost` is sufficient — you do not need `inferhost[gateway]` anymore.

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
uv tool install --force 'inferhost==0.5.0'
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

- **llama-server** — a prebuilt TurboQuant-enabled binary from inferhost's own releases, matching your platform (CUDA, CPU, or Metal).
- **llama-swap** — the lazy-loading proxy from [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap).

You'll see a progress bar for each. After that, the dashboard opens and you're ready to add a model.

## Verify

After the install screen, the dashboard's top bar shows the live endpoint:

```
● litellm http://localhost:9001/v1
```

The green ● means the LiteLLM gateway is up. Press **`a`** to add your first model.

[Continue to Usage →](usage.md)
