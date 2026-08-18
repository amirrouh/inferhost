# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Responses
- **Be extremely brief.** Answer in as few words as possible. Lead with the
  action or the answer. No preamble, no restating the question, no summarizing
  what you just did, no listing alternatives you didn't take. If the answer is a
  command, output the command and nothing else. Expand only when asked.

## "Fix it" means fix it
- When I say **fix**, do the fix. Don't hand it back to me as something "to
  flag", "to note", or "worth mentioning" — that's just the work relabelled as
  a status update.
- This includes problems **you** created. If you notice you did something
  wrong, repair it in the same turn, then tell me it's repaired. Don't report
  your own mistake and leave it sitting there.
- Only stop and ask when fixing it would destroy something I can't get back, or
  when there are genuinely two reasonable fixes and picking wrong wastes real
  work. "This is a bit disruptive" is not a reason to ask — do it.
- Tell me what you fixed after it's done, in one line.

## Commits
- **Do NOT add Claude as an author or co-author.** Never append a
  `Co-Authored-By: Claude ...` trailer (or any "Generated with Claude" line) to
  commit messages or PR descriptions. Commits are authored solely by the repo
  owner. Write a normal commit message and stop — no attribution trailer.
- Only commit/push when explicitly asked.

## Commands

`run.sh` is the supported entry point and wraps a uv venv:

```bash
./run.sh test                      # pytest -v
./run.sh test -k vision            # single test / pattern
./run.sh lint                      # ruff check src tests
./run.sh install                   # dev install into the venv
./run.sh start-bg | stop | restart | status
./run.sh update [bNNNN] [--rebuild]  # refresh upstream binaries
./run.sh reset                     # wipe local state
./run.sh docker-smoke | docker-functional | docker-test
```

CI runs exactly `uv run ruff check src tests` and `uv run pytest -v` on
3.11–3.13. Working directly with uv (`uv run pytest -q`, `uv run ruff check
src tests`) is equivalent and faster in a loop.

**Release**: bump `version` in `pyproject.toml`, `uv lock`, commit, then push a
`vX.Y.Z` tag — `.github/workflows/publish.yml` publishes to PyPI via trusted
publishing on any `v*` tag.

## Architecture

inferhost **never performs inference**. It is a control plane that downloads
upstream binaries (`llama-server`, `llama-swap`, `sd-server`), renders their
config files, and supervises them as daemons. Almost every feature is therefore
either *config generation* or *binary capability probing* — not serving code.

**The pipeline.** `models.toml` → `registry.Model` → `configs.render_llama_swap()`
/ `render_litellm()` → `~/.config/inferhost/{llama-swap,litellm}.yaml` → daemons.
Nothing is served that isn't rendered into those files, so behavior changes go in
`core/configs.py`. `_llama_server_cmd()` builds the argv for one model and is
where context, KV quant, MoE offload, vision, and speculative decoding converge.

**Two entry points.** `cli.py` launches the Textual TUI (`tui/`); `_ops.py` holds
the headless verbs (`start`/`stop`/`update`/`prune`/`autostart`) and is what both
the CLI and `run.sh` dispatch into. TUI screens do work in `@work(thread=True)`
workers and must not block the event loop.

**Binary capability probing** is the load-bearing pattern. The package and the
upstream binaries version independently, so a box can hold a llama.cpp that
predates a model. Rather than maintain version tables, inferhost asks the binary
on disk what it can do, and every probe **fails open** (assume supported) so a
failed probe never blocks a config that would have worked:

- `binaries.binary_supports_arch()` / `binary_supports_ggml_type()` scan the
  executable and its shared libraries for C string literals. Architecture names
  live in `libllama`, ggml type names (`nvfp4`, `mxfp4`) in `libggml-base` —
  the two probes pass different glob sets to the shared scanner for that reason.
- `llama_caps.supports_spec_type()` / `supported_cache_types()` parse
  `llama-server --help`.
- `configs.server_binary()` routes a single model to the managed binary when the
  user's custom build can't read its architecture or weight format.
- `_ops._heal_unknown_architectures()` fetches a newer llama.cpp at startup when
  no binary on the box can load a registered model.

**Custom binaries.** `INFERHOST_LLAMA_SERVER_PATH` points at a self-compiled
llama-server (necessary for CUDA on Linux — upstream ships no CUDA tarball for
it). `update` then won't overwrite it; `update --rebuild` recompiles it in place,
reusing the flags in its existing `CMakeCache.txt`. Custom builds must be
statically linked: the generated llama-swap config pins `LD_LIBRARY_PATH` to
inferhost's `bin/`, whose managed `libggml*.so` would otherwise shadow theirs.

**Settings** are env-only via pydantic-settings (`settings.py`), read from `.env`
in the cwd and then `<config_dir>/inferhost.env` (TUI-managed overrides, which
win). Per-model overrides on `Model` use sentinels — `-1`, `""`, `0` mean
"inherit the global". Tests must call `reload_settings()` after mutating env.

**Model kinds** share one registry but not one engine. `Model.kind` is
`chat`/`image`/`tts`; chat and Orpheus TTS are fronted by llama-swap, other TTS
runs in the `tts_serve.py` daemon, and image models run on `sd-server`. The TTS
engine is inferred from the `vocoder_path` extension, not stored.

**Speculative decoding** picks one lane per model in `_llama_server_cmd()`:
DFlash (external draft) > MTP (heads baked into the GGUF) > none, with the
model-free `ngram-mod` lane stacked on top. Vision (`--mmproj`) suppresses only
the external-draft lane. The ordering and the vision interaction are heavily
commented there because both are upstream-behavior-dependent and have changed.

## Testing

Tests are pure Python — no network, no daemons, no GPU. They monkeypatch
collaborators on `_ops`/`binaries`/`configs` and use `tmp_path` plus
`INFERHOST_DATA_DIR`/`INFERHOST_CONFIG_DIR` for isolation. Binary probes are
exercised against fake files containing NUL-delimited tokens rather than real
binaries.

Unit tests cannot prove that a generated argv actually runs. For anything that
changes what gets passed to a binary, run a real workload — `./run.sh
docker-functional`, or a real model on the user's own box — before calling it
verified.
