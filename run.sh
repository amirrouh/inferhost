#!/usr/bin/env bash
# inferhost local dev wrapper.
# Reads .env (if present), ensures a venv exists, and forwards subcommands to the inferhost CLI.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Load .env if present, ignoring comments/blank lines.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source <(grep -E '^[A-Z_][A-Z0-9_]*=' "${PROJECT_ROOT}/.env")
  set +o allexport
fi

usage() {
  cat <<'EOF'
inferhost run.sh — dev wrapper around the inferhost CLI.

Usage: ./run.sh <command> [args]

Top-level commands:
  install            Create venv, pip install -e ".[dev]", then download binaries.
  uninstall          Remove the local venv and runtime data (does NOT touch HF cache).
  start              Start llama-swap (and gateway if installed).
  stop               Stop all daemons.
  status             Show daemon + model status.
  doctor             Health-check: binaries, GPU, paths.
  serve <hf_repo>    Add + start serving a Hugging Face model.
  tui                Launch the dashboard.
  test               Run the test suite (pytest).
  lint               Run ruff over src/ and tests/.
  shell              Open a shell with the venv activated.
  reset              Stop daemons and clear the model registry.
  -- <args...>       Pass-through to inferhost (e.g. `./run.sh -- logs swap -f`).

Environment overrides live in .env (see .env.example for the full list):
  INFERHOST_GATEWAY_PORT  INFERHOST_SWAP_PORT
  INFERHOST_DATA_DIR      INFERHOST_CONFIG_DIR   INFERHOST_HF_CACHE
  INFERHOST_GPU_LAYERS    INFERHOST_DEFAULT_CTX  INFERHOST_FLASH_ATTENTION
  INFERHOST_LLAMACPP_VERSION   INFERHOST_LLAMASWAP_VERSION
  INFERHOST_LLAMACPP_BACKEND   (vulkan|cuda|rocm|sycl|openvino|cpu; auto by default)
EOF
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo ">>> Creating venv at ${VENV_DIR}"
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 "${VENV_DIR}"
    else
      python3 -m venv "${VENV_DIR}"
    fi
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

install_dev() {
  ensure_venv
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[dev]"
  else
    pip install -e ".[dev]"
  fi
  echo ">>> Running inferhost install (downloads llama.cpp + llama-swap binaries) ..."
  inferhost install
}

uninstall_local() {
  echo ">>> Stopping any running daemons"
  ensure_venv 2>/dev/null && inferhost stop --all 2>/dev/null || true
  echo ">>> Removing venv: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
  echo ">>> Removing runtime data: ${INFERHOST_DATA_DIR:-$HOME/.local/share/inferhost}"
  rm -rf "${INFERHOST_DATA_DIR:-$HOME/.local/share/inferhost}"
  echo ">>> Removing config: ${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}"
  rm -rf "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}"
  echo "Done. (Hugging Face model cache at ${INFERHOST_HF_CACHE:-$HOME/.cache/huggingface} was kept.)"
}

reset_state() {
  ensure_venv
  inferhost stop --all || true
  rm -f "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}/models.toml"
  rm -f "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}/llama-swap.yaml"
  rm -f "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}/litellm.yaml"
  echo "Registry cleared. (Model files remain in HF cache.)"
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  install)    install_dev ;;
  uninstall)  uninstall_local ;;
  start)      ensure_venv; inferhost start "$@" ;;
  stop)       ensure_venv; inferhost stop "$@" ;;
  status)     ensure_venv; inferhost status "$@" ;;
  doctor)     ensure_venv; inferhost doctor "$@" ;;
  serve)      ensure_venv; inferhost serve "$@" ;;
  tui)        ensure_venv; inferhost tui ;;
  test)       ensure_venv; pytest -v "$@" ;;
  lint)       ensure_venv; ruff check src tests "$@" ;;
  shell)      ensure_venv; exec "${SHELL:-bash}" ;;
  reset)      reset_state ;;
  --)         ensure_venv; inferhost "$@" ;;
  help|-h|--help|"") usage ;;
  *)          ensure_venv; inferhost "${cmd}" "$@" ;;
esac
