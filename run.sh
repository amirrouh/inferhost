#!/usr/bin/env bash
# inferhost local dev wrapper.
# Reads .env (if present), ensures a venv exists, and routes high-level commands
# to the inferhost package. The user-facing inferhost command is TUI-only —
# this wrapper exists for repo-development convenience (install / start / stop / status).

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
inferhost run.sh — repo wrapper around the TUI-only inferhost package.

Usage: ./run.sh <command>

Commands:
  install         Create the project venv and pip install -e ".[dev]".
                  Runtime binaries (llama.cpp, llama-swap) are auto-downloaded
                  on the first launch of the TUI.
  start           Launch the TUI (alias of `run`). The TUI is the only UI.
  run             Launch the TUI.
  stop            Stop llama-swap (and the LiteLLM gateway, if running).
  status          Print daemon + endpoint status (no UI).
  uninstall       Remove the venv and runtime data dir (keeps HF model cache).
  reset           Stop daemons and clear the model registry / generated configs.
  test            Run pytest.
  lint            Run ruff over src/ and tests/.
  shell           Open a shell with the venv activated.
  help            Show this help.

Configuration lives in .env (see .env.example). Variables include:
  INFERHOST_SWAP_PORT       INFERHOST_GATEWAY_PORT
  INFERHOST_DATA_DIR        INFERHOST_CONFIG_DIR        INFERHOST_HF_CACHE
  INFERHOST_GPU_LAYERS      INFERHOST_DEFAULT_CTX       INFERHOST_FLASH_ATTENTION
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
  echo ">>> Install complete. Run './run.sh start' to launch the TUI."
  echo "    (Runtime binaries are downloaded automatically on first launch.)"
}

uninstall_local() {
  echo ">>> Stopping any running daemons"
  ensure_venv 2>/dev/null && python -m inferhost._ops stop 2>/dev/null || true
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
  python -m inferhost._ops stop || true
  rm -f "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}/models.toml"
  rm -f "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}/llama-swap.yaml"
  rm -f "${INFERHOST_CONFIG_DIR:-$HOME/.config/inferhost}/litellm.yaml"
  echo "Registry cleared. (Model files remain in HF cache.)"
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  install)         install_dev ;;
  uninstall)       uninstall_local ;;
  start|run|tui)   ensure_venv; inferhost ;;
  stop)            ensure_venv; python -m inferhost._ops stop ;;
  status)          ensure_venv; python -m inferhost._ops status ;;
  reset)           reset_state ;;
  test)            ensure_venv; pytest -v "$@" ;;
  lint)            ensure_venv; ruff check src tests "$@" ;;
  shell)           ensure_venv; exec "${SHELL:-bash}" ;;
  help|-h|--help|"") usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 2 ;;
esac
