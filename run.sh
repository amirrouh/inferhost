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
                  on the first launch of the TUI. inferhost pulls llama-server
                  straight from upstream ggml-org/llama.cpp releases — no
                  cmake required. The variant (Vulkan / ROCm / SYCL / OpenVINO /
                  CPU / Metal) is picked by hardware probe and overridable via
                  INFERHOST_LLAMACPP_BACKEND or the TUI Settings screen. Set
                  INFERHOST_LLAMA_SERVER_PATH to use a custom binary instead
                  (e.g. a self-built CUDA llama-server).
                  Exception: the optional Qwen3-TTS engine (qwen3-tts.cpp) has
                  no prebuilt release and IS built from source — on demand,
                  the first time you add a Qwen3-TTS model — which needs
                  git/cmake/a C++ compiler on the host. Every other engine
                  (llama-server, llama-swap, sd-server, llama-tts) stays
                  prebuilt-binary-only, no compiling.
  start           Launch the TUI (alias of `run`). The TUI is the only UI.
  run             Launch the TUI.
  start-bg        Start llama-swap + LiteLLM gateway as background daemons,
                  no TUI. They survive your shell session — kill them with
                  `./run.sh stop`. Requires at least one model already
                  registered (use the TUI to add one the first time). If a
                  text-to-speech model is registered, the inferhost-tts daemon
                  (serves /v1/audio/speech) is started too. Image-generation
                  models (stable-diffusion.cpp) ride llama-swap and serve
                  /v1/images/generations automatically — no extra daemon.
  stop            Stop llama-swap, the LiteLLM gateway, and inferhost-tts.
  restart         Stop + start-bg in one shot. Picks up any config edits.
  status          Print daemon + endpoint status (no UI). Note: llama-swap
                  (swap) listens on 127.0.0.1 by default (internal); the
                  user-visible gateway is the LiteLLM port (INFERHOST_GATEWAY_PORT).
                  To expose llama-swap externally, set INFERHOST_SWAP_HOST=0.0.0.0.
  uninstall       Remove the venv and runtime data dir (keeps HF model cache).
  reset           Stop daemons and clear the model registry / generated configs.
  test            Run pytest.
  lint            Run ruff over src/ and tests/.
  shell           Open a shell with the venv activated.
  docker-build       Build the inferhost-test:v0.5 image (CUDA 12.4 base, GPU-ready).
  docker-smoke       In-container static smoke (imports, pytest, GPU visibility,
                     release availability). Fast — no model download. Requires
                     the NVIDIA Container Toolkit on the host (--gpus all).
  docker-functional  Full end-to-end test: downloads a tiny GGUF, registers it,
                     starts llama-swap, sends a real chat completion through
                     llama-server, and verifies the live process has the default
                     -ctk q8_0 -ctv q8_0 KV flags. Model is cached in a named
                     volume so subsequent runs are offline.
  docker-test        Run pytest inside the container.
  docker-shell       Drop into a bash shell in the running container.
  docker-clean       Stop the test container and remove its named volumes.
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

COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.test.yml"
DOCKER_SVC="inferhost"

docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "${COMPOSE_FILE}" "$@"
  else
    docker-compose -f "${COMPOSE_FILE}" "$@"
  fi
}

docker_ensure_running() {
  if ! docker_compose ps --status running --services 2>/dev/null | grep -qx "${DOCKER_SVC}"; then
    echo ">>> Starting test container"
    docker_compose up -d
  fi
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  install)         install_dev ;;
  uninstall)       uninstall_local ;;
  start|run|tui)   ensure_venv; inferhost ;;
  start-bg)        ensure_venv; python -m inferhost._ops start ;;
  stop)            ensure_venv; python -m inferhost._ops stop ;;
  restart)         ensure_venv; python -m inferhost._ops restart ;;
  status)          ensure_venv; python -m inferhost._ops status ;;
  reset)           reset_state ;;
  test)            ensure_venv; pytest -v "$@" ;;
  lint)            ensure_venv; ruff check src tests "$@" ;;
  shell)           ensure_venv; exec "${SHELL:-bash}" ;;
  docker-build)      docker_compose build "$@" ;;
  docker-smoke)      docker_ensure_running; docker_compose exec "${DOCKER_SVC}" inferhost-smoke ;;
  docker-functional) docker_ensure_running; docker_compose exec "${DOCKER_SVC}" inferhost-functional ;;
  docker-test)       docker_ensure_running; docker_compose exec "${DOCKER_SVC}" pytest tests/ -v "$@" ;;
  docker-shell)      docker_ensure_running; docker_compose exec "${DOCKER_SVC}" bash ;;
  docker-clean)      docker_compose down -v --remove-orphans ;;
  help|-h|--help|"") usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 2 ;;
esac
