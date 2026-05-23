#!/usr/bin/env bash
# In-container smoke test for inferhost v0.5+.
# Exits non-zero on hard failures; soft-fails (warns, returns 0) on the
# llama-server download step if the CI hasn't published a llama-v* release yet.

set -euo pipefail
shopt -s lastpipe

pass() { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; exit 1; }
section() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

section "Environment"
python --version | sed 's/^/  /'
python -c "import sys; print('  prefix:', sys.prefix)"

section "GPU visibility (via NVIDIA Container Toolkit)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        | while read -r line; do pass "$line"; done
else
    warn "nvidia-smi not in PATH — GPU features will be skipped"
fi

section "Imports"
python - <<'PY'
from inferhost.core import binaries, configs, registry, processes, vram
from inferhost.tui.screens import dashboard, model_settings, warning, add_model, settings as settings_screen
from inferhost.settings import settings
s = settings()
assert s.kv_quant_k == "q8_0", f"expected K=q8_0, got {s.kv_quant_k}"
assert s.kv_quant_v == "turbo3", f"expected V=turbo3, got {s.kv_quant_v}"
assert binaries.LLAMACPP_REPO == "amirrouh/inferhost", f"wrong repo: {binaries.LLAMACPP_REPO}"
assert hasattr(processes, "force_load_model")
assert hasattr(processes, "force_unload_model")
assert hasattr(vram, "can_pin")
print("  all imports + contracts OK")
PY
pass "imports + v0.5 contracts present"

section "Unit tests"
cd /workspace && pytest tests/ -v --tb=short 2>&1 | tail -8

section "CLI smoke (status before any install)"
python -m inferhost._ops status 2>&1 | sed 's/^/  /' || warn "status returned non-zero (expected before install)"

section "Prebuilt llama-server availability"
python - <<'PY' || warn "no llama-v* release yet — CI may still be building; retry with: docker compose run --rm inferhost inferhost-smoke"
from inferhost.core import binaries
try:
    rel = binaries._llamacpp_release_json("latest")
    tag = rel.get("tag_name", "?")
    n_assets = len(rel.get("assets", []))
    print(f"  found release {tag} with {n_assets} asset(s)")
except Exception as e:
    raise SystemExit(f"no release: {e}")
PY

section "Done"
pass "v0.5 container smoke complete"
