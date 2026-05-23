#!/usr/bin/env bash
# Functional smoke test for inferhost v0.5+ — exercises the real install
# → register → serve → inference path with a tiny GGUF, and verifies the
# actually-running llama-server has the v0.5 TurboQuant K/V flags.
#
# Runs inside the inferhost-test container (via `./run.sh docker-functional`).
# Requires --gpus all on the host (NVIDIA Container Toolkit).
#
# Model used: Qwen/Qwen2.5-0.5B-Instruct-GGUF q4_k_m (~470 MiB, downloaded
# once into the named hf-cache volume; subsequent runs are offline).
set -euo pipefail
shopt -s lastpipe

pass() { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; exit 1; }
section() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_NAME="qwen-tiny"
MODEL_PORT=8081
EXPECTED_K="q8_0"
EXPECTED_V="turbo3"

# System libs the prebuilt CPU/CUDA binaries link against. libgomp1 is part
# of glibc-runtime on almost every distro; minimal containers like ours miss it.
section "Install runtime deps (libgomp1, jq, curl)"
apt-get update -qq && apt-get install -y -qq libgomp1 jq curl >/dev/null 2>&1
pass "deps installed"

section "Install llama-server + llama-swap (from amirrouh/inferhost releases)"
python - <<'PY'
from inferhost.core.binaries import install_llama_server, install_llama_swap
ls = install_llama_server()
sw = install_llama_swap()
print(f"  llama-server: {ls.path} ({ls.version})")
print(f"  llama-swap:   {sw.path} ({sw.version})")
PY
pass "binaries installed"

section "Download tiny model (~470 MiB, cached in hf-cache volume)"
python - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="$MODEL_REPO",
    filename="$MODEL_FILE",
    cache_dir="/inferhost/hf-cache",
)
import os; print(f"  {p}  ({os.path.getsize(p)/1024/1024:.1f} MiB)")
PY
pass "model in hf-cache"

section "Register model + generate configs"
python - <<PY
from pathlib import Path
from inferhost.core import registry as reg_mod
from inferhost.core.registry import Model
from inferhost.core.configs import write_all, _llama_server_cmd

g = next(Path("/inferhost/hf-cache").rglob("$MODEL_FILE"), None)
assert g, "model not in hf-cache"
reg = reg_mod.load()
m = Model(name="$MODEL_NAME", repo_id="$MODEL_REPO", filename="$MODEL_FILE",
          local_path=str(g), port=$MODEL_PORT, ctx=2048, size_gib=0.4)
reg.add(m); reg_mod.save(reg)
swap_yaml, lite_yaml = write_all(reg)
cmd = _llama_server_cmd(m)
assert "-ctk $EXPECTED_K" in cmd, f"-ctk $EXPECTED_K not in cmd: {cmd}"
assert "-ctv $EXPECTED_V" in cmd, f"-ctv $EXPECTED_V not in cmd: {cmd}"
print(f"  registered {m.name!r} -> 127.0.0.1:$MODEL_PORT")
print(f"  generated llama-swap.yaml + litellm.yaml")
print(f"  cmd builder emits: -ctk $EXPECTED_K -ctv $EXPECTED_V  (asymmetric KV)")
PY
pass "registry + configs OK"

section "Start llama-swap (loopback only)"
python -c "from inferhost.core.processes import start_swap; print(start_swap())" >/dev/null
sleep 2
python -m inferhost._ops status | sed 's/^/  /'
pass "llama-swap running"

section "Send real chat completion (warm-loads llama-server)"
RESP=$(curl -s --max-time 120 http://127.0.0.1:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word OK and nothing else.\"}],\"max_tokens\":10,\"temperature\":0}")
CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content // empty')
TOKENS=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
if [[ -z "$CONTENT" ]]; then
    fail "no content in response: $(echo "$RESP" | head -c 300)"
fi
pass "model returned: \"$CONTENT\" ($TOKENS completion tokens)"

section "Verify -ctk $EXPECTED_K -ctv $EXPECTED_V on the LIVE process"
LIVE_CMD=$(ps -ef | grep "[l]lama-server" | grep -v "^root.*bash" | head -1)
if [[ -z "$LIVE_CMD" ]]; then
    fail "no llama-server process found"
fi
if ! echo "$LIVE_CMD" | grep -q -- "-ctk $EXPECTED_K"; then
    fail "live cmdline missing -ctk $EXPECTED_K:\n  $LIVE_CMD"
fi
if ! echo "$LIVE_CMD" | grep -q -- "-ctv $EXPECTED_V"; then
    fail "live cmdline missing -ctv $EXPECTED_V:\n  $LIVE_CMD"
fi
pass "live process running with -ctk $EXPECTED_K -ctv $EXPECTED_V"

section "Clean up"
python -c "from inferhost.core.processes import stop_all; stop_all()" 2>/dev/null || true
sleep 1
python -m inferhost._ops status | sed 's/^/  /'
pass "daemons stopped"

section "Done"
pass "v0.5 functional smoke complete — real model inference verified"
