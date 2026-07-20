#!/usr/bin/env bash
# Functional smoke test for inferhost — exercises the real install
# → register → serve → inference path with a tiny GGUF, and verifies the
# actually-running llama-server has the default K/V flags.
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
EXPECTED_V="q8_0"

# System libs the prebuilt CPU/CUDA binaries link against. libgomp1 is part
# of glibc-runtime on almost every distro; minimal containers like ours miss it.
section "Install runtime deps (libgomp1, jq, curl)"
apt-get update -qq && apt-get install -y -qq libgomp1 jq curl >/dev/null 2>&1
pass "deps installed"

section "Install llama-server (ggml-org/llama.cpp) + llama-swap"
python - <<'PY'
from inferhost.core.binaries import install_llama_server, install_llama_swap
ls = install_llama_server()
sw = install_llama_swap()
print(f"  llama-server: {ls.path} ({ls.version})")
print(f"  llama-swap:   {sw.path} ({sw.version})")
PY
pass "binaries installed"

section "Reinstall on top of existing dir (must purge stale libs)"
# Plants a fake old-versioned .so so we can prove the next install actually
# removes it rather than leaving an ABI mismatch behind. Without the purge,
# llama-server would later segfault with a stale libggml-base.so.<OLD>.
python - <<'PY'
from pathlib import Path
from inferhost.core import paths
from inferhost.core.binaries import install_llama_server

bin_dir = Path(paths.bin_dir())
stale = bin_dir / "libggml-base.so.0.99.99"
stale.write_bytes(b"STALE")  # bogus version that must not survive reinstall
assert stale.exists()

install_llama_server()

# The purge must have wiped our fake stale lib, and all libfoo.so symlinks
# must point at the freshly-installed real files (not symlink loops, not
# missing targets).
assert not stale.exists(), f"stale lib not purged: {stale}"
for link in bin_dir.iterdir():
    if not link.is_symlink():
        continue
    target = (bin_dir / link.readlink()).resolve()
    assert target.exists(), f"dangling symlink after reinstall: {link} -> {target}"
print("  reinstall purged stale lib + all symlinks resolve to real files")
PY
pass "reinstall is hermetic — no leftover ABI-mismatched libs"

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

# --- Multi-part GGUF: split, register via shard 1, prove real auto-discovery -
# hf.list_ggufs' shard-grouping and download_gguf_parts_with_progress's
# cumulative-progress math are unit-tested (hermetic, mocked) in
# test_hf_naming.py. The one thing that can't be proven without the real
# binary is whether llama-server, given only shard 1's path as local_path
# (which is all the registry ever stores), actually finds and loads its
# siblings. We split the already-downloaded tiny model with llama-gguf-split
# (extracted from the same llama.cpp release as the installed llama-server,
# so it shares its libs via RUNPATH=$ORIGIN — no separate build/download of a
# whole new test fixture needed) and serve the result for real.
MULTIPART_NAME="qwen-tiny-multipart"
MULTIPART_PORT=8090

section "Multi-part: fetch llama-gguf-split from the already-installed llama.cpp release"
python - <<'PY'
from pathlib import Path
from inferhost.core import paths
from inferhost.core.binaries import _llamacpp_release_json, _pick_llamacpp_asset, _download, _extract_archive
from inferhost.core.probe import probe
from inferhost.settings import settings

s = settings()
rel = _llamacpp_release_json(s.llamacpp_version)
asset = _pick_llamacpp_asset(rel["assets"], want_gpu=probe().has_gpu, backend=s.llamacpp_backend)
blob = _download(asset.download_url)
extracted = _extract_archive(blob, asset.name, Path(paths.bin_dir()), want_basenames=("llama-gguf-split",))
assert extracted, f"llama-gguf-split not found in {asset.name}"
print(f"  {extracted[0]}")
PY
pass "llama-gguf-split extracted alongside llama-server"

section "Multi-part: split the tiny model into shards (~200 MiB max each)"
python - <<PY
import subprocess
from pathlib import Path
from inferhost.core import paths
from inferhost.core.hf import _PART_RE

src = next(Path("/inferhost/hf-cache").rglob("$MODEL_FILE"))
out_dir = Path("/inferhost/hf-cache/gguf-split")
out_dir.mkdir(parents=True, exist_ok=True)
prefix = out_dir / "$MULTIPART_NAME"
tool = paths.bin_dir() / "llama-gguf-split"
subprocess.run(
    [str(tool), "--split", "--split-max-size", "200M", str(src), str(prefix)], check=True
)

shards = sorted(out_dir.glob("$MULTIPART_NAME-*-of-*.gguf"))
assert len(shards) >= 2, f"expected >=2 shards from the split, got {shards}"
for sh in shards:
    # Same regex hf.list_ggufs groups real HF repo listings with — pins the
    # real tool's naming convention to the production grouping code.
    assert _PART_RE.match(sh.name), f"{sh.name} doesn't match hf.py's _PART_RE"
    print(f"  {sh.name}  ({sh.stat().st_size/1024/1024:.1f} MiB)")
PY
pass "model split into shards matching hf.py's multi-part naming convention"

section "Multi-part: register with local_path = shard 1 only (what download_gguf_parts_with_progress returns)"
python - <<PY
from pathlib import Path
from inferhost.core import registry as reg_mod
from inferhost.core.registry import Model
from inferhost.core.configs import write_all

shard1 = sorted(Path("/inferhost/hf-cache/gguf-split").glob("$MULTIPART_NAME-*-of-*.gguf"))[0]
reg = reg_mod.load()
if not any(m.name == "$MULTIPART_NAME" for m in reg.models):
    reg.add(Model(name="$MULTIPART_NAME", repo_id="$MODEL_REPO", filename=shard1.name,
                  local_path=str(shard1), port=$MULTIPART_PORT, ctx=2048, size_gib=0.4))
reg_mod.save(reg); write_all(reg)
print(f"  registered {shard1.name!r} -> 127.0.0.1:$MULTIPART_PORT (siblings not referenced anywhere)")
PY
pass "registry + configs OK"

section "Multi-part: force-load + real chat completion (proves shard auto-discovery on the live binary)"
python -c "
from inferhost.core.processes import reload_if_running, force_load_model
import sys
reload_if_running()
ok = force_load_model('$MULTIPART_NAME', timeout=60.0)
sys.exit(0 if ok else 2)
" || fail "multi-part model failed to load — check ${MULTIPART_NAME}.err.log; llama-server may not be auto-discovering sibling shards from shard 1's path on this build"
RESP=$(curl -s --max-time 60 http://127.0.0.1:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MULTIPART_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word OK and nothing else.\"}],\"max_tokens\":10,\"temperature\":0}")
CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content // empty')
if [[ -z "$CONTENT" ]]; then
    fail "multi-part model returned no content (shard auto-discovery likely failed): $(echo "$RESP" | head -c 300)"
fi
pass "multi-part model (registered via shard 1 only) answered: \"$CONTENT\" — llama-server found the sibling shards on its own"

section "Multi-part: unload to free VRAM before the multi-model tests below"
python -c "from inferhost.core.processes import force_unload_model; force_unload_model('$MULTIPART_NAME', timeout=15.0)" >/dev/null
pass "multi-part model unloaded"

# --- Multi-model pin / unpin lifecycle ---------------------------------------
# Register a SECOND entry pointing at the same GGUF on a different port and pin
# both. Verify VRAM accounting, can_pin guard, and force_unload_model actually
# tears the model down (catches stale llama-swap admin URL + VRAM-trace bugs).
section "Multi-model: register tiny-b (port $((MODEL_PORT+1))) and pin both"
python - <<PY
from pathlib import Path
from inferhost.core import registry as reg_mod
from inferhost.core.registry import Model
from inferhost.core.configs import write_all
g = next(Path("/inferhost/hf-cache").rglob("$MODEL_FILE"))
reg = reg_mod.load()
# qwen-tiny already exists; mark it pinned and add a sibling
for m in reg.models:
    if m.name == "$MODEL_NAME":
        m.pin = True
if not any(m.name == "${MODEL_NAME}-b" for m in reg.models):
    reg.add(Model(name="${MODEL_NAME}-b", repo_id="$MODEL_REPO", filename="$MODEL_FILE",
                  local_path=str(g), port=$((MODEL_PORT+1)), ctx=2048, size_gib=0.4, pin=True))
reg_mod.save(reg); write_all(reg)
print(f"  pinned: {[m.name for m in reg.models if m.pin]}")
PY
python -c "from inferhost.core.processes import reload_if_running; reload_if_running()" >/dev/null
sleep 2

VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
python -c "from inferhost.core.processes import force_load_model
import sys
ok_a = force_load_model('$MODEL_NAME', timeout=60.0)
ok_b = force_load_model('${MODEL_NAME}-b', timeout=60.0)
sys.exit(0 if ok_a and ok_b else 2)"
VRAM_BOTH=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
PROC_COUNT=$(ps -ef | grep "[l]lama-server" | grep -v grep | wc -l)
if (( VRAM_BOTH - VRAM_BEFORE < 800 )); then
    fail "VRAM did not jump on dual-load (delta only ${VRAM_BOTH}-${VRAM_BEFORE} MiB)"
fi
if (( PROC_COUNT < 2 )); then fail "expected 2 llama-server procs, got $PROC_COUNT"; fi
pass "both pinned + loaded — VRAM ${VRAM_BEFORE}→${VRAM_BOTH} MiB across $PROC_COUNT procs"

section "Multi-model: can_pin refuses an oversized request"
python - <<PY
from inferhost.core import registry as reg_mod, vram
from inferhost.core.registry import Model
huge = Model(name="huge-fake", repo_id="x/y", filename="f.gguf", local_path="/dev/null",
             port=8099, ctx=131072, size_gib=999.0, pin=False)
ok, need, free = vram.can_pin(reg_mod.load(), huge)
assert not ok, f"can_pin should refuse 999 GiB, but ok={ok}"
print(f"  refused: needed={need:.1f} GiB, free={free:.1f} GiB")
PY
pass "VRAM guard rejects oversized pin"

section "Multi-model: unpin $MODEL_NAME → force_unload (llama-swap admin endpoint)"
UNLOAD_OK=$(python -c "from inferhost.core.processes import force_unload_model
print('true' if force_unload_model('$MODEL_NAME', timeout=15.0) else 'false')")
sleep 3
VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
PROC_AFTER=$(ps -ef | grep "[l]lama-server" | grep -v grep | wc -l)
if [[ "$UNLOAD_OK" != "true" ]]; then fail "force_unload_model returned False"; fi
if (( VRAM_BOTH - VRAM_AFTER < 400 )); then
    fail "VRAM did not drop after unload (was $VRAM_BOTH MiB, now $VRAM_AFTER MiB)"
fi
if (( PROC_AFTER != 1 )); then fail "expected 1 proc after unload, got $PROC_AFTER"; fi
pass "unload OK — VRAM ${VRAM_BOTH}→${VRAM_AFTER} MiB, procs $PROC_COUNT→$PROC_AFTER"

section "Multi-model: sibling still serves while $MODEL_NAME is unloaded"
RESP=$(curl -s --max-time 20 http://127.0.0.1:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}-b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":5,\"temperature\":0}")
CB=$(echo "$RESP" | jq -r '.choices[0].message.content // empty')
if [[ -z "$CB" ]]; then fail "${MODEL_NAME}-b stopped responding: $RESP"; fi
pass "${MODEL_NAME}-b still alive: \"$CB\""

# --- Lifecycle: rename, stop, start, per-model + global config change --------
# Exercises the daemon-restart paths through LiteLLM (port 9001), catches
# dep regressions like litellm[proxy] missing, and confirms config edits
# actually propagate into the running llama-server argv.
port_in_use() { ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":$1\$"; }
proc_count_exe() { ls -l /proc/*/exe 2>/dev/null | awk -v p="$1" '{n=split($NF,a,"/"); if(a[n]==p) c++} END{print c+0}'; }
wait_for_model_in_both() {
    for i in $(seq 1 15); do
        if curl -sf http://127.0.0.1:9001/v1/models 2>/dev/null | jq -r '.data[].id' 2>/dev/null | grep -qx "$1" \
        && curl -sf http://127.0.0.1:9090/v1/models 2>/dev/null | jq -r '.data[].id' 2>/dev/null | grep -qx "$1"; then
            return 0
        fi
        sleep 1
    done; return 1
}

GATEWAY_PORT=${INFERHOST_GATEWAY_PORT:-9001}
section "Lifecycle: gateway warmup (LiteLLM bind on :$GATEWAY_PORT)"
python -c "from inferhost.core.processes import start_swap, start_gateway; start_swap(); start_gateway()" >/dev/null
wait_for_model_in_both "${MODEL_NAME}-b" || fail "${MODEL_NAME}-b not visible in LiteLLM (gateway may have crashed — check litellm[proxy] dep)"
pass "gateway up and proxying"

section "Lifecycle: RENAME ${MODEL_NAME}-b → bard"
python - <<PY
from inferhost.core import registry as reg_mod
from inferhost.core.configs import write_all
from inferhost.core.processes import reload_if_running, force_unload_model
reg = reg_mod.load()
assert reg.rename("${MODEL_NAME}-b", "bard"), "rename failed"
reg_mod.save(reg); write_all(reg)
force_unload_model("${MODEL_NAME}-b", timeout=5.0); reload_if_running()
PY
wait_for_model_in_both bard || fail "bard not visible after rename"
RESP=$(curl -s --max-time 60 http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"bard\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":5}")
C=$(echo "$RESP" | jq -r '.choices[0].message.content // empty')
[[ -z "$C" ]] && fail "bard not serving: $RESP"
CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}-b\",\"messages\":[{\"role\":\"user\",\"content\":\"X\"}],\"max_tokens\":2}")
[[ "$CODE" -ge 400 ]] || fail "old name still served (HTTP $CODE)"
pass "rename: bard -> \"$C\", old name -> HTTP $CODE"

section "Lifecycle: STOP (verify daemons + ports + VRAM all released)"
python -c "from inferhost.core.processes import stop_all; stop_all()"
sleep 3
LS=$(proc_count_exe llama-server); SW=$(proc_count_exe llama-swap)
port_in_use "$GATEWAY_PORT" && fail "gateway port still bound"
port_in_use 9090 && fail "swap port still bound"
[[ "$LS" -eq 0 && "$SW" -eq 0 ]] || fail "leftover procs (server=$LS swap=$SW)"
pass "clean shutdown — no zombies, ports released"

section "Lifecycle: START again (cold restart)"
python -c "from inferhost.core.processes import start_swap, start_gateway; start_swap(); start_gateway()" >/dev/null
wait_for_model_in_both bard || fail "bard missing after restart"
RESP=$(curl -s --max-time 60 http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"bard\",\"messages\":[{\"role\":\"user\",\"content\":\"After restart.\"}],\"max_tokens\":5}")
C=$(echo "$RESP" | jq -r '.choices[0].message.content // empty')
[[ -z "$C" ]] && fail "bard broken after restart: $RESP"
pass "bard alive after cold start -> \"$C\""

section "Lifecycle: CHANGE per-model ctx 2048 → 4096 (visible in live argv)"
python - <<PY
from inferhost.core import registry as reg_mod
from inferhost.core.configs import write_all
from inferhost.core.processes import reload_if_running, force_unload_model
reg = reg_mod.load(); reg.get("bard").ctx = 4096
reg_mod.save(reg); write_all(reg)
force_unload_model("bard", timeout=10.0); reload_if_running()
PY
wait_for_model_in_both bard || fail "swap not ready after reload"
curl -s --max-time 60 http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"bard\",\"messages\":[{\"role\":\"user\",\"content\":\"X\"}],\"max_tokens\":2}" >/dev/null
sleep 3
LIVE_CTX=$(ps -eo cmd | grep -E '^.*llama-server ' | grep -oE '\-c [0-9]+' | head -1)
[[ "$LIVE_CTX" == "-c 4096" ]] || fail "ctx mismatch in live argv: $LIVE_CTX"
pass "per-model ctx flowed through to llama-server argv: $LIVE_CTX"

section "Lifecycle: CHANGE global gpu_layers 99 → 50 via save_overrides"
python - <<PY
from inferhost.settings import save_overrides, reload_settings
from inferhost.core.configs import write_all
from inferhost.core import registry as reg_mod
from inferhost.core.processes import reload_if_running, force_unload_model
save_overrides({"gpu_layers": 50}); reload_settings()
write_all(reg_mod.load())
force_unload_model("bard", timeout=10.0); reload_if_running()
PY
wait_for_model_in_both bard || fail "swap not ready"
curl -s --max-time 60 http://127.0.0.1:${GATEWAY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"bard\",\"messages\":[{\"role\":\"user\",\"content\":\"Y\"}],\"max_tokens\":2}" >/dev/null
sleep 3
LIVE_NGL=$(ps -eo cmd | grep -E '^.*llama-server ' | grep -oE '\-ngl [0-9]+' | head -1)
[[ "$LIVE_NGL" == "-ngl 50" ]] || fail "ngl mismatch in live argv: $LIVE_NGL"
pass "global override flowed through to llama-server argv: $LIVE_NGL"

# Restore default override so re-runs start fresh
python - <<PY
from inferhost.settings import overrides_env_path
p = overrides_env_path()
if p.exists():
    p.unlink()
PY

# --- Text-to-speech: real llama-tts synthesis through the gateway ------------
# Proves the bundled llama-tts binary + inferhost-tts daemon + LiteLLM
# audio_speech passthrough actually produce a WAV. OuteTTS needs a separate
# vocoder GGUF; both are cached in the hf-cache volume after first run.
TTS_MODEL_REPO="OuteAI/OuteTTS-0.2-500M-GGUF"
TTS_MODEL_FILE="OuteTTS-0.2-500M-Q4_K_M.gguf"
TTS_VOCODER_REPO="ggml-org/WavTokenizer"
TTS_VOCODER_FILE="WavTokenizer-Large-75-F16.gguf"

section "TTS: llama-tts binary was bundled alongside llama-server"
python - <<'PY'
from inferhost.core import paths
p = paths.llama_tts_path()
assert p.exists(), f"llama-tts not extracted at {p}"
print(f"  {p}")
PY
pass "llama-tts present"

section "TTS: download OuteTTS model + WavTokenizer vocoder (cached in volume)"
python - <<PY
from huggingface_hub import hf_hub_download
m = hf_hub_download(repo_id="$TTS_MODEL_REPO", filename="$TTS_MODEL_FILE", cache_dir="/inferhost/hf-cache")
v = hf_hub_download(repo_id="$TTS_VOCODER_REPO", filename="$TTS_VOCODER_FILE", cache_dir="/inferhost/hf-cache")
import os
print(f"  model:   {m}  ({os.path.getsize(m)/1024/1024:.0f} MiB)")
print(f"  vocoder: {v}  ({os.path.getsize(v)/1024/1024:.0f} MiB)")
PY
pass "TTS model + vocoder in hf-cache"

section "TTS: register model (vocoder => TTS), verify config routing"
python - <<PY
from pathlib import Path
from inferhost.core import registry as reg_mod
from inferhost.core.registry import Model
from inferhost.core.configs import write_all, render_llama_swap, render_litellm

m = next(Path("/inferhost/hf-cache").rglob("$TTS_MODEL_FILE"))
v = next(Path("/inferhost/hf-cache").rglob("$TTS_VOCODER_FILE"))
reg = reg_mod.load()
reg.add(Model(name="oute", repo_id="$TTS_MODEL_REPO", filename="$TTS_MODEL_FILE",
              local_path=str(m), vocoder_path=str(v), size_gib=0.4))
reg_mod.save(reg); write_all(reg)

# TTS model must NOT appear in llama-swap, and must be audio_speech in litellm.
swap = render_llama_swap(reg)
assert "oute" not in swap["models"], "TTS model leaked into llama-swap"
lite = {e["model_name"]: e for e in render_litellm(reg)["model_list"]}
assert lite["oute"]["model_info"]["mode"] == "audio_speech", lite["oute"]
print("  oute excluded from llama-swap; registered as audio_speech in litellm")
PY
pass "TTS routing config correct"

section "TTS: restart stack (brings up inferhost-tts daemon)"
python - <<'PY'
import inferhost.core.processes as p
p.start_swap(); p.start_gateway()
p.reload_if_running()  # picks up new configs + starts inferhost-tts for the TTS model
print(f"  tts running: {p.tts_status().running}")
PY
sleep 2
python -m inferhost._ops status | sed 's/^/  /'
TTS_PORT=${INFERHOST_TTS_PORT:-9092}
for i in $(seq 1 30); do
    curl -sf "http://127.0.0.1:${TTS_PORT}/health" >/dev/null 2>&1 && break
    sleep 0.5
done
curl -sf "http://127.0.0.1:${TTS_PORT}/health" >/dev/null || fail "inferhost-tts did not come up on :$TTS_PORT"
pass "inferhost-tts daemon up on :$TTS_PORT"

section "TTS: synthesize directly via inferhost-tts /v1/audio/speech"
curl -s --max-time 180 "http://127.0.0.1:${TTS_PORT}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"model":"oute","input":"Functional smoke test.","voice":"default"}' \
    -o /tmp/tts-direct.wav
if [[ "$(head -c 4 /tmp/tts-direct.wav)" != "RIFF" ]]; then
    fail "direct TTS did not return a WAV: $(head -c 200 /tmp/tts-direct.wav)"
fi
pass "direct TTS produced a WAV ($(stat -c%s /tmp/tts-direct.wav) bytes)"

section "TTS: synthesize through the LiteLLM gateway (:$GATEWAY_PORT)"
curl -s --max-time 180 "http://127.0.0.1:${GATEWAY_PORT}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"model":"oute","input":"Gateway routed.","voice":"default"}' \
    -o /tmp/tts-gateway.wav
if [[ "$(head -c 4 /tmp/tts-gateway.wav)" != "RIFF" ]]; then
    fail "gateway TTS did not return a WAV: $(head -c 200 /tmp/tts-gateway.wav)"
fi
pass "gateway TTS routed to daemon and produced a WAV ($(stat -c%s /tmp/tts-gateway.wav) bytes)"

# --- Image generation: real sd-server txt2img through the gateway ------------
# Proves the bundled sd-server (stable-diffusion.cpp), fronted by llama-swap,
# produces a PNG via the OpenAI /v1/images/generations endpoint. SD1.5 is a
# self-contained single-file checkpoint (~1.6 GiB), cached in the volume.
IMG_MODEL_REPO="second-state/stable-diffusion-v1-5-GGUF"
IMG_MODEL_FILE="stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf"

section "Image: sd-server binary was bundled (isolated in bin/sd/)"
python - <<'PY'
from inferhost.core import binaries, paths
if binaries.needs_sdcpp_refresh():
    binaries.install_stable_diffusion()
p = paths.sd_server_path()
assert p.exists(), f"sd-server not installed at {p}"
# Isolation guard: a llama.cpp reinstall must NOT wipe libstable-diffusion.so.
binaries.install_llama_server()
assert (paths.sd_bin_dir() / "libstable-diffusion.so").exists(), \
    "llama.cpp reinstall wiped libstable-diffusion.so — isolation broke!"
print(f"  {p}  (survives llama.cpp reinstall)")
PY
pass "sd-server bundled and isolated from the llama.cpp purge"

section "Image: download SD1.5 single-file checkpoint (cached in volume)"
python - <<PY
from huggingface_hub import hf_hub_download
m = hf_hub_download(repo_id="$IMG_MODEL_REPO", filename="$IMG_MODEL_FILE", cache_dir="/inferhost/hf-cache")
import os; print(f"  model: {m}  ({os.path.getsize(m)/1024/1024:.0f} MiB)")
PY
pass "image model in hf-cache"

section "Image: register (kind=image), verify sd-server cmd + image_generation"
python - <<PY
from pathlib import Path
from inferhost.core import registry as reg_mod
from inferhost.core.registry import Model
from inferhost.core.configs import write_all, render_llama_swap, render_litellm
g = next(Path("/inferhost/hf-cache").rglob("$IMG_MODEL_FILE"))
reg = reg_mod.load()
reg.add(Model(name="sd15", repo_id="$IMG_MODEL_REPO", filename="$IMG_MODEL_FILE",
              local_path=str(g), kind="image", size_gib=1.6,
              port=reg.next_port(9090), extra_args="--steps 8"))
reg_mod.save(reg); write_all(reg)
cmd = render_llama_swap(reg)["models"]["sd15"]["cmd"]
assert "sd-server" in cmd and "-m " in cmd, cmd
lite = {e["model_name"]: e["model_info"].get("mode") for e in render_litellm(reg)["model_list"]}
assert lite["sd15"] == "image_generation", lite
print("  sd15 -> sd-server cmd; litellm mode=image_generation")
PY
pass "image routing config correct"

section "Image: restart stack + generate via gateway /v1/images/generations"
python -c "from inferhost.core.processes import start_swap, start_gateway, reload_if_running; start_swap(); start_gateway(); reload_if_running()" >/dev/null
wait_for_model_in_both sd15 || warn "sd15 not yet listed (sd-server loads lazily on first request)"
curl -s --max-time 600 "http://127.0.0.1:${GATEWAY_PORT}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{"model":"sd15","prompt":"a red apple on a wooden table","size":"256x256"}' \
    -o /tmp/img-gen.json
python - <<'PY'
import base64, json
d = json.load(open("/tmp/img-gen.json"))
b64 = d["data"][0]["b64_json"]
raw = base64.b64decode(b64)
open("/tmp/img-gen.png", "wb").write(raw)
assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {raw[:16]!r}"
print(f"  generated PNG: {len(raw)} bytes")
PY
pass "gateway -> llama-swap -> sd-server produced a PNG"

section "Clean up"
python -c "from inferhost.core.processes import stop_all; stop_all()" 2>/dev/null || true
sleep 1
python -m inferhost._ops status | sed 's/^/  /'
pass "daemons stopped"

section "Done"
pass "v0.5 functional smoke complete — real model inference verified"
