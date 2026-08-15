import struct

import pytest

from inferhost.core.configs import render_litellm, render_llama_swap
from inferhost.core.registry import Model, Registry
from inferhost.settings import reload_settings


def _write_gguf(path, arch: str, ctx: int) -> None:
    """Write a minimal valid GGUF header advertising ``<arch>.context_length``."""
    def gstr(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kvs = [
        gstr("general.architecture") + struct.pack("<I", 8) + gstr(arch),
        gstr(f"{arch}.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx),
    ]
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path.write_bytes(header + b"".join(kvs))


def _force_supported_cache_types(monkeypatch, values: frozenset[str]) -> None:
    """Override the cached capability probe so tests are hermetic.

    configs.py uses `from inferhost.core.llama_caps import supported_cache_types`,
    which captures the function reference at import time — patching it on
    `llama_caps` alone doesn't reach the binding inside `configs`. Patch
    both namespaces so any caller hits the stub.
    """
    from inferhost.core import configs, llama_caps
    llama_caps.supported_cache_types.cache_clear()
    monkeypatch.setattr(llama_caps, "supported_cache_types", lambda: values)
    monkeypatch.setattr(configs, "supported_cache_types", lambda: values)


def _force_spec_support(monkeypatch, supported: bool) -> None:
    """Override the DFlash capability probe so tests are hermetic.

    configs.py does `from inferhost.core.llama_caps import supports_spec_type`,
    capturing the reference at import time — patch both namespaces so any
    caller hits the stub. ``supported`` maps to the fail-open contract:
    supports_spec_type returns True for every spec-type when supported, False
    otherwise.
    """
    from inferhost.core import configs, llama_caps
    llama_caps._help_text.cache_clear()
    stub = (lambda name: True) if supported else (lambda name: False)
    monkeypatch.setattr(llama_caps, "supports_spec_type", stub)
    monkeypatch.setattr(configs, "supports_spec_type", stub)


def test_render_llama_swap_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(
        monkeypatch,
        frozenset({"f16", "q8_0", "q5_0", "q4_0"}),
    )

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF", filename="qwen-Q4_K_M.gguf",
              quant="Q4_K_M", ctx=8192, port=8081, size_gib=4.4, local_path="/tmp/qwen.gguf"),
    ])
    cfg = render_llama_swap(reg)
    assert "qwen" in cfg["models"]
    entry = cfg["models"]["qwen"]
    cmd = entry["cmd"]

    # Basic flags
    assert "--model" in cmd
    assert "/tmp/qwen.gguf" in cmd
    assert "--port 8081" in cmd

    # Proxy must use loopback — never an external/LAN address
    assert entry["proxy"] == "http://127.0.0.1:8081"

    # KV cache default: K=q8_0, V=q8_0 (q8_0 is near-lossless, ~2x compression).
    assert "-ctk q8_0" in cmd
    assert "-ctv q8_0" in cmd

    # --jinja must be present so the model's native chat template (with
    # tool-call and vision content-block support) is used instead of the
    # legacy built-in template.
    assert "--jinja" in cmd


def test_render_llama_swap_attaches_mmproj_for_vision_model(tmp_path, monkeypatch):
    """A model with mmproj_path set must pass --mmproj to llama-server."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(
        monkeypatch,
        frozenset({"f16", "q8_0", "q5_0", "q4_0"}),
    )

    reg = Registry(models=[
        Model(
            name="qwen3vl",
            repo_id="Qwen/Qwen3VL-8B-Instruct-GGUF",
            filename="qwen3vl.gguf",
            quant="Q8_0",
            ctx=32768,
            port=8081,
            local_path="/tmp/qwen3vl.gguf",
            mmproj_path="/tmp/mmproj-qwen3vl-F16.gguf",
        ),
    ])
    cmd = render_llama_swap(reg)["models"]["qwen3vl"]["cmd"]
    assert "--mmproj /tmp/mmproj-qwen3vl-F16.gguf" in cmd
    # The short alias -mm was replaced with the long form; make sure the
    # rendered cmd is greppable.
    assert " -mm " not in cmd


def test_render_substitutes_unsupported_kv_quant(tmp_path, monkeypatch):
    """When the configured KV quant isn't in the binary's allow-list, fall back."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_KV_QUANT_V", "q4_0")
    reload_settings()
    # Minimal build that doesn't expose q4_0 — should fall through to q4_1.
    _force_supported_cache_types(
        monkeypatch,
        frozenset({"f16", "bf16", "q8_0", "q4_1"}),
    )

    reg = Registry(models=[
        Model(name="qwen", repo_id="x", filename="x.gguf", port=8081, local_path="/tmp/x.gguf"),
    ])
    notices: list[str] = []
    cfg = render_llama_swap(reg, notices=notices)
    cmd = cfg["models"]["qwen"]["cmd"]

    assert "-ctv q4_0" not in cmd
    assert "-ctv q4_1" in cmd
    # K side was already q8_0 (supported) so it's untouched.
    assert "-ctk q8_0" in cmd
    # A notice must be emitted so the user knows their setting was substituted.
    assert any("q4_0" in n and "q4_1" in n for n in notices)


def test_write_all_persists_and_consumes_notices(tmp_path, monkeypatch):
    """write_all writes notices to disk; consume_notices reads and clears them."""
    from inferhost.core import configs, paths
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_KV_QUANT_V", "q4_0")
    reload_settings()
    _force_supported_cache_types(
        monkeypatch,
        frozenset({"f16", "q8_0", "q4_1"}),  # no q4_0
    )

    reg = Registry(models=[
        Model(name="qwen", repo_id="x", filename="x.gguf", port=8081, local_path="/tmp/x.gguf"),
        Model(name="llama", repo_id="y", filename="y.gguf", port=8082, local_path="/tmp/y.gguf"),
    ])
    configs.write_all(reg)
    assert paths.notices_path().exists()

    notes = configs.consume_notices()
    assert notes  # at least one notice was written
    # Dedupe: the q4_0 warning fires once even with 2 models
    assert sum("q4_0" in n for n in notes) == 1
    # consume should have removed the file
    assert not paths.notices_path().exists()

    # Second consume returns empty (file already gone)
    assert configs.consume_notices() == []


def test_per_model_overrides_win_over_global(tmp_path, monkeypatch):
    """Per-model fields on Model override the global Settings in the rendered cmd."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    # Globals deliberately set to something different from the per-model values
    # below — we then assert the per-model values are what made it into the cmd.
    monkeypatch.setenv("INFERHOST_GPU_LAYERS", "99")
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "1")
    monkeypatch.setenv("INFERHOST_FLASH_ATTENTION", "on")
    monkeypatch.setenv("INFERHOST_KV_QUANT_K", "q8_0")
    monkeypatch.setenv("INFERHOST_KV_QUANT_V", "q8_0")
    reload_settings()
    _force_supported_cache_types(
        monkeypatch,
        frozenset({"f16", "q8_0", "q5_0", "q4_0"}),
    )

    reg = Registry(models=[
        Model(
            name="tuned",
            repo_id="x/y",
            filename="y.gguf",
            port=8081,
            local_path="/tmp/y.gguf",
            kv_quant_k="f16",
            kv_quant_v="q4_0",
            gpu_layers=42,
            parallel_slots=4,
            flash_attention="off",
        ),
    ])
    cmd = render_llama_swap(reg)["models"]["tuned"]["cmd"]
    assert "-ctk f16" in cmd
    assert "-ctv q4_0" in cmd
    assert "-ngl 42" in cmd
    assert "--parallel 4" in cmd
    assert "-fa off" in cmd
    # Globals must NOT have been applied
    assert "-ngl 99" not in cmd
    assert "--parallel 1" not in cmd
    assert "-fa on" not in cmd


def test_render_litellm_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SWAP_PORT", "8080")
    reload_settings()

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/x", filename="x.gguf", port=8081),
        Model(name="qwenvl", repo_id="Qwen/y", filename="y.gguf", port=8082,
              mmproj_path="/tmp/mmproj.gguf"),
    ])
    cfg = render_litellm(reg)
    assert len(cfg["model_list"]) == 2
    entry = cfg["model_list"][0]
    assert entry["model_name"] == "qwen"

    # LiteLLM proxies to llama-swap on loopback — must never be an external address
    api_base = entry["litellm_params"]["api_base"]
    assert api_base == "http://127.0.0.1:8080/v1"
    assert "127.0.0.1" in api_base

    # Capability flags: tool-calling is always advertised (llama.cpp's jinja
    # template parses tool calls for any tool-trained GGUF); vision is gated
    # on having an mmproj projector.
    text_info = cfg["model_list"][0]["model_info"]
    vl_info = cfg["model_list"][1]["model_info"]
    assert text_info["supports_function_calling"] is True
    assert text_info["supports_vision"] is False
    assert vl_info["supports_vision"] is True


def test_ctx_clamped_to_native_trained_context(tmp_path, monkeypatch):
    """A configured -c above the GGUF's native window is clamped on serve AND
    in what's advertised to agents, with a notice explaining the clamp."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SWAP_PORT", "8080")
    # One slot, so `-c` is the per-request window unscaled (see
    # test_ctx_is_scaled_by_parallel_slots for the multi-slot case).
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "1")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    gguf_path = tmp_path / "small.gguf"
    _write_gguf(gguf_path, arch="qwen3", ctx=4096)

    reg = Registry(models=[
        Model(name="small", repo_id="x/y", filename="small.gguf", port=8081,
              ctx=8192, local_path=str(gguf_path)),
    ])

    notices: list[str] = []
    cmd = render_llama_swap(reg, notices=notices)["models"]["small"]["cmd"]
    assert "-c 4096" in cmd
    assert "-c 8192" not in cmd
    assert any("native trained context 4096" in n for n in notices)

    # The advertised window must match the clamped served window.
    info = render_litellm(reg)["model_list"][0]["model_info"]
    assert info["max_input_tokens"] == 4096
    assert info["max_tokens"] == 4096


def test_ctx_below_native_is_left_alone(tmp_path, monkeypatch):
    """When -c is at or below native, serve exactly what the user configured."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "1")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    gguf_path = tmp_path / "big.gguf"
    _write_gguf(gguf_path, arch="qwen3", ctx=262144)

    reg = Registry(models=[
        Model(name="big", repo_id="x/y", filename="big.gguf", port=8081,
              ctx=65536, local_path=str(gguf_path)),
    ])
    notices: list[str] = []
    cmd = render_llama_swap(reg, notices=notices)["models"]["big"]["cmd"]
    assert "-c 65536" in cmd
    assert notices == []


def test_ctx_is_scaled_by_parallel_slots(tmp_path, monkeypatch):
    """`-c` must be sized for ALL slots.

    llama-server divides -c across --parallel, so emitting the raw configured
    window served only ctx/slots tokens per request (`-c 8192 --parallel 3` ->
    "request exceeds the available context size (2816 tokens)") while litellm
    still advertised 8192. Serve ctx x slots so a single request really gets
    the configured window.
    """
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "3")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="y.gguf", port=8081,
              ctx=8192, local_path=str(tmp_path / "y.gguf")),
    ])
    cmd = render_llama_swap(reg)["models"]["qwen"]["cmd"]
    assert "-c 24576" in cmd
    assert "--parallel 3" in cmd

    # Advertised window stays the per-request window the user configured.
    info = render_litellm(reg)["model_list"][0]["model_info"]
    assert info["max_input_tokens"] == 8192


def test_ctx_scaling_uses_per_model_slot_override(tmp_path, monkeypatch):
    """A per-model parallel_slots override scales -c too, and the native clamp
    applies to the per-request window (not the multiplied total)."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "1")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    gguf_path = tmp_path / "small.gguf"
    _write_gguf(gguf_path, arch="qwen3", ctx=4096)

    reg = Registry(models=[
        Model(name="small", repo_id="x/y", filename="small.gguf", port=8081,
              ctx=8192, parallel_slots=2, local_path=str(gguf_path)),
    ])
    cmd = render_llama_swap(reg)["models"]["small"]["cmd"]
    # clamped to native 4096 per request, x2 slots
    assert "-c 8192" in cmd
    assert "--parallel 2" in cmd
    assert render_litellm(reg)["model_list"][0]["model_info"]["max_tokens"] == 4096


def test_slots_reduced_when_ctx_times_slots_exceeds_vram(tmp_path, monkeypatch):
    """Scaling -c by the slot count must never render a config that OOMs.

    A 27B at ctx 65536 x 3 slots asks for a ~196k-token cache on a 24 GiB card;
    llama-server dies with ErrorOutOfDeviceMemory and llama-swap restarts it
    forever. The slot count gives way (it's the throughput knob) and the
    configured context — which we advertise to clients — is preserved.
    """
    from inferhost.core import vram

    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "3")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    monkeypatch.setattr(vram, "total_vram_gib", lambda gpu_index=0: 24.0)

    reg = Registry(models=[
        Model(name="big", repo_id="x/y", filename="y.gguf", port=8081,
              ctx=65536, size_gib=17.2, local_path=str(tmp_path / "y.gguf")),
    ])
    notices: list[str] = []
    cmd = render_llama_swap(reg, notices=notices)["models"]["big"]["cmd"]

    assert "--parallel 1" in cmd
    assert "-c 65536" in cmd  # full configured window still served
    assert any("parallel slots" in n and "big" in n for n in notices)

    # And the advertised window is untouched by the slot reduction.
    assert render_litellm(reg)["model_list"][0]["model_info"]["max_input_tokens"] == 65536


def test_slots_kept_when_everything_fits(tmp_path, monkeypatch):
    """A small model at the same slot count fits, so concurrency is preserved."""
    from inferhost.core import vram

    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "3")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    monkeypatch.setattr(vram, "total_vram_gib", lambda gpu_index=0: 24.0)

    reg = Registry(models=[
        Model(name="small", repo_id="x/y", filename="y.gguf", port=8081,
              ctx=4096, size_gib=1.5, local_path=str(tmp_path / "y.gguf")),
    ])
    notices: list[str] = []
    cmd = render_llama_swap(reg, notices=notices)["models"]["small"]["cmd"]
    assert "--parallel 3" in cmd
    assert "-c 12288" in cmd
    assert notices == []


def test_slots_not_second_guessed_without_gpu_info(tmp_path, monkeypatch):
    """On a CPU box (or a failed probe) the fit question is unanswerable —
    honor the requested slot count rather than silently serializing."""
    from inferhost.core import vram

    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "2")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    monkeypatch.setattr(vram, "total_vram_gib", lambda gpu_index=0: 0.0)

    reg = Registry(models=[
        Model(name="huge", repo_id="x/y", filename="y.gguf", port=8081,
              ctx=32768, size_gib=60.0, local_path=str(tmp_path / "y.gguf")),
    ])
    cmd = render_llama_swap(reg)["models"]["huge"]["cmd"]
    assert "--parallel 2" in cmd
    assert "-c 65536" in cmd


def test_vram_estimate_scales_with_parallel_slots(tmp_path, monkeypatch):
    """Each slot holds its own full context, so the KV estimate must scale —
    otherwise pin feasibility is off by the slot count."""
    from inferhost.core import vram

    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "1")
    reload_settings()
    m = Model(name="a", repo_id="x/y", filename="y.gguf", ctx=8192, size_gib=10.0)
    one_slot = vram.estimate_model_vram_gib(m)

    monkeypatch.setenv("INFERHOST_PARALLEL_SLOTS", "4")
    reload_settings()
    four_slots = vram.estimate_model_vram_gib(m)

    kv_one = one_slot - 10.0 * 1.05
    assert four_slots == pytest.approx(10.0 * 1.05 + kv_one * 4)


def test_mtp_draft_override_wins_over_global(tmp_path, monkeypatch):
    """Per-model spec_draft_n_max_override controls --spec-draft-n-max; -1
    inherits the global, 0 disables the MTP lane entirely."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SPEC_DRAFT_N_MAX", "2")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    # 'mtp' in the name marks the model MTP-capable.
    def mk(override):
        return Registry(models=[
            Model(name="qwen-mtp", repo_id="x/y", filename="qwen-mtp.gguf",
                  port=8081, local_path="/tmp/qwen-mtp.gguf",
                  spec_draft_n_max_override=override),
        ])

    inherit = render_llama_swap(mk(-1))["models"]["qwen-mtp"]["cmd"]
    assert "--spec-type draft-mtp" in inherit
    assert "--spec-draft-n-max 2" in inherit  # global default

    tuned = render_llama_swap(mk(5))["models"]["qwen-mtp"]["cmd"]
    assert "--spec-draft-n-max 5" in tuned

    off = render_llama_swap(mk(0))["models"]["qwen-mtp"]["cmd"]
    assert "draft-mtp" not in off


def test_reasoning_off_does_not_pin_budget(tmp_path, monkeypatch):
    """`--reasoning off` suppresses thinking via enable_thinking=false. We must
    NOT also force --reasoning-budget 0: the budget-0 hard stop injects the
    end-of-thinking tag at token 0, which makes some finetuned/MTP models run
    away instead of answering. So an "off" model still inherits the global
    budget verbatim."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_REASONING_BUDGET", "-1")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="qwen.gguf", port=8081,
              local_path="/tmp/qwen.gguf", reasoning="off", reasoning_budget=-2),
    ])
    cmd = render_llama_swap(reg)["models"]["qwen"]["cmd"]
    assert "--reasoning off" in cmd
    assert "--reasoning-budget -1" in cmd  # inherited, NOT pinned to 0


def test_per_model_reasoning_overrides_global(tmp_path, monkeypatch):
    """A per-model reasoning value beats the global setting — this is why a
    model with reasoning='on' keeps thinking even when the global config is set
    to 'off'."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_REASONING", "off")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="qwen.gguf", port=8081,
              local_path="/tmp/qwen.gguf", reasoning="on", reasoning_budget=-2),
    ])
    cmd = render_llama_swap(reg)["models"]["qwen"]["cmd"]
    assert "--reasoning on" in cmd  # per-model "on" wins over global "off"


def test_tts_model_excluded_from_llama_swap(tmp_path, monkeypatch):
    """A model with a vocoder is TTS-served, so it must NOT appear in llama-swap."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="qwen.gguf", port=8081,
              local_path="/tmp/qwen.gguf"),
        Model(name="oute", repo_id="a/b", filename="oute.gguf", port=8082,
              local_path="/tmp/oute.gguf", vocoder_path="/tmp/wavtok.gguf"),
    ])
    cfg = render_llama_swap(reg)
    assert "qwen" in cfg["models"]
    assert "oute" not in cfg["models"]  # TTS model never runs under llama-swap
    # And it must not leak into either lifecycle group.
    members = [n for g in cfg.get("groups", {}).values() for n in g["members"]]
    assert "oute" not in members


def test_tts_model_registered_as_audio_speech_in_litellm(tmp_path, monkeypatch):
    """A TTS model is exposed to LiteLLM as an audio_speech endpoint pointing at
    the inferhost-tts daemon, not as a chat model on llama-swap."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SWAP_PORT", "8080")
    monkeypatch.setenv("INFERHOST_TTS_PORT", "8092")
    reload_settings()

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="qwen.gguf", port=8081),
        Model(name="oute", repo_id="a/b", filename="oute.gguf", port=8082,
              vocoder_path="/tmp/wavtok.gguf"),
    ])
    entries = {e["model_name"]: e for e in render_litellm(reg)["model_list"]}

    tts = entries["oute"]
    assert tts["model_info"]["mode"] == "audio_speech"
    assert tts["litellm_params"]["api_base"] == "http://127.0.0.1:8092/v1"
    assert tts["litellm_params"]["model"] == "openai/oute"
    # The chat model still routes to llama-swap and has no audio_speech mode.
    assert entries["qwen"]["litellm_params"]["api_base"] == "http://127.0.0.1:8080/v1"
    assert entries["qwen"]["model_info"].get("mode") != "audio_speech"


def test_image_model_single_file_uses_sd_server(tmp_path, monkeypatch):
    """A single-file image model renders an sd-server cmd with -m and a
    checkEndpoint, and is registered as image_generation in litellm."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SWAP_PORT", "8080")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="sdxl", repo_id="a/b", filename="sdxl.gguf", kind="image",
              port=8085, local_path="/m/sdxl.gguf"),
    ])
    entry = render_llama_swap(reg)["models"]["sdxl"]
    cmd = entry["cmd"]
    assert "sd-server" in cmd
    assert "-m /m/sdxl.gguf" in cmd
    assert "--diffusion-model" not in cmd  # single-file path
    assert entry["checkEndpoint"] == "/v1/models"

    info = {e["model_name"]: e for e in render_litellm(reg)["model_list"]}["sdxl"]
    assert info["model_info"]["mode"] == "image_generation"
    assert info["litellm_params"]["api_base"] == "http://127.0.0.1:8080/v1"


def test_image_model_split_uses_diffusion_model_flags(tmp_path, monkeypatch):
    """A Flux/SD3 split image model passes --diffusion-model + encoder/VAE flags."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="flux", repo_id="c/d", filename="flux.gguf", kind="image",
              port=8086, local_path="/m/flux.gguf",
              vae_path="/m/ae.safetensors", clip_l_path="/m/clip_l.safetensors",
              t5xxl_path="/m/t5.safetensors"),
    ])
    cmd = render_llama_swap(reg)["models"]["flux"]["cmd"]
    assert "--diffusion-model /m/flux.gguf" in cmd
    assert "--vae /m/ae.safetensors" in cmd
    assert "--clip_l /m/clip_l.safetensors" in cmd
    assert "--t5xxl /m/t5.safetensors" in cmd
    assert " -m /m/flux.gguf" not in cmd  # split mode, not single-file


def test_image_model_qwen_text_encoder_uses_llm_flag(tmp_path, monkeypatch):
    """Z-Image / Qwen-Image carry a Qwen text encoder -> --llm (split load)."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="zimage", repo_id="leejet/Z-Image-Turbo-GGUF", filename="z.gguf",
              kind="image", port=8088, local_path="/m/z.gguf",
              vae_path="/m/ae.safetensors", text_encoder_path="/m/qwen.gguf"),
    ])
    cmd = render_llama_swap(reg)["models"]["zimage"]["cmd"]
    assert "--diffusion-model /m/z.gguf" in cmd
    assert "--vae /m/ae.safetensors" in cmd
    assert "--llm /m/qwen.gguf" in cmd  # current flag (not deprecated --qwen2vl)
    assert " -m /m/z.gguf" not in cmd   # split mode


def test_image_model_qwen_edit_vision_encoder_uses_llm_vision(tmp_path, monkeypatch):
    """Qwen-Image-Edit adds a vision ViT/mmproj via --llm_vision."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="qie", repo_id="QuantStack/Qwen-Image-Edit-GGUF", filename="qie.gguf",
              kind="image", port=8089, local_path="/m/qie.gguf",
              vae_path="/m/qvae.safetensors", text_encoder_path="/m/qwen-vl.gguf",
              vision_encoder_path="/m/mmproj.gguf"),
    ])
    cmd = render_llama_swap(reg)["models"]["qie"]["cmd"]
    assert "--llm /m/qwen-vl.gguf" in cmd
    assert "--llm_vision /m/mmproj.gguf" in cmd


def test_image_model_joins_swappable_group(tmp_path, monkeypatch):
    """Image models ride llama-swap and must be in the swappable group so they
    swap VRAM with LLMs (unlike TTS, which is excluded entirely)."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="q.gguf", port=8081, local_path="/m/q.gguf"),
        Model(name="sdxl", repo_id="a/b", filename="s.gguf", kind="image", port=8085, local_path="/m/s.gguf"),
    ])
    cfg = render_llama_swap(reg)
    assert "sdxl" in cfg["models"]
    assert set(cfg["groups"]["swappable"]["members"]) == {"qwen", "sdxl"}


def _dflash_model(**over):
    base = dict(
        name="qwen3-27b", repo_id="Qwen/Qwen3.6-27B", filename="qwen3.6-27b-Q4_K_M.gguf",
        port=8081, local_path="/tmp/target.gguf",
        draft_model_path="/tmp/draft.gguf", draft_repo_id="Alittlehammmer/x",
        draft_size_gib=0.9,
    )
    base.update(over)
    return Registry(models=[Model(**base)])


def test_dflash_emits_draft_flags_when_supported(tmp_path, monkeypatch):
    """A draft-attached model on a DFlash-capable binary emits --model-draft +
    --spec-type draft-dflash + --spec-draft-n-max (global default 4)."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    cmd = render_llama_swap(_dflash_model())["models"]["qwen3-27b"]["cmd"]
    assert "--model-draft /tmp/draft.gguf" in cmd
    assert "--spec-type draft-dflash" in cmd
    assert "--spec-draft-n-max 4" in cmd  # global spec_dflash_n_max default
    # ngram-mod stacks orthogonally with the draft lane.
    assert "--spec-type ngram-mod" in cmd


def test_dflash_beats_mtp_when_draft_attached(tmp_path, monkeypatch):
    """When a draft is attached, DFlash wins even if the filename says 'mtp' —
    draft-mtp must NOT be emitted (they're alternative drafting strategies)."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    reg = _dflash_model(filename="qwen3-mtp-Q4_K_M.gguf")
    cmd = render_llama_swap(reg)["models"]["qwen3-27b"]["cmd"]
    assert "draft-dflash" in cmd
    assert "draft-mtp" not in cmd


def test_dflash_per_model_override_and_zero_disables(tmp_path, monkeypatch):
    """spec_draft_n_max_override drives the DFlash depth: >=0 wins over the
    global, 0 disables the draft lane (no --model-draft) but leaves the model
    servable."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    tuned = render_llama_swap(_dflash_model(spec_draft_n_max_override=8))
    cmd = tuned["models"]["qwen3-27b"]["cmd"]
    assert "--spec-draft-n-max 8" in cmd
    assert "draft-dflash" in cmd

    off = render_llama_swap(_dflash_model(spec_draft_n_max_override=0))
    off_cmd = off["models"]["qwen3-27b"]["cmd"]
    assert "draft-dflash" not in off_cmd
    assert "--model-draft" not in off_cmd
    # Still a valid servable model (has --model and a port).
    assert "--model /tmp/target.gguf" in off_cmd


def test_dflash_global_setting_controls_depth(tmp_path, monkeypatch):
    """INFERHOST_SPEC_DFLASH_N_MAX sets the depth when no per-model override."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SPEC_DFLASH_N_MAX", "12")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    cmd = render_llama_swap(_dflash_model())["models"]["qwen3-27b"]["cmd"]
    assert "--spec-draft-n-max 12" in cmd


def test_dflash_unsupported_binary_emits_notice_not_flags(tmp_path, monkeypatch):
    """On a binary without draft-dflash, no dflash flags are rendered (would
    abort the swap entry) — a notice is emitted and the model serves draftless."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=False)

    notices: list[str] = []
    cmd = render_llama_swap(_dflash_model(), notices=notices)["models"]["qwen3-27b"]["cmd"]
    assert "draft-dflash" not in cmd
    assert "--model-draft" not in cmd
    assert "--model /tmp/target.gguf" in cmd  # still servable
    assert any("b9831" in n and "qwen3-27b" in n for n in notices)


def test_dflash_thinking_warning_notice(tmp_path, monkeypatch):
    """Draft attached + reasoning 'on' emits an acceptance-drop caveat notice;
    'auto' does not."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    on_notices: list[str] = []
    render_llama_swap(_dflash_model(reasoning="on"), notices=on_notices)
    assert any("acceptance drops" in n for n in on_notices)

    auto_notices: list[str] = []
    render_llama_swap(_dflash_model(reasoning="auto"), notices=auto_notices)
    assert not any("acceptance drops" in n for n in auto_notices)


def test_vision_model_suppresses_dflash_lane(tmp_path, monkeypatch):
    """A vision model (mmproj set) with a DFlash draft attached must NOT emit any
    draft-based speculative flags — llama-server aborts every image request with
    'failed to process speculative batch' when a draft context has to decode an
    image-expanded batch. The draft fields stay attached but no flags render; a
    notice explains it, and the model still serves with ngram-mod."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    notices: list[str] = []
    reg = _dflash_model(mmproj_path="/tmp/mmproj.gguf")
    cmd = render_llama_swap(reg, notices=notices)["models"]["qwen3-27b"]["cmd"]
    # No draft-based lane at all (external draft OR MTP heads).
    assert "draft-dflash" not in cmd
    assert "--model-draft" not in cmd
    assert "draft-mtp" not in cmd
    # mmproj is still attached and the model is still servable.
    assert "--mmproj /tmp/mmproj.gguf" in cmd
    assert "--model /tmp/target.gguf" in cmd
    # ngram-mod (model-free) is the one safe speculative lane and stays on.
    assert "--spec-type ngram-mod" in cmd
    # A clear notice names the model, DFlash, and disablement.
    assert any(
        "qwen3-27b" in n and "DFlash" in n and "disabled" in n for n in notices
    )


def test_vision_model_suppresses_mtp_lane(tmp_path, monkeypatch):
    """A vision model whose GGUF is MTP-capable (filename says 'mtp', no external
    draft) must also suppress the draft-mtp lane — MTP + vision corrupts slots /
    OOMs upstream. ngram-mod stays; notice names MTP."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SPEC_DRAFT_N_MAX", "2")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    notices: list[str] = []
    reg = Registry(models=[
        Model(name="qwen-mtp-vl", repo_id="x/y", filename="qwen-mtp.gguf",
              port=8081, local_path="/tmp/qwen-mtp.gguf",
              mmproj_path="/tmp/mmproj.gguf"),
    ])
    cmd = render_llama_swap(reg, notices=notices)["models"]["qwen-mtp-vl"]["cmd"]
    assert "draft-mtp" not in cmd
    assert "--model-draft" not in cmd
    assert "--mmproj /tmp/mmproj.gguf" in cmd
    assert "--spec-type ngram-mod" in cmd
    assert any(
        "qwen-mtp-vl" in n and "MTP" in n and "disabled" in n for n in notices
    )


def test_vision_toggle_off_restores_dflash_lane(tmp_path, monkeypatch):
    """vision_enabled=False on a projector-attached model drops --mmproj and
    lets the DFlash lane render again — the user's explicit trade of image
    input for draft speed. No vision-suppression notice fires."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    notices: list[str] = []
    reg = _dflash_model(mmproj_path="/tmp/mmproj.gguf", vision_enabled=False)
    cmd = render_llama_swap(reg, notices=notices)["models"]["qwen3-27b"]["cmd"]
    assert "--mmproj" not in cmd
    assert "--spec-type draft-dflash" in cmd
    assert "--model-draft /tmp/draft.gguf" in cmd
    assert "--spec-type ngram-mod" in cmd
    assert not any("vision model" in n for n in notices)


def test_vision_toggle_off_restores_mtp_lane(tmp_path, monkeypatch):
    """Same trade for an MTP-capable vision model with no external draft:
    vision off → no --mmproj, draft-mtp lane back on."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SPEC_DRAFT_N_MAX", "2")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    reg = Registry(models=[
        Model(name="qwen-mtp-vl", repo_id="x/y", filename="qwen-mtp.gguf",
              port=8081, local_path="/tmp/qwen-mtp.gguf",
              mmproj_path="/tmp/mmproj.gguf", vision_enabled=False),
    ])
    cmd = render_llama_swap(reg)["models"]["qwen-mtp-vl"]["cmd"]
    assert "--mmproj" not in cmd
    assert "--spec-type draft-mtp" in cmd
    assert "--spec-type ngram-mod" in cmd


def test_vision_toggle_off_unadvertises_vision_in_litellm(tmp_path, monkeypatch):
    """supports_vision follows the toggle, not just the projector: a client
    must not be told it can send images to a model served text-only."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()

    reg = Registry(models=[
        Model(name="vl-on", repo_id="x/y", filename="a.gguf", port=8081,
              mmproj_path="/tmp/mmproj.gguf"),
        Model(name="vl-off", repo_id="x/y", filename="b.gguf", port=8082,
              mmproj_path="/tmp/mmproj.gguf", vision_enabled=False),
    ])
    infos = {
        e["model_name"]: e["model_info"]
        for e in render_litellm(reg)["model_list"]
    }
    assert infos["vl-on"]["supports_vision"] is True
    assert infos["vl-off"]["supports_vision"] is False


def test_non_vision_draft_unchanged(tmp_path, monkeypatch):
    """Regression guard: the vision suppression must NOT touch a non-vision model
    — a draft-attached model with no mmproj still emits the full DFlash lane."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    _force_spec_support(monkeypatch, supported=True)

    notices: list[str] = []
    cmd = render_llama_swap(
        _dflash_model(), notices=notices
    )["models"]["qwen3-27b"]["cmd"]
    assert "--spec-type draft-dflash" in cmd
    assert "--model-draft /tmp/draft.gguf" in cmd
    assert not any("vision model" in n for n in notices)


def test_mtp_path_unchanged_with_no_draft(tmp_path, monkeypatch):
    """Regression guard: a model with NO draft attached renders exactly the same
    MTP + ngram-mod lanes as before DFlash existed."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SPEC_DRAFT_N_MAX", "2")
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    # Even on a DFlash-capable binary, a model with no draft uses the MTP lane.
    _force_spec_support(monkeypatch, supported=True)

    reg = Registry(models=[
        Model(name="qwen-mtp", repo_id="x/y", filename="qwen-mtp.gguf",
              port=8081, local_path="/tmp/qwen-mtp.gguf"),
    ])
    cmd = render_llama_swap(reg)["models"]["qwen-mtp"]["cmd"]
    assert "--spec-type draft-mtp" in cmd
    assert "--spec-draft-n-max 2" in cmd
    assert "--spec-type ngram-mod" in cmd
    # No DFlash artifacts leak into a draftless model.
    assert "draft-dflash" not in cmd
    assert "--model-draft" not in cmd


def test_max_output_tokens_cap(tmp_path, monkeypatch):
    """max_output_tokens advertises the full window by default (0) and a capped
    value when set, while max_input_tokens always stays at the full window."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_MAX_OUTPUT_TOKENS", "8192")
    reload_settings()

    reg = Registry(models=[
        Model(name="qwen", repo_id="x/y", filename="x.gguf", port=8081, ctx=65536),
    ])
    info = render_litellm(reg)["model_list"][0]["model_info"]
    assert info["max_input_tokens"] == 65536  # full window for the prompt
    assert info["max_output_tokens"] == 8192  # capped completion


def _pin_guest_registry():
    return Registry(models=[
        Model(name="big-pin", repo_id="x/p", filename="p.gguf", port=8081,
              local_path="/m/p.gguf", pin=True, size_gib=18.0),
        Model(name="guest", repo_id="x/g", filename="g.gguf", port=8082,
              local_path="/m/g.gguf", size_gib=7.0),
    ])


def test_groups_cofit_lets_guests_load_beside_pins(tmp_path, monkeypatch):
    """When pinned + every swappable model fit in VRAM together, the swappable
    group is NOT exclusive, so loading a guest never evicts the pins."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    from inferhost.core import vram
    monkeypatch.setattr(vram, "total_vram_gib", lambda gpu_index=0: 24.0)
    monkeypatch.setattr(vram, "estimate_model_vram_gib", lambda m, slots=None: 5.0)

    cfg = render_llama_swap(_pin_guest_registry())
    assert cfg["groups"]["pinned"]["exclusive"] is False
    assert cfg["groups"]["swappable"]["exclusive"] is False
    assert cfg["groups"]["swappable"]["swap"] is True


def test_groups_oversized_guest_swaps_both_directions(tmp_path, monkeypatch):
    """A swappable model that can't co-fit beside the pinned set makes BOTH
    groups exclusive (symmetric graceful eviction, no VRAM fight) and emits a
    notice naming the oversized model."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    from inferhost.core import vram
    monkeypatch.setattr(vram, "total_vram_gib", lambda gpu_index=0: 24.0)
    monkeypatch.setattr(vram, "estimate_model_vram_gib", lambda m, slots=None: 13.0)

    notices: list[str] = []
    cfg = render_llama_swap(_pin_guest_registry(), notices=notices)
    assert cfg["groups"]["pinned"]["exclusive"] is True
    assert cfg["groups"]["swappable"]["exclusive"] is True
    assert any("guest" in n and "big-pin" in n for n in notices)


def test_groups_no_gpu_info_keeps_legacy_flags(tmp_path, monkeypatch):
    """Without GPU info the co-fit question is unanswerable — keep the
    historical flags (swappable exclusive, pinned not)."""
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))
    from inferhost.core import vram
    monkeypatch.setattr(vram, "total_vram_gib", lambda gpu_index=0: 0.0)

    cfg = render_llama_swap(_pin_guest_registry())
    assert cfg["groups"]["pinned"]["exclusive"] is False
    assert cfg["groups"]["swappable"]["exclusive"] is True


def test_orpheus_tts_swap_fronted_pinnable_and_routed_to_tts_daemon(tmp_path, monkeypatch):
    """Orpheus is the one TTS engine llama-swap fronts: its GGUF gets a
    llama-server entry (so swapping AND pinning work like a chat model's),
    while LiteLLM still routes its /v1/audio/speech to inferhost-tts, which
    decodes the generated SNAC tokens. Kokoro/OuteTTS stay out of llama-swap.
    """
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()
    _force_supported_cache_types(monkeypatch, frozenset({"f16", "q8_0"}))

    reg = Registry(models=[
        Model(name="orpheus", repo_id="unsloth/orpheus-3b-0.1-ft-GGUF",
              filename="orpheus-3b-0.1-ft-Q4_K_M.gguf", quant="Q4_K_M",
              port=8085, size_gib=2.3, local_path="/tmp/orpheus.gguf",
              vocoder_path="/tmp/decoder_model.onnx", pin=True),
        Model(name="kokoro", repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
              filename="onnx/model.onnx", port=8086, size_gib=0.3,
              local_path="/tmp/model.onnx", vocoder_path="/tmp/voices.npz",
              pin=True),
        Model(name="oute", repo_id="OuteAI/OuteTTS-0.2-500M-GGUF",
              filename="OuteTTS-0.2-500M-Q8_0.gguf", port=8087, size_gib=0.6,
              local_path="/tmp/oute.gguf", vocoder_path="/tmp/wavtokenizer.gguf"),
    ])
    cfg = render_llama_swap(reg)
    assert "orpheus" in cfg["models"]
    assert "kokoro" not in cfg["models"]
    assert "oute" not in cfg["models"]
    # Pinned: ttl=0 + membership in the pinned (swap: false) group.
    assert cfg["models"]["orpheus"]["ttl"] == 0
    assert cfg["groups"]["pinned"]["members"] == ["orpheus"]
    # llama-server (not sd-server / llama-tts) serves the GGUF.
    assert "/tmp/orpheus.gguf" in cfg["models"]["orpheus"]["cmd"]

    # All three are audio_speech endpoints pointed at inferhost-tts.
    from inferhost.settings import settings
    tts_base = f"http://127.0.0.1:{settings().tts_port}/v1"
    ml = render_litellm(reg)["model_list"]
    for name in ("orpheus", "kokoro", "oute"):
        entry = next(e for e in ml if e["model_name"] == name)
        assert entry["model_info"]["mode"] == "audio_speech"
        assert entry["litellm_params"]["api_base"] == tts_base


# ---- per-model binary fallback (custom build too old for a model) ----

def _arch_env(monkeypatch, tmp_path, *, custom_archs, managed_archs):
    """Point the custom + managed binaries at fakes carrying the given arch tables."""
    from inferhost import settings as settings_mod
    from inferhost.core import gguf, paths

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    custom = tmp_path / "mybuild" / "llama-server"
    custom.parent.mkdir(parents=True, exist_ok=True)
    for path, archs in ((custom, custom_archs), (bin_dir / "llama-server", managed_archs)):
        path.write_bytes(b"\x00".join(a.encode() for a in archs) + b"\x00")
    monkeypatch.setattr(paths, "bin_dir", lambda: bin_dir)
    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", str(custom))
    settings_mod.reload_settings()
    monkeypatch.setattr(gguf, "architecture_cached", lambda _p: "muse-glimmer")
    return custom, bin_dir / "llama-server"


def test_model_falls_back_to_the_managed_binary_for_a_new_architecture(
        monkeypatch, tmp_path):
    """A custom build is frozen at the commit the user compiled. When a model's
    architecture landed upstream later, serving it with the managed binary
    keeps it running instead of crash-looping on "unknown model architecture"."""
    from inferhost import settings as settings_mod
    from inferhost.core import configs
    from inferhost.core.registry import Model

    custom, managed = _arch_env(
        monkeypatch, tmp_path,
        custom_archs=["llama", "qwen3"], managed_archs=["llama", "muse-glimmer"])
    try:
        notices: list[str] = []
        m = Model(name="muse", repo_id="x", filename="m.gguf", local_path="/m.gguf")
        assert configs.server_binary(m, notices) == managed
        assert any("muse-glimmer" in n for n in notices)
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


def test_custom_binary_is_kept_when_it_knows_the_architecture(monkeypatch, tmp_path):
    """The fallback is per-model and only for what the custom build can't do —
    everything else still runs on the user's build."""
    from inferhost import settings as settings_mod
    from inferhost.core import configs
    from inferhost.core.registry import Model

    custom, _ = _arch_env(
        monkeypatch, tmp_path,
        custom_archs=["llama", "muse-glimmer"], managed_archs=["llama"])
    try:
        notices: list[str] = []
        m = Model(name="muse", repo_id="x", filename="m.gguf", local_path="/m.gguf")
        assert configs.server_binary(m, notices) == custom
        assert notices == []
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


def test_custom_binary_is_kept_when_neither_knows_the_architecture(
        monkeypatch, tmp_path):
    """If the managed binary is no better, don't silently switch engines — let
    the error come from the binary the user chose."""
    from inferhost import settings as settings_mod
    from inferhost.core import configs
    from inferhost.core.registry import Model

    custom, _ = _arch_env(
        monkeypatch, tmp_path,
        custom_archs=["llama"], managed_archs=["llama"])
    try:
        notices: list[str] = []
        m = Model(name="muse", repo_id="x", filename="m.gguf", local_path="/m.gguf")
        assert configs.server_binary(m, notices) == custom
        assert notices == []
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


def test_model_falls_back_to_the_managed_binary_for_a_new_weight_format(
        monkeypatch, tmp_path):
    """Same fallback, one table over: an NVFP4 GGUF needs a build carrying that
    ggml tensor type. A hand-built llama.cpp from before it landed knows the
    architecture and still aborts with "unknown type N"."""
    from inferhost import settings as settings_mod
    from inferhost.core import configs
    from inferhost.core.registry import Model

    custom, managed = _arch_env(
        monkeypatch, tmp_path,
        custom_archs=["llama", "muse-glimmer", "q4_K"],
        managed_archs=["llama", "muse-glimmer", "q4_K", "nvfp4"])
    try:
        notices: list[str] = []
        m = Model(name="qwen", repo_id="x", filename="q-NVFP4.gguf", local_path="/q.gguf")
        assert configs.server_binary(m, notices) == managed
        assert any("nvfp4" in n for n in notices)
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


def test_custom_binary_is_kept_for_a_weight_format_it_already_has(monkeypatch, tmp_path):
    from inferhost import settings as settings_mod
    from inferhost.core import configs
    from inferhost.core.registry import Model

    custom, _ = _arch_env(
        monkeypatch, tmp_path,
        custom_archs=["llama", "muse-glimmer", "nvfp4"],
        managed_archs=["llama", "muse-glimmer"])
    try:
        notices: list[str] = []
        m = Model(name="qwen", repo_id="x", filename="q-NVFP4.gguf", local_path="/q.gguf")
        assert configs.server_binary(m, notices) == custom
        assert notices == []
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()
