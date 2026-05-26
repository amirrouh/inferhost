from inferhost.core.configs import render_litellm, render_llama_swap
from inferhost.core.registry import Model, Registry
from inferhost.settings import reload_settings


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
