from inferhost.core.configs import render_litellm, render_llama_swap
from inferhost.core.registry import Model, Registry
from inferhost.settings import reload_settings


def test_render_llama_swap_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    reload_settings()

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

    # Asymmetric KV: TurboQuant guidance is K=q8_0, V=turbo3 by default.
    # Both flags appear, with different values per the authors' rec.
    assert "-ctk q8_0" in cmd
    assert "-ctv turbo3" in cmd


def test_render_litellm_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_SWAP_PORT", "8080")
    reload_settings()

    reg = Registry(models=[
        Model(name="qwen", repo_id="Qwen/x", filename="x.gguf", port=8081),
    ])
    cfg = render_litellm(reg)
    assert len(cfg["model_list"]) == 1
    entry = cfg["model_list"][0]
    assert entry["model_name"] == "qwen"

    # LiteLLM proxies to llama-swap on loopback — must never be an external address
    api_base = entry["litellm_params"]["api_base"]
    assert api_base == "http://127.0.0.1:8080/v1"
    assert "127.0.0.1" in api_base
