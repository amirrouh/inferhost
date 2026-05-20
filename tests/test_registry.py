from pathlib import Path

import inferhost.core.paths as paths_mod
from inferhost.core.registry import Model, Registry, load, save
from inferhost.settings import reload_settings


def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path / "data"))
    reload_settings()

    reg = Registry()
    reg.add(Model(name="m1", repo_id="org/model", filename="m1.Q4_K_M.gguf", quant="Q4_K_M", ctx=4096, port=8081, size_gib=4.0, local_path="/tmp/m1.gguf"))
    reg.add(Model(name="m2", repo_id="org/model2", filename="m2.Q5_K_M.gguf", quant="Q5_K_M", ctx=8192, port=8082, size_gib=4.8, local_path="/tmp/m2.gguf"))
    save(reg)

    reg2 = load()
    assert {m.name for m in reg2.models} == {"m1", "m2"}
    m1 = reg2.get("m1")
    assert m1 is not None
    assert m1.quant == "Q4_K_M"
    assert m1.port == 8081


def test_registry_next_port():
    reg = Registry(models=[
        Model(name="a", repo_id="x/a", filename="a.gguf", port=8081),
        Model(name="b", repo_id="x/b", filename="b.gguf", port=8082),
    ])
    assert reg.next_port(8080) == 8083


def test_registry_remove():
    reg = Registry(models=[
        Model(name="a", repo_id="x/a", filename="a.gguf"),
    ])
    assert reg.remove("a") is True
    assert reg.remove("missing") is False
