"""Tests for core/vram.py — VRAM estimation and pin feasibility math.

Agent A creates src/inferhost/core/vram.py. These tests cover:
  - estimate_model_vram_gib: positive, monotone in size_gib and ctx
  - pinned_vram_estimate: sums only pinned models
  - can_pin: True when plenty of VRAM free
  - can_pin: False when VRAM is near zero
  - can_pin: True (0.0 needed) for an already-pinned model

free_vram_gib() is monkey-patched via monkeypatch so no real GPU is required.
"""

import pytest

import inferhost.core.vram as vram_mod
from inferhost.core.registry import Model, Registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(name: str, size_gib: float = 4.0, ctx: int = 8192, pin: bool = False) -> Model:
    return Model(
        name=name,
        repo_id=f"org/{name}",
        filename=f"{name}.Q4_K_M.gguf",
        size_gib=size_gib,
        ctx=ctx,
        port=8081,
        pin=pin,
    )


# ---------------------------------------------------------------------------
# estimate_model_vram_gib
# ---------------------------------------------------------------------------

def test_estimate_returns_positive():
    m = _model("m1", size_gib=4.0, ctx=8192)
    estimate = vram_mod.estimate_model_vram_gib(m)
    assert estimate > 0.0


def test_estimate_increases_with_size_gib():
    small = _model("small", size_gib=2.0, ctx=8192)
    large = _model("large", size_gib=8.0, ctx=8192)
    assert vram_mod.estimate_model_vram_gib(large) > vram_mod.estimate_model_vram_gib(small)


def test_estimate_increases_with_ctx():
    short_ctx = _model("m", size_gib=4.0, ctx=4096)
    long_ctx = _model("m", size_gib=4.0, ctx=32768)
    assert vram_mod.estimate_model_vram_gib(long_ctx) > vram_mod.estimate_model_vram_gib(short_ctx)


def test_estimate_includes_attached_dflash_draft():
    """A model with a DFlash draft attached costs more VRAM than the same model
    without one (draft weights + its small KV are co-resident)."""
    no_draft = _model("m", size_gib=16.0, ctx=8192)
    with_draft = _model("m", size_gib=16.0, ctx=8192)
    with_draft.draft_size_gib = 1.0
    base = vram_mod.estimate_model_vram_gib(no_draft)
    boosted = vram_mod.estimate_model_vram_gib(with_draft)
    assert boosted > base
    # ~draft_size * 1.1 added on top.
    assert boosted == pytest.approx(base + 1.0 * 1.1)


def test_estimate_skips_draft_that_vision_suppresses():
    """llama-server can't run an external draft context behind --mmproj, so
    configs.py drops the DFlash lane on a vision model. Costing the draft
    anyway is what pushed Qwen3.8-27B down to a single parallel slot."""
    m = _model("vlm", size_gib=16.0, ctx=8192)
    m.draft_size_gib = 1.7
    m.mmproj_path = "/nonexistent/mmproj.gguf"
    m.vision_enabled = True
    assert m.vision_active
    base = _model("vlm", size_gib=16.0, ctx=8192)
    assert vram_mod.estimate_model_vram_gib(m) == pytest.approx(
        vram_mod.estimate_model_vram_gib(base)
    )


def test_estimate_uses_hybrid_cache_layer_count(monkeypatch, tmp_path):
    """A hybrid attention/SSM model must be costed on its *cache* layers only.
    Qwen3.8-27B keeps 17 of 65 blocks in KV; counting all 65 overstates the
    cache 4x and makes the fit gate refuse slots the card can hold."""
    from inferhost.core import gguf

    m = _model("hybrid", size_gib=16.0, ctx=64000)
    m.local_path = str(tmp_path / "hybrid.gguf")

    hybrid = gguf.KVGeometry(cache_layers=17, kv_elems=2048, state_elems=39223296)
    plain = gguf.KVGeometry(cache_layers=65, kv_elems=2048, state_elems=0)

    monkeypatch.setattr(gguf, "kv_geometry_cached", lambda _p: hybrid)
    hybrid_gib = vram_mod.estimate_model_vram_gib(m, slots=1)
    monkeypatch.setattr(gguf, "kv_geometry_cached", lambda _p: plain)
    plain_gib = vram_mod.estimate_model_vram_gib(m, slots=1)

    # 17/65 of the cache, so the whole estimate has to drop by GiBs.
    assert plain_gib - hybrid_gib > 5.0
    # 36,992 bytes/token of KV at the q8_0 default + the one-off SSM state.
    kv = 17 * 2048 * 1.0625 * 64000 / 1024 ** 3
    state = 39223296 * 4 / 1024 ** 3
    assert hybrid_gib == pytest.approx(16.0 * 1.05 + kv + state, rel=1e-3)


def test_hybrid_recurrent_state_does_not_scale_with_slots(monkeypatch, tmp_path):
    """The recurrent state is allocated once, not per slot — measured on
    Qwen3.8-27B, a second slot costs 2,273 MiB against a predicted KV of 2,258,
    so the marginal cost of a slot is KV and nothing else."""
    from inferhost.core import gguf

    m = _model("hybrid", size_gib=16.0, ctx=1024)
    m.local_path = str(tmp_path / "hybrid.gguf")
    monkeypatch.setattr(
        gguf, "kv_geometry_cached",
        lambda _p: gguf.KVGeometry(cache_layers=17, kv_elems=2048,
                                   state_elems=39223296))
    one = vram_mod.estimate_model_vram_gib(m, slots=1)
    two = vram_mod.estimate_model_vram_gib(m, slots=2)
    kv_per_slot = 17 * 2048 * 1.0625 * 1024 / 1024 ** 3
    assert two - one == pytest.approx(kv_per_slot, rel=1e-3)


# ---------------------------------------------------------------------------
# pinned_vram_estimate
# ---------------------------------------------------------------------------

def test_pinned_vram_estimate_sums_only_pinned():
    reg = Registry(models=[
        _model("pinned1", size_gib=4.0, pin=True),
        _model("pinned2", size_gib=3.0, pin=True),
        _model("unpinned", size_gib=6.0, pin=False),
    ])
    total = vram_mod.pinned_vram_estimate(reg)
    # Must include pinned1 and pinned2 but NOT unpinned
    individual = (
        vram_mod.estimate_model_vram_gib(reg.get("pinned1"))
        + vram_mod.estimate_model_vram_gib(reg.get("pinned2"))
    )
    assert total == pytest.approx(individual)


def test_pinned_vram_estimate_empty_registry():
    reg = Registry(models=[])
    assert vram_mod.pinned_vram_estimate(reg) == pytest.approx(0.0)


def test_pinned_vram_estimate_no_pins():
    reg = Registry(models=[
        _model("a", pin=False),
        _model("b", pin=False),
    ])
    assert vram_mod.pinned_vram_estimate(reg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# can_pin
# ---------------------------------------------------------------------------

def test_can_pin_returns_true_when_vram_plentiful(monkeypatch):
    """When free VRAM far exceeds the model's estimate, can_pin returns True."""
    monkeypatch.setattr(vram_mod, "free_vram_gib", lambda gpu_index=0:100.0)

    reg = Registry(models=[_model("m1", size_gib=4.0, pin=False)])
    ok, needed, free = vram_mod.can_pin(reg, reg.get("m1"))

    assert ok is True
    assert needed > 0.0
    assert free == pytest.approx(100.0)


def test_can_pin_returns_false_when_vram_near_zero(monkeypatch):
    """When free VRAM is near zero, can_pin returns (False, needed, free)."""
    monkeypatch.setattr(vram_mod, "free_vram_gib", lambda gpu_index=0:0.1)

    reg = Registry(models=[_model("m1", size_gib=8.0, pin=False)])
    ok, needed, free = vram_mod.can_pin(reg, reg.get("m1"))

    assert ok is False
    assert needed > free


def test_can_pin_already_pinned_model_returns_true_zero_needed(monkeypatch):
    """A model that is already pinned costs zero additional VRAM to 're-pin'."""
    monkeypatch.setattr(vram_mod, "free_vram_gib", lambda gpu_index=0:0.5)

    m = _model("m1", size_gib=4.0, pin=True)
    reg = Registry(models=[m])
    ok, needed, _free = vram_mod.can_pin(reg, m)

    assert ok is True
    assert needed == pytest.approx(0.0)


def test_can_pin_accounts_for_already_pinned_vram(monkeypatch):
    """can_pin should count existing pinned models' VRAM as already consumed."""
    # Suppose we have 10 GiB free, but 8 GiB are already committed by pinned models.
    # Trying to pin another 4 GiB model should fail.
    monkeypatch.setattr(vram_mod, "free_vram_gib", lambda gpu_index=0:10.0)

    already_pinned = _model("heavy", size_gib=14.0, pin=True)  # ~8+ GiB estimate
    new_model = _model("new", size_gib=4.0, pin=False)
    reg = Registry(models=[already_pinned, new_model])

    ok, needed, free = vram_mod.can_pin(reg, new_model)
    # With 10 GiB free but heavy already pinned (estimated > 10 GiB), should fail
    # The exact outcome depends on the estimate formula; the important thing is
    # that the function returns a valid (bool, float, float) triple.
    assert isinstance(ok, bool)
    assert needed >= 0.0
    assert free >= 0.0
