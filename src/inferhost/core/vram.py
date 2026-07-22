"""VRAM feasibility checks for pinning models.

The numbers here are rough estimates — exact VRAM cost depends on the
model architecture (n_layers, head_dim, n_kv_heads), the quant of the
weights, KV cache quant, batch/parallel slots, and whatever scratch buffers
llama-server allocates. We just need a 'would this fit?' gate that's
conservative enough to prevent obvious OOMs.
"""
from __future__ import annotations

from inferhost.core import processes
from inferhost.core.registry import Model, Registry

# Approximate bytes per element for each cache-type value we support.
# Doesn't have to be exact — just in the right ballpark so the pinned-overflow
# warning doesn't trip on a phantom 20 GiB of KV that doesn't actually exist.
_KV_QUANT_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q5_0": 0.6875, "q5_1": 0.75, "q4_0": 0.5625, "q4_1": 0.625,
    "iq4_nl": 0.5,
    "off": 2.0,
}


def _kv_cache_estimate_gib(m: Model) -> float:
    """Rough KV cache estimate that respects the configured K/V quant."""
    # Layers heuristic: small models ~28, mid ~32-40, large ~80
    if m.size_gib < 2:
        n_layers = 24
    elif m.size_gib < 8:
        n_layers = 32
    elif m.size_gib < 30:
        n_layers = 48
    else:
        n_layers = 80
    hidden_dim = 4096 if m.size_gib >= 4 else 2048
    try:
        from inferhost.settings import settings as _settings
        s = _settings()
        k_bytes = _KV_QUANT_BYTES.get(getattr(s, "kv_quant_k", "q8_0"), 1.0625)
        v_bytes = _KV_QUANT_BYTES.get(getattr(s, "kv_quant_v", "q8_0"), 1.0625)
    except Exception:  # noqa: BLE001
        k_bytes, v_bytes = 1.0625, 1.0625
    bytes_per_token = n_layers * hidden_dim * (k_bytes + v_bytes)
    return (bytes_per_token * m.ctx) / (1024 ** 3)


def estimate_model_vram_gib(m: Model) -> float:
    """Conservative VRAM cost for hosting model `m` with its declared ctx.

    Includes the DFlash draft weights (plus its small KV cache) when one is
    attached — the draft is co-resident with the target during serving, so its
    footprint has to count toward pin feasibility and the dashboard estimate.
    """
    draft = m.draft_size_gib * 1.1 if m.draft_size_gib > 0 else 0.0
    return m.size_gib * 1.05 + _kv_cache_estimate_gib(m) + draft


def pinned_vram_estimate(reg: Registry) -> float:
    return sum(estimate_model_vram_gib(m) for m in reg.models if m.pin)


def free_vram_gib(gpu_index: int = 0) -> float:
    gpus = processes.query_gpus()
    if not gpus:
        return float("inf")  # No GPU info => don't block the user
    for g in gpus:
        if g.index == gpu_index:
            return (g.mem_total_mib - g.mem_used_mib) / 1024
    return float("inf")


def total_vram_gib(gpu_index: int = 0) -> float:
    """Total capacity of the given GPU in GiB, or 0.0 when no GPU info exists.

    Capacity (not current free space) is the right input for config-render
    decisions: whether a swappable model can ever co-reside with the pinned
    set is a property of the card, independent of what happens to be loaded
    at render time.
    """
    for g in processes.query_gpus():
        if g.index == gpu_index:
            return g.mem_total_mib / 1024
    return 0.0


def can_pin(reg: Registry, m: Model, gpu_index: int = 0) -> tuple[bool, float, float]:
    """Returns (ok, needed_gib, free_gib).

    If the model is already pinned, returns (True, 0, free). Otherwise compares
    estimate_model_vram_gib(m) against free_vram_gib.
    """
    if m.pin:
        return True, 0.0, free_vram_gib(gpu_index)
    needed = estimate_model_vram_gib(m)
    free = free_vram_gib(gpu_index)
    return (needed <= free, needed, free)
