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


def _kv_cache_estimate_gib(m: Model) -> float:
    """Very rough KV cache estimate.

    Assumes n_layers ~ scales with model size and TurboQuant ~4.9x compression.
    """
    # Layers heuristic: small models ~28, mid ~32-40, large ~80
    if m.size_gib < 2:
        n_layers = 24
    elif m.size_gib < 8:
        n_layers = 32
    elif m.size_gib < 30:
        n_layers = 48
    else:
        n_layers = 80
    # Per-token KV: 2 (K+V) * n_layers * hidden_dim_guess * 2 bytes (fp16) / turboquant_compression
    hidden_dim = 4096 if m.size_gib >= 4 else 2048
    bytes_per_token = 2 * n_layers * hidden_dim * 2 / 4.9
    return (bytes_per_token * m.ctx) / (1024 ** 3)


def estimate_model_vram_gib(m: Model) -> float:
    """Conservative VRAM cost for hosting model `m` with its declared ctx."""
    return m.size_gib * 1.05 + _kv_cache_estimate_gib(m)


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
