"""VRAM estimation helpers.

Estimates per-model VRAM requirements and checks whether a new model can
be pinned without exceeding available GPU memory.

NOTE: This module is a stub — Agent A will supply the full implementation.
The signatures below are the contract the TUI depends on.
"""
from __future__ import annotations

from inferhost.core.registry import Model, Registry
from inferhost.core import processes


def estimate_model_vram_gib(m: Model) -> float:
    """Return estimated VRAM usage in GiB for model *m*.

    Falls back to ``m.size_gib * 1.15`` (weights + KV cache headroom) when
    a better estimate is not available.
    """
    if m.size_gib > 0:
        return m.size_gib * 1.15
    return 0.0


def pinned_vram_estimate(reg: Registry) -> float:
    """Return the total estimated VRAM for all pinned models in *reg*."""
    return sum(estimate_model_vram_gib(m) for m in reg.models if m.pin)


def free_vram_gib(gpu_index: int = 0) -> float:
    """Return free VRAM in GiB for the given GPU index via nvidia-smi."""
    gpus = processes.query_gpus()
    for g in gpus:
        if g.index == gpu_index:
            used = g.mem_used_mib / 1024
            total = g.mem_total_mib / 1024
            return max(0.0, total - used)
    return 0.0


def can_pin(
    reg: Registry, m: Model, gpu_index: int = 0
) -> tuple[bool, float, float]:
    """Check whether model *m* can be pinned without exhausting VRAM.

    Returns ``(ok, needed_gib, free_gib)``.
    ``ok`` is False when the model cannot be pinned because there is not
    enough free VRAM to accommodate it alongside the already-pinned models.
    """
    needed = estimate_model_vram_gib(m)
    free = free_vram_gib(gpu_index)
    return (needed <= free, needed, free)
