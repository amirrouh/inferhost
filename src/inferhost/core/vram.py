"""VRAM feasibility checks for pinning models.

The numbers here are rough estimates — exact VRAM cost depends on the
model architecture (n_layers, head_dim, n_kv_heads), the quant of the
weights, KV cache quant, batch/parallel slots, and whatever scratch buffers
llama-server allocates. We just need a 'would this fit?' gate that's
conservative enough to prevent obvious OOMs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

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


def requested_parallel_slots(m: Model) -> int:
    """Slots the user asked for: per-model override wins, 0 inherits global."""
    try:
        from inferhost.settings import settings as _settings
        s = _settings()
        return max(1, m.parallel_slots if m.parallel_slots > 0 else s.parallel_slots)
    except Exception:  # noqa: BLE001
        return 1


def _kv_cache_estimate_gib(m: Model, slots: int) -> float:
    """Rough KV cache estimate that respects the configured K/V quant.

    Scales with ``slots``: llama-server gives each parallel slot its own full
    context window (we size ``-c`` as ctx x slots precisely so it does), so N
    slots cost N times the cache.

    Reads the real attention geometry out of the GGUF when it can. The old
    size-based heuristic assumed full multi-head attention and so overstated
    the cache several-fold on any grouped-query model (i.e. all of them) — it
    put a 27B at 64k context over a 24 GiB card that in fact runs it in 20.5.
    """
    from inferhost.core import gguf  # local: gguf imports nothing from here

    try:
        from inferhost.settings import settings as _settings
        s = _settings()
        k_bytes = _KV_QUANT_BYTES.get(getattr(s, "kv_quant_k", "q8_0"), 1.0625)
        v_bytes = _KV_QUANT_BYTES.get(getattr(s, "kv_quant_v", "q8_0"), 1.0625)
    except Exception:  # noqa: BLE001
        k_bytes, v_bytes = 1.0625, 1.0625

    geom = gguf.kv_geometry_cached(m.local_path) if m.local_path else None
    state_gib = 0.0
    if geom:
        # kv_elems counts K and V elements together; both sides are usually
        # quantized alike, so the mean byte width is the right multiplier.
        # cache_layers already excludes the recurrent layers of a hybrid stack.
        bytes_per_token = geom.cache_layers * geom.kv_elems * ((k_bytes + v_bytes) / 2)
        # Hybrid models also hold a recurrent state for their non-attention
        # layers. It neither grows with context nor scales with slots: measured
        # on Qwen3.8-27B, the marginal cost of a second slot is 2,273 MiB
        # against a predicted KV of 2,258 — i.e. KV and nothing else.
        state_gib = (geom.state_elems * 4.0) / (1024 ** 3)
        state_gib += _mtp_overhead_gib(m, slots, geom.kv_elems)
    else:
        # Fallback when the file isn't on disk yet (pre-download sizing) or the
        # metadata is unreadable. Deliberately coarse — see docstring.
        if m.size_gib < 2:
            n_layers = 24
        elif m.size_gib < 8:
            n_layers = 32
        elif m.size_gib < 30:
            n_layers = 48
        else:
            n_layers = 80
        # Assume 8 KV heads x 128 head dim (the common GQA shape) rather than
        # full MHA, so the fallback errs far less wildly.
        bytes_per_token = n_layers * 8 * 128 * (k_bytes + v_bytes)
    return (bytes_per_token * m.ctx * max(1, slots)) / (1024 ** 3) + state_gib


def _mtp_overhead_gib(m: Model, slots: int, kv_elems: int) -> float:
    """VRAM the MTP speculative lane adds on top of the target model.

    ``--spec-type draft-mtp`` builds a *second* context against the same
    weights, holding the MTP block's own KV cache (one layer, f16 — the
    ``-ctk``/``-ctv`` values do not propagate to the draft context) plus a
    per-draft-token graph arena. On a 24 GiB card this is the difference
    between a model that loads and one that aborts on the way up, so the fit
    gate has to see it.

    The three terms are fitted to 15 measured configurations of Qwen3.8-27B at
    64k (draft depth 1-8, one and two slots) and reproduce each within ~2%.
    """
    if not _mtp_will_run(m):
        return 0.0
    depth = _effective_draft_depth(m)
    if depth <= 0:
        return 0.0
    slots = max(1, slots)
    # The MTP block's own context: one cache layer, f16, full context, per slot.
    draft_kv = (kv_elems * 2.0 * m.ctx * slots) / (1024 ** 3)
    # Graph/scratch arena, which scales with how many tokens are drafted.
    arena = 0.15 * depth * slots
    # Fixed cost of standing the second context up at all.
    return draft_kv + arena + 0.4


def _mtp_will_run(m: Model) -> bool:
    """Whether configs.py will actually emit ``--spec-type draft-mtp``.

    Mirrors the lane ladder in :func:`configs._llama_server_cmd` — which we
    can't import here (configs imports this module) — so keep the two in step:
    an attached DFlash draft wins, except behind ``--mmproj`` where llama-server
    can't run an external draft context and MTP takes over.
    """
    from inferhost.core import gguf

    if not m.local_path or not gguf.has_mtp_heads_cached(m.local_path):
        return False
    return not (m.draft_model_path and not m.vision_active)


def _effective_draft_depth(m: Model) -> int:
    """``--spec-draft-n-max`` for the MTP lane: per-model override, else global."""
    if m.spec_draft_n_max_override >= 0:
        return m.spec_draft_n_max_override
    try:
        from inferhost.settings import settings as _settings
        return _settings().spec_draft_n_max
    except Exception:  # noqa: BLE001
        return 2


def _mmproj_gib(m: Model) -> float:
    """Size of the multimodal projector, 0 when vision isn't actually served."""
    if not m.vision_active:
        return 0.0
    try:
        return os.path.getsize(m.mmproj_path) / (1024 ** 3)
    except OSError:
        return 0.0


def _estimate_with_slots(m: Model, slots: int) -> float:
    """Cost of `m` at an explicit slot count. Private so the fit loop below is
    unaffected by tests/callers that stub out `estimate_model_vram_gib`."""
    # A DFlash draft is co-resident with its target — but llama-server can't
    # run an external draft context behind --mmproj, so configs.py suppresses
    # the lane on a vision model and the draft weights never reach VRAM. Costing
    # them anyway is what pushed this model down to one slot.
    draft = (m.draft_size_gib * 1.1
             if m.draft_size_gib > 0 and not m.vision_active else 0.0)
    # The vision projector is loaded alongside the weights when vision is on
    # (~0.9 GiB on a 27B) and is otherwise invisible to size_gib.
    return (m.size_gib * 1.05 + _kv_cache_estimate_gib(m, slots) + draft
            + _mmproj_gib(m))


def estimate_model_vram_gib(m: Model, slots: int | None = None) -> float:
    """Conservative VRAM cost for hosting model `m` with its declared ctx.

    ``slots`` defaults to the count the user asked for; pass the value from
    :func:`fit_parallel_slots` to cost what will actually be served. Kept free
    of any GPU probe so the estimate stays a pure function of the model.

    Includes the DFlash draft weights (plus its small KV cache) when one is
    attached — the draft is co-resident with the target during serving, so its
    footprint has to count toward pin feasibility and the dashboard estimate.
    """
    if slots is None:
        slots = requested_parallel_slots(m)
    return _estimate_with_slots(m, slots)


def fit_parallel_slots(m: Model, notices: list[str] | None = None) -> int:
    """Requested slot count, reduced until the model plausibly fits in VRAM.

    Each slot holds a full context window, so honoring both `ctx` and a high
    slot count can ask for far more VRAM than the card has — llama-server then
    dies on load with ErrorOutOfDeviceMemory and llama-swap restarts it in a
    loop. Between the two knobs, `ctx` is the one the user set deliberately and
    the one we advertise to clients, so concurrency is what gives way.

    Never returns less than 1: if even a single slot doesn't fit we serve it
    anyway (llama-server can still spill layers to CPU) and say so.
    """
    want = requested_parallel_slots(m)
    total = total_vram_gib()
    if total <= 0:
        return want  # No GPU info (CPU box / probe failure) => don't second-guess
    fits = want
    while fits > 1 and _estimate_with_slots(m, fits) > total:
        fits -= 1
    if fits < want and notices is not None:
        notices.append(
            f"{m.name}: {want} parallel slots x {m.ctx:,}-token context needs more "
            f"than the {total:.0f} GiB on this GPU — serving {fits} slot"
            f"{'s' if fits != 1 else ''} so the full context still fits. Lower the "
            f"context to run more slots concurrently."
        )
    return fits


def pinned_vram_estimate(reg: Registry) -> float:
    return sum(
        estimate_model_vram_gib(m, slots=fit_parallel_slots(m))
        for m in reg.models
        if m.pin
    )


@dataclass(frozen=True)
class FitReport:
    """How a model's configuration lands against the GPU it will run on.

    Drives the dashboard's degradation banner: the point is to make a slow
    model *explicable* — you should never have to wonder whether you're waiting
    on CPU offload or on something else.
    """
    requested_slots: int
    served_slots: int
    needed_gib: float          # estimated cost at served_slots
    total_gib: float           # GPU capacity, 0.0 when unknown
    cpu_offload_configured: bool   # user explicitly capped -ngl / offloaded experts

    @property
    def known(self) -> bool:
        """False on a CPU box or a failed probe — say nothing rather than guess."""
        return self.total_gib > 0

    @property
    def slots_reduced(self) -> bool:
        return self.served_slots < self.requested_slots

    @property
    def degraded(self) -> bool:
        """Only conditions we know to be true, never ones we merely estimate.

        Deliberately does NOT include "needed_gib > total_gib": the estimate
        can't model hybrid/recurrent architectures (where most layers keep a
        constant-size state instead of a growing KV cache) and overstates them
        several-fold. Claiming "this is running on CPU" about a model that is
        in fact fully GPU-resident would make the banner noise. Real load
        failures are reported from the server's own stderr instead.
        """
        return self.cpu_offload_configured or (self.known and self.slots_reduced)


def fit_report(m: Model) -> FitReport:
    """Describe how ``m`` will actually be served versus how it was configured."""
    served = fit_parallel_slots(m)
    return FitReport(
        requested_slots=requested_parallel_slots(m),
        served_slots=served,
        needed_gib=_estimate_with_slots(m, served),
        total_gib=total_vram_gib(),
        # -ngl below "all" or MoE experts pinned to CPU are deliberate offload:
        # no estimation involved, the model IS partly on CPU.
        cpu_offload_configured=(0 <= m.gpu_layers < 99) or m.cpu_moe_layers >= 0,
    )


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
    needed = estimate_model_vram_gib(m, slots=fit_parallel_slots(m))
    free = free_vram_gib(gpu_index)
    return (needed <= free, needed, free)
