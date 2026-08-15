"""GGUF quantization parsing and selection by available VRAM/RAM."""
from __future__ import annotations

import re

# Ranked best (highest fidelity) to worst. Lower index = better quality.
QUANT_PRIORITY: tuple[str, ...] = (
    "F16",
    "BF16",
    "Q8_0",
    "Q6_K_XL",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_1",
    "Q5_0",
    # Block-scaled FP4 (ggml types 39/40). Both are ~4 bits per weight, but the
    # scale format decides where they land against the K-quants: NVFP4 pairs a
    # 16-weight block with an FP8 (E4M3) scale, which holds up better than
    # Q4_K_M, while MXFP4's 32-weight block and coarse E8M0 (power-of-two) scale
    # puts it just under. Both need a llama-server new enough to carry the type
    # — see `binaries.binary_supports_ggml_type`.
    "NVFP4",
    "Q4_K_M",
    "MXFP4",
    "Q4_K_S",
    "Q4_1",
    "Q4_0",
    "IQ4_NL",
    "IQ4_XS",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "IQ3_M",
    "IQ3_XS",
    "IQ3_XXS",
    "IQ2_M",
    "IQ2_XS",
    "IQ2_XXS",
    "Q2_K",
    # Sub-2.2-bit ternary / binary formats (BitNet TQ*, prism-ml Bonsai Q2/Q1).
    # Ranked below the conventional quants: they only ever compete inside a
    # dedicated ternary repo, where the mainline-runnable group-64 packing must
    # outrank the fork-only group-128 files (plain Q2_0 / PQ2_0 need the
    # PrismML llama.cpp fork; *_g64 runs on the upstream llama-server we ship).
    "TQ2_0",
    "TQ1_0",
    "Q2_0_G64",
    "Q2_G64",
    "Q2_0",
    "PQ2_0",
    "Q1_0",
)

QUANT_RANK: dict[str, int] = {q: i for i, q in enumerate(QUANT_PRIORITY)}

# Quants carried by a ggml tensor type new enough that an older llama-server
# won't have it compiled in, mapped to that type's `type_name` in ggml.c. Such a
# file loads only on a binary that knows the type; anything older aborts with
# "unknown type N". Everything else in QUANT_PRIORITY has been in ggml for years
# and needs no probe.
RECENT_GGML_TYPES: dict[str, str] = {
    "MXFP4": "mxfp4",
    "NVFP4": "nvfp4",
}

_QUANT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(F16|BF16|Q8_0|Q6_K_XL|Q6_K|Q5_K_M|Q5_K_S|Q5_1|Q5_0|NVFP4|MXFP4|"
    r"Q4_K_M|Q4_K_S|Q4_1|Q4_0|"
    r"IQ4_NL|IQ4_XS|Q3_K_L|Q3_K_M|Q3_K_S|IQ3_M|IQ3_XS|IQ3_XXS|IQ2_M|IQ2_XS|IQ2_XXS|Q2_K|"
    r"TQ2_0|TQ1_0|Q2_0_G64|Q2_G64|PQ2_0|Q2_0|Q1_0)"
    r"(?![A-Za-z0-9])"
)


def extract_quant(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def pick_best_fitting(files, available_gib: float, overhead_gib: float = 1.5):
    """Pick the best-quality GGUF file that fits in available memory.

    `files`: iterable of objects with .size_gib (float) and .quant_rank (int) attrs.
    Returns the best file, or None if nothing fits.
    """
    budget = max(0.0, available_gib - overhead_gib)
    fitting = [f for f in files if f.size_gib <= budget]
    if not fitting:
        return None
    fitting.sort(key=lambda f: (f.quant_rank, -f.size_gib))
    return fitting[0]


def pick_best(files, available_gib: float, overhead_gib: float = 1.5):
    """Like pick_best_fitting but if nothing fits, returns the smallest available file."""
    best = pick_best_fitting(files, available_gib, overhead_gib)
    if best is not None:
        return best
    smallest = sorted(files, key=lambda f: f.size_bytes)
    return smallest[0] if smallest else None
