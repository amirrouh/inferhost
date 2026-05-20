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
    "Q5_0",
    "Q4_K_M",
    "Q4_K_S",
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
)

QUANT_RANK: dict[str, int] = {q: i for i, q in enumerate(QUANT_PRIORITY)}

_QUANT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(F16|BF16|Q8_0|Q6_K_XL|Q6_K|Q5_K_M|Q5_K_S|Q5_0|Q4_K_M|Q4_K_S|Q4_0|"
    r"IQ4_NL|IQ4_XS|Q3_K_L|Q3_K_M|Q3_K_S|IQ3_M|IQ3_XS|IQ3_XXS|IQ2_M|IQ2_XS|IQ2_XXS|Q2_K)"
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
