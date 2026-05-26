"""Probe the installed llama-server for supported features.

Different llama-server builds expose different `-ctk` / `-ctv` cache-type
values (e.g. vendor forks may add custom codecs). This module probes
`llama-server --help` once per process to discover the allowed values, then
`pick_kv_quant` returns a safe substitute when the requested value isn't
supported. Defaults to assuming everything works if the binary is missing
or the probe fails, so we never downgrade a config that would have worked.
"""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache

from inferhost.core import paths

_FALLBACK_ORDER: dict[str, tuple[str, ...]] = {
    "q4_0": ("q4_1", "iq4_nl", "q5_0", "q8_0", "f16"),
    "q4_1": ("q4_0", "iq4_nl", "q5_0", "q8_0", "f16"),
    "iq4_nl": ("q4_0", "q4_1", "q5_0", "q8_0", "f16"),
    "q5_0": ("q5_1", "q4_0", "q8_0", "f16"),
    "q5_1": ("q5_0", "q4_0", "q8_0", "f16"),
    "q8_0": ("f16", "bf16"),
    "f16": ("bf16", "f32"),
    "bf16": ("f16", "f32"),
    "f32": ("f16", "bf16"),
}

_ALL_KNOWN: frozenset[str] = frozenset({
    "f32", "f16", "bf16",
    "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl",
})

_ALLOWED_VALUES_RE = re.compile(r"allowed values:\s*([a-zA-Z0-9_,\s]+)")


def parse_supported_cache_types(help_text: str) -> frozenset[str]:
    """Extract the union of allowed -ctk/-ctv values from `llama-server --help`.

    Public so tests can exercise the parser without a real binary.
    """
    found: set[str] = set()
    for line in help_text.splitlines():
        m = _ALLOWED_VALUES_RE.search(line)
        if not m:
            continue
        for token in m.group(1).split(","):
            token = token.strip().lower()
            if token:
                found.add(token)
    return frozenset(found)


@lru_cache(maxsize=1)
def supported_cache_types() -> frozenset[str]:
    """Return the set of -ctk/-ctv values the installed llama-server accepts.

    Cached per process. Returns `_ALL_KNOWN` if the binary doesn't exist or
    the probe fails — better to emit the user's chosen value and let
    llama-server reject it with its own message than to silently rewrite
    configs based on a failed probe.
    """
    bin_path = paths.llama_server_path()
    if not os.path.isfile(bin_path):
        return _ALL_KNOWN
    try:
        out = subprocess.run(
            [str(bin_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _ALL_KNOWN
    parsed = parse_supported_cache_types(out.stdout + out.stderr)
    return parsed if parsed else _ALL_KNOWN


def pick_kv_quant(
    requested: str, supported: frozenset[str] | None = None
) -> tuple[str, str | None]:
    """Resolve a requested KV quant against what the binary actually supports.

    Returns `(value_to_use, warning_or_None)`. If `requested` is already
    supported, returns it with no warning. Otherwise walks the fallback
    ladder and returns the first supported alternative plus a one-line
    warning explaining the substitution.
    """
    if supported is None:
        supported = supported_cache_types()
    req_lower = requested.lower()
    if req_lower in supported:
        return requested, None
    for alt in _FALLBACK_ORDER.get(req_lower, ()):
        if alt in supported:
            return alt, (
                f"llama-server build does not support '{requested}' — "
                f"using '{alt}' instead. Set INFERHOST_KV_QUANT_K/V to a "
                f"supported value to silence this notice."
            )
    for safe in ("q8_0", "f16"):
        if safe in supported:
            return safe, (
                f"llama-server build does not support '{requested}' — "
                f"using '{safe}' as a safe default."
            )
    return requested, None
