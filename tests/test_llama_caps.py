"""Tests for llama-server capability probing and KV-quant fallback."""
from __future__ import annotations

from inferhost.core.llama_caps import (
    _ALL_KNOWN,
    parse_supported_cache_types,
    pick_kv_quant,
)

VANILLA_HELP = """
-ctk,  --cache-type-k TYPE              KV cache data type for K
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (default: f16)
-ctv,  --cache-type-v TYPE              KV cache data type for V
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (default: f16)
"""


def test_parse_vanilla_help():
    s = parse_supported_cache_types(VANILLA_HELP)
    assert "q8_0" in s
    assert "f16" in s


def test_parse_empty_help():
    assert parse_supported_cache_types("") == frozenset()


def test_pick_supported_passthrough():
    """When requested value is supported, return it unchanged with no warning."""
    val, warn = pick_kv_quant("q8_0", frozenset({"q8_0", "f16"}))
    assert val == "q8_0"
    assert warn is None


def test_pick_q4_0_falls_back_on_minimal_build():
    """q4_0 -> q4_1 is the first-choice fallback per _FALLBACK_ORDER."""
    minimal = frozenset({"f16", "q8_0", "q4_1"})
    val, warn = pick_kv_quant("q4_0", minimal)
    assert val == "q4_1"
    assert warn is not None
    assert "q4_0" in warn


def test_pick_unknown_value_safe_default():
    """Unknown quant name falls back to q8_0/f16 with a warning."""
    val, warn = pick_kv_quant("made_up_quant", frozenset({"q8_0", "f16"}))
    assert val == "q8_0"
    assert warn is not None


def test_pick_case_insensitive():
    """User might set INFERHOST_KV_QUANT_V=Q8_0 — should still match."""
    val, warn = pick_kv_quant("Q8_0", frozenset({"q8_0", "f16"}))
    assert val == "Q8_0"  # returned as-given when supported
    assert warn is None


def test_all_known_covers_upstream_quants():
    """Sanity: the optimistic default set covers every upstream value."""
    assert "q8_0" in _ALL_KNOWN
    assert "f16" in _ALL_KNOWN
    assert "iq4_nl" in _ALL_KNOWN
