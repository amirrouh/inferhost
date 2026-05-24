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

TURBO_HELP = """
-ctk,  --cache-type-k TYPE              KV cache data type for K
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo2, turbo3, turbo4
-ctv,  --cache-type-v TYPE              KV cache data type for V
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo2, turbo3, turbo4
"""


def test_parse_vanilla_help():
    s = parse_supported_cache_types(VANILLA_HELP)
    assert "q8_0" in s
    assert "f16" in s
    assert "turbo3" not in s  # vanilla doesn't have turbo


def test_parse_turbo_help():
    s = parse_supported_cache_types(TURBO_HELP)
    assert "turbo3" in s
    assert "turbo2" in s
    assert "turbo4" in s
    assert "q8_0" in s


def test_parse_empty_help():
    assert parse_supported_cache_types("") == frozenset()


def test_pick_supported_passthrough():
    """When requested value is supported, return it unchanged with no warning."""
    val, warn = pick_kv_quant("turbo3", frozenset({"turbo3", "q8_0", "f16"}))
    assert val == "turbo3"
    assert warn is None


def test_pick_turbo3_falls_back_to_q5_0_on_vanilla():
    """turbo3 -> q5_0 is the first-choice fallback per _FALLBACK_ORDER."""
    vanilla = parse_supported_cache_types(VANILLA_HELP)
    val, warn = pick_kv_quant("turbo3", vanilla)
    assert val == "q5_0"
    assert warn is not None
    assert "turbo3" in warn
    assert "q5_0" in warn


def test_pick_turbo2_falls_back_to_q4_0_on_vanilla():
    vanilla = parse_supported_cache_types(VANILLA_HELP)
    val, warn = pick_kv_quant("turbo2", vanilla)
    assert val == "q4_0"
    assert warn is not None


def test_pick_turbo4_falls_back_to_q5_1_on_vanilla():
    vanilla = parse_supported_cache_types(VANILLA_HELP)
    val, warn = pick_kv_quant("turbo4", vanilla)
    assert val == "q5_1"
    assert warn is not None


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


def test_all_known_includes_turbo():
    """Sanity: the optimistic default set knows about turbo* values."""
    assert "turbo3" in _ALL_KNOWN
    assert "q8_0" in _ALL_KNOWN
