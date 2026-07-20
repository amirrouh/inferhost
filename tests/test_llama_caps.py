"""Tests for llama-server capability probing and KV-quant fallback."""
from __future__ import annotations

from inferhost.core import llama_caps
from inferhost.core.llama_caps import (
    _ALL_KNOWN,
    parse_supported_cache_types,
    pick_kv_quant,
    supports_spec_type,
)

DFLASH_HELP = """
--spec-type TYPE                       speculative decoding draft type
                                        allowed: draft-mtp, draft-dflash, ngram-mod
"""

OLD_HELP = """
--spec-type TYPE                       speculative decoding draft type
                                        allowed: draft-mtp, ngram-mod
"""

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


def test_supports_spec_type_true_when_advertised(monkeypatch):
    """A modern binary lists draft-dflash in --help -> supported."""
    llama_caps._help_text.cache_clear()
    monkeypatch.setattr(llama_caps, "_help_text", lambda: DFLASH_HELP)
    assert supports_spec_type("draft-dflash") is True
    assert supports_spec_type("draft-mtp") is True


def test_supports_spec_type_false_when_absent(monkeypatch):
    """An older binary that lacks draft-dflash -> NOT supported (so configs.py
    won't emit a flag that would abort the model)."""
    llama_caps._help_text.cache_clear()
    monkeypatch.setattr(llama_caps, "_help_text", lambda: OLD_HELP)
    assert supports_spec_type("draft-dflash") is False
    # But it still advertises the older spec types.
    assert supports_spec_type("draft-mtp") is True


def test_supports_spec_type_fail_open_on_empty_help(monkeypatch):
    """Empty help (binary missing / probe failed) -> fail open (True), same
    optimistic philosophy as supported_cache_types."""
    llama_caps._help_text.cache_clear()
    monkeypatch.setattr(llama_caps, "_help_text", lambda: "")
    assert supports_spec_type("draft-dflash") is True


def test_supports_spec_type_case_insensitive(monkeypatch):
    llama_caps._help_text.cache_clear()
    monkeypatch.setattr(llama_caps, "_help_text", lambda: DFLASH_HELP.upper())
    assert supports_spec_type("draft-dflash") is True


def test_all_known_covers_upstream_quants():
    """Sanity: the optimistic default set covers every upstream value."""
    assert "q8_0" in _ALL_KNOWN
    assert "f16" in _ALL_KNOWN
    assert "iq4_nl" in _ALL_KNOWN
