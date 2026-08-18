"""Tests for the minimal GGUF header reader.

We build tiny valid GGUF headers in-memory (magic + metadata KV block, no
tensors) so the reader can be exercised without shipping a multi-GiB fixture.
"""
from __future__ import annotations

import struct

from inferhost.core import gguf

# GGUF metadata value types we use below.
_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _gstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key: str, value: str) -> bytes:
    return _gstr(key) + struct.pack("<I", _T_STRING) + _gstr(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _gstr(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    arr = struct.pack("<I", _T_STRING) + struct.pack("<Q", len(values))
    arr += b"".join(_gstr(v) for v in values)
    return _gstr(key) + struct.pack("<I", _T_ARRAY) + arr


def _write_gguf(path, kvs: list[bytes]) -> None:
    header = b"GGUF" + struct.pack("<I", 3)  # version 3
    header += struct.pack("<Q", 0)           # tensor_count
    header += struct.pack("<Q", len(kvs))    # metadata_kv_count
    path.write_bytes(header + b"".join(kvs))


def test_reads_arch_context_length(tmp_path):
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "qwen3"),
        _kv_uint32("qwen3.context_length", 262144),
    ])
    assert gguf.native_context(p) == 262144


def test_skips_arrays_before_context_length(tmp_path):
    """A big string array (like the tokenizer) before the key must be skipped,
    not parsed entry-by-entry — and the reader must still find the context."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "llama"),
        _kv_string_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(500)]),
        _kv_uint32("llama.context_length", 131072),
    ])
    assert gguf.native_context(p) == 131072


def test_single_context_key_without_matching_arch(tmp_path):
    """If the architecture key is absent but exactly one *.context_length
    exists, trust it (covers odd metadata orderings / nonstandard arch names)."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_uint32("phi3.context_length", 4096),
    ])
    assert gguf.native_context(p) == 4096


def test_detects_mtp_heads_from_metadata(tmp_path):
    """A model advertising nextn predict layers > 0 is detected as MTP-capable."""
    p = tmp_path / "mtp.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "qwen3moe"),
        _kv_uint32("qwen3moe.nextn_predict_layers", 1),
    ])
    assert gguf.has_mtp_heads(p) is True


def test_no_mtp_heads_when_absent_or_zero(tmp_path):
    """A normal model (no nextn key) — and one with a zero count — are NOT MTP."""
    p1 = tmp_path / "plain.gguf"
    _write_gguf(p1, [
        _kv_string("general.architecture", "qwen3moe"),
        _kv_uint32("qwen3moe.context_length", 262144),
    ])
    assert gguf.has_mtp_heads(p1) is False

    p2 = tmp_path / "zero.gguf"
    _write_gguf(p2, [
        _kv_uint32("qwen3moe.nextn_predict_layers", 0),  # present but 0 = no heads
    ])
    assert gguf.has_mtp_heads(p2) is False


def test_non_gguf_file_returns_none(tmp_path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"this is not a gguf file at all")
    assert gguf.native_context(p) is None


def test_missing_file_returns_none(tmp_path):
    assert gguf.native_context_cached(tmp_path / "nope.gguf") is None
    assert gguf.native_context_cached("") is None


def test_cache_reflects_file_swap(tmp_path):
    """The cache is keyed on (path, size, mtime), so replacing the file with
    different content at the same path must surface the new value."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "qwen3"),
        _kv_uint32("qwen3.context_length", 8192),
    ])
    assert gguf.native_context_cached(p) == 8192
    # Rewrite with a different native context and a different size so the
    # (size, mtime) key changes even on coarse-grained filesystem clocks.
    _write_gguf(p, [
        _kv_string("general.architecture", "qwen3"),
        _kv_uint32("qwen3.context_length", 262144),
        _kv_string("general.name", "padding-to-change-size"),
    ])
    assert gguf.native_context_cached(p) == 262144


# ---- general.architecture ----

def test_reads_architecture(tmp_path):
    """The arch string is what llama.cpp matches its own table against, so it's
    also the token that appears in "unknown model architecture: 'X'"."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.name", "Muse Glimmer 30B"),
        _kv_string("general.architecture", "muse-glimmer"),
        _kv_uint32("muse-glimmer.context_length", 262144),
    ])
    assert gguf.architecture(p) == "muse-glimmer"


def test_architecture_skips_arrays_before_the_key(tmp_path):
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
        _kv_string("general.architecture", "gemma4"),
    ])
    assert gguf.architecture(p) == "gemma4"


def test_architecture_none_for_missing_or_non_gguf(tmp_path):
    assert gguf.architecture(tmp_path / "nope.gguf") is None
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"NOTGGUF" + b"\x00" * 64)
    assert gguf.architecture(junk) is None
    assert gguf.architecture_cached("") is None


def test_kv_geometry_plain_transformer(tmp_path):
    """No hybrid keys: every block holds a KV cache and there is no SSM state."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "qwen3"),
        _kv_uint32("qwen3.block_count", 48),
        _kv_uint32("qwen3.attention.head_count", 40),
        _kv_uint32("qwen3.attention.head_count_kv", 8),
        _kv_uint32("qwen3.attention.key_length", 128),
        _kv_uint32("qwen3.attention.value_length", 128),
    ])
    geom = gguf.kv_geometry(p)
    assert geom.cache_layers == 48
    assert geom.kv_elems == 8 * (128 + 128)
    assert geom.state_elems == 0


def test_kv_geometry_hybrid_counts_only_attention_layers(tmp_path):
    """Qwen3.8-27B's real shape: 65 blocks, but only every 4th layer of the
    64-layer body keeps a KV cache, plus the trailing MTP block — 17, not 65.
    Counting all 65 overstates the cache 4x and costs the user parallel slots."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "qwen35"),
        _kv_uint32("qwen35.block_count", 65),
        _kv_uint32("qwen35.attention.head_count", 24),
        _kv_uint32("qwen35.attention.head_count_kv", 4),
        _kv_uint32("qwen35.attention.key_length", 256),
        _kv_uint32("qwen35.attention.value_length", 256),
        _kv_uint32("qwen35.nextn_predict_layers", 1),
        _kv_uint32("qwen35.full_attention_interval", 4),
        _kv_uint32("qwen35.ssm.conv_kernel", 4),
        _kv_uint32("qwen35.ssm.state_size", 128),
        _kv_uint32("qwen35.ssm.group_count", 16),
        _kv_uint32("qwen35.ssm.inner_size", 6144),
    ])
    geom = gguf.kv_geometry(p)
    assert geom.cache_layers == 17
    assert geom.kv_elems == 4 * (256 + 256)
    # 48 recurrent layers x (inner*state + (conv_kernel-1)*conv_dim)
    conv_dim = 6144 + 2 * 16 * 128
    assert geom.state_elems == 48 * (6144 * 128 + 3 * conv_dim)
    # 36,992 bytes/token at q8_0 — matches what llama-server allocates.
    assert geom.cache_layers * geom.kv_elems * 1.0625 == 36992


def test_kv_geometry_hybrid_without_ssm_keys(tmp_path):
    """A hybrid that declares the interval but no SSM metadata still gets the
    reduced layer count; the state term degrades to zero rather than raising."""
    p = tmp_path / "model.gguf"
    _write_gguf(p, [
        _kv_string("general.architecture", "hyb"),
        _kv_uint32("hyb.block_count", 32),
        _kv_uint32("hyb.attention.head_count", 32),
        _kv_uint32("hyb.attention.head_count_kv", 8),
        _kv_uint32("hyb.attention.key_length", 128),
        _kv_uint32("hyb.attention.value_length", 128),
        _kv_uint32("hyb.full_attention_interval", 4),
    ])
    geom = gguf.kv_geometry(p)
    assert geom.cache_layers == 8
    assert geom.state_elems == 0
