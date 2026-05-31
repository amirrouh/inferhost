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
