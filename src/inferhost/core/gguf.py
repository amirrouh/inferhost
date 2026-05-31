"""Minimal GGUF header reader — extracts a model's native trained context.

Only the metadata key/value block at the head of the file is parsed (never the
tensor data), so this is cheap even for multi-GiB models. We early-exit the
moment the architecture and its ``<arch>.context_length`` are known, which in
practice happens before the giant tokenizer arrays near the end of the block.

Any malformed / unexpected layout returns ``None`` rather than raising: a
missing native context simply means inferhost trusts the user-configured ``-c``
value instead of clamping to the file's real capability.

GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

_MAGIC = b"GGUF"

# GGUF metadata value-type enum.
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32 = 0, 1, 2, 3, 4, 5
_T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY = 6, 7, 8, 9
_T_UINT64, _T_INT64, _T_FLOAT64 = 10, 11, 12

# Fixed-width scalar types -> struct format. STRING / ARRAY are variable-length
# and handled separately.
_SCALAR_FMT: dict[int, str] = {
    _T_UINT8: "<B", _T_INT8: "<b",
    _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i",
    _T_FLOAT32: "<f", _T_BOOL: "<?",
    _T_UINT64: "<Q", _T_INT64: "<q",
    _T_FLOAT64: "<d",
}
_SCALAR_SIZE: dict[int, int] = {t: struct.calcsize(f) for t, f in _SCALAR_FMT.items()}


class _Reader:
    """Sequential little-endian reader over a seekable binary file."""

    def __init__(self, f) -> None:
        self.f = f

    def _read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected end of GGUF header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self._read(n).decode("utf-8", "replace")

    def scalar(self, t: int):
        return struct.unpack(_SCALAR_FMT[t], self._read(_SCALAR_SIZE[t]))[0]

    def value(self, t: int):
        """Read and return a metadata value of type ``t``."""
        if t == _T_STRING:
            return self.string()
        if t == _T_ARRAY:
            elem_t = self.u32()
            count = self.u64()
            return [self.value(elem_t) for _ in range(count)]
        if t in _SCALAR_FMT:
            return self.scalar(t)
        raise ValueError(f"unknown GGUF value type {t}")

    def skip_value(self, t: int) -> None:
        """Advance past a metadata value without materializing it.

        Uses ``seek`` for fixed-width scalars and scalar arrays so skipping a
        150k-entry tokenizer array costs one seek, not 150k reads.
        """
        if t == _T_STRING:
            self.f.seek(self.u64(), os.SEEK_CUR)
        elif t == _T_ARRAY:
            elem_t = self.u32()
            count = self.u64()
            if elem_t == _T_STRING:
                for _ in range(count):
                    self.f.seek(self.u64(), os.SEEK_CUR)
            elif elem_t == _T_ARRAY:
                for _ in range(count):
                    self.skip_value(_T_ARRAY)
            else:
                self.f.seek(_SCALAR_SIZE[elem_t] * count, os.SEEK_CUR)
        elif t in _SCALAR_SIZE:
            self.f.seek(_SCALAR_SIZE[t], os.SEEK_CUR)
        else:
            raise ValueError(f"unknown GGUF value type {t}")


def native_context(path: str | os.PathLike) -> int | None:
    """Return the model's native trained context (``<arch>.context_length``).

    ``None`` if the file is missing, isn't a GGUF, or the key can't be found.
    """
    try:
        with open(path, "rb") as f:
            r = _Reader(f)
            if r._read(4) != _MAGIC:
                return None
            version = r.u32()
            if version < 2:  # v1 used 32-bit lengths; effectively extinct
                return None
            r.u64()  # tensor_count (unused)
            kv_count = r.u64()

            arch: str | None = None
            ctx_by_key: dict[str, int] = {}
            for _ in range(kv_count):
                key = r.string()
                vtype = r.u32()
                if key == "general.architecture":
                    v = r.value(vtype)
                    arch = v if isinstance(v, str) else arch
                elif key.endswith(".context_length"):
                    v = r.value(vtype)
                    if isinstance(v, (int, float)):
                        ctx_by_key[key] = int(v)
                else:
                    r.skip_value(vtype)
                # Early-exit once the architecture-specific key is in hand — we
                # almost never reach the tokenizer arrays this way.
                if arch and f"{arch}.context_length" in ctx_by_key:
                    return ctx_by_key[f"{arch}.context_length"]

            if arch and f"{arch}.context_length" in ctx_by_key:
                return ctx_by_key[f"{arch}.context_length"]
            # Architecture key absent but exactly one context_length present —
            # trust it (covers odd metadata orderings / nonstandard arch names).
            if len(ctx_by_key) == 1:
                return next(iter(ctx_by_key.values()))
            return None
    except (OSError, EOFError, ValueError, struct.error):
        return None


# (path, size, mtime) -> native context. Keyed on file identity so a swapped or
# re-downloaded GGUF at the same path is automatically re-read — which is the
# exact "disk differs from what's served" drift we want to catch.
_cache: dict[tuple[str, int, int], int | None] = {}


def native_context_cached(path: str | os.PathLike) -> int | None:
    """``native_context`` with an in-process cache keyed on file identity."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(Path(path)), st.st_size, int(st.st_mtime))
    if key not in _cache:
        _cache[key] = native_context(path)
    return _cache[key]
