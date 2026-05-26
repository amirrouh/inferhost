"""Tests for llama.cpp binary install/purge logic.

The bug this guards against: re-installing llama.cpp on top of an existing
install used to leave the old versioned .so files in place. _link_so_versions
then symlinked libfoo.so to whichever version iterdir() hit last, which on
some hosts was the OLD one — producing an ABI mismatch between
libllama.so.0.0.<NEW> and libggml-base.so.<OLD> that crashed llama-server
with SIGSEGV on the next load.

The fix is _purge_llamacpp_files, which runs before each extract and wipes
every llama.cpp-shipped file (libs + llama-* launchers) while preserving
the unrelated llama-swap binary and the source marker.
"""
from __future__ import annotations

from pathlib import Path

from inferhost.core.binaries import _purge_llamacpp_files


def _touch(p: Path, content: bytes = b"") -> None:
    p.write_bytes(content)


def test_purge_removes_old_versioned_libs_and_symlinks(tmp_path: Path) -> None:
    # Simulate a dir left over from a previous install: a mix of old and new
    # versioned libs plus the symlinks _link_so_versions would have produced.
    _touch(tmp_path / "libggml-base.so.0.12.0", b"OLD")
    _touch(tmp_path / "libggml-base.so.0.13.0", b"NEW")
    (tmp_path / "libggml-base.so.0").symlink_to("libggml-base.so.0.12.0")
    (tmp_path / "libggml-base.so").symlink_to("libggml-base.so.0.12.0")
    _touch(tmp_path / "libggml-cuda.so.0.12.0", b"OLD")
    (tmp_path / "libggml-cuda.so").symlink_to("libggml-cuda.so.0.12.0")
    _touch(tmp_path / "libggml-vulkan.so", b"NEW")
    _touch(tmp_path / "llama-server", b"NEW")
    # Files we must NOT touch: llama-swap (different repo) + source marker.
    _touch(tmp_path / "llama-swap", b"keep-me")
    _touch(tmp_path / ".llama-server.source", b"ggml-org/llama.cpp\nb9329\n")

    _purge_llamacpp_files(tmp_path)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [".llama-server.source", "llama-swap"], remaining
    # Confirm the preserved files weren't disturbed.
    assert (tmp_path / "llama-swap").read_bytes() == b"keep-me"
    assert (tmp_path / ".llama-server.source").read_bytes().startswith(b"ggml-org")


def test_purge_is_idempotent_on_empty_dir(tmp_path: Path) -> None:
    # First-time install: nothing to purge, must not raise.
    _purge_llamacpp_files(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_purge_handles_missing_dir(tmp_path: Path) -> None:
    # Path that doesn't exist yet — must be a silent no-op, not a crash.
    missing = tmp_path / "does-not-exist"
    _purge_llamacpp_files(missing)
    assert not missing.exists()


def test_purge_leaves_unrelated_files_alone(tmp_path: Path) -> None:
    # A user might have dropped a custom file in the bin dir (e.g. a notes
    # file). Purge should only touch things it recognizes as llama.cpp output.
    _touch(tmp_path / "notes.md", b"my notes")
    _touch(tmp_path / "libllama.so.0.0.9329", b"")
    _purge_llamacpp_files(tmp_path)
    assert (tmp_path / "notes.md").exists()
    assert not (tmp_path / "libllama.so.0.0.9329").exists()
