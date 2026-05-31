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


def test_extract_replaces_existing_binary(tmp_path: Path) -> None:
    """_extract_archive must overwrite an existing same-named binary (the
    unlink-before-write path that makes a running llama-swap replaceable —
    ETXTBSY otherwise). We can't run a real binary in a unit test, but we prove
    the existing file is unlinked and rewritten from the archive."""
    import io
    import zipfile

    from inferhost.core.binaries import _extract_archive, _unlink_before_write

    # Pre-existing "old" llama-swap that an upgrade must replace.
    old = tmp_path / "llama-swap"
    old.write_bytes(b"OLD-VERSION")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("llama-swap", b"NEW-VERSION")
    extracted = _extract_archive(buf.getvalue(), "x.zip", tmp_path,
                                 want_basenames=("llama-swap",))

    assert (tmp_path / "llama-swap").read_bytes() == b"NEW-VERSION"
    assert extracted == [tmp_path / "llama-swap"]

    # Helper is a no-op (no raise) when the target is absent.
    _unlink_before_write(tmp_path / "does-not-exist")


def test_pick_sdcpp_asset_prefers_vulkan_then_cpu(monkeypatch) -> None:
    """On a Vulkan-capable Linux box the picker takes the Linux Vulkan zip; with
    no GPU it falls back to the plain CPU x86_64 zip. (.zip + x86_64 naming.)"""
    import platform as _plat

    from inferhost.core import binaries

    monkeypatch.setattr(_plat, "system", lambda: "Linux")
    monkeypatch.setattr(_plat, "machine", lambda: "x86_64")

    assets = [
        {"name": "sd-master-x-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
         "browser_download_url": "https://e/v", "size": 1},
        {"name": "sd-master-x-bin-Linux-Ubuntu-24.04-x86_64.zip",
         "browser_download_url": "https://e/c", "size": 1},
        {"name": "sd-master-x-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.2.1.zip",
         "browser_download_url": "https://e/r", "size": 1},
        {"name": "sd-master-x-bin-win-cuda12-x64.zip",
         "browser_download_url": "https://e/w", "size": 1},
        {"name": "cudart-sd-bin-win-cu12-x64.zip",
         "browser_download_url": "https://e/cu", "size": 1},
    ]
    gpu = binaries._pick_sdcpp_asset(assets, want_gpu=True, backend="auto")
    assert gpu.name.endswith("x86_64-vulkan.zip")
    cpu = binaries._pick_sdcpp_asset(assets, want_gpu=False, backend="auto")
    assert cpu.name.endswith("x86_64.zip") and "vulkan" not in cpu.name
    rocm = binaries._pick_sdcpp_asset(assets, want_gpu=True, backend="rocm")
    assert "rocm" in rocm.name


def test_latest_release_with_empty_assets_falls_back(monkeypatch) -> None:
    """GitHub publishes the tag before the asset tarballs finish uploading.
    During that gap /releases/latest returns the new tag with assets:[].
    The fallback must walk /releases and return the most recent one with
    assets, instead of failing the install."""
    import httpx as _httpx

    from inferhost.core import binaries

    NEW_EMPTY = {"tag_name": "b9330", "assets": []}
    PREV_WITH_ASSETS = {
        "tag_name": "b9329",
        "assets": [{"name": "llama-b9329-bin-ubuntu-vulkan-x64.tar.gz",
                    "browser_download_url": "https://example/x", "size": 1}],
    }

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self._payload

    def fake_get(url, **_kw):
        if url.endswith("/releases/latest"):
            return _Resp(NEW_EMPTY)
        if url.endswith("/releases"):
            return _Resp([NEW_EMPTY, PREV_WITH_ASSETS])
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(_httpx, "get", fake_get)

    rel = binaries._llamacpp_release_json("latest")
    assert rel["tag_name"] == "b9329"
    assert rel["assets"], "fallback must return a release with assets attached"


def test_latest_release_with_assets_returns_directly(monkeypatch) -> None:
    """Happy path: /releases/latest already has its assets — no fallback needed."""
    import httpx as _httpx

    from inferhost.core import binaries

    OK = {
        "tag_name": "b9330",
        "assets": [{"name": "llama-b9330-bin-ubuntu-vulkan-x64.tar.gz",
                    "browser_download_url": "https://example/x", "size": 1}],
    }

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self._payload

    calls: list[str] = []

    def fake_get(url, **_kw):
        calls.append(url)
        if url.endswith("/releases/latest"):
            return _Resp(OK)
        raise AssertionError(
            f"happy path must not hit /releases — only /releases/latest. got: {url}"
        )

    monkeypatch.setattr(_httpx, "get", fake_get)

    rel = binaries._llamacpp_release_json("latest")
    assert rel["tag_name"] == "b9330"
    assert sum(c.endswith("/releases") for c in calls) == 0
