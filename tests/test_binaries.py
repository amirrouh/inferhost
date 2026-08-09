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


# ---- B4: DFlash min-build version gate ----

def test_parse_build_number_parses_bnnnn_only() -> None:
    from inferhost.core.binaries import _parse_build_number

    assert _parse_build_number("b9831") == 9831
    assert _parse_build_number("  b9840  ") == 9840
    # Non-standard tags return None -> "leave it alone" (no re-download thrash).
    assert _parse_build_number("custom") is None
    assert _parse_build_number("unknown") is None
    assert _parse_build_number("") is None
    assert _parse_build_number("9831") is None  # missing the 'b'
    assert _parse_build_number("b9831-dirty") is None


def _write_marker(bin_dir, repo: str, tag: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / ".llama-server.source").write_text(f"{repo}\n{tag}\n", encoding="utf-8")


def _setup_installed(monkeypatch, tmp_path, tag: str):
    """Simulate a full existing install (server + tts present) with a marker tag."""
    from inferhost.core import binaries, paths

    bin_dir = tmp_path / "bin"
    server = bin_dir / "llama-server"
    tts = bin_dir / "llama-tts"
    bin_dir.mkdir(parents=True, exist_ok=True)
    server.write_bytes(b"x")
    tts.write_bytes(b"x")
    _write_marker(bin_dir, binaries.LLAMACPP_REPO, tag)
    monkeypatch.setattr(paths, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(paths, "llama_server_path", lambda: server)
    monkeypatch.setattr(paths, "llama_tts_path", lambda: tts)
    monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)


def test_needs_refresh_true_for_old_build_below_min(tmp_path, monkeypatch) -> None:
    """An installed build older than the DFlash floor (b9831) forces a refresh."""
    from inferhost.core import binaries

    _setup_installed(monkeypatch, tmp_path, tag="b9500")
    assert binaries.needs_llama_server_refresh() is True


def test_needs_refresh_false_for_build_at_or_above_min(tmp_path, monkeypatch) -> None:
    from inferhost.core import binaries

    _setup_installed(monkeypatch, tmp_path, tag="b9831")
    assert binaries.needs_llama_server_refresh() is False
    _setup_installed(monkeypatch, tmp_path, tag="b9999")
    assert binaries.needs_llama_server_refresh() is False


def test_needs_refresh_false_for_custom_or_unparseable_tag(tmp_path, monkeypatch) -> None:
    """A 'custom'/unparseable marker tag is left alone — we never thrash-download
    over a build we can't reason about."""
    from inferhost.core import binaries

    _setup_installed(monkeypatch, tmp_path, tag="custom")
    assert binaries.needs_llama_server_refresh() is False
    _setup_installed(monkeypatch, tmp_path, tag="unknown")
    assert binaries.needs_llama_server_refresh() is False


def test_needs_refresh_ignores_min_build_for_custom_binary_path(tmp_path, monkeypatch) -> None:
    """INFERHOST_LLAMA_SERVER_PATH short-circuits to False regardless of build —
    we never overwrite the user's own binary."""
    from inferhost import settings as settings_mod
    from inferhost.core import binaries

    _setup_installed(monkeypatch, tmp_path, tag="b9500")
    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", "/opt/custom/llama-server")
    # settings() is cached, and the value normally arrives via inferhost.env
    # rather than the process env — reload so the setting is actually seen.
    settings_mod.reload_settings()
    try:
        assert binaries.needs_llama_server_refresh() is False
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


# ---- custom llama-server path ----

def test_llama_server_path_honours_the_setting(monkeypatch, tmp_path):
    """The custom path must win in paths.llama_server_path() itself, not only
    inside install_llama_server(). Custom-binary mode skips the installer
    entirely, so a path resolved only there would never reach the generated
    llama-swap config and the user's build would sit unused."""
    from inferhost import settings as settings_mod
    from inferhost.core import paths

    custom = tmp_path / "cuda" / "llama-server"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"x")
    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", str(custom))
    settings_mod.reload_settings()
    try:
        assert paths.llama_server_path() == custom
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


def test_llama_server_path_defaults_to_managed_binary(monkeypatch, tmp_path):
    """Unset (the common case) still resolves to inferhost's own bin dir."""
    from inferhost import settings as settings_mod
    from inferhost.core import paths

    monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
    settings_mod.reload_settings()
    monkeypatch.setattr(paths, "bin_dir", lambda: tmp_path / "bin")
    assert paths.llama_server_path() == tmp_path / "bin" / "llama-server"


def test_llama_server_path_expands_user(monkeypatch):
    """~ in the env file must expand — it's a hand-edited file, people type ~."""
    from inferhost import settings as settings_mod
    from inferhost.core import paths

    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", "~/src/llama.cpp/build/bin/llama-server")
    settings_mod.reload_settings()
    try:
        assert "~" not in str(paths.llama_server_path())
        assert str(paths.llama_server_path()).endswith("build/bin/llama-server")
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()
