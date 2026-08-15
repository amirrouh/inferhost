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


# ---- installed build tag (what `update` reports as the "from" version) ----

def test_installed_tag_reads_the_marker(tmp_path, monkeypatch) -> None:
    from inferhost.core import binaries

    _setup_installed(monkeypatch, tmp_path, tag="b10068")
    assert binaries.installed_llama_server_tag() == "b10068"


def test_installed_tag_is_none_without_a_marker(tmp_path, monkeypatch) -> None:
    """A fresh box (or a marker we can't read) must report unknown, not crash —
    `update` prints "unknown -> bNNNN" rather than failing before it downloads."""
    from inferhost.core import binaries, paths

    monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setattr(paths, "bin_dir", lambda: tmp_path / "empty")
    assert binaries.installed_llama_server_tag() is None


def test_installed_tag_reports_custom_binary_mode(tmp_path, monkeypatch) -> None:
    from inferhost import settings as settings_mod
    from inferhost.core import binaries

    _setup_installed(monkeypatch, tmp_path, tag="b10068")
    monkeypatch.setenv("INFERHOST_LLAMA_SERVER_PATH", "/opt/custom/llama-server")
    settings_mod.reload_settings()
    try:
        assert binaries.installed_llama_server_tag() == "custom"
    finally:
        monkeypatch.delenv("INFERHOST_LLAMA_SERVER_PATH", raising=False)
        settings_mod.reload_settings()


# ---- architecture capability probe ----
#
# llama.cpp keeps every architecture it can build in one table of literal names
# and matches general.architecture against it, so the names are embedded in the
# binary. Searching for the token is how inferhost decides whether the
# llama-server on disk predates a model.

def _fake_binary(path, names: list[bytes], filler: bytes = b"\x00" * 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(filler + b"\x00".join(names) + b"\x00" + filler)


def test_probe_finds_a_supported_architecture(tmp_path) -> None:
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"llama", b"qwen3moe", b"muse-glimmer"])
    assert binaries.binary_supports_arch(exe, "muse-glimmer") is True
    assert binaries.binary_supports_arch(exe, "llama") is True


def test_probe_rejects_an_architecture_the_binary_predates(tmp_path) -> None:
    """The case that matters: an older build simply has no such entry."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"llama", b"qwen3moe"])
    assert binaries.binary_supports_arch(exe, "muse-glimmer") is False


def test_probe_does_not_match_a_longer_architecture_name(tmp_path) -> None:
    """"qwen3" must not be satisfied by "qwen3moe" alone — they're different
    architectures, and a prefix match would call a stale binary current."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"qwen3moe", b"gemma4n"])
    assert binaries.binary_supports_arch(exe, "qwen3") is False
    assert binaries.binary_supports_arch(exe, "gemma4") is False


def test_probe_accepts_a_tail_merged_name(tmp_path) -> None:
    """Linkers store a short constant as the tail of a longer one — "bert"
    inside "modern_bert\\0". Requiring a delimiter in FRONT of the token
    reports these perfectly supported architectures as missing."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"modern_bert", b"rwkv6qwen2", b"deepseek-r1-qwen"])
    assert binaries.binary_supports_arch(exe, "bert") is True
    assert binaries.binary_supports_arch(exe, "qwen2") is True
    assert binaries.binary_supports_arch(exe, "qwen") is True


def test_probe_spans_read_chunk_boundaries(tmp_path) -> None:
    """The scan reads in 8 MiB chunks; a name straddling the seam must still
    be found, or support would depend on where in the file it happens to sit."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    exe.parent.mkdir(parents=True, exist_ok=True)
    chunk = 8 * 1024 * 1024
    name = b"muse-glimmer"
    # Straddle the seam: half the token before it, half after.
    lead = b"\x00" * (chunk - len(name) // 2)
    exe.write_bytes(lead + name + b"\x00" + b"\x00" * 1024)
    assert binaries.binary_supports_arch(exe, "muse-glimmer") is True


def test_probe_checks_the_shared_library_too(tmp_path) -> None:
    """Official tarballs are dynamically linked: the arch table lives in
    libllama.so, and the launcher next to it is a ~17 KiB stub."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"stub"])
    _fake_binary(tmp_path / "libllama.so.0", [b"muse-glimmer"])
    assert binaries.binary_supports_arch(exe, "muse-glimmer") is True


def test_probe_errs_toward_supported_when_it_cannot_tell(tmp_path) -> None:
    """A missing binary or unknown arch must not strip a model of its server —
    let the binary raise its own clear error instead."""
    from inferhost.core import binaries

    assert binaries.binary_supports_arch(tmp_path / "absent", "muse-glimmer") is True
    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"llama"])
    assert binaries.binary_supports_arch(exe, "") is True


def test_probe_finds_a_ggml_tensor_type(tmp_path) -> None:
    """NVFP4 weights need a build carrying that ggml type; ggml stores every
    type it can read as a literal `.type_name` in type_traits[]."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"q4_K", b"mxfp4", b"nvfp4"])
    assert binaries.binary_supports_ggml_type(exe, "nvfp4") is True
    assert binaries.binary_supports_ggml_type(exe, "mxfp4") is True


def test_probe_rejects_a_ggml_type_the_binary_predates(tmp_path) -> None:
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    _fake_binary(exe, [b"q4_K", b"mxfp4"])
    assert binaries.binary_supports_ggml_type(exe, "nvfp4") is False


def test_ggml_type_probe_reads_libggml_base(tmp_path) -> None:
    """The official tarballs are dynamically linked and ggml's type_traits[]
    lives in libggml-base.so — one layer below the libllama that holds the
    architecture table. Probing only libllama reports every build as too old."""
    from inferhost.core import binaries

    exe = tmp_path / "llama-server"
    exe.write_bytes(b"\x00" * 64)  # stub launcher, no tables of its own
    _fake_binary(tmp_path / "libllama.so.0.1.0", [b"llama", b"qwen3"])
    _fake_binary(tmp_path / "libggml-base.so.0.19.0", [b"q4_K", b"nvfp4"])
    assert binaries.binary_supports_ggml_type(exe, "nvfp4") is True
    assert binaries.binary_supports_ggml_type(exe, "mxfp4") is False
    assert binaries.binary_supports_arch(exe, "qwen3") is True


def _fake_checkout(tmp_path):
    """A minimal llama.cpp checkout: worktree markers + a CMake build dir."""
    src = tmp_path / "src" / "llama.cpp"
    (src / "build" / "bin").mkdir(parents=True)
    (src / ".git").mkdir()
    (src / "CMakeLists.txt").write_text("project(llama.cpp)\n")
    (src / "build" / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=ON\n")
    exe = src / "build" / "bin" / "llama-server"
    exe.write_bytes(b"\x00" * 32)
    return src, exe


def test_find_custom_build_tree_locates_the_checkout(tmp_path) -> None:
    """`update --rebuild` has to find the tree from the binary alone — the user
    only ever told us INFERHOST_LLAMA_SERVER_PATH."""
    from inferhost.core import binaries

    src, exe = _fake_checkout(tmp_path)
    tree = binaries.find_custom_build_tree(exe)
    assert tree is not None
    assert tree.source == src.resolve()
    assert tree.build == (src / "build").resolve()


def test_find_custom_build_tree_rejects_a_loose_binary(tmp_path) -> None:
    """A binary copied to /usr/local/bin has no tree behind it. Returning a
    bogus one would point cmake at an unrelated directory."""
    from inferhost.core import binaries

    exe = tmp_path / "bin" / "llama-server"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"\x00" * 32)
    assert binaries.find_custom_build_tree(exe) is None


def test_find_custom_build_tree_needs_a_git_worktree(tmp_path) -> None:
    """A build dir inside an unpacked tarball can't be checked out to a tag."""
    from inferhost.core import binaries

    src, exe = _fake_checkout(tmp_path)
    (src / ".git").rmdir()
    assert binaries.find_custom_build_tree(exe) is None


def test_rebuild_reuses_the_cmake_cache_not_our_own_flags(tmp_path, monkeypatch) -> None:
    """The whole point: `cmake --build` inherits GGML_CUDA / CUDA arch / nvcc
    path from CMakeCache.txt. Passing configure flags here would be how a
    rebuild silently downgrades a CUDA build to CPU."""
    from inferhost.core import binaries

    src, exe = _fake_checkout(tmp_path)
    tree = binaries.find_custom_build_tree(exe)
    calls: list[list[str]] = []

    class _Proc:
        stdout = iter(())
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(binaries.subprocess, "Popen",
                        lambda cmd, **kw: calls.append(cmd) or _Proc())
    tag = binaries.rebuild_custom_llama_server(tree, version="b10448")

    assert tag == "b10448"
    assert calls[0][:2] == ["git", "fetch"] and "b10448" in calls[0]
    assert calls[1][:2] == ["git", "checkout"] and "b10448" in calls[1]
    build = calls[2]
    assert build[:2] == ["cmake", "--build"]
    assert "--target" in build and "llama-server" in build
    # No configure-time flags: nothing that would re-derive the build type.
    assert not any(a.startswith("-DGGML") or a.startswith("-DCMAKE") for a in build)


def test_rebuild_surfaces_a_failed_command(tmp_path, monkeypatch) -> None:
    """A silent failure would leave the user believing they upgraded."""
    import pytest

    from inferhost.core import binaries

    src, exe = _fake_checkout(tmp_path)
    tree = binaries.find_custom_build_tree(exe)

    class _Proc:
        stdout = iter(("nvcc fatal : Unsupported gpu architecture\n",))
        returncode = 1

        def wait(self):
            return 1

    monkeypatch.setattr(binaries.subprocess, "Popen", lambda cmd, **kw: _Proc())
    with pytest.raises(RuntimeError, match="failed"):
        binaries.rebuild_custom_llama_server(tree, version="b10448")
