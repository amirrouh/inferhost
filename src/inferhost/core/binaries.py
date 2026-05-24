"""Download and manage prebuilt binaries: llama.cpp (llama-server) and llama-swap."""
from __future__ import annotations

import contextlib
import io
import os
import platform
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from inferhost.core import paths
from inferhost.core.probe import probe
from inferhost.settings import settings

ProgressCallback = Callable[[int, int], None]

LLAMACPP_REPO = "amirrouh/inferhost"
LLAMASWAP_REPO = "mostlygeek/llama-swap"

GH_API = "https://api.github.com"


@dataclass
class ReleaseAsset:
    name: str
    download_url: str
    size: int


@dataclass
class InstalledBinary:
    path: Path
    version: str


def _release_json(repo: str, version: str) -> dict:
    url = (
        f"{GH_API}/repos/{repo}/releases/latest"
        if version == "latest"
        else f"{GH_API}/repos/{repo}/releases/tags/{version}"
    )
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _llamacpp_release_json(version: str) -> dict:
    """Fetch a llama-server release from amirrouh/inferhost.

    Prebuilt llama-server assets live under tags that start with ``llama-v``
    (e.g. ``llama-v1.0.0``). The repo also has ``v*`` PyPI-release tags which
    must be skipped. When version is "latest" we list all releases and pick the
    most-recent one whose tag starts with ``llama-v``.
    """
    repo = LLAMACPP_REPO
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if version != "latest":
        # Caller passed a specific tag; just fetch it directly.
        tag = version if version.startswith("llama-v") else f"llama-v{version}"
        url = f"{GH_API}/repos/{repo}/releases/tags/{tag}"
        r = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        r.raise_for_status()
        return r.json()

    # "latest" — walk paginated release list to find the newest llama-v* tag.
    url = f"{GH_API}/repos/{repo}/releases?per_page=30&page=1"
    r = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
    r.raise_for_status()
    releases = r.json()
    if not isinstance(releases, list):
        raise RuntimeError(
            f"Unexpected response from GitHub releases API for {repo}: {releases!r}"
        )
    for rel in releases:
        tag = rel.get("tag_name", "")
        if tag.startswith("llama-v"):
            return rel
    raise RuntimeError(
        "No llama-v* release found in amirrouh/inferhost. "
        "The CI workflow that builds llama-server prebuilt assets has not run yet. "
        "Please wait for the CI to publish a release under a 'llama-v*' tag and retry."
    )


def _platform_keys() -> tuple[str, str, str]:
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    if sysname == "darwin":
        os_key = "macos"
    elif sysname == "linux":
        os_key = "linux"
    else:
        raise RuntimeError(f"Unsupported OS: {sysname}")
    if machine in ("x86_64", "amd64"):
        cpp_arch = "x64"
        swap_arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        cpp_arch = "arm64"
        swap_arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported arch: {machine}")
    return os_key, cpp_arch, swap_arch  # plus swap_os derivable from sysname


def _pick_llamacpp_asset(assets: list[dict], want_gpu: bool) -> ReleaseAsset:
    """Pick the correct llama-server asset from amirrouh/inferhost prebuilt releases.

    The CI publishes exactly three asset filenames:
      - llama-server-linux-x86_64-cuda12.tar.gz   (Linux + CUDA 12)
      - llama-server-linux-x86_64-cpu.tar.gz      (Linux, CPU-only)
      - llama-server-macos-arm64-metal.tar.gz     (macOS arm64, Metal)

    Selection logic:
      - macOS arm64  -> always metal
      - Linux + GPU  -> cuda12, fall back to cpu
      - Linux + CPU  -> cpu
    """
    sysname = platform.system().lower()
    machine = platform.machine().lower()

    if sysname == "darwin" and machine in ("arm64", "aarch64"):
        preferred = "llama-server-macos-arm64-metal.tar.gz"
        fallbacks: tuple[str, ...] = ()
    elif sysname == "linux" and want_gpu:
        preferred = "llama-server-linux-x86_64-cuda12.tar.gz"
        fallbacks = ("llama-server-linux-x86_64-cpu.tar.gz",)
    elif sysname == "linux":
        preferred = "llama-server-linux-x86_64-cpu.tar.gz"
        fallbacks = ()
    else:
        raise RuntimeError(
            f"No prebuilt llama-server asset for platform {sysname}/{machine}. "
            "Only Linux x86_64 and macOS arm64 are supported."
        )

    asset_map = {a["name"]: a for a in assets if "name" in a and "browser_download_url" in a}
    for candidate in (preferred,) + fallbacks:
        if candidate in asset_map:
            a = asset_map[candidate]
            return ReleaseAsset(name=a["name"], download_url=a["browser_download_url"], size=a.get("size", 0))

    available = list(asset_map.keys())
    raise RuntimeError(
        f"Expected asset '{preferred}' not found in release. "
        f"Available assets: {available}. "
        "The CI may not have finished publishing all platform builds yet."
    )


def _pick_llamaswap_asset(assets: list[dict], swap_os: str, swap_arch: str) -> ReleaseAsset:
    for a in assets:
        name = a.get("name", "").lower()
        if not name.endswith(".tar.gz"):
            continue
        if swap_os in name and swap_arch in name:
            return ReleaseAsset(name=a["name"], download_url=a["browser_download_url"], size=a.get("size", 0))
    raise RuntimeError(
        f"No llama-swap asset found for os={swap_os} arch={swap_arch}. "
        f"Available: {[a.get('name') for a in assets][:10]}"
    )


def _download(url: str, progress_cb: ProgressCallback | None = None) -> bytes:
    chunks: list[bytes] = []
    downloaded = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        if progress_cb:
            progress_cb(0, total)
        for chunk in r.iter_bytes(chunk_size=64 * 1024):
            if not chunk:
                continue
            chunks.append(chunk)
            downloaded += len(chunk)
            if progress_cb:
                progress_cb(downloaded, total)
    return b"".join(chunks)


def _is_lib_or_binary(name: str) -> bool:
    base = Path(name).name.lower()
    return base.startswith("lib") and (".so" in base or ".dylib" in base)


def _extract_archive(
    blob: bytes,
    name: str,
    dest_dir: Path,
    want_basenames: tuple[str, ...],
    take_libs: bool = False,
) -> list[Path]:
    """Extract archive. Pulls out files matching want_basenames; if take_libs, also pulls .so/.dylib."""
    extracted: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    def wants(member_name: str, is_file: bool) -> bool:
        if not is_file:
            return False
        base = Path(member_name).name
        stem = base.split(".")[0]
        if base in want_basenames or stem in want_basenames:
            return True
        return bool(take_libs and _is_lib_or_binary(member_name))

    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for info in z.infolist():
                if not wants(info.filename, not info.is_dir()):
                    continue
                target = dest_dir / Path(info.filename).name
                with z.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                extracted.append(target)
    elif name.lower().endswith(".tar.gz"):
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
            for member in t.getmembers():
                if not wants(member.name, member.isfile()):
                    continue
                f = t.extractfile(member)
                if f is None:
                    continue
                target = dest_dir / Path(member.name).name
                with target.open("wb") as dst:
                    shutil.copyfileobj(f, dst)
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                extracted.append(target)
    else:
        raise RuntimeError(f"Unsupported archive type: {name}")
    return extracted


def _link_so_versions(directory: Path) -> None:
    """For each libfoo.so.MAJOR.MINOR[.PATCH], create symlinks libfoo.so.MAJOR and libfoo.so."""
    import re

    so_pattern = re.compile(r"^(lib[\w\-]+\.so)\.([\d.]+)$")
    dylib_pattern = re.compile(r"^(lib[\w\-]+)\.([\d.]+)\.dylib$")

    for f in directory.iterdir():
        # Skip symlinks: only drive link creation from the real .so.MAJOR.MINOR[.PATCH]
        # file. Otherwise a re-install can iterate "lib.so.0" (a symlink) and create a
        # self-loop, then process the real file too late to recover.
        if not f.is_file() or f.is_symlink():
            continue
        m = so_pattern.match(f.name)
        if m:
            base = m.group(1)  # "libfoo.so"
            version = m.group(2)  # e.g. "0.0.9244"
            major = version.split(".")[0]
            for link in (f"{base}.{major}", base):
                link_path = directory / link
                if link_path.exists() or link_path.is_symlink():
                    try:
                        link_path.unlink()
                    except OSError:
                        continue
                with contextlib.suppress(OSError):
                    link_path.symlink_to(f.name)
            continue
        m = dylib_pattern.match(f.name)
        if m:
            base = m.group(1)  # "libfoo"
            link_path = directory / f"{base}.dylib"
            if link_path.exists() or link_path.is_symlink():
                try:
                    link_path.unlink()
                except OSError:
                    continue
            with contextlib.suppress(OSError):
                link_path.symlink_to(f.name)


def install_llama_server(
    version: str | None = None, progress_cb: ProgressCallback | None = None
) -> InstalledBinary:
    # Escape hatch: if the user has pre-built or pre-installed llama-server,
    # they can point INFERHOST_LLAMA_SERVER_PATH at it to skip all download/extract.
    custom_path = os.environ.get("INFERHOST_LLAMA_SERVER_PATH", "")
    if custom_path:
        exe = Path(custom_path)
        if not exe.exists():
            raise RuntimeError(
                f"INFERHOST_LLAMA_SERVER_PATH is set to '{custom_path}' "
                "but that file does not exist."
            )
        target = paths.llama_server_path()
        paths.ensure_dirs()
        if target.resolve() != exe.resolve():
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(exe.resolve())
        return InstalledBinary(path=target, version="custom")

    paths.ensure_dirs()
    version = version or settings().llamacpp_version
    rel = _llamacpp_release_json(version)
    asset = _pick_llamacpp_asset(rel["assets"], want_gpu=probe().has_gpu)
    blob = _download(asset.download_url, progress_cb=progress_cb)
    _extract_archive(
        blob,
        asset.name,
        paths.bin_dir(),
        want_basenames=("llama-server",),
        take_libs=True,
    )
    target = paths.llama_server_path()
    if not target.exists():
        raise RuntimeError(f"llama-server not found inside {asset.name}")
    _link_so_versions(paths.bin_dir())
    return InstalledBinary(path=target, version=rel.get("tag_name", "unknown"))


def install_llama_swap(
    version: str | None = None, progress_cb: ProgressCallback | None = None
) -> InstalledBinary:
    paths.ensure_dirs()
    version = version or settings().llamaswap_version
    rel = _release_json(LLAMASWAP_REPO, version)
    sysname = platform.system().lower()
    swap_os = "darwin" if sysname == "darwin" else "linux"
    _, _, swap_arch = _platform_keys()
    asset = _pick_llamaswap_asset(rel["assets"], swap_os, swap_arch)
    blob = _download(asset.download_url, progress_cb=progress_cb)
    extracted = _extract_archive(
        blob,
        asset.name,
        paths.bin_dir(),
        want_basenames=("llama-swap",),
    )
    if not extracted:
        raise RuntimeError(f"llama-swap binary not found inside {asset.name}")
    target = paths.llama_swap_path()
    if extracted[0] != target:
        shutil.move(str(extracted[0]), str(target))
    return InstalledBinary(path=target, version=rel.get("tag_name", "unknown"))


def installed_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"llama-server": None, "llama-swap": None}
    for label, p in (("llama-server", paths.llama_server_path()), ("llama-swap", paths.llama_swap_path())):
        if p.exists():
            out[label] = "installed"
    return out
