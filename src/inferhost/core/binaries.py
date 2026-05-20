"""Download and manage prebuilt binaries: llama.cpp (llama-server) and llama-swap."""
from __future__ import annotations

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

ProgressCallback = Callable[[int, int], None]

from inferhost.core import paths
from inferhost.core.probe import probe
from inferhost.settings import settings

LLAMACPP_REPO = "ggml-org/llama.cpp"
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


def _platform_keys() -> tuple[str, str, str]:
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    if sysname == "darwin":
        os_key = "macos"
        swap_os = "darwin"
    elif sysname == "linux":
        os_key = "linux"
        swap_os = "linux"
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


_BACKEND_TAGS = ("cuda", "cu12", "cu11", "vulkan", "rocm", "hip", "sycl", "openvino", "kompute")


def _asset_backend(name: str) -> str:
    n = name.lower()
    for tag in _BACKEND_TAGS:
        if tag in n:
            return "cuda" if tag in ("cuda", "cu12", "cu11") else tag
    return "cpu"


def _pick_llamacpp_asset(
    assets: list[dict], os_key: str, arch: str, want_gpu: bool, preferred_backend: str | None = None
) -> ReleaseAsset:
    candidates = []
    for a in assets:
        name = a.get("name", "")
        lname = name.lower()
        if not (lname.endswith(".zip") or lname.endswith(".tar.gz")):
            continue
        if os_key == "linux" and not ("linux" in lname or "ubuntu" in lname):
            continue
        if os_key == "macos" and "macos" not in lname:
            continue
        if arch not in lname:
            continue
        candidates.append(a)

    if not candidates:
        raise RuntimeError(
            f"No llama.cpp asset found for os={os_key} arch={arch}. "
            f"Available: {[a.get('name') for a in assets][:10]}"
        )

    # macOS releases are universally Metal-accelerated; pick the plain build.
    if os_key == "macos":
        for a in candidates:
            if _asset_backend(a["name"]) == "cpu":
                return ReleaseAsset(name=a["name"], download_url=a["browser_download_url"], size=a.get("size", 0))
        a = candidates[0]
        return ReleaseAsset(name=a["name"], download_url=a["browser_download_url"], size=a.get("size", 0))

    if want_gpu:
        ranked_backends = (
            (preferred_backend,) if preferred_backend else ()
        ) + ("cuda", "vulkan", "rocm", "sycl", "openvino", "cpu")
    else:
        ranked_backends = ("cpu", "vulkan", "openvino", "sycl", "rocm", "cuda")

    by_backend: dict[str, dict] = {}
    for a in candidates:
        b = _asset_backend(a["name"])
        if b not in by_backend or a.get("size", 0) < by_backend[b].get("size", 0):
            by_backend[b] = a

    for backend in ranked_backends:
        if backend and backend in by_backend:
            a = by_backend[backend]
            return ReleaseAsset(name=a["name"], download_url=a["browser_download_url"], size=a.get("size", 0))

    a = candidates[0]
    return ReleaseAsset(name=a["name"], download_url=a["browser_download_url"], size=a.get("size", 0))


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
    if base.startswith("lib") and (".so" in base or ".dylib" in base):
        return True
    return False


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
        if take_libs and _is_lib_or_binary(member_name):
            return True
        return False

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
        if not f.is_file():
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
                try:
                    link_path.symlink_to(f.name)
                except OSError:
                    pass
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
            try:
                link_path.symlink_to(f.name)
            except OSError:
                pass


def install_llama_server(
    version: str | None = None, progress_cb: ProgressCallback | None = None
) -> InstalledBinary:
    paths.ensure_dirs()
    version = version or settings().llamacpp_version
    rel = _release_json(LLAMACPP_REPO, version)
    os_key, arch, _ = _platform_keys()
    preferred = os.environ.get("INFERHOST_LLAMACPP_BACKEND")
    asset = _pick_llamacpp_asset(rel["assets"], os_key, arch, want_gpu=probe().has_gpu, preferred_backend=preferred)
    blob = _download(asset.download_url, progress_cb=progress_cb)
    extracted = _extract_archive(
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
