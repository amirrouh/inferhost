"""Download and manage prebuilt binaries: llama.cpp (llama-server) and llama-swap."""
from __future__ import annotations

import contextlib
import io
import os
import platform
import re
import shutil
import stat
import sys
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

LLAMACPP_REPO = "ggml-org/llama.cpp"
LLAMASWAP_REPO = "mostlygeek/llama-swap"
STABLE_DIFFUSION_REPO = "leejet/stable-diffusion.cpp"

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
    """Fetch a llama-server release from upstream ggml-org/llama.cpp.

    Upstream tags follow the ``bNNNN`` format (e.g. ``b9320``). When version
    is ``"latest"`` we hit ``releases/latest``; otherwise we resolve the tag
    directly. Accepts the user-supplied value with or without a leading ``b``.

    Race-condition handling for ``"latest"``: upstream publishes the release
    tag first, then uploads the per-platform tarballs over the next ~10 min.
    During that gap, ``releases/latest`` returns the new tag with
    ``assets: []`` and the install fails. We detect this and walk the
    ``releases`` list to find the most recent release that actually has
    assets attached.
    """
    repo = LLAMACPP_REPO
    if version == "latest":
        rel = _release_json(repo, "latest")
        if rel.get("assets"):
            return rel
        return _find_latest_release_with_assets(repo, skip_tag=rel.get("tag_name"))
    tag = version if version.startswith("b") else f"b{version}"
    return _release_json(repo, tag)


def _find_latest_release_with_assets(repo: str, skip_tag: str | None = None) -> dict:
    """Walk recent releases until one with non-empty assets is found.

    GitHub's ``/releases`` endpoint returns releases newest-first. We skip
    the tag passed in ``skip_tag`` (the just-published one with no assets
    yet) and return the next release down. Page size 10 is plenty —
    upstream publishes roughly daily and the assets-attached lag is minutes.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = httpx.get(
        f"{GH_API}/repos/{repo}/releases",
        params={"per_page": "10"},
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    r.raise_for_status()
    for rel in r.json():
        if rel.get("tag_name") == skip_tag:
            continue
        if rel.get("assets"):
            return rel
    raise RuntimeError(
        f"No recent {repo} release has assets attached yet. "
        "Upstream may be mid-publish — try again in a few minutes, or set "
        "INFERHOST_LLAMACPP_VERSION to a specific tag (e.g. b9329)."
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


def _backend_substring_order(want_gpu: bool, backend: str) -> tuple[str, ...]:
    """Map platform + backend choice -> ordered substrings to match upstream asset names.

    Upstream ggml-org/llama.cpp asset names look like:
      llama-bNNNN-bin-ubuntu-x64.tar.gz                 (Linux x64 CPU)
      llama-bNNNN-bin-ubuntu-arm64.tar.gz               (Linux arm64 CPU)
      llama-bNNNN-bin-ubuntu-vulkan-x64.tar.gz          (Linux x64 Vulkan)
      llama-bNNNN-bin-ubuntu-vulkan-arm64.tar.gz        (Linux arm64 Vulkan)
      llama-bNNNN-bin-ubuntu-rocm-7.2-x64.tar.gz        (Linux x64 ROCm / AMD)
      llama-bNNNN-bin-ubuntu-sycl-fp16-x64.tar.gz       (Linux x64 SYCL / Intel)
      llama-bNNNN-bin-ubuntu-openvino-2026.0-x64.tar.gz (Linux x64 OpenVINO)
      llama-bNNNN-bin-macos-arm64.tar.gz                (macOS arm64 Metal)
      llama-bNNNN-bin-macos-x64.tar.gz                  (macOS x64 CPU)

    Note: upstream does NOT publish a Linux CUDA prebuilt — NVIDIA users on
    Linux should use Vulkan (works on every NVIDIA driver) or supply their
    own CUDA build via INFERHOST_LLAMA_SERVER_PATH.
    """
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")

    if sysname == "darwin":
        # macOS bundles Metal in the default build; "metal" is just an alias.
        if backend in ("auto", "metal", "cpu"):
            return ("bin-macos-arm64",) if is_arm else ("bin-macos-x64",)
        raise RuntimeError(
            f"Backend '{backend}' is not available on macOS. "
            "macOS uses Metal, which is bundled in the default build."
        )

    if sysname != "linux":
        raise RuntimeError(
            f"Unsupported OS '{sysname}'. inferhost supports Linux and macOS."
        )

    arch_suffix = "arm64" if is_arm else "x64"

    if backend == "auto":
        # NVIDIA detected -> Vulkan (no upstream Linux CUDA build available).
        # No GPU -> CPU. AMD/Intel users should set INFERHOST_LLAMACPP_BACKEND.
        if want_gpu:
            return (f"bin-ubuntu-vulkan-{arch_suffix}", f"bin-ubuntu-{arch_suffix}")
        return (f"bin-ubuntu-{arch_suffix}",)
    if backend == "vulkan":
        return (f"bin-ubuntu-vulkan-{arch_suffix}",)
    if backend == "rocm":
        return ("bin-ubuntu-rocm-", f"-{arch_suffix}.tar.gz")  # match version-suffixed name
    if backend == "sycl":
        return ("bin-ubuntu-sycl-",)
    if backend == "openvino":
        return ("bin-ubuntu-openvino-",)
    if backend == "cpu":
        return (f"bin-ubuntu-{arch_suffix}",)
    if backend == "cuda":
        raise RuntimeError(
            "Upstream ggml-org/llama.cpp does not publish a Linux CUDA prebuilt. "
            "Use INFERHOST_LLAMACPP_BACKEND=vulkan (works on every NVIDIA driver) "
            "or set INFERHOST_LLAMA_SERVER_PATH to a self-built CUDA binary."
        )
    raise RuntimeError(
        f"Unknown INFERHOST_LLAMACPP_BACKEND='{backend}'. "
        "Accepted: auto | vulkan | rocm | sycl | openvino | cpu."
    )


def _pick_llamacpp_asset(
    assets: list[dict], want_gpu: bool, backend: str = "auto"
) -> ReleaseAsset:
    """Pick a llama-server tarball from an upstream ggml-org/llama.cpp release.

    Selection is by substring match against the candidate list returned by
    :func:`_backend_substring_order`. We exclude ``cudart-*`` redistributable
    archives (Windows-only CUDA runtime) and anything ending in ``.zip``
    (Windows builds) since inferhost is Linux/macOS only.
    """
    candidates = _backend_substring_order(want_gpu=want_gpu, backend=backend)

    usable = [
        a for a in assets
        if a.get("name", "").endswith(".tar.gz")
        and not a["name"].startswith("cudart-")
        and "browser_download_url" in a
    ]

    for needle in candidates:
        for a in usable:
            if needle in a["name"]:
                return ReleaseAsset(
                    name=a["name"],
                    download_url=a["browser_download_url"],
                    size=a.get("size", 0),
                )

    available = [a["name"] for a in usable]
    raise RuntimeError(
        f"No matching llama.cpp asset for backend='{backend}' "
        f"(tried substrings: {list(candidates)}). "
        f"Available assets in this release: {available}. "
        "Set INFERHOST_LLAMACPP_BACKEND to one of "
        "vulkan/rocm/sycl/openvino/cpu, or use INFERHOST_LLAMA_SERVER_PATH "
        "to point at a custom build."
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


def _unlink_before_write(target: Path) -> None:
    """Remove ``target`` before rewriting it, so a *running* binary can be replaced.

    Opening an executable that's currently running for writing fails with
    ``[Errno 26] Text file busy`` (ETXTBSY) — this bit llama-swap upgrades while
    the daemon was live. Unlinking first drops the directory entry while the
    running process keeps the old inode, then the fresh write creates a new
    inode. Same trick `_purge_llamacpp_files` uses for llama.cpp; doing it here
    makes every extracted binary (notably llama-swap) replace-safe.
    """
    with contextlib.suppress(FileNotFoundError, OSError):
        if target.is_symlink() or target.exists():
            target.unlink()


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
                _unlink_before_write(target)
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
                _unlink_before_write(target)
                with target.open("wb") as dst:
                    shutil.copyfileobj(f, dst)
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                extracted.append(target)
    else:
        raise RuntimeError(f"Unsupported archive type: {name}")
    return extracted


def _link_so_versions(directory: Path) -> None:
    """For each libfoo.so.MAJOR.MINOR[.PATCH], create symlinks libfoo.so.MAJOR and libfoo.so."""
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
    # Read through Settings, not os.environ — the value usually lives in
    # ~/.config/inferhost/inferhost.env, which pydantic-settings loads into the
    # model without ever exporting it to the process environment.
    custom_path = settings().llama_server_path.strip()
    if custom_path:
        exe = Path(custom_path).expanduser()
        if not exe.exists():
            raise RuntimeError(
                f"INFERHOST_LLAMA_SERVER_PATH is set to '{custom_path}' "
                "but that file does not exist."
            )
        return InstalledBinary(path=exe, version="custom")

    paths.ensure_dirs()
    s = settings()
    version = version or s.llamacpp_version
    rel = _llamacpp_release_json(version)
    asset = _pick_llamacpp_asset(
        rel["assets"], want_gpu=probe().has_gpu, backend=s.llamacpp_backend
    )
    blob = _download(asset.download_url, progress_cb=progress_cb)
    # Purge any leftover llama.cpp files from a previous install BEFORE
    # extracting new ones. Without this, both old and new versioned .so
    # files coexist (e.g. libggml-base.so.0.12.0 and libggml-base.so.0.13.0)
    # and _link_so_versions picks one arbitrarily based on iterdir order —
    # the resulting ABI mismatch makes llama-server segfault on load.
    # Preserves llama-swap (different repo) and the source marker.
    _purge_llamacpp_files(paths.bin_dir())
    # Pull both the chat server and the standalone TTS tool. llama-tts is a
    # separate one-shot binary (OuteTTS+vocoder synthesis lives only there, not
    # in llama-server) — bundling it here means `install`/`update` light up the
    # /v1/audio/speech endpoint with no manual steps. It shares the same .so set.
    _extract_archive(
        blob,
        asset.name,
        paths.bin_dir(),
        want_basenames=("llama-server", "llama-tts"),
        take_libs=True,
    )
    _write_source_marker(paths.bin_dir(), LLAMACPP_REPO, rel.get("tag_name", "unknown"))
    target = paths.llama_server_path()
    if not target.exists():
        raise RuntimeError(f"llama-server not found inside {asset.name}")
    if not paths.llama_tts_path().exists():
        # Non-fatal: a backend variant might omit llama-tts. Chat still works;
        # only TTS models are affected, and they surface a clear error at serve
        # time. Warn so the absence isn't silent.
        print(
            f"inferhost: note — {asset.name} did not contain llama-tts; "
            "text-to-speech models won't be servable until a build that ships it "
            "is installed.",
            file=sys.stderr,
        )
    _link_so_versions(paths.bin_dir())
    return InstalledBinary(path=target, version=rel.get("tag_name", "unknown"))


def _purge_llamacpp_files(bin_dir: Path) -> None:
    """Remove llama.cpp-shipped files from `bin_dir` so a reinstall is hermetic.

    Keeps llama-swap (different upstream repo, different release cadence) and
    the source marker untouched. Targets: the llama-server launcher, every
    lib*.so* / lib*.dylib (versioned files + symlinks), and any sibling
    llama-* / *.metallib / *.json that may have been extracted alongside.
    """
    if not bin_dir.exists():
        return
    keep = {"llama-swap", _SOURCE_MARKER}
    for f in bin_dir.iterdir():
        if f.name in keep:
            continue
        name = f.name
        is_lib = (
            ("." in name and ".so" in name and name.startswith("lib"))
            or (name.startswith("lib") and name.endswith(".dylib"))
        )
        is_llama_exe = name == "llama-server" or name.startswith("llama-")
        if not (is_lib or is_llama_exe):
            continue
        with contextlib.suppress(OSError):
            if f.is_symlink() or f.is_file():
                f.unlink()


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
    out: dict[str, str | None] = {"llama-server": None, "llama-swap": None, "sd-server": None}
    for label, p in (
        ("llama-server", paths.llama_server_path()),
        ("llama-swap", paths.llama_swap_path()),
        ("sd-server", paths.sd_server_path()),
    ):
        if p.exists():
            out[label] = "installed"
    return out


# ---- stable-diffusion.cpp (image generation via sd-server) ----

_SD_SOURCE_MARKER = ".sd-server.source"


def _sdcpp_release_json(version: str) -> dict:
    """Fetch a stable-diffusion.cpp release.

    Upstream publishes rolling ``master-*`` tags. ``"latest"`` hits
    ``releases/latest`` and falls back to the newest release that actually has
    assets if the just-published one is still uploading (same race as llama.cpp).
    """
    if version == "latest":
        rel = _release_json(STABLE_DIFFUSION_REPO, "latest")
        if rel.get("assets"):
            return rel
        return _find_latest_release_with_assets(
            STABLE_DIFFUSION_REPO, skip_tag=rel.get("tag_name")
        )
    return _release_json(STABLE_DIFFUSION_REPO, version)


def _sdcpp_backend_substrings(want_gpu: bool, backend: str) -> tuple[str, ...]:
    """Map platform + backend -> ordered substrings matching sd.cpp asset names.

    Asset names look like:
      sd-master-<hash>-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip   (Linux Vulkan)
      sd-master-<hash>-bin-Linux-Ubuntu-24.04-x86_64.zip          (Linux CPU)
      sd-master-<hash>-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.2.1.zip (Linux ROCm)
      sd-master-<hash>-bin-Darwin-macOS-15.7.7-arm64.zip          (macOS arm64)

    Like llama.cpp, there is no Linux CUDA build — NVIDIA Linux uses Vulkan.
    Note these are ``.zip`` (not ``.tar.gz``) and use ``x86_64`` (not ``x64``).
    """
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")

    if sysname == "darwin":
        if is_arm:
            return ("-arm64.zip",)
        raise RuntimeError(
            "stable-diffusion.cpp publishes only an arm64 macOS build; "
            "Intel macOS is unsupported for image generation."
        )
    if sysname != "linux":
        raise RuntimeError(f"Unsupported OS '{sysname}'. inferhost supports Linux and macOS.")
    if is_arm:
        raise RuntimeError(
            "stable-diffusion.cpp has no Linux arm64 prebuilt; image generation "
            "needs an x86_64 Linux host."
        )

    if backend == "auto":
        if want_gpu:
            return ("-x86_64-vulkan.zip", "-x86_64.zip")  # Vulkan, else CPU
        return ("-x86_64.zip",)
    if backend in ("vulkan", "cuda"):
        # No Linux CUDA build; Vulkan covers every NVIDIA driver.
        return ("-x86_64-vulkan.zip",)
    if backend == "rocm":
        return ("-x86_64-rocm",)  # version-suffixed
    if backend == "cpu":
        return ("-x86_64.zip",)
    raise RuntimeError(
        f"Backend '{backend}' has no stable-diffusion.cpp Linux build. "
        "Use vulkan, rocm, or cpu."
    )


def _pick_sdcpp_asset(assets: list[dict], want_gpu: bool, backend: str = "auto") -> ReleaseAsset:
    """Pick an sd-server zip from a leejet/stable-diffusion.cpp release."""
    candidates = _sdcpp_backend_substrings(want_gpu=want_gpu, backend=backend)
    usable = [
        a for a in assets
        if a.get("name", "").endswith(".zip")
        and not a["name"].startswith("cudart-")
        and "-win-" not in a["name"]
        and "browser_download_url" in a
    ]
    for needle in candidates:
        for a in usable:
            if needle in a["name"]:
                return ReleaseAsset(
                    name=a["name"],
                    download_url=a["browser_download_url"],
                    size=a.get("size", 0),
                )
    raise RuntimeError(
        f"No matching stable-diffusion.cpp asset for backend='{backend}' "
        f"(tried: {list(candidates)}). Available: {[a['name'] for a in usable]}."
    )


def install_stable_diffusion(
    version: str | None = None, progress_cb: ProgressCallback | None = None
) -> InstalledBinary:
    """Download sd-server + libstable-diffusion.so into the isolated sd/ subdir.

    Kept separate from the llama.cpp install so the llama.cpp purge never touches
    these files (see :func:`paths.sd_bin_dir`).
    """
    paths.ensure_dirs()
    s = settings()
    version = version or s.sdcpp_version
    rel = _sdcpp_release_json(version)
    asset = _pick_sdcpp_asset(rel["assets"], want_gpu=probe().has_gpu, backend=s.llamacpp_backend)
    blob = _download(asset.download_url, progress_cb=progress_cb)
    _extract_archive(
        blob,
        asset.name,
        paths.sd_bin_dir(),
        want_basenames=("sd-server", "sd-cli"),
        take_libs=True,
    )
    target = paths.sd_server_path()
    if not target.exists():
        raise RuntimeError(f"sd-server not found inside {asset.name}")
    _link_so_versions(paths.sd_bin_dir())
    with contextlib.suppress(OSError):
        (paths.sd_bin_dir() / _SD_SOURCE_MARKER).write_text(
            f"{STABLE_DIFFUSION_REPO}\n{rel.get('tag_name', 'unknown')}\n", encoding="utf-8"
        )
    return InstalledBinary(path=target, version=rel.get("tag_name", "unknown"))


def needs_sdcpp_refresh() -> bool:
    """True when sd-server should be (re)fetched — i.e. it's not installed yet.

    Lets existing installs pick up image-generation support on the next start
    without a manual reinstall (mirrors :func:`needs_llama_server_refresh`).
    """
    return not paths.sd_server_path().exists()


_SOURCE_MARKER = ".llama-server.source"

# Minimum upstream llama.cpp build that ships DFlash speculative decoding
# (--spec-type draft-dflash). An installed binary older than this can't serve
# DFlash drafts, so needs_llama_server_refresh forces a one-time re-fetch to
# pull a build that can — but ONLY when the marker's tag parses confidently as
# bNNNN below this floor (a "custom"/unparseable tag is left alone so we never
# thrash-download over a user's hand-picked build).
_LLAMACPP_MIN_BUILD = 9831


def _parse_build_number(tag: str) -> int | None:
    """Parse an upstream ``bNNNN`` release tag into its integer build number.

    Returns None for anything that isn't confidently a ``bNNNN`` tag —
    "custom", "unknown", a git hash, an empty string, etc. Callers treat None
    as "don't touch it" so a non-standard binary is never re-downloaded on a
    guess.
    """
    tag = (tag or "").strip()
    m = re.fullmatch(r"b(\d+)", tag)
    return int(m.group(1)) if m else None


def _write_source_marker(bin_dir: Path, repo: str, tag: str) -> None:
    """Record where the installed llama-server came from.

    Used by :func:`needs_llama_server_refresh` so a user upgrading from an
    older inferhost that pulled binaries from a different repo
    automatically re-downloads from the current source.
    """
    # Marker is advisory; failing to write must not break the install.
    with contextlib.suppress(OSError):
        (bin_dir / _SOURCE_MARKER).write_text(f"{repo}\n{tag}\n", encoding="utf-8")


def needs_llama_server_refresh() -> bool:
    """Return True when the installed llama-server should be re-fetched.

    Triggers a refresh in three cases:
      1. No binary on disk yet.
      2. Binary on disk, but the marker file says it came from a different
         repo than the current ``LLAMACPP_REPO`` (e.g. user is upgrading
         from a build of inferhost that bundled a fork).
      3. llama-server is present but llama-tts is missing — an install that
         predates TTS support. Re-fetching pulls llama-tts alongside, so
         existing users get the /v1/audio/speech endpoint on their next start.

    When ``INFERHOST_LLAMA_SERVER_PATH`` is set, the user is in custom-binary
    mode and we never overwrite their choice.
    """
    if settings().llama_server_path.strip():
        return False
    if not paths.llama_server_path().exists():
        return True
    if not paths.llama_tts_path().exists():
        return True
    marker = paths.bin_dir() / _SOURCE_MARKER
    if not marker.exists():
        # No marker means the binary predates this tracking — assume stale
        # so the upgrade path lands on the current source.
        return True
    try:
        marker_lines = marker.read_text(encoding="utf-8").splitlines()
        recorded_repo = marker_lines[0].strip()
    except (OSError, IndexError):
        return True
    if recorded_repo != LLAMACPP_REPO:
        return True
    # Min-build gate for DFlash (draft-dflash landed at b9831). Force a
    # one-time refresh ONLY when the marker's tag parses confidently as a
    # bNNNN build below the floor. A "custom"/unparseable tag (or a missing
    # tag line) returns None from _parse_build_number and is left untouched —
    # we never re-download over a build we can't reason about, which avoids
    # thrash-downloads on every start.
    recorded_tag = marker_lines[1].strip() if len(marker_lines) > 1 else ""
    build = _parse_build_number(recorded_tag)
    return build is not None and build < _LLAMACPP_MIN_BUILD
