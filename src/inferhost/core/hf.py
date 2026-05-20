"""Hugging Face Hub: list GGUF files, derive names, download models."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from inferhost.core.quant import QUANT_RANK, extract_quant

ProgressCallback = Callable[[int, int], None]


@dataclass
class GgufFile:
    repo_id: str
    filename: str
    size_bytes: int
    quant: str | None  # e.g. "Q4_K_M", or None if not detectable

    @property
    def size_gib(self) -> float:
        return round(self.size_bytes / (1024**3), 2)

    @property
    def quant_rank(self) -> int:
        return QUANT_RANK.get(self.quant or "", 99)


_INVALID_NAME_CHARS = re.compile(r"[^a-z0-9._-]+")


def normalize_name(repo_id: str) -> str:
    base = repo_id.split("/", 1)[-1].lower()
    for suffix in ("-gguf", ".gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = _INVALID_NAME_CHARS.sub("-", base).strip("-.")
    return base or repo_id.lower().replace("/", "-")


def _api() -> HfApi:
    return HfApi()


def list_ggufs(repo_id: str) -> list[GgufFile]:
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError as e:
        raise ValueError(f"Hugging Face repo not found: {repo_id}") from e
    except HfHubHTTPError as e:
        raise ValueError(f"Unable to fetch repo metadata for {repo_id}: {e}") from e

    files: list[GgufFile] = []
    for sib in info.siblings:
        fname = sib.rfilename
        if not fname.endswith(".gguf"):
            continue
        # Skip multi-part files for now (e.g. -00001-of-00003.gguf) — handled in v0.2
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", fname):
            continue
        files.append(
            GgufFile(
                repo_id=repo_id,
                filename=fname,
                size_bytes=sib.size or 0,
                quant=extract_quant(fname),
            )
        )
    files.sort(key=lambda f: (f.quant_rank, f.size_bytes))
    return files


def download_gguf(repo_id: str, filename: str, cache_dir: Path | None = None) -> Path:
    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def blobs_dir(repo_id: str, cache_dir: Path | None = None) -> Path:
    """The Hugging Face cache subdirectory where blobs (and *.incomplete temp files) live."""
    root = Path(cache_dir) if cache_dir else Path(HF_HUB_CACHE)
    folder = "models--" + repo_id.replace("/", "--")
    return root / folder / "blobs"


def download_gguf_with_progress(
    repo_id: str,
    filename: str,
    expected_bytes: int,
    progress_cb: ProgressCallback,
    cache_dir: Path | None = None,
    poll_interval: float = 0.3,
) -> Path:
    """Download a GGUF and report progress by polling the HF cache blobs directory.

    Spawns a daemon thread that watches `<cache>/models--<org>--<repo>/blobs/` for the
    growing `.incomplete` file (or the final blob if the download is small enough to skip
    the temp file). Reports `(downloaded_bytes, total_bytes)` to the callback.
    """
    import threading
    import time

    bdir = blobs_dir(repo_id, cache_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    pre_existing = {p.name for p in bdir.iterdir()} if bdir.exists() else set()
    start_time = time.time()
    done = threading.Event()
    result: dict[str, object] = {}

    def _worker() -> None:
        try:
            result["path"] = download_gguf(repo_id, filename, cache_dir=cache_dir)
        except Exception as e:  # noqa: BLE001
            result["error"] = e
        finally:
            done.set()

    def _watcher() -> None:
        import contextlib

        last = -1
        while not done.is_set():
            current = _current_progress(bdir, pre_existing, start_time)
            if current != last:
                with contextlib.suppress(Exception):
                    progress_cb(current, expected_bytes)
                last = current
            time.sleep(poll_interval)

    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_watcher, daemon=True).start()
    done.wait()
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    progress_cb(expected_bytes, expected_bytes)
    return result["path"]  # type: ignore[return-value]


def _current_progress(bdir: Path, pre_existing: set[str], start_time: float) -> int:
    if not bdir.exists():
        return 0
    best = 0
    for f in bdir.iterdir():
        if not f.is_file():
            continue
        if f.name in pre_existing:
            try:
                if f.stat().st_mtime < start_time:
                    continue
            except OSError:
                continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size > best:
            best = size
    return best
