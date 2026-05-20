"""Hugging Face Hub: list GGUF files, derive names, download models."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from inferhost.core.quant import QUANT_RANK, extract_quant


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
