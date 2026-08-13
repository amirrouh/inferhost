"""Hugging Face Hub: list GGUF files, derive names, download models."""
from __future__ import annotations

import contextlib
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from inferhost.core import paths
from inferhost.core.quant import QUANT_RANK, extract_quant

ProgressCallback = Callable[[int, int], None]


@dataclass
class GgufFile:
    repo_id: str
    filename: str
    size_bytes: int
    quant: str | None  # e.g. "Q4_K_M", or None if not detectable
    # Non-empty for a multi-part GGUF (e.g. model-00001-of-00003.gguf): every
    # shard as (filename, size_bytes), ordered 1..N. `filename`/`size_bytes`
    # above describe shard 1 (what gets stored as local_path — llama-server
    # auto-discovers the rest from its path) and the summed size respectively.
    # Empty tuple = a normal single-file GGUF.
    parts: tuple[tuple[str, int], ...] = ()

    @property
    def size_gib(self) -> float:
        return round(self.size_bytes / (1024**3), 2)

    @property
    def quant_rank(self) -> int:
        return QUANT_RANK.get(self.quant or "", 99)


_INVALID_NAME_CHARS = re.compile(r"[^a-z0-9._-]+")


def parse_repo_id(text: str) -> str:
    """Turn user input into a bare ``owner/name`` repo id.

    Accepts a plain repo id (``Qwen/Qwen2.5-7B-Instruct-GGUF``) unchanged, or a
    full Hugging Face URL — with or without scheme, ``/tree/…`` / ``/blob/…``
    path, query string, or a trailing ``.git`` — and strips it down to the
    ``owner/name`` the Hub API expects, so users can paste a link straight from
    the browser.
    """
    s = text.strip()
    if not s:
        return s
    # Drop scheme and any host (huggingface.co, hf.co, www.…).
    s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", s)
    s = re.sub(r"^(?:www\.)?(?:huggingface\.co|hf\.co)/", "", s)
    # Strip query string / fragment.
    s = s.split("?", 1)[0].split("#", 1)[0]
    # Drop a HF path suffix like /tree/main, /blob/main/file.gguf, /resolve/….
    parts = s.strip("/").split("/")
    for i, seg in enumerate(parts):
        if seg in ("tree", "blob", "resolve", "raw"):
            parts = parts[:i]
            break
    s = "/".join(parts)
    if s.endswith(".git"):
        s = s[: -len(".git")]
    return s.strip("/")


def normalize_name(repo_id: str) -> str:
    base = repo_id.split("/", 1)[-1].lower()
    for suffix in ("-gguf", ".gguf", "-onnx"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = _INVALID_NAME_CHARS.sub("-", base).strip("-.")
    return base or repo_id.lower().replace("/", "-")


def _api() -> HfApi:
    return HfApi()


# Matches one shard of a multi-part GGUF, e.g. "qwen3-235b-Q4_K_M-00001-of-00003.gguf".
_PART_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def list_ggufs(repo_id: str) -> list[GgufFile]:
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError as e:
        raise ValueError(f"Hugging Face repo not found: {repo_id}") from e
    except HfHubHTTPError as e:
        raise ValueError(f"Unable to fetch repo metadata for {repo_id}: {e}") from e

    files: list[GgufFile] = []
    # Multi-part GGUFs (e.g. -00001-of-00003.gguf) are grouped into a single
    # GgufFile per complete run rather than skipped — see the grouping pass
    # below. base -> {shard index: (filename, size_bytes)}, plus the total
    # shard count each base declares (taken from whichever shard we see it on).
    part_groups: dict[str, dict[int, tuple[str, int]]] = {}
    part_totals: dict[str, int] = {}
    for sib in info.siblings:
        fname = sib.rfilename
        if not fname.endswith(".gguf"):
            continue
        size = sib.size or 0
        m = _PART_RE.match(fname)
        if m is None:
            files.append(
                GgufFile(repo_id=repo_id, filename=fname, size_bytes=size, quant=extract_quant(fname))
            )
            continue
        base = m.group("base")
        part_groups.setdefault(base, {})[int(m.group("idx"))] = (fname, size)
        part_totals[base] = int(m.group("total"))

    for base, shards in part_groups.items():
        total = part_totals[base]
        # Only a complete run (every shard 1..total present) is servable — an
        # incomplete/mixed group (partial mirror, or shards from two repos
        # colliding on a base name) is silently dropped rather than offered
        # as a broken pick.
        if sorted(shards) != list(range(1, total + 1)):
            continue
        ordered = tuple(shards[i] for i in range(1, total + 1))
        shard1_name, _ = ordered[0]
        files.append(
            GgufFile(
                repo_id=repo_id,
                filename=shard1_name,
                size_bytes=sum(sz for _, sz in ordered),
                quant=extract_quant(base),
                parts=ordered,
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


def repo_file_size(repo_id: str, filename: str) -> int:
    """Return the byte size of one file in a repo, or 0 if unknown/missing.

    Used to give companion downloads (mmproj / vocoder / same-repo aux files)
    a real progress bar instead of an unbounded "N MiB downloaded so far"
    counter — see ``add_model.py``'s ``_download_companion``.
    """
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return 0
    for sib in info.siblings:
        if sib.rfilename == filename:
            return sib.size or 0
    return 0


_MMPROJ_PAT = re.compile(r"(?:^|[/-])mmproj[^/]*\.gguf$", re.IGNORECASE)

# Companion attachments that ride along with a main model in the same repo and
# are never served standalone: vision projectors (mmproj) and prism-ml's DSpark
# speculative-decoding drafters. Deliberately does NOT match "dflash" — DFlash
# drafts live in dedicated repos where the draft IS the main pick.
_COMPANION_PAT = re.compile(r"(?:^|[-_.])(?:mmproj|dspark)(?:[-_.]|$)", re.IGNORECASE)


def is_companion_file(filename: str) -> bool:
    """True if ``filename`` names a companion attachment (vision projector /
    DSpark drafter), so the add-model picker never recommends it as the main
    model even when its quant outranks the real candidates."""
    return bool(_COMPANION_PAT.search(filename))


def find_mmproj(repo_id: str) -> str | None:
    """Return the best-matching mmproj filename in the repo, or None if absent.

    Multimodal projector files are typically named like ``mmproj-<model>-<dtype>.gguf``
    or ``<model>-mmproj-<dtype>.gguf``. We prefer the smallest one (f16 is half the
    size of bf16 and works on virtually all GPUs).
    """
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return None
    candidates = [
        (sib.size or 0, sib.rfilename)
        for sib in info.siblings
        if _MMPROJ_PAT.search(sib.rfilename or "")
    ]
    if not candidates:
        return None
    candidates.sort()  # smallest first
    return candidates[0][1]


# Scoped to specific companion-file naming, not a bare "tokenizer" — that
# would false-positive on chat-model tokenizer.gguf-ish names. wavtokenizer
# covers OuteTTS-style repos.
_VOCODER_PAT = re.compile(r"(?:wavtokenizer|vocoder)[^/]*\.gguf$", re.IGNORECASE)


def find_vocoder(repo_id: str) -> str | None:
    """Return the best-matching vocoder filename in the repo, or None if absent.

    TTS models (OuteTTS et al.) pair a generation model with a WavTokenizer /
    vocoder GGUF that decodes audio codes into a waveform. Its presence is what
    marks a repo as a text-to-speech repo for inferhost (mirrors ``find_mmproj``
    for vision). Files are typically named ``WavTokenizer-*.gguf`` or contain
    ``vocoder``. Prefer the smallest match. The main model file never matches
    this pattern, so it won't be mistaken for the vocoder.
    """
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return None
    candidates = [
        (sib.size or 0, sib.rfilename)
        for sib in info.siblings
        if _VOCODER_PAT.search(sib.rfilename or "")
    ]
    if not candidates:
        return None
    candidates.sort()  # smallest first
    return candidates[0][1]


# ---- Orpheus (llama-server + SNAC) ----
#
# Orpheus-3B is a llama-architecture speech-LLM: its GGUF runs on the stack's
# normal llama-server (fronted by llama-swap), emitting SNAC audio tokens that
# the inferhost-tts daemon decodes to a waveform. The decoder is a small ONNX
# graph fetched from a fixed community export at add time (mirrors how Kokoro
# bundles its voices .npz) — onnxruntime already ships with inferhost, so no
# torch dependency. Detection is by repo name: every Orpheus base/finetune
# repo carries the family name, and its GGUFs are indistinguishable from a
# plain chat llama otherwise.
SNAC_ONNX_REPO = "onnx-community/snac_24khz-ONNX"
SNAC_DECODER_FILE = "onnx/decoder_model.onnx"


def is_orpheus_repo(repo_id: str) -> bool:
    """True when a repo holds an Orpheus-family TTS model (name-based)."""
    return "orpheus" in repo_id.lower()


# ---- Kokoro (kokoro-onnx) ----
#
# Kokoro-82M's official repo (hexgrad/Kokoro-82M) ships only PyTorch weights;
# the servable ONNX export lives in a community companion repo. Pasting either
# link works — the alias resolves the PyTorch repo to its ONNX counterpart so
# the usual paste-repo → pick-file flow stays unchanged.
KOKORO_ONNX_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
_TTS_REPO_ALIASES = {
    "hexgrad/kokoro-82m": KOKORO_ONNX_REPO,
}

# The ONNX repo names its precision variants by suffix; map them onto the
# quant labels the picker already knows how to rank/display.
_ONNX_QUANT_LABELS = {
    "model": "F32",
    "model_fp16": "F16",
    "model_quantized": "Q8_0",
    "model_q8f16": "Q8_F16",
    "model_q4": "Q4",
    "model_q4f16": "Q4_F16",
    "model_uint8": "U8",
    "model_uint8f16": "U8_F16",
}


def resolve_tts_repo(repo_id: str) -> str:
    """Map a pasted TTS repo onto the repo that holds its servable files."""
    return _TTS_REPO_ALIASES.get(repo_id.lower(), repo_id)


def list_tts_files(repo_id: str) -> list[GgufFile]:
    """List candidate main TTS model files in a repo (excludes companions).

    Two engine families, told apart by what the repo ships:
    - OuteTTS-style GGUF repos: every .gguf except the WavTokenizer/vocoder
      companion that find_vocoder fetches separately.
    - Kokoro ONNX repos: the .onnx precision variants; the per-voice style
      vectors under voices/ are bundled separately (list_kokoro_voice_files).
    """
    repo_id = resolve_tts_repo(repo_id)
    ggufs = [f for f in list_ggufs(repo_id) if not _VOCODER_PAT.search(f.filename)]
    if ggufs:
        return ggufs
    return _list_onnx_files(repo_id)


def _list_onnx_files(repo_id: str) -> list[GgufFile]:
    """List a repo's .onnx model files (Kokoro), reusing the GgufFile shape."""
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError as e:
        raise ValueError(f"Hugging Face repo not found: {repo_id}") from e
    except HfHubHTTPError as e:
        raise ValueError(f"Unable to fetch repo metadata for {repo_id}: {e}") from e
    files: list[GgufFile] = []
    for sib in info.siblings:
        fname = sib.rfilename or ""
        if not fname.endswith(".onnx"):
            continue
        stem = Path(fname).stem.lower()
        files.append(
            GgufFile(
                repo_id=repo_id,
                filename=fname,
                size_bytes=sib.size or 0,
                quant=_ONNX_QUANT_LABELS.get(stem, extract_quant(fname)),
            )
        )
    files.sort(key=lambda f: (f.quant_rank, f.size_bytes))
    return files


def list_kokoro_voice_files(repo_id: str) -> list[tuple[str, int]]:
    """(filename, size) for every per-voice style vector in a Kokoro repo.

    The ONNX repo ships one raw ``voices/<name>.bin`` per voice; the add-model
    flow downloads them all and bundles them into the single .npz that
    kokoro-onnx loads (build_kokoro_voices_npz).
    """
    repo_id = resolve_tts_repo(repo_id)
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return []
    return sorted(
        (sib.rfilename, sib.size or 0)
        for sib in info.siblings
        if (sib.rfilename or "").startswith("voices/") and sib.rfilename.endswith(".bin")
    )


def build_kokoro_voices_npz(voice_paths: dict[str, Path], out_path: Path) -> Path:
    """Bundle per-voice .bin style vectors into the single .npz kokoro-onnx loads.

    Each .bin is a raw float32 array of per-phoneme-count style vectors with
    shape (N, 1, 256) — kokoro-onnx indexes it by the phoneme length of the
    input at synthesis time. The np.savez keys become the selectable voice
    names. ``out_path`` must end in .npz (np.savez would otherwise append it).
    """
    import numpy as np

    voices = {
        name: np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256)
        for name, path in voice_paths.items()
    }
    np.savez(out_path, **voices)
    return out_path


def list_repo_files(repo_id: str) -> list[GgufFile]:
    """List ALL .gguf/.safetensors files in a repo (no companion filtering).

    Used by the component picker, where the user is choosing a VAE / text
    encoder / CLIP / T5 file — exactly the files list_image_files excludes.
    """
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError as e:
        raise ValueError(f"Hugging Face repo not found: {repo_id}") from e
    except HfHubHTTPError as e:
        raise ValueError(f"Unable to fetch repo metadata for {repo_id}: {e}") from e
    files: list[GgufFile] = []
    for sib in info.siblings:
        fname = sib.rfilename or ""
        if not (fname.endswith(".gguf") or fname.endswith(".safetensors")):
            continue
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", fname):
            continue
        files.append(
            GgufFile(repo_id=repo_id, filename=fname, size_bytes=sib.size or 0,
                     quant=extract_quant(fname))
        )
    files.sort(key=lambda f: (f.quant_rank, f.size_bytes))
    return files


def list_image_files(repo_id: str) -> list[GgufFile]:
    """List diffusion-model candidate files in a repo (.gguf and .safetensors).

    Image checkpoints ship as either GGUF (quantized) or .safetensors. Reuses the
    GgufFile shape so the add-model picker can show them the same way as LLMs;
    quant is None for safetensors (sorts last, which is fine).
    """
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError as e:
        raise ValueError(f"Hugging Face repo not found: {repo_id}") from e
    except HfHubHTTPError as e:
        raise ValueError(f"Unable to fetch repo metadata for {repo_id}: {e}") from e
    files: list[GgufFile] = []
    for sib in info.siblings:
        fname = sib.rfilename or ""
        if not (fname.endswith(".gguf") or fname.endswith(".safetensors")):
            continue
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", fname):
            continue
        # Skip obvious companion files (VAE / text encoders) — those are picked
        # up by find_sd_aux, not chosen as the main model.
        if _is_sd_aux(fname):
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


# Companion-file patterns for split (Flux/SD3) image models. Each maps a Model
# field to a filename regex. `ae.safetensors` is Flux's VAE.
_SD_AUX_PATTERNS: dict[str, re.Pattern] = {
    "vae_path": re.compile(r"(?:^|[/_-])(?:vae|ae)[^/]*\.(?:gguf|safetensors)$", re.IGNORECASE),
    "clip_l_path": re.compile(r"clip[_-]?l[^/]*\.(?:gguf|safetensors)$", re.IGNORECASE),
    "clip_g_path": re.compile(r"clip[_-]?g[^/]*\.(?:gguf|safetensors)$", re.IGNORECASE),
    "t5xxl_path": re.compile(r"t5[_-]?xxl[^/]*\.(?:gguf|safetensors)$", re.IGNORECASE),
    # Qwen/LLM text encoder (Qwen-Image, Z-Image). Conservative: only obvious
    # encoder names — NOT plain Qwen chat GGUFs — to avoid false positives. The
    # encoder usually lives in a separate repo anyway (use the component picker).
    "text_encoder_path": re.compile(
        r"(?:text[_-]?encoder|qwen[0-9._]*[_-]?vl|qwenvl)[^/]*\.(?:gguf|safetensors)$",
        re.IGNORECASE,
    ),
    # Vision ViT for Qwen-Image-Edit (sd-server --llm_vision). Named like
    # *mmproj* (multimodal projector), same convention as vision LLMs.
    "vision_encoder_path": re.compile(r"mmproj[^/]*\.(?:gguf|safetensors)$", re.IGNORECASE),
}


def _is_sd_aux(fname: str) -> bool:
    """True if a filename looks like a companion (VAE / encoder), not a main model.

    Derived from _SD_AUX_PATTERNS so the two never drift apart.
    """
    return any(p.search(fname) for p in _SD_AUX_PATTERNS.values())


def repo_tags(repo_id: str) -> list[str]:
    """Return a repo's HF tags (used to classify image-model families), or []."""
    try:
        info = _api().repo_info(repo_id)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return []
    return list(getattr(info, "tags", []) or [])


def find_sd_aux(repo_id: str) -> dict[str, str]:
    """Detect companion VAE / CLIP / T5 files for a split image model in a repo.

    Returns a dict of {Model field name: filename} for each companion found
    (smallest match per slot). Empty when the repo has none (single-file model).
    Mirrors find_mmproj / find_vocoder.
    """
    try:
        info = _api().repo_info(repo_id, files_metadata=True)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return {}
    found: dict[str, str] = {}
    for field, pat in _SD_AUX_PATTERNS.items():
        matches = sorted(
            (sib.size or 0, sib.rfilename)
            for sib in info.siblings
            if pat.search(sib.rfilename or "")
        )
        if matches:
            found[field] = matches[0][1]
    return found


def blobs_dir(repo_id: str, cache_dir: Path | None = None) -> Path:
    """The Hugging Face cache subdirectory where blobs (and *.incomplete temp files) live."""
    root = Path(cache_dir) if cache_dir else Path(HF_HUB_CACHE)
    folder = "models--" + repo_id.replace("/", "--")
    return root / folder / "blobs"


def repo_dir(repo_id: str, cache_dir: Path | None = None) -> Path:
    """The Hugging Face cache directory holding one repo's blobs and snapshots."""
    root = Path(cache_dir) if cache_dir else Path(HF_HUB_CACHE)
    return root / ("models--" + repo_id.replace("/", "--"))


def _shard_siblings(path: Path) -> list[Path]:
    """Every shard of a multi-part GGUF, given any one of them.

    A sharded model is stored as ``...-00001-of-00003.gguf``; the registry keeps
    only shard 1 (llama-server opens the rest itself), so deleting just that
    path would strand the other shards on disk forever.
    """
    m = _PART_RE.match(path.name)
    if not m:
        return [path]
    base, total = m.group("base"), int(m.group("total"))
    found = [
        sib for i in range(1, total + 1)
        if (sib := path.with_name(f"{base}-{i:05d}-of-{total:05d}.gguf")).exists()
    ]
    return found or [path]


def _resolve_targets(path: Path) -> list[Path]:
    """The paths to unlink to actually reclaim ``path``'s bytes.

    Files in the HF cache are symlinks from ``snapshots/<rev>/`` into
    ``blobs/<sha>``, and the bytes live in the blob. Removing only the link
    frees nothing, which is the whole complaint — so return both.
    """
    out: list[Path] = []
    for p in _shard_siblings(path):
        out.append(p)
        if p.is_symlink():
            with contextlib.suppress(OSError):
                blob = p.resolve()
                if blob.exists():
                    out.append(blob)
    return out


def _own_bytes(path: Path) -> int:
    """Disk actually reclaimed by unlinking ``path``.

    A snapshot entry is a symlink into ``blobs/``; its own inode holds only the
    target string. The blob is returned alongside it and carries the real size,
    so counting the link too would inflate every figure we show the user.
    """
    if path.is_symlink():
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def model_file_paths(m) -> list[Path]:
    """Every weight file a registry entry owns, shards included.

    Only paths inside the Hugging Face cache or inferhost's own models dir are
    returned. A model added from an arbitrary location on disk is the user's
    file that inferhost merely pointed at — deleting a registry entry must
    never delete that.
    """
    fields = (
        m.local_path, m.mmproj_path, m.vocoder_path, m.vae_path, m.clip_l_path,
        m.clip_g_path, m.t5xxl_path, m.text_encoder_path, m.vision_encoder_path,
        m.draft_model_path,
    )
    owned_roots = [Path(HF_HUB_CACHE).resolve(), paths.models_dir().resolve()]
    out: list[Path] = []
    for raw in fields:
        if not raw:
            continue
        p = Path(raw)
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if not any(resolved.is_relative_to(root) or p.is_relative_to(root)
                   for root in owned_roots):
            continue
        out.extend(_resolve_targets(p))
    # De-duplicate while preserving order: several fields can share one blob.
    seen: set[Path] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def paths_used_by_others(reg, exclude_name: str) -> set[Path]:
    """Files every OTHER registered model owns — the keep-list when deleting.

    Two entries routinely share bytes: two quants from one repo sharing an
    mmproj, or a target and its DFlash draft. Deleting one must not break the
    other.
    """
    keep: set[Path] = set()
    for other in reg.models:
        if other.name == exclude_name:
            continue
        keep.update(model_file_paths(other))
    return keep


def deletable_bytes(m, keep: Iterable[Path] = ()) -> int:
    """Bytes freed by :func:`delete_model_files` — for the confirm prompt."""
    keep_set = {Path(k) for k in keep}
    return sum(_own_bytes(p) for p in model_file_paths(m) if p not in keep_set)


def delete_model_files(m, keep: Iterable[Path] = ()) -> int:
    """Delete ``m``'s weights and return the bytes reclaimed.

    ``keep`` lists paths still owned by other registered models — two entries
    can share one repo (different quants sharing an mmproj, a base model and
    its DFlash draft), and deleting a file another model still serves would
    break that model.
    """
    keep_set = {Path(k) for k in keep}
    freed = 0
    for p in model_file_paths(m):
        if p in keep_set:
            continue
        size = _own_bytes(p)
        try:
            p.unlink()
        except OSError:
            continue
        freed += size
    _prune_empty_repo_dirs(m)
    return freed


def _prune_empty_repo_dirs(m) -> None:
    """Remove the repo's cache skeleton once its last blob is gone.

    Without this the cache keeps a tree of empty snapshots/refs/blobs dirs per
    deleted model, so `du` on the cache still lists repos that hold nothing.
    """
    for repo_id in {m.repo_id, m.draft_repo_id}:
        if not repo_id:
            continue
        d = repo_dir(repo_id)
        blobs = d / "blobs"
        if not d.is_dir() or not blobs.is_dir():
            continue
        if any(blobs.iterdir()):
            continue  # another quant from the same repo is still cached
        with contextlib.suppress(OSError):
            shutil.rmtree(d)


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
                # suppress(Exception) here only guards the progress *callback*
                # (e.g. a UI update racing a widget teardown) — it must never
                # swallow the download itself. The actual download error is
                # captured in result["error"] by _worker above and re-raised
                # below, so callers still see real failures.
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


def download_gguf_parts_with_progress(
    repo_id: str,
    parts: tuple[tuple[str, int], ...],
    progress_cb: ProgressCallback,
    cache_dir: Path | None = None,
) -> Path:
    """Download every shard of a multi-part GGUF, reporting cumulative progress.

    ``parts`` is ``GgufFile.parts`` — (filename, size_bytes) per shard, ordered
    shard 1..N. Downloads sequentially (shard order) via
    :func:`download_gguf_with_progress`, translating each shard's own 0..size
    progress into its slice of the overall 0..total_bytes range. Returns shard
    1's path: llama-server auto-discovers the sibling shards from that path
    (they land in the same HF-cache blob layout), so the registry only ever
    needs to store shard 1's local_path.
    """
    if not parts:
        raise ValueError("download_gguf_parts_with_progress: no parts given")
    total_bytes = sum(sz for _, sz in parts) or 1
    shard1_path: Path | None = None
    done_before = 0
    for filename, size in parts:
        # Bind `done_before`'s current value into the default arg — a plain
        # closure over the loop variable would have every shard's callback
        # see the FINAL value of done_before once the loop finishes.
        def _shard_progress(done: int, _total: int, base: int = done_before) -> None:
            progress_cb(base + done, total_bytes)

        path = download_gguf_with_progress(
            repo_id=repo_id,
            filename=filename,
            expected_bytes=max(size, 1),
            progress_cb=_shard_progress,
            cache_dir=cache_dir,
        )
        if shard1_path is None:
            shard1_path = path
        done_before += size
    progress_cb(total_bytes, total_bytes)
    return shard1_path  # type: ignore[return-value]


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
