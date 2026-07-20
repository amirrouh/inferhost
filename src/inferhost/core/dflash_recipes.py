"""Built-in target->draft pairings for DFlash speculative decoding.

DFlash accelerates a large "target" model by attaching a small z-lab
block-diffusion "draft" model that proposes several tokens per step, which the
target then verifies in one pass (`--model-draft ... --spec-type draft-dflash`).
The draft has to be trained specifically for its target — you can't pair an
arbitrary small model — so knowing *which* published draft GGUF goes with a
given target is exactly the kind of thing a user shouldn't have to memorize.

A pairing maps a recognizable target family (matched by repo-id substring) to
the community-published draft repo that serves it. The dashboard's Suggest /
`f`-key flow uses this to fetch the right draft automatically; the manual
Browse picker (paste any draft repo URL) stays as the always-works fallback for
anything unmatched or newly released.

Pure data — adding a newly published family is a one-tuple edit. The draft
repos are community GGUF conversions and can move/rename upstream; the Browse
picker is the escape hatch when a pairing goes stale.

Note on MoE targets: a Mixture-of-Experts target (…-A3B / …-A4B, only a few
billion params active per token) is already cheap to run per step, so
speculative decoding buys a smaller wall-clock speedup than on a dense target
of similar total size. Those entries carry a note saying so.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFlashPairing:
    key: str
    label: str
    # Substrings matched (case-insensitive) against the target model's repo id.
    target_patterns: tuple[str, ...]
    # HF repo id of the community DFlash draft GGUF for this target family.
    draft_repo: str
    # Optional advisory shown in the UI (e.g. the MoE "smaller speedup" caveat).
    note: str = ""


# Order matters: most-specific families first, so a MoE / coder variant wins
# over a shorter dense-family substring (e.g. "qwen3-coder-30b-a3b" must beat a
# bare "qwen3", "35b-a3b" must beat "27b"). match_pairing returns the first hit.
PAIRINGS: tuple[DFlashPairing, ...] = (
    DFlashPairing(
        key="qwen3-coder-30b-a3b",
        label="Qwen3-Coder-30B-A3B",
        target_patterns=("qwen3-coder-30b-a3b", "qwen3-coder-30b", "qwen3coder-30b-a3b"),
        draft_repo="AtomicChat/Qwen3-Coder-30B-A3B-DFlash-GGUF",
        note="MoE target (A3B active) — DFlash gives a smaller speedup than on a dense model.",
    ),
    DFlashPairing(
        key="qwen3.6-35b-a3b",
        label="Qwen3.6-35B-A3B",
        target_patterns=("qwen3.6-35b-a3b", "qwen3-6-35b-a3b", "qwen3.6-35b"),
        draft_repo="Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp",
        note="MoE target (A3B active) — DFlash gives a smaller speedup than on a dense model.",
    ),
    DFlashPairing(
        key="gemma-4-26b-a4b",
        label="Gemma-4-26B-A4B",
        target_patterns=("gemma-4-26b-a4b", "gemma4-26b-a4b", "gemma-4-26b"),
        draft_repo="Alittlehammmer/gemma-4-26B-A4B-it-DFlash-GGUF-llama.cpp",
        note="MoE target (A4B active) — DFlash gives a smaller speedup than on a dense model.",
    ),
    DFlashPairing(
        key="qwen3.6-27b",
        label="Qwen3.6-27B",
        target_patterns=("qwen3.6-27b", "qwen3-6-27b"),
        draft_repo="Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp",
    ),
    DFlashPairing(
        key="gemma-4-31b",
        label="Gemma-4-31B",
        target_patterns=("gemma-4-31b", "gemma4-31b"),
        draft_repo="Alittlehammmer/gemma-4-31B-it-DFlash-GGUF-llama.cpp",
    ),
    DFlashPairing(
        key="gemma-4-12b",
        label="Gemma-4-12B",
        target_patterns=("gemma-4-12b", "gemma4-12b"),
        draft_repo="williamliao/gemma-4-12B-it-DFlash-GGUF",
    ),
    DFlashPairing(
        key="qwen3.5-27b",
        label="Qwen3.5-27B",
        target_patterns=("qwen3.5-27b", "qwen3-5-27b"),
        draft_repo="AtomicChat/Qwen3.5-27B-DFlash-GGUF",
    ),
    DFlashPairing(
        key="qwen3.5-9b",
        label="Qwen3.5-9B",
        target_patterns=("qwen3.5-9b", "qwen3-5-9b"),
        draft_repo="Anbeeld/Qwen3.5-9B-DFlash-GGUF",
    ),
)


def match_pairing(repo_id: str) -> DFlashPairing | None:
    """Return the DFlash pairing whose target family matches ``repo_id``, else None.

    Checked in PAIRINGS order (most-specific first), by case-insensitive
    repo-id substring so a browser-pasted URL or a bare ``owner/name`` both work.
    """
    rid = (repo_id or "").lower()
    if not rid:
        return None
    for pairing in PAIRINGS:
        if any(p in rid for p in pairing.target_patterns):
            return pairing
    return None


def suggest_gguf_repo(repo_id: str) -> str | None:
    """Reverse lookup: given a *draft* repo the user pasted, suggest its paired GGUF repo.

    The official z-lab drafts (e.g. ``z-lab/Qwen3.5-27B-DFlash``) ship as raw
    ``model.safetensors`` for vLLM/SGLang — no GGUFs — while the community
    conversion llama.cpp actually needs lives in a separate, differently-named
    repo (e.g. ``AtomicChat/Qwen3.5-27B-DFlash-GGUF``). Someone who pastes the
    z-lab repo into the picker gets an empty file list with no clue why, so this
    redirects them to the known GGUF conversion when we recognize the family.

    Reuses :func:`match_pairing`: the family substrings (e.g. "qwen3.5-27b")
    that identify a *target* also appear in the matching *draft* repo's name,
    so matching the pasted draft repo against the same table works unchanged.
    Returns ``pairing.draft_repo`` only if it differs (case-insensitively) from
    the repo the user pasted — i.e. only when there's actually somewhere new to
    redirect to — else None.
    """
    pairing = match_pairing(repo_id)
    if pairing is None:
        return None
    if pairing.draft_repo.lower() == (repo_id or "").strip().lower():
        return None
    return pairing.draft_repo
