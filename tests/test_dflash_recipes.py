"""Tests for the built-in DFlash target->draft pairing matcher."""
from inferhost.core.dflash_recipes import PAIRINGS, match_pairing, suggest_gguf_repo


def test_matches_qwen_and_gemma_families():
    assert match_pairing("Qwen/Qwen3.6-27B").key == "qwen3.6-27b"
    assert match_pairing("google/gemma-4-12B-it").key == "gemma-4-12b"
    assert match_pairing("google/gemma-4-31B").key == "gemma-4-31b"
    assert match_pairing("Qwen/Qwen3.5-9B-Instruct").key == "qwen3.5-9b"


def test_moe_pairings_carry_smaller_speedup_note():
    for key in ("qwen3-coder-30b-a3b", "qwen3.6-35b-a3b", "gemma-4-26b-a4b"):
        p = next(pp for pp in PAIRINGS if pp.key == key)
        assert "smaller speedup" in p.note.lower()


def test_specificity_moe_beats_dense_and_coder_beats_base():
    # An MoE variant repo must match its A3B/A4B pairing, not a shorter dense one.
    assert match_pairing("Qwen/Qwen3.6-35B-A3B-Instruct").key == "qwen3.6-35b-a3b"
    assert match_pairing("google/gemma-4-26B-A4B").key == "gemma-4-26b-a4b"
    # Qwen3-Coder must beat any bare qwen3 substring.
    assert match_pairing("Qwen/Qwen3-Coder-30B-A3B-Instruct").key == "qwen3-coder-30b-a3b"


def test_dense_27b_not_swallowed_by_moe():
    # The dense Qwen3.6-27B must NOT match the 35B-A3B MoE pairing.
    assert match_pairing("Qwen/Qwen3.6-27B").key == "qwen3.6-27b"


def test_unknown_and_empty_return_none():
    assert match_pairing("meta-llama/Llama-3.1-8B") is None
    assert match_pairing("") is None
    assert match_pairing("Qwen/Qwen2.5-7B-Instruct-GGUF") is None


def test_all_pairings_have_a_draft_repo():
    for p in PAIRINGS:
        assert "/" in p.draft_repo, f"{p.key} draft_repo isn't an owner/name"
        assert p.target_patterns, f"{p.key} has no target patterns"
        assert p.label


def test_suggest_gguf_repo_redirects_raw_safetensors_draft_to_gguf():
    # The official z-lab draft is raw safetensors (vLLM/SGLang) — no GGUFs —
    # so pasting it should redirect to the community GGUF conversion.
    assert (
        suggest_gguf_repo("z-lab/Qwen3.5-27B-DFlash")
        == "AtomicChat/Qwen3.5-27B-DFlash-GGUF"
    )


def test_suggest_gguf_repo_none_when_already_the_gguf_repo():
    # Pasting the GGUF repo itself (any casing) — nowhere new to redirect to.
    assert suggest_gguf_repo("AtomicChat/Qwen3.5-27B-DFlash-GGUF") is None
    assert suggest_gguf_repo("atomicchat/qwen3.5-27b-dflash-gguf") is None


def test_suggest_gguf_repo_none_for_unknown_or_empty():
    assert suggest_gguf_repo("meta-llama/Llama-3.1-8B") is None
    assert suggest_gguf_repo("") is None
