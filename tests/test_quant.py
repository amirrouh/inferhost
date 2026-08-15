from inferhost.core.quant import QUANT_RANK, extract_quant, pick_best


def test_extract_quant_basic():
    assert extract_quant("model.Q4_K_M.gguf") == "Q4_K_M"
    assert extract_quant("Qwen2.5-7B-Instruct-Q5_K_M.gguf") == "Q5_K_M"
    assert extract_quant("model-IQ4_XS.gguf") == "IQ4_XS"
    assert extract_quant("model-Q4_1.gguf") == "Q4_1"
    assert extract_quant("model-Q5_1.gguf") == "Q5_1"
    assert extract_quant("model.gguf") is None


def test_extract_quant_case_insensitive():
    assert extract_quant("model-q4_k_m.gguf") == "Q4_K_M"


def test_quant_priority_ordering():
    assert QUANT_RANK["Q8_0"] < QUANT_RANK["Q4_K_M"]
    assert QUANT_RANK["Q4_K_M"] < QUANT_RANK["IQ3_XS"]


def test_extract_quant_ternary_formats():
    # prism-ml Bonsai ternary / binary packings.
    assert extract_quant("Ternary-Bonsai-27B-Q2_g64.gguf") == "Q2_G64"
    assert extract_quant("Ternary-Bonsai-8B-Q2_0_g64.gguf") == "Q2_0_G64"
    assert extract_quant("Ternary-Bonsai-27B-Q2_0.gguf") == "Q2_0"
    assert extract_quant("Ternary-Bonsai-27B-PQ2_0.gguf") == "PQ2_0"
    assert extract_quant("Bonsai-27B-Q1_0.gguf") == "Q1_0"
    # BitNet ternary types from mainline llama.cpp.
    assert extract_quant("bitnet-b1.58-2B-TQ1_0.gguf") == "TQ1_0"
    assert extract_quant("bitnet-b1.58-2B-TQ2_0.gguf") == "TQ2_0"
    # F16 in the same repo must still parse as F16, not a ternary name.
    assert extract_quant("Ternary-Bonsai-27B-F16.gguf") == "F16"


def test_ternary_ranking_prefers_mainline_g64():
    # Ternary formats rank below every conventional quant, and the
    # mainline-runnable group-64 packing outranks the fork-only group-128.
    assert QUANT_RANK["Q2_K"] < QUANT_RANK["TQ2_0"]
    assert QUANT_RANK["Q2_G64"] < QUANT_RANK["Q2_0"]
    assert QUANT_RANK["Q2_0_G64"] < QUANT_RANK["Q2_0"]
    assert QUANT_RANK["Q2_0"] < QUANT_RANK["PQ2_0"]
    assert QUANT_RANK["PQ2_0"] < QUANT_RANK["Q1_0"]


class _F:
    def __init__(self, name, size_gib, quant):
        self.filename = name
        self.size_gib = size_gib
        self.size_bytes = int(size_gib * (1024**3))
        self.quant = quant

    @property
    def quant_rank(self):
        return QUANT_RANK.get(self.quant or "", 99)


def test_pick_best_fits():
    files = [
        _F("a.Q8_0.gguf", 7.6, "Q8_0"),
        _F("a.Q5_K_M.gguf", 4.8, "Q5_K_M"),
        _F("a.Q4_K_M.gguf", 4.1, "Q4_K_M"),
    ]
    # With 6 GiB available, only Q4_K_M fits (with 1.5 overhead → budget 4.5)
    pick = pick_best(files, 6.0)
    assert pick.quant == "Q4_K_M"


def test_pick_best_no_fit_returns_smallest():
    files = [
        _F("a.Q5_K_M.gguf", 4.8, "Q5_K_M"),
        _F("a.Q8_0.gguf", 7.6, "Q8_0"),
    ]
    pick = pick_best(files, 1.0)  # nothing fits
    assert pick.size_gib == 4.8


def test_pick_best_bonsai_repo_without_companions():
    """The prism-ml Ternary-Bonsai-27B layout: with companion files (mmproj,
    dspark) filtered out — as the add-model picker does — the recommendation
    lands on the mainline-runnable g64 file, not the fork-only Q2_0/PQ2_0."""
    from inferhost.core.hf import is_companion_file

    files = [
        _F("Ternary-Bonsai-27B-F16.gguf", 50.11, "F16"),
        _F("Ternary-Bonsai-27B-mmproj-BF16.gguf", 0.87, "BF16"),
        _F("Ternary-Bonsai-27B-dspark-bf16.gguf", 6.79, "BF16"),
        _F("Ternary-Bonsai-27B-mmproj-Q8_0.gguf", 0.59, "Q8_0"),
        _F("Ternary-Bonsai-27B-dspark-Q4_1.gguf", 1.81, None),
        _F("Ternary-Bonsai-27B-PQ2_0.gguf", 6.67, "PQ2_0"),
        _F("Ternary-Bonsai-27B-Q2_0.gguf", 6.67, "Q2_0"),
        _F("Ternary-Bonsai-27B-Q2_g64.gguf", 7.06, "Q2_G64"),
    ]
    main = [f for f in files if not is_companion_file(f.filename)]
    pick = pick_best(main, 24.0)  # 24 GiB card; F16 doesn't fit
    assert pick.filename == "Ternary-Bonsai-27B-Q2_g64.gguf"


def test_extract_quant_block_scaled_fp4():
    from inferhost.core.quant import RECENT_GGML_TYPES

    assert extract_quant("Qwen3.8-27B-NVFP4-MTP.gguf") == "NVFP4"
    assert extract_quant("gpt-oss-20b-mxfp4.gguf") == "MXFP4"
    # Both need a llama-server carrying the ggml type, so both must map to one.
    assert RECENT_GGML_TYPES["NVFP4"] == "nvfp4"
    assert RECENT_GGML_TYPES["MXFP4"] == "mxfp4"


def test_fp4_ranking_against_the_k_quants():
    # NVFP4's FP8 (E4M3) scale over a 16-weight block holds up better than
    # Q4_K_M; MXFP4's coarse E8M0 scale over 32 weights lands just under it.
    assert QUANT_RANK["Q5_0"] < QUANT_RANK["NVFP4"] < QUANT_RANK["Q4_K_M"]
    assert QUANT_RANK["Q4_K_M"] < QUANT_RANK["MXFP4"] < QUANT_RANK["Q4_K_S"]


def test_fp4_files_are_pickable_at_all():
    """The regression this guards: an unparsed quant falls to rank 99 and loses
    the recommendation star to every K-quant in the repo, whatever its size."""
    files = [
        _F("m-NVFP4.gguf", 15.0, extract_quant("m-NVFP4.gguf")),
        _F("m-Q3_K_M.gguf", 13.0, extract_quant("m-Q3_K_M.gguf")),
    ]
    assert pick_best(files, 32.0).filename == "m-NVFP4.gguf"
