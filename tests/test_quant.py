from inferhost.core.quant import QUANT_RANK, extract_quant, pick_best


def test_extract_quant_basic():
    assert extract_quant("model.Q4_K_M.gguf") == "Q4_K_M"
    assert extract_quant("Qwen2.5-7B-Instruct-Q5_K_M.gguf") == "Q5_K_M"
    assert extract_quant("model-IQ4_XS.gguf") == "IQ4_XS"
    assert extract_quant("model.gguf") is None


def test_extract_quant_case_insensitive():
    assert extract_quant("model-q4_k_m.gguf") == "Q4_K_M"


def test_quant_priority_ordering():
    assert QUANT_RANK["Q8_0"] < QUANT_RANK["Q4_K_M"]
    assert QUANT_RANK["Q4_K_M"] < QUANT_RANK["IQ3_XS"]


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
