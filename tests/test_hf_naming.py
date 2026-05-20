from inferhost.core.hf import normalize_name


def test_normalize_name_strips_org():
    assert normalize_name("Qwen/Qwen2.5-7B-Instruct-GGUF") == "qwen2.5-7b-instruct"


def test_normalize_name_strips_lowercase():
    assert normalize_name("meta-llama/Llama-3.2-3B-Instruct") == "llama-3.2-3b-instruct"


def test_normalize_name_handles_no_org():
    assert normalize_name("solo-model") == "solo-model"
