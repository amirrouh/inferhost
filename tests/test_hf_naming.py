from types import SimpleNamespace

from inferhost.core import hf
from inferhost.core.hf import normalize_name


class _FakeApi:
    def __init__(self, filenames):
        self._sibs = [SimpleNamespace(rfilename=f, size=s) for f, s in filenames]

    def repo_info(self, repo_id, files_metadata=True):  # noqa: ARG002
        return SimpleNamespace(siblings=self._sibs)


def _patch_repo(monkeypatch, filenames):
    monkeypatch.setattr(hf, "_api", lambda: _FakeApi(filenames))


def test_find_vocoder_detects_wavtokenizer(monkeypatch):
    _patch_repo(monkeypatch, [
        ("OuteTTS-1.0-0.6B-Q8_0.gguf", 600),
        ("WavTokenizer-Large-75-F16.gguf", 80),
    ])
    assert hf.find_vocoder("oute/repo") == "WavTokenizer-Large-75-F16.gguf"


def test_find_vocoder_detects_generic_vocoder_name(monkeypatch):
    _patch_repo(monkeypatch, [
        ("model.gguf", 600),
        ("some-vocoder-f16.gguf", 50),
    ])
    assert hf.find_vocoder("x/y") == "some-vocoder-f16.gguf"


def test_find_vocoder_none_for_plain_chat_repo(monkeypatch):
    # A normal chat repo (no vocoder) — and the main model must never be picked.
    _patch_repo(monkeypatch, [
        ("Qwen2.5-7B-Instruct-Q4_K_M.gguf", 4400),
        ("mmproj-qwen-F16.gguf", 600),
    ])
    assert hf.find_vocoder("Qwen/x") is None


def test_find_sd_aux_detects_flux_companions(monkeypatch):
    _patch_repo(monkeypatch, [
        ("flux1-dev-Q4_K_S.gguf", 6000),
        ("ae.safetensors", 300),
        ("clip_l.safetensors", 240),
        ("t5xxl_fp16.safetensors", 9000),
    ])
    aux = hf.find_sd_aux("black-forest-labs/flux-gguf")
    assert aux["vae_path"] == "ae.safetensors"
    assert aux["clip_l_path"] == "clip_l.safetensors"
    assert aux["t5xxl_path"] == "t5xxl_fp16.safetensors"
    assert "clip_g_path" not in aux  # SDXL-only, not present here


def test_find_sd_aux_detects_qwen_text_encoder_conservatively(monkeypatch):
    # An obvious qwen *vl* encoder is detected; a plain Qwen chat GGUF is NOT
    # (avoids grabbing a chat model as a text encoder).
    _patch_repo(monkeypatch, [
        ("qwen_image-Q4_K.gguf", 12000),
        ("qwen2.5-vl-7b-Q4_K_M.gguf", 4000),
    ])
    aux = hf.find_sd_aux("x/qwen-image")
    assert aux.get("text_encoder_path") == "qwen2.5-vl-7b-Q4_K_M.gguf"

    _patch_repo(monkeypatch, [
        ("z_image_turbo-Q4_K.gguf", 4000),
        ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2500),  # plain chat GGUF, no vl/encoder token
    ])
    # Plain Qwen chat GGUF must NOT be mistaken for an encoder (cross-repo only).
    assert "text_encoder_path" not in hf.find_sd_aux("x/z")


def test_list_repo_files_includes_companions(monkeypatch):
    # Unlike list_image_files, the component picker lister returns VAE/encoders.
    _patch_repo(monkeypatch, [
        ("ae.safetensors", 300),
        ("t5xxl_fp16.safetensors", 9000),
    ])
    names = {f.filename for f in hf.list_repo_files("x/y")}
    assert names == {"ae.safetensors", "t5xxl_fp16.safetensors"}


def test_find_sd_aux_empty_for_single_file_repo(monkeypatch):
    _patch_repo(monkeypatch, [("sd_xl_base_1.0.safetensors", 6900)])
    assert hf.find_sd_aux("stabilityai/sdxl") == {}


def test_list_image_files_includes_safetensors_excludes_companions(monkeypatch):
    _patch_repo(monkeypatch, [
        ("flux1-dev-Q4_K_S.gguf", 6000),
        ("sd_xl.safetensors", 6900),
        ("ae.safetensors", 300),          # companion -> excluded from main picker
        ("t5xxl_fp16.safetensors", 9000), # companion -> excluded
        ("README.md", 1),                  # non-model -> excluded
    ])
    names = {f.filename for f in hf.list_image_files("x/y")}
    assert names == {"flux1-dev-Q4_K_S.gguf", "sd_xl.safetensors"}


def test_normalize_name_strips_org():
    assert normalize_name("Qwen/Qwen2.5-7B-Instruct-GGUF") == "qwen2.5-7b-instruct"


def test_normalize_name_strips_lowercase():
    assert normalize_name("meta-llama/Llama-3.2-3B-Instruct") == "llama-3.2-3b-instruct"


def test_normalize_name_handles_no_org():
    assert normalize_name("solo-model") == "solo-model"
