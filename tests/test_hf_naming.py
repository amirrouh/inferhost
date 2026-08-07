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


# ---- A4: multi-part GGUF grouping ----

def test_list_ggufs_groups_complete_multipart_run(monkeypatch):
    _patch_repo(monkeypatch, [
        ("model-Q4_K_M-00001-of-00003.gguf", 1000),
        ("model-Q4_K_M-00002-of-00003.gguf", 1000),
        ("model-Q4_K_M-00003-of-00003.gguf", 500),
    ])
    files = hf.list_ggufs("x/y")
    assert len(files) == 1
    f = files[0]
    assert f.filename == "model-Q4_K_M-00001-of-00003.gguf"  # shard 1
    assert f.size_bytes == 2500  # summed
    assert f.quant == "Q4_K_M"
    assert f.parts == (
        ("model-Q4_K_M-00001-of-00003.gguf", 1000),
        ("model-Q4_K_M-00002-of-00003.gguf", 1000),
        ("model-Q4_K_M-00003-of-00003.gguf", 500),
    )


def test_list_ggufs_drops_incomplete_multipart_run(monkeypatch):
    # Shard 2 of 3 is missing (partial upload/mirror) — must not be offered.
    _patch_repo(monkeypatch, [
        ("model-Q4_K_M-00001-of-00003.gguf", 1000),
        ("model-Q4_K_M-00003-of-00003.gguf", 500),
    ])
    assert hf.list_ggufs("x/y") == []


def test_list_ggufs_mixed_single_and_multipart(monkeypatch):
    _patch_repo(monkeypatch, [
        ("small-model-Q8_0.gguf", 4000),
        ("big-model-Q4_K_M-00001-of-00002.gguf", 2000),
        ("big-model-Q4_K_M-00002-of-00002.gguf", 2000),
    ])
    files = hf.list_ggufs("x/y")
    names = {f.filename for f in files}
    assert names == {"small-model-Q8_0.gguf", "big-model-Q4_K_M-00001-of-00002.gguf"}
    by_name = {f.filename: f for f in files}
    assert by_name["small-model-Q8_0.gguf"].parts == ()
    assert len(by_name["big-model-Q4_K_M-00001-of-00002.gguf"].parts) == 2


def test_download_gguf_parts_with_progress_reports_cumulative(monkeypatch):
    parts = (("a-00001-of-00002.gguf", 100), ("a-00002-of-00002.gguf", 200))
    calls: list[tuple[int, int]] = []
    downloaded: list[str] = []

    def fake_download_with_progress(repo_id, filename, expected_bytes, progress_cb, cache_dir=None, poll_interval=0.3):  # noqa: ARG001
        # Simulate the shard completing fully — one progress tick at full size.
        progress_cb(expected_bytes, expected_bytes)
        downloaded.append(filename)
        return f"/fake/{filename}"

    monkeypatch.setattr(hf, "download_gguf_with_progress", fake_download_with_progress)

    result = hf.download_gguf_parts_with_progress(
        repo_id="x/y", parts=parts, progress_cb=lambda d, t: calls.append((d, t)),
    )
    assert downloaded == ["a-00001-of-00002.gguf", "a-00002-of-00002.gguf"]
    assert str(result) == "/fake/a-00001-of-00002.gguf"  # shard 1's path
    total = 300
    # First shard completes at 100/300, second at 300/300 — cumulative, not
    # each shard resetting back to its own 0..size range.
    assert (100, total) in calls
    assert (300, total) in calls
    assert calls[-1] == (total, total)  # final "done" tick from the function itself


def test_download_gguf_parts_with_progress_rejects_empty_parts():
    import pytest

    with pytest.raises(ValueError):
        hf.download_gguf_parts_with_progress(repo_id="x/y", parts=(), progress_cb=lambda d, t: None)


# ---- A3: repo_file_size ----

def test_repo_file_size_returns_matching_size(monkeypatch):
    _patch_repo(monkeypatch, [
        ("model.gguf", 4000),
        ("mmproj-model-f16.gguf", 600),
    ])
    assert hf.repo_file_size("x/y", "mmproj-model-f16.gguf") == 600


def test_repo_file_size_missing_file_returns_zero(monkeypatch):
    _patch_repo(monkeypatch, [("model.gguf", 4000)])
    assert hf.repo_file_size("x/y", "does-not-exist.gguf") == 0


# ---- A7: list_tts_files (OuteTTS GGUF + Kokoro ONNX) ----

def test_list_tts_files_excludes_vocoder(monkeypatch):
    _patch_repo(monkeypatch, [
        ("OuteTTS-1.0-0.6B-Q8_0.gguf", 600),
        ("WavTokenizer-Large-75-F16.gguf", 80),
    ])
    names = {f.filename for f in hf.list_tts_files("oute/repo")}
    assert names == {"OuteTTS-1.0-0.6B-Q8_0.gguf"}


def test_list_tts_files_kokoro_onnx_variants(monkeypatch):
    _patch_repo(monkeypatch, [
        ("config.json", 1_000),
        ("onnx/model.onnx", 325_000_000),
        ("onnx/model_fp16.onnx", 163_000_000),
        ("onnx/model_quantized.onnx", 92_000_000),
        ("voices/af_heart.bin", 522_240),
    ])
    files = hf.list_tts_files("onnx-community/Kokoro-82M-v1.0-ONNX")
    quants = {f.filename: f.quant for f in files}
    assert quants == {
        "onnx/model.onnx": "F32",
        "onnx/model_fp16.onnx": "F16",
        "onnx/model_quantized.onnx": "Q8_0",
    }


def test_resolve_tts_repo_aliases_kokoro_pytorch_repo_to_onnx():
    assert hf.resolve_tts_repo("hexgrad/Kokoro-82M") == hf.KOKORO_ONNX_REPO
    assert hf.resolve_tts_repo("OuteAI/OuteTTS-0.2-500M-GGUF") == "OuteAI/OuteTTS-0.2-500M-GGUF"


def test_list_kokoro_voice_files_returns_sorted_voice_bins(monkeypatch):
    _patch_repo(monkeypatch, [
        ("onnx/model.onnx", 325_000_000),
        ("voices/am_michael.bin", 522_240),
        ("voices/af_heart.bin", 522_240),
        ("README.md", 100),
    ])
    assert hf.list_kokoro_voice_files("x/kokoro") == [
        ("voices/af_heart.bin", 522_240),
        ("voices/am_michael.bin", 522_240),
    ]


def test_build_kokoro_voices_npz_bundles_named_voices(tmp_path):
    np = __import__("numpy")
    a = (np.arange(510 * 256, dtype=np.float32)).reshape(-1)
    (tmp_path / "af_heart.bin").write_bytes(a.tobytes())
    (tmp_path / "am_michael.bin").write_bytes((a * 2).tobytes())
    out = hf.build_kokoro_voices_npz(
        {"af_heart": tmp_path / "af_heart.bin", "am_michael": tmp_path / "am_michael.bin"},
        tmp_path / "voices.npz",
    )
    loaded = np.load(out)
    assert set(loaded.keys()) == {"af_heart", "am_michael"}
    assert loaded["af_heart"].shape == (510, 1, 256)


def test_is_companion_file_matches_mmproj_and_dspark():
    assert hf.is_companion_file("Ternary-Bonsai-27B-mmproj-BF16.gguf")
    assert hf.is_companion_file("mmproj-Qwen3VL-8B-Instruct-F16.gguf")
    assert hf.is_companion_file("Ternary-Bonsai-27B-dspark-Q4_1.gguf")
    assert hf.is_companion_file("Bonsai-27B-dspark-bf16.gguf")


def test_is_companion_file_leaves_main_and_dflash_files_alone():
    assert not hf.is_companion_file("Ternary-Bonsai-27B-Q2_g64.gguf")
    assert not hf.is_companion_file("Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    # DFlash drafts live in dedicated repos where the draft IS the main pick.
    assert not hf.is_companion_file("Qwen3.6-35B-A3B-DFlash-BF16.gguf")


def test_is_orpheus_repo_detects_family_by_name():
    assert hf.is_orpheus_repo("unsloth/orpheus-3b-0.1-ft-GGUF")
    assert hf.is_orpheus_repo("lex-au/Orpheus-3b-FT-Q8_0.gguf")
    assert hf.is_orpheus_repo("isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF")
    assert not hf.is_orpheus_repo("OuteAI/OuteTTS-0.2-500M-GGUF")
    assert not hf.is_orpheus_repo("onnx-community/Kokoro-82M-v1.0-ONNX")
