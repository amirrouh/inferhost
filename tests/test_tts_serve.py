"""Tests for the inferhost-tts daemon, with llama-tts stubbed out.

These cover the request/synthesis logic without a GPU or the real binary: the
subprocess call is monkeypatched to write a fake WAV, so we exercise argument
construction, model lookup, and the OpenAI-shape error handling.
"""
import json
from pathlib import Path

import pytest

from inferhost import tts_serve
from inferhost.core import paths, registry
from inferhost.core.registry import Model, Registry
from inferhost.settings import reload_settings

FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-bytes"


@pytest.fixture
def tts_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(tmp_path / "data"))
    reload_settings()
    paths.ensure_dirs()
    # Pretend llama-tts is installed.
    tts_bin = paths.llama_tts_path()
    tts_bin.write_bytes(b"#!/bin/true\n")
    # Register one TTS model + one plain chat model.
    reg = Registry()
    reg.add(Model(name="oute", repo_id="a/b", filename="oute.gguf",
                  local_path="/tmp/oute.gguf", vocoder_path="/tmp/wavtok.gguf"))
    reg.add(Model(name="qwen", repo_id="x/y", filename="qwen.gguf",
                  local_path="/tmp/qwen.gguf"))
    registry.save(reg)
    return tmp_path


def _stub_llama_tts(monkeypatch, captured):
    """Replace subprocess.run so it records argv and writes a fake WAV to -o."""
    class _Result:
        returncode = 0
        stderr = b""
        stdout = b""

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ARG001
        captured["cmd"] = cmd
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(FAKE_WAV)
        return _Result()

    monkeypatch.setattr(tts_serve.subprocess, "run", fake_run)


def test_synthesize_builds_correct_llama_tts_command(tts_env, monkeypatch):
    captured: dict = {}
    _stub_llama_tts(monkeypatch, captured)

    model = registry.load().get("oute")
    audio = tts_serve.synthesize(model, "hello world")

    assert audio == FAKE_WAV
    cmd = captured["cmd"]
    assert str(paths.llama_tts_path()) == cmd[0]
    assert "-m" in cmd and "/tmp/oute.gguf" in cmd
    assert "-mv" in cmd and "/tmp/wavtok.gguf" in cmd
    assert "-p" in cmd and "hello world" in cmd
    assert "--tts-use-guide-tokens" in cmd


def test_synthesize_raises_with_stderr_on_failure(tts_env, monkeypatch):
    class _Result:
        returncode = 1
        stderr = b"some llama-tts error\nGGML_ASSERT failed"
        stdout = b""

    monkeypatch.setattr(tts_serve.subprocess, "run", lambda *a, **k: _Result())
    model = registry.load().get("oute")
    with pytest.raises(RuntimeError, match="GGML_ASSERT"):
        tts_serve.synthesize(model, "hi")


def test_tts_models_lists_only_tts(tts_env):
    names = set(tts_serve._tts_models())
    assert names == {"oute"}  # the plain chat model 'qwen' is excluded


# ---- HTTP handler tests (without binding a socket) ----

class _FakeWFile:
    def __init__(self):
        self.data = b""

    def write(self, b):  # noqa: ANN001
        self.data += b


def _make_handler(method, path, body=None):
    """Construct a _Handler without running BaseHTTPRequestHandler.__init__."""
    h = tts_serve._Handler.__new__(tts_serve._Handler)
    raw = json.dumps(body).encode() if body is not None else b""
    h.command = method
    h.path = path
    h.headers = {"Content-Length": str(len(raw))}
    import io
    h.rfile = io.BytesIO(raw)
    h.wfile = _FakeWFile()
    h.requestline = f"{method} {path}"
    h.request_version = "HTTP/1.1"
    h.client_address = ("127.0.0.1", 0)
    h._status = None
    h._headers_sent = {}

    def send_response(code, message=None):  # noqa: ANN001, ARG001
        h._status = code

    def send_header(k, v):  # noqa: ANN001
        h._headers_sent[k] = v

    def end_headers():
        pass

    h.send_response = send_response
    h.send_header = send_header
    h.end_headers = end_headers
    return h


def test_handler_rejects_unknown_model(tts_env):
    h = _make_handler("POST", "/v1/audio/speech", {"model": "nope", "input": "hi"})
    h.do_POST()
    assert h._status == 404
    assert "unknown TTS model" in h.wfile.data.decode()


def test_handler_rejects_missing_input(tts_env):
    h = _make_handler("POST", "/v1/audio/speech", {"model": "oute"})
    h.do_POST()
    assert h._status == 400
    assert "input" in h.wfile.data.decode()


def test_handler_synthesizes_wav(tts_env, monkeypatch):
    captured: dict = {}
    _stub_llama_tts(monkeypatch, captured)
    h = _make_handler("POST", "/v1/audio/speech", {"model": "oute", "input": "hello"})
    h.do_POST()
    assert h._status == 200
    assert h._headers_sent["Content-Type"] == "audio/wav"
    assert h.wfile.data == FAKE_WAV


def test_handler_health(tts_env):
    h = _make_handler("GET", "/health")
    h.do_GET()
    assert h._status == 200


# ---- Kokoro engine helpers (no model download needed) ----

class _FakeKokoro:
    def __init__(self, voices):
        self._voices = voices

    def get_voices(self):
        return list(self._voices)


def test_tts_engine_discriminates_all_three_families():
    from inferhost.core.configs import tts_engine

    kokoro = Model(name="k", repo_id="x/y", filename="onnx/model.onnx",
                   local_path="/models/model.onnx", vocoder_path="/models/v.npz")
    oute = Model(name="o", repo_id="x/y", filename="OuteTTS-Q8_0.gguf",
                 local_path="/models/OuteTTS-Q8_0.gguf", vocoder_path="/models/w.gguf")
    orpheus = Model(name="orp", repo_id="unsloth/orpheus-3b-0.1-ft-GGUF",
                    filename="orpheus-3b-0.1-ft-Q4_K_M.gguf",
                    local_path="/models/orpheus-3b-0.1-ft-Q4_K_M.gguf",
                    vocoder_path="/models/decoder_model.onnx")
    chat = Model(name="c", repo_id="x/y", filename="chat-Q4_K_M.gguf",
                 local_path="/models/chat-Q4_K_M.gguf")
    assert tts_engine(kokoro) == "kokoro"
    assert tts_engine(oute) == "outetts"
    assert tts_engine(orpheus) == "orpheus"
    assert tts_engine(chat) == ""


def test_resolve_kokoro_voice_exact_alias_default_fallback(tts_env):
    k = _FakeKokoro(["af_heart", "af_alloy", "am_michael"])
    # Exact Kokoro name wins.
    assert tts_serve._resolve_kokoro_voice(k, "am_michael") == "am_michael"
    # OpenAI preset maps to its Kokoro equivalent.
    assert tts_serve._resolve_kokoro_voice(k, "alloy") == "af_alloy"
    # Unknown / missing falls back to the default voice, never errors.
    assert tts_serve._resolve_kokoro_voice(k, "not-a-voice") == "af_heart"
    assert tts_serve._resolve_kokoro_voice(k, None) == "af_heart"


def test_resolve_kokoro_voice_bundle_without_default(tts_env):
    # Default voice absent from the bundle -> first available, deterministic.
    k = _FakeKokoro(["jf_alpha", "am_adam"])
    assert tts_serve._resolve_kokoro_voice(k, None) == "am_adam"


def test_input_casting_session_casts_to_graph_dtypes():
    np = __import__("numpy")

    class _FakeSess:
        def get_inputs(self):
            from types import SimpleNamespace as NS
            return [NS(name="input_ids", type="tensor(int64)"),
                    NS(name="style", type="tensor(float)"),
                    NS(name="speed", type="tensor(float)")]

        def run(self, output_names, inputs):  # noqa: ARG002
            return inputs

    sess = tts_serve._InputCastingSession(_FakeSess())
    # kokoro-onnx 0.5.0 sends speed as int32 and tokens as a plain list.
    out = sess.run(None, {
        "input_ids": [[0, 1, 2, 0]],
        "style": np.zeros((1, 256), dtype=np.float32),
        "speed": np.array([1], dtype=np.int32),
    })
    assert out["input_ids"].dtype == np.int64
    assert out["style"].dtype == np.float32
    assert out["speed"].dtype == np.float32  # the upstream-bug case


def test_to_wav_bytes_produces_valid_16bit_mono_wav():
    import io
    import wave

    np = __import__("numpy")
    samples = np.sin(np.linspace(0, 100, 24000, dtype=np.float32))
    data = tts_serve._to_wav_bytes(samples, 24000)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000
        assert w.getnframes() == 24000


# ---- Orpheus engine helpers (no model download needed) ----

def test_orpheus_snac_codes_parsing_and_layer_layout():
    """Token stream -> 3 SNAC layers: each accepted token's code is
    N - 10 - (i % 7) * 4096; non-audio specials are skipped WITHOUT advancing
    the frame position; a trailing partial frame is dropped."""
    codes = [5, 100, 200, 300, 400, 500, 600,   # frame 0, positions 0..6
             7, 101, 201, 301, 401, 501, 601]   # frame 1
    parts = ["<custom_token_4>"]  # start marker: 4-10 < 0 -> skipped
    for i, c in enumerate(codes):
        parts.append(f"<custom_token_{c + 10 + (i % 7) * 4096}>")
    parts.append("<custom_token_9>")   # end marker: 9-10 < 0 -> skipped
    parts.append("<custom_token_52>")  # lone frame-position-0 token: partial frame
    generated = "prefix noise " + "".join(parts)

    c0, c1, c2 = tts_serve.orpheus_snac_codes(generated)
    assert c0 == [5, 7]                                      # position 0
    assert c1 == [100, 400, 101, 401]                        # positions 1, 4
    assert c2 == [200, 300, 500, 600, 201, 301, 501, 601]    # positions 2,3,5,6


def test_orpheus_snac_codes_empty_for_plain_text():
    assert tts_serve.orpheus_snac_codes("hello, no audio tokens here") == ([], [], [])


def test_resolve_orpheus_voice_exact_alias_default_fallback(tts_env):
    # Exact Orpheus voice wins.
    assert tts_serve._resolve_orpheus_voice("leo") == "leo"
    # OpenAI preset maps to its Orpheus equivalent.
    assert tts_serve._resolve_orpheus_voice("alloy") == "tara"
    assert tts_serve._resolve_orpheus_voice("shimmer") == "zoe"
    # Unknown / missing falls back to the default voice, never errors.
    assert tts_serve._resolve_orpheus_voice("not-a-voice") == "tara"
    assert tts_serve._resolve_orpheus_voice(None) == "tara"
