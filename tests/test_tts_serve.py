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
