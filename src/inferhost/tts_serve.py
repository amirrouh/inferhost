"""inferhost-tts: a tiny OpenAI-compatible text-to-speech daemon.

OuteTTS (and other WavTokenizer-based TTS models) can only be synthesized by the
standalone ``llama-tts`` binary — ``llama-server`` has no endpoint for them. This
daemon wraps ``llama-tts`` behind ``POST /v1/audio/speech`` so the model is
reachable over the same OpenAI wire protocol as everything else in the stack.
LiteLLM routes the gateway's ``/v1/audio/speech`` here for any model registered
with ``mode: audio_speech``; the daemon is also reachable directly on its port.

``llama-tts`` is one-shot — it loads the model, synthesizes, and exits — so each
request pays the model's load cost. Synthesis is serialized with a lock so two
concurrent requests can't fight over VRAM.

Run with ``python -m inferhost.tts_serve``. No third-party dependencies (stdlib
``http.server`` only) so it stays a lightweight sidecar.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inferhost.core import paths, registry
from inferhost.settings import settings

# Only one synthesis at a time: llama-tts loads the full model into VRAM for
# each call, so overlapping runs would race for memory and likely OOM.
_SYNTH_LOCK = threading.Lock()

# Generous: a cold model load + synthesis of a long paragraph can take a while
# on a busy GPU. Better to wait than to truncate a legitimate request.
_SYNTH_TIMEOUT = 300.0


def _model_path(m: registry.Model) -> str:
    return m.local_path or str(paths.models_dir() / m.filename)


def _tts_models() -> dict[str, registry.Model]:
    """Name -> Model for every registered TTS model (those with a vocoder)."""
    return {m.name: m for m in registry.load().models if m.vocoder_path}


# ── Qwen3-TTS (qwen3-tts.cpp) engine ───────────────────────────────────────
# Qwen3-TTS's architecture (28-layer talker + 5-layer code predictor +
# WavTokenizer decoder) is NOT supported by mainline llama.cpp / llama-tts, so
# it's served by the standalone ``qwen3-tts-cli`` binary from the qwen3-tts.cpp
# fork, cloned and built alongside the other engines under
# ``<inferhost-data>/qwen3-tts.cpp``. The CLI takes a model *directory*
# containing ``qwen3-tts-0.6b-{f16,q8_0}.gguf`` + ``qwen3-tts-tokenizer-f16.gguf``
# and writes a 24 kHz mono WAV. A TTS model is routed here (instead of
# llama-tts) when its GGUF filename starts with ``qwen3-tts``.

def _qwen3_tts_root() -> Path:
    return paths.bin_dir().parent / "qwen3-tts.cpp"


def _qwen3_tts_cli() -> Path:
    return _qwen3_tts_root() / "build" / "qwen3-tts-cli"


def _qwen3_tts_libdir() -> Path:
    return _qwen3_tts_root() / "ggml" / "build" / "src"


def _is_qwen3_tts(model: registry.Model) -> bool:
    return Path(_model_path(model)).name.lower().startswith("qwen3-tts")


def _synthesize_qwen3_tts(
    model: registry.Model, text: str, speaker_file: str | None = None
) -> bytes:
    """Synthesize via qwen3-tts.cpp's ``qwen3-tts-cli`` and return WAV bytes."""
    cli = _qwen3_tts_cli()
    if not cli.exists():
        raise RuntimeError(
            f"qwen3-tts-cli not found at {cli}. Build the qwen3-tts.cpp engine "
            "(git clone + cmake) under the inferhost data dir."
        )
    libdir = _qwen3_tts_libdir()
    # qwen3-tts-cli auto-selects the GPU when libggml-cuda.so is reachable;
    # otherwise it falls back to CPU on its own.
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": os.pathsep.join(
            p for p in (
                str(libdir),
                str(libdir / "ggml-cuda"),
                os.environ.get("LD_LIBRARY_PATH", ""),
            ) if p
        ),
    }
    model_dir = str(Path(_model_path(model)).parent)
    with _SYNTH_LOCK, tempfile.TemporaryDirectory(prefix="inferhost-tts-") as tmp:
        out_path = Path(tmp) / "speech.wav"
        cmd = [str(cli), "-m", model_dir, "-t", text, "-o", str(out_path)]
        if speaker_file:
            cmd += ["-r", speaker_file]
        proc = subprocess.run(
            cmd, capture_output=True, env=env, timeout=_SYNTH_TIMEOUT, check=False,
        )
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            detail = " | ".join(tail[-5:]) if tail else "no stderr"
            raise RuntimeError(f"qwen3-tts-cli failed (exit {proc.returncode}): {detail}")
        return out_path.read_bytes()


def synthesize(model: registry.Model, text: str, speaker_file: str | None = None) -> bytes:
    """Run the model's TTS engine for ``text`` and return the WAV bytes.

    Dispatches to qwen3-tts.cpp for Qwen3-TTS models, else to llama-tts.
    Raises RuntimeError with the engine's stderr tail on failure.
    """
    if _is_qwen3_tts(model):
        return _synthesize_qwen3_tts(model, text, speaker_file=speaker_file)

    tts_bin = paths.llama_tts_path()
    if not tts_bin.exists():
        raise RuntimeError(
            "llama-tts binary not found. Run `inferhost start` (or reinstall) to "
            "fetch it from the llama.cpp release."
        )
    s = settings()
    # llama-tts links the shared .so set next to the binary.
    env = {**os.environ, "LD_LIBRARY_PATH": str(paths.bin_dir())}

    def _run(out_path: Path, gpu: bool) -> subprocess.CompletedProcess:
        cmd = [
            str(tts_bin),
            "-m", _model_path(model),
            "-mv", model.vocoder_path,
            "-p", text,
            "-o", str(out_path),
            # Guide tokens keep the model from dropping/repeating words — cheap
            # quality win that matters most for the short prompts TTS gets.
            "--tts-use-guide-tokens",
        ]
        if gpu:
            cmd += ["-ngl", str(s.gpu_layers)]
        else:
            # True CPU-only: on a Vulkan/CUDA build, -ngl 0 alone still tries to
            # allocate compute buffers on the GPU, so we must also drop the
            # device. This is the fallback when the GPU is out of memory (e.g. a
            # large chat model is resident) — the TTS model is small enough to
            # synthesize on CPU in a few seconds.
            cmd += ["-dev", "none", "-ngl", "0"]
        if speaker_file:
            cmd += ["--tts-speaker-file", speaker_file]
        return subprocess.run(
            cmd, capture_output=True, env=env, timeout=_SYNTH_TIMEOUT, check=False,
        )

    with _SYNTH_LOCK, tempfile.TemporaryDirectory(prefix="inferhost-tts-") as tmp:
        out_path = Path(tmp) / "speech.wav"
        proc = _run(out_path, gpu=True)
        if proc.returncode != 0 or not out_path.exists():
            stderr = (proc.stderr or b"").decode("utf-8", "replace")
            # Retry on CPU when the GPU couldn't fit the model — common when a
            # chat model already owns the VRAM. Other failures fall through.
            if "alloc" in stderr.lower() or "out of" in stderr.lower() or "vk::" in stderr:
                print("inferhost-tts: GPU allocation failed, retrying on CPU", flush=True)
                proc = _run(out_path, gpu=False)
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            detail = " | ".join(tail[-5:]) if tail else "no stderr"
            raise RuntimeError(f"llama-tts failed (exit {proc.returncode}): {detail}")
        return out_path.read_bytes()


class _Handler(BaseHTTPRequestHandler):
    # Silence the default per-request stderr logging; the daemon log captures
    # what we choose to print instead.
    def log_message(self, *args) -> None:  # noqa: D401, ANN002
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok"})
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            data = [{"id": name, "object": "model"} for name in _tts_models()]
            self._send_json(200, {"object": "list", "data": data})
            return
        self._error(404, f"not found: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/audio/speech", "/audio/speech"):
            self._error(404, f"not found: {self.path}")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            self._error(400, f"invalid JSON body: {e}")
            return

        model_name = body.get("model")
        text = body.get("input")
        response_format = (body.get("response_format") or "wav").lower()
        # OpenAI's `voice` is a named preset; llama-tts uses a speaker JSON file.
        # We treat `voice` as a speaker-file path when it points at a real file,
        # otherwise ignore it (the model's built-in speaker is used).
        voice = body.get("voice")
        speaker_file = voice if voice and Path(str(voice)).is_file() else None

        if not model_name:
            self._error(400, "missing required field: model")
            return
        if not text or not str(text).strip():
            self._error(400, "missing required field: input")
            return
        if response_format not in ("wav", "pcm"):
            self._error(
                400,
                f"unsupported response_format {response_format!r}; this endpoint "
                "only produces 'wav'.",
            )
            return

        model = _tts_models().get(model_name)
        if model is None:
            self._error(404, f"unknown TTS model: {model_name}")
            return

        try:
            audio = synthesize(model, str(text), speaker_file=speaker_file)
        except subprocess.TimeoutExpired:
            self._error(504, "llama-tts timed out")
            return
        except Exception as e:  # noqa: BLE001
            self._error(500, str(e))
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


def main(argv: list[str] | None = None) -> int:
    s = settings()
    host, port = s.tts_host, s.tts_port
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"inferhost-tts listening on {host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
