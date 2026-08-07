"""inferhost-tts: a tiny OpenAI-compatible text-to-speech daemon.

Serves ``POST /v1/audio/speech`` for every registered TTS model, over the same
OpenAI wire protocol as everything else in the stack. LiteLLM routes the
gateway's ``/v1/audio/speech`` here for any model registered with
``mode: audio_speech``; the daemon is also reachable directly on its port.

Three engines, dispatched per model (see ``configs.tts_engine``):
- **Kokoro** (model file ends in ``.onnx``): synthesized in-process via the
  kokoro-onnx package (ONNX Runtime, CPU). The model loads once and stays
  resident, so requests after the first are fast. Pinned Kokoro models are
  pre-warmed at daemon startup so even the first request is fast.
- **Orpheus** (model ``.gguf`` + SNAC decoder ``.onnx`` as vocoder_path): a
  llama-architecture speech-LLM. The GGUF is served by llama-server behind
  llama-swap like any chat model — this daemon prompts it over
  ``/v1/completions``, collects the ``<custom_token_N>`` SNAC audio tokens it
  generates, and decodes them to a waveform in-process (ONNX Runtime, CPU).
  llama-swap owns the VRAM lifecycle, so Orpheus swaps against chat models
  and honors pinning exactly like they do.
- **OuteTTS-style GGUF** (model ``.gguf`` + WavTokenizer ``.gguf``): wraps
  llama.cpp's standalone ``llama-tts`` binary — the only way to render
  OuteTTS+WavTokenizer; ``llama-server`` has no endpoint for them.
  ``llama-tts`` is one-shot, so each request pays the model's load cost.

Synthesis is serialized with a lock so concurrent requests can't fight over
memory. Run with ``python -m inferhost.tts_serve``.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inferhost.core import paths, registry
from inferhost.core.configs import tts_engine
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
    """Name -> Model for every registered TTS model.

    Non-empty vocoder_path marks a model as TTS: a WavTokenizer GGUF for
    OuteTTS-style models, or the bundled voices .npz for Kokoro.
    """
    return {m.name: m for m in registry.load().models if m.vocoder_path}


# ── Kokoro (kokoro-onnx) engine ────────────────────────────────────────────
# Kokoro-82M is a StyleTTS2-derived architecture with no llama.cpp support, so
# it's synthesized in-process via the kokoro-onnx package (ONNX Runtime on
# CPU — at 82M params that's well faster than realtime, and it leaves VRAM to
# the chat models). The instance is cached, so the model loads once and stays
# resident. A TTS model is routed here when its model file ends in ``.onnx``;
# its vocoder_path points at the voices .npz bundled at add time.

_KOKORO_CACHE: dict[str, object] = {}

# OpenAI's named voice presets mapped onto Kokoro's closest equivalents, so
# off-the-shelf clients sending voice="alloy" etc. keep working.
_OPENAI_VOICE_MAP = {
    "alloy": "af_alloy",
    "ash": "am_adam",
    "ballad": "bf_alice",
    "coral": "af_heart",
    "echo": "am_echo",
    "fable": "bm_fable",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "sage": "af_sarah",
    "shimmer": "af_sky",
    "verse": "am_michael",
}

# Kokoro voice names encode language+gender in their prefix (af_* = American
# female, bm_* = British male, jf_* = Japanese female, ...). The first letter
# picks the phonemizer language.
_KOKORO_LANGS = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


class _InputCastingSession:
    """Wraps the ONNX session to cast inputs to the graph's declared dtypes.

    kokoro-onnx 0.5.0 hardcodes ``speed`` as int32 for ``input_ids``-style
    exports, but the onnx-community graphs declare float32, which makes every
    request fail with INVALID_ARGUMENT. Casting by the graph's own signature
    fixes that for any export variant, and is a no-op when dtypes already
    match. Everything except ``run`` proxies straight to the real session.
    """

    _NP_TYPES = {
        "tensor(float)": "float32",
        "tensor(float16)": "float16",
        "tensor(int64)": "int64",
        "tensor(int32)": "int32",
    }

    def __init__(self, sess) -> None:
        self._sess = sess
        self._input_types = {i.name: i.type for i in sess.get_inputs()}

    def __getattr__(self, name: str):
        return getattr(self._sess, name)

    def run(self, output_names, inputs: dict):
        import numpy as np

        cast = {}
        for name, value in inputs.items():
            arr = np.asarray(value)
            want = self._NP_TYPES.get(self._input_types.get(name, ""))
            cast[name] = arr.astype(want) if want and arr.dtype != np.dtype(want) else arr
        return self._sess.run(output_names, cast)


def _get_kokoro(model: registry.Model):
    inst = _KOKORO_CACHE.get(model.name)
    if inst is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise RuntimeError(
                "kokoro-onnx is not installed in inferhost's environment — "
                "upgrade/reinstall inferhost (e.g. `uv tool install --force "
                "inferhost` or `pipx reinstall inferhost`) to serve Kokoro."
            ) from e
        inst = Kokoro(_model_path(model), model.vocoder_path)
        inst.sess = _InputCastingSession(inst.sess)
        _KOKORO_CACHE[model.name] = inst
    return inst


def _resolve_kokoro_voice(kokoro, requested: str | None) -> str:
    """Pick a real voice from the bundle for whatever the client sent.

    Order: exact Kokoro name → OpenAI preset alias → the INFERHOST_TTS_VOICE
    default → first voice in the bundle. Never errors on an unknown name; TTS
    clients hardcode voices and a fallback beats a 400.
    """
    available = set(kokoro.get_voices())
    req = (requested or "").strip()
    for cand in (req, _OPENAI_VOICE_MAP.get(req.lower()), settings().tts_voice):
        if cand and cand in available:
            return cand
    return sorted(available)[0]


def _to_wav_bytes(samples, sample_rate: int) -> bytes:
    """float32 [-1, 1] samples -> 16-bit mono WAV bytes."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _synthesize_kokoro(
    model: registry.Model, text: str, voice: str | None, speed: float
) -> bytes:
    kokoro = _get_kokoro(model)
    name = _resolve_kokoro_voice(kokoro, voice)
    lang = _KOKORO_LANGS.get(name[:1], "en-us")
    speed = min(max(float(speed or 1.0), 0.5), 2.0)
    with _SYNTH_LOCK:
        samples, sample_rate = kokoro.create(text, voice=name, speed=speed, lang=lang)
    return _to_wav_bytes(samples, sample_rate)


# ── Orpheus (llama-server + SNAC) engine ───────────────────────────────────
# Orpheus-3B generates audio as a stream of <custom_token_N> strings, 7 tokens
# per 2048-sample frame across SNAC's 3 codebook layers. llama-swap fronts the
# GGUF (lazy-load, swap groups, pinning all apply); this daemon only formats
# the prompt, strips the token stream back into SNAC codes, and runs the small
# ONNX decoder on CPU. Prompt format, sampling params, and the token→code
# arithmetic follow the reference llama.cpp serving stack (Orpheus-FastAPI).

_ORPHEUS_SAMPLE_RATE = 24000
_ORPHEUS_MAX_TOKENS = 8192  # ~ 1.5 min of audio; llama-server stops at EOS sooner
_ORPHEUS_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")

# OpenAI preset voices mapped onto the closest Orpheus voice, mirroring the
# Kokoro map above so voice="alloy" etc. works against any TTS model.
_ORPHEUS_OPENAI_VOICE_MAP = {
    "alloy": "tara",
    "ash": "dan",
    "ballad": "leah",
    "coral": "mia",
    "echo": "leo",
    "fable": "dan",
    "nova": "jess",
    "onyx": "zac",
    "sage": "leah",
    "shimmer": "zoe",
    "verse": "leo",
}

_ORPHEUS_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

_SNAC_CACHE: dict[str, object] = {}


def _get_snac(decoder_path: str):
    sess = _SNAC_CACHE.get(decoder_path)
    if sess is None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is not installed in inferhost's environment — "
                "upgrade/reinstall inferhost to serve Orpheus TTS."
            ) from e
        sess = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
        _SNAC_CACHE[decoder_path] = sess
    return sess


def _resolve_orpheus_voice(requested: str | None) -> str:
    req = (requested or "").strip().lower()
    for cand in (req, _ORPHEUS_OPENAI_VOICE_MAP.get(req), settings().tts_voice):
        if cand and cand in _ORPHEUS_VOICES:
            return cand
    return _ORPHEUS_VOICES[0]


def orpheus_snac_codes(generated: str) -> tuple[list[int], list[int], list[int]]:
    """Parse llama-server's generated text into the 3 SNAC codebook layers.

    Each ``<custom_token_N>`` carries one code: ``N - 10 - (i % 7) * 4096``
    where ``i`` counts the accepted tokens so far — the model emits codes for
    the 7 frame positions with disjoint 4096-wide id ranges. Out-of-range
    results (the model's non-audio special tokens, e.g. start/end markers)
    are skipped without advancing ``i``, matching the reference decoder. The
    flat 7-token frames are then dealt to SNAC's layers: position 0 → layer 0,
    positions 1/4 → layer 1, positions 2/3/5/6 → layer 2 (1:2:4 temporal
    resolution). A trailing partial frame is dropped.
    """
    ids: list[int] = []
    for match in _ORPHEUS_TOKEN_RE.finditer(generated):
        code = int(match.group(1)) - 10 - ((len(ids) % 7) * 4096)
        if 0 < code < 4096:
            ids.append(code)
    frames = len(ids) // 7
    c0: list[int] = []
    c1: list[int] = []
    c2: list[int] = []
    for j in range(frames):
        f = ids[j * 7 : (j + 1) * 7]
        c0.append(f[0])
        c1 += [f[1], f[4]]
        c2 += [f[2], f[3], f[5], f[6]]
    return c0, c1, c2


def _orpheus_generate(model: registry.Model, prompt: str) -> str:
    """Ask llama-server (via llama-swap) to generate the audio-token stream.

    Going through llama-swap — not the model port directly — is what makes
    the whole VRAM story work: the request lazy-loads the model, swap groups
    evict what must be evicted, and a pinned Orpheus stays resident.
    """
    import httpx

    url = f"http://127.0.0.1:{settings().swap_port}/v1/completions"
    payload = {
        "model": model.name,
        "prompt": prompt,
        "max_tokens": _ORPHEUS_MAX_TOKENS,
        # Reference sampling for Orpheus: creative enough to sound natural,
        # repeat_penalty keeps it from looping on a syllable.
        "temperature": 0.6,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "stream": False,
    }
    try:
        r = httpx.post(url, json=payload, timeout=_SYNTH_TIMEOUT)
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"llama-swap unreachable ({e}) — Orpheus needs the stack running "
            "(`inferhost start`) so llama-server can generate its audio tokens."
        ) from e
    if r.status_code != 200:
        tail = r.text.strip()[-300:]
        raise RuntimeError(f"llama-server error {r.status_code} for {model.name}: {tail}")
    try:
        return r.json()["choices"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"unexpected completion response for {model.name}: {e}") from e


def _synthesize_orpheus(model: registry.Model, text: str, voice: str | None) -> bytes:
    import numpy as np

    name = _resolve_orpheus_voice(voice)
    # "<|audio|>voice: text<|eot_id|>" is the trained prompt shape; the voice
    # prefix selects the speaker. (Orpheus has no speed control — the OpenAI
    # `speed` param is ignored for this engine.)
    prompt = f"<|audio|>{name}: {text}<|eot_id|>"
    with _SYNTH_LOCK:
        generated = _orpheus_generate(model, prompt)
        c0, c1, c2 = orpheus_snac_codes(generated)
        if not c0:
            raise RuntimeError(
                f"{model.name} produced no audio tokens — is this actually an "
                "Orpheus-family GGUF? (check the model's .err.log)"
            )
        snac = _get_snac(model.vocoder_path)
        (audio,) = snac.run(
            None,
            {
                "audio_codes.0": np.asarray([c0], dtype=np.int64),
                "audio_codes.1": np.asarray([c1], dtype=np.int64),
                "audio_codes.2": np.asarray([c2], dtype=np.int64),
            },
        )
    return _to_wav_bytes(audio[0, 0], _ORPHEUS_SAMPLE_RATE)


def synthesize(
    model: registry.Model, text: str, voice: str | None = None, speed: float = 1.0
) -> bytes:
    """Run the model's TTS engine for ``text`` and return the WAV bytes.

    Dispatches on ``configs.tts_engine``: kokoro-onnx in-process, Orpheus via
    llama-swap + SNAC, else llama-tts. ``voice`` is a voice name (or OpenAI
    preset) for Kokoro/Orpheus; for llama-tts it's honored only when it's a
    path to a speaker JSON file. Raises RuntimeError with the engine's error
    tail on failure.
    """
    engine = tts_engine(model)
    if engine == "kokoro":
        return _synthesize_kokoro(model, text, voice, speed)
    if engine == "orpheus":
        return _synthesize_orpheus(model, text, voice)

    speaker_file = voice if voice and Path(str(voice)).is_file() else None
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
        # For Kokoro, `voice` picks one of the bundled voices (OpenAI preset
        # names like "alloy" are mapped). For llama-tts it's honored only when
        # it's a path to a speaker JSON file; otherwise the built-in speaker.
        voice = body.get("voice")
        try:
            speed = float(body.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0

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
            audio = synthesize(model, str(text), voice=str(voice) if voice else None, speed=speed)
        except subprocess.TimeoutExpired:
            self._error(504, "TTS synthesis timed out")
            return
        except Exception as e:  # noqa: BLE001
            self._error(500, str(e))
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


def _prewarm_pinned_kokoro() -> None:
    """Load pinned Kokoro models into the in-process cache at daemon startup.

    This is what "pinned" means for the Kokoro engine: the model is always
    loaded and ready (system RAM, CPU inference), so even the first request
    after a restart pays no load cost. Unpinned Kokoro models still lazy-load
    on first use and stay resident after. Orpheus pinning is llama-swap's job
    (ttl=0 group + pinwatch), not ours.
    """
    for m in _tts_models().values():
        if m.pin and tts_engine(m) == "kokoro":
            try:
                _get_kokoro(m)
                print(f"inferhost-tts: pre-warmed pinned Kokoro model {m.name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"inferhost-tts: pre-warm failed for {m.name}: {e}", flush=True)


def main(argv: list[str] | None = None) -> int:
    s = settings()
    host, port = s.tts_host, s.tts_port
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"inferhost-tts listening on {host}:{port}", flush=True)
    # Off-thread so a slow model load never delays serving the port.
    threading.Thread(target=_prewarm_pinned_kokoro, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
