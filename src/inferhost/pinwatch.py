"""inferhost-pinwatch daemon: keep pinned models resident in VRAM.

"Pinned" promises the user "this model stays in VRAM" — but three things
silently break that promise, because llama-swap only *lazy*-loads models:

1. Exclusive-group swap: a swappable model that can't co-fit next to the pins
   evicts them on load (graceful, by design). When it later idle-unloads, the
   GPU sits empty and the pins stay cold.
2. Crashes: a pinned llama-server that dies (OOM, driver hiccup) is never
   restarted by llama-swap on its own.
3. Daemon (re)starts: a reboot or config reload brings llama-swap up with
   nothing loaded.

This daemon closes all three gaps with one rule: whenever a pinned model is
not resident AND no swappable model is using (or transitioning in/out of) the
GPU, load the pin back. It never preempts a resident swappable model — that
side wins until it goes idle and unloads, exactly like the swap groups intend.

Runs as ``python -m inferhost.pinwatch``, spawned/stopped alongside llama-swap
by :mod:`inferhost.core.processes`. Exits on its own when llama-swap stays
gone, so it can't linger as an orphan.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

# Consecutive polls a pin must stay warm-eligible before we load it. One poll
# of grace absorbs the race where a swappable model idle-unloads and is
# immediately re-requested — without it we'd load the pin just to have the
# returning guest evict it again a second later.
GRACE_POLLS = 2

# After a failed warm attempt (OOM, bad file, llama-server crash-loop), leave
# the model alone for this long instead of hammering it every poll.
FAIL_BACKOFF_S = 300.0

# Polls llama-swap may be unreachable before we conclude it's gone and exit.
SWAP_GONE_POLLS = 3


def pins_to_warm(states: dict[str, str], pinned: list[str]) -> list[str]:
    """Which pinned models should be loaded right now, given llama-swap state.

    ``states`` is model_name -> llama-swap state (from ``GET /running``);
    models absent from it are not resident. Returns ``[]`` whenever any
    non-pinned model is active in ANY live state — 'ready' means a guest holds
    the VRAM (never preempt it), 'starting'/'stopping' mean a swap transition
    is mid-flight and we must not pile a load on top of it. 'stopped' entries
    are exited processes llama-swap still lists; they hold no VRAM.
    """
    busy = any(
        state != "stopped" for name, state in states.items() if name not in pinned
    )
    if busy:
        return []
    return [
        name
        for name in pinned
        if states.get(name, "stopped") == "stopped"
    ]


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _pinned_swap_models() -> list[str]:
    """Names of pinned models that llama-swap fronts.

    Chat/vision/image models — and Orpheus TTS, whose GGUF runs under
    llama-server like any chat model. Kokoro pins are the inferhost-tts
    daemon's job (in-process pre-warm), not ours.
    """
    from inferhost.core import registry
    from inferhost.core.configs import is_swap_fronted

    return [m.name for m in registry.load().models if m.pin and is_swap_fronted(m)]


def main() -> int:
    from inferhost.core import processes
    from inferhost.settings import settings

    interval = max(2, settings().pinwatch_poll_s)
    _log(f"inferhost-pinwatch up (poll every {interval}s)")
    swap_gone = 0
    streak: dict[str, int] = {}  # model -> consecutive warm-eligible polls
    backoff_until: dict[str, float] = {}  # model -> monotonic time of next try
    while True:
        time.sleep(interval)
        if not processes.swap_status().running:
            swap_gone += 1
            if swap_gone >= SWAP_GONE_POLLS:
                _log("llama-swap is gone — exiting")
                return 0
            continue
        swap_gone = 0
        pinned = _pinned_swap_models()
        if not pinned:
            streak.clear()
            continue
        states = processes.model_states(timeout=2.0)
        eligible = pins_to_warm(states, pinned)
        for name in list(streak):
            if name not in eligible:
                del streak[name]
        now = time.monotonic()
        for name in eligible:
            streak[name] = streak.get(name, 0) + 1
            if streak[name] < GRACE_POLLS or now < backoff_until.get(name, 0.0):
                continue
            _log(f"{name}: pinned but not resident and GPU side is idle — loading")
            if processes.force_load_model(name, timeout=300.0):
                _log(f"{name}: back in VRAM")
                backoff_until.pop(name, None)
            else:
                _log(
                    f"{name}: load failed — retrying in {FAIL_BACKOFF_S:.0f}s "
                    "(check the model's .err.log)"
                )
                backoff_until[name] = time.monotonic() + FAIL_BACKOFF_S
            streak.pop(name, None)


if __name__ == "__main__":
    sys.exit(main())
