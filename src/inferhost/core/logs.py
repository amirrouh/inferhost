"""Log file helpers: tailing and following."""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

from inferhost.core import paths


def log_path(model_name: str | None) -> Path:
    if model_name is None or model_name == "":
        return paths.swap_log_path()
    if model_name == "gateway":
        return paths.gateway_log_path()
    if model_name == "swap":
        return paths.swap_log_path()
    return paths.model_log_path(model_name)


def tail(path: Path, n: int = 200) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
            lines = data.splitlines()[-n:]
            return [line.decode("utf-8", errors="replace") for line in lines]
        except OSError:
            return []


def tail_err_log(model_name: str, n: int = 5) -> list[str]:
    """Last non-blank lines of ``<model_name>.err.log`` — the raw stderr
    llama-server wrote before dying (see ``configs.py``'s ``err_log`` wiring).

    Used to enrich generic "Failed"/"Load failed" toasts with the real reason
    instead of leaving the user to go dig through the log panel themselves.
    Over-fetches (4x n) before filtering blanks so a log with sparse blank
    lines near the tail still yields ``n`` real lines.
    """
    path = paths.logs_dir() / f"{model_name}.err.log"
    lines = [ln for ln in tail(path, n * 4) if ln.strip()]
    return lines[-n:]


def follow(path: Path, from_end: bool = True, poll_sec: float = 0.5) -> Iterator[str]:
    while not path.exists():
        time.sleep(poll_sec)
    with path.open("r", errors="replace") as f:
        if from_end:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_sec)
                continue
            yield line.rstrip("\n")
