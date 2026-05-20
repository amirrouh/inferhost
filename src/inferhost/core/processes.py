"""Daemon lifecycle for llama-swap and the LiteLLM gateway."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from inferhost.core import paths
from inferhost.settings import settings


@dataclass
class DaemonStatus:
    name: str
    running: bool
    pid: int | None
    port: int | None
    log_path: Path | None


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


def _kill_pid(pid: int, timeout: float = 8.0) -> None:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = proc.children(recursive=True)
    for c in children:
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.terminate()
    except psutil.NoSuchProcess:
        return
    end = time.time() + timeout
    while time.time() < end and proc.is_running():
        time.sleep(0.1)
    if proc.is_running():
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    for c in children:
        try:
            if c.is_running():
                c.kill()
        except psutil.NoSuchProcess:
            pass


def _spawn(cmd: list[str], log_path: Path, pid_file: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    pid_file.write_text(str(proc.pid))
    return proc.pid


# ---- llama-swap ----

def swap_status() -> DaemonStatus:
    pid = _read_pid(paths.swap_pid_file())
    running = pid is not None and _alive(pid)
    if pid is not None and not running:
        try:
            paths.swap_pid_file().unlink(missing_ok=True)
        except OSError:
            pass
    return DaemonStatus(
        name="llama-swap",
        running=running,
        pid=pid if running else None,
        port=settings().swap_port,
        log_path=paths.swap_log_path(),
    )


def start_swap() -> DaemonStatus:
    binary = paths.llama_swap_path()
    if not binary.exists():
        raise RuntimeError(
            "llama-swap binary not found. Run `inferhost install` first."
        )
    cfg = paths.llama_swap_config_path()
    if not cfg.exists():
        raise RuntimeError(
            "llama-swap config not found. Add a model with `inferhost add <repo>` first."
        )
    st = swap_status()
    if st.running:
        return st
    port = settings().swap_port
    if not _port_free_local(port):
        raise RuntimeError(
            f"Port {port} is already in use. Set INFERHOST_SWAP_PORT to a free port in .env "
            f"(or your environment) and try again."
        )
    cmd = [str(binary), "--config", str(cfg), "--listen", f":{port}"]
    pid = _spawn(cmd, paths.swap_log_path(), paths.swap_pid_file())
    # Wait up to 3s for it to bind; fail fast if it died.
    for _ in range(30):
        time.sleep(0.1)
        if not _alive(pid):
            raise RuntimeError(
                f"llama-swap exited shortly after launch. "
                f"Check the log: {paths.swap_log_path()}"
            )
        if not _port_free_local(port):  # i.e. swap has bound it
            break
    return swap_status()


def _port_free_local(port: int) -> bool:
    import socket
    # Check 0.0.0.0 binding — that's what llama-swap uses (--listen :port).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
        return True


def stop_swap() -> None:
    pid = _read_pid(paths.swap_pid_file())
    if pid is not None and _alive(pid):
        _kill_pid(pid)
    try:
        paths.swap_pid_file().unlink(missing_ok=True)
    except OSError:
        pass


# ---- litellm gateway (optional) ----

def gateway_available() -> bool:
    return shutil.which("litellm") is not None


def gateway_status() -> DaemonStatus:
    pid = _read_pid(paths.gateway_pid_file())
    running = pid is not None and _alive(pid)
    if pid is not None and not running:
        try:
            paths.gateway_pid_file().unlink(missing_ok=True)
        except OSError:
            pass
    return DaemonStatus(
        name="litellm",
        running=running,
        pid=pid if running else None,
        port=settings().gateway_port,
        log_path=paths.gateway_log_path(),
    )


def start_gateway() -> DaemonStatus:
    if not gateway_available():
        raise RuntimeError(
            "litellm not found on PATH. Install the gateway extra: "
            "pip install 'inferhost[gateway]'"
        )
    cfg = paths.litellm_config_path()
    if not cfg.exists():
        raise RuntimeError(
            "litellm config not found. Add a model with `inferhost add <repo>` first."
        )
    st = gateway_status()
    if st.running:
        return st
    cmd = [
        "litellm",
        "--config", str(cfg),
        "--host", "0.0.0.0",
        "--port", str(settings().gateway_port),
    ]
    _spawn(cmd, paths.gateway_log_path(), paths.gateway_pid_file())
    time.sleep(0.5)
    return gateway_status()


def stop_gateway() -> None:
    pid = _read_pid(paths.gateway_pid_file())
    if pid is not None and _alive(pid):
        _kill_pid(pid)
    try:
        paths.gateway_pid_file().unlink(missing_ok=True)
    except OSError:
        pass


def stop_all() -> None:
    stop_gateway()
    stop_swap()
