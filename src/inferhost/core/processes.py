"""Daemon lifecycle for llama-swap and the LiteLLM gateway.

Also exposes two read-only state queries used by the TUI status bars:
``query_gpus()`` (nvidia-smi snapshot) and ``currently_loaded()`` (which
model llama-swap currently has resident in VRAM, via its ``/running``
HTTP endpoint).
"""
from __future__ import annotations

import shutil
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
    cmd = [str(binary), "--config", str(cfg), "--listen", f"127.0.0.1:{port}"]
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
    # Check 127.0.0.1 binding — llama-swap now binds loopback only (--listen 127.0.0.1:port).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
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


@dataclass
class GpuStat:
    index: int
    name: str
    mem_used_mib: int
    mem_total_mib: int
    util_pct: int


def query_gpus(timeout: float = 1.0) -> list[GpuStat]:
    """Snapshot per-GPU VRAM + utilization via nvidia-smi.

    Returns ``[]`` if nvidia-smi is missing, errors, or hangs — the TUI then
    just hides the GPU bar instead of showing stale data.
    """
    bin_path = shutil.which("nvidia-smi")
    if bin_path is None:
        return []
    try:
        out = subprocess.run(
            [
                bin_path,
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
            text=True,
        ).stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return []
    gpus: list[GpuStat] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                GpuStat(
                    index=int(parts[0]),
                    name=parts[1],
                    mem_used_mib=int(parts[2]),
                    mem_total_mib=int(parts[3]),
                    util_pct=int(parts[4]),
                )
            )
        except ValueError:
            continue
    return gpus


def currently_loaded(timeout: float = 0.5) -> list[str]:
    """Names of models llama-swap currently has resident in VRAM.

    Hits llama-swap's ``GET /running`` endpoint. Returns ``[]`` if swap isn't
    running, the call fails, or the JSON shape is unexpected.
    """
    if not swap_status().running:
        return []
    # Lazy import: keeps `inferhost status` and other non-TUI paths from paying
    # the httpx import cost.
    import httpx

    port = settings().swap_port
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/running", timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:  # noqa: BLE001
        return []
    items = data.get("running") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            n = item.get("model")
            if isinstance(n, str) and n:
                names.append(n)
    return names


def force_load_model(name: str, timeout: float = 30.0) -> bool:
    """Force llama-swap to load ``name`` into VRAM by issuing a tiny chat request.

    Returns True on success, False otherwise. Does not raise on timeout or error.
    The response body is discarded — what matters is that the model becomes resident.
    """
    import httpx

    port = settings().swap_port
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
    }
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:  # noqa: BLE001
        return False


def force_unload_model(name: str, timeout: float = 5.0) -> bool:
    """Tell llama-swap to evict ``name`` from VRAM.

    Uses ``POST /api/models/unload/{name}`` (llama-swap v217+). Older versions
    exposed ``/upstream/{name}/unload`` instead; we try that as a fallback so
    pinned users on an older llama-swap still get unloads. Returns True on
    success, False if every variant rejected the request.
    """
    import httpx

    port = settings().swap_port
    candidates = (
        f"http://127.0.0.1:{port}/api/models/unload/{name}",
        f"http://127.0.0.1:{port}/upstream/{name}/unload",
    )
    for url in candidates:
        try:
            r = httpx.post(url, timeout=timeout)
            if r.status_code < 400:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def reload_if_running() -> tuple[bool, bool]:
    """Restart any daemons currently running so they pick up new on-disk configs.

    LiteLLM and llama-swap both read their config once at startup, so a registry
    mutation (add / remove / rename) is invisible until the proxy is restarted.

    Returns ``(swap_reloaded, gateway_reloaded)``.
    """
    swap_was_running = swap_status().running
    gateway_was_running = gateway_status().running
    if swap_was_running:
        stop_swap()
        start_swap()
    if gateway_was_running:
        stop_gateway()
        start_gateway()
    return swap_was_running, gateway_was_running
