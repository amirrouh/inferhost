"""Hardware probing: GPU count, VRAM, RAM, OS, arch."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_mib: int
    vram_free_mib: int
    driver: str = ""

    @property
    def vram_total_gib(self) -> float:
        return round(self.vram_total_mib / 1024, 2)

    @property
    def vram_free_gib(self) -> float:
        return round(self.vram_free_mib / 1024, 2)


@dataclass
class ProbeResult:
    os: str
    arch: str
    ram_gib: float
    gpus: list[GPUInfo] = field(default_factory=list)
    nvidia_smi: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def primary_vram_gib(self) -> float:
        return self.gpus[0].vram_total_gib if self.gpus else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_gpu"] = self.has_gpu
        d["primary_vram_gib"] = self.primary_vram_gib
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _read_ram_gib() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return 0.0


def _probe_nvidia() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mib=int(parts[2]),
                    vram_free_mib=int(parts[3]),
                    driver=parts[4],
                )
            )
        except ValueError:
            continue
    return gpus


def probe() -> ProbeResult:
    gpus = _probe_nvidia()
    notes: list[str] = []
    if not gpus:
        notes.append("No NVIDIA GPU detected; will run on CPU (slow for large models).")
    return ProbeResult(
        os=platform.system(),
        arch=platform.machine(),
        ram_gib=_read_ram_gib(),
        gpus=gpus,
        nvidia_smi=bool(gpus) or bool(shutil.which("nvidia-smi")),
        notes=notes,
    )
