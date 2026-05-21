"""TOML-backed registry of locally configured models."""
from __future__ import annotations

import socket
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w

from inferhost.core import paths


def _port_free(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


@dataclass
class Model:
    name: str
    repo_id: str
    filename: str
    quant: str | None = None
    ctx: int = 8192
    port: int = 0
    size_gib: float = 0.0
    local_path: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quant"] = d["quant"] or ""
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(
            name=d["name"],
            repo_id=d["repo_id"],
            filename=d["filename"],
            quant=d.get("quant") or None,
            ctx=int(d.get("ctx", 8192)),
            port=int(d.get("port", 0)),
            size_gib=float(d.get("size_gib", 0.0)),
            local_path=d.get("local_path", ""),
        )


@dataclass
class Registry:
    models: list[Model] = field(default_factory=list)

    def add(self, model: Model) -> None:
        self.remove(model.name)
        self.models.append(model)

    def remove(self, name: str) -> bool:
        before = len(self.models)
        self.models = [m for m in self.models if m.name != name]
        return len(self.models) < before

    def rename(self, old: str, new: str) -> bool:
        """Rename a model in-place. Returns False if ``old`` is missing or ``new`` is taken."""
        if old == new:
            return False
        if self.get(new) is not None:
            return False
        m = self.get(old)
        if m is None:
            return False
        m.name = new
        return True

    def get(self, name: str) -> Model | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def names(self) -> list[str]:
        return [m.name for m in self.models]

    def next_port(self, base: int) -> int:
        used = {m.port for m in self.models if m.port}
        candidate = base + 1
        # Skip ports used in registry OR currently held by a foreign process
        while candidate in used or not _port_free(candidate):
            candidate += 1
            if candidate > base + 200:
                raise RuntimeError(f"Could not find free port near {base}")
        return candidate


def _path() -> Path:
    return paths.registry_path()


def load() -> Registry:
    p = _path()
    if not p.exists():
        return Registry()
    with p.open("rb") as f:
        data = tomllib.load(f)
    models = [Model.from_dict(m) for m in data.get("models", [])]
    return Registry(models=models)


def save(reg: Registry) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"models": [m.to_dict() for m in reg.models]}
    with p.open("wb") as f:
        tomli_w.dump(data, f)
