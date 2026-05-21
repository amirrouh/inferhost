"""Runtime configuration loaded from environment / .env file.

Configuration precedence (highest wins):

1. Real process environment variables (`INFERHOST_*`).
2. A TUI-managed overrides file at ``<config_dir>/inferhost.env`` — written when the
   user edits settings in the TUI.
3. A project-local ``.env`` in the current working directory.
4. Built-in defaults.

Users who prefer to manage everything by hand can stick to step 3 (a `.env` file).
Users who change settings in the TUI get step 2 — the TUI just writes another
`.env`-style file in the user config dir.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _default_config_dir() -> Path:
    return _expand("~/.config/inferhost")


def overrides_env_path() -> Path:
    """Where the TUI persists user-edited settings."""
    return _default_config_dir() / "inferhost.env"


# pydantic-settings reads env_file entries in order; later entries override earlier ones.
# So: .env (project-local) is loaded first, then the TUI-managed file overrides it,
# then real env vars override both.
_ENV_FILES: tuple[str, ...] = (
    ".env",
    str(Path.cwd() / ".env"),
    str(overrides_env_path()),
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERHOST_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gateway_port: int = 9001
    swap_port: int = 9090

    data_dir: Path = Field(default=Path("~/.local/share/inferhost"))
    config_dir: Path = Field(default=Path("~/.config/inferhost"))
    hf_cache: Path = Field(default=Path("~/.cache/huggingface"))

    gpu_layers: int = 99
    default_ctx: int = 8192
    flash_attention: str = "on"

    # Number of parallel request slots per llama-server instance (--parallel N).
    # 1 is the safest default — one in-flight request at a time per model, no KV
    # cache contention. Bump this if you need concurrency on the same model.
    parallel_slots: int = 1

    # Speculative decoding — only applied to MTP-capable models (filename contains "mtp").
    # Two lanes are stacked: --spec-type draft-mtp AND --spec-type ngram-mod.
    # Set any of these to 0 to disable that lane individually.
    spec_draft_n_max: int = 2          # MTP draft tokens per step (PR author: 2 is sweet spot)
    spec_ngram_mod_n_match: int = 24   # min matching sequence length before ngram drafts
    spec_ngram_mod_n_min: int = 48     # min context window to search back through
    spec_ngram_mod_n_max: int = 64     # max draft tokens ngram-mod proposes on strong match

    llamacpp_version: str = "latest"
    llamaswap_version: str = "latest"

    log_level: str = "INFO"

    def model_post_init(self, __ctx) -> None:  # type: ignore[override]
        self.data_dir = _expand(self.data_dir)
        self.config_dir = _expand(self.config_dir)
        self.hf_cache = _expand(self.hf_cache)


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings


# ---- user-editable overrides (written by the TUI) -------------------------------

# Whitelist of fields the TUI is allowed to persist. Keeps the override file tidy
# and prevents accidental writes of internal / path-y settings.
EDITABLE_FIELDS: tuple[str, ...] = (
    "swap_port",
    "gateway_port",
    "default_ctx",
    "gpu_layers",
    "flash_attention",
    "parallel_slots",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Managed by inferhost — written when you edit Settings in the TUI.",
        "# You can also edit this file by hand; values follow KEY=VALUE format.",
        "",
    ]
    for key in sorted(values):
        lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_overrides() -> dict[str, str]:
    """Read the TUI-managed overrides file as a dict of {bare_field: value}.

    Strips the ``INFERHOST_`` prefix so callers see field names matching ``Settings``.
    """
    raw = _parse_env_file(overrides_env_path())
    prefix = "INFERHOST_"
    return {
        k[len(prefix):].lower(): v
        for k, v in raw.items()
        if k.startswith(prefix)
    }


def save_overrides(updates: dict[str, object]) -> Path:
    """Merge ``updates`` into the TUI-managed overrides file and reload settings.

    Keys must be ``Settings`` field names (e.g. ``swap_port``). Only fields listed in
    ``EDITABLE_FIELDS`` are persisted; the rest are silently ignored.
    """
    path = overrides_env_path()
    existing = _parse_env_file(path)
    prefix = "INFERHOST_"
    for key, value in updates.items():
        if key not in EDITABLE_FIELDS:
            continue
        existing[f"{prefix}{key.upper()}"] = str(value)
    _write_env_file(path, existing)
    reload_settings()
    return path
