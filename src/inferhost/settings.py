"""Runtime configuration loaded from environment / .env file."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERHOST_",
        env_file=(".env", Path.cwd() / ".env"),
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
