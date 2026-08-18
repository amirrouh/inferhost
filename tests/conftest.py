"""Test isolation from the developer's own inferhost installation.

Settings are env-only, but pydantic-settings also reads `.env` in the cwd and
`<config_dir>/inferhost.env` — so without this fixture the suite's result
depends on how the machine running it happens to be configured. Setting
`INFERHOST_LLAMA_SERVER_PATH` in your own `inferhost.env` (the supported way to
point inferhost at a self-compiled CUDA build) made six tests fail on a
developer box while CI, which has no such file, stayed green.

The fixture redirects the config and data dirs at a per-test tmp dir and clears
every `INFERHOST_*` variable, so tests see stock defaults unless they set
something themselves. Individual tests still monkeypatch these to their own
tmp_path; that runs after this fixture and wins.
"""
from __future__ import annotations

import os

import pytest

from inferhost import settings as settings_mod
from inferhost.settings import reload_settings


@pytest.fixture(autouse=True)
def _isolate_inferhost_config(tmp_path_factory, monkeypatch):
    for key in [k for k in os.environ if k.startswith("INFERHOST_")]:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path_factory.mktemp("inferhost-isolated")
    monkeypatch.setenv("INFERHOST_CONFIG_DIR", str(root / "config"))
    monkeypatch.setenv("INFERHOST_DATA_DIR", str(root / "data"))
    # The overrides file lives at a fixed ~/.config/inferhost/inferhost.env
    # (it is deliberately NOT relative to INFERHOST_CONFIG_DIR), and the file
    # list is baked into model_config at import time — so point it, and the
    # project-local `.env`, at the empty tmp dir instead.
    monkeypatch.chdir(root)
    monkeypatch.setitem(
        settings_mod.Settings.model_config, "env_file", (str(root / "none.env"),))
    reload_settings()
    yield
    reload_settings()
