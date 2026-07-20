"""Tests for core/logs.py — err-log tailing used to enrich failure toasts
(dashboard.py force_load_model / _apply_changes_worker / start / restart
paths, see A5 in the UX-fix plan)."""
from __future__ import annotations

from inferhost.core import logs, paths


def test_tail_err_log_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    assert logs.tail_err_log("no-such-model") == []


def test_tail_err_log_returns_last_n_non_blank_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    err = tmp_path / "my-model.err.log"
    lines = [f"line {i}" for i in range(10)]
    err.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = logs.tail_err_log("my-model", n=3)
    assert result == ["line 7", "line 8", "line 9"]


def test_tail_err_log_filters_blank_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    err = tmp_path / "gappy.err.log"
    # A run of blank lines right before the real error must not crowd out the
    # actual content — the last n NON-blank lines are what matter.
    err.write_text("real error one\n\n\n\nreal error two\n\n", encoding="utf-8")

    result = logs.tail_err_log("gappy", n=2)
    assert result == ["real error one", "real error two"]


def test_tail_err_log_empty_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    (tmp_path / "empty.err.log").write_text("", encoding="utf-8")
    assert logs.tail_err_log("empty") == []


def test_tail_err_log_uses_the_configs_naming_convention(tmp_path, monkeypatch):
    """Must read `<name>.err.log`, matching the file configs.py wires
    llama-server's stderr redirect (`2>>...`) into — a naming mismatch here
    would silently make every enriched toast come up empty."""
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    (tmp_path / "some-model.err.log").write_text("boom\n", encoding="utf-8")
    (tmp_path / "some-model.log").write_text("stdout noise, not this one\n", encoding="utf-8")
    assert logs.tail_err_log("some-model") == ["boom"]
