from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout

import pytest

from nightrunner import cli


def test_root_help_contains_all_commands(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in [
        "init",
        "setup",
        "baseline",
        "night",
        "run",
        "report",
        "apply",
        "clean",
        "auth",
        "status",
        "tail",
    ]:
        assert command in out


def test_setup_help_keeps_current_flags(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["setup", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    expected_flags = [
        "--project",
        "--editable",
        "--train-command",
        "--metric",
        "--api-key-env",
        "--base-url",
        "--model",
        "--run-baseline",
        "--yes",
        "--lower-is-better",
        "--higher-is-better",
        "--metric-regex",
    ]
    for flag in expected_flags:
        assert flag in out


def test_auth_defaults_to_status(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    out_auth = io.StringIO()
    with redirect_stdout(out_auth):
        code_auth = cli.main(["auth"])

    out_status = io.StringIO()
    with redirect_stdout(out_status):
        code_status = cli.main(["auth", "status"])

    assert code_auth == code_status == 1
    assert out_auth.getvalue() == out_status.getvalue()
    assert "No API key found" in out_auth.getvalue()
    assert "sk-" not in out_auth.getvalue()


def test_dispatch_routes_auth_to_auth_cli(monkeypatch) -> None:
    captured: dict[str, str | None] = {"auth_command": None}

    def _fake_handle(auth_command: str | None) -> int:
        captured["auth_command"] = auth_command
        return 7

    monkeypatch.setattr(cli, "handle_auth_command", _fake_handle)
    args = argparse.Namespace(command="auth", auth_command="status", project=None)
    assert cli.dispatch(args) == 7
    assert captured["auth_command"] == "status"
