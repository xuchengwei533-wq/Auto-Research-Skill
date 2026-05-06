from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

from nightrunner.cli import main


def test_auth_and_status_without_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    out = io.StringIO()
    with redirect_stdout(out):
        auth_exit = main(["auth"])
    assert auth_exit == 1
    assert "No API key found" in out.getvalue()

    out = io.StringIO()
    with redirect_stdout(out):
        status_exit = main(["auth", "status"])
    assert status_exit == 1
    assert "No API key found" in out.getvalue()


def test_auth_login_status_logout(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-test-1234567890")

    out = io.StringIO()
    with redirect_stdout(out):
        login_exit = main(["auth", "login"])
    assert login_exit == 0
    assert "saved to" in out.getvalue()
    assert "1234567890" not in out.getvalue()

    out = io.StringIO()
    with redirect_stdout(out):
        status_exit = main(["auth", "status"])
    assert status_exit == 0
    status_text = out.getvalue()
    assert "user config" in status_text
    assert "sk-****7890" in status_text
    assert "1234567890" not in status_text

    out = io.StringIO()
    with redirect_stdout(out):
        logout_exit = main(["auth", "logout"])
    assert logout_exit == 0
    assert "removed from user config" in out.getvalue()


def test_auth_login_empty_returns_non_zero(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "   ")

    err = io.StringIO()
    with redirect_stderr(err):
        exit_code = main(["auth", "login"])
    assert exit_code == 1
    assert "No API key entered." in err.getvalue()
