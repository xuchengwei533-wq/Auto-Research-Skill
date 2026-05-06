from __future__ import annotations

import pytest

from nightrunner.agent_api import resolve_api_settings


def test_environment_variable_takes_priority_over_user_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-9876wxyz")

    from nightrunner.auth_store import save_api_key

    save_api_key("deepseek", "sk-config-1234abcd", base_url="https://saved.example.com")

    api_key, base_url = resolve_api_settings(
        api_key_env="DEEPSEEK_API_KEY",
        provider="deepseek",
        base_url="https://api.deepseek.com",
    )

    assert api_key == "sk-env-9876wxyz"
    assert base_url == "https://api.deepseek.com"


def test_user_config_is_used_when_environment_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    from nightrunner.auth_store import save_api_key

    save_api_key("deepseek", "sk-config-1234abcd", base_url="https://saved.example.com")

    api_key, base_url = resolve_api_settings(
        api_key_env="DEEPSEEK_API_KEY",
        provider="deepseek",
        base_url="https://api.deepseek.com",
    )

    assert api_key == "sk-config-1234abcd"
    assert base_url == "https://saved.example.com"


def test_missing_key_mentions_auth_login(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        resolve_api_settings(
            api_key_env="DEEPSEEK_API_KEY",
            provider="deepseek",
            base_url="https://api.deepseek.com",
        )

    message = str(excinfo.value)
    assert "No API key found." in message
    assert "nightrunner auth login" in message
