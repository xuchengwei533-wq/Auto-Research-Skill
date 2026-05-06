from __future__ import annotations

import os
from pathlib import Path

from nightrunner import auth_store


def test_save_load_delete_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("HOME", str(tmp_path))

    path = auth_store.save_api_key(
        "deepseek",
        "sk-test-1234abcd",
        base_url="https://api.deepseek.com",
    )

    if os.name == "nt":
        expected = tmp_path / "AppData" / "Roaming" / "nightrunner" / "config.json"
    else:
        expected = tmp_path / ".config" / "nightrunner" / "config.json"
    assert path == expected
    assert path.exists()
    assert auth_store.load_api_key("deepseek") == "sk-test-1234abcd"

    data = auth_store.load_auth_config()
    assert data == {
        "providers": {
            "deepseek": {
                "api_key": "sk-test-1234abcd",
                "base_url": "https://api.deepseek.com",
            }
        }
    }

    assert auth_store.delete_api_key("deepseek") is True
    assert auth_store.load_api_key("deepseek") is None


def test_mask_api_key_does_not_leak_full_value() -> None:
    masked = auth_store.mask_api_key("sk-secret-1234abcd")
    assert masked == "sk-****abcd"
    assert "secret" not in masked
    assert "1234" not in masked
