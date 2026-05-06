"""User-level API credential storage for NightRunner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def get_config_path() -> Path:
    """Return the user-level auth config path."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "nightrunner" / "config.json"


def load_auth_config() -> dict[str, Any]:
    """Load auth config from user config directory."""
    path = get_config_path()
    if not path.exists():
        return {"providers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"providers": {}}
    if not isinstance(data, dict):
        return {"providers": {}}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        data["providers"] = {}
    return data


def save_auth_config(data: dict[str, Any]) -> Path:
    """Persist auth config with best-effort restrictive permissions."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def save_api_key(provider: str, api_key: str, base_url: str | None = None) -> Path:
    """Save provider API key and optional base_url."""
    data = load_auth_config()
    providers = data.setdefault("providers", {})
    provider_cfg = providers.setdefault(provider, {})
    provider_cfg["api_key"] = api_key
    if base_url:
        provider_cfg["base_url"] = base_url
    return save_auth_config(data)


def load_api_key(provider: str) -> str | None:
    """Load provider API key from user config."""
    data = load_auth_config()
    provider_cfg = data.get("providers", {}).get(provider, {})
    if not isinstance(provider_cfg, dict):
        return None
    value = provider_cfg.get("api_key")
    return value if isinstance(value, str) and value.strip() else None


def delete_api_key(provider: str) -> bool:
    """Delete provider API key from user config."""
    data = load_auth_config()
    providers = data.get("providers", {})
    if not isinstance(providers, dict) or provider not in providers:
        return False
    del providers[provider]
    save_auth_config(data)
    return True


def mask_api_key(api_key: str) -> str:
    """Return a masked key for display."""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    if api_key.startswith("sk-"):
        return f"sk-****{api_key[-4:]}"
    return f"{api_key[:2]}****{api_key[-4:]}"
