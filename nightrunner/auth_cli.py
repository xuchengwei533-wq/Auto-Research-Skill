from __future__ import annotations

import getpass
import os

from .auth_store import (
    delete_api_key,
    get_config_path as get_auth_config_path,
    load_api_key,
    mask_api_key,
    save_api_key,
)


def auth_status() -> dict[str, str | bool | None]:
    env_name = "DEEPSEEK_API_KEY"
    env_value = os.environ.get(env_name)
    if env_value:
        return {
            "ok": True,
            "message": f"{env_name} is set in environment ({mask_api_key(env_value)}).",
            "source": "environment",
            "masked_key": mask_api_key(env_value),
        }
    stored_key = load_api_key("deepseek")
    if stored_key:
        return {
            "ok": True,
            "message": f"DeepSeek API key is configured in user config ({mask_api_key(stored_key)}).",
            "source": "user_config",
            "masked_key": mask_api_key(stored_key),
        }
    return {
        "ok": False,
        "message": "No API key found. Run `nightrunner auth login` or set DEEPSEEK_API_KEY.",
        "source": "missing",
        "masked_key": None,
    }


def handle_auth_command(auth_command: str | None) -> int:
    resolved = auth_command or "status"
    if resolved == "login":
        api_key = getpass.getpass("Paste DeepSeek API key: ").strip()
        if not api_key:
            raise RuntimeError("No API key entered.")
        save_api_key("deepseek", api_key, base_url="https://api.deepseek.com")
        print(f"DeepSeek API key saved to {get_auth_config_path()}")
        return 0
    if resolved == "logout":
        deleted = delete_api_key("deepseek")
        if deleted:
            print("DeepSeek API key removed from user config.")
        else:
            print("No saved DeepSeek API key found.")
        return 0

    result = auth_status()
    print(result["message"])
    return 0 if result["ok"] else 1
