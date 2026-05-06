from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from nightrunner.agent_api import resolve_api_settings
from nightrunner.auth_store import save_api_key
from nightrunner.cli import main


class AuthCliAndAgentTests(unittest.TestCase):
    def test_auth_status_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "APPDATA": tmp,
                "XDG_CONFIG_HOME": str(Path(tmp) / ".config"),
                "HOME": tmp,
            }
            with patch.dict(os.environ, env, clear=False):
                out = io.StringIO()
                with redirect_stdout(out):
                    exit_code = main(["auth", "status"])
                self.assertEqual(exit_code, 1)
                self.assertIn("No API key found", out.getvalue())

    def test_environment_variable_takes_priority_over_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "APPDATA": tmp,
                "XDG_CONFIG_HOME": str(Path(tmp) / ".config"),
                "HOME": tmp,
            }
            with patch.dict(os.environ, env, clear=False):
                save_api_key("deepseek", "sk-config-1234abcd", base_url="https://saved.example.com")
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-env-9876wxyz"}, clear=False):
                    api_key, base_url = resolve_api_settings(
                        api_key_env="DEEPSEEK_API_KEY",
                        provider="deepseek",
                        base_url="https://api.deepseek.com",
                    )
                self.assertEqual(api_key, "sk-env-9876wxyz")
                self.assertEqual(base_url, "https://api.deepseek.com")

    def test_user_config_is_used_when_environment_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "APPDATA": tmp,
                "XDG_CONFIG_HOME": str(Path(tmp) / ".config"),
                "HOME": tmp,
            }
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("DEEPSEEK_API_KEY", None)
                save_api_key("deepseek", "sk-config-1234abcd", base_url="https://saved.example.com")
                api_key, base_url = resolve_api_settings(
                    api_key_env="DEEPSEEK_API_KEY",
                    provider="deepseek",
                    base_url="https://api.deepseek.com",
                )
                self.assertEqual(api_key, "sk-config-1234abcd")
                self.assertEqual(base_url, "https://saved.example.com")


if __name__ == "__main__":
    unittest.main()
