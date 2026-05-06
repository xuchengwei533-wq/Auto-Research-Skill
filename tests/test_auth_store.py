from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nightrunner import auth_store


class AuthStoreTests(unittest.TestCase):
    def test_save_load_delete_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "APPDATA": tmp,
                "XDG_CONFIG_HOME": str(Path(tmp) / ".config"),
                "HOME": tmp,
            }
            with patch.dict(os.environ, env, clear=False):
                path = auth_store.save_api_key(
                    "deepseek",
                    "sk-test-1234abcd",
                    base_url="https://api.deepseek.com",
                )
                self.assertTrue(path.exists())
                self.assertEqual(auth_store.load_api_key("deepseek"), "sk-test-1234abcd")
                data = auth_store.load_auth_config()
                self.assertEqual(
                    data["providers"]["deepseek"]["base_url"],
                    "https://api.deepseek.com",
                )
                self.assertEqual(auth_store.mask_api_key("sk-test-1234abcd"), "sk-****abcd")
                self.assertTrue(auth_store.delete_api_key("deepseek"))
                self.assertIsNone(auth_store.load_api_key("deepseek"))


if __name__ == "__main__":
    unittest.main()
