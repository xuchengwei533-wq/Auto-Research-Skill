"""Configuration loading and defaults for NightRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import write_text

CONFIG_FILE_NAME = "nightrunner.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "Auto-Research-Skill"},
    "files": {
        "editable": ["train.py"],
        "protected": [
            "prepare.py",
            "program.md",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            ".env",
            ".env.local",
            "nightrunner.yaml",
        ],
    },
    "execution": {"train_command": "uv run train.py", "timeout_seconds": 900},
    "metric": {"name": "val_bpb", "lower_is_better": True},
    "agent": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-v4-pro",
        "reasoning_effort": "high",
        "thinking_enabled": True,
    },
    "safety": {
        "require_clean_git": True,
        "auto_apply_to_main": False,
        "allow_new_files": False,
        "allow_dependency_changes": False,
    },
    "logging": {"save_request": True, "save_response": True, "save_run_log": True},
}

NIGHTRUNNER_GITIGNORE_LINES = [
    "# NightRunner local state",
    ".nightrunner/worktrees/",
    ".nightrunner/tmp/",
    ".nightrunner/cache/",
    ".nightrunner/runs/*/run.log",
    ".nightrunner/runs/*/request.json",
    ".nightrunner/runs/*/response.json",
    ".nightrunner/config.local.yaml",
    "",
    "# Secrets",
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
]


def get_config_path(project_root: Path) -> Path:
    return project_root / CONFIG_FILE_NAME


def write_default_config_if_missing(project_root: Path) -> Path:
    path = get_config_path(project_root)
    if path.exists():
        return path
    write_text(path, yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False, allow_unicode=True))
    return path


def load_config(project_root: Path) -> dict[str, Any]:
    path = get_config_path(project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"在 {project_root} 中未找到 {CONFIG_FILE_NAME}。请先执行 `nightrunner init`。"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的配置格式无效，根节点必须是映射对象。")
    return raw


def save_config(project_root: Path, config: dict[str, Any]) -> Path:
    """Persist config to nightrunner.yaml."""
    path = get_config_path(project_root)
    write_text(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path
