"""Configuration loading and defaults for NightRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import write_text

CONFIG_FILE_NAME = "nightrunner.yaml"

def build_default_config(
    project_name: str,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
) -> dict[str, Any]:
    return {
        "project": {"name": project_name},
        "files": {
            "editable": editable_files or ["train.py"],
            "protected": [
                ".env",
                ".env.local",
                "pyproject.toml",
                "requirements.txt",
                "uv.lock",
                "README.md",
                "nightrunner.yaml",
            ],
        },
        "execution": {"train_command": train_command, "timeout_seconds": 3600},
        "metric": {"name": metric_name, "lower_is_better": lower_is_better},
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


DEFAULT_CONFIG: dict[str, Any] = build_default_config("Auto-Research-Skill")

NIGHTRUNNER_GITIGNORE_LINES = [
    "# NightRunner local state",
    ".nightrunner/",
    "nightrunner_summary.md",
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


def write_default_config_if_missing(
    project_root: Path,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
) -> Path:
    path = get_config_path(project_root)
    if path.exists():
        return path
    default = build_default_config(
        project_name=project_root.name,
        editable_files=editable_files,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
    )
    write_text(path, yaml.safe_dump(default, sort_keys=False, allow_unicode=True))
    return path


def load_config(project_root: Path) -> dict[str, Any]:
    path = get_config_path(project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"{CONFIG_FILE_NAME} not found in {project_root}. Run `nightrunner init` first."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config format in {path}. Root must be a mapping.")
    metric = raw.get("metric", {})
    if isinstance(metric, dict) and "regex" in metric and metric["regex"] is not None:
        if not isinstance(metric["regex"], str):
            raise ValueError("metric.regex must be a string when provided.")
    return raw


def save_config(project_root: Path, config: dict[str, Any]) -> Path:
    """Persist config to nightrunner.yaml."""
    path = get_config_path(project_root)
    write_text(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path
