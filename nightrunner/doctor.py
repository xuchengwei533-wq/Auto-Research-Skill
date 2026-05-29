"""Project diagnostics and setup helpers for NightRunner."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from .auth_cli import auth_status
from .config import get_config_path, load_config
from .git_ops import is_git_repo, run_git
from .setup_flow import scan_python_files

COMMON_METRIC_PATTERNS: dict[str, str] = {
    "accuracy": r"accuracy\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    "loss": r"loss\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    "val_loss": r"val[_ ]loss\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    "rmse": r"rmse\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    "mae": r"mae\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    "f1": r"f1\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    "auc": r"auc\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
}


def collect_doctor_info(project_root: Path) -> dict[str, Any]:
    config_path = get_config_path(project_root)
    config_exists = config_path.exists()
    config = load_config(project_root) if config_exists else {}
    git_repo = is_git_repo(project_root)
    git_status = "not_repo"
    if git_repo:
        try:
            git_status = "clean" if run_git(["status", "--porcelain"], project_root).strip() == "" else "dirty"
        except Exception:
            git_status = "unknown"

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    auth = auth_status()
    return {
        "project_root": str(project_root),
        "python_executable": sys.executable,
        "conda_environment": conda_env,
        "conda_prefix": conda_prefix,
        "nightrunner_package_path": str(Path(__file__).resolve().parent),
        "git_repository": git_repo,
        "git_status": git_status,
        "config_found": config_exists,
        "config_path": str(config_path),
        "train_command": config.get("execution", {}).get("train_command"),
        "editable_files": config.get("files", {}).get("editable", []),
        "auth_ok": bool(auth.get("ok")),
        "auth_message": auth.get("message"),
        "backend": config.get("execution", {}).get("backend", "sandbox"),
        "candidate_files": suggest_editable_files(project_root),
    }


def suggest_editable_files(project_root: Path) -> list[str]:
    candidates = scan_python_files(project_root)
    preferred: list[str] = []
    for name in ("train.py", "main.py", "model.py", "config.py"):
        preferred.extend([path for path in candidates if path.endswith(name)])
    seen: set[str] = set()
    ordered: list[str] = []
    for path in [*preferred, *candidates]:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered[:20]


def detect_metrics_from_text(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    lowered = text.lower()
    for name, pattern in COMMON_METRIC_PATTERNS.items():
        if re.search(pattern, lowered, flags=re.MULTILINE):
            found.append({"name": name, "regex": pattern})
    return found
