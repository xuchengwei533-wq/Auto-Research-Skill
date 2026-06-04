"""Project diagnostics and setup helpers for NightRunner."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .auth_cli import auth_status
from .config import get_config_path, load_config
from .git_ops import is_git_repo, run_git
from .log_parser import detect_metric_candidates_from_text
from .setup_flow import scan_python_files


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
        "nightrunner_executable_path": sys.executable,
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
        "metric_name": config.get("metric", {}).get("name"),
        "metric_regex": config.get("metric", {}).get("regex"),
        "higher_is_better": not bool(config.get("metric", {}).get("lower_is_better", True)),
        "auth_ok": bool(auth.get("ok")),
        "auth_message": auth.get("message"),
        "backend": config.get("execution", {}).get("backend", "sandbox"),
        "state_dir": str(project_root / ".nightrunner"),
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
    return [
        {"name": str(item["name"]), "regex": str(item["regex"])}
        for item in detect_metric_candidates_from_text(text)
    ]


def format_doctor_report(info: dict[str, Any]) -> str:
    git_hint = ""
    if info.get("git_repository") and info.get("git_status") == "dirty":
        git_hint = (
            "Your Git working tree has uncommitted changes. This is okay in sandbox mode.\n"
            "NightRunner will copy your current files into isolated sandboxes."
        )
    elif not info.get("git_repository"):
        git_hint = (
            "Git is not detected. NightRunner can still run sandbox experiments, "
            "but diff/apply safety may be reduced."
        )
    direction = "higher is better" if info.get("higher_is_better") else "lower is better"
    lines = [
        "NightRunner Doctor",
        "",
        f"NightRunner executable path: {info.get('nightrunner_executable_path')}",
        f"NightRunner package path: {info.get('nightrunner_package_path')}",
        f"Project root: {info.get('project_root')}",
        f"Python executable: {info.get('python_executable')}",
        f"Conda environment: {info.get('conda_environment') or '-'}",
        f"Git repository: {'yes' if info.get('git_repository') else 'no'}",
        f"Git status: {info.get('git_status')}",
        f"Config file: {'found' if info.get('config_found') else 'missing'}",
        f"Config path: {info.get('config_path')}",
        f"Backend: {info.get('backend')}",
        f"Editable files: {', '.join(info.get('editable_files') or []) or '-'}",
        f"Train command: {info.get('train_command') or '-'}",
        f"Metric: {info.get('metric_name') or '-'} ({direction})",
        f"Auth status: {info.get('auth_message') or '-'}",
        f".nightrunner state dir: {info.get('state_dir')}",
    ]
    if git_hint:
        lines.extend(["", git_hint])
    return "\n".join(lines)
