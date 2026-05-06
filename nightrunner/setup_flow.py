from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    NIGHTRUNNER_GITIGNORE_LINES,
    load_config,
    save_config,
    write_default_config_if_missing,
)
from .git_ops import is_git_repo
from .project_context import build_project_context, ensure_project_layout
from .utils import now_iso, write_json, write_text


def is_git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0
    except OSError:
        return False


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{prompt} {suffix}\n> ").strip().lower()
    if not raw:
        return default_yes
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return default_yes


def update_gitignore(project_root: Path) -> None:
    path = project_root / ".gitignore"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = current.splitlines()
    existing = set(lines)
    to_add = [line for line in NIGHTRUNNER_GITIGNORE_LINES if line not in existing]
    if to_add:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(to_add)
        write_text(path, "\n".join(lines) + "\n")


def _has_git_repo_marker(path: Path) -> bool:
    marker = path / ".git"
    return marker.is_file() or marker.is_dir()


def _is_nested_git_repo_dir(project_root: Path, path: Path) -> bool:
    if path == project_root:
        return False
    return _has_git_repo_marker(path)


def _is_inside_nested_git_repo(project_root: Path, relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").strip("/")
    if not normalized:
        return False
    current = project_root
    parts = [part for part in Path(normalized).parts if part not in {"", "."}]
    if not parts:
        return False
    for part in parts[:-1]:
        current = current / part
        if _is_nested_git_repo_dir(project_root, current):
            return True
    return False


def validate_editable_paths(project_root: Path, editable_files: list[str]) -> None:
    for path in editable_files:
        normalized = path.replace("\\", "/").rstrip("/")
        if _is_inside_nested_git_repo(project_root, normalized):
            raise RuntimeError(
                "This file is inside a Git submodule or nested Git repository. "
                "Run NightRunner inside that repository instead."
            )


def scan_python_files(project_root: Path) -> list[str]:
    ignore_dirs = {
        ".venv",
        "venv",
        ".git",
        ".nightrunner",
        "__pycache__",
        "site-packages",
        "build",
        "dist",
    }
    all_files: list[str] = []
    for current_root, dirnames, filenames in os.walk(project_root):
        current_path = Path(current_root)
        if current_path != project_root and _has_git_repo_marker(current_path):
            dirnames[:] = []
            continue
        kept_dirnames: list[str] = []
        for name in dirnames:
            if name in ignore_dirs:
                continue
            child = current_path / name
            if _has_git_repo_marker(child):
                continue
            kept_dirnames.append(name)
        dirnames[:] = kept_dirnames
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            file_path = current_path / filename
            rel = file_path.relative_to(project_root)
            if any(part in ignore_dirs for part in rel.parts):
                continue
            all_files.append(rel.as_posix())

    def _priority(path: str) -> tuple[int, str]:
        if path == "train.py":
            return (0, path)
        if path == "main.py":
            return (1, path)
        if path == "model.py":
            return (2, path)
        if path.startswith("src/"):
            return (3, path)
        if "/" not in path:
            return (4, path)
        return (5, path)

    return sorted(all_files, key=_priority)


def _parse_editable_selection(raw: str, candidates: list[str]) -> list[str]:
    if not raw.strip():
        return [candidates[0]] if candidates else ["train.py"]
    chosen: list[str] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if token.isdigit():
            idx = int(token)
            if idx < 1 or idx > len(candidates):
                raise ValueError(f"Selection index out of range: {token}")
            chosen.append(candidates[idx - 1])
        else:
            normalized = token.replace("\\", "/")
            if normalized in candidates:
                chosen.append(normalized)
            else:
                chosen.append(normalized)
    deduped: list[str] = []
    for path in chosen:
        if path not in deduped:
            deduped.append(path)
    return deduped or (["train.py"] if not candidates else [candidates[0]])


def _detect_train_command_default(candidates: list[str]) -> str:
    if "train.py" in candidates:
        return "python train.py"
    if "main.py" in candidates:
        return "python main.py"
    if candidates:
        return f"python {candidates[0]}"
    return "python train.py"


@dataclass
class SetupOptions:
    editable_files: list[str] | None = None
    train_command: str | None = None
    metric_name: str | None = None
    lower_is_better: bool | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    model: str | None = None
    run_baseline_now: bool = False
    yes: bool = False
    metric_regex: str | None = None


@dataclass
class SetupResult:
    project_root: str
    config: str
    nightrunner_dir: str
    editable_files: list[str]
    train_command: str
    metric_name: str
    lower_is_better: bool
    metric_regex: str | None


def init_project(
    project_root: Path,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
    update_gitignore_enabled: bool = True,
) -> dict[str, Any]:
    if not is_git_repo(project_root):
        raise RuntimeError(
            "NightRunner requires a Git repository for safe worktree isolation.\n"
            "Please run:\n"
            "git init\n"
            "git add .\n"
            "git commit -m \"Initial commit\""
        )

    ctx = build_project_context(project_root)
    ensure_project_layout(ctx)
    cfg_path = write_default_config_if_missing(
        project_root,
        editable_files=editable_files,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
    )

    best_path = ctx.state_dir / "best.json"
    if not best_path.exists():
        write_json(best_path, {})
    exp_path = ctx.state_dir / "experiments.jsonl"
    if not exp_path.exists():
        write_text(exp_path, "")
    project_meta = ctx.state_dir / "project.json"
    if not project_meta.exists():
        write_json(
            project_meta,
            {
                "project_root": str(project_root.resolve()),
                "created_at": now_iso(),
                "nightrunner_version": __version__,
            },
        )
    if update_gitignore_enabled:
        update_gitignore(project_root)
    return {
        "project_root": str(project_root.resolve()),
        "config": str(cfg_path),
        "nightrunner_dir": str(ctx.nightrunner_dir),
    }


def run_setup(project_root: Path, options: SetupOptions) -> dict[str, Any]:
    candidates = scan_python_files(project_root)

    selected_editable = options.editable_files
    if selected_editable is None:
        if options.yes:
            selected_editable = [candidates[0]] if candidates else ["train.py"]
        else:
            print("")
            print("Detected Python files:")
            for idx, path in enumerate(candidates, start=1):
                print(f"{idx:>3}. {path}")
            print("")
            print("Choose editable files (comma-separated indices or paths).")
            print("Press Enter to use first detected file.")
            raw = input("\nEditable files:\n> ")
            selected_editable = _parse_editable_selection(raw, candidates)
    selected_editable = [x.replace("\\", "/") for x in (selected_editable or ["train.py"])]
    validate_editable_paths(project_root, selected_editable)

    if options.train_command is None:
        train_default = _detect_train_command_default(selected_editable or candidates)
        if options.yes:
            train_command = train_default
        else:
            train_command = input(f"\nTrain command [{train_default}]:\n> ").strip() or train_default
    else:
        train_command = options.train_command

    metric_name = options.metric_name
    if metric_name is None:
        if options.yes:
            metric_name = "val_loss"
        else:
            print("")
            metric_name = input("Metric name [val_loss]:\n> ").strip() or "val_loss"

    lower_is_better = options.lower_is_better
    if lower_is_better is None:
        lower_is_better = True if options.yes else prompt_yes_no(
            "Is lower better for this metric?",
            default_yes=True,
        )

    metric_regex_value = options.metric_regex
    if metric_regex_value is None and options.yes:
        metric_regex_value = ""
    elif metric_regex_value is None:
        print("")
        print("Optional metric regex.")
        print('Leave empty to parse formats like "val_loss: 0.123".')
        print("Example: Average loss:\\s*([0-9.]+)")
        metric_regex_value = input("\nMetric regex:\n> ").strip()

    api_key_env = options.api_key_env
    if api_key_env is None:
        api_key_env = "DEEPSEEK_API_KEY"
    if not api_key_env:
        api_key_env = "DEEPSEEK_API_KEY"

    should_update_gitignore = True if options.yes else prompt_yes_no(
        "Add NightRunner runtime artifacts to .gitignore?",
        default_yes=True,
    )

    from . import runner as runner_module

    init_result = runner_module.init_project(
        project_root=project_root,
        editable_files=selected_editable,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=bool(lower_is_better),
        update_gitignore=should_update_gitignore,
    )

    cfg = load_config(project_root)
    cfg.setdefault("agent", {})["api_key_env"] = api_key_env
    if options.base_url:
        cfg.setdefault("agent", {})["base_url"] = options.base_url
    if options.model:
        cfg.setdefault("agent", {})["model"] = options.model
    if metric_regex_value:
        cfg.setdefault("metric", {})["regex"] = metric_regex_value
    else:
        cfg.setdefault("metric", {}).pop("regex", None)
    save_config(project_root, cfg)

    result = SetupResult(
        project_root=init_result["project_root"],
        config=init_result["config"],
        nightrunner_dir=init_result["nightrunner_dir"],
        editable_files=selected_editable,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=bool(lower_is_better),
        metric_regex=metric_regex_value or None,
    )
    return asdict(result)
