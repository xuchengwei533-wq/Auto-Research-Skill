"""Patch safety guard for changed files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .git_ops import run_git

DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "poetry.lock",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}


def _norm(path: str | Path) -> str:
    raw = str(path).strip().replace("\\", "/")
    if raw.endswith("/") and raw != "/":
        # Keep explicit directory rule semantics.
        return raw
    return Path(raw).as_posix()


def _matches_rule(path: str, rule: str) -> bool:
    rule_norm = _norm(rule)
    if rule_norm.endswith("/"):
        return path.startswith(rule_norm)
    return path == rule_norm


def get_changed_files(worktree_path: Path) -> list[str]:
    """Return changed tracked files from git diff."""
    out = run_git(["diff", "--name-only"], worktree_path)
    return [Path(line.strip()).as_posix() for line in out.splitlines() if line.strip()]


def get_new_files(worktree_path: Path) -> list[str]:
    """Return newly added files from git status porcelain."""
    out = run_git(["status", "--porcelain"], worktree_path)
    files: list[str] = []
    for line in out.splitlines():
        if line.startswith("?? "):
            files.append(Path(line[3:].strip()).as_posix())
        elif line.startswith("A "):
            files.append(Path(line[2:].strip()).as_posix())
        elif line.startswith("A  "):
            files.append(Path(line[3:].strip()).as_posix())
    return files


def validate_changed_files(
    changed_files: list[str],
    editable_files: list[str],
    protected_files: list[str],
    allow_new_files: bool,
    allow_dependency_changes: bool,
    new_files: list[str] | None = None,
) -> dict[str, Any]:
    """Validate changed files against editable/protected/dependency rules."""
    editable_norm = [_norm(x) for x in editable_files]
    protected_norm = [_norm(x) for x in protected_files]
    changed_norm = [_norm(x) for x in changed_files]
    new_norm = [_norm(x) for x in (new_files or [])]

    violations: list[dict[str, str]] = []
    for changed in changed_norm:
        # Protected has highest priority.
        if any(_matches_rule(changed, p) for p in protected_norm):
            violations.append({"type": "protected_file", "file": changed})
            continue
        if not any(_matches_rule(changed, e) for e in editable_norm):
            violations.append({"type": "not_editable", "file": changed})
            continue
        if (not allow_dependency_changes) and Path(changed).name in DEPENDENCY_FILES:
            violations.append({"type": "dependency_change", "file": changed})

    if not allow_new_files:
        for new_file in new_norm:
            violations.append({"type": "new_file", "file": new_file})

    return {"ok": len(violations) == 0, "violations": violations}
