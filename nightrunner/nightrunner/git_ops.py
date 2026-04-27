"""Git operations via subprocess."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .utils import ensure_dir


class GitError(RuntimeError):
    """Raised when a git command fails."""


def run_git(args: list[str], cwd: Path) -> str:
    """Run git command and return stdout, raising GitError on failure."""
    cmd = ["git", *args]
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise GitError(
            f"Git command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def is_git_repo(path: Path) -> bool:
    try:
        run_git(["rev-parse", "--is-inside-work-tree"], path)
        return True
    except GitError:
        return False


def ensure_git_repo(path: Path) -> None:
    if not is_git_repo(path):
        raise RuntimeError(f"{path} is not a Git repository.")


def is_worktree_clean(path: Path) -> bool:
    out = run_git(["status", "--porcelain"], path)
    return out.strip() == ""


def ensure_clean_worktree(path: Path) -> None:
    if not is_worktree_clean(path):
        raise RuntimeError(
            "Working tree is not clean. Commit/stash your changes or set safety.require_clean_git=false."
        )


def create_worktree(project_root: Path, exp_id: str) -> Path:
    worktrees_root = ensure_dir(project_root / ".nightrunner" / "worktrees")
    worktree_path = worktrees_root / exp_id
    branch_name = f"nightrunner/{exp_id}"

    if worktree_path.exists():
        remove_worktree(project_root, worktree_path)

    try:
        run_git(["worktree", "add", str(worktree_path), "-b", branch_name, "HEAD"], project_root)
        return worktree_path
    except GitError as exc:
        # Branch may already exist from previous failed cleanup.
        unique_branch = f"{branch_name}-{int(time.time())}"
        try:
            run_git(
                ["worktree", "add", str(worktree_path), "-b", unique_branch, "HEAD"],
                project_root,
            )
            return worktree_path
        except GitError:
            raise exc


def remove_worktree(project_root: Path, worktree_path: Path) -> None:
    """Remove a worktree, force if normal remove fails."""
    if not worktree_path.exists():
        return
    try:
        run_git(["worktree", "remove", str(worktree_path)], project_root)
    except GitError:
        run_git(["worktree", "remove", "--force", str(worktree_path)], project_root)


def save_diff(worktree_path: Path, output_path: Path) -> None:
    diff = run_git(["diff"], worktree_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diff, encoding="utf-8")


def apply_patch_to_project(project_root: Path, patch_path: Path) -> None:
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    run_git(["apply", str(patch_path)], project_root)
