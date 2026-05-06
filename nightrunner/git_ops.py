"""Git operations via subprocess."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .utils import ensure_dir


class GitError(RuntimeError):
    """Raised when a git command fails."""


class WorktreeSetupError(RuntimeError):
    """Raised when git worktree setup fails."""


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
        raise RuntimeError(
            "Current directory is not a Git repository.\n"
            "Run:\n"
            "git init\n"
            "git add .\n"
            "git commit -m \"Initial commit\""
        )


def is_worktree_clean(path: Path) -> bool:
    out = run_git(["status", "--porcelain"], path)
    return out.strip() == ""


def ensure_clean_worktree(path: Path) -> None:
    if not is_worktree_clean(path):
        raise RuntimeError(
            "Working tree is not clean.\n"
            "Run one of:\n"
            "git add .\n"
            "git commit -m \"Save current work before NightRunner\"\n"
            "or:\n"
            "git stash push -u -m \"before nightrunner\""
        )


def get_worktrees_root(project_root: Path) -> Path:
    """Return the user cache root for isolated worktrees."""
    project_hash = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()[:16]
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return ensure_dir(base / "nightrunner" / "worktrees" / project_hash)


def _list_worktrees(project_root: Path) -> list[dict[str, str]]:
    out = run_git(["worktree", "list", "--porcelain"], project_root)
    blocks = [block for block in out.split("\n\n") if block.strip()]
    items: list[dict[str, str]] = []
    for block in blocks:
        item: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("worktree "):
                item["worktree"] = line.split(" ", 1)[1].strip()
            elif line.startswith("branch "):
                branch_ref = line.split(" ", 1)[1].strip()
                item["branch"] = branch_ref.removeprefix("refs/heads/")
        if item:
            items.append(item)
    return items


def _find_worktree_branch(project_root: Path, worktree_path: Path) -> str | None:
    target = str(worktree_path.resolve())
    for item in _list_worktrees(project_root):
        candidate = item.get("worktree")
        if candidate and Path(candidate).resolve() == Path(target):
            branch = item.get("branch")
            return branch if branch else None
    return None


def _delete_branch_if_temporary(project_root: Path, branch_name: str | None) -> bool:
    if not branch_name or not branch_name.startswith("nightrunner/"):
        return False
    run_git(["branch", "-D", branch_name], project_root)
    return True


def list_temporary_branches(project_root: Path) -> list[str]:
    """List leftover NightRunner branches."""
    out = run_git(["branch", "--list", "nightrunner/*", "--format=%(refname:short)"], project_root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def clean_temporary_branches(project_root: Path) -> list[str]:
    """Delete leftover NightRunner branches."""
    removed: list[str] = []
    for branch_name in list_temporary_branches(project_root):
        run_git(["branch", "-D", branch_name], project_root)
        removed.append(branch_name)
    return removed


def create_worktree(project_root: Path, exp_id: str) -> tuple[Path, str]:
    worktrees_root = get_worktrees_root(project_root)
    worktree_path = worktrees_root / exp_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"nightrunner/{exp_id}_{timestamp}"

    if worktree_path.exists():
        remove_worktree(project_root, worktree_path)

    try:
        run_git(["worktree", "add", str(worktree_path), "-b", branch_name, "HEAD"], project_root)
        return worktree_path, branch_name
    except GitError as exc:
        # Branch may already exist from previous failed cleanup.
        unique_branch = f"{branch_name}_{int(time.time())}"
        try:
            run_git(
                ["worktree", "add", str(worktree_path), "-b", unique_branch, "HEAD"],
                project_root,
            )
            return worktree_path, unique_branch
        except GitError:
            raise WorktreeSetupError(f"Failed to create worktree for {exp_id}: {exc}") from exc


def remove_worktree(project_root: Path, worktree_path: Path, branch_name: str | None = None) -> bool:
    """Remove a worktree, force if normal remove fails."""
    if not worktree_path.exists():
        if branch_name:
            return _delete_branch_if_temporary(project_root, branch_name)
        return False
    branch_to_delete = branch_name or _find_worktree_branch(project_root, worktree_path)
    try:
        run_git(["worktree", "remove", str(worktree_path)], project_root)
    except GitError:
        run_git(["worktree", "remove", "--force", str(worktree_path)], project_root)
    return _delete_branch_if_temporary(project_root, branch_to_delete)


def save_diff(worktree_path: Path, output_path: Path) -> None:
    diff = run_git(["diff"], worktree_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diff, encoding="utf-8")


def apply_patch_to_project(project_root: Path, patch_path: Path) -> None:
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    ensure_clean_worktree(project_root)
    try:
        run_git(["apply", "--check", str(patch_path)], project_root)
        run_git(["apply", str(patch_path)], project_root)
    except GitError as exc:
        raise RuntimeError(
            "Patch could not be applied cleanly.\n"
            "Review your working tree with `git status` and `git diff`, then try again.\n"
            f"Details: {exc}"
        ) from exc
