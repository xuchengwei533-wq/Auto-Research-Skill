from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .git_ops import get_worktrees_root
from .utils import ensure_dir


@dataclass(frozen=True)
class ProjectContext:
    root: Path
    config_path: Path
    nightrunner_dir: Path
    state_dir: Path
    runs_dir: Path
    experiments_dir: Path
    sandboxes_root: Path
    cache_dir: Path
    tmp_dir: Path
    worktrees_root: Path


def build_project_context(project_root: Path) -> ProjectContext:
    root = project_root.resolve()
    nightrunner_dir = root / ".nightrunner"
    return ProjectContext(
        root=root,
        config_path=root / "nightrunner.yaml",
        nightrunner_dir=nightrunner_dir,
        state_dir=nightrunner_dir / "state",
        runs_dir=nightrunner_dir / "runs",
        experiments_dir=nightrunner_dir / "experiments",
        sandboxes_root=nightrunner_dir / "sandboxes",
        cache_dir=nightrunner_dir / "cache",
        tmp_dir=nightrunner_dir / "tmp",
        worktrees_root=get_worktrees_root(root),
    )


def ensure_project_layout(ctx: ProjectContext) -> None:
    ensure_dir(ctx.state_dir)
    ensure_dir(ctx.runs_dir)
    ensure_dir(ctx.experiments_dir)
    ensure_dir(ctx.sandboxes_root)
    ensure_dir(ctx.cache_dir)
    ensure_dir(ctx.tmp_dir)
    ensure_dir(ctx.worktrees_root)
