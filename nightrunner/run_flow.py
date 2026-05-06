from __future__ import annotations

from pathlib import Path


def run_baseline(project_root: Path, force: bool = False):
    from . import runner as runner_module

    return runner_module.run_baseline(project_root, force=force)


def run_night(project_root: Path, rounds: int = 1, dry_run: bool = False, plain: bool = False):
    from . import runner as runner_module

    return runner_module.run_night(project_root, rounds=rounds, dry_run=dry_run, plain=plain)


def run(project_root: Path, rounds: int = 1, dry_run: bool = False, plain: bool = False):
    from . import runner as runner_module

    return runner_module.run(project_root, rounds=rounds, dry_run=dry_run, plain=plain)
