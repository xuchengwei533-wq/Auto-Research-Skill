from __future__ import annotations

from pathlib import Path

import pytest

from nightrunner import runner
from nightrunner.config import build_default_config, save_config


def _fake_init_project(project_root: Path, editable_files=None, train_command="python train.py", metric_name="val_loss", lower_is_better=True, update_gitignore=True):
    (project_root / ".nightrunner" / "state").mkdir(parents=True, exist_ok=True)
    save_config(
        project_root,
        build_default_config(
            project_name=project_root.name,
            editable_files=editable_files or ["train.py"],
            train_command=train_command,
            metric_name=metric_name,
            lower_is_better=lower_is_better,
        ),
    )
    return {
        "project_root": str(project_root),
        "config": str(project_root / "nightrunner.yaml"),
        "nightrunner_dir": str(project_root / ".nightrunner"),
    }


def test_setup_does_not_run_baseline_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "_is_git_available", lambda: True)
    monkeypatch.setattr(runner, "is_git_repo", lambda path: True)
    monkeypatch.setattr(runner, "init_project", _fake_init_project)
    monkeypatch.setattr(runner, "check_auth", lambda project_root=None: {"ok": True, "message": "ok", "source": "environment"})

    def fail_run_baseline(project_root: Path, force: bool = False) -> Path:
        raise AssertionError("run_baseline should not be called by default setup")

    monkeypatch.setattr(runner, "run_baseline", fail_run_baseline)

    result = runner.setup(
        tmp_path,
        editable_files=["train.py"],
        train_command="python train.py",
        metric_name="val_loss",
        lower_is_better=True,
        yes=True,
    )

    assert result["config"].endswith("nightrunner.yaml")
    assert (tmp_path / "nightrunner.yaml").exists()


def test_setup_explicit_baseline_with_dirty_tree_shows_specific_guidance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "_is_git_available", lambda: True)
    monkeypatch.setattr(runner, "is_git_repo", lambda path: True)
    monkeypatch.setattr(runner, "init_project", _fake_init_project)
    monkeypatch.setattr(runner, "check_auth", lambda project_root=None: {"ok": True, "message": "ok", "source": "environment"})
    monkeypatch.setattr(
        runner,
        "run_git",
        lambda args, cwd: " M .gitignore\n?? nightrunner.yaml\n" if args == ["status", "--porcelain"] else "",
    )

    def fail_run_baseline(project_root: Path, force: bool = False) -> Path:
        raise AssertionError("run_baseline should not be called when setup dirtied the tree")

    monkeypatch.setattr(runner, "run_baseline", fail_run_baseline)

    with pytest.raises(RuntimeError) as excinfo:
        runner.setup(
            tmp_path,
            editable_files=["train.py"],
            train_command="python train.py",
            metric_name="val_loss",
            lower_is_better=True,
            yes=True,
            run_baseline_now=True,
        )

    message = str(excinfo.value)
    assert "NightRunner setup created or modified project config files." in message
    assert "git add nightrunner.yaml .gitignore" in message
    assert "nightrunner baseline" in message


def test_update_gitignore_does_not_ignore_nightrunner_yaml(tmp_path: Path) -> None:
    runner._update_gitignore(tmp_path)
    text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".nightrunner/" in text
    assert "nightrunner_summary.md" in text
    assert "nightrunner.yaml" not in text
