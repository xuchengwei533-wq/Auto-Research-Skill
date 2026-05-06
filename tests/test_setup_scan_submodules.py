from __future__ import annotations

from pathlib import Path

import pytest

from nightrunner import runner
from nightrunner.config import build_default_config, save_config


def _fake_init_project(
    project_root: Path,
    editable_files=None,
    train_command="python train.py",
    metric_name="val_loss",
    lower_is_better=True,
    update_gitignore=True,
):
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


def test_scan_python_files_skips_submodule_and_keeps_normal_files(tmp_path: Path) -> None:
    submodule = tmp_path / "submodule"
    submodule.mkdir()
    (submodule / ".git").write_text("gitdir: ../.git/modules/submodule\n", encoding="utf-8")
    (submodule / "main.py").write_text("print('submodule')\n", encoding="utf-8")

    normal = tmp_path / "examples" / "mnist"
    normal.mkdir(parents=True)
    (normal / "main.py").write_text("print('normal')\n", encoding="utf-8")

    files = runner._scan_python_files(tmp_path)

    assert "submodule/main.py" not in files
    assert "examples/mnist/main.py" in files


def test_setup_rejects_manual_editable_inside_submodule(monkeypatch, tmp_path: Path) -> None:
    submodule = tmp_path / "submodule"
    submodule.mkdir()
    (submodule / ".git").write_text("gitdir: ../.git/modules/submodule\n", encoding="utf-8")
    (submodule / "main.py").write_text("print('submodule')\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_is_git_available", lambda: True)
    monkeypatch.setattr(runner, "is_git_repo", lambda path: True)
    monkeypatch.setattr(runner, "init_project", _fake_init_project)
    monkeypatch.setattr(
        runner,
        "check_auth",
        lambda project_root=None: {"ok": True, "message": "ok", "source": "environment"},
    )

    with pytest.raises(RuntimeError) as excinfo:
        runner.setup(
            tmp_path,
            editable_files=["submodule/main.py"],
            train_command="python train.py",
            metric_name="val_loss",
            lower_is_better=True,
            yes=True,
        )

    assert (
        "This file is inside a Git submodule or nested Git repository. "
        "Run NightRunner inside that repository instead."
    ) in str(excinfo.value)
