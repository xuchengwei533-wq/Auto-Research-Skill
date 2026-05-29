from __future__ import annotations

from pathlib import Path

import pytest

from nightrunner import runner, setup_flow
from nightrunner.config import build_default_config, load_config, save_config


def _fake_init_project(
    project_root: Path,
    editable_files=None,
    train_command="python train.py",
    metric_name="val_loss",
    lower_is_better=True,
    update_gitignore=False,
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
    if update_gitignore:
        setup_flow.update_gitignore(project_root)
    elif (project_root / ".git").exists():
        setup_flow.update_git_exclude(project_root)
    return {
        "project_root": str(project_root),
        "config": str(project_root / "nightrunner.yaml"),
        "nightrunner_dir": str(project_root / ".nightrunner"),
    }


def test_scan_python_files_respects_ignore_and_nested_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "examples" / "mnist").mkdir(parents=True)
    (tmp_path / "examples" / "mnist" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".venv" / "a.py").parent.mkdir(parents=True)
    (tmp_path / ".venv" / "a.py").write_text("print('skip')\n", encoding="utf-8")
    (tmp_path / "nested" / ".git").mkdir(parents=True)
    (tmp_path / "nested" / "main.py").write_text("print('skip nested')\n", encoding="utf-8")
    (tmp_path / "submodule").mkdir()
    (tmp_path / "submodule" / ".git").write_text("gitdir: ../.git/modules/submodule\n", encoding="utf-8")
    (tmp_path / "submodule" / "main.py").write_text("print('skip submodule')\n", encoding="utf-8")

    files = setup_flow.scan_python_files(tmp_path)
    assert "examples/mnist/main.py" in files
    assert ".venv/a.py" not in files
    assert "nested/main.py" not in files
    assert "submodule/main.py" not in files


def test_validate_editable_paths_accepts_normal_and_rejects_nested(tmp_path: Path) -> None:
    (tmp_path / "examples" / "mnist").mkdir(parents=True)
    (tmp_path / "examples" / "mnist" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    setup_flow.validate_editable_paths(tmp_path, ["examples/mnist/main.py"])

    (tmp_path / "nested" / ".git").mkdir(parents=True)
    with pytest.raises(RuntimeError) as excinfo:
        setup_flow.validate_editable_paths(tmp_path, ["nested/main.py"])
    assert "inside a Git submodule or nested Git repository" in str(excinfo.value)


def test_run_setup_yes_mode_generates_config_without_api_key(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "examples" / "mnist").mkdir(parents=True)
    (tmp_path / "examples" / "mnist" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".git" / "info").mkdir(parents=True)
    monkeypatch.setattr(runner, "init_project", _fake_init_project)
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(AssertionError("input() should not be called")))

    result = setup_flow.run_setup(
        tmp_path,
        setup_flow.SetupOptions(
            editable_files=["examples/mnist/main.py"],
            train_command="python examples/mnist/main.py",
            metric_name="Average loss",
            lower_is_better=True,
            api_key_env="DEEPSEEK_API_KEY",
            yes=True,
        ),
    )

    assert result["config"].endswith("nightrunner.yaml")
    cfg = load_config(tmp_path)
    serialized = (tmp_path / "nightrunner.yaml").read_text(encoding="utf-8")
    assert "sk-" not in serialized
    assert "api_key:" not in serialized
    assert cfg.get("agent", {}).get("api_key_env") == "DEEPSEEK_API_KEY"

    assert not (tmp_path / ".gitignore").exists()
    exclude = (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert ".nightrunner/" in exclude
    assert "nightrunner_summary.md" in exclude
    assert "nightrunner.yaml" not in exclude
