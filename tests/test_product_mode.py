from __future__ import annotations

import re
from pathlib import Path

from nightrunner import cli, git_ops, runner
from nightrunner.report import generate_summary_report
from nightrunner.state_store import append_experiment, save_best


def test_cli_supports_project_arg(monkeypatch) -> None:
    captured: dict[str, Path] = {}

    def _fake_init(project_root: Path, **kwargs):
        captured["project_root"] = project_root
        return {
            "project_root": str(project_root),
            "config": str(project_root / "nightrunner.yaml"),
            "nightrunner_dir": str(project_root / ".nightrunner"),
        }

    monkeypatch.setattr(cli, "init_project", _fake_init)
    code = cli.main(["init", "--project", "C:\\demo\\mlproj"])
    assert code == 0
    assert captured["project_root"] == Path("C:/demo/mlproj").resolve()


def test_init_writes_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "is_git_repo", lambda path: True)
    result = runner.init_project(
        tmp_path,
        editable_files=["train.py", "model.py"],
        train_command="python train.py --config cfg.yaml",
        metric_name="val_acc",
        lower_is_better=False,
    )
    assert Path(result["config"]).exists()
    cfg_text = (tmp_path / "nightrunner.yaml").read_text(encoding="utf-8")
    assert "model.py" in cfg_text
    assert "python train.py --config cfg.yaml" in cfg_text
    assert "val_acc" in cfg_text
    assert "lower_is_better: false" in cfg_text


def test_report_writes_two_summary_files(tmp_path: Path) -> None:
    append_experiment(
        tmp_path,
        {
            "id": "baseline",
            "status": "keep",
            "metric_value": 1.0,
            "hypothesis": "baseline",
        },
    )
    save_best(
        tmp_path,
        {
            "experiment_id": "baseline",
            "metric_name": "val_loss",
            "metric_value": 1.0,
            "patch_path": None,
        },
    )
    summary_path = generate_summary_report(tmp_path)
    assert summary_path.exists()
    assert (tmp_path / "nightrunner_summary.md").exists()


def test_create_worktree_branch_name_unique(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _fake_run_git(args: list[str], cwd: Path) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(git_ops, "run_git", _fake_run_git)
    monkeypatch.setattr(git_ops, "remove_worktree", lambda project_root, worktree_path: None)
    worktree = git_ops.create_worktree(tmp_path, "exp_0001")
    assert worktree.name == "exp_0001"
    add_call = calls[0]
    assert add_call[:3] == ["worktree", "add", str(worktree)]
    assert "-b" in add_call
    branch = add_call[add_call.index("-b") + 1]
    assert re.match(r"^nightrunner/exp_0001_\d{8}_\d{6}$", branch)
