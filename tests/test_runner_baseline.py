from __future__ import annotations

from pathlib import Path

import pytest

from nightrunner import runner


def test_run_baseline_writes_best(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True)
    saved: list[dict] = []

    monkeypatch.setattr(runner, "ensure_git_repo", lambda project_root: None)
    monkeypatch.setattr(runner, "_ensure_layout", lambda project_root: None)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda project_root: {
            "execution": {"train_command": "uv run train.py", "timeout_seconds": 60},
            "metric": {"name": "val_bpb", "lower_is_better": True},
            "safety": {"require_clean_git": False},
        },
    )
    monkeypatch.setattr(runner, "load_best", lambda project_root: None)
    monkeypatch.setattr(runner, "create_worktree", lambda project_root, exp_id: worktree)
    monkeypatch.setattr(
        runner,
        "run_training",
        lambda command, cwd, log_path, timeout_seconds: {
            "returncode": 0,
            "timeout": False,
            "duration_seconds": 1.0,
            "error": None,
        },
    )
    monkeypatch.setattr(
        runner,
        "parse_metrics",
        lambda log_path, metric_name: {
            "metric_name": metric_name,
            "metric_value": 1.234,
            "peak_vram_mb": 1234.0,
            "training_seconds": 60.0,
            "num_steps": 100,
            "crashed": False,
        },
    )
    monkeypatch.setattr(runner, "save_best", lambda project_root, best: saved.append(best))
    monkeypatch.setattr(runner, "generate_experiment_report", lambda project_root, exp_id, data: None)
    monkeypatch.setattr(runner, "remove_worktree", lambda project_root, worktree_path: None)

    report_path = runner.run_baseline(tmp_path)

    assert report_path == tmp_path / ".nightrunner" / "runs" / "baseline" / "report.md"
    assert saved
    assert saved[0]["experiment_id"] == "baseline"
    assert saved[0]["metric_name"] == "val_bpb"
    assert saved[0]["metric_value"] == 1.234
    assert saved[0]["is_baseline"] is True


def test_run_baseline_requires_force_when_best_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "ensure_git_repo", lambda project_root: None)
    monkeypatch.setattr(runner, "_ensure_layout", lambda project_root: None)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda project_root: {"safety": {"require_clean_git": False}},
    )
    monkeypatch.setattr(
        runner,
        "load_best",
        lambda project_root: {"experiment_id": "exp_0001", "metric_value": 1.0},
    )

    with pytest.raises(RuntimeError):
        runner.run_baseline(tmp_path, force=False)


def test_run_night_requires_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ensure_git_repo", lambda project_root: None)
    monkeypatch.setattr(runner, "_ensure_layout", lambda project_root: None)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda project_root: {"safety": {"require_clean_git": False}},
    )
    monkeypatch.setattr(runner, "load_best", lambda project_root: None)

    with pytest.raises(RuntimeError):
        runner.run_night(tmp_path, rounds=1, dry_run=False)
