from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nightrunner import runner
from nightrunner.config import load_config, save_config, write_default_config_if_missing
from nightrunner.experiments import collect_file_hashes, get_experiment_paths, load_experiment_metadata, save_experiment_metadata
from nightrunner.log_parser import detect_metric_candidates_from_text, parse_metrics
from nightrunner.state_store import append_experiment, load_best, load_experiments
from nightrunner.web_ui import _root_html, _run_test_command


def test_detect_metric_candidates_prefers_validation_loss_and_last_value() -> None:
    text = "\n".join(
        [
            "epoch=1 loss: 1.2 val_loss: 0.9 accuracy: 0.70",
            "epoch=2 loss: 1.0 val_loss: 0.6 accuracy: 0.75",
        ]
    )

    candidates = detect_metric_candidates_from_text(text)

    assert candidates[0]["name"] == "val_loss"
    assert candidates[0]["value"] == 0.6
    assert candidates[0]["lower_is_better"] is True


def test_metric_regex_parses_last_match(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text("val_loss: 0.9\nval_loss: 0.4\n", encoding="utf-8")

    metrics = parse_metrics(log_path, "val_loss", r"val_loss:\s*([0-9.]+)")

    assert metrics["metric_value"] == 0.4


def test_setup_yes_auto_detects_metric_from_sandbox_run(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        "print('epoch=1 val_loss: 0.8')\nprint('epoch=2 val_loss: 0.5')\n",
        encoding="utf-8",
    )

    runner.setup(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name=None,
        lower_is_better=None,
        yes=True,
    )

    cfg = load_config(tmp_path)
    assert cfg["metric"]["name"] == "val_loss"
    assert cfg["metric"]["lower_is_better"] is True
    assert cfg["metric"]["regex"]


def test_default_config_does_not_emit_yaml_aliases(tmp_path: Path) -> None:
    config_path = write_default_config_if_missing(tmp_path, editable_files=["train.py"])

    text = config_path.read_text(encoding="utf-8")

    assert "&id" not in text
    assert "*id" not in text


def test_failed_baseline_reruns_without_force(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('no metric yet')\n", encoding="utf-8")
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )

    runner.run_baseline(tmp_path, force=True)
    assert load_experiment_metadata(tmp_path, "baseline")["status"] == "baseline_error"

    (tmp_path / "train.py").write_text("print('val_loss: 0.3')\n", encoding="utf-8")
    runner.run_baseline(tmp_path, force=False)

    metadata = load_experiment_metadata(tmp_path, "baseline")
    best = load_best(tmp_path)
    assert metadata["status"] == "baseline"
    assert metadata["metric_value"] == 0.3
    assert best is not None
    assert best["metric_value"] == 0.3


def test_apply_blocks_non_keep_experiments(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('val_loss: 0.5')\n", encoding="utf-8")
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )
    sandbox_dir = tmp_path / ".nightrunner" / "sandboxes" / "exp_0001"
    sandbox_dir.mkdir(parents=True)
    (sandbox_dir / "train.py").write_text("print('val_loss: 0.4')\n", encoding="utf-8")
    paths = get_experiment_paths(tmp_path, "exp_0001")
    paths.patch_path.write_text("--- a/train.py\n+++ b/train.py\n", encoding="utf-8")
    save_experiment_metadata(
        tmp_path,
        "exp_0001",
        {
            "id": "exp_0001",
            "status": "discard",
            "editable_files": ["train.py"],
            "base_file_hashes": collect_file_hashes(tmp_path, ["train.py"]),
            "sandbox_dir": ".nightrunner/sandboxes/exp_0001",
            "changed_files": ["train.py"],
        },
    )

    preview = runner.preview_experiment(tmp_path, "exp_0001")
    assert preview["can_apply"] is False
    assert preview["safe_to_apply"] is False
    assert "status 'keep'" in preview["apply_block_reason"]

    with pytest.raises(RuntimeError, match="status 'keep'"):
        runner.apply_experiment(tmp_path, "exp_0001", confirm=False)


def test_web_ui_template_renders_metric_detection_controls() -> None:
    html = _root_html(Path("."))

    assert "优化指标" in html
    assert "metric_candidates" in html
    assert "JSON.stringify({" in html
    assert "global_notice" in html
    assert "handleAction" in html
    assert "apply_selected_btn" in html
    assert "can_apply" in html


def test_web_ui_test_run_returns_selected_metric(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        "print('epoch=1 val_loss: 0.8')\nprint('epoch=2 val_loss: 0.4')\n",
        encoding="utf-8",
    )
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )

    result = _run_test_command(tmp_path, f'"{sys.executable}" train.py')

    assert result["selected_metric"]["name"] == "val_loss"
    assert result["selected_metric"]["value"] == 0.4


def test_dry_run_preflight_does_not_call_model_or_create_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('val_loss: 0.5')\n", encoding="utf-8")
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )

    def fail_request_patch(**kwargs):
        raise AssertionError("dry-run preflight must not call the model API")

    def fail_run_training(*args, **kwargs):
        raise AssertionError("dry-run preflight must not run training")

    monkeypatch.setattr(runner, "request_patch", fail_request_patch)
    monkeypatch.setattr(runner, "run_training", fail_run_training)

    summary = runner.run_night(tmp_path, rounds=1, dry_run=True, plain=True)

    assert summary.exists()
    assert load_experiments(tmp_path) == []
    assert not (tmp_path / ".nightrunner" / "sandboxes" / "dry_run_preflight").exists()


def test_run_dry_run_skips_auth_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('val_loss: 0.5')\n", encoding="utf-8")
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )

    def fail_check_auth(project_root=None):
        raise AssertionError("run --dry-run should not require API auth")

    monkeypatch.setattr(runner, "check_auth", fail_check_auth)
    monkeypatch.setattr(runner, "request_patch", lambda **kwargs: (_ for _ in ()).throw(AssertionError("model API called")))
    monkeypatch.setattr(runner, "run_training", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training called")))

    summary = runner.run(tmp_path, rounds=1, dry_run=True, plain=True)

    assert summary.exists()
    assert load_experiments(tmp_path) == []


def test_night_requires_auth_before_creating_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('val_loss: 0.5')\n", encoding="utf-8")
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(runner, "load_api_key", lambda provider: None)

    with pytest.raises(RuntimeError, match="No API key found"):
        runner.run_night(tmp_path, rounds=1, dry_run=False, plain=True)

    assert load_experiments(tmp_path) == []


def test_status_warns_when_metric_changes_after_baseline(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('val_loss: 0.4 accuracy: 0.7')\n", encoding="utf-8")
    runner.init_project(
        tmp_path,
        editable_files=["train.py"],
        train_command=f'"{sys.executable}" train.py',
        metric_name="val_loss",
        lower_is_better=True,
    )
    baseline_record = {
        "id": "baseline",
        "status": "baseline",
        "metric_name": "val_loss",
        "metric_config": {"name": "val_loss", "regex": None, "lower_is_better": True},
        "metric_value": 0.4,
        "editable_files": ["train.py"],
        "base_file_hashes": collect_file_hashes(tmp_path, ["train.py"]),
    }
    save_experiment_metadata(tmp_path, "baseline", baseline_record)
    append_experiment(tmp_path, baseline_record)
    config = load_config(tmp_path)
    config["metric"]["name"] = "accuracy"
    config["metric"]["lower_is_better"] = False
    save_config(tmp_path, config)

    runner.status(tmp_path, plain=True)

    out = capsys.readouterr().out
    assert "Baseline warning" in out
    assert "baseline --force" in out
