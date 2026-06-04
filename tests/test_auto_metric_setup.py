from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nightrunner import runner
from nightrunner.config import load_config, write_default_config_if_missing
from nightrunner.experiments import collect_file_hashes, get_experiment_paths, load_experiment_metadata, save_experiment_metadata
from nightrunner.log_parser import detect_metric_candidates_from_text, parse_metrics
from nightrunner.state_store import load_best
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

    with pytest.raises(RuntimeError, match="status 'keep'"):
        runner.apply_experiment(tmp_path, "exp_0001", confirm=False)


def test_web_ui_template_renders_metric_detection_controls() -> None:
    html = _root_html(Path("."))

    assert "优化指标" in html
    assert "metric_candidates" in html
    assert "JSON.stringify({" in html


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
