from pathlib import Path

from nightrunner.log_parser import parse_metrics


def test_parse_val_bpb_colon(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("val_bpb: 0.9981\npeak_vram_mb: 1234.5\n", encoding="utf-8")
    m = parse_metrics(log, "val_bpb")
    assert m["crashed"] is False
    assert m["metric_value"] == 0.9981
    assert m["peak_vram_mb"] == 1234.5


def test_parse_val_loss_equals(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("epoch 1 val_loss=0.123\nnum_steps: 100\n", encoding="utf-8")
    m = parse_metrics(log, "val_loss")
    assert m["crashed"] is False
    assert m["metric_value"] == 0.123
    assert m["num_steps"] == 100


def test_metric_missing_marks_crashed(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("training started\nno metric found\n", encoding="utf-8")
    m = parse_metrics(log, "val_bpb")
    assert m["crashed"] is True
    assert m["metric_value"] is None
