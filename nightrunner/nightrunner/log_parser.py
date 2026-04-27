"""Training log parser for metric extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


NUM_PATTERN = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def _last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return int(matches[-1])


def parse_metrics(log_path: Path, metric_name: str) -> dict[str, Any]:
    """Parse primary metric and selected runtime stats from run.log."""
    content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    metric_patterns = [
        rf"{re.escape(metric_name)}\s*[:=]\s*{NUM_PATTERN}",
        rf"{re.escape(metric_name)}\s+{NUM_PATTERN}",
    ]
    metric_value = None
    for pat in metric_patterns:
        metric_value = _last_float(pat, content)
        if metric_value is not None:
            break

    peak_vram_mb = _last_float(rf"peak_vram_mb\s*[:=]\s*{NUM_PATTERN}", content)
    training_seconds = _last_float(rf"training_seconds\s*[:=]\s*{NUM_PATTERN}", content)
    num_steps = _last_int(r"num_steps\s*[:=]\s*(\d+)", content)

    crashed = metric_value is None
    return {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "peak_vram_mb": peak_vram_mb,
        "training_seconds": training_seconds,
        "num_steps": num_steps,
        "crashed": crashed,
    }
