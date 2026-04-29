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


def parse_metrics(log_path: Path, metric_name: str, metric_regex: str | None = None) -> dict[str, Any]:
    """Parse primary metric and selected runtime stats from run.log."""
    content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    metric_value = None
    metric_error: str | None = None
    if metric_regex:
        try:
            m = re.search(metric_regex, content, flags=re.MULTILINE)
        except re.error as exc:
            metric_error = f"Invalid metric.regex: {exc}"
            m = None
        if m:
            if m.lastindex is None or m.lastindex < 1:
                metric_error = "metric.regex must include at least one capture group."
            else:
                try:
                    metric_value = float(m.group(1))
                except ValueError:
                    metric_error = "metric.regex group(1) is not a float."
        elif metric_error is None:
            metric_error = "metric.regex did not match run.log content."
    else:
        metric_patterns = [
            rf"{re.escape(metric_name)}\s*[:=]\s*{NUM_PATTERN}",
            rf"{re.escape(metric_name)}\s+{NUM_PATTERN}",
        ]
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
        "metric_regex": metric_regex,
        "metric_error": metric_error,
        "peak_vram_mb": peak_vram_mb,
        "training_seconds": training_seconds,
        "num_steps": num_steps,
        "crashed": crashed,
    }
