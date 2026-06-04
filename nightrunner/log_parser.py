"""Training log parser for metric extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


NUM_PATTERN = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

METRIC_CANDIDATE_PATTERNS = [
    ("val_loss", r"val[_\s-]?loss", True, 0),
    ("validation_loss", r"validation[_\s-]?loss", True, 1),
    ("eval_loss", r"eval[_\s-]?loss", True, 2),
    ("loss", r"loss", True, 8),
    ("val_accuracy", r"val[_\s-]?(?:accuracy|acc)", False, 10),
    ("validation_accuracy", r"validation[_\s-]?(?:accuracy|acc)", False, 11),
    ("accuracy", r"(?:accuracy|acc)", False, 18),
    ("roc_auc", r"roc[_\s-]?auc", False, 20),
    ("auc", r"auc", False, 21),
    ("f1", r"f1(?:[_\s-]?score)?", False, 22),
    ("rmse", r"rmse", True, 30),
    ("mae", r"mae", True, 31),
    ("mse", r"mse", True, 32),
]


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


def _metric_regex(name_pattern: str) -> str:
    return rf"\b{name_pattern}\b\s*[:=]\s*{NUM_PATTERN}"


def detect_metric_candidates_from_text(text: str) -> list[dict[str, Any]]:
    """Return likely optimization metrics found in training logs."""
    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for name, name_pattern, lower_is_better, priority in METRIC_CANDIDATE_PATTERNS:
        if name in seen_names:
            continue
        pattern = _metric_regex(name_pattern)
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if not matches:
            continue
        match = matches[-1]
        try:
            value = float(match.group(1))
        except (IndexError, ValueError):
            continue
        candidates.append(
            {
                "name": name,
                "value": value,
                "lower_is_better": lower_is_better,
                "higher_is_better": not lower_is_better,
                "regex": pattern,
                "source": "log",
                "occurrences": len(matches),
                "priority": priority,
            }
        )
        seen_names.add(name)
    return sorted(candidates, key=lambda item: (int(item["priority"]), str(item["name"])))


def detect_metric_candidates(log_path: Path) -> list[dict[str, Any]]:
    content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return detect_metric_candidates_from_text(content)


def parse_metrics(log_path: Path, metric_name: str, metric_regex: str | None = None) -> dict[str, Any]:
    """Parse primary metric and selected runtime stats from run.log."""
    content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    metric_value = None
    metric_error: str | None = None
    if metric_regex:
        try:
            matches = list(re.finditer(metric_regex, content, flags=re.IGNORECASE | re.MULTILINE))
        except re.error as exc:
            metric_error = f"Invalid metric.regex: {exc}"
            matches = []
        if matches:
            m = matches[-1]
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
