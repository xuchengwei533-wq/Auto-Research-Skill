"""Basic diff analyzer for reporting."""

from __future__ import annotations

import re
from typing import Any

HYPERPARAM_KEYS = [
    "lr",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "dropout",
    "hidden_dim",
    "num_layers",
]


def _extract_assignments(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    key_pat = "|".join(re.escape(k) for k in HYPERPARAM_KEYS)
    pattern = re.compile(rf"\b({key_pat})\b\s*[:=]\s*([^\s,#]+)")
    for line in lines:
        m = pattern.search(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def analyze_diff(diff_text: str) -> dict[str, Any]:
    """Analyze git diff text for changed files, line stats, and likely hyperparameters."""
    files_changed: list[str] = []
    added_lines = 0
    removed_lines = 0
    plus_payload: list[str] = []
    minus_payload: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files_changed.append(line[len("+++ b/") :].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines += 1
            plus_payload.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines += 1
            minus_payload.append(line[1:])

    removed_assign = _extract_assignments(minus_payload)
    added_assign = _extract_assignments(plus_payload)

    changes = []
    for key in HYPERPARAM_KEYS:
        before = removed_assign.get(key)
        after = added_assign.get(key)
        if before is not None and after is not None and before != after:
            changes.append({"name": key, "before": before, "after": after})

    return {
        "files_changed": sorted(set(files_changed)),
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "possible_hyperparameter_changes": changes,
    }
