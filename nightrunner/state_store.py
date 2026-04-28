"""Persistent state storage for NightRunner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


def _state_dir(project_root: Path) -> Path:
    return ensure_dir(project_root / ".nightrunner" / "state")


def _best_path(project_root: Path) -> Path:
    return _state_dir(project_root) / "best.json"


def _experiments_path(project_root: Path) -> Path:
    return _state_dir(project_root) / "experiments.jsonl"


def load_best(project_root: Path) -> dict[str, Any] | None:
    path = _best_path(project_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return data
    except json.JSONDecodeError:
        return None
    return None


def save_best(project_root: Path, best: dict[str, Any]) -> None:
    write_json(_best_path(project_root), best)


def append_experiment(project_root: Path, record: dict[str, Any]) -> None:
    path = _experiments_path(project_root)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_experiments(project_root: Path) -> list[dict[str, Any]]:
    path = _experiments_path(project_root)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                records.append(data)
        except json.JSONDecodeError:
            continue
    return records


def next_experiment_id(project_root: Path) -> str:
    experiments = load_experiments(project_root)
    max_id = 0
    for rec in experiments:
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid.startswith("exp_"):
            continue
        try:
            max_id = max(max_id, int(rid.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return f"exp_{max_id + 1:04d}"
