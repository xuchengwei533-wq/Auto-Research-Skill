"""Experiment artifact helpers for sandbox-backed NightRunner runs."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_json, read_text, write_json, write_text


@dataclass(frozen=True)
class ExperimentPaths:
    exp_id: str
    dir: Path
    metadata_path: Path
    log_path: Path
    patch_path: Path
    idea_path: Path
    changed_files_path: Path
    report_path: Path
    status_path: Path


def experiments_root(project_root: Path) -> Path:
    return ensure_dir(project_root / ".nightrunner" / "experiments")


def get_experiment_paths(project_root: Path, exp_id: str) -> ExperimentPaths:
    exp_dir = ensure_dir(experiments_root(project_root) / exp_id)
    return ExperimentPaths(
        exp_id=exp_id,
        dir=exp_dir,
        metadata_path=exp_dir / "metadata.json",
        log_path=exp_dir / "train.log",
        patch_path=exp_dir / "patch.diff",
        idea_path=exp_dir / "idea.md",
        changed_files_path=exp_dir / "changed_files.json",
        report_path=exp_dir / "report.md",
        status_path=exp_dir / "status.json",
    )


def load_experiment_metadata(project_root: Path, exp_id: str) -> dict[str, Any]:
    legacy_path = project_root / ".nightrunner" / "runs" / exp_id / "status.json"
    metadata = read_json(get_experiment_paths(project_root, exp_id).metadata_path, default=None)
    if isinstance(metadata, dict):
        return metadata
    legacy = read_json(legacy_path, default={})
    return legacy if isinstance(legacy, dict) else {}


def save_experiment_metadata(project_root: Path, exp_id: str, metadata: dict[str, Any]) -> ExperimentPaths:
    paths = get_experiment_paths(project_root, exp_id)
    write_json(paths.metadata_path, metadata)
    status_payload = {
        "status": metadata.get("status"),
        "metric_value": metadata.get("metric_value"),
        "current_round": metadata.get("current_round"),
        "total_rounds": metadata.get("total_rounds"),
        "stage": metadata.get("stage"),
        "sandbox_dir": metadata.get("sandbox_dir"),
    }
    write_json(paths.status_path, status_payload)
    return paths


def write_experiment_idea(paths: ExperimentPaths, metadata: dict[str, Any]) -> None:
    content = "\n".join(
        [
            f"# {paths.exp_id}",
            "",
            "## 思路",
            str(metadata.get("hypothesis", "")),
            "",
            "## 原因",
            str(metadata.get("reason", "")),
            "",
            "## 预期效果",
            str(metadata.get("expected_effect", "")),
            "",
            "## 风险",
            str(metadata.get("risk", "")),
            "",
        ]
    )
    write_text(paths.idea_path, content)


def write_changed_files(paths: ExperimentPaths, changed_files: list[str]) -> None:
    write_json(paths.changed_files_path, {"files": changed_files})


def calculate_file_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_file_hashes(project_root: Path, relative_paths: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for rel_path in relative_paths:
        normalized = rel_path.replace("\\", "/")
        result[normalized] = calculate_file_hash(project_root / normalized)
    return result


def diff_editable_files(project_root: Path, sandbox_root: Path, editable_files: list[str]) -> tuple[str, list[str]]:
    diff_parts: list[str] = []
    changed_files: list[str] = []
    for rel_path in editable_files:
        normalized = rel_path.replace("\\", "/")
        project_file = project_root / normalized
        sandbox_file = sandbox_root / normalized
        project_text = read_text(project_file)
        sandbox_text = read_text(sandbox_file)
        if project_text == sandbox_text:
            continue
        changed_files.append(normalized)
        diff = difflib.unified_diff(
            project_text.splitlines(),
            sandbox_text.splitlines(),
            fromfile=f"a/{normalized}",
            tofile=f"b/{normalized}",
            lineterm="",
        )
        diff_parts.append("\n".join(diff))
    return ("\n\n".join(part for part in diff_parts if part).strip() + ("\n" if diff_parts else "")), changed_files


def load_diff_text(project_root: Path, exp_id: str) -> str:
    paths = get_experiment_paths(project_root, exp_id)
    if paths.patch_path.exists():
        return paths.patch_path.read_text(encoding="utf-8", errors="replace")
    legacy_path = project_root / ".nightrunner" / "runs" / exp_id / "patch.diff"
    return read_text(legacy_path)


def list_experiment_ids(project_root: Path) -> list[str]:
    root = experiments_root(project_root)
    ids = [item.name for item in root.iterdir() if item.is_dir()]
    return sorted(ids)
