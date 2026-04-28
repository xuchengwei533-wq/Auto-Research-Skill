"""Prompt building for NightRunner agent calls."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt() -> str:
    """Return strict system prompt for one experiment patch generation."""
    return (
        "You are NightRunner's AI coding researcher.\n"
        "Current project is a fixed benchmark training project.\n"
        "Your task is to propose exactly one experiment.\n"
        "You may only modify editable files.\n"
        "Default editable file is train.py.\n"
        "You must not modify prepare.py.\n"
        "You must not modify program.md.\n"
        "You must not modify pyproject.toml, uv.lock, or dependency files.\n"
        "You must not change evaluation logic.\n"
        "You must not add any external dependency.\n"
        "You must return strict JSON.\n"
        "Do not output any text outside JSON.\n"
        "JSON must include field 'patch'.\n"
        "patch must be a unified diff that git apply can apply.\n"
        "Return only valid JSON. Do not use Markdown fences. "
        "Do not include explanations outside JSON."
    )


def build_user_prompt(
    config: dict[str, Any],
    best: dict[str, Any] | None,
    recent_experiments: list[dict[str, Any]],
    editable_file_contents: dict[str, str],
) -> str:
    """Build user prompt with project context and output schema."""
    payload = {
        "project_name": config.get("project", {}).get("name", "default-project"),
        "editable_files": config.get("files", {}).get("editable", []),
        "protected_files": config.get("files", {}).get("protected", []),
        "train_command": config.get("execution", {}).get("train_command"),
        "metric.name": config.get("metric", {}).get("name"),
        "lower_is_better": config.get("metric", {}).get("lower_is_better", True),
        "best.json": best,
        "recent_experiments_summary": recent_experiments,
        "editable_file_full_contents": editable_file_contents,
        "json_schema": {
            "hypothesis": "string",
            "reason": "string",
            "expected_effect": "string",
            "risk": "string",
            "files_to_modify": ["string"],
            "patch": "string",
        },
    }
    return (
        "Generate one experiment patch proposal using this context.\n"
        "Return only valid JSON. Do not use Markdown fences. "
        "Do not include explanations outside JSON.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
