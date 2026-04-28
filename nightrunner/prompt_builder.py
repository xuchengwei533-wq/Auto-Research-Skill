"""Prompt building for NightRunner agent calls."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt() -> str:
    """Return strict system prompt for one experiment generation."""
    return (
        "You are NightRunner's AI coding researcher.\n"
        "Your task is to propose exactly one code experiment.\n"
        "You MUST return only valid JSON.\n"
        "Do not output Markdown.\n"
        "Do not output git diff.\n"
        "Do not output unified diff.\n"
        "Do not output ```diff or ```json fences.\n"
        "Do not include any text outside JSON.\n"
        "You must use search-replace edits.\n"
        "Each edit must include: file, old_text, new_text.\n"
        "old_text must be copied exactly from the provided file content.\n"
        "old_text must exist exactly once in that file.\n"
        "Use relative paths only.\n"
        "Do not use absolute Windows paths.\n"
        "Only modify editable files.\n"
        "Do not modify protected files.\n"
        "Do not modify prepare.py.\n"
        "Do not modify pyproject.toml.\n"
        "Do not modify uv.lock.\n"
        "Do not modify nightrunner.yaml.\n"
        "Do not add dependencies.\n"
        "Do not change evaluation logic.\n"
        "For the current Auto-Research-Skill demo, the default editable file is train.py."
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
        "metric_name": config.get("metric", {}).get("name"),
        "lower_is_better": config.get("metric", {}).get("lower_is_better", True),
        "current_best": best,
        "recent_experiments_summary": recent_experiments,
        "editable_file_full_contents": editable_file_contents,
        "json_schema": {
            "hypothesis": "string",
            "reason": "string",
            "expected_effect": "string",
            "risk": "string",
            "edits": [
                {
                    "file": "train.py",
                    "old_text": "exact old text from the file",
                    "new_text": "replacement text",
                }
            ],
        },
        "schema_example": {
            "hypothesis": "Increase ASPECT_RATIO from 64 to 96 to test a wider model.",
            "reason": "The model may be under-capacity within the current benchmark.",
            "expected_effect": "A wider model may reduce validation bpb if the extra capacity is useful.",
            "risk": "It may increase VRAM usage and reduce training speed.",
            "edits": [
                {
                    "file": "train.py",
                    "old_text": "ASPECT_RATIO = 64        # model dim = depth * ASPECT_RATIO",
                    "new_text": "ASPECT_RATIO = 96        # model dim = depth * ASPECT_RATIO",
                }
            ],
        },
    }
    return (
        "Generate exactly one experiment with a small focused change.\n"
        "Return only valid JSON. Do not use Markdown fences. Do not include explanations outside JSON.\n"
        "old_text must be copied verbatim from provided file content and must be unique in the target file.\n"
        "Use edits only. Do not return a patch field unless fallback is explicitly required.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
