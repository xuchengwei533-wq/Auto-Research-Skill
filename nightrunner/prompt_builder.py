"""Prompt building for NightRunner agent calls."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt() -> str:
    """Return strict system prompt for one experiment generation."""
    return (
        "You are NightRunner's AI coding researcher.\n"
        "Propose exactly one experiment.\n"
        "Return only valid JSON.\n"
        "Use search-replace edits.\n"
        "Do not output git diff.\n"
        "Do not output unified diff.\n"
        "Do not output markdown fences.\n"
        "Only modify editable files.\n"
        "Do not modify protected files.\n"
        "Editable file permission only defines where NightRunner may propose changes.\n"
        "Protected keys and protected regions are still enforced inside editable files.\n"
        "Do not add dependencies unless explicitly allowed.\n"
        "Do not change evaluation logic, evaluation metrics, test dataset paths, or split protocol.\n"
        "Do not change reproducibility settings such as seed, random_state, manual_seed, cudnn flags, or hashing seed.\n"
        "Do not change train/val/test split definitions, test_size, validation_split, loader assignments, labels, or targets.\n"
        "Do not try to improve results by modifying protected terms or test protocol.\n"
        "Even inside editable files, do not modify protected reproducibility or evaluation terms.\n"
        "Do not modify random seeds, random_state, manual_seed, torch.manual_seed, np.random.seed, data split logic, test dataset paths, label columns, target columns, metric calculation, evaluation protocol, test_loader, val_loader, or validation/test split settings.\n"
        "Do not improve metrics by changing the evaluation protocol or test data.\n"
        "Search phase must keep randomness and evaluation protocol fixed.\n"
        "Keep randomness fixed during search. Multi-seed validation may only be done as a separate validation phase, not by changing seeds during optimization.\n"
        "If multiple seeds are desired, mention them as a later validation idea instead of editing seeds now.\n"
        "Keep changes small, local, reversible, and easy to review.\n"
        "Prefer hyperparameter and localized training/model changes.\n"
        "Architecture changes are allowed only when editable and simple.\n"
        "Each edit must include: file, old_text, new_text.\n"
        "old_text must be copied exactly from the provided file content.\n"
        "old_text must appear exactly once in the target file.\n"
        "Use relative paths only."
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
        "protected_terms": config.get("safety", {}).get("protected_terms", []),
        "semantic_guard": config.get("safety", {}).get("semantic_guard", True),
        "allow_protected_term_edits": config.get("safety", {}).get("allow_protected_term_edits", False),
        "protected_terms_rule": (
            "Even inside editable files, do not modify protected reproducibility or evaluation terms. "
            "Do not improve metrics by changing evaluation protocol or test data."
        ),
        "editing_policy": (
            "Editable file permission only defines where NightRunner may propose changes. "
            "Protected keys and protected regions are still enforced inside editable files."
        ),
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
        "Generate exactly one focused experiment.\n"
        "Return JSON only with no extra text.\n"
        "Use edits as the primary format.\n"
        "old_text must be exact and unique in the target file.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
