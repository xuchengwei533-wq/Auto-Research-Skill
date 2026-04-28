"""Prompt building for NightRunner agent calls."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt() -> str:
    """Return strict system prompt for one experiment generation."""
    return (
        "你是 NightRunner 的 AI 代码研究员。\n"
        "你的任务是提出且只提出 1 个代码实验。\n"
        "你必须只返回合法 JSON。\n"
        "不要输出 Markdown。\n"
        "不要输出 git diff。\n"
        "不要输出 unified diff。\n"
        "不要输出 ```diff 或 ```json 代码块。\n"
        "不要在 JSON 之外输出任何文本。\n"
        "你必须使用 search-replace edits。\n"
        "每个 edit 必须包含: file, old_text, new_text。\n"
        "old_text 必须从提供的文件内容中逐字复制。\n"
        "old_text 在目标文件中必须恰好出现 1 次。\n"
        "file 只能使用相对路径。\n"
        "不要使用 Windows 绝对路径。\n"
        "只能修改 editable_files 里的文件。\n"
        "不要修改 protected_files。\n"
        "不要修改 prepare.py。\n"
        "不要修改 pyproject.toml。\n"
        "不要修改 uv.lock。\n"
        "不要修改 nightrunner.yaml。\n"
        "不要新增依赖。\n"
        "不要改变评估逻辑。\n"
        "请使用中文撰写 hypothesis、reason、expected_effect、risk 字段。\n"
        "当前 Auto-Research-Skill demo 默认可编辑文件是 train.py。"
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
        "只生成 1 个实验，并保持改动小且聚焦。\n"
        "必须只返回合法 JSON，不要使用 Markdown 代码块，不要在 JSON 外补充说明。\n"
        "old_text 必须从提供文件中逐字复制，并且在目标文件里唯一。\n"
        "只使用 edits，不要把 patch 作为主输出。\n"
        "请使用中文撰写 hypothesis、reason、expected_effect、risk。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
