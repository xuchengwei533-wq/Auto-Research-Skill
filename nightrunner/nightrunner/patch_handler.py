"""Model response parsing and patch apply helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .git_ops import GitError, run_git


class InvalidModelResponse(ValueError):
    """Raised when model response cannot be parsed or validated."""


class PatchApplyError(RuntimeError):
    """Raised when git apply fails."""


REQUIRED_FIELDS = {
    "hypothesis",
    "reason",
    "expected_effect",
    "risk",
    "files_to_modify",
    "patch",
}


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(raw_text)):
        char = raw_text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_text[start : idx + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    return None
    return None


def parse_model_response(raw_text: str) -> dict[str, Any]:
    """Parse model response to validated JSON payload."""
    data: dict[str, Any] | None
    try:
        parsed = json.loads(raw_text)
        data = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        data = None

    if data is None:
        data = _extract_first_json_object(raw_text)

    if data is None:
        raise InvalidModelResponse("Model response is not valid JSON.")

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise InvalidModelResponse(f"Model response missing required fields: {sorted(missing)}")
    if not isinstance(data.get("patch"), str) or not data["patch"].strip():
        raise InvalidModelResponse("Model response field 'patch' must be a non-empty string.")
    return data


def apply_patch(worktree_path: Path, patch_text: str) -> None:
    """Apply unified diff patch inside worktree using git apply."""
    if not patch_text.strip():
        raise PatchApplyError("Patch is empty.")
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as f:
        f.write(patch_text)
        patch_file = Path(f.name)
    try:
        run_git(["apply", str(patch_file)], worktree_path)
    except GitError as exc:
        raise PatchApplyError(str(exc)) from exc
    finally:
        patch_file.unlink(missing_ok=True)
