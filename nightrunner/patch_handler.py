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


class SearchReplaceError(RuntimeError):
    """Raised when search-replace edits cannot be applied safely."""


REQUIRED_COMMON_FIELDS = {"hypothesis", "reason", "expected_effect", "risk"}


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

    missing = REQUIRED_COMMON_FIELDS - set(data.keys())
    if missing:
        raise InvalidModelResponse(f"Model response missing required fields: {sorted(missing)}")

    if "edits" in data:
        edits = data.get("edits")
        if not isinstance(edits, list) or not edits:
            raise InvalidModelResponse("Model response field 'edits' must be a non-empty list.")
        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                raise InvalidModelResponse(f"Edit #{idx} must be an object.")
            for key in ("file", "old_text", "new_text"):
                if key not in edit:
                    raise InvalidModelResponse(f"Edit #{idx} missing required field '{key}'.")
                if not isinstance(edit[key], str):
                    raise InvalidModelResponse(f"Edit #{idx} field '{key}' must be a string.")
            if not edit["old_text"]:
                raise InvalidModelResponse(f"Edit #{idx} field 'old_text' must not be empty.")
        return data

    patch = data.get("patch")
    if isinstance(patch, str) and patch.strip():
        return data

    raise InvalidModelResponse("Model response must include non-empty 'edits' or fallback 'patch'.")


def apply_search_replace_edits(worktree_path: Path, edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply search-replace edits to files in worktree and return edit summaries."""
    applied: list[dict[str, Any]] = []
    for edit in edits:
        edit_file = str(edit["file"])
        old_text = str(edit["old_text"])
        new_text = str(edit["new_text"])

        edit_path = Path(edit_file)
        if edit_path.is_absolute():
            raise SearchReplaceError(f"Edit file must be a relative path: {edit_file}")
        if ".." in edit_path.parts:
            raise SearchReplaceError(f"Edit file must not contain '..': {edit_file}")

        target_path = (worktree_path / edit_path).resolve()
        try:
            target_path.relative_to(worktree_path.resolve())
        except ValueError as exc:
            raise SearchReplaceError(f"Edit path escapes worktree: {edit_file}") from exc
        if not target_path.exists() or not target_path.is_file():
            raise SearchReplaceError(f"Edit target file not found: {edit_file}")

        content = target_path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            preview = old_text[:200]
            raise SearchReplaceError(
                f"old_text not found in {edit_file}. old_text preview: {preview}"
            )
        if count > 1:
            raise SearchReplaceError(
                f"old_text is not unique in {edit_file}. occurrences={count}"
            )

        updated = content.replace(old_text, new_text, 1)
        target_path.write_text(updated, encoding="utf-8")
        applied.append(
            {
                "file": edit_path.as_posix(),
                "old_text_preview": old_text[:200],
                "new_text_preview": new_text[:200],
                "old_length": len(old_text),
                "new_length": len(new_text),
            }
        )

    return applied


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
        raise PatchApplyError(f"git apply failed: {exc}") from exc
    finally:
        patch_file.unlink(missing_ok=True)
