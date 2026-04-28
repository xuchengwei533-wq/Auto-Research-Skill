from __future__ import annotations

from pathlib import Path

import pytest

from nightrunner.patch_handler import (
    SearchReplaceError,
    apply_search_replace_edits,
    parse_model_response,
)


def test_parse_model_response_edits_json() -> None:
    raw = """
{
  "hypothesis": "test",
  "reason": "test",
  "expected_effect": "test",
  "risk": "test",
  "edits": [
    {
      "file": "train.py",
      "old_text": "A = 1",
      "new_text": "A = 2"
    }
  ]
}
"""
    parsed = parse_model_response(raw)
    assert parsed["edits"][0]["file"] == "train.py"


def test_apply_search_replace_edits_success(tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    train_py.write_text("A = 1\nB = 2\n", encoding="utf-8")
    edits = [{"file": "train.py", "old_text": "A = 1", "new_text": "A = 2"}]
    applied = apply_search_replace_edits(tmp_path, edits)
    assert "A = 2" in train_py.read_text(encoding="utf-8")
    assert applied[0]["file"] == "train.py"


def test_apply_search_replace_edits_old_text_not_found(tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    train_py.write_text("A = 1\nB = 2\n", encoding="utf-8")
    edits = [{"file": "train.py", "old_text": "C = 3", "new_text": "C = 4"}]
    with pytest.raises(SearchReplaceError):
        apply_search_replace_edits(tmp_path, edits)


def test_apply_search_replace_edits_old_text_not_unique(tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    train_py.write_text("A = 1\nA = 1\n", encoding="utf-8")
    edits = [{"file": "train.py", "old_text": "A = 1", "new_text": "A = 2"}]
    with pytest.raises(SearchReplaceError):
        apply_search_replace_edits(tmp_path, edits)


def test_apply_search_replace_edits_absolute_path(tmp_path: Path) -> None:
    target = (tmp_path / "train.py").resolve()
    target.write_text("A = 1\n", encoding="utf-8")
    edits = [{"file": str(target), "old_text": "A = 1", "new_text": "A = 2"}]
    with pytest.raises(SearchReplaceError):
        apply_search_replace_edits(tmp_path, edits)


def test_apply_search_replace_edits_parent_path(tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    train_py.write_text("A = 1\n", encoding="utf-8")
    edits = [{"file": "../train.py", "old_text": "A = 1", "new_text": "A = 2"}]
    with pytest.raises(SearchReplaceError):
        apply_search_replace_edits(tmp_path, edits)
