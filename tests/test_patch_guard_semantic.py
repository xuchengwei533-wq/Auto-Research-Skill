from __future__ import annotations

from pathlib import Path

from nightrunner.patch_guard import validate_semantic_changes


def test_semantic_guard_rejects_seed_change(tmp_path: Path) -> None:
    original_root = tmp_path / "project"
    candidate_root = tmp_path / "sandbox"
    (original_root / "examples" / "mnist").mkdir(parents=True)
    (candidate_root / "examples" / "mnist").mkdir(parents=True)
    original = "seed = 42\nlearning_rate = 0.01\n"
    candidate = "seed = 1337\nlearning_rate = 0.01\n"
    (original_root / "examples" / "mnist" / "main.py").write_text(original, encoding="utf-8")
    (candidate_root / "examples" / "mnist" / "main.py").write_text(candidate, encoding="utf-8")

    result = validate_semantic_changes(
        original_root=original_root,
        candidate_root=candidate_root,
        changed_files=["examples/mnist/main.py"],
        protected_terms=["seed", "random_state"],
        enabled=True,
        allow_protected_term_edits=False,
    )

    assert result["ok"] is False
    assert result["violations"][0]["term"] == "seed"
    assert "Rejected by safety guard" in result["violations"][0]["message"]


def test_semantic_guard_allows_learning_rate_change(tmp_path: Path) -> None:
    original_root = tmp_path / "project"
    candidate_root = tmp_path / "sandbox"
    (original_root / "configs").mkdir(parents=True)
    (candidate_root / "configs").mkdir(parents=True)
    original = "learning_rate: 0.01\nseed: 42\n"
    candidate = "learning_rate: 0.02\nseed: 42\n"
    (original_root / "configs" / "train.yaml").write_text(original, encoding="utf-8")
    (candidate_root / "configs" / "train.yaml").write_text(candidate, encoding="utf-8")

    result = validate_semantic_changes(
        original_root=original_root,
        candidate_root=candidate_root,
        changed_files=["configs/train.yaml"],
        protected_terms=["seed", "random_state"],
        enabled=True,
        allow_protected_term_edits=False,
    )

    assert result["ok"] is True
