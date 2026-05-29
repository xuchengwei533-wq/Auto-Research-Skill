from __future__ import annotations

from pathlib import Path

from nightrunner.patch_guard import validate_semantic_changes


def _semantic_result(tmp_path: Path, rel_path: str, original: str, candidate: str):
    original_root = tmp_path / "project"
    candidate_root = tmp_path / "sandbox"
    target_original = original_root / rel_path
    target_candidate = candidate_root / rel_path
    target_original.parent.mkdir(parents=True, exist_ok=True)
    target_candidate.parent.mkdir(parents=True, exist_ok=True)
    target_original.write_text(original, encoding="utf-8")
    target_candidate.write_text(candidate, encoding="utf-8")
    return validate_semantic_changes(
        original_root=original_root,
        candidate_root=candidate_root,
        changed_files=[rel_path],
        protected_terms=[
            "seed",
            "random_state",
            "torch.manual_seed",
            "train_test_split",
            "test_path",
            "label_column",
        ],
        enabled=True,
        allow_protected_term_edits=False,
    )


def test_semantic_guard_rejects_seed_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "examples/mnist/main.py",
        "seed = 42\nlearning_rate = 0.01\n",
        "seed = 1337\nlearning_rate = 0.01\n",
    )
    assert result["ok"] is False
    assert result["violations"][0]["term"] == "seed"
    assert "Rejected by safety guard" in result["violations"][0]["message"]


def test_semantic_guard_allows_learning_rate_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "configs/train.yaml",
        "learning_rate: 0.01\nseed: 42\n",
        "learning_rate: 0.02\nseed: 42\n",
    )
    assert result["ok"] is True


def test_semantic_guard_rejects_random_state_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "configs/train.yaml",
        "random_state: 42\n",
        "random_state: 7\n",
    )
    assert result["ok"] is False
    assert result["violations"][0]["term"] == "random_state"


def test_semantic_guard_rejects_torch_manual_seed_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "examples/mnist/main.py",
        "torch.manual_seed(42)\n",
        "torch.manual_seed(123)\n",
    )
    assert result["ok"] is False
    assert result["violations"][0]["term"] == "torch.manual_seed"


def test_semantic_guard_rejects_train_test_split_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "train.py",
        "train_test_split(data, test_size=0.2)\n",
        "train_test_split(data, test_size=0.1)\n",
    )
    assert result["ok"] is False
    assert result["violations"][0]["term"] == "train_test_split"


def test_semantic_guard_rejects_test_path_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "config.yaml",
        "test_path: data/test.csv\n",
        "test_path: data/private_test.csv\n",
    )
    assert result["ok"] is False
    assert result["violations"][0]["term"] == "test_path"


def test_semantic_guard_rejects_label_column_change(tmp_path: Path) -> None:
    result = _semantic_result(
        tmp_path,
        "config.yaml",
        "label_column: label\n",
        "label_column: target\n",
    )
    assert result["ok"] is False
    assert result["violations"][0]["term"] == "label_column"
