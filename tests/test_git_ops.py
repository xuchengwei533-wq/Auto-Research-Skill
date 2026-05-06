from __future__ import annotations

from pathlib import Path

from nightrunner import git_ops


def test_apply_patch_to_project_runs_check_before_apply(monkeypatch, tmp_path: Path) -> None:
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text("diff --git a/a.txt b/a.txt\n", encoding="utf-8")

    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_ensure_clean_worktree(project_root: Path) -> None:
        calls.append((("ensure_clean_worktree",), project_root))

    def fake_run_git(args: list[str], cwd: Path) -> str:
        calls.append((tuple(args), cwd))
        return ""

    monkeypatch.setattr(git_ops, "ensure_clean_worktree", fake_ensure_clean_worktree)
    monkeypatch.setattr(git_ops, "run_git", fake_run_git)

    git_ops.apply_patch_to_project(tmp_path, patch_path)

    assert calls == [
        (("ensure_clean_worktree",), tmp_path),
        (("apply", "--check", str(patch_path)), tmp_path),
        (("apply", str(patch_path)), tmp_path),
    ]


def test_cleanup_nightrunner_branches_reports_removed_and_skipped(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        git_ops,
        "_list_worktrees",
        lambda project_root: [{"worktree": str(tmp_path / ".nightrunner" / "worktrees" / "exp_0001"), "branch": "nightrunner/active"}],
    )
    monkeypatch.setattr(
        git_ops,
        "list_temporary_branches",
        lambda project_root: ["nightrunner/active", "nightrunner/stale"],
    )

    deleted: list[str] = []

    def fake_run_git(args: list[str], cwd: Path) -> str:
        if args[:2] == ["branch", "-D"]:
            deleted.append(args[2])
        return ""

    monkeypatch.setattr(git_ops, "run_git", fake_run_git)

    result = git_ops.cleanup_nightrunner_branches(tmp_path)

    assert result["removed_branches"] == ["nightrunner/stale"]
    assert result["skipped_branches"] == ["nightrunner/active"]
    assert deleted == ["nightrunner/stale"]
