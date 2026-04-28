"""Core orchestration for NightRunner CLI."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from . import __version__
from .agent_api import request_patch
from .config import NIGHTRUNNER_GITIGNORE_LINES, load_config, write_default_config_if_missing
from .diff_analyzer import analyze_diff
from .git_ops import (
    GitError,
    apply_patch_to_project,
    create_worktree,
    ensure_clean_worktree,
    ensure_git_repo,
    remove_worktree,
    run_git,
    save_diff,
)
from .log_parser import parse_metrics
from .patch_guard import get_changed_files, get_new_files, validate_changed_files
from .patch_handler import InvalidModelResponse, PatchApplyError, apply_patch, parse_model_response
from .prompt_builder import build_system_prompt, build_user_prompt
from .report import generate_experiment_report, generate_summary_report
from .state_store import append_experiment, load_best, load_experiments, next_experiment_id, save_best
from .train_runner import run_training
from .utils import ensure_dir, now_iso, write_json, write_text


def _nightrunner_dir(project_root: Path) -> Path:
    return project_root / ".nightrunner"


def _state_dir(project_root: Path) -> Path:
    return _nightrunner_dir(project_root) / "state"


def _ensure_layout(project_root: Path) -> None:
    base = _nightrunner_dir(project_root)
    ensure_dir(base / "state")
    ensure_dir(base / "runs")
    ensure_dir(base / "worktrees")
    ensure_dir(base / "cache")
    ensure_dir(base / "tmp")


def _update_gitignore(project_root: Path) -> None:
    path = project_root / ".gitignore"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = current.splitlines()
    existing = set(lines)
    to_add = [line for line in NIGHTRUNNER_GITIGNORE_LINES if line not in existing]
    if to_add:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(to_add)
        write_text(path, "\n".join(lines) + "\n")


def _collect_editable_contents(worktree: Path, editable: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in editable:
        p = worktree / entry
        if entry.endswith("/") and p.exists() and p.is_dir():
            for fp in p.rglob("*"):
                if fp.is_file():
                    rel = fp.relative_to(worktree).as_posix()
                    out[rel] = fp.read_text(encoding="utf-8", errors="replace")
        elif p.exists() and p.is_file():
            out[p.relative_to(worktree).as_posix()] = p.read_text(
                encoding="utf-8", errors="replace"
            )
    return out


def _is_better(new_value: float, best_value: float | None, lower_is_better: bool) -> bool:
    if best_value is None:
        return True
    return new_value < best_value if lower_is_better else new_value > best_value


def _save_status(run_dir: Path, status_payload: dict[str, Any]) -> None:
    write_json(run_dir / "status.json", status_payload)


def _files_from_patch(patch_text: str) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    new_files: list[str] = []
    lines = patch_text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("+++ b/"):
            f = line[len("+++ b/") :].strip()
            changed.append(f)
            prev = lines[idx - 1] if idx > 0 else ""
            if prev.startswith("--- /dev/null"):
                new_files.append(f)
    return sorted(set(changed)), sorted(set(new_files))


def init_project(project_root: Path) -> dict[str, Any]:
    """Initialize NightRunner runtime files in a git project."""
    ensure_git_repo(project_root)
    _ensure_layout(project_root)
    cfg_path = write_default_config_if_missing(project_root)

    state = _state_dir(project_root)
    best_path = state / "best.json"
    if not best_path.exists():
        write_json(best_path, {})
    exp_path = state / "experiments.jsonl"
    if not exp_path.exists():
        write_text(exp_path, "")
    project_meta = state / "project.json"
    if not project_meta.exists():
        write_json(
            project_meta,
            {
                "project_root": str(project_root.resolve()),
                "created_at": now_iso(),
                "nightrunner_version": __version__,
            },
        )
    _update_gitignore(project_root)
    return {
        "project_root": str(project_root.resolve()),
        "config": str(cfg_path),
        "nightrunner_dir": str(_nightrunner_dir(project_root)),
    }


def run_night(project_root: Path, rounds: int, dry_run: bool = False) -> Path:
    """Run N rounds of NightRunner experiments."""
    ensure_git_repo(project_root)
    _ensure_layout(project_root)
    config = load_config(project_root)
    if config.get("safety", {}).get("require_clean_git", True):
        ensure_clean_worktree(project_root)

    for _ in range(rounds):
        exp_id = next_experiment_id(project_root)
        run_dir = ensure_dir(project_root / ".nightrunner" / "runs" / exp_id)
        record: dict[str, Any] = {
            "id": exp_id,
            "status": "unknown",
            "metric_name": config.get("metric", {}).get("name", "metric"),
            "metric_value": None,
            "hypothesis": "",
            "reason": "",
            "run_dir": f".nightrunner/runs/{exp_id}",
            "patch_path": f".nightrunner/runs/{exp_id}/patch.diff",
            "created_at": now_iso(),
        }
        worktree_path: Path | None = None
        parsed: dict[str, Any] | None = None
        try:
            worktree_path = create_worktree(project_root, exp_id)
            editable = config.get("files", {}).get("editable", [])
            protected = config.get("files", {}).get("protected", [])
            recent = load_experiments(project_root)[-5:]
            best = load_best(project_root)
            file_contents = _collect_editable_contents(worktree_path, editable)
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(config, best, recent, file_contents)

            if config.get("logging", {}).get("save_request", True):
                write_json(
                    run_dir / "request.json",
                    {"system_prompt": system_prompt, "user_prompt": user_prompt},
                )

            try:
                raw = request_patch(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=config.get("agent", {}).get("model", "deepseek-v4-pro"),
                    reasoning_effort=config.get("agent", {}).get("reasoning_effort", "high"),
                    thinking_enabled=config.get("agent", {}).get("thinking_enabled", True),
                )
            except Exception as exc:
                record["status"] = "api_error"
                record["error"] = str(exc)
                _save_status(run_dir, {"status": "api_error", "error": str(exc)})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            if config.get("logging", {}).get("save_response", True):
                write_json(run_dir / "response.json", {"raw": raw})

            try:
                parsed = parse_model_response(raw)
            except InvalidModelResponse as exc:
                record["status"] = "invalid_response"
                record["error"] = str(exc)
                _save_status(run_dir, {"status": "invalid_response", "error": str(exc)})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            write_text(run_dir / "model.patch.diff", parsed["patch"])
            record["hypothesis"] = parsed.get("hypothesis", "")
            record["reason"] = parsed.get("reason", "")
            record["expected_effect"] = parsed.get("expected_effect", "")
            record["risk"] = parsed.get("risk", "")

            try:
                apply_patch(worktree_path, parsed["patch"])
            except PatchApplyError as exc:
                record["status"] = "patch_error"
                record["error"] = str(exc)
                _save_status(run_dir, {"status": "patch_error", "error": str(exc)})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            changed_files = get_changed_files(worktree_path)
            new_files = get_new_files(worktree_path)
            guard_result = validate_changed_files(
                changed_files=changed_files,
                editable_files=editable,
                protected_files=protected,
                allow_new_files=config.get("safety", {}).get("allow_new_files", False),
                allow_dependency_changes=config.get("safety", {}).get(
                    "allow_dependency_changes", False
                ),
                new_files=new_files,
                run_dir=run_dir,
            )
            if not guard_result["ok"]:
                record["status"] = "violation"
                record["violations"] = guard_result["violations"]
                write_text(run_dir / "illegal.patch.diff", parsed["patch"])
                _save_status(
                    run_dir,
                    {"status": "violation", "violations": guard_result["violations"]},
                )
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            save_diff(worktree_path, run_dir / "patch.diff")
            diff_text = (run_dir / "patch.diff").read_text(encoding="utf-8")
            diff_summary = analyze_diff(diff_text)
            write_json(run_dir / "diff_summary.json", diff_summary)
            record["files_changed"] = changed_files
            record["diff_summary"] = diff_summary

            if dry_run:
                record["status"] = "discard"
                record["decision"] = "Dry run enabled: patch validated, training skipped."
                _save_status(run_dir, {"status": "discard", "dry_run": True})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            log_path = run_dir / "run.log"
            run_result = run_training(
                command=config.get("execution", {}).get("train_command", "uv run train.py"),
                cwd=worktree_path,
                log_path=log_path,
                timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 3600)),
            )
            record["run_log_path"] = str(log_path)

            if run_result.get("timeout"):
                record["status"] = "timeout"
                record["decision"] = "Training timed out."
                _save_status(run_dir, {"status": "timeout"})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            if run_result.get("returncode") not in (0,):
                record["status"] = "crash"
                record["decision"] = "Training process crashed."
                _save_status(run_dir, {"status": "crash", "run_result": run_result})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                continue

            metric_cfg = config.get("metric", {})
            metrics = parse_metrics(log_path, metric_cfg.get("name", "val_bpb"))
            write_json(run_dir / "metrics.json", metrics)
            record["metrics"] = metrics
            record["metric_value"] = metrics.get("metric_value")

            if metrics.get("crashed") or metrics.get("metric_value") is None:
                record["status"] = "crash"
                record["decision"] = "Primary metric not found in run log."
            else:
                best = load_best(project_root)
                best_value = best.get("metric_value") if isinstance(best, dict) else None
                is_keep = _is_better(
                    float(metrics["metric_value"]),
                    float(best_value) if best_value is not None else None,
                    bool(metric_cfg.get("lower_is_better", True)),
                )
                if is_keep:
                    record["status"] = "keep"
                    record["decision"] = "Metric improved over current best."
                    save_best(
                        project_root,
                        {
                            "experiment_id": exp_id,
                            "metric_name": metrics.get("metric_name"),
                            "metric_value": metrics.get("metric_value"),
                            "patch_path": f".nightrunner/runs/{exp_id}/patch.diff",
                            "updated_at": now_iso(),
                        },
                    )
                else:
                    record["status"] = "discard"
                    record["decision"] = "Metric did not beat current best."

            _save_status(run_dir, {"status": record["status"], "metrics": record.get("metrics")})
            generate_experiment_report(project_root, exp_id, record)
            append_experiment(project_root, record)
        except Exception as exc:  # pragma: no cover
            record["status"] = "crash"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            _save_status(run_dir, {"status": "crash", "error": str(exc)})
            write_text(run_dir / "error.txt", record["traceback"])
            generate_experiment_report(project_root, exp_id, record)
            append_experiment(project_root, record)
        finally:
            # Keep successful worktree for inspection; cleanup failed/non-keep runs.
            if worktree_path and worktree_path.exists() and record.get("status") != "keep":
                try:
                    remove_worktree(project_root, worktree_path)
                except GitError:
                    pass

    return generate_summary_report(project_root)


def apply_experiment(project_root: Path, exp_id: str) -> Path:
    """Apply selected experiment patch to main worktree after guard checks."""
    ensure_git_repo(project_root)
    ensure_clean_worktree(project_root)
    config = load_config(project_root)

    patch_path = project_root / ".nightrunner" / "runs" / exp_id / "patch.diff"
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")
    patch_text = patch_path.read_text(encoding="utf-8")
    changed_files, new_files = _files_from_patch(patch_text)
    guard_result = validate_changed_files(
        changed_files=changed_files,
        editable_files=config.get("files", {}).get("editable", []),
        protected_files=config.get("files", {}).get("protected", []),
        allow_new_files=config.get("safety", {}).get("allow_new_files", False),
        allow_dependency_changes=config.get("safety", {}).get("allow_dependency_changes", False),
        new_files=new_files,
    )
    if not guard_result["ok"]:
        raise RuntimeError(f"Patch guard failed: {json.dumps(guard_result, ensure_ascii=False)}")

    # Validate patch can apply cleanly before actual apply.
    run_git(["apply", "--check", str(patch_path)], project_root)
    apply_patch_to_project(project_root, patch_path)
    return patch_path


def clean(project_root: Path) -> dict[str, Any]:
    """Cleanup temporary worktrees only."""
    ensure_git_repo(project_root)
    worktrees_root = project_root / ".nightrunner" / "worktrees"
    removed = 0
    failed: list[str] = []
    if not worktrees_root.exists():
        return {"removed": 0, "failed": []}

    for child in worktrees_root.iterdir():
        if not child.is_dir():
            continue
        try:
            remove_worktree(project_root, child)
            removed += 1
        except Exception:
            failed.append(child.name)
    return {"removed": removed, "failed": failed}


def check_auth() -> dict[str, Any]:
    """Check DeepSeek auth environment variable without printing secret value."""
    exists = bool(os.environ.get("DEEPSEEK_API_KEY"))
    return {
        "ok": exists,
        "message": (
            "DEEPSEEK_API_KEY is set."
            if exists
            else "DEEPSEEK_API_KEY is not set. Please set it in your environment variables."
        ),
    }
