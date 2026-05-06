"""Core orchestration for NightRunner CLI."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from .agent_api import request_patch
from .auth_store import load_api_key, mask_api_key
from .console_ui import RunUI
from .config import load_config
from .diff_analyzer import analyze_diff
from .git_ops import (
    GitError,
    WorktreeSetupError,
    apply_patch_to_project,
    cleanup_nightrunner_branches,
    create_worktree,
    ensure_clean_worktree,
    ensure_git_repo,
    get_worktrees_root,
    is_git_repo,
    remove_worktree,
    run_git,
    save_diff,
)
from .log_parser import parse_metrics
from .patch_guard import get_changed_files, get_new_files, validate_changed_files
from .patch_handler import (
    InvalidModelResponse,
    PatchApplyError,
    SearchReplaceError,
    apply_patch,
    apply_search_replace_edits,
    parse_model_response,
)
from .prompt_builder import build_system_prompt, build_user_prompt
from .project_context import build_project_context, ensure_project_layout
from .report import generate_experiment_report, generate_summary_report
from .setup_flow import (
    SetupOptions,
    init_project as setup_init_project,
    is_git_available as setup_is_git_available,
    prompt_yes_no as setup_prompt_yes_no,
    run_setup as run_setup_flow,
    scan_python_files as setup_scan_python_files,
    update_gitignore as setup_update_gitignore,
    validate_editable_paths as setup_validate_editable_paths,
)
from .state_store import append_experiment, load_best, load_experiments, next_experiment_id, save_best
from .train_runner import run_training
from .utils import ensure_dir, now_iso, write_json, write_text


def _nightrunner_dir(project_root: Path) -> Path:
    return build_project_context(project_root).nightrunner_dir


def _state_dir(project_root: Path) -> Path:
    return build_project_context(project_root).state_dir


def _ensure_layout(project_root: Path) -> None:
    ensure_project_layout(build_project_context(project_root))


def _update_gitignore(project_root: Path) -> None:
    setup_update_gitignore(project_root)


def _is_git_available() -> bool:
    return setup_is_git_available()


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    return setup_prompt_yes_no(prompt, default_yes=default_yes)


def _has_git_repo_marker(path: Path) -> bool:
    marker = path / ".git"
    return marker.is_file() or marker.is_dir()


def _is_nested_git_repo_dir(project_root: Path, path: Path) -> bool:
    if path == project_root:
        return False
    return _has_git_repo_marker(path)


def _is_inside_nested_git_repo(project_root: Path, relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").strip("/")
    if not normalized:
        return False
    current = project_root
    parts = [part for part in Path(normalized).parts if part not in {"", "."}]
    if not parts:
        return False
    for part in parts[:-1]:
        current = current / part
        if _is_nested_git_repo_dir(project_root, current):
            return True
    return False


def _validate_editable_paths(project_root: Path, editable_files: list[str]) -> None:
    setup_validate_editable_paths(project_root, editable_files)


def _scan_python_files(project_root: Path) -> list[str]:
    return setup_scan_python_files(project_root)


def _parse_editable_selection(raw: str, candidates: list[str]) -> list[str]:
    if not raw.strip():
        return [candidates[0]] if candidates else ["train.py"]
    chosen: list[str] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if token.isdigit():
            idx = int(token)
            if idx < 1 or idx > len(candidates):
                raise ValueError(f"Selection index out of range: {token}")
            chosen.append(candidates[idx - 1])
        else:
            normalized = token.replace("\\", "/")
            if normalized in candidates:
                chosen.append(normalized)
            else:
                chosen.append(normalized)
    deduped: list[str] = []
    for path in chosen:
        if path not in deduped:
            deduped.append(path)
    return deduped or (["train.py"] if not candidates else [candidates[0]])


def _detect_train_command_default(candidates: list[str]) -> str:
    if "train.py" in candidates:
        return "python train.py"
    if "main.py" in candidates:
        return "python main.py"
    return "python train.py"


def _baseline_exists(project_root: Path) -> bool:
    status_path = project_root / ".nightrunner" / "runs" / "baseline" / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict) and status.get("status") == "baseline":
                return True
        except json.JSONDecodeError:
            pass
    best = load_best(project_root)
    if isinstance(best, dict) and (best.get("is_baseline") is True or best.get("experiment_id") == "baseline"):
        return True
    experiments = load_experiments(project_root)
    return any(rec.get("id") == "baseline" and rec.get("status") == "baseline" for rec in experiments)


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


def _fmt_seconds(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}s"


def _git_status_paths(project_root: Path) -> list[str]:
    """Return dirty paths from git status --porcelain."""
    try:
        out = run_git(["status", "--porcelain"], project_root)
    except Exception:
        return []
    paths: list[str] = []
    for raw_line in out.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        path_part = line[3:] if len(line) > 3 else line
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        normalized = path_part.strip().replace("\\", "/")
        if normalized:
            paths.append(normalized)
    return paths


def _format_setup_commit_hint(project_root: Path) -> str:
    gitignore_path = project_root / ".gitignore"
    add_targets = ["nightrunner.yaml"]
    if gitignore_path.exists():
        add_targets.append(".gitignore")
    add_line = " ".join(add_targets)
    return (
        "NightRunner setup completed.\n"
        "Before running baseline, commit the generated config:\n\n"
        f"  git add {add_line}\n"
        "  git commit -m \"Configure NightRunner\"\n\n"
        "Then run:\n\n"
        "  nightrunner baseline\n"
        "  nightrunner run --rounds 8"
    )


def _raise_setup_dirty_baseline_error(project_root: Path, dirty_paths: list[str]) -> None:
    normalized = {path.replace("\\", "/") for path in dirty_paths}
    setup_related = {
        path
        for path in normalized
        if path == "nightrunner.yaml" or path == ".gitignore" or path.startswith(".nightrunner/")
    }
    if setup_related:
        gitignore_path = project_root / ".gitignore"
        add_targets = ["nightrunner.yaml"]
        if gitignore_path.exists():
            add_targets.append(".gitignore")
        add_line = " ".join(add_targets)
        raise RuntimeError(
            "NightRunner setup created or modified project config files.\n"
            "Commit them first:\n\n"
            f"  git add {add_line}\n"
            "  git commit -m \"Configure NightRunner\"\n\n"
            "Then run:\n\n"
            "  nightrunner baseline"
        )
    ensure_clean_worktree(project_root)


def _short(text: str, limit: int = 60) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _format_metric(metric_cfg: dict[str, Any]) -> str:
    name = str(metric_cfg.get("name", "val_loss"))
    lower = bool(metric_cfg.get("lower_is_better", True))
    direction = "lower is better" if lower else "higher is better"
    return f"{name}, {direction}"


def _print_config_summary(
    project_root: Path,
    config: dict[str, Any],
    title: str = "NightRunner configuration",
    emit: callable | None = None,
) -> None:
    out = emit or print
    files = config.get("files", {})
    metric_cfg = config.get("metric", {})
    safety = config.get("safety", {})
    agent = config.get("agent", {})
    out("")
    out(title)
    out("")
    out("Project root:")
    out(project_root)
    out("")
    out("Editable files:")
    for path in files.get("editable", []) or []:
        out(f"- {path}")
    out("")
    out("Protected files:")
    for path in files.get("protected", []) or []:
        out(f"- {path}")
    out("")
    out("Train command:")
    out(config.get("execution", {}).get("train_command", "python train.py"))
    out("")
    out("Metric:")
    out(_format_metric(metric_cfg))
    out("")
    out("Metric regex:")
    out(metric_cfg.get("regex"))
    out("")
    out("Worktree root:")
    out(get_worktrees_root(project_root))
    out("")
    out("Agent:")
    out(f"provider: {agent.get('provider', 'deepseek')}")
    out(f"model: {agent.get('model', 'deepseek-v4-pro')}")
    out(f"base_url: {agent.get('base_url', 'https://api.deepseek.com')}")
    out(f"api_key_env: {agent.get('api_key_env', 'DEEPSEEK_API_KEY')}")
    out("")
    out("Safety:")
    out(f"require_clean_git: {bool(safety.get('require_clean_git', True))}")
    out(f"auto_apply_to_main: {bool(safety.get('auto_apply_to_main', False))}")
    out(f"allow_new_files: {bool(safety.get('allow_new_files', False))}")
    out(f"allow_dependency_changes: {bool(safety.get('allow_dependency_changes', False))}")
    out("")


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


def init_project(
    project_root: Path,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
    update_gitignore: bool = True,
) -> dict[str, Any]:
    """Initialize NightRunner runtime files in a git project."""
    return setup_init_project(
        project_root=project_root,
        editable_files=editable_files,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        update_gitignore_enabled=update_gitignore,
    )


def _require_best_baseline(project_root: Path) -> dict[str, Any]:
    best = load_best(project_root)
    if not isinstance(best, dict) or best.get("metric_value") is None:
        raise RuntimeError(
            "No baseline metric found. Run `nightrunner baseline` first."
        )
    return best


def run_baseline(project_root: Path, force: bool = False) -> Path:
    """Run baseline training once in project root and save best.json."""
    ensure_git_repo(project_root)
    _ensure_layout(project_root)
    config = load_config(project_root)
    if config.get("safety", {}).get("require_clean_git", True):
        ensure_clean_worktree(project_root)

    existing_best = load_best(project_root)
    if existing_best and not force:
        raise RuntimeError(
            "best.json already exists. Use `nightrunner baseline --force` to rerun baseline."
        )

    run_dir = ensure_dir(project_root / ".nightrunner" / "runs" / "baseline")
    record: dict[str, Any] = {
        "id": "baseline",
        "status": "unknown",
        "metric_name": config.get("metric", {}).get("name", "metric"),
        "metric_value": None,
        "hypothesis": "Run baseline training without AI edits.",
        "reason": "Capture a stable reference metric before automated experiments.",
        "expected_effect": "best.json stores baseline metric for future keep/discard decisions.",
        "risk": "Baseline run may fail, timeout, or miss metric extraction.",
        "run_dir": ".nightrunner/runs/baseline",
        "patch_path": None,
        "created_at": now_iso(),
        "is_baseline": True,
    }
    try:
        log_path = run_dir / "run.log"
        run_result = run_training(
            command=config.get("execution", {}).get("train_command", "python train.py"),
            cwd=project_root,
            log_path=log_path,
            timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 3600)),
        )
        record["run_log_path"] = str(log_path)

        if run_result.get("timeout"):
            record["status"] = "baseline_error"
            record["decision"] = "Baseline training timed out."
            _save_status(run_dir, {"status": "baseline_error", "is_baseline": True})
            generate_experiment_report(project_root, "baseline", record)
            append_experiment(project_root, record)
            generate_summary_report(project_root)
            return run_dir / "report.md"

        if run_result.get("returncode") not in (0,):
            record["status"] = "baseline_error"
            record["decision"] = "Baseline training process crashed."
            _save_status(
                run_dir, {"status": "baseline_error", "run_result": run_result, "is_baseline": True}
            )
            generate_experiment_report(project_root, "baseline", record)
            append_experiment(project_root, record)
            generate_summary_report(project_root)
            return run_dir / "report.md"

        metric_cfg = config.get("metric", {})
        metrics = parse_metrics(
            log_path,
            metric_cfg.get("name", "val_loss"),
            metric_cfg.get("regex"),
        )
        write_json(run_dir / "metrics.json", metrics)
        record["metrics"] = metrics
        record["metric_value"] = metrics.get("metric_value")

        if metrics.get("crashed") or metrics.get("metric_value") is None:
            record["status"] = "baseline_error"
            record["decision"] = "Baseline run completed but metric was not found in run.log."
            _save_status(run_dir, {"status": "baseline_error", "metrics": metrics, "is_baseline": True})
            generate_experiment_report(project_root, "baseline", record)
            append_experiment(project_root, record)
            generate_summary_report(project_root)
            return run_dir / "report.md"

        save_best(
            project_root,
            {
                "experiment_id": "baseline",
                "metric_name": metrics.get("metric_name"),
                "metric_value": metrics.get("metric_value"),
                "patch_path": None,
                "run_dir": ".nightrunner/runs/baseline",
                "updated_at": now_iso(),
                "is_baseline": True,
            },
        )
        record["status"] = "keep"
        record["status"] = "baseline"
        record["decision"] = "Baseline metric saved to best.json."
        _save_status(run_dir, {"status": "baseline", "metrics": metrics, "is_baseline": True})
        generate_experiment_report(project_root, "baseline", record)
        append_experiment(project_root, record)
        generate_summary_report(project_root)
        return run_dir / "report.md"
    except Exception as exc:  # pragma: no cover
        record["status"] = "baseline_error"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        _save_status(run_dir, {"status": "baseline_error", "error": str(exc), "is_baseline": True})
        write_text(run_dir / "error.txt", record["traceback"])
        generate_experiment_report(project_root, "baseline", record)
        append_experiment(project_root, record)
        generate_summary_report(project_root)
        return run_dir / "report.md"


def run_night(
    project_root: Path,
    rounds: int,
    dry_run: bool = False,
    plain: bool = False,
    ui: RunUI | None = None,
) -> Path:
    """Run N rounds of NightRunner experiments."""
    ensure_git_repo(project_root)
    _ensure_layout(project_root)
    config = load_config(project_root)
    ui = ui or RunUI(enabled=not plain)
    _print_config_summary(project_root, config, "NightRunner configuration", emit=ui.log)
    ui.start_run(project_root, config, rounds, baseline_info="checked at startup")
    if config.get("safety", {}).get("require_clean_git", True):
        ensure_clean_worktree(project_root)
    if not dry_run:
        _require_best_baseline(project_root)

    for i in range(rounds):
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
            "timings": {
                "worktree_seconds": 0.0,
                "prompt_seconds": 0.0,
                "api_seconds": 0.0,
                "patch_seconds": 0.0,
                "training_seconds": 0.0,
                "metric_parse_seconds": 0.0,
                "report_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
        worktree_path: Path | None = None
        parsed: dict[str, Any] | None = None
        t_total = perf_counter()
        ui.start_experiment(exp_id, i + 1, rounds)
        try:
            ui.set_stage("Preparing experiment")
            editable = config.get("files", {}).get("editable", [])
            protected = config.get("files", {}).get("protected", [])

            t_stage = perf_counter()
            ui.set_stage("Creating git worktree")
            try:
                worktree_path, branch_name = create_worktree(project_root, exp_id)
                record["branch_name"] = branch_name
                record["worktree_path"] = str(worktree_path)
                write_json(
                    run_dir / "worktree.json",
                    {"branch_name": branch_name, "worktree_path": str(worktree_path)},
                )
            except WorktreeSetupError as exc:
                record["status"] = "setup_error"
                record["error"] = str(exc)
                record["timings"]["worktree_seconds"] = perf_counter() - t_stage
                record["timings"]["total_seconds"] = perf_counter() - t_total
                _save_status(run_dir, {"status": "setup_error", "error": str(exc), "timings": record["timings"]})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                generate_summary_report(project_root)
                ui.finish_experiment(record)
                continue
            record["timings"]["worktree_seconds"] = perf_counter() - t_stage

            t_stage = perf_counter()
            ui.set_stage("Reading editable files")
            recent = load_experiments(project_root)[-5:]
            best = _require_best_baseline(project_root) if not dry_run else load_best(project_root)
            file_contents = _collect_editable_contents(worktree_path, editable)
            ui.set_stage("Building prompt")
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(config, best, recent, file_contents)
            record["timings"]["prompt_seconds"] = perf_counter() - t_stage

            if config.get("logging", {}).get("save_request", True):
                write_json(run_dir / "request.json", {"system_prompt": system_prompt, "user_prompt": user_prompt})

            t_stage = perf_counter()
            ui.set_stage("Calling LLM API")
            ui.set_stage("Waiting for LLM response")
            try:
                raw = request_patch(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=config.get("agent", {}).get("model", "deepseek-v4-pro"),
                    base_url=config.get("agent", {}).get("base_url", "https://api.deepseek.com"),
                    api_key_env=config.get("agent", {}).get("api_key_env", "DEEPSEEK_API_KEY"),
                    reasoning_effort=config.get("agent", {}).get("reasoning_effort", "high"),
                    thinking_enabled=config.get("agent", {}).get("thinking_enabled", True),
                    on_start=lambda: ui.log(f"[{exp_id}] Calling LLM API..."),
                    on_success=lambda secs: ui.log(f"[{exp_id}] LLM response received in {secs:.1f}s"),
                    on_retry=lambda idx, total, wait, err: ui.api_retry(idx, total, wait, err),
                )
            except Exception as exc:
                record["status"] = "api_error"
                record["error"] = str(exc)
                record["timings"]["api_seconds"] = perf_counter() - t_stage
                record["timings"]["total_seconds"] = perf_counter() - t_total
                _save_status(run_dir, {"status": "api_error", "error": str(exc), "timings": record["timings"]})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                generate_summary_report(project_root)
                ui.finish_experiment(record)
                continue
            record["timings"]["api_seconds"] = perf_counter() - t_stage

            if config.get("logging", {}).get("save_response", True):
                write_json(run_dir / "response.json", {"raw": raw})

            ui.set_stage("Parsing LLM response")
            try:
                parsed = parse_model_response(raw)
            except InvalidModelResponse:
                record["status"] = "invalid_response"
                record["error"] = "failed to parse JSON"
                record["timings"]["total_seconds"] = perf_counter() - t_total
                ui.log(f"[{exp_id}] Invalid LLM response: failed to parse JSON.")
                ui.log(f"[{exp_id}] Response saved to {_safe_relative(run_dir / 'response.json', project_root)}")
                _save_status(run_dir, {"status": "invalid_response", "error": record["error"], "timings": record["timings"]})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                generate_summary_report(project_root)
                ui.finish_experiment(record)
                continue

            record["hypothesis"] = parsed.get("hypothesis", "")
            record["reason"] = parsed.get("reason", "")
            record["expected_effect"] = parsed.get("expected_effect", "")
            record["risk"] = parsed.get("risk", "")
            record["used_legacy_patch_mode"] = False

            t_stage = perf_counter()
            ui.set_stage("Applying edits")
            try:
                if "edits" in parsed:
                    model_edits = parsed["edits"]
                    write_json(run_dir / "model_edits.json", {"edits": model_edits})
                    applied_edits = apply_search_replace_edits(worktree_path, model_edits)
                    write_json(run_dir / "applied_edits.json", {"applied_edits": applied_edits})
                    record["applied_edits"] = applied_edits
                elif isinstance(parsed.get("patch"), str) and parsed["patch"].strip():
                    write_text(run_dir / "model.patch.diff", parsed["patch"])
                    apply_patch(worktree_path, parsed["patch"])
                    record["used_legacy_patch_mode"] = True
                else:
                    raise InvalidModelResponse("Model response contains neither valid 'edits' nor fallback 'patch'.")
            except (PatchApplyError, SearchReplaceError, InvalidModelResponse) as exc:
                record["status"] = "patch_error"
                record["error"] = str(exc)
                if isinstance(exc, SearchReplaceError):
                    record["error_type"] = "search_replace_error"
                elif isinstance(exc, PatchApplyError):
                    record["error_type"] = "patch_apply_error"
                else:
                    record["error_type"] = "invalid_model_response"
                record["timings"]["patch_seconds"] = perf_counter() - t_stage
                record["timings"]["total_seconds"] = perf_counter() - t_total
                _save_status(
                    run_dir,
                    {
                        "status": "patch_error",
                        "error": str(exc),
                        "error_type": record["error_type"],
                        "timings": record["timings"],
                    },
                )
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                generate_summary_report(project_root)
                ui.finish_experiment(record)
                continue

            ui.set_stage("Running patch guard")
            changed_files = get_changed_files(worktree_path)
            new_files = get_new_files(worktree_path)
            guard_result = validate_changed_files(
                changed_files=changed_files,
                editable_files=editable,
                protected_files=protected,
                allow_new_files=config.get("safety", {}).get("allow_new_files", False),
                allow_dependency_changes=config.get("safety", {}).get("allow_dependency_changes", False),
                new_files=new_files,
                run_dir=run_dir,
            )
            if not guard_result["ok"]:
                record["status"] = "violation"
                record["violations"] = guard_result["violations"]
                record["timings"]["patch_seconds"] = perf_counter() - t_stage
                record["timings"]["total_seconds"] = perf_counter() - t_total
                _save_status(run_dir, {"status": "violation", "violations": guard_result["violations"], "timings": record["timings"]})
                generate_experiment_report(project_root, exp_id, record)
                append_experiment(project_root, record)
                generate_summary_report(project_root)
                ui.finish_experiment(record)
                continue

            ui.set_stage("Generating patch.diff")
            save_diff(worktree_path, run_dir / "patch.diff")
            diff_text = (run_dir / "patch.diff").read_text(encoding="utf-8")
            diff_summary = analyze_diff(diff_text)
            write_json(run_dir / "diff_summary.json", diff_summary)
            record["files_changed"] = changed_files
            record["diff_summary"] = diff_summary
            record["timings"]["patch_seconds"] = perf_counter() - t_stage

            if dry_run:
                record["status"] = "discard"
                record["decision"] = "Dry run enabled: patch validated, training skipped."
                record["timings"]["total_seconds"] = perf_counter() - t_total
                ui.set_stage("Writing report")
                t_report = perf_counter()
                generate_experiment_report(project_root, exp_id, record)
                record["timings"]["report_seconds"] = perf_counter() - t_report
                append_experiment(project_root, record)
                _save_status(
                    run_dir,
                    {
                        "status": "discard",
                        "dry_run": True,
                        "timings": record["timings"],
                        "branch_name": record.get("branch_name"),
                        "worktree_path": record.get("worktree_path"),
                    },
                )
                ui.set_stage("Updating summary")
                generate_summary_report(project_root)
                ui.finish_experiment(record)
                continue

            ui.set_stage("Running train command")
            log_path = run_dir / "run.log"
            t_train = perf_counter()
            run_result = run_training(
                command=config.get("execution", {}).get("train_command", "python train.py"),
                cwd=worktree_path,
                log_path=log_path,
                timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 3600)),
            )
            record["run_log_path"] = str(log_path)
            record["timings"]["training_seconds"] = perf_counter() - t_train
            ui.log(f"[{exp_id}] Training completed in {record['timings']['training_seconds']:.1f}s")

            if run_result.get("timeout"):
                record["status"] = "timeout"
                record["decision"] = "Training timed out."
                ui.log(f"[{exp_id}] Training failed. See: {_safe_relative(log_path, project_root)}")
            elif run_result.get("returncode") not in (0,):
                record["status"] = "crash"
                record["decision"] = "Training process crashed."
                ui.log(f"[{exp_id}] Training failed. See: {_safe_relative(log_path, project_root)}")
            else:
                ui.set_stage("Parsing metric")
                metric_cfg = config.get("metric", {})
                t_parse = perf_counter()
                metrics = parse_metrics(log_path, metric_cfg.get("name", "val_loss"), metric_cfg.get("regex"))
                record["timings"]["metric_parse_seconds"] = perf_counter() - t_parse
                write_json(run_dir / "metrics.json", metrics)
                record["metrics"] = metrics
                record["metric_value"] = metrics.get("metric_value")

                if metrics.get("crashed") or metrics.get("metric_value") is None:
                    record["status"] = "crash"
                    record["decision"] = "Primary metric not found in run.log."
                else:
                    best = _require_best_baseline(project_root)
                    baseline_value = best.get("metric_value") if isinstance(best, dict) else None
                    if isinstance(baseline_value, (int, float)):
                        record["delta_vs_baseline"] = float(metrics["metric_value"]) - float(baseline_value)
                    ui.log(f"[{exp_id}] Parsed {metrics.get('metric_name')}={metrics.get('metric_value')}")
                    is_keep = _is_better(
                        float(metrics["metric_value"]),
                        float(baseline_value) if baseline_value is not None else None,
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

            record["timings"]["total_seconds"] = perf_counter() - t_total
            ui.set_stage("Writing report")
            t_report = perf_counter()
            generate_experiment_report(project_root, exp_id, record)
            record["timings"]["report_seconds"] = perf_counter() - t_report
            append_experiment(project_root, record)
            ui.set_stage("Updating summary")
            generate_summary_report(project_root)
            _save_status(
                run_dir,
                {
                    "status": record["status"],
                    "metrics": record.get("metrics"),
                    "timings": record["timings"],
                    "branch_name": record.get("branch_name"),
                    "worktree_path": record.get("worktree_path"),
                },
            )
            ui.finish_experiment(record)
        except Exception as exc:  # pragma: no cover
            record["status"] = "crash"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            record["timings"]["total_seconds"] = perf_counter() - t_total
            _save_status(run_dir, {"status": "crash", "error": str(exc), "timings": record["timings"]})
            write_text(run_dir / "error.txt", record["traceback"])
            generate_experiment_report(project_root, exp_id, record)
            append_experiment(project_root, record)
            generate_summary_report(project_root)
            ui.error(f"[{exp_id}] {str(exc)[:180]}")
            ui.finish_experiment(record)
        finally:
            if worktree_path and worktree_path.exists() and record.get("status") != "keep":
                try:
                    remove_worktree(project_root, worktree_path, record.get("branch_name"))
                except GitError:
                    pass

    summary = generate_summary_report(project_root)
    ui.finish_run(summary)
    return summary


def run(project_root: Path, rounds: int = 1, dry_run: bool = False, plain: bool = False) -> Path:
    """Product entrypoint: baseline check + night + summary."""
    if rounds <= 0:
        raise ValueError("--rounds must be > 0")
    try:
        config = load_config(project_root)
    except FileNotFoundError as exc:
        raise RuntimeError("No nightrunner.yaml found.\nRun `nightrunner setup` first.") from exc

    ui = RunUI(enabled=not plain)
    _print_config_summary(project_root, config, "NightRunner run configuration", emit=ui.log)
    if not _is_git_available():
        raise RuntimeError("Git is not available in PATH. Please install Git first.")
    ensure_git_repo(project_root)
    if config.get("safety", {}).get("require_clean_git", True):
        ensure_clean_worktree(project_root)
    auth = check_auth(project_root)
    if not auth["ok"]:
        env_name = str(config.get("agent", {}).get("api_key_env", "DEEPSEEK_API_KEY"))
        raise RuntimeError(
            f"No API key found.\nRun `nightrunner auth login`\n"
            f"or set {env_name} in your environment."
        )

    if not _baseline_exists(project_root):
        do_baseline = _prompt_yes_no("No baseline found. Run baseline now?", default_yes=True)
        if do_baseline:
            run_baseline(project_root)
        else:
            raise RuntimeError("Baseline is required. Run `nightrunner baseline` first.")

    run_night(project_root, rounds=rounds, dry_run=dry_run, plain=plain)
    summary_path = generate_summary_report(project_root)
    print("NightRunner run completed.")
    print("Summary:")
    print(project_root / "nightrunner_summary.md")
    print(summary_path)
    return summary_path


def setup(
    project_root: Path,
    editable_files: list[str] | None = None,
    train_command: str | None = None,
    metric_name: str | None = None,
    metric_regex: str | None = None,
    lower_is_better: bool | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    run_baseline_now: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Interactive or semi-automated setup wizard for first-time onboarding."""
    print(f"Project root: {project_root}")
    if not _is_git_available():
        raise RuntimeError("Git is not available in PATH. Please install Git first.")

    if not is_git_repo(project_root):
        print("NightRunner requires Git for safe experiment isolation.")
        print("Please run:")
        print("git init")
        print("git add .")
        print("git commit -m \"Initial commit\"")
        auto_init = yes or _prompt_yes_no("Run these commands now?", default_yes=False)
        if not auto_init:
            raise RuntimeError("Setup cancelled. Initialize Git repository and run setup again.")
        run_git(["init"], project_root)
        run_git(["add", "."], project_root)
        try:
            run_git(["commit", "-m", "Initial commit"], project_root)
        except Exception as exc:
            if "nothing to commit" not in str(exc).lower():
                raise

    setup_result = run_setup_flow(
        project_root,
        SetupOptions(
            editable_files=editable_files,
            train_command=train_command,
            metric_name=metric_name,
            metric_regex=metric_regex,
            lower_is_better=lower_is_better,
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            run_baseline_now=run_baseline_now,
            yes=yes,
        ),
    )
    cfg = load_config(project_root)

    print("")
    print("NightRunner setup completed.")
    print("")
    print("Project root:")
    print(project_root)
    print("")
    print("Editable files:")
    for path in setup_result.get("editable_files", []):
        print(f"- {path}")
    print("")
    print("Train command:")
    print(setup_result.get("train_command"))
    print("")
    print("Metric:")
    print(_format_metric(cfg.get("metric", {})))
    print("")
    print("Metric regex:")
    print(cfg.get("metric", {}).get("regex"))
    print("")
    print("Config file:")
    print(project_root / "nightrunner.yaml")
    print("")
    print("State dir:")
    print(project_root / ".nightrunner")
    print("")
    print(_format_setup_commit_hint(project_root))

    auth = check_auth(project_root)
    if not auth["ok"]:
        print("")
        print("No API key found. Run `nightrunner auth login` before `nightrunner run`.")

    should_run_baseline = bool(run_baseline_now)
    if not should_run_baseline and not yes:
        print("")
        print("Commit the generated NightRunner config, then run `nightrunner baseline`.")
        should_run_baseline = _prompt_yes_no("Run baseline now?", default_yes=False)
    if should_run_baseline:
        if cfg.get("safety", {}).get("require_clean_git", True):
            dirty_paths = _git_status_paths(project_root)
            if dirty_paths:
                _raise_setup_dirty_baseline_error(project_root, dirty_paths)
        run_baseline(project_root)
    return setup_result


def status(project_root: Path, plain: bool = False) -> None:
    """Show current NightRunner status without starting new experiments."""
    ui = RunUI(enabled=not plain)
    ensure_git_repo(project_root)
    experiments = load_experiments(project_root)
    best = load_best(project_root) or {}
    baseline = next((r for r in reversed(experiments) if r.get("id") == "baseline"), None)
    latest = experiments[-1] if experiments else {}

    ui.log(f"Project root: {project_root}")
    ui.log(f"Total experiments: {len(experiments)}")
    ui.log(f"Baseline: {baseline.get('status') if isinstance(baseline, dict) else None}")
    ui.log(f"Current best: {best.get('experiment_id')} ({best.get('metric_value')})")
    ui.log(f"Latest experiment: {latest.get('id')}")
    ui.log(f"Latest status: {latest.get('status')}")
    ui.log(f"Latest metric: {latest.get('metric_value')}")
    if latest.get("id"):
        ui.log(f"Latest report: .nightrunner/runs/{latest.get('id')}/report.md")
        ui.log(f"Latest run.log: .nightrunner/runs/{latest.get('id')}/run.log")

    use_rich_table = False
    rich_console = None
    rich_table = None
    if not plain:
        try:
            from rich.console import Console
            from rich.table import Table

            rich_console = Console()
            rich_table = Table
            use_rich_table = bool(getattr(sys.stdout, "isatty", lambda: False)())
        except Exception:
            use_rich_table = False

    ui.log("")
    ui.log("Recent experiments table:")
    if not use_rich_table:
        ui.log("ID | Status | Metric | Delta | API Time | Train Time | Total Time | Hypothesis")
    else:
        assert rich_table is not None and rich_console is not None
        table = rich_table()
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Metric")
        table.add_column("Delta")
        table.add_column("API Time")
        table.add_column("Train Time")
        table.add_column("Total Time")
        table.add_column("Hypothesis")
    baseline_value = None
    if isinstance(baseline, dict) and isinstance(baseline.get("metric_value"), (int, float)):
        baseline_value = float(baseline["metric_value"])
    for rec in experiments[-10:]:
        rid = str(rec.get("id", ""))
        st = str(rec.get("status", ""))
        mv = rec.get("metric_value")
        timings = rec.get("timings", {}) or {}
        api_t = timings.get("api_seconds")
        train_t = timings.get("training_seconds")
        total = timings.get("total_seconds")
        delta = ""
        if baseline_value is not None and isinstance(mv, (int, float)):
            delta = f"{float(mv) - baseline_value:.6f}"
        hyp = str(rec.get("hypothesis", "")).replace("\n", " ")
        if use_rich_table:
            table.add_row(
                rid,
                st,
                f"{mv}",
                delta,
                _fmt_seconds(api_t),
                _fmt_seconds(train_t),
                _fmt_seconds(total),
                _short(hyp, 60),
            )
        else:
            ui.log(
                f"{rid:10} | {st:12} | metric={mv} | delta={delta} | "
                f"api={_fmt_seconds(api_t)} | train={_fmt_seconds(train_t)} | total={_fmt_seconds(total)} | {_short(hyp, 60)}"
            )
    if use_rich_table:
        rich_console.print(table)


def tail(project_root: Path, exp: str | None = None, lines: int = 80, follow: bool = False) -> None:
    """Tail run.log for latest or selected experiment."""
    ensure_git_repo(project_root)
    experiments = load_experiments(project_root)
    if exp is None:
        exp_records = [r for r in experiments if isinstance(r.get("id"), str) and str(r.get("id")).startswith("exp_")]
        if not exp_records:
            raise RuntimeError("No experiment run.log found.")
        exp = str(exp_records[-1]["id"])

    log_path = project_root / ".nightrunner" / "runs" / exp / "run.log"
    if not log_path.exists():
        raise FileNotFoundError(f"run.log not found: {log_path}")

    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-max(1, lines) :]:
        print(line)
    if not follow:
        return

    pos = log_path.stat().st_size
    try:
        while True:
            sleep(0.5)
            if not log_path.exists():
                break
            new_size = log_path.stat().st_size
            if new_size < pos:
                pos = 0
            if new_size == pos:
                continue
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
            pos = new_size
            if chunk:
                print(chunk, end="")
    except KeyboardInterrupt:
        return


def apply_experiment(project_root: Path, exp_id: str) -> Path:
    """Apply selected experiment patch to main worktree after guard checks."""
    ensure_git_repo(project_root)
    try:
        ensure_clean_worktree(project_root)
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot apply experiment while your project has uncommitted changes.\n{exc}") from exc
    config = load_config(project_root)

    patch_path = project_root / ".nightrunner" / "runs" / exp_id / "patch.diff"
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
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
    try:
        apply_patch_to_project(project_root, patch_path)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to apply patch for {exp_id}.\n"
            "Please inspect the patch and current project state before retrying.\n"
            f"Patch: {patch_path}\n"
            f"Details: {exc}"
        ) from exc
    return patch_path


def clean(project_root: Path, branches: bool = False) -> dict[str, Any]:
    """Cleanup temporary worktrees and optionally leftover temporary branches."""
    ensure_git_repo(project_root)
    worktrees_root = get_worktrees_root(project_root)
    removed = 0
    failed: list[str] = []
    removed_branches: list[str] = []
    skipped_branches: list[str] = []
    if not worktrees_root.exists():
        if branches:
            branch_result = cleanup_nightrunner_branches(project_root)
            removed_branches = branch_result["removed_branches"]
            skipped_branches = branch_result["skipped_branches"]
        return {
            "removed": 0,
            "failed": [],
            "removed_branches": removed_branches,
            "skipped_branches": skipped_branches,
        }

    for child in worktrees_root.iterdir():
        if not child.is_dir():
            continue
        try:
            remove_worktree(project_root, child)
            removed += 1
        except Exception:
            failed.append(child.name)
    if branches:
        branch_result = cleanup_nightrunner_branches(project_root)
        removed_branches = branch_result["removed_branches"]
        skipped_branches = branch_result["skipped_branches"]
    return {
        "removed": removed,
        "failed": failed,
        "removed_branches": removed_branches,
        "skipped_branches": skipped_branches,
    }


def check_auth(project_root: Path | None = None) -> dict[str, Any]:
    """Check API key from environment first, then user config."""
    env_name = "DEEPSEEK_API_KEY"
    if project_root is not None:
        try:
            cfg = load_config(project_root)
            env_name = str(cfg.get("agent", {}).get("api_key_env", env_name))
        except Exception:
            pass
    env_value = os.environ.get(env_name)
    if env_value:
        return {
            "ok": True,
            "source": "environment",
            "message": f"{env_name} is set in environment ({mask_api_key(env_value)}).",
            "masked_key": mask_api_key(env_value),
        }
    stored_key = load_api_key("deepseek")
    if stored_key:
        return {
            "ok": True,
            "source": "user_config",
            "message": f"DeepSeek API key is configured in user config ({mask_api_key(stored_key)}).",
            "masked_key": mask_api_key(stored_key),
        }
    return {
        "ok": False,
        "source": "missing",
        "message": f"No API key found. Run `nightrunner auth login` or set {env_name}.",
        "masked_key": None,
    }
