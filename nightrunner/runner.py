"""Core orchestration for NightRunner CLI and Web UI."""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from .agent_api import request_patch
from .auth_store import load_api_key, mask_api_key
from .config import load_config, write_default_config_if_missing
from .console_ui import RunUI
from .diff_analyzer import analyze_diff
from .doctor import collect_doctor_info
from .experiments import (
    collect_file_hashes,
    diff_editable_files,
    get_experiment_paths,
    load_diff_text,
    load_experiment_metadata,
    save_experiment_metadata,
    write_changed_files,
    write_experiment_idea,
)
from .git_ops import (
    GitError,
    cleanup_nightrunner_branches,
    create_worktree,
    ensure_clean_worktree,
    is_git_repo,
    remove_worktree,
    run_git,
    save_diff,
)
from .setup_flow import scan_python_files as setup_scan_python_files
from .setup_flow import validate_editable_paths as setup_validate_editable_paths
from .log_parser import parse_metrics
from .patch_guard import (
    get_changed_files,
    get_new_files,
    validate_changed_files,
    validate_semantic_changes,
)
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
from .sandbox import SandboxManager
from .setup_flow import (
    SetupOptions,
    is_git_available as setup_is_git_available,
    prompt_yes_no as setup_prompt_yes_no,
    run_setup as run_setup_flow,
    update_gitignore as setup_update_gitignore,
)
from .state_store import append_experiment, load_best, load_experiments, next_experiment_id, save_best
from .train_runner import run_training
from .utils import ensure_dir, now_iso, read_json, read_text, write_json, write_text


def _ensure_layout(project_root: Path) -> None:
    ensure_project_layout(build_project_context(project_root))


def _update_gitignore(project_root: Path) -> None:
    setup_update_gitignore(project_root)


def _is_git_available() -> bool:
    return setup_is_git_available()


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    return setup_prompt_yes_no(prompt, default_yes=default_yes)


def _validate_editable_paths(project_root: Path, editable_files: list[str]) -> None:
    setup_validate_editable_paths(project_root, editable_files)


def _scan_python_files(project_root: Path) -> list[str]:
    return setup_scan_python_files(project_root)


def _backend(config: dict[str, Any]) -> str:
    value = str(config.get("execution", {}).get("backend", "sandbox")).strip()
    return "worktree" if value in {"worktree", "git_worktree"} else "sandbox"


def _is_git_dirty(project_root: Path) -> bool:
    if not is_git_repo(project_root):
        return False
    try:
        return bool(run_git(["status", "--porcelain"], project_root).strip())
    except Exception:
        return False


def _emit_backend_warning(project_root: Path, config: dict[str, Any], emit: callable | None = None) -> None:
    out = emit or print
    backend = _backend(config)
    if backend == "sandbox" and _is_git_dirty(project_root):
        out("Your Git working tree has uncommitted changes. This is okay in sandbox mode.")
        out("NightRunner will copy your current files into isolated sandboxes.")
    elif backend == "sandbox" and not is_git_repo(project_root):
        out("Git is not detected. NightRunner can still run sandbox experiments, but diff/apply safety may be reduced.")


def _ensure_backend_ready(project_root: Path, config: dict[str, Any], emit: callable | None = None) -> None:
    backend = _backend(config)
    if backend == "worktree":
        if not is_git_repo(project_root):
            raise RuntimeError("Worktree backend requires a Git repository.")
        if bool(config.get("safety", {}).get("require_clean_git", True)):
            ensure_clean_worktree(project_root)
        return
    _emit_backend_warning(project_root, config, emit=emit)


def _ensure_config_only_mode(config: dict[str, Any]) -> None:
    mode = str(config.get("optimization", {}).get("mode", "standard")).strip().lower()
    if mode != "config_only":
        return
    editable = list(config.get("files", {}).get("editable", []))
    allowed = {".yaml", ".yml", ".json", ".toml"}
    invalid = [path for path in editable if Path(path).suffix.lower() not in allowed]
    if invalid:
        raise RuntimeError(
            "Config files only - safest 模式只允许修改 YAML / JSON / TOML 文件。\n"
            + "\n".join(f"- {path}" for path in invalid)
        )


def _baseline_exists(project_root: Path) -> bool:
    best = load_best(project_root)
    if isinstance(best, dict) and (best.get("is_baseline") is True or best.get("experiment_id") == "baseline"):
        return True
    return any(rec.get("id") == "baseline" and rec.get("status") == "baseline" for rec in load_experiments(project_root))


def _require_best_baseline(project_root: Path) -> dict[str, Any]:
    best = load_best(project_root)
    if not isinstance(best, dict) or best.get("metric_value") is None:
        raise RuntimeError("No baseline metric found. Run `nightrunner baseline` first.")
    return best


def _collect_editable_contents(root: Path, editable: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in editable:
        target = root / entry
        if target.exists() and target.is_file():
            out[target.relative_to(root).as_posix()] = target.read_text(encoding="utf-8", errors="replace")
    return out


def _is_better(new_value: float, best_value: float | None, lower_is_better: bool) -> bool:
    if best_value is None:
        return True
    return new_value < best_value if lower_is_better else new_value > best_value


def _fmt_seconds(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}s"


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _format_metric(metric_cfg: dict[str, Any]) -> str:
    name = str(metric_cfg.get("name", "val_loss"))
    lower = bool(metric_cfg.get("lower_is_better", True))
    return f"{name}, {'lower is better' if lower else 'higher is better'}"


def _print_config_summary(
    project_root: Path,
    config: dict[str, Any],
    title: str = "NightRunner configuration",
    emit: callable | None = None,
) -> None:
    out = emit or print
    out("")
    out(title)
    out("")
    out("Project root:")
    out(project_root)
    out("")
    out("Editable files:")
    for path in config.get("files", {}).get("editable", []):
        out(f"- {path}")
    out("")
    out("Train command:")
    out(config.get("execution", {}).get("train_command"))
    out("")
    out("Execution backend:")
    out(config.get("execution", {}).get("backend", "sandbox"))
    out("")
    out("Sandbox root:")
    out(config.get("sandbox", {}).get("root", ".nightrunner/sandboxes"))
    out("")
    out("Metric:")
    out(_format_metric(config.get("metric", {})))
    out("")


def _session_state_path(project_root: Path) -> Path:
    return ensure_dir(project_root / ".nightrunner" / "state") / "ui_session.json"


def _save_session_state(project_root: Path, payload: dict[str, Any]) -> None:
    write_json(_session_state_path(project_root), payload)


def _load_session_state(project_root: Path) -> dict[str, Any]:
    return read_json(_session_state_path(project_root), default={"running": False}) or {"running": False}


def _load_baseline_record(project_root: Path) -> dict[str, Any] | None:
    for rec in reversed(load_experiments(project_root)):
        if rec.get("id") == "baseline":
            return rec
    return None


def _create_workspace(project_root: Path, config: dict[str, Any], exp_id: str) -> dict[str, Any]:
    backend = _backend(config)
    if backend == "worktree":
        if not is_git_repo(project_root):
            raise RuntimeError("Git worktree backend requires a Git repository.")
        worktree_path, branch_name = create_worktree(project_root, exp_id)
        return {
            "backend": backend,
            "project_dir": worktree_path,
            "worktree_path": worktree_path,
            "branch_name": branch_name,
        }

    manager = SandboxManager.from_config(project_root, config)
    sandbox = manager.create_sandbox(exp_id)
    return {
        "backend": "sandbox",
        "project_dir": sandbox.project_dir,
        "sandbox_dir": sandbox.root_dir,
    }


def _cleanup_workspace(project_root: Path, workspace: dict[str, Any] | None, keep: bool) -> None:
    if not workspace:
        return
    if workspace.get("backend") == "worktree":
        worktree_path = workspace.get("worktree_path")
        if isinstance(worktree_path, Path) and worktree_path.exists() and not keep:
            try:
                remove_worktree(project_root, worktree_path, workspace.get("branch_name"))
            except GitError:
                pass
    # Sandboxes are preserved for later diff review and explicit apply.


def _persist_record(project_root: Path, record: dict[str, Any]) -> None:
    save_experiment_metadata(project_root, str(record["id"]), record)
    _save_session_state(
        project_root,
        {
            "running": record.get("status") == "running",
            "current_exp_id": record.get("id"),
            "current_round": record.get("current_round"),
            "total_rounds": record.get("total_rounds"),
            "stage": record.get("stage"),
            "status": record.get("status"),
            "best": load_best(project_root),
            "updated_at": now_iso(),
        },
    )


def init_project(
    project_root: Path,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
    update_gitignore: bool = False,
) -> dict[str, Any]:
    """Initialize NightRunner runtime files in a project."""
    project_root = project_root.resolve()
    _ensure_layout(project_root)
    config_path = write_default_config_if_missing(
        project_root,
        editable_files=editable_files,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
    )
    state_dir = project_root / ".nightrunner" / "state"
    if not (state_dir / "best.json").exists():
        write_json(state_dir / "best.json", {})
    if not (state_dir / "experiments.jsonl").exists():
        write_text(state_dir / "experiments.jsonl", "")
    if update_gitignore:
        _update_gitignore(project_root)
    return {
        "project_root": str(project_root),
        "config": str(config_path),
        "nightrunner_dir": str(project_root / ".nightrunner"),
    }


def run_baseline(project_root: Path, force: bool = False) -> Path:
    """Run baseline training in an isolated sandbox and save best.json."""
    _ensure_layout(project_root)
    config = load_config(project_root)
    _ensure_config_only_mode(config)
    _ensure_backend_ready(project_root, config)
    paths = get_experiment_paths(project_root, "baseline")
    if paths.metadata_path.exists() and not force:
        existing = load_experiment_metadata(project_root, "baseline")
        if existing.get("status") == "baseline" and existing.get("metric_value") is not None:
            return paths.report_path

    editable = list(config.get("files", {}).get("editable", []))
    record: dict[str, Any] = {
        "id": "baseline",
        "status": "running",
        "stage": "运行基线",
        "metric_name": config.get("metric", {}).get("name", "metric"),
        "metric_value": None,
        "baseline_metric": None,
        "is_improvement": False,
        "editable_files": editable,
        "base_file_hashes": collect_file_hashes(project_root, editable),
        "created_at": now_iso(),
        "backend": _backend(config),
        "train_command": config.get("execution", {}).get("train_command", "python train.py"),
        "hypothesis": "在隔离沙箱中运行基线训练，建立参考指标。",
        "reason": "NightRunner 需要 baseline 才能比较实验改进。",
        "expected_effect": "记录 baseline 指标并确保主项目不被 NightRunner 直接改写。",
        "risk": "训练命令可能失败、超时，或者日志中没有目标指标。",
        "timings": {
            "workspace_seconds": 0.0,
            "training_seconds": 0.0,
            "metric_parse_seconds": 0.0,
            "total_seconds": 0.0,
        },
        "is_baseline": True,
    }
    _persist_record(project_root, record)
    workspace = _create_workspace(project_root, config, "baseline")
    if workspace.get("backend") == "sandbox":
        record["sandbox_dir"] = _safe_relative(Path(workspace["project_dir"]), project_root)
    t_total = perf_counter()
    try:
        t_train = perf_counter()
        run_result = run_training(
            command=record["train_command"],
            cwd=Path(workspace["project_dir"]),
            log_path=paths.log_path,
            timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 3600)),
        )
        record["timings"]["training_seconds"] = perf_counter() - t_train
        record["run_log_path"] = _safe_relative(paths.log_path, project_root)
        if run_result.get("timeout"):
            record["status"] = "baseline_error"
            record["decision"] = "基线训练超时。"
        elif run_result.get("returncode") not in (0,):
            record["status"] = "baseline_error"
            record["decision"] = "基线训练进程失败。"
        else:
            t_parse = perf_counter()
            metrics = parse_metrics(paths.log_path, config.get("metric", {}).get("name", "val_loss"), config.get("metric", {}).get("regex"))
            record["timings"]["metric_parse_seconds"] = perf_counter() - t_parse
            record["metrics"] = metrics
            record["metric_value"] = metrics.get("metric_value")
            write_json(paths.dir / "metrics.json", metrics)
            if metrics.get("metric_value") is None:
                record["status"] = "baseline_error"
                record["decision"] = "基线训练完成，但没有从日志中提取到指标。"
            else:
                record["status"] = "baseline"
                record["decision"] = "基线指标已保存。"
                save_best(
                    project_root,
                    {
                        "experiment_id": "baseline",
                        "metric_name": metrics.get("metric_name"),
                        "metric_value": metrics.get("metric_value"),
                        "patch_path": None,
                        "updated_at": now_iso(),
                        "is_baseline": True,
                    },
                )
    except Exception as exc:  # pragma: no cover
        record["status"] = "baseline_error"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        write_text(paths.dir / "error.txt", record["traceback"])
    finally:
        record["completed_at"] = now_iso()
        record["timings"]["total_seconds"] = perf_counter() - t_total
        _persist_record(project_root, record)
        generate_experiment_report(project_root, "baseline", record)
        append_experiment(project_root, record)
        generate_summary_report(project_root)
        _cleanup_workspace(project_root, workspace, keep=False)
    return paths.report_path


def run_night(
    project_root: Path,
    rounds: int,
    dry_run: bool = False,
    plain: bool = False,
    ui: RunUI | None = None,
) -> Path:
    """Run N rounds of NightRunner experiments."""
    if rounds <= 0:
        raise ValueError("--rounds must be > 0")
    _ensure_layout(project_root)
    config = load_config(project_root)
    _ensure_config_only_mode(config)
    _ensure_backend_ready(project_root, config, emit=(ui.log if ui else print))
    if not dry_run and not _baseline_exists(project_root):
        run_baseline(project_root)

    ui = ui or RunUI(enabled=not plain)
    _print_config_summary(project_root, config, "NightRunner configuration", emit=ui.log)
    ui.start_run(project_root, config, rounds, baseline_info="sandbox baseline")
    _save_session_state(project_root, {"running": True, "current_round": 0, "total_rounds": rounds, "stage": "准备运行", "updated_at": now_iso()})

    editable = list(config.get("files", {}).get("editable", []))
    protected = list(config.get("files", {}).get("protected", []))
    metric_cfg = config.get("metric", {})

    for index in range(rounds):
        exp_id = next_experiment_id(project_root)
        paths = get_experiment_paths(project_root, exp_id)
        baseline = _load_baseline_record(project_root) or {}
        record: dict[str, Any] = {
            "id": exp_id,
            "status": "running",
            "stage": "创建隔离环境",
            "metric_name": metric_cfg.get("name", "metric"),
            "metric_value": None,
            "baseline_metric": baseline.get("metric_value"),
            "is_improvement": False,
            "editable_files": editable,
            "base_file_hashes": collect_file_hashes(project_root, editable),
            "created_at": now_iso(),
            "current_round": index + 1,
            "total_rounds": rounds,
            "backend": _backend(config),
            "train_command": config.get("execution", {}).get("train_command", "python train.py"),
            "timings": {
                "workspace_seconds": 0.0,
                "prompt_seconds": 0.0,
                "api_seconds": 0.0,
                "patch_seconds": 0.0,
                "training_seconds": 0.0,
                "metric_parse_seconds": 0.0,
                "report_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
        workspace: dict[str, Any] | None = None
        ui.start_experiment(exp_id, index + 1, rounds)
        t_total = perf_counter()
        try:
            _persist_record(project_root, record)
            t_stage = perf_counter()
            workspace = _create_workspace(project_root, config, exp_id)
            record["timings"]["workspace_seconds"] = perf_counter() - t_stage
            if workspace.get("backend") == "sandbox":
                record["sandbox_dir"] = _safe_relative(Path(workspace["project_dir"]), project_root)
            else:
                record["worktree_path"] = _safe_relative(Path(workspace["project_dir"]), project_root)
                record["branch_name"] = workspace.get("branch_name")
            _persist_record(project_root, record)

            ui.set_stage("读取可编辑文件")
            record["stage"] = "读取可编辑文件"
            _persist_record(project_root, record)
            t_stage = perf_counter()
            recent = load_experiments(project_root)[-5:]
            best = _require_best_baseline(project_root) if not dry_run else load_best(project_root)
            file_contents = _collect_editable_contents(Path(workspace["project_dir"]), editable)
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(config, best, recent, file_contents)
            record["timings"]["prompt_seconds"] = perf_counter() - t_stage
            if config.get("logging", {}).get("save_request", True):
                write_json(paths.dir / "request.json", {"system_prompt": system_prompt, "user_prompt": user_prompt})

            ui.set_stage("调用模型")
            record["stage"] = "调用模型"
            _persist_record(project_root, record)
            t_stage = perf_counter()
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
            record["timings"]["api_seconds"] = perf_counter() - t_stage
            if config.get("logging", {}).get("save_response", True):
                write_json(paths.dir / "response.json", {"raw": raw})

            ui.set_stage("解析模型输出")
            record["stage"] = "解析模型输出"
            _persist_record(project_root, record)
            parsed = parse_model_response(raw)
            record["hypothesis"] = parsed.get("hypothesis", "")
            record["reason"] = parsed.get("reason", "")
            record["expected_effect"] = parsed.get("expected_effect", "")
            record["risk"] = parsed.get("risk", "")
            write_experiment_idea(paths, record)

            ui.set_stage("应用改动")
            record["stage"] = "应用改动"
            _persist_record(project_root, record)
            t_stage = perf_counter()
            if workspace.get("backend") == "sandbox" and "edits" not in parsed:
                raise InvalidModelResponse("Sandbox backend only supports edits output.")
            if "edits" in parsed:
                applied_edits = apply_search_replace_edits(Path(workspace["project_dir"]), parsed["edits"])
                record["applied_edits"] = applied_edits
                write_json(paths.dir / "applied_edits.json", {"applied_edits": applied_edits})
                changed_files = sorted(
                    {
                        str(item.get("file", "")).replace("\\", "/")
                        for item in applied_edits
                        if item.get("file")
                    }
                )
                new_files: list[str] = []
            else:
                write_text(paths.dir / "model.patch.diff", parsed["patch"])
                apply_patch(Path(workspace["project_dir"]), parsed["patch"])
                changed_files = get_changed_files(Path(workspace["project_dir"]))
                new_files = get_new_files(Path(workspace["project_dir"]))

            guard_result = validate_changed_files(
                changed_files=changed_files,
                editable_files=editable,
                protected_files=protected,
                allow_new_files=config.get("safety", {}).get("allow_new_files", False),
                allow_dependency_changes=config.get("safety", {}).get("allow_dependency_changes", False),
                new_files=new_files,
                run_dir=paths.dir,
            )
            if not guard_result["ok"]:
                record["status"] = "violation"
                record["violations"] = guard_result["violations"]
                raise RuntimeError("Patch guard failed.")

            semantic_result = validate_semantic_changes(
                original_root=project_root,
                candidate_root=Path(workspace["project_dir"]),
                changed_files=changed_files,
                protected_terms=config.get("safety", {}).get("protected_terms", []),
                enabled=bool(config.get("safety", {}).get("semantic_guard", True)),
                allow_protected_term_edits=bool(config.get("safety", {}).get("allow_protected_term_edits", False)),
                run_dir=paths.dir,
            )
            if not semantic_result["ok"]:
                record["status"] = "violation"
                record["violations"] = semantic_result["violations"]
                details = semantic_result["violations"][0].get("message") if semantic_result["violations"] else "Semantic guard failed."
                raise RuntimeError(str(details))

            if workspace.get("backend") == "sandbox":
                diff_text, changed_files = diff_editable_files(project_root, Path(workspace["project_dir"]), editable)
                write_text(paths.patch_path, diff_text)
            else:
                save_diff(Path(workspace["project_dir"]), paths.patch_path)
                changed_files = get_changed_files(Path(workspace["project_dir"]))
                diff_text = read_text(paths.patch_path)

            record["changed_files"] = changed_files
            record["files_changed"] = changed_files
            record["diff_summary"] = analyze_diff(diff_text) if diff_text.strip() else {}
            write_changed_files(paths, changed_files)
            record["timings"]["patch_seconds"] = perf_counter() - t_stage
            _persist_record(project_root, record)

            if dry_run:
                record["status"] = "discard"
                record["decision"] = "Dry run 模式：已校验改动，但未执行训练。"
            else:
                ui.set_stage("运行训练命令")
                record["stage"] = "运行训练命令"
                _persist_record(project_root, record)
                t_train = perf_counter()
                run_result = run_training(
                    command=config.get("execution", {}).get("train_command", "python train.py"),
                    cwd=Path(workspace["project_dir"]),
                    log_path=paths.log_path,
                    timeout_seconds=int(config.get("execution", {}).get("timeout_seconds", 3600)),
                )
                record["timings"]["training_seconds"] = perf_counter() - t_train
                record["run_log_path"] = _safe_relative(paths.log_path, project_root)
                if run_result.get("timeout"):
                    record["status"] = "timeout"
                    record["decision"] = "训练超时。"
                elif run_result.get("returncode") not in (0,):
                    record["status"] = "crash"
                    record["decision"] = "训练进程失败。"
                else:
                    ui.set_stage("解析指标")
                    record["stage"] = "解析指标"
                    _persist_record(project_root, record)
                    t_parse = perf_counter()
                    metrics = parse_metrics(paths.log_path, metric_cfg.get("name", "val_loss"), metric_cfg.get("regex"))
                    record["timings"]["metric_parse_seconds"] = perf_counter() - t_parse
                    write_json(paths.dir / "metrics.json", metrics)
                    record["metrics"] = metrics
                    record["metric_value"] = metrics.get("metric_value")
                    if metrics.get("metric_value") is None:
                        record["status"] = "crash"
                        record["decision"] = "训练完成，但没有从日志中提取到主指标。"
                    else:
                        baseline_metric = record.get("baseline_metric")
                        current_best = load_best(project_root) or {}
                        current_best_value = current_best.get("metric_value") if isinstance(current_best, dict) else None
                        metric_value = float(metrics["metric_value"])
                        if isinstance(baseline_metric, (int, float)):
                            record["delta_vs_baseline"] = metric_value - float(baseline_metric)
                            record["is_improvement"] = _is_better(metric_value, float(baseline_metric), bool(metric_cfg.get("lower_is_better", True)))
                        is_best = _is_better(
                            metric_value,
                            float(current_best_value) if isinstance(current_best_value, (int, float)) else None,
                            bool(metric_cfg.get("lower_is_better", True)),
                        )
                        if is_best:
                            record["status"] = "keep"
                            record["decision"] = "指标优于当前 best experiment。"
                            save_best(
                                project_root,
                                {
                                    "experiment_id": exp_id,
                                    "metric_name": metrics.get("metric_name"),
                                    "metric_value": metrics.get("metric_value"),
                                    "patch_path": f".nightrunner/experiments/{exp_id}/patch.diff",
                                    "updated_at": now_iso(),
                                },
                            )
                        else:
                            record["status"] = "discard"
                            record["decision"] = "指标没有超过当前 best experiment。"
        except (InvalidModelResponse, PatchApplyError, SearchReplaceError) as exc:
            record["status"] = "patch_error" if not isinstance(exc, InvalidModelResponse) else "invalid_response"
            record["error"] = str(exc)
        except Exception as exc:  # pragma: no cover
            if record.get("status") == "running":
                record["status"] = "crash"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            write_text(paths.dir / "error.txt", record.get("traceback", ""))
        finally:
            record["stage"] = "已完成"
            record["completed_at"] = now_iso()
            record["timings"]["total_seconds"] = perf_counter() - t_total
            _persist_record(project_root, record)
            generate_experiment_report(project_root, exp_id, record)
            append_experiment(project_root, record)
            generate_summary_report(project_root)
            ui.finish_experiment(record)
            keep_workspace = bool(record.get("status") == "keep" and workspace and workspace.get("backend") == "worktree")
            _cleanup_workspace(project_root, workspace, keep=keep_workspace)

    summary = generate_summary_report(project_root)
    _save_session_state(project_root, {"running": False, "current_round": rounds, "total_rounds": rounds, "stage": "完成", "best": load_best(project_root), "updated_at": now_iso()})
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
    _ensure_config_only_mode(config)
    _ensure_backend_ready(project_root, config)
    auth = check_auth(project_root)
    if not auth["ok"]:
        env_name = str(config.get("agent", {}).get("api_key_env", "DEEPSEEK_API_KEY"))
        raise RuntimeError(f"No API key found.\nRun `nightrunner auth login`\nor set {env_name} in your environment.")
    if not dry_run and not _baseline_exists(project_root):
        run_baseline(project_root)
    return run_night(project_root, rounds=rounds, dry_run=dry_run, plain=plain)


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
    if is_git_repo(project_root):
        print("Git repository: yes")
        try:
            _emit_backend_warning(project_root, load_config(project_root))
        except FileNotFoundError:
            print("Your Git working tree may be dirty. This is okay in sandbox mode.")
            print("NightRunner will copy your current files into isolated sandboxes.")
    else:
        print("Git is not detected. NightRunner can still run sandbox experiments.")
    print("Editable file permission only defines where NightRunner may propose changes.")
    print("Protected keys and protected regions are still enforced inside editable files.")
    result = run_setup_flow(
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
    print("")
    print("NightRunner runs experiments in isolated sandboxes.")
    print("Your main project files will not be changed unless you apply an experiment.")
    print("Git commits are optional and not required before running.")
    if run_baseline_now:
        run_baseline(project_root)
    return result


def doctor(project_root: Path) -> dict[str, Any]:
    """Collect project diagnostics for CLI/UI display."""
    return collect_doctor_info(project_root)


def status(project_root: Path, plain: bool = False) -> None:
    """Show current NightRunner status without starting new experiments."""
    ui = RunUI(enabled=not plain)
    info = collect_doctor_info(project_root)
    experiments = load_experiments(project_root)
    best = load_best(project_root) or {}
    baseline = _load_baseline_record(project_root) or {}
    latest = experiments[-1] if experiments else {}
    session = _load_session_state(project_root)
    ui.log(f"Project root: {project_root}")
    ui.log(f"Git repository: {'yes' if info.get('git_repository') else 'no'}")
    ui.log(f"Git status: {info.get('git_status')}")
    ui.log(f"Backend: {info.get('backend')}")
    ui.log(f"Running: {session.get('running')}")
    ui.log(f"Current stage: {session.get('stage')}")
    ui.log(f"Baseline: {baseline.get('metric_value')}")
    ui.log(f"Current best: {best.get('experiment_id')} ({best.get('metric_value')})")
    ui.log(f"Latest experiment: {latest.get('id')}")
    ui.log(f"Latest status: {latest.get('status')}")
    ui.log(f"Latest metric: {latest.get('metric_value')}")
    ui.log("")
    ui.log("Recent experiments table:")
    ui.log("ID | Status | Metric | Delta | Train Time | Total Time | Hypothesis")
    baseline_value = baseline.get("metric_value") if isinstance(baseline.get("metric_value"), (int, float)) else None
    for rec in experiments[-10:]:
        timings = rec.get("timings", {}) or {}
        delta = ""
        if baseline_value is not None and isinstance(rec.get("metric_value"), (int, float)):
            delta = f"{float(rec['metric_value']) - float(baseline_value):.6f}"
        ui.log(
            f"{str(rec.get('id', '')):10} | {str(rec.get('status', '')):12} | metric={rec.get('metric_value')} | "
            f"delta={delta} | train={_fmt_seconds(timings.get('training_seconds'))} | "
            f"total={_fmt_seconds(timings.get('total_seconds'))} | {str(rec.get('hypothesis', ''))[:60]}"
        )


def tail(project_root: Path, exp: str | None = None, lines: int = 80, follow: bool = False) -> None:
    """Tail train.log for latest or selected experiment."""
    experiments = load_experiments(project_root)
    if exp is None:
        exp_records = [r for r in experiments if isinstance(r.get("id"), str) and str(r.get("id")).startswith("exp_")]
        if not exp_records:
            raise RuntimeError("No experiment train.log found.")
        exp = str(exp_records[-1]["id"])
    paths = get_experiment_paths(project_root, exp)
    log_path = paths.log_path
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
            new_size = log_path.stat().st_size
            if new_size < pos:
                pos = 0
            if new_size == pos:
                continue
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
            pos = new_size
            if chunk:
                print(chunk, end="")
    except KeyboardInterrupt:
        return


def preview_experiment(project_root: Path, exp_id: str) -> dict[str, Any]:
    """Return diff and conflict information before apply."""
    metadata = load_experiment_metadata(project_root, exp_id)
    if not metadata:
        raise FileNotFoundError(f"Experiment metadata not found: {exp_id}")
    editable_files = list(metadata.get("editable_files", []))
    base_hashes = metadata.get("base_file_hashes", {}) if isinstance(metadata.get("base_file_hashes"), dict) else {}
    current_hashes = collect_file_hashes(project_root, editable_files)
    conflicts = [path for path in editable_files if current_hashes.get(path) != base_hashes.get(path)]
    return {
        "metadata": metadata,
        "diff": load_diff_text(project_root, exp_id),
        "conflicts": conflicts,
        "safe_to_apply": len(conflicts) == 0,
    }


def apply_experiment(
    project_root: Path,
    exp_id: str,
    confirm: bool = True,
    allow_non_keep: bool = False,
) -> Path:
    """Apply selected experiment changes back to the main project."""
    preview = preview_experiment(project_root, exp_id)
    metadata = preview["metadata"]
    status_value = str(metadata.get("status", "unknown"))
    if not allow_non_keep and status_value != "keep":
        raise RuntimeError(
            f"Only experiments with status 'keep' can be applied. "
            f"{exp_id} currently has status '{status_value}'."
        )
    if preview["conflicts"]:
        raise RuntimeError(
            "Cannot apply experiment because original editable files changed:\n"
            + "\n".join(f"- {path}" for path in preview["conflicts"])
        )
    if confirm and sys.stdin.isatty():
        print(preview["diff"] or "(no diff)")
        if not _prompt_yes_no("Apply this experiment to your main project now?", default_yes=False):
            raise RuntimeError("Apply cancelled.")
    sandbox_dir = metadata.get("sandbox_dir")
    changed_files = list(metadata.get("changed_files") or metadata.get("files_changed") or [])
    if isinstance(sandbox_dir, str) and sandbox_dir:
        sandbox_root = (project_root / sandbox_dir).resolve()
        for rel_path in changed_files:
            src = sandbox_root / rel_path
            dst = project_root / rel_path
            if not src.exists():
                raise FileNotFoundError(f"Sandbox file not found: {src}")
            ensure_dir(dst.parent)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return get_experiment_paths(project_root, exp_id).patch_path
    raise RuntimeError("Experiment cannot be applied because sandbox files are unavailable.")


def clean(project_root: Path, branches: bool = False) -> dict[str, Any]:
    """Cleanup local sandboxes and optionally leftover worktrees/branches."""
    ctx = build_project_context(project_root)
    removed = 0
    failed: list[str] = []
    for root in [ctx.sandboxes_root, ctx.worktrees_root]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                if root == ctx.worktrees_root and is_git_repo(project_root):
                    remove_worktree(project_root, child)
                else:
                    shutil.rmtree(child)
                removed += 1
            except Exception:
                failed.append(child.name)
    removed_branches: list[str] = []
    skipped_branches: list[str] = []
    if branches and is_git_repo(project_root):
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
            env_name = str(load_config(project_root).get("agent", {}).get("api_key_env", env_name))
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
