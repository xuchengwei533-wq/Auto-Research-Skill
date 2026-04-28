"""Report generation for NightRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .state_store import load_best, load_experiments
from .utils import write_text


def generate_experiment_report(project_root: Path, exp_id: str, data: dict[str, Any]) -> None:
    """Generate per-experiment report markdown."""
    run_dir = project_root / ".nightrunner" / "runs" / exp_id
    metrics = data.get("metrics", {})
    files_changed = data.get("files_changed", [])
    diff_summary = data.get("diff_summary", {})
    applied_edits = data.get("applied_edits", [])
    used_legacy_patch_mode = bool(data.get("used_legacy_patch_mode", False))

    lines = [
        f"# NightRunner 实验报告 {exp_id}",
        "",
        "## 状态",
        str(data.get("status", "unknown")),
        "",
        "## 假设",
        str(data.get("hypothesis", "")),
        "",
        "## 原因",
        str(data.get("reason", "")),
        "",
        "## 预期效果",
        str(data.get("expected_effect", "")),
        "",
        "## 风险",
        str(data.get("risk", "")),
        "",
        "## 变更文件",
        ", ".join(files_changed) if files_changed else "(无)",
        "",
        "## 超参数 / 代码变更摘要",
        str(diff_summary),
        "",
        "## 应用的 Edits",
    ]
    if applied_edits:
        for edit in applied_edits:
            lines.append(f"- file: {edit.get('file')}")
            lines.append(f"- old_text_preview: {edit.get('old_text_preview')}")
            lines.append(f"- new_text_preview: {edit.get('new_text_preview')}")
    elif used_legacy_patch_mode:
        lines.append("本次实验使用了 legacy patch 兼容模式。")
    else:
        lines.append("(无)")

    lines += [
        "",
        "## 指标",
        f"- metric_name: {metrics.get('metric_name')}",
        f"- metric_value: {metrics.get('metric_value')}",
        f"- peak_vram_mb: {metrics.get('peak_vram_mb')}",
        f"- training_seconds: {metrics.get('training_seconds')}",
        f"- num_steps: {metrics.get('num_steps')}",
        "",
        "## 决策",
        str(data.get("decision", "")),
        "",
        "## Patch",
        str(data.get("patch_path", run_dir / "patch.diff")),
        "",
        "## 运行日志",
        str(data.get("run_log_path", run_dir / "run.log")),
        "",
    ]
    write_text(run_dir / "report.md", "\n".join(lines))


def generate_summary_report(project_root: Path) -> Path:
    """Generate summary report markdown from state."""
    experiments = load_experiments(project_root)
    best = load_best(project_root)
    baseline_rec = next((r for r in reversed(experiments) if r.get("id") == "baseline"), None)

    counts = {
        "keep": 0,
        "discard": 0,
        "crash": 0,
        "timeout": 0,
        "violation": 0,
        "api_error": 0,
        "invalid_response": 0,
        "patch_error": 0,
        "setup_error": 0,
        "baseline_error": 0,
    }
    for rec in experiments:
        status = rec.get("status")
        if status in counts:
            counts[status] += 1
    baseline_status = baseline_rec.get("status") if isinstance(baseline_rec, dict) else None
    baseline_metric = baseline_rec.get("metric_value") if isinstance(baseline_rec, dict) else None

    lines = [
        "# NightRunner Summary",
        "",
        "## Project",
        f"- Project root: {project_root}",
        f"- Config: {project_root / 'nightrunner.yaml'}",
        f"- Summary file: {project_root / 'nightrunner_summary.md'}",
        "",
        "## Overall",
        f"- Total experiments: {len(experiments)}",
        f"- Baseline: {baseline_status}",
        f"- Keep: {counts['keep']}",
        f"- Discard: {counts['discard']}",
        f"- Crash: {counts['crash']}",
        f"- Timeout: {counts['timeout']}",
        f"- Violation: {counts['violation']}",
        f"- API Error: {counts['api_error']}",
        f"- Invalid Response: {counts['invalid_response']}",
        f"- Patch Error: {counts['patch_error']}",
        f"- Setup Error: {counts['setup_error']}",
        f"- Baseline Error: {counts['baseline_error']}",
        "",
        "## Baseline",
        f"- Metric: {baseline_metric}",
        f"- Report: .nightrunner/runs/baseline/report.md",
        "",
        "## Current Best",
        f"- Experiment: {(best or {}).get('experiment_id')}",
        f"- Metric: {(best or {}).get('metric_name')}={(best or {}).get('metric_value')}",
        f"- Patch: {(best or {}).get('patch_path')}",
        "",
        "## Experiment Table",
        "| ID | Status | Metric | Δ vs Baseline | Hypothesis | Report |",
        "|---|---|---:|---:|---|---|",
    ]
    baseline_value: float | None = None
    if isinstance(baseline_metric, (int, float)):
        baseline_value = float(baseline_metric)
    for rec in experiments:
        rid = rec.get("id", "")
        status = rec.get("status", "")
        metric = rec.get("metric_value")
        hypothesis = str(rec.get("hypothesis", "")).replace("|", "/")
        report_path = f".nightrunner/runs/{rid}/report.md"
        delta = ""
        if baseline_value is not None and isinstance(metric, (int, float)):
            delta = f"{float(metric) - baseline_value:.6f}"
        lines.append(f"| {rid} | {status} | {metric} | {delta} | {hypothesis} | {report_path} |")

    lines += [
        "",
        "## Recommended Next Step",
    ]
    best_exp = (best or {}).get("experiment_id")
    if isinstance(best_exp, str) and best_exp and best_exp != "baseline":
        lines.append(f"`nightrunner apply {best_exp}`")
    else:
        lines.append("先运行 `nightrunner night --rounds 1` 生成可应用实验。")
    summary_path = project_root / ".nightrunner" / "summary.md"
    summary_root_path = project_root / "nightrunner_summary.md"
    content = "\n".join(lines)
    write_text(summary_path, content)
    write_text(summary_root_path, content)
    return summary_path
