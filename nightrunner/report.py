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

    counts = {
        "keep": 0,
        "discard": 0,
        "crash": 0,
        "timeout": 0,
        "violation": 0,
        "api_error": 0,
        "invalid_response": 0,
        "patch_error": 0,
    }
    for rec in experiments:
        status = rec.get("status")
        if status in counts:
            counts[status] += 1

    lines = [
        "# NightRunner 汇总",
        "",
        "## 总览",
        f"- 实验总数: {len(experiments)}",
        f"- Keep: {counts['keep']}",
        f"- Discard: {counts['discard']}",
        f"- Crash: {counts['crash']}",
        f"- Timeout: {counts['timeout']}",
        f"- Violation: {counts['violation']}",
        f"- API Error: {counts['api_error']}",
        f"- Invalid Response: {counts['invalid_response']}",
        f"- Patch Error: {counts['patch_error']}",
        "",
        "## 当前最佳",
        f"- 实验: {(best or {}).get('experiment_id')}",
        f"- 指标: {(best or {}).get('metric_name')}={(best or {}).get('metric_value')}",
        f"- 补丁: {(best or {}).get('patch_path')}",
        "",
        "## 实验表",
        "| ID | Status | Metric | Hypothesis | Report |",
        "|---|---|---:|---|---|",
    ]
    for rec in experiments:
        rid = rec.get("id", "")
        status = rec.get("status", "")
        metric = rec.get("metric_value")
        hypothesis = str(rec.get("hypothesis", "")).replace("|", "/")
        report_path = f".nightrunner/runs/{rid}/report.md"
        lines.append(f"| {rid} | {status} | {metric} | {hypothesis} | {report_path} |")

    lines += [
        "",
        "## 建议下一步",
        "查看当前最佳实验的报告和补丁，如果确认应用，请手动执行 `nightrunner apply <exp_id>`。",
    ]
    summary_path = project_root / ".nightrunner" / "summary.md"
    write_text(summary_path, "\n".join(lines))
    return summary_path
