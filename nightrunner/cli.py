"""Argparse CLI entrypoint for NightRunner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .report import generate_summary_report
from .runner import apply_experiment, check_auth, clean, init_project, run_night
from .state_store import load_best, load_experiments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nightrunner", description="NightRunner 命令行工具")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="在当前 Git 项目中初始化 NightRunner。")

    p_night = sub.add_parser("night", help="运行 N 轮自动化实验。")
    p_night.add_argument("--rounds", type=int, default=1, help="实验轮数。")
    p_night.add_argument(
        "--dry-run",
        action="store_true",
        help="仅执行到补丁校验，跳过训练。",
    )

    p_report = sub.add_parser("report", help="生成并打印汇总报告。")
    p_report.add_argument("--json", action="store_true", help="以 JSON 格式打印汇总。")

    p_apply = sub.add_parser("apply", help="将某个实验补丁应用到主工作区。")
    p_apply.add_argument("exp_id", help="实验 ID，例如 exp_0003")

    sub.add_parser("clean", help="删除 NightRunner 临时 worktree。")
    sub.add_parser("auth", help="检查是否已配置 DEEPSEEK_API_KEY。")
    return parser


def _project_root() -> Path:
    return Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = _project_root()

    try:
        if args.command == "init":
            result = init_project(project_root)
            print("NightRunner 初始化完成。")
            print(f"项目根目录: {result['project_root']}")
            print(f"配置文件: {result['config']}")
            print(f"状态目录: {result['nightrunner_dir']}")
            return 0

        if args.command == "night":
            if args.rounds <= 0:
                raise ValueError("--rounds 必须大于 0")
            summary_path = run_night(project_root, rounds=args.rounds, dry_run=bool(args.dry_run))
            print(f"夜间运行完成。汇总: {summary_path}")
            return 0

        if args.command == "report":
            summary_path = generate_summary_report(project_root)
            experiments = load_experiments(project_root)
            best = load_best(project_root) or {}
            counts: dict[str, int] = {}
            for rec in experiments:
                status = str(rec.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
            print(f"汇总已生成: {summary_path}")
            print(f"实验总数: {len(experiments)}")
            print(f"keep: {counts.get('keep', 0)}")
            print(f"discard: {counts.get('discard', 0)}")
            print(f"crash: {counts.get('crash', 0)}")
            print(f"violation: {counts.get('violation', 0)}")
            print(f"当前最佳实验: {best.get('experiment_id')}")
            print(f"当前最佳指标: {best.get('metric_value')}")
            print(f"建议查看报告: {summary_path}")
            if args.json:
                payload = {
                    "summary_path": str(summary_path),
                    "total_experiments": len(experiments),
                    "counts": counts,
                    "best": best,
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "apply":
            patch_path = apply_experiment(project_root, args.exp_id)
            print(f"已应用补丁: {patch_path}")
            print("补丁已应用到主工作区。请检查变更后手动提交。")
            return 0

        if args.command == "clean":
            result = clean(project_root)
            print(f"已删除 worktree 数量: {result['removed']}")
            if result["failed"]:
                print("删除失败:", ", ".join(result["failed"]))
            return 0

        if args.command == "auth":
            result = check_auth()
            print(result["message"])
            return 0 if result["ok"] else 1

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
