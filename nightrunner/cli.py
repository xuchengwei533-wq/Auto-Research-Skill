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
    parser = argparse.ArgumentParser(prog="nightrunner", description="NightRunner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize NightRunner in current git project.")

    p_night = sub.add_parser("night", help="Run N rounds of automated experiments.")
    p_night.add_argument("--rounds", type=int, default=1, help="Number of experiment rounds.")
    p_night.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run to patch validation, skip training execution.",
    )

    p_report = sub.add_parser("report", help="Generate and print summary report.")
    p_report.add_argument("--json", action="store_true", help="Print summary in JSON format.")

    p_apply = sub.add_parser("apply", help="Apply one experiment patch to main workspace.")
    p_apply.add_argument("exp_id", help="Experiment id, e.g. exp_0003")

    sub.add_parser("clean", help="Remove temporary NightRunner worktrees.")
    sub.add_parser("auth", help="Check whether DEEPSEEK_API_KEY is configured.")
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
            print("NightRunner initialized.")
            print(f"Project root: {result['project_root']}")
            print(f"Config file: {result['config']}")
            print(f"State dir: {result['nightrunner_dir']}")
            return 0

        if args.command == "night":
            if args.rounds <= 0:
                raise ValueError("--rounds must be > 0")
            summary_path = run_night(project_root, rounds=args.rounds, dry_run=bool(args.dry_run))
            print(f"Night run completed. Summary: {summary_path}")
            return 0

        if args.command == "report":
            summary_path = generate_summary_report(project_root)
            experiments = load_experiments(project_root)
            best = load_best(project_root) or {}
            counts: dict[str, int] = {}
            for rec in experiments:
                status = str(rec.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
            print(f"Summary generated: {summary_path}")
            print(f"Total experiments: {len(experiments)}")
            print(f"keep: {counts.get('keep', 0)}")
            print(f"discard: {counts.get('discard', 0)}")
            print(f"crash: {counts.get('crash', 0)}")
            print(f"violation: {counts.get('violation', 0)}")
            print(f"Current best experiment: {best.get('experiment_id')}")
            print(f"Current best metric: {best.get('metric_value')}")
            print(f"Recommended report path: {summary_path}")
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
            print(f"Applied patch: {patch_path}")
            print("Patch applied to main workspace. Review changes and commit manually.")
            return 0

        if args.command == "clean":
            result = clean(project_root)
            print(f"Worktrees removed: {result['removed']}")
            if result["failed"]:
                print("Failed to remove:", ", ".join(result["failed"]))
            return 0

        if args.command == "auth":
            result = check_auth()
            print(result["message"])
            return 0 if result["ok"] else 1

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
