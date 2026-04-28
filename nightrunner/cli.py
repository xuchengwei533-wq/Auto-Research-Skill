"""Argparse CLI entrypoint for NightRunner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .report import generate_summary_report
from .runner import apply_experiment, check_auth, clean, init_project, run_baseline, run_night
from .state_store import load_best, load_experiments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nightrunner", description="NightRunner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize NightRunner in target Git project.")
    p_init.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_init.add_argument(
        "--editable",
        action="append",
        default=[],
        help="Editable file (repeatable), e.g. --editable train.py --editable model.py",
    )
    p_init.add_argument(
        "--train-command", type=str, default="python train.py", help="Training command, e.g. uv run train.py"
    )
    p_init.add_argument("--metric", type=str, default="val_loss", help="Primary metric name.")
    mode = p_init.add_mutually_exclusive_group()
    mode.add_argument("--lower-is-better", action="store_true", default=True)
    mode.add_argument("--higher-is-better", action="store_true")

    p_night = sub.add_parser("night", help="Run N rounds of automated experiments.")
    p_night.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_night.add_argument("--rounds", type=int, default=1, help="Number of rounds.")
    p_night.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate changes only and skip training.",
    )

    p_baseline = sub.add_parser("baseline", help="Run baseline training and write best.json.")
    p_baseline.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_baseline.add_argument(
        "--force",
        action="store_true",
        help="Force baseline rerun even if best.json already exists.",
    )

    p_report = sub.add_parser("report", help="Generate and print summary report.")
    p_report.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_report.add_argument("--json", action="store_true", help="Print summary as JSON.")

    p_apply = sub.add_parser("apply", help="Apply one experiment patch to main workspace.")
    p_apply.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_apply.add_argument("exp_id", help="Experiment ID, e.g. exp_0003")

    p_clean = sub.add_parser("clean", help="Remove temporary NightRunner worktrees.")
    p_clean.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_auth = sub.add_parser("auth", help="Check API key environment variable.")
    p_auth.add_argument("--project", type=str, default=None, help="Optional project path.")
    return parser


def _project_root(project_arg: str | None) -> Path:
    if project_arg:
        return Path(project_arg).expanduser().resolve()
    return Path.cwd().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = _project_root(getattr(args, "project", None))

    try:
        if args.command == "init":
            editable_files = list(args.editable) if args.editable else ["train.py"]
            lower_is_better = not bool(args.higher_is_better)
            result = init_project(
                project_root,
                editable_files=editable_files,
                train_command=args.train_command,
                metric_name=args.metric,
                lower_is_better=lower_is_better,
            )
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

        if args.command == "baseline":
            report_path = run_baseline(project_root, force=bool(args.force))
            print(f"Baseline completed. Report: {report_path}")
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
            print("Patch applied. Please review changes:")
            print("git diff")
            print("git add .")
            print(f"git commit -m \"Apply NightRunner experiment {args.exp_id}\"")
            return 0

        if args.command == "clean":
            result = clean(project_root)
            print(f"Worktrees removed: {result['removed']}")
            if result["failed"]:
                print("Failed to remove:", ", ".join(result["failed"]))
            return 0

        if args.command == "auth":
            result = check_auth(project_root)
            print(result["message"])
            return 0 if result["ok"] else 1

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
