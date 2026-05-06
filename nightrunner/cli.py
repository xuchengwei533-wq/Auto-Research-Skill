"""Argparse CLI entrypoint for NightRunner."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .auth_store import delete_api_key, get_config_path as get_auth_config_path, save_api_key
from .config import load_config
from .report import generate_summary_report
from .runner import (
    apply_experiment,
    check_auth,
    clean,
    init_project,
    run,
    run_baseline,
    run_night,
    setup,
    status,
    tail,
)
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
    p_night.add_argument("--plain", action="store_true", help="Force plain text output (disable rich UI).")

    p_run = sub.add_parser("run", help="Run baseline check + night + report in one command.")
    p_run.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_run.add_argument("--rounds", type=int, default=1, help="Number of rounds.")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate changes only and skip training.",
    )
    p_run.add_argument("--plain", action="store_true", help="Force plain text output (disable rich UI).")

    p_setup = sub.add_parser("setup", help="Interactive setup wizard for first-time use.")
    p_setup.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_setup.add_argument(
        "--editable",
        action="append",
        default=[],
        help="Editable file (repeatable), e.g. --editable train.py --editable model.py",
    )
    p_setup.add_argument("--train-command", type=str, default=None, help="Training command.")
    p_setup.add_argument("--metric", type=str, default=None, help="Primary metric name.")
    p_setup.add_argument("--api-key-env", type=str, default=None, help="API key environment variable name.")
    p_setup.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL.")
    p_setup.add_argument("--model", type=str, default=None, help="Model name, e.g. deepseek-v4-pro.")
    p_setup.add_argument("--run-baseline", action="store_true", help="Run baseline after setup completes.")
    p_setup.add_argument("--yes", action="store_true", help="Accept defaults and minimize prompts.")
    p_setup_mode = p_setup.add_mutually_exclusive_group()
    p_setup_mode.add_argument("--lower-is-better", action="store_true")
    p_setup_mode.add_argument("--higher-is-better", action="store_true")

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
    p_clean.add_argument(
        "--branches",
        action="store_true",
        help="Also delete leftover nightrunner/* branches after removing worktrees.",
    )
    p_auth = sub.add_parser("auth", help="Manage DeepSeek API key for NightRunner.")
    p_auth.add_argument("--project", type=str, default=None, help="Optional project path.")
    auth_sub = p_auth.add_subparsers(dest="auth_command")
    auth_sub.add_parser("login", help="Save DeepSeek API key to user config.")
    auth_sub.add_parser("status", help="Show API key status without printing the full key.")
    auth_sub.add_parser("logout", help="Delete saved DeepSeek API key from user config.")

    p_status = sub.add_parser("status", help="Show current run status and recent experiments.")
    p_status.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_status.add_argument("--plain", action="store_true", help="Force plain text output.")

    p_tail = sub.add_parser("tail", help="Show tail of latest or selected run.log.")
    p_tail.add_argument("--project", type=str, default=None, help="Target project path (default: cwd).")
    p_tail.add_argument("--exp", type=str, default=None, help="Experiment ID, e.g. exp_0005.")
    p_tail.add_argument("--lines", type=int, default=80, help="Number of lines from tail.")
    p_tail.add_argument("-f", "--follow", action="store_true", help="Follow appended log output.")
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
            summary_path = run_night(
                project_root,
                rounds=args.rounds,
                dry_run=bool(args.dry_run),
                plain=bool(args.plain),
            )
            print(f"Night run completed. Summary: {summary_path}")
            return 0

        if args.command == "run":
            if args.rounds <= 0:
                raise ValueError("--rounds must be > 0")
            summary_path = run(
                project_root,
                rounds=args.rounds,
                dry_run=bool(args.dry_run),
                plain=bool(args.plain),
            )
            print(f"Run completed. Summary: {summary_path}")
            return 0

        if args.command == "setup":
            lower_is_better = None
            if bool(args.higher_is_better):
                lower_is_better = False
            elif bool(args.lower_is_better):
                lower_is_better = True
            result = setup(
                project_root,
                editable_files=list(args.editable) if args.editable else None,
                train_command=args.train_command,
                metric_name=args.metric,
                lower_is_better=lower_is_better,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                run_baseline_now=bool(args.run_baseline),
                yes=bool(args.yes),
            )
            print(f"Config file: {Path(result['config']).name}")
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
                rec_status = str(rec.get("status", "unknown"))
                counts[rec_status] = counts.get(rec_status, 0) + 1
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
            result = clean(project_root, branches=bool(args.branches))
            print(f"Worktrees removed: {result['removed']}")
            if result["failed"]:
                print("Failed to remove:", ", ".join(result["failed"]))
            if result.get("removed_branches"):
                print("Branches removed:", ", ".join(result["removed_branches"]))
            return 0

        if args.command == "auth":
            auth_command = getattr(args, "auth_command", None) or "status"
            if auth_command == "login":
                api_key = getpass.getpass("Paste DeepSeek API key: ").strip()
                if not api_key:
                    raise RuntimeError("No API key entered.")
                save_api_key("deepseek", api_key, base_url="https://api.deepseek.com")
                print(f"DeepSeek API key saved to {get_auth_config_path()}")
                return 0
            if auth_command == "logout":
                deleted = delete_api_key("deepseek")
                if deleted:
                    print("DeepSeek API key removed from user config.")
                else:
                    print("No saved DeepSeek API key found.")
                return 0
            try:
                load_config(project_root)
                result = check_auth(project_root)
            except Exception:
                result = check_auth()
            print(result["message"])
            return 0 if result["ok"] else 1

        if args.command == "status":
            status(project_root, plain=bool(args.plain))
            return 0

        if args.command == "tail":
            if args.lines <= 0:
                raise ValueError("--lines must be > 0")
            tail(project_root, exp=args.exp, lines=int(args.lines), follow=bool(args.follow))
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
