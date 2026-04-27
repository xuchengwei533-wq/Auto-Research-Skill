"""Argparse CLI entrypoint for NightRunner."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import sys
from pathlib import Path

from .config import load_config, save_config, write_default_config_if_missing
from .report import generate_summary_report
from .runner import apply_experiment, clean, init_project, run_night
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
    p_auth = sub.add_parser(
        "auth",
        help="Configure API base URL and input your own API key (key is not written to files).",
    )
    p_auth.add_argument("--base-url", help="API base URL, e.g. https://api.deepseek.com")
    p_auth.add_argument("--key-env", help="Environment variable name for API key.")
    p_auth.add_argument("--api-key", help="API key value. If omitted, prompt securely.")
    return parser


def _project_root() -> Path:
    return Path.cwd()


def _prompt_with_default(title: str, default: str) -> str:
    value = input(f"{title} [{default}]: ").strip()
    return value or default


def _configure_auth(project_root: Path, base_url: str | None, key_env: str | None, api_key: str | None) -> None:
    write_default_config_if_missing(project_root)
    config = load_config(project_root)
    agent = config.setdefault("agent", {})
    default_base_url = str(agent.get("base_url", "https://api.deepseek.com"))
    default_key_env = str(agent.get("api_key_env", "DEEPSEEK_API_KEY"))

    chosen_base_url = base_url or _prompt_with_default("API Base URL", default_base_url)
    chosen_key_env = key_env or _prompt_with_default("API Key env name", default_key_env)
    chosen_api_key = api_key or getpass.getpass(f"API Key value for {chosen_key_env}: ").strip()
    if not chosen_api_key:
        raise ValueError("API key cannot be empty.")

    agent["base_url"] = chosen_base_url
    agent["api_key_env"] = chosen_key_env
    save_config(project_root, config)

    # Only set for this process/session; do not persist to any file.
    os.environ[chosen_key_env] = chosen_api_key

    print("Auth setup complete.")
    print(f"Configured API Base URL: {chosen_base_url}")
    print(f"Configured API Key env: {chosen_key_env}")
    print("API key is loaded in current process only and was not written to disk.")
    if platform.system().lower().startswith("win"):
        print("To persist on Windows, run in PowerShell:")
        print(f'  setx {chosen_key_env} "YOUR_KEY"')
    else:
        print("To persist on Linux/macOS, add to shell profile:")
        print(f'  export {chosen_key_env}="YOUR_KEY"')


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
            _configure_auth(
                project_root=project_root,
                base_url=args.base_url,
                key_env=args.key_env,
                api_key=args.api_key,
            )
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
