#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${NIGHTRUNNER_REPO:-git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop}"
FORCE_PIP="${NIGHTRUNNER_FORCE_PIP:-0}"

step() {
  printf '\n==> %s\n' "$1"
}

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

if ! command -v git >/dev/null 2>&1; then
  die "Git is required but was not found in PATH."
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  die "Python 3.10+ is required but was not found in PATH."
fi

step "Checking Python"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python 3.10+ is required."

if [[ "$FORCE_PIP" != "1" ]] && command -v uv >/dev/null 2>&1; then
  step "Installing NightRunner with uv"
  uv tool install --force "$REPO_URL"
else
  step "Installing NightRunner with pip"
  "$PYTHON_BIN" -m pip install --user --upgrade "$REPO_URL"
fi

RUNNER_CMD="nightrunner"
step "Verifying installation"
if command -v nightrunner >/dev/null 2>&1; then
  nightrunner --help >/dev/null
else
  printf 'nightrunner is not on PATH yet.\n'
  printf 'You can still run:\n'
  printf '  %s -m nightrunner.cli --help\n' "$PYTHON_BIN"
  "$PYTHON_BIN" -m nightrunner.cli --help >/dev/null
  RUNNER_CMD="$PYTHON_BIN -m nightrunner.cli"
fi

step "Installed successfully"
printf 'Next steps:\n'
printf '  %s auth login\n' "$RUNNER_CMD"
printf '  cd /path/to/your/ml/project\n'
printf '  %s setup\n' "$RUNNER_CMD"
printf '  %s run --rounds 8\n' "$RUNNER_CMD"
