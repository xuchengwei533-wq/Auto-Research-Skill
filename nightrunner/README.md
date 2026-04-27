# NightRunner

NightRunner is a standalone Python CLI for overnight ML experimentation.  
It runs in your Git project, creates isolated Git worktrees, asks an LLM for a patch, validates the patch, runs training, parses metrics, and saves reports.

## Why

NightRunner helps you automate repetitive experiment cycles at night:

- propose one code experiment
- run training
- collect metric and logs
- keep or discard by metric
- never auto-merge to your main workspace by default

## Install

```bash
pip install -e .
```

or

```bash
uv tool install .
```

## Quick Start

```bash
cd your-project
nightrunner init
```

Edit `nightrunner.yaml` if needed, then set API key and run:

```bash
nightrunner auth
nightrunner night --rounds 5
nightrunner report
```

Two practical enhancements:

- `nightrunner night --dry-run`: validates patch and safety checks without running training.
- `nightrunner report --json`: prints machine-readable summary JSON.

## API Key

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
```

Persistent on Windows:

```powershell
setx DEEPSEEK_API_KEY "your-key"
```

Linux/macOS:

```bash
export DEEPSEEK_API_KEY="your-key"
```

## Safety Model

- Main workspace is not directly modified in `night` runs.
- All AI edits happen in `.nightrunner/worktrees/<exp_id>/`.
- Only `files.editable` are allowed to change.
- Protected files are blocked.
- Dependency file changes are blocked by default.
- API key is read only from `DEEPSEEK_API_KEY`.
- You can configure API endpoint and key env name via `nightrunner auth`.
- Applying to main requires explicit command: `nightrunner apply <exp_id>`.

## Default nightrunner.yaml

```yaml
project:
  name: default-project

files:
  editable:
    - train.py
  protected:
    - .env
    - .env.local
    - pyproject.toml
    - requirements.txt
    - uv.lock
    - README.md
    - nightrunner.yaml

execution:
  train_command: "uv run train.py"
  timeout_seconds: 3600

metric:
  name: val_bpb
  lower_is_better: true

agent:
  provider: deepseek
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  model: deepseek-v4-pro
  reasoning_effort: high
  thinking_enabled: true

safety:
  require_clean_git: true
  auto_apply_to_main: false
  allow_new_files: false
  allow_dependency_changes: false

logging:
  save_request: true
  save_response: true
  save_run_log: true
```

## .nightrunner Directory

Inside your target project:

```text
.nightrunner/
  state/
    best.json
    experiments.jsonl
    project.json
  runs/
  worktrees/
  cache/
  tmp/
```

## FAQ

- Not a Git repo: initialize Git first (`git init`) and commit baseline.
- Missing API key: set `DEEPSEEK_API_KEY` in environment variables.
- How to input your own API and API key: run `nightrunner auth`, enter your API Base URL and key.
- Patch apply failure: check `.nightrunner/runs/<exp_id>/response.json` and `model.patch.diff`.
- Metric not found: run will be marked as `crash`; adjust parser metric name or training output.
- How to apply best experiment: `nightrunner apply <exp_id>`, then review and commit manually.
