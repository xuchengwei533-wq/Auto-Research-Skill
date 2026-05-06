# NightRunner

NightRunner is an AI-powered overnight experiment runner for machine learning projects.

It is installed as a CLI tool and runs inside your own Git project.

## Quick Start

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/xuchengwei533-wq/Auto-Research-Skill/develop/install.ps1 | iex"
```

macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/xuchengwei533-wq/Auto-Research-Skill/develop/install.sh | bash
```

Then:

```powershell
nightrunner auth login
cd D:\MyDeepLearningProject
nightrunner setup
git add nightrunner.yaml .gitignore
git commit -m "Configure NightRunner"
nightrunner run --rounds 8
```

## Manual Install

Recommended with `uv`:

```powershell
uv tool install --force "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"
```

Compatible with `pip`:

```powershell
python -m pip install --user --upgrade "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"
```

`uv tool install` is the recommended way to install an isolated CLI tool. `pip` is the compatibility option.
NightRunner itself is lightweight and does not install training dependencies such as `torch`, `tensorflow`, `numpy`, or `pandas`.
Your training project's dependencies stay in your own project environment.
If `nightrunner` is not found on Windows after a `uv` install, run `uv tool update-shell` or restart your terminal.

## Advanced Usage

```powershell
nightrunner init --editable main.py --train-command "python main.py" --metric val_loss --lower-is-better
nightrunner setup --project D:\MyDeepLearningProject --yes
nightrunner baseline
nightrunner night --rounds 8
nightrunner report

# monitoring
nightrunner status
nightrunner tail --follow
```

Core commands:

- `nightrunner setup`
- `nightrunner run`
- `nightrunner status`
- `nightrunner tail`
- `nightrunner init`
- `nightrunner baseline`
- `nightrunner night`
- `nightrunner report`
- `nightrunner apply`
- `nightrunner clean`
- `nightrunner auth login`
- `nightrunner auth status`
- `nightrunner auth logout`

Authentication commands:

- `nightrunner auth login`
- `nightrunner auth status`
- `nightrunner auth logout`

All project-related commands support `--project <path>` and default to `Path.cwd()`.

## Why Git Is Required

NightRunner requires Git because:

- It uses `git worktree` to isolate experiments safely.
- It uses `git diff` to generate `patch.diff`.
- It uses `git apply` to apply selected experiments.
- It checks `git status` to avoid modifying dirty workspaces.
- It keeps AI edits isolated from your main workspace until you explicitly apply them.

If your ML project is not a Git repository yet, initialize it first:

```powershell
git init
git add .
git commit -m "Initial commit"
```

## Safety Model

- AI edits are applied only inside `.nightrunner/worktrees/`.
- Main workspace changes only after `nightrunner apply exp_xxxx`.
- API key is read from `DEEPSEEK_API_KEY` first, then from the user-level NightRunner auth config created by `nightrunner auth login`.
- API key is never written to the target project directory or `nightrunner.yaml`.
- Training dependencies belong to the user project, not NightRunner.
- `.nightrunner/` stores lightweight local state, reports, and patch metadata.
- `nightrunner_summary.md` is written to your project root.

## API Key Configuration

For normal interactive use:

```powershell
nightrunner auth login
nightrunner auth status
```

For CI or temporary shells, prefer an environment variable:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
```

NightRunner checks credentials in this order:

1. Environment variable from `agent.api_key_env`
2. User config stored by `nightrunner auth login`

User auth config location:

- Windows: `%APPDATA%/nightrunner/config.json`
- macOS/Linux: `~/.config/nightrunner/config.json`

## Terminal UI / Monitoring

During a run, NightRunner shows:

- current experiment ID
- current stage
- editable files
- train command
- metric
- current best
- recent experiments
- API/train/total time

Monitoring commands:

```powershell
nightrunner run --rounds 58

# in another terminal
nightrunner status
nightrunner tail --follow
```

If you prefer plain logs (or in CI):

```powershell
nightrunner run --rounds 58 --plain
nightrunner night --rounds 58 --plain
```

## Configuration

NightRunner reads `nightrunner.yaml` from the target project root.

Key sections:

- `project`: display name for reporting.
- `files.editable`: files/directories AI is allowed to modify in worktrees.
- `files.protected`: files always blocked by patch guard.
- `execution.train_command`: training command executed in baseline/night runs.
- `metric.name` and `metric.lower_is_better`: primary metric extraction and comparison direction.
- `metric.regex` (optional): custom regex with one capture group for metric extraction.
- `agent`: DeepSeek/OpenAI-compatible API settings.
- `safety`: clean-git requirement and change restrictions.
- `logging`: request/response/run artifact persistence options.

## Metric Parsing

Default parsing:

```yaml
metric:
  name: val_loss
  lower_is_better: true
```

Regex-based parsing:

```yaml
metric:
  name: val_loss
  lower_is_better: true
  regex: "Average loss:\\s*([0-9.]+)"
```
