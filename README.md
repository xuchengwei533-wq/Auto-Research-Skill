# NightRunner

NightRunner is a local experiment UI for machine learning projects.

It installs into your Python or conda environment, starts from the command line, and runs experiments in isolated sandboxes so your main project is not modified unless you explicitly apply an experiment.

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
nightrunner ui
```

Open the local browser UI, then choose:

- goal type
- editable files
- train command
- metric and direction
- experiment rounds

NightRunner copies your current project files into `.nightrunner/sandboxes/` and runs experiments there. Git commits are optional and not required before running.

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

## CLI Usage

```powershell
nightrunner init --editable main.py --train-command "python main.py" --metric val_loss --lower-is-better
nightrunner setup --project D:\MyDeepLearningProject --yes
nightrunner doctor
nightrunner ui
nightrunner baseline
nightrunner night --rounds 8
nightrunner report

# monitoring
nightrunner status
nightrunner tail --follow
```

Core commands:

- `nightrunner setup`
- `nightrunner start`
- `nightrunner ui`
- `nightrunner doctor`
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

## Sandbox Execution

NightRunner now defaults to a sandbox copy backend:

- experiments run inside `.nightrunner/sandboxes/exp_xxxx`
- your current uncommitted files are copied into the sandbox
- Git dirty state is allowed and is not a blocker
- Git is optional for sandbox mode
- `git worktree` can still be kept as an advanced backend option

## Safety Model

- AI edits are applied only inside isolated sandboxes by default.
- Main workspace changes only after `nightrunner apply exp_xxxx` or clicking Apply in the Web UI.
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

## Local Web UI

Start the UI:

```powershell
nightrunner ui
```

The first version includes:

- Project Doctor
- Setup Wizard
- Run Monitor
- Experiments list
- Diff and Apply view

If you prefer CLI monitoring:

```powershell
nightrunner doctor
nightrunner status
nightrunner tail --follow
```

## Configuration

NightRunner reads `nightrunner.yaml` from the target project root.

Key sections:

- `project`: display name for reporting.
- `editable_files`: editable files for the UI and compatibility with older configs.
- `files.editable`: files/directories AI is allowed to modify in worktrees.
- `files.protected`: files always blocked by patch guard.
- `execution.backend`: `sandbox` by default.
- `execution.train_command`: training command executed in baseline/night runs.
- `sandbox.root` and `sandbox.ignore`: sandbox location and ignore rules.
- `optimization`: UI-facing goal and metric preferences.
- `metric.name` and `metric.lower_is_better`: primary metric extraction and comparison direction.
- `metric.regex` (optional): custom regex with one capture group for metric extraction.
- `agent`: DeepSeek/OpenAI-compatible API settings.
- `safety`: change restrictions. Clean Git is no longer required by default.
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
