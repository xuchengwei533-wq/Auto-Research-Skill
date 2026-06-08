# NightRunner

NightRunner is a local experiment UI for machine learning projects.

It installs into your Python or conda environment, starts from the command line, and runs experiments in isolated sandboxes so your main project is not modified unless you explicitly apply an experiment.

## Quick Start

```powershell
python -m pip install --upgrade "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"
cd your-ml-project
nightrunner ui
```

Recommended first steps:

- run `nightrunner doctor` if you want a quick environment check
- run `nightrunner setup --yes` if you prefer the CLI; it probes your training log and auto-detects the optimization metric when possible
- run `nightrunner ui` as the default entrypoint for new users
- in the UI, set editable files and the train command, then use Test Run to auto-detect metric candidates before starting experiments

NightRunner runs experiments in isolated sandboxes.
Your main project files are not modified unless you explicitly apply an experiment.
Git commits are optional and not required before running.
The default backend is `sandbox`.
If your Git working tree is dirty, NightRunner only shows a warning and still runs.
NightRunner copies your current working tree snapshot into `.nightrunner/sandboxes/exp_xxxx`.
During experiments, NightRunner does not modify your main project.
Only an explicit Apply writes an experiment back to your main project.

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
nightrunner setup --project D:\MyDeepLearningProject --yes
nightrunner doctor
nightrunner ui
nightrunner baseline
nightrunner night --dry-run
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

Backend options:

- `sandbox`: default, recommended for normal users, works with dirty Git and even without Git
- `worktree`: advanced mode, relies on Git worktree and is intended for clean Git repositories

Git is optional but useful. NightRunner can run sandbox experiments without requiring a clean Git working tree. If your project is a Git repository, NightRunner can use Git information for diagnostics and diff safety, but commits are not required before running.

## Safety Model

- AI edits are applied only inside isolated sandboxes by default.
- Main workspace changes only after `nightrunner apply exp_xxxx` or clicking Apply in the Web UI.
- Editable file permission only defines where NightRunner may propose changes. Protected keys and protected regions are still enforced inside editable files.
- Semantic guard rejects edits that touch protected terms such as `seed`, `random_state`, `manual_seed`, split definitions, evaluation metrics, and test dataset paths.
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
- Config files only - safest mode for YAML / JSON / TOML tuning

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
- `optimization.mode`: `standard` or `config_only`.
- `metric.name` and `metric.lower_is_better`: primary metric extraction and comparison direction.
- `metric.regex` (optional): custom regex with one capture group for metric extraction.
- `agent`: DeepSeek/OpenAI-compatible API settings.
- `safety`: change restrictions. Clean Git is no longer required by default in sandbox mode.
- `safety.semantic_guard`: semantic validator for protected terms inside editable files.
- `safety.protected_terms`: protected reproducibility / split / metric / test-path terms.
- `logging`: request/response/run artifact persistence options.

## Metric Parsing

`nightrunner setup --yes` and the Web UI Test Run can auto-detect common metrics such as `val_loss`, `loss`, `accuracy`, `f1`, `auc`, and related validation/evaluation names from the latest matching value in the training log. If auto-detection picks the wrong candidate, you can still override the metric name, direction, or regex in `nightrunner.yaml` or in the UI.

If you change the metric after a baseline already exists, run `nightrunner baseline --force` so later experiments are compared against the same metric configuration.

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
