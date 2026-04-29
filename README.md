# NightRunner

NightRunner is an AI-powered overnight experiment runner for machine learning projects.

It is installed as a CLI tool and runs inside your own Git project.

## Install

Recommended:

```powershell
uv tool install "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"
```

Without uv:

```powershell
python -m pip install "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"
```

`uv` is the recommended way to install an isolated CLI tool. `pip` is the compatibility option.
NightRunner itself is lightweight and does not install training dependencies such as `torch`, `torchvision`, `tensorflow`, `numpy`, or `pandas`.
Your training project's dependencies stay in your project environment.

## Quick Start

```powershell
python -m pip install "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"

cd D:\MyDeepLearningProject

nightrunner setup
nightrunner run --rounds 8

notepad nightrunner_summary.md
```

## Advanced Usage

```powershell
nightrunner init --editable main.py --train-command "python main.py" --metric val_loss --lower-is-better
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
- `nightrunner auth`

All project-related commands support `--project <path>` and default to `Path.cwd()`.

## Why Git Is Required

NightRunner requires Git because:

- It uses `git worktree` to isolate experiments safely.
- It uses `git diff` to generate `patch.diff`.
- It uses `git apply` to apply selected experiments.
- It checks `git status` to avoid modifying dirty workspaces.

## Safety Model

- AI edits are applied only inside `.nightrunner/worktrees/`.
- Main workspace changes only after `nightrunner apply exp_xxxx`.
- API key is read from environment variables only.
- Training dependencies belong to the user project, not NightRunner.
- `.nightrunner/` stores local experiment state.
- `nightrunner_summary.md` is written to your project root.

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
