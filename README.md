# NightRunner

NightRunner is an AI-powered overnight experiment runner for machine learning projects.

It is installed as a CLI tool and runs inside your own Git project.

## Install from GitHub

```powershell
uv tool install "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop"
```

## Safety Model

- NightRunner does not directly modify your main workspace.
- AI edits are applied only inside `.nightrunner/worktrees/`.
- Main workspace changes only if you run `nightrunner apply exp_xxxx`.
- API key is read from environment variables only.
- `.nightrunner/` stores local experiment state.
- `nightrunner_summary.md` is written to your project root.

## Commands

- `nightrunner init`
- `nightrunner baseline`
- `nightrunner night`
- `nightrunner report`
- `nightrunner apply`
- `nightrunner clean`
- `nightrunner auth`

All project-related commands support `--project <path>` and default to the current working directory.

## Quick Start

```powershell
cd D:\Research\MyDLProject
nightrunner init --editable train.py --train-command "uv run train.py" --metric val_loss --lower-is-better
$env:DEEPSEEK_API_KEY="your-key"
nightrunner baseline
nightrunner night --rounds 8
nightrunner report
notepad nightrunner_summary.md
```

## Configuration

NightRunner reads `nightrunner.yaml` from the target project root.

Key sections:

- `project`: display name for reporting.
- `files.editable`: files/directories AI is allowed to modify in worktrees.
- `files.protected`: files always blocked by patch guard.
- `execution.train_command`: training command executed in baseline/night runs.
- `metric.name` and `metric.lower_is_better`: primary metric extraction and comparison direction.
- `agent`: DeepSeek/OpenAI-compatible API settings.
- `safety`: clean-git requirement and change restrictions.
- `logging`: request/response/run artifact persistence options.
