# NightRunner

NightRunner is an AI-powered overnight experiment runner for machine learning projects.

It is installed as a CLI tool and runs inside your own Git project.

## Install

```powershell
git clone -b develop https://github.com/xuchengwei533-wq/Auto-Research-Skill.git D:\Tools\Auto-Research-Skill
cd D:\Tools\Auto-Research-Skill\packages\nightrunner
uv tool install -e .
```

Or:

```powershell
uv tool install "git+https://github.com/xuchengwei533-wq/Auto-Research-Skill.git@develop#subdirectory=packages/nightrunner"
```

## Use In Your Project

```powershell
cd D:\Research\MyDLProject
nightrunner init --editable train.py --train-command "uv run train.py" --metric val_loss --lower-is-better
$env:DEEPSEEK_API_KEY="your-key"
nightrunner baseline
nightrunner night --rounds 8
nightrunner report
notepad nightrunner_summary.md
```
