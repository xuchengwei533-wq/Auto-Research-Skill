from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nightrunner import cli, runner


def test_setup_help_includes_metric_regex(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["setup", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--metric-regex" in captured.out


def test_cli_setup_passes_metric_regex(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_setup(
        project_root: Path,
        editable_files=None,
        train_command=None,
        metric_name=None,
        metric_regex=None,
        lower_is_better=None,
        api_key_env=None,
        base_url=None,
        model=None,
        run_baseline_now=False,
        yes=False,
    ):
        captured["project_root"] = project_root
        captured["metric_regex"] = metric_regex
        return {"config": str(project_root / "nightrunner.yaml")}

    monkeypatch.setattr(cli, "setup", fake_setup)

    exit_code = cli.main(
        [
            "setup",
            "--project",
            str(tmp_path),
            "--yes",
            "--editable",
            "examples/mnist/main.py",
            "--train-command",
            "python examples/mnist/main.py",
            "--metric",
            "Average loss",
            "--lower-is-better",
            "--metric-regex",
            r"Average loss:\s*([0-9.]+)",
        ]
    )

    assert exit_code == 0
    assert captured["metric_regex"] == r"Average loss:\s*([0-9.]+)"


def test_noninteractive_setup_writes_metric_regex(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "examples" / "mnist").mkdir(parents=True)
    (tmp_path / "examples" / "mnist" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()

    monkeypatch.setattr(runner, "_is_git_available", lambda: True)
    monkeypatch.setattr(runner, "check_auth", lambda project_root=None: {"ok": True, "message": "ok", "source": "environment"})

    result = runner.setup(
        tmp_path,
        editable_files=["examples/mnist/main.py"],
        train_command="python examples/mnist/main.py",
        metric_name="Average loss",
        metric_regex=r"Average loss:\s*([0-9.]+)",
        lower_is_better=True,
        yes=True,
    )

    config_path = Path(result["config"])
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["metric"]["regex"] == r"Average loss:\s*([0-9.]+)"
