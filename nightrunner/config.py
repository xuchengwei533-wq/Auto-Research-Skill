"""Configuration loading and defaults for NightRunner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .utils import write_text

CONFIG_FILE_NAME = "nightrunner.yaml"

DEFAULT_SANDBOX_IGNORE = [
    ".git",
    ".nightrunner/sandboxes",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "runs",
    "wandb",
    "checkpoints",
    "outputs",
    ".tmp-auth",
]

DEFAULT_PROTECTED_TERMS = [
    "seed",
    "random_seed",
    "random_state",
    "split_seed",
    "dataloader_seed",
    "manual_seed",
    "torch.manual_seed",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
    "np.random.seed",
    "numpy.random.seed",
    "random.seed",
    "PYTHONHASHSEED",
    "cudnn.deterministic",
    "cudnn.benchmark",
    "train_test_split",
    "StratifiedKFold",
    "KFold",
    "GroupKFold",
    "ShuffleSplit",
    "random_split",
    "train_indices",
    "val_indices",
    "test_indices",
    "validation_split",
    "test_size",
    "metric",
    "metrics",
    "accuracy",
    "f1",
    "auc",
    "roc_auc",
    "rmse",
    "mae",
    "mse",
    "evaluate",
    "test",
    "test_loader",
    "val_loader",
    "validation_loader",
    "test_data",
    "test_path",
    "test_dir",
    "label_column",
    "target_column",
    "labels",
    "targets",
]


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=_NoAliasSafeDumper, sort_keys=False, allow_unicode=True)


def build_default_config(
    project_name: str,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
) -> dict[str, Any]:
    editable = list(editable_files or ["train.py"])
    return {
        "project": {"name": project_name},
        "editable_files": list(editable),
        "files": {
            "editable": list(editable),
            "protected": [
                ".env",
                ".env.local",
                "pyproject.toml",
                "requirements.txt",
                "uv.lock",
                "README.md",
                "nightrunner.yaml",
            ],
        },
        "execution": {
            "backend": "sandbox",
            "train_command": train_command,
            "timeout_seconds": 3600,
        },
        "sandbox": {
            "root": ".nightrunner/sandboxes",
            "ignore": list(DEFAULT_SANDBOX_IGNORE),
        },
        "metric": {"name": metric_name, "lower_is_better": lower_is_better},
        "optimization": {
            "goal": "Tune hyperparameters",
            "mode": "standard",
            "metric": metric_name,
            "metric_regex": None,
            "higher_is_better": not lower_is_better,
        },
        "agent": {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": "deepseek-v4-pro",
            "reasoning_effort": "high",
            "thinking_enabled": True,
        },
        "safety": {
            "require_clean_git": False,
            "auto_apply_to_main": False,
            "allow_new_files": False,
            "allow_dependency_changes": False,
            "semantic_guard": True,
            "allow_protected_term_edits": False,
            "protected_terms": list(DEFAULT_PROTECTED_TERMS),
        },
        "logging": {"save_request": True, "save_response": True, "save_run_log": True},
    }


DEFAULT_CONFIG: dict[str, Any] = build_default_config("Auto-Research-Skill")

NIGHTRUNNER_GITIGNORE_LINES = [
    "# NightRunner generated artifacts",
    ".nightrunner/",
    "nightrunner_summary.md",
    "",
    "# Secrets",
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
]


def get_config_path(project_root: Path) -> Path:
    return project_root / CONFIG_FILE_NAME


def write_default_config_if_missing(
    project_root: Path,
    editable_files: list[str] | None = None,
    train_command: str = "python train.py",
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
) -> Path:
    path = get_config_path(project_root)
    if path.exists():
        return path
    default = build_default_config(
        project_name=project_root.name,
        editable_files=editable_files,
        train_command=train_command,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
    )
    write_text(path, _dump_yaml(default))
    return path


def load_config(project_root: Path) -> dict[str, Any]:
    path = get_config_path(project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"{CONFIG_FILE_NAME} not found in {project_root}. Run `nightrunner init` first."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config format in {path}. Root must be a mapping.")

    cfg = deepcopy(build_default_config(project_root.name))
    _deep_merge(cfg, raw)
    _normalize_config(cfg)

    metric = cfg.get("metric", {})
    if isinstance(metric, dict) and "regex" in metric and metric["regex"] is not None:
        if not isinstance(metric["regex"], str):
            raise ValueError("metric.regex must be a string when provided.")
    return cfg


def save_config(project_root: Path, config: dict[str, Any]) -> Path:
    """Persist config to nightrunner.yaml."""
    path = get_config_path(project_root)
    write_text(path, _dump_yaml(config))
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _normalize_config(config: dict[str, Any]) -> None:
    editable = config.get("editable_files")
    if isinstance(editable, list) and editable:
        config.setdefault("files", {})["editable"] = editable
    else:
        config["editable_files"] = list(config.get("files", {}).get("editable", ["train.py"]))

    execution = config.setdefault("execution", {})
    backend = str(execution.get("backend", "sandbox")).strip() or "sandbox"
    if backend == "git_worktree":
        backend = "worktree"
    if backend not in {"sandbox", "worktree"}:
        backend = "sandbox"
    execution["backend"] = backend
    execution.setdefault("train_command", "python train.py")
    execution.setdefault("timeout_seconds", 3600)

    sandbox = config.setdefault("sandbox", {})
    sandbox.setdefault("root", ".nightrunner/sandboxes")
    sandbox_ignore = sandbox.get("ignore")
    if not isinstance(sandbox_ignore, list):
        sandbox["ignore"] = list(DEFAULT_SANDBOX_IGNORE)

    metric = config.setdefault("metric", {})
    optimization = config.setdefault("optimization", {})
    if optimization.get("metric") and not metric.get("name"):
        metric["name"] = optimization["metric"]
    if optimization.get("metric_regex") and not metric.get("regex"):
        metric["regex"] = optimization["metric_regex"]
    if "higher_is_better" in optimization and "lower_is_better" not in metric:
        metric["lower_is_better"] = not bool(optimization["higher_is_better"])
    metric.setdefault("name", "val_loss")
    metric.setdefault("lower_is_better", True)
    optimization.setdefault("goal", "Tune hyperparameters")
    optimization.setdefault("mode", "standard")
    optimization["metric"] = metric["name"]
    optimization["metric_regex"] = metric.get("regex")
    optimization["higher_is_better"] = not bool(metric.get("lower_is_better", True))

    safety = config.setdefault("safety", {})
    if execution["backend"] == "worktree":
        safety.setdefault("require_clean_git", True)
    else:
        safety.setdefault("require_clean_git", False)
    safety.setdefault("auto_apply_to_main", False)
    safety.setdefault("allow_new_files", False)
    safety.setdefault("allow_dependency_changes", False)
    safety.setdefault("semantic_guard", True)
    safety.setdefault("allow_protected_term_edits", False)
    protected_terms = safety.get("protected_terms")
    if not isinstance(protected_terms, list) or not protected_terms:
        safety["protected_terms"] = list(DEFAULT_PROTECTED_TERMS)
