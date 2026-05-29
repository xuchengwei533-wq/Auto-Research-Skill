"""Sandbox copy backend for isolated NightRunner experiments."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import ensure_dir

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
    "*.pyc",
    "*.pyo",
]


@dataclass(frozen=True)
class Sandbox:
    exp_id: str
    root_dir: Path
    project_dir: Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class SandboxManager:
    """Create isolated experiment sandboxes by copying the current project."""

    def __init__(
        self,
        project_root: Path,
        sandbox_root: Path,
        ignore_patterns: Iterable[str] | None = None,
        respect_gitignore: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.sandbox_root = sandbox_root.resolve()
        self.ignore_patterns = [self._normalize_pattern(p) for p in (ignore_patterns or DEFAULT_SANDBOX_IGNORE)]
        self.respect_gitignore = respect_gitignore
        self.gitignore_patterns = self._load_gitignore_patterns() if respect_gitignore else []

    @classmethod
    def from_config(cls, project_root: Path, config: dict) -> "SandboxManager":
        sandbox_cfg = config.get("sandbox", {}) if isinstance(config.get("sandbox"), dict) else {}
        root_value = sandbox_cfg.get("root", ".nightrunner/sandboxes")
        sandbox_root = (project_root / str(root_value)).resolve() if not Path(str(root_value)).is_absolute() else Path(str(root_value)).resolve()
        ignore = sandbox_cfg.get("ignore")
        if not isinstance(ignore, list):
            ignore = DEFAULT_SANDBOX_IGNORE
        return cls(project_root=project_root, sandbox_root=sandbox_root, ignore_patterns=ignore)

    def create_sandbox(self, exp_id: str) -> Sandbox:
        root_dir = ensure_dir(self.sandbox_root)
        project_dir = root_dir / exp_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        ensure_dir(project_dir)
        self._copy_project_tree(project_dir)
        return Sandbox(exp_id=exp_id, root_dir=project_dir, project_dir=project_dir)

    def remove_sandbox(self, sandbox: Sandbox | Path | None) -> None:
        if sandbox is None:
            return
        root_dir = sandbox.root_dir if isinstance(sandbox, Sandbox) else Path(sandbox)
        if root_dir.exists():
            shutil.rmtree(root_dir)

    def cleanup_sandbox(self, exp_id: str) -> None:
        self.remove_sandbox(self.sandbox_root / exp_id)

    def file_hashes(self, relative_paths: Iterable[str]) -> dict[str, str | None]:
        hashes: dict[str, str | None] = {}
        for rel_path in relative_paths:
            normalized = rel_path.replace("\\", "/")
            target = self.project_root / normalized
            hashes[normalized] = sha256_file(target) if target.exists() and target.is_file() else None
        return hashes

    def _copy_project_tree(self, destination_root: Path) -> None:
        for current_root, dirnames, filenames in os.walk(self.project_root):
            current_path = Path(current_root)
            rel_dir = current_path.relative_to(self.project_root)
            rel_dir_str = "" if rel_dir == Path(".") else rel_dir.as_posix()

            kept_dirs: list[str] = []
            for dirname in dirnames:
                rel_path = f"{rel_dir_str}/{dirname}".strip("/")
                if self._should_ignore(rel_path, is_dir=True):
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs

            for filename in filenames:
                rel_path = f"{rel_dir_str}/{filename}".strip("/")
                if self._should_ignore(rel_path, is_dir=False):
                    continue
                src = current_path / filename
                dst = destination_root / rel_path
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)

    def _load_gitignore_patterns(self) -> list[str]:
        path = self.project_root / ".gitignore"
        if not path.exists():
            return []
        patterns: list[str] = []
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            patterns.append(self._normalize_pattern(line))
        return patterns

    def _should_ignore(self, relative_path: str, is_dir: bool) -> bool:
        normalized = relative_path.replace("\\", "/").strip("/")
        if not normalized:
            return False
        for pattern in [*self.ignore_patterns, *self.gitignore_patterns]:
            if self._matches_pattern(normalized, pattern, is_dir=is_dir):
                return True
        return False

    @staticmethod
    def _normalize_pattern(pattern: str) -> str:
        return pattern.replace("\\", "/").strip()

    @staticmethod
    def _matches_pattern(relative_path: str, pattern: str, is_dir: bool) -> bool:
        pattern = pattern.strip()
        if not pattern:
            return False
        if pattern.endswith("/"):
            pattern = pattern.rstrip("/")
            return relative_path == pattern or relative_path.startswith(f"{pattern}/")
        if "/" not in pattern and not any(ch in pattern for ch in "*?[]"):
            parts = relative_path.split("/")
            return pattern in parts if is_dir else pattern in parts[:-1] or parts[-1] == pattern
        if fnmatch.fnmatch(relative_path, pattern):
            return True
        if "/" not in pattern and fnmatch.fnmatch(Path(relative_path).name, pattern):
            return True
        return relative_path == pattern or relative_path.startswith(f"{pattern}/")
