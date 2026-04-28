"""Training command runner."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any


def run_training(command: str, cwd: Path, log_path: Path, timeout_seconds: int) -> dict[str, Any]:
    """Run training command and save combined stdout/stderr to log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                text=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        return {
            "returncode": proc.returncode,
            "timeout": False,
            "duration_seconds": round(time.time() - start, 3),
            "error": None,
        }
    except subprocess.TimeoutExpired as exc:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[NightRunner] 训练超时，超过 {timeout_seconds}s。\n")
            if exc.stdout:
                f.write(str(exc.stdout))
            if exc.stderr:
                f.write(str(exc.stderr))
        return {
            "returncode": None,
            "timeout": True,
            "duration_seconds": round(time.time() - start, 3),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[NightRunner] 训练执行器错误: {exc}\n")
        return {
            "returncode": None,
            "timeout": False,
            "duration_seconds": round(time.time() - start, 3),
            "error": str(exc),
        }
