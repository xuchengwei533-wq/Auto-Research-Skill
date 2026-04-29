"""Lightweight terminal UI for NightRunner (rich with plain fallback)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _short(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class _ExpRow:
    exp_id: str
    status: str
    metric: str
    delta: str
    api_time: str
    train_time: str
    total_time: str
    hypothesis: str


@dataclass
class _UIState:
    project_root: str = ""
    round_index: int = 0
    total_rounds: int = 0
    exp_id: str = ""
    stage: str = ""
    recent: list[_ExpRow] = field(default_factory=list)


class RunUI:
    def __init__(self, enabled: bool = True) -> None:
        self._plain_forced = not enabled
        self._state = _UIState()
        self._rich_ok = False
        self._console = None
        self._live = None
        self._Table = None
        self._Panel = None
        self._Layout = None
        self._Text = None
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if not self._plain_forced and self._is_tty:
            try:
                from rich.console import Console
                from rich.layout import Layout
                from rich.live import Live
                from rich.panel import Panel
                from rich.table import Table
                from rich.text import Text

                self._console = Console()
                self._live = Live
                self._Table = Table
                self._Panel = Panel
                self._Layout = Layout
                self._Text = Text
                self._rich_ok = True
            except Exception:
                self._rich_ok = False

        self.enabled = self._rich_ok
        self._fallback_print = print
        self._live_ctx = None

    def _status_style(self, status: str) -> str:
        mapping = {
            "keep": "green",
            "discard": "yellow",
            "crash": "red",
            "api_error": "red",
            "patch_error": "red",
            "violation": "red",
            "timeout": "red",
            "baseline": "cyan",
        }
        return mapping.get(status, "white")

    def _render(self) -> Any:
        if not self.enabled:
            return None
        assert self._Table and self._Panel and self._Layout
        layout = self._Layout()
        layout.split_column(
            self._Layout(name="top", size=8),
            self._Layout(name="bottom"),
        )
        header = (
            f"[bold]Project:[/bold] {self._state.project_root}\n"
            f"[bold]Round:[/bold] {self._state.round_index}/{self._state.total_rounds}\n"
            f"[bold]Experiment:[/bold] {self._state.exp_id or '-'}\n"
            f"[bold]Stage:[/bold] {self._state.stage or '-'}"
        )
        layout["top"].update(self._Panel(header, title="NightRunner"))

        table = self._Table(title="Recent Experiments", expand=True)
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Metric", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("API", justify="right")
        table.add_column("Train", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Hypothesis")
        for row in self._state.recent[-10:]:
            table.add_row(
                row.exp_id,
                f"[{self._status_style(row.status)}]{row.status}[/{self._status_style(row.status)}]",
                row.metric,
                row.delta,
                row.api_time,
                row.train_time,
                row.total_time,
                _short(row.hypothesis, 60),
            )
        layout["bottom"].update(table)
        return layout

    def _refresh(self) -> None:
        if not self.enabled:
            return
        if self._live_ctx is not None:
            self._live_ctx.update(self._render(), refresh=True)

    def start_run(self, project_root: Path, config: dict[str, Any], total_rounds: int, baseline_info: str) -> None:
        self._state.project_root = str(project_root)
        self._state.total_rounds = total_rounds
        self._state.stage = "Preparing run"
        if self.enabled and self._live is not None:
            self._live_ctx = self._live(self._render(), refresh_per_second=4, transient=False)
            self._live_ctx.__enter__()
        self.log(f"Baseline: {baseline_info}")

    def start_experiment(self, exp_id: str, round_index: int, total_rounds: int) -> None:
        self._state.exp_id = exp_id
        self._state.round_index = round_index
        self._state.total_rounds = total_rounds
        self._state.stage = "Preparing experiment"
        self._refresh()
        self.log(f"[{exp_id}/{total_rounds}] Preparing experiment...")

    def set_stage(self, stage: str) -> None:
        self._state.stage = stage
        self._refresh()
        if not self.enabled:
            exp = self._state.exp_id or "run"
            total = self._state.total_rounds or 1
            self._fallback_print(f"[{exp}/{total}] {stage}...")

    def log(self, message: str) -> None:
        if self.enabled and self._console is not None:
            self._console.print(message)
        else:
            self._fallback_print(message)

    def api_retry(self, retry_index: int, retry_total: int, wait_seconds: int, error: str) -> None:
        exp = self._state.exp_id or "run"
        self.log(f"[{exp}] API request failed: {_short(error, 180)}")
        self.log(f"[{exp}] Retry {retry_index}/{retry_total} after {wait_seconds}s...")

    def finish_experiment(self, result: dict[str, Any]) -> None:
        timings = result.get("timings", {}) or {}
        metric = result.get("metric_value")
        delta = result.get("delta_vs_baseline")
        row = _ExpRow(
            exp_id=str(result.get("id", "-")),
            status=str(result.get("status", "unknown")),
            metric="" if metric is None else f"{metric}",
            delta="" if delta is None else f"{delta}",
            api_time=f"{float(timings.get('api_seconds', 0.0)):.1f}s",
            train_time=f"{float(timings.get('training_seconds', 0.0)):.1f}s",
            total_time=f"{float(timings.get('total_seconds', 0.0)):.1f}s",
            hypothesis=str(result.get("hypothesis", "")),
        )
        self._state.recent.append(row)
        self._state.stage = "Finished"
        self._refresh()
        self.log(
            f"[{row.exp_id}/{self._state.total_rounds}] Decision: {row.status} "
            f"(metric={row.metric}, total={row.total_time})"
        )

    def finish_run(self, summary_path: Path) -> None:
        self._state.stage = "Completed"
        self._refresh()
        self.log(f"NightRunner run completed. Summary: {summary_path}")
        if self._live_ctx is not None:
            self._live_ctx.__exit__(None, None, None)
            self._live_ctx = None

    def error(self, message: str) -> None:
        self.log(f"Error: {message}")

