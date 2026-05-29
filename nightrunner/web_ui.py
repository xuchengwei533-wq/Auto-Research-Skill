"""Minimal local Web UI for NightRunner."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .config import build_default_config, load_config, save_config
from .doctor import detect_metrics_from_text
from .experiments import get_experiment_paths, load_diff_text, load_experiment_metadata
from .runner import apply_experiment, check_auth, doctor, preview_experiment, run_night
from .sandbox import SandboxManager
from .state_store import load_best, load_experiments
from .train_runner import run_training
from .utils import ensure_dir, now_iso, read_json, read_text, write_json


class RunCoordinator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, rounds: int) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("NightRunner 已在运行中。")
            self._thread = threading.Thread(target=self._run_worker, args=(rounds,), daemon=True)
            self._thread.start()

    def _run_worker(self, rounds: int) -> None:
        try:
            run_night(self.project_root, rounds=rounds, plain=True)
        except Exception as exc:  # pragma: no cover
            write_json(
                ensure_dir(self.project_root / ".nightrunner" / "state") / "ui_session.json",
                {
                    "running": False,
                    "stage": "失败",
                    "error": str(exc),
                    "updated_at": now_iso(),
                },
            )


def _load_or_default_config(project_root: Path) -> dict[str, Any]:
    try:
        return load_config(project_root)
    except FileNotFoundError:
        return build_default_config(project_root.name)


def _save_ui_config(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = _load_or_default_config(project_root)
    editable_files = [str(item).replace("\\", "/") for item in payload.get("editable_files", []) if str(item).strip()]
    if editable_files:
        config["editable_files"] = editable_files
        config.setdefault("files", {})["editable"] = editable_files
    train_command = str(payload.get("train_command", "")).strip()
    if train_command:
        config.setdefault("execution", {})["train_command"] = train_command
    goal = str(payload.get("goal", "")).strip()
    if goal:
        config.setdefault("optimization", {})["goal"] = goal
    mode = str(payload.get("mode", "standard")).strip() or "standard"
    config.setdefault("optimization", {})["mode"] = mode
    metric = str(payload.get("metric", "")).strip()
    if metric:
        config.setdefault("metric", {})["name"] = metric
        config.setdefault("optimization", {})["metric"] = metric
    metric_regex = str(payload.get("metric_regex", "") or "").strip()
    if metric_regex:
        config.setdefault("metric", {})["regex"] = metric_regex
        config.setdefault("optimization", {})["metric_regex"] = metric_regex
    else:
        config.setdefault("metric", {}).pop("regex", None)
        config.setdefault("optimization", {})["metric_regex"] = None
    higher_is_better = bool(payload.get("higher_is_better", True))
    config.setdefault("metric", {})["lower_is_better"] = not higher_is_better
    config.setdefault("optimization", {})["higher_is_better"] = higher_is_better
    config.setdefault("execution", {})["backend"] = "sandbox"
    config.setdefault("safety", {})["semantic_guard"] = True
    config.setdefault("safety", {})["allow_protected_term_edits"] = False
    save_config(project_root, config)
    return config


def _run_test_command(project_root: Path, command: str) -> dict[str, Any]:
    config = _load_or_default_config(project_root)
    manager = SandboxManager.from_config(project_root, config)
    sandbox = manager.create_sandbox("ui_test_run")
    tmp_dir = ensure_dir(project_root / ".nightrunner" / "tmp")
    log_path = tmp_dir / "test_run.log"
    result = run_training(command=command, cwd=sandbox.project_dir, log_path=log_path, timeout_seconds=300)
    content = read_text(log_path)
    suggestions = detect_metrics_from_text(content)
    return {
        "returncode": result.get("returncode"),
        "timeout": result.get("timeout"),
        "log_tail": "\n".join(content.splitlines()[-80:]),
        "metric_suggestions": suggestions,
    }


def _root_html(project_root: Path) -> str:
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NightRunner UI</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; }}
    main {{ max-width:1200px; margin:0 auto; padding:24px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; }}
    .card {{ background:#111827; border:1px solid #334155; border-radius:12px; padding:16px; }}
    h1,h2 {{ margin:0 0 12px; }}
    textarea,input,select {{ width:100%; box-sizing:border-box; padding:10px; border-radius:8px; border:1px solid #475569; background:#020617; color:#e2e8f0; }}
    textarea {{ min-height:90px; }}
    button {{ border:0; border-radius:8px; padding:10px 14px; background:#2563eb; color:white; cursor:pointer; margin-right:8px; }}
    button.secondary {{ background:#475569; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#020617; padding:12px; border-radius:8px; border:1px solid #334155; max-height:320px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; }}
    td,th {{ border-bottom:1px solid #334155; padding:8px; text-align:left; vertical-align:top; }}
    .muted {{ color:#94a3b8; }}
  </style>
</head>
<body>
  <main>
    <h1>NightRunner 本地实验 UI</h1>
    <p class="muted">NightRunner 在隔离 sandbox 中运行实验。除非你显式点击 Apply，否则不会改动主项目文件。Git commit 是可选项，不是运行前置条件。</p>
    <div class="grid">
      <section class="card">
        <h2>1. Project Doctor</h2>
        <pre id="doctor">加载中...</pre>
      </section>
      <section class="card">
        <h2>2. Setup Wizard</h2>
        <label>目标类型</label>
        <select id="goal">
          <option>Tune hyperparameters</option>
          <option>Improve model architecture</option>
          <option>Improve training strategy</option>
          <option>Custom goal</option>
        </select>
        <label>运行模式</label>
        <select id="mode">
          <option value="standard">标准模式</option>
          <option value="config_only">Config files only - safest</option>
        </select>
        <label>Editable files（每行一个）</label>
        <textarea id="editable_files"></textarea>
        <div class="muted" style="margin:6px 0 10px;">Editable file permission only defines where NightRunner may propose changes. Protected keys and protected regions are still enforced inside editable files.</div>
        <label>训练命令</label>
        <textarea id="train_command"></textarea>
        <label>指标名</label>
        <input id="metric" />
        <label>指标正则（可选）</label>
        <input id="metric_regex" />
        <label>方向</label>
        <select id="higher_is_better">
          <option value="true">higher is better</option>
          <option value="false">lower is better</option>
        </select>
        <label>实验轮数</label>
        <input id="rounds" type="number" value="3" />
        <div style="margin-top:12px;">
          <button onclick="saveConfig()">保存配置</button>
          <button class="secondary" onclick="testRun()">Test Run</button>
          <button onclick="startRun()">开始运行</button>
        </div>
        <pre id="setup_result">尚未保存配置</pre>
      </section>
      <section class="card">
        <h2>3. Run Monitor</h2>
        <pre id="run_status">尚未启动</pre>
        <pre id="live_log">暂无日志</pre>
      </section>
      <section class="card">
        <h2>4. Experiments</h2>
        <div id="experiments_table" class="muted">暂无实验</div>
      </section>
      <section class="card" style="grid-column:1/-1;">
        <h2>5. Diff and Apply</h2>
        <div id="selected_exp" class="muted">点击上方实验的查看按钮。</div>
        <pre id="diff_view">暂无 diff</pre>
        <button onclick="applySelected()">Apply this experiment</button>
      </section>
    </div>
  </main>
  <script>
    const state = {{ selectedExp: null }};
    async function fetchJson(url, options) {{
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }}
    async function loadDoctor() {{
      const data = await fetchJson('/api/doctor');
      document.getElementById('doctor').textContent = JSON.stringify(data, null, 2);
    }}
    async function loadConfig() {{
      const data = await fetchJson('/api/config');
      document.getElementById('goal').value = data.optimization.goal || 'Tune hyperparameters';
      document.getElementById('mode').value = data.optimization.mode || 'standard';
      document.getElementById('editable_files').value = (data.files.editable || []).join('\\n');
      document.getElementById('train_command').value = data.execution.train_command || '';
      document.getElementById('metric').value = data.metric.name || '';
      document.getElementById('metric_regex').value = data.metric.regex || '';
      document.getElementById('higher_is_better').value = (!data.metric.lower_is_better).toString();
    }}
    async function saveConfig() {{
      const payload = {{
        goal: document.getElementById('goal').value.trim(),
        mode: document.getElementById('mode').value.trim(),
        editable_files: document.getElementById('editable_files').value.split('\\n').map(x => x.trim()).filter(Boolean),
        train_command: document.getElementById('train_command').value.trim(),
        metric: document.getElementById('metric').value.trim(),
        metric_regex: document.getElementById('metric_regex').value.trim(),
        higher_is_better: document.getElementById('higher_is_better').value === 'true'
      }};
      const data = await fetchJson('/api/config', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload) }});
      document.getElementById('setup_result').textContent = JSON.stringify(data, null, 2);
      await loadDoctor();
    }}
    async function testRun() {{
      const payload = {{ command: document.getElementById('train_command').value.trim() }};
      const data = await fetchJson('/api/test-run', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload) }});
      document.getElementById('setup_result').textContent = JSON.stringify(data, null, 2);
    }}
    async function startRun() {{
      await saveConfig();
      const payload = {{ rounds: parseInt(document.getElementById('rounds').value || '3', 10) }};
      const data = await fetchJson('/api/run', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload) }});
      document.getElementById('run_status').textContent = JSON.stringify(data, null, 2);
    }}
    async function loadRunStatus() {{
      const data = await fetchJson('/api/run-status');
      document.getElementById('run_status').textContent = JSON.stringify(data, null, 2);
      if (data.latest_log_tail) document.getElementById('live_log').textContent = data.latest_log_tail;
    }}
    async function loadExperiments() {{
      const data = await fetchJson('/api/experiments');
      const rows = data.items.map(item => `<tr>
        <td>${{item.id}}</td><td>${{item.status}}</td><td>${{item.metric_value ?? ''}}</td><td>${{(item.changed_files || []).join(', ')}}</td><td>${{item.created_at || ''}}</td>
        <td><button onclick=\"viewExperiment('${{item.id}}')\">查看</button></td>
      </tr>`).join('');
      document.getElementById('experiments_table').innerHTML = `<table><thead><tr><th>ID</th><th>Status</th><th>Metric</th><th>Changed files</th><th>Created at</th><th>Actions</th></tr></thead><tbody>${{rows}}</tbody></table>`;
    }}
    async function viewExperiment(expId) {{
      state.selectedExp = expId;
      const data = await fetchJson(`/api/experiments/${{expId}}`);
      document.getElementById('selected_exp').textContent = JSON.stringify(data.metadata, null, 2);
      document.getElementById('diff_view').textContent = data.diff || '(no diff)';
      document.getElementById('live_log').textContent = data.log_tail || '暂无日志';
    }}
    async function applySelected() {{
      if (!state.selectedExp) return;
      if (!confirm('确认将该实验写回主项目吗？')) return;
      const data = await fetchJson(`/api/experiments/${{state.selectedExp}}/apply`, {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{ confirm: true }}) }});
      alert(JSON.stringify(data, null, 2));
      await loadDoctor();
      await loadExperiments();
    }}
    async function tick() {{
      await loadRunStatus();
      await loadExperiments();
    }}
    loadDoctor(); loadConfig(); tick(); setInterval(tick, 2000);
  </script>
</body>
</html>
"""


def create_app(project_root: Path) -> FastAPI:
    app = FastAPI(title="NightRunner UI")
    coordinator = RunCoordinator(project_root)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_root_html(project_root))

    @app.get("/api/doctor")
    async def api_doctor() -> JSONResponse:
        return JSONResponse(doctor(project_root))

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        return JSONResponse(_load_or_default_config(project_root))

    @app.post("/api/config")
    async def api_save_config(payload: dict[str, Any]) -> JSONResponse:
        return JSONResponse(_save_ui_config(project_root, payload))

    @app.post("/api/test-run")
    async def api_test_run(payload: dict[str, Any]) -> JSONResponse:
        command = str(payload.get("command", "")).strip()
        if not command:
            raise HTTPException(status_code=400, detail="缺少训练命令。")
        return JSONResponse(_run_test_command(project_root, command))

    @app.post("/api/run")
    async def api_run(payload: dict[str, Any]) -> JSONResponse:
        auth = check_auth(project_root)
        if not auth.get("ok"):
            raise HTTPException(status_code=400, detail=str(auth.get("message")))
        rounds = int(payload.get("rounds", 3))
        coordinator.start(rounds)
        return JSONResponse({"ok": True, "running": True, "rounds": rounds})

    @app.get("/api/run-status")
    async def api_run_status() -> JSONResponse:
        session = read_json(project_root / ".nightrunner" / "state" / "ui_session.json", default={"running": False}) or {"running": False}
        experiments = load_experiments(project_root)
        latest_log_tail = ""
        if experiments:
            latest = experiments[-1]
            exp_id = latest.get("id")
            if isinstance(exp_id, str):
                log_path = get_experiment_paths(project_root, exp_id).log_path
                latest_log_tail = "\n".join(read_text(log_path).splitlines()[-80:])
        return JSONResponse(
            {
                "session": session,
                "best": load_best(project_root),
                "running": coordinator.is_running(),
                "latest_log_tail": latest_log_tail,
                "doctor": doctor(project_root),
            }
        )

    @app.get("/api/experiments")
    async def api_experiments() -> JSONResponse:
        items = list(reversed(load_experiments(project_root)))
        return JSONResponse({"items": items})

    @app.get("/api/experiments/{exp_id}")
    async def api_experiment_detail(exp_id: str) -> JSONResponse:
        meta = load_experiment_metadata(project_root, exp_id)
        if not meta:
            raise HTTPException(status_code=404, detail="实验不存在。")
        log_path = get_experiment_paths(project_root, exp_id).log_path
        return JSONResponse(
            {
                "metadata": meta,
                "diff": load_diff_text(project_root, exp_id),
                "log_tail": "\n".join(read_text(log_path).splitlines()[-120:]),
            }
        )

    @app.post("/api/experiments/{exp_id}/apply")
    async def api_apply_experiment(exp_id: str, payload: dict[str, Any]) -> JSONResponse:
        if not bool(payload.get("confirm")):
            raise HTTPException(status_code=400, detail="需要确认后才能 apply。")
        preview = preview_experiment(project_root, exp_id)
        if preview["conflicts"]:
            return JSONResponse({"ok": False, "conflicts": preview["conflicts"], "diff": preview["diff"]}, status_code=409)
        patch_path = apply_experiment(project_root, exp_id, confirm=False)
        return JSONResponse({"ok": True, "patch_path": str(patch_path)})

    return app


def launch_ui(project_root: Path, host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    app = create_app(project_root)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
