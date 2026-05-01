import json
import locale
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .filesystem import read_status, write_status, log_file_for_stage


def load_stage_config(root: Path, path_override: Path | None = None) -> Dict:
    """Load stages config from override path or default config/stages.yaml."""
    if path_override is not None:
        cfg_path = (root / path_override) if not Path(path_override).is_absolute() else Path(path_override)
    else:
        cfg_path = root / "config" / "stages.yaml"
    if not cfg_path.exists():
        return {"stages": []}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


class StageRunner:
    def __init__(
        self,
        root: Path,
        stages_path: Path | None = None,
        workspace_root: Path | None = None,
        profile_id: str = "",
    ):
        self.root = root
        self.stages_path = stages_path
        self.cfg = load_stage_config(root, stages_path)
        self.workspace_root = (workspace_root or (root / "workspace")).resolve()
        self.profile_id = str(profile_id or "").strip()

    def _format_args(self, args: List[str], ep_info: Dict) -> List[str]:
        return [a.format(**ep_info) for a in args]

    def _script_exists(self, script_rel: str) -> bool:
        return (self.root / script_rel).exists()

    def _prepare_step(self, ep_info: Dict, step: Dict) -> Tuple[str, str, list[str], dict[str, str]]:
        step_name = step.get("name", "step")
        step_type = step.get("type", "python")
        script_rel = step.get("script")
        args = self._format_args(step.get("args", []), ep_info)

        if step.get("skip_if_dir_has_files"):
            check_dir = (ep_info["path"] / step["skip_if_dir_has_files"]).resolve()
            if check_dir.exists() and any(check_dir.iterdir()):
                raise RuntimeError(f"skip {step_name}: already has files in {check_dir}")

        if step_type == "python":
            script_abs = (self.root / script_rel).resolve()
            if not script_abs.exists():
                if step.get("optional"):
                    raise FileNotFoundError(f"optional script missing: {script_rel}")
                raise FileNotFoundError(f"script not found: {script_rel}")
            cmd = [sys.executable, "-u", str(script_abs)] + args
        else:
            cmd = [script_rel] + args

        env = dict(os.environ)
        env["CAP_WORKSPACE_ROOT"] = str(self.workspace_root)
        if self.profile_id:
            env["CAP_PROFILE_ID"] = self.profile_id
        if step_type == "python":
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
        return step_name, step_type, cmd, env

    def _proc_meta_path(self, log_path: Path) -> Path:
        return log_path.with_suffix(log_path.suffix + ".proc.json")

    def _proc_last_meta_path(self, log_path: Path) -> Path:
        return log_path.with_suffix(log_path.suffix + ".proc.last.json")

    def _ts_label(self, ts: int | float | None) -> str:
        if not ts:
            return ""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    def _task_state_from_log(self, log_path: Path, run_id: str) -> Dict:
        proc_meta_path = self._proc_meta_path(log_path)
        proc_last_meta_path = self._proc_last_meta_path(log_path)
        state = {
            "running": False,
            "pid": None,
            "log": str(log_path),
            "proc_meta": str(proc_meta_path),
            "proc_last_meta": str(proc_last_meta_path),
            "started_at": None,
            "ended_at": None,
            "step_name": None,
        }
        if not proc_meta_path.exists():
            if proc_last_meta_path.exists():
                try:
                    last_meta = json.loads(proc_last_meta_path.read_text(encoding="utf-8"))
                    state["pid"] = int(last_meta.get("pid", 0) or 0)
                    state["started_at"] = last_meta.get("started_at")
                    state["ended_at"] = last_meta.get("ended_at")
                    state["step_name"] = last_meta.get("step_name")
                except Exception:
                    pass
            return state
        try:
            meta = json.loads(proc_meta_path.read_text(encoding="utf-8"))
        except Exception:
            return state

        pid = int(meta.get("pid", 0) or 0)
        state["pid"] = pid
        state["started_at"] = meta.get("started_at")
        state["step_name"] = meta.get("step_name")
        running = self._is_pid_running(pid)
        state["running"] = running
        if not running:
            ended_at = int(log_path.stat().st_mtime) if log_path.exists() else int(time.time())
            state["ended_at"] = ended_at
            last_meta = {
                "pid": pid,
                "started_at": meta.get("started_at"),
                "ended_at": ended_at,
                "step_name": meta.get("step_name"),
            }
            try:
                proc_last_meta_path.write_text(
                    json.dumps(last_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"==== END {run_id} | {meta.get('step_name', 'step')} | "
                        f"{self._ts_label(ended_at)} ===\n"
                    )
            except Exception:
                pass
            try:
                proc_meta_path.unlink()
            except FileNotFoundError:
                pass
        return state

    def _start_logged_process(self, log_path: Path, run_id: str, step_name: str, cmd: List[str], env: Dict[str, str]) -> Dict:
        proc_meta_path = self._proc_meta_path(log_path)
        proc_last_meta_path = self._proc_last_meta_path(log_path)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"\n==== RUN {run_id} | {step_name} | {self._ts_label(time.time())} ====: {cmd}\n"
            )
            p = subprocess.Popen(
                cmd,
                cwd=str(self.root),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        proc_meta_path.write_text(
            json.dumps(
                {
                    "pid": p.pid,
                    "started_at": int(time.time()),
                    "step_name": step_name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            proc_last_meta_path.unlink()
        except FileNotFoundError:
            pass
        return {"ok": True, "started": True, "message": "started", "pid": p.pid, "log": str(log_path)}

    def _is_pid_running(self, pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding=locale.getpreferredencoding(False) or "utf-8",
                errors="ignore",
            )
            out = (result.stdout or "").strip()
            return bool(out) and "No tasks are running" not in out and out.startswith('"')
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def get_substep_state(self, ep_info: Dict, sub_no: str) -> Dict:
        log_path = log_file_for_stage(ep_info["path"], f"sub{sub_no}")
        return self._task_state_from_log(log_path, f"sub{sub_no}")

    def get_stage_state(self, ep_info: Dict, stage_id: str) -> Dict:
        log_path = log_file_for_stage(ep_info["path"], str(stage_id))
        return self._task_state_from_log(log_path, str(stage_id))

    def start_substep(self, ep_info: Dict, sub_no: str, step: Dict) -> Dict:
        log_path = log_file_for_stage(ep_info["path"], f"sub{sub_no}")
        current = self.get_substep_state(ep_info, sub_no)
        if current["running"]:
            return {"ok": True, "started": False, "message": "already running", "log": str(log_path)}

        try:
            step_name, _step_type, cmd, env = self._prepare_step(ep_info, step)
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\n==== FAIL sub{sub_no} | {step.get('name', 'step')} | "
                    f"{self._ts_label(time.time())} | {e} ===\n"
                )
            return {"ok": False, "started": False, "message": str(e), "log": str(log_path)}

        return self._start_logged_process(log_path, f"sub{sub_no}", step_name, cmd, env)

    def start_stage(self, ep_info: Dict, stage_id: str) -> Dict:
        log_path = log_file_for_stage(ep_info["path"], str(stage_id))
        current = self.get_stage_state(ep_info, stage_id)
        if current["running"]:
            return {"ok": True, "started": False, "message": "already running", "log": str(log_path)}

        stage_def = None
        for s in self.cfg.get("stages", []):
            if str(s.get("id")) == str(stage_id):
                stage_def = s
                break
        if not stage_def:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\n==== FAIL {stage_id} | Stage {stage_id} | "
                    f"{self._ts_label(time.time())} | StageNotDefined ===\n"
                )
            return {"ok": False, "started": False, "message": f"stage {stage_id} not defined", "log": str(log_path)}

        status = read_status(ep_info["path"]) or {"stages": {}}
        status.setdefault("stages", {})
        status["stages"][str(stage_id)] = "running"
        write_status(ep_info["path"], status)

        runner_script = (self.root / "scripts" / "run_stage_background.py").resolve()
        stage_name = stage_def.get("name", f"Stage {stage_id}")
        cmd = [
            sys.executable,
            "-u",
            str(runner_script),
            "--root",
            str(self.root),
            "--workspace-root",
            str(self.workspace_root),
            "--ep",
            str(ep_info["ep"]),
            "--stage",
            str(stage_id),
        ]
        if self.profile_id:
            cmd.extend(["--profile-id", self.profile_id])
        if self.stages_path:
            stage_cfg_path = self.stages_path if Path(self.stages_path).is_absolute() else (self.root / self.stages_path)
            cmd.extend(["--stages-path", str(stage_cfg_path.resolve())])

        env = dict(os.environ)
        env["CAP_WORKSPACE_ROOT"] = str(self.workspace_root)
        if self.profile_id:
            env["CAP_PROFILE_ID"] = self.profile_id
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return self._start_logged_process(log_path, str(stage_id), stage_name, cmd, env)

    def _run_step(self, ep_info: Dict, stage_id: str, step: Dict, log_path: Path) -> Tuple[bool, str]:
        """Run a single step. Returns (ok, message)."""
        step_name = step.get("name", "step")
        with open(log_path, "a", encoding="utf-8", buffering=1) as lf:
            try:
                step_name, step_type, cmd, env = self._prepare_step(ep_info, step)
                started_at = int(time.time())
                lf.write(f"\n==== RUN {stage_id} | {step_name} | {self._ts_label(started_at)} ====: {cmd}\n")
                lf.flush()
                p = subprocess.Popen(
                    cmd,
                    cwd=str(self.root),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                code = p.wait()
                ended_at = int(time.time())
                if code == 0:
                    lf.write(f"==== OK {stage_id} | {step_name} | {self._ts_label(ended_at)} ===\n")
                    lf.flush()
                    return True, "ok"
                else:
                    lf.write(f"==== FAIL {stage_id} | {step_name} | {self._ts_label(ended_at)} | code={code} ===\n")
                    lf.flush()
                    return False, f"exit {code}"
            except RuntimeError as e:
                lf.write(f"==== SKIP {stage_id} | {step_name} | {self._ts_label(time.time())} | {e} ===\n")
                lf.flush()
                return True, str(e)
            except FileNotFoundError:
                lf.write(f"==== FAIL {stage_id} | {step_name} | {self._ts_label(time.time())} | FileNotFoundError ===\n")
                lf.flush()
                return False, "FileNotFound"
            except Exception as e:
                lf.write(f"==== FAIL {stage_id} | {step_name} | {self._ts_label(time.time())} | {e} ===\n")
                lf.flush()
                return False, str(e)

    def run_stage(self, ep_info: Dict, stage_id: str) -> Dict:
        status = read_status(ep_info["path"]) or {"stages": {}}
        status.setdefault("stages", {})
        log_path = log_file_for_stage(ep_info["path"], stage_id)

        status["stages"][stage_id] = "running"
        write_status(ep_info["path"], status)

        stage_def = None
        for s in self.cfg.get("stages", []):
            if str(s.get("id")) == str(stage_id):
                stage_def = s
                break
        if not stage_def:
            status["stages"][stage_id] = "error"
            write_status(ep_info["path"], status)
            Path(log_path).write_text(f"Stage {stage_id} not defined in config\n", encoding="utf-8")
            return status

        ok_all = True
        for step in stage_def.get("steps", []):
            ok, _msg = self._run_step(ep_info, stage_id, step, log_path)
            if not ok:
                ok_all = False
                break

        status["stages"][stage_id] = "done" if ok_all else "error"
        write_status(ep_info["path"], status)
        return status

    def run_substep(self, ep_info: Dict, sub_no: str, step: Dict) -> Dict:
        """Run a single ad-hoc substep using StageRunner's execution pipeline.
        Logs to stage_sub<no>.log and does not change stage status.json fields.
        Returns a dict with ok/message and log path for convenience.
        """
        from .filesystem import log_file_for_stage

        log_path = log_file_for_stage(ep_info["path"], f"sub{sub_no}")
        ok, msg = self._run_step(ep_info, f"sub{sub_no}", step, log_path)
        return {"ok": ok, "message": msg, "log": str(log_path)}
