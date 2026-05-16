import json
import locale
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def schedule_jobs_dir(root: Path, profile_id: str) -> Path:
    path = root / "runtime" / "schedule_jobs" / str(profile_id or "default").strip()
    path.mkdir(parents=True, exist_ok=True)
    return path


def schedule_job_meta_path(root: Path, profile_id: str, job_id: str) -> Path:
    return schedule_jobs_dir(root, profile_id) / f"{job_id}.json"


def schedule_job_log_path(root: Path, profile_id: str, job_id: str) -> Path:
    return schedule_jobs_dir(root, profile_id) / f"{job_id}.log"


def schedule_job_launcher_path(root: Path, profile_id: str, job_id: str) -> Path:
    return schedule_jobs_dir(root, profile_id) / f"{job_id}.launch.ps1"


def _proc_meta_path(log_path: Path) -> Path:
    return log_path.with_suffix(log_path.suffix + ".proc.json")


def _proc_last_meta_path(log_path: Path) -> Path:
    return log_path.with_suffix(log_path.suffix + ".proc.last.json")


def schedule_task_name(job_id: str) -> str:
    return f"CAP2000_Schedule_{job_id}"


def _build_schedule_runner_cmd(
    root: Path,
    workspace_root: Path,
    stages_path: Path | None,
    job_id: str,
    profile_id: str,
) -> list[str]:
    runner_script = (root / "scripts" / "run_stage_schedule_background.py").resolve()
    cmd = [
        str(Path(sys.executable).resolve()),
        "-u",
        str(runner_script),
        "--root",
        str(root),
        "--workspace-root",
        str(workspace_root),
        "--job-id",
        str(job_id),
        "--profile-id",
        str(profile_id),
    ]
    if stages_path:
        stage_cfg_path = stages_path if Path(stages_path).is_absolute() else (root / stages_path)
        cmd.extend(["--stages-path", str(stage_cfg_path.resolve())])
    return cmd


def _write_schedule_launcher(
    root: Path,
    profile_id: str,
    job_id: str,
    cmd: list[str],
) -> Path:
    launcher_path = schedule_job_launcher_path(root, profile_id, job_id)
    launcher_lines = [
        '$ErrorActionPreference = "Stop"',
        '$env:PYTHONUNBUFFERED = "1"',
        "& " + " ".join([json.dumps(part, ensure_ascii=False) for part in cmd]),
        "exit $LASTEXITCODE",
        "",
    ]
    launcher_path.write_text("\n".join(launcher_lines), encoding="utf-8")
    return launcher_path


def read_schedule_job(root: Path, profile_id: str, job_id: str) -> Dict:
    path = schedule_job_meta_path(root, profile_id, job_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_schedule_job(root: Path, profile_id: str, job_id: str, data: Dict) -> Path:
    path = schedule_job_meta_path(root, profile_id, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def update_schedule_job(root: Path, profile_id: str, job_id: str, updates: Dict) -> Path:
    current = read_schedule_job(root, profile_id, job_id)
    current.update(updates)
    return write_schedule_job(root, profile_id, job_id, current)


def list_schedule_jobs(root: Path, profile_id: str) -> List[Dict]:
    out: List[Dict] = []
    for meta_path in schedule_jobs_dir(root, profile_id).glob("*.json"):
        if meta_path.name.endswith(".log.proc.json") or meta_path.name.endswith(".log.proc.last.json"):
            continue
        try:
            item = json.loads(meta_path.read_text(encoding="utf-8"))
            job_id = str(item.get("id", "")).strip()
            if not job_id:
                continue
            item["_meta_path"] = str(meta_path)
            item["_log_path"] = str(schedule_job_log_path(root, profile_id, job_id))
            out.append(item)
        except Exception:
            continue
    out.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return out


def _ts_label(ts: int | float | None) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _is_pid_running(pid: int) -> bool:
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


def _kill_pid_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False) or "utf-8",
            errors="ignore",
            check=False,
        )
        return
    try:
        os.kill(pid, 9)
    except OSError:
        pass


def _unregister_schedule_task(root: Path, task_name: str) -> None:
    task_name = str(task_name or "").strip()
    if not task_name:
        return
    unregister_script = (root / "scripts" / "unregister_schedule_task.ps1").resolve()
    try:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(unregister_script),
                "-TaskName",
                task_name,
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False) or "utf-8",
            errors="ignore",
            check=False,
        )
    except Exception:
        pass


def get_schedule_job_state(root: Path, profile_id: str, job_id: str) -> Dict:
    log_path = schedule_job_log_path(root, profile_id, job_id)
    proc_meta_path = _proc_meta_path(log_path)
    proc_last_meta_path = _proc_last_meta_path(log_path)
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
    running = _is_pid_running(pid)
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
                    f"==== END schedule:{job_id} | {meta.get('step_name', 'schedule')} | "
                    f"{_ts_label(ended_at)} ===\n"
                )
        except Exception:
            pass
        try:
            proc_meta_path.unlink()
        except FileNotFoundError:
            pass
    return state


def stop_schedule_job(root: Path, profile_id: str, job_id: str) -> Dict:
    job = read_schedule_job(root, profile_id, job_id)
    state = get_schedule_job_state(root, profile_id, job_id)
    if not job:
        return {"ok": False, "message": "schedule job not found"}

    if not state.get("running"):
        update_schedule_job(
            root,
            profile_id,
            job_id,
            {
                "status": "stopped",
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "current_episode": None,
                "last_message": "stopped",
            },
        )
        _unregister_schedule_task(root, str(job.get("task_name", "")))
        return {"ok": True, "stopped": False, "message": "job was not running"}

    pid = int(state.get("pid") or 0)
    if pid > 0:
        _kill_pid_tree(pid)

    _unregister_schedule_task(root, str(job.get("task_name", "")))

    log_path = schedule_job_log_path(root, profile_id, job_id)
    ended_at = int(time.time())
    last_meta = {
        "pid": pid,
        "started_at": state.get("started_at"),
        "ended_at": ended_at,
        "step_name": state.get("step_name") or f"schedule:{job_id}",
    }
    try:
        _proc_last_meta_path(log_path).write_text(
            json.dumps(last_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    try:
        _proc_meta_path(log_path).unlink()
    except FileNotFoundError:
        pass
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"\n==== STOP schedule:{job_id} | {last_meta['step_name']} | "
                f"{_ts_label(ended_at)} ====: interrupted by user\n"
            )
    except Exception:
        pass

    update_schedule_job(
        root,
        profile_id,
        job_id,
        {
            "status": "stopped",
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ended_at)),
            "current_episode": None,
            "last_message": "stopped by user",
        },
    )
    return {"ok": True, "stopped": True, "message": "stopped", "pid": pid}


def start_schedule_job(
    root: Path,
    profile_id: str,
    workspace_root: Path,
    stages_path: Path | None,
    job_id: str,
    step_name: str,
) -> Dict:
    log_path = schedule_job_log_path(root, profile_id, job_id)
    current = get_schedule_job_state(root, profile_id, job_id)
    if current["running"]:
        return {"ok": True, "started": False, "message": "already running", "log": str(log_path)}

    job = read_schedule_job(root, profile_id, job_id)
    schedule_at_text = str(job.get("schedule_at", "")).strip()
    if not schedule_at_text:
        return {"ok": False, "started": False, "message": "missing schedule_at", "log": str(log_path)}

    register_script = (root / "scripts" / "register_schedule_task.ps1").resolve()
    task_name = schedule_task_name(job_id)
    cmd = _build_schedule_runner_cmd(root, workspace_root, stages_path, job_id, profile_id)
    launcher_path = _write_schedule_launcher(root, profile_id, job_id, cmd)

    register_cmd = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(register_script),
        "-TaskName",
        task_name,
        "-Command",
        "powershell.exe",
        "-Arguments",
        subprocess.list2cmdline(
            [
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher_path),
            ]
        ),
        "-StartAt",
        schedule_at_text,
        "-Description",
        step_name,
    ]
    result = subprocess.run(
        register_cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False) or "utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "started": False,
            "message": (result.stderr or result.stdout or f"register task failed: {result.returncode}").strip(),
            "log": str(log_path),
        }

    register_info = {}
    stdout_text = (result.stdout or "").strip()
    if stdout_text:
        for line in reversed(stdout_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                register_info = json.loads(line)
                break
            except Exception:
                continue
    wake_to_run = bool(register_info.get("wake_to_run", True))
    task_mode = str(register_info.get("mode", "")).strip()
    register_note = str(register_info.get("note", "")).strip()

    update_schedule_job(
        root,
        profile_id,
        job_id,
        {
            "task_name": task_name,
            "task_mode": task_mode,
            "wake_to_run": wake_to_run,
            "launcher_path": str(launcher_path),
            "status": "queued",
            "last_message": (
                f"waiting until {schedule_at_text.replace('T', ' ')}"
                + (" (wake-to-run unavailable)" if not wake_to_run else "")
            ),
        },
    )
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"\n==== REGISTER schedule:{job_id} | {step_name} | "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ====: {task_name} @ {schedule_at_text} "
            f"| mode={task_mode or '-'} | wake_to_run={str(wake_to_run).lower()} | note={register_note or '-'}\n"
        )
    return {
        "ok": True,
        "started": True,
        "message": "scheduled",
        "task_name": task_name,
        "task_mode": task_mode,
        "wake_to_run": wake_to_run,
        "note": register_note,
        "log": str(log_path),
    }


def start_schedule_job_now(
    root: Path,
    profile_id: str,
    workspace_root: Path,
    stages_path: Path | None,
    job_id: str,
    step_name: str,
) -> Dict:
    log_path = schedule_job_log_path(root, profile_id, job_id)
    current = get_schedule_job_state(root, profile_id, job_id)
    if current["running"]:
        return {"ok": True, "started": False, "message": "already running", "log": str(log_path)}

    cmd = _build_schedule_runner_cmd(root, workspace_root, stages_path, job_id, profile_id)
    launcher_path = _write_schedule_launcher(root, profile_id, job_id, cmd)

    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher_path),
            ],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:
        return {
            "ok": False,
            "started": False,
            "message": f"start immediate job failed: {exc}",
            "log": str(log_path),
        }

    now_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    update_schedule_job(
        root,
        profile_id,
        job_id,
        {
            "task_name": "",
            "task_mode": "immediate",
            "wake_to_run": False,
            "launcher_path": str(launcher_path),
            "status": "queued",
            "last_message": "starting immediately",
        },
    )
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"\n==== START schedule:{job_id} | {step_name} | {now_text} ====: "
            f"immediate launcher pid={process.pid}\n"
        )
    return {
        "ok": True,
        "started": True,
        "message": "started",
        "task_mode": "immediate",
        "wake_to_run": False,
        "launcher_pid": process.pid,
        "log": str(log_path),
    }


def delete_schedule_job(root: Path, profile_id: str, job_id: str) -> Dict:
    job = read_schedule_job(root, profile_id, job_id)
    state = get_schedule_job_state(root, profile_id, job_id)
    meta_status = str(job.get("status", "queued")).strip().lower() or "queued"
    if state.get("running"):
        display_status = "running"
    elif meta_status == "queued":
        display_status = "waiting"
    else:
        display_status = meta_status

    if state.get("running") and display_status != "waiting":
        return {"ok": False, "message": "running job cannot be deleted"}

    if state.get("running") and state.get("pid"):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(int(state["pid"])), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    encoding=locale.getpreferredencoding(False) or "utf-8",
                    errors="ignore",
                    check=False,
                )
            else:
                os.kill(int(state["pid"]), 9)
        except Exception:
            pass

    _unregister_schedule_task(root, str(job.get("task_name", "")))

    log_path = schedule_job_log_path(root, profile_id, job_id)
    targets = [
        schedule_job_meta_path(root, profile_id, job_id),
        log_path,
        schedule_job_launcher_path(root, profile_id, job_id),
        _proc_meta_path(log_path),
        _proc_last_meta_path(log_path),
    ]
    removed = []
    for path in targets:
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except Exception:
            pass
    return {"ok": True, "message": "deleted", "removed": removed}


def cleanup_schedule_history(root: Path, profile_id: str) -> Dict:
    jobs = list_schedule_jobs(root, profile_id)
    deleted_ids: List[str] = []
    skipped_ids: List[str] = []
    errors: List[Dict[str, str]] = []

    for job in jobs:
        job_id = str(job.get("id", "")).strip()
        if not job_id:
            continue
        state = get_schedule_job_state(root, profile_id, job_id)
        if state.get("running"):
            skipped_ids.append(job_id)
            continue
        res = delete_schedule_job(root, profile_id, job_id)
        if res.get("ok"):
            deleted_ids.append(job_id)
        else:
            errors.append({"id": job_id, "message": str(res.get("message", ""))})

    return {
        "ok": len(errors) == 0,
        "deleted_ids": deleted_ids,
        "skipped_ids": skipped_ids,
        "errors": errors,
    }
