import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        self.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def run_batch_animation_job(root: Path, workspace_root: Path, ep_num: int) -> str:
    script_path = (root / "scripts" / "generate_all_scene_animations.py").resolve()
    env = os.environ.copy()
    env["CAP_WORKSPACE_ROOT"] = str(workspace_root)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", str(script_path), "--ep", str(ep_num)]

    print(f"[schedule] launch batch animation command={cmd}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    batch_status = ""
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("batch_status="):
            batch_status = str(line.split("=", 1)[1]).strip().lower()
    return_code = process.wait()
    print(f"[schedule] batch animation exit_code={return_code}", flush=True)
    if batch_status in {"done", "partial", "error"}:
        return batch_status
    return "done" if return_code == 0 else "error"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--stages-path", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app_utils.filesystem import find_episode_dir, parse_episode_info
    from app_utils.pipeline import StageRunner
    from app_utils.power import keep_system_awake
    from app_utils.schedules import read_schedule_job, schedule_job_log_path, update_schedule_job

    log_path = schedule_job_log_path(root, args.profile_id, args.job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    try:
        job = read_schedule_job(root, args.profile_id, args.job_id)
        if not job:
            print(f"[schedule] job not found: {args.job_id}", flush=True)
            return 1

        stage_id = str(job.get("stage_id", "")).strip()
        episodes = sorted(int(x) for x in (job.get("episodes") or []))
        schedule_at_text = str(job.get("schedule_at", "")).strip()
        schedule_at = datetime.fromisoformat(schedule_at_text) if schedule_at_text else datetime.now()
        stages_path = Path(args.stages_path) if args.stages_path else None
        runner = StageRunner(root, stages_path=stages_path, workspace_root=workspace_root, profile_id=args.profile_id)
        proc_meta_path = log_path.with_suffix(log_path.suffix + ".proc.json")
        proc_last_meta_path = log_path.with_suffix(log_path.suffix + ".proc.last.json")

        proc_meta_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": int(time.time()),
                    "step_name": f"schedule:{args.job_id}",
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

        print(
            f"\n==== RUN schedule:{args.job_id} | stage {stage_id} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====: episodes={episodes} "
            f"scheduled_at={schedule_at.isoformat(sep=' ', timespec='minutes')}",
            flush=True,
        )
        update_schedule_job(
            root,
            args.profile_id,
            args.job_id,
            {
                "status": "queued",
                "started_at": None,
                "ended_at": None,
                "current_episode": None,
                "completed_episodes": [],
                "partial_episodes": [],
                "failed_episodes": [],
                "last_message": f"waiting until {schedule_at.isoformat(sep=' ', timespec='minutes')}",
            },
        )

        with keep_system_awake(f"Scheduled stage {stage_id} ({args.job_id})"):
            while True:
                now = datetime.now()
                delta = (schedule_at - now).total_seconds()
                if delta <= 0:
                    break
                print(
                    f"[schedule] waiting {int(delta)}s until {schedule_at.isoformat(sep=' ', timespec='minutes')}",
                    flush=True,
                )
                time.sleep(min(max(int(delta), 1), 60))

            print("[schedule] starting queued stage run", flush=True)
            update_schedule_job(
                root,
                args.profile_id,
                args.job_id,
                {
                    "status": "running",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "partial_episodes": [],
                    "last_message": "running",
                },
            )

            completed_episodes = []
            partial_episodes = []
            failed_episodes = []

            for index, ep_num in enumerate(episodes, start=1):
                update_schedule_job(
                    root,
                    args.profile_id,
                    args.job_id,
                    {
                        "current_episode": ep_num,
                        "last_message": f"running episode {ep_num} ({index}/{len(episodes)})",
                    },
                )
                print(f"[schedule] start episode {ep_num} ({index}/{len(episodes)}) stage {stage_id}", flush=True)

                ep_path = find_episode_dir(root, ep_num, str(workspace_root.relative_to(root)))
                if ep_path is None:
                    print(f"[schedule] episode folder not found for ep={ep_num}", flush=True)
                    failed_episodes.append(ep_num)
                    update_schedule_job(
                        root,
                        args.profile_id,
                        args.job_id,
                        {
                            "failed_episodes": failed_episodes,
                            "partial_episodes": partial_episodes,
                            "last_message": f"episode {ep_num} not found",
                        },
                    )
                    continue

                ep_info = parse_episode_info(ep_path)
                try:
                    if stage_id == "4.1":
                        ep_stage_status = run_batch_animation_job(root, workspace_root, ep_num)
                    else:
                        result = runner.run_stage(ep_info, stage_id)
                        ep_stage_status = str((result.get("stages") or {}).get(stage_id, "error"))
                except Exception as exc:
                    ep_stage_status = "error"
                    print(f"[schedule] episode {ep_num} raised error: {exc}", flush=True)

                if ep_stage_status == "done":
                    completed_episodes.append(ep_num)
                    print(f"[schedule] episode {ep_num} completed stage {stage_id}", flush=True)
                elif ep_stage_status == "partial":
                    partial_episodes.append(ep_num)
                    print(f"[schedule] episode {ep_num} partially completed stage {stage_id}", flush=True)
                else:
                    failed_episodes.append(ep_num)
                    print(f"[schedule] episode {ep_num} failed stage {stage_id} ({ep_stage_status})", flush=True)

                update_schedule_job(
                    root,
                    args.profile_id,
                    args.job_id,
                    {
                        "completed_episodes": completed_episodes,
                        "partial_episodes": partial_episodes,
                        "failed_episodes": failed_episodes,
                        "last_message": f"episode {ep_num} => {ep_stage_status}",
                    },
                )

        final_status = "error" if failed_episodes else ("partial" if partial_episodes else "done")
        update_schedule_job(
            root,
            args.profile_id,
            args.job_id,
            {
                "status": final_status,
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "current_episode": None,
                "completed_episodes": completed_episodes,
                "partial_episodes": partial_episodes,
                "failed_episodes": failed_episodes,
                "last_message": f"finished with status={final_status}",
            },
        )
        print(
            f"[schedule] finished status={final_status} completed={completed_episodes} partial={partial_episodes} failed={failed_episodes}",
            flush=True,
        )
        return 0 if final_status in {"done", "partial"} else 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
