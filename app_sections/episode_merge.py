import os
import sys
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd
import streamlit as st

from app_utils.filesystem import (
    list_episode_dirs,
    parse_episode_info,
    subtitles_fixed_path,
    video_output_path,
)

ROOT: Path | None = None
profiles: dict[str, dict] = {}


def _not_configured(*args, **kwargs):
    raise RuntimeError("Episode merge section is not configured. Call configure_episode_merge first.")


story_subtitles_srt_path = _not_configured
media_duration_seconds = _not_configured
seconds_to_label = _not_configured
get_schedule_status_label = _not_configured
read_json_file = _not_configured
parse_srt_entries = _not_configured
safe_float = _not_configured
read_text_file = _not_configured
youtube_auth_file_signatures = _not_configured
list_youtube_playlists_cached = _not_configured
reauthorize_youtube_for_app = _not_configured
resolve_default_playlist_id = _not_configured


def configure_episode_merge(root: Path, profile_map: dict[str, dict], deps: dict) -> None:
    global ROOT, profiles
    global story_subtitles_srt_path, media_duration_seconds, seconds_to_label, get_schedule_status_label
    global read_json_file, parse_srt_entries, safe_float, read_text_file
    global youtube_auth_file_signatures, list_youtube_playlists_cached, reauthorize_youtube_for_app
    global resolve_default_playlist_id

    ROOT = root
    profiles = profile_map
    story_subtitles_srt_path = deps["story_subtitles_srt_path"]
    media_duration_seconds = deps["media_duration_seconds"]
    seconds_to_label = deps["seconds_to_label"]
    get_schedule_status_label = deps["get_schedule_status_label"]
    read_json_file = deps["read_json_file"]
    parse_srt_entries = deps["parse_srt_entries"]
    safe_float = deps["safe_float"]
    read_text_file = deps["read_text_file"]
    youtube_auth_file_signatures = deps["youtube_auth_file_signatures"]
    list_youtube_playlists_cached = deps["list_youtube_playlists_cached"]
    reauthorize_youtube_for_app = deps["reauthorize_youtube_for_app"]
    resolve_default_playlist_id = deps["resolve_default_playlist_id"]


def _root() -> Path:
    if ROOT is None:
        raise RuntimeError("Episode merge section is not configured. Call configure_episode_merge first.")
    return ROOT


MERGE_PROFILE_ID = "episode_merge"

def path_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
        return (int(stat.st_mtime), int(stat.st_size))
    except OSError:
        return (0, 0)

def merge_jobs_dir() -> Path:
    path = _root() / "runtime" / "merge_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def merge_output_root() -> Path:
    path = _root() / "workspaces" / "episode_merge"
    path.mkdir(parents=True, exist_ok=True)
    return path

def merge_saved_configs_dir() -> Path:
    path = merge_output_root() / "saved_configs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def merge_saved_config_path(config_id: str) -> Path:
    safe_id = slugify_filename(config_id, fallback="merge_config")
    return merge_saved_configs_dir() / f"{safe_id}.json"

def merge_job_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.json"

def merge_log_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.log"

def merge_launcher_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.launch.ps1"

def merge_proc_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.proc.json"

def merge_proc_last_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.proc.last.json"

def slugify_filename(text: str, fallback: str = "merged_episode") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(text or "").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:48] or fallback

def merge_source_profiles() -> dict[str, dict]:
    return {
        pid: prof
        for pid, prof in profiles.items()
        if pid not in {MERGE_PROFILE_ID, "youtube_ai"} and prof.get("workspace")
    }

def source_video_path(profile_id: str, ep_path: Path) -> Path:
    if profile_id == "story":
        return ep_path / "06_video" / "final_video.mp4"
    return video_output_path(ep_path)
def source_video_variants(profile_id: str, ep_path: Path) -> dict[str, str]:
    default_path = source_video_path(profile_id, ep_path)
    if profile_id == "story":
        return {
            "default": str(default_path),
            "no_outro": str(default_path),
            "with_outro": str(default_path),
        }
    output_dir = ep_path / "05_output"
    no_outro_path = output_dir / "final_video_no_outro.mp4"
    with_outro_path = output_dir / "final_video_with_outro.mp4"
    return {
        "default": str(default_path),
        "no_outro": str(no_outro_path if no_outro_path.exists() else default_path),
        "with_outro": str(with_outro_path if with_outro_path.exists() else default_path),
    }

def selected_merge_video_path(item: dict) -> str:
    include_outro = bool(item.get("include_outro"))
    if include_outro:
        return str(item.get("video_path_with_outro") or item.get("merge_video_path") or item.get("video_path") or "")
    return str(item.get("video_path_no_outro") or item.get("merge_video_path") or item.get("video_path") or "")

def normalize_merge_item(item: dict, order: int | None = None, default_include_outro: bool | None = None) -> dict:
    clean = dict(item)
    if default_include_outro is not None and "include_outro" not in clean:
        clean["include_outro"] = bool(default_include_outro)
    clean["include_outro"] = bool(clean.get("include_outro"))
    if not clean.get("video_path_no_outro"):
        clean["video_path_no_outro"] = clean.get("video_path", "")
    if not clean.get("video_path_with_outro"):
        clean["video_path_with_outro"] = clean.get("video_path", "")
    clean["merge_video_path"] = selected_merge_video_path(clean)
    if order is not None:
        clean["order"] = order
    return clean

def source_subtitle_path(profile_id: str, ep_path: Path) -> Path | None:
    if profile_id == "story":
        path = story_subtitles_srt_path(ep_path)
        return path if path.exists() else None
    for path in [subtitles_fixed_path(ep_path), ep_path / "02_subtitles" / "notebooklm_audio.srt"]:
        if path.exists():
            return path
    return None

def merge_episode_sources_signature(profile_id: str) -> tuple:
    profile_data = profiles.get(profile_id, {})
    workspace = profile_data.get("workspace", "")
    sig = []
    for ep_path in list_episode_dirs(_root(), workspace):
        video_path = source_video_path(profile_id, ep_path)
        variants = source_video_variants(profile_id, ep_path)
        subtitle_path = source_subtitle_path(profile_id, ep_path)
        sig.append(
            (
                ep_path.name,
                path_signature(video_path),
                path_signature(Path(variants.get("no_outro", ""))),
                path_signature(Path(variants.get("with_outro", ""))),
                path_signature(subtitle_path) if subtitle_path else (0, 0),
            )
        )
    return tuple(sig)

@st.cache_data(show_spinner=False, max_entries=32)
def _list_merge_episode_sources_cached(
    root_path: str,
    profile_id: str,
    profile_name: str,
    workspace: str,
    signature: tuple,
) -> list[dict]:
    rows = []
    root = Path(root_path)
    for ep_path in list_episode_dirs(root, workspace):
        info = parse_episode_info(ep_path)
        video_path = source_video_path(profile_id, ep_path)
        video_variants = source_video_variants(profile_id, ep_path)
        subtitle_path = source_subtitle_path(profile_id, ep_path)
        duration = None
        rows.append(
            {
                "profile_id": profile_id,
                "profile_name": profile_name,
                "ep": int(info.get("ep") or 0),
                "range": f"{int(info.get('start') or 0):04d}-{int(info.get('end') or 0):04d}",
                "ep_path": str(ep_path),
                "video_path": str(video_path),
                "video_path_no_outro": video_variants.get("no_outro", str(video_path)),
                "video_path_with_outro": video_variants.get("with_outro", str(video_path)),
                "include_outro": False,
                "merge_video_path": video_variants.get("no_outro", str(video_path)),
                "subtitle_path": str(subtitle_path or ""),
                "video_ready": video_path.exists(),
                "subtitle_ready": bool(subtitle_path and subtitle_path.exists()),
                "duration": duration,
            }
        )
    return sorted(rows, key=lambda row: int(row.get("ep") or 0))

def list_merge_episode_sources(profile_id: str) -> list[dict]:
    profile_data = profiles.get(profile_id, {})
    workspace = str(profile_data.get("workspace", "") or "")
    profile_name = str(profile_data.get("name", profile_id) or profile_id)
    return _list_merge_episode_sources_cached(
        str(_root()),
        profile_id,
        profile_name,
        workspace,
        merge_episode_sources_signature(profile_id),
    )

def merge_source_label(row: dict) -> str:
    ep_text = f"Ep{int(row.get('ep') or 0):02d}"
    ready_text = "影片就緒" if row.get("video_ready") else "缺影片"
    subtitle_text = "字幕就緒" if row.get("subtitle_ready") else "缺字幕"
    duration = row.get("duration")
    duration_text = seconds_to_label(duration) if duration is not None else "--"
    return f"{ep_text} ({row.get('range')}) | {ready_text} / {subtitle_text} | {duration_text}"

def read_merge_job(job_id: str) -> dict:
    path = merge_job_path(job_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_merge_job(job_id: str, payload: dict) -> Path:
    path = merge_job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def list_merge_saved_configs() -> list[dict]:
    configs = []
    for path in merge_saved_configs_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            configs.append(payload)
        except Exception:
            continue
    configs.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    return configs

def read_merge_saved_config(config_id: str) -> dict:
    path = merge_saved_config_path(config_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_merge_saved_config(title: str, items: list[dict], config_id: str = "") -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    clean_title = str(title or "").strip()
    if not config_id:
        config_id = f"cfg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify_filename(clean_title, fallback='merge_config')}"
    existing = read_merge_saved_config(config_id)
    clean_items = []
    for idx, item in enumerate(items, start=1):
        item = normalize_merge_item(item, idx, default_include_outro=(idx == len(items)))
        duration = merge_item_duration_seconds(item, probe_missing=True)
        clean_items.append(
            {
                "order": idx,
                "profile_id": item.get("profile_id"),
                "profile_name": item.get("profile_name"),
                "ep": int(item.get("ep") or 0),
                "range": item.get("range", ""),
                "ep_path": item.get("ep_path", ""),
                "video_path": item.get("video_path", ""),
                "video_path_no_outro": item.get("video_path_no_outro", ""),
                "video_path_with_outro": item.get("video_path_with_outro", ""),
                "include_outro": bool(item.get("include_outro")),
                "merge_video_path": item.get("merge_video_path", ""),
                "subtitle_path": item.get("subtitle_path", ""),
                "duration": duration,
            }
        )
    payload = {
        "id": config_id,
        "title": clean_title,
        "items": clean_items,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    path = merge_saved_config_path(config_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["_path"] = str(path)
    return payload

def merge_config_items_signature(items: list[dict]) -> list[dict]:
    sig = []
    for idx, item in enumerate(items, start=1):
        item = normalize_merge_item(item, idx, default_include_outro=(idx == len(items)))
        sig.append(
            {
                "order": idx,
                "profile_id": str(item.get("profile_id") or ""),
                "ep": int(item.get("ep") or 0),
                "video_path": str(item.get("video_path") or ""),
                "include_outro": bool(item.get("include_outro")),
                "merge_video_path": str(item.get("merge_video_path") or ""),
                "subtitle_path": str(item.get("subtitle_path") or ""),
            }
        )
    return sig

def find_matching_merge_saved_config(title: str, items: list[dict]) -> dict:
    clean_title = str(title or "").strip()
    target_sig = merge_config_items_signature(items)
    for config in list_merge_saved_configs():
        if str(config.get("title") or "").strip() != clean_title:
            continue
        if merge_config_items_signature(list(config.get("items") or [])) == target_sig:
            return config
    return {}

def get_or_write_merge_saved_config(title: str, items: list[dict], config_id: str = "") -> dict:
    config_id = str(config_id or "").strip()
    if config_id:
        return write_merge_saved_config(title, items, config_id)
    existing = find_matching_merge_saved_config(title, items)
    if existing:
        return existing
    return write_merge_saved_config(title, items, "")

def update_merge_saved_config_title(config_id: str, new_title: str) -> dict:
    clean_id = str(config_id or "").strip()
    clean_title = str(new_title or "").strip()
    if not clean_id or not clean_title:
        return {}
    payload = read_merge_saved_config(clean_id)
    if not payload:
        return {}
    payload["title"] = clean_title
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path = merge_saved_config_path(clean_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["_path"] = str(path)
    return payload

def delete_merge_saved_config(config_id: str) -> bool:
    clean_id = str(config_id or "").strip()
    if not clean_id:
        return False
    path = merge_saved_config_path(clean_id)
    if not path.exists():
        return False
    path.unlink()
    return True

def merge_saved_config_label(config: dict) -> str:
    title = str(config.get("title") or "未命名設定").strip()
    updated = str(config.get("updated_at") or config.get("created_at") or "").replace("T", " ")
    count = len(config.get("items") or [])
    return f"{title} | {count} 支影片 | {updated}"

def merge_item_duration_seconds(item: dict, probe_missing: bool = True) -> float | None:
    duration = item.get("duration")
    if duration not in (None, ""):
        try:
            return max(float(duration), 0.0)
        except Exception:
            pass
    if not probe_missing:
        return None
    video_path = selected_merge_video_path(item).strip() or str(item.get("video_path") or "").strip()
    if not video_path:
        return None
    return media_duration_seconds(video_path)

def merge_queue_duration_seconds(items: list[dict], probe_missing: bool = True) -> tuple[float, int]:
    total = 0.0
    missing = 0
    for item in items:
        duration = merge_item_duration_seconds(item, probe_missing=probe_missing)
        if duration is None:
            missing += 1
        else:
            total += duration
    return total, missing

def queue_from_editor(queue: list[dict], edited: pd.DataFrame | None) -> list[dict]:
    if edited is None or edited.empty:
        return list(queue)
    keep_indices = [idx for idx, row in edited.iterrows() if not bool(row.get("移除"))]
    reordered = []
    for idx in keep_indices:
        try:
            order_val = int(edited.at[idx, "順序"] or (idx + 1))
        except Exception:
            order_val = idx + 1
        if 0 <= int(idx) < len(queue):
            item = dict(queue[int(idx)])
            if "合併片尾" in edited.columns:
                item["include_outro"] = bool(edited.at[idx, "合併片尾"])
            reordered.append((order_val, int(idx), normalize_merge_item(item)))
    reordered.sort(key=lambda row: (row[0], row[1]))
    return [normalize_merge_item(row[2], idx, default_include_outro=(idx == len(reordered))) for idx, row in enumerate(reordered, start=1)]

def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        out = (result.stdout or "").strip()
        return bool(out) and "No tasks are running" not in out and out.startswith('"')
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_merge_job_state(job_id: str) -> dict:
    job = read_merge_job(job_id)
    proc_path = merge_proc_path(job_id)
    state = {"running": False, "pid": None, "job": job, "log": str(merge_log_path(job_id))}
    if proc_path.exists():
        try:
            proc = json.loads(proc_path.read_text(encoding="utf-8"))
            pid = int(proc.get("pid") or 0)
            state["pid"] = pid
            state["running"] = is_pid_running(pid)
        except Exception:
            pass
    return state

def get_merge_upload_state(job_id: str) -> dict:
    proc_path = merge_upload_proc_path(job_id)
    state = {"running": False, "pid": None, "log": str(merge_upload_log_path(job_id))}
    if proc_path.exists():
        try:
            proc = json.loads(proc_path.read_text(encoding="utf-8"))
            pid = int(proc.get("pid") or 0)
            state["pid"] = pid
            state["running"] = is_pid_running(pid)
        except Exception:
            pass
    return state

def start_merge_upload_job(job: dict, privacy: str, publish_at: str = "", playlist_id: str = "") -> dict:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return {"ok": False, "message": "missing merge job id"}
    state = get_merge_upload_state(job_id)
    if state.get("running"):
        return {"ok": True, "started": False, "message": "already running", "log": state.get("log")}

    job_path = merge_job_path(job_id)
    script_path = _root() / "scripts" / "upload_merge_to_youtube.py"
    log_path = merge_upload_log_path(job_id)
    proc_path = merge_upload_proc_path(job_id)
    cmd = [
        str(Path(sys.executable).resolve()),
        "-u",
        str(script_path),
        "--job",
        str(job_path.resolve()),
        "--privacy",
        str(privacy or "private"),
        "--proc_path",
        str(proc_path.resolve()),
    ]
    if publish_at:
        cmd.extend(["--publish_at", publish_at])
    if playlist_id:
        cmd.extend(["--playlist_id", playlist_id])
    cover_path = merge_job_cover_path(job)
    if cover_path:
        cmd.extend(["--cover_path", str(cover_path.resolve())])
    try:
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        log_file.write(f"\n==== START merge-upload:{job_id} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd=str(_root()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env=env,
        )
        log_file.close()
        return {"ok": True, "started": True, "pid": process.pid, "log": str(log_path)}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "log": str(log_path)}

def merge_jobs_signature() -> tuple:
    sig = []
    for path in merge_jobs_dir().glob("merge_*.json"):
        sig.append((path.name, path_signature(path)))
    return tuple(sorted(sig))

@st.cache_data(show_spinner=False, max_entries=64)
def _list_merge_jobs_for_config_cached(jobs_dir: str, config_id: str, signature: tuple) -> list[dict]:
    jobs = []
    for path in Path(jobs_dir).glob("merge_*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if config_id and str(job.get("saved_config_id") or "") != str(config_id):
                continue
            jobs.append(job)
        except Exception:
            continue
    jobs.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return jobs

def list_merge_jobs_for_config(config_id: str = "") -> list[dict]:
    return _list_merge_jobs_for_config_cached(
        str(merge_jobs_dir()),
        str(config_id or ""),
        merge_jobs_signature(),
    )

def merge_job_output_video_path(job: dict) -> Path:
    explicit = str(job.get("output_video") or "").strip()
    if explicit:
        return Path(explicit)
    return Path(str(job.get("output_dir") or "")) / "final_video.mp4"

def merge_job_output_subtitle_path(job: dict) -> Path:
    explicit = str(job.get("output_subtitle") or "").strip()
    if explicit:
        return Path(explicit)
    return Path(str(job.get("output_dir") or "")) / "merged_subtitles.srt"

def merge_job_metadata_path(job: dict) -> Path:
    return Path(str(job.get("output_dir") or "")) / "youtube_meta.json"

def merge_job_upload_record_path(job: dict) -> Path:
    return Path(str(job.get("output_dir") or "")) / "upload_record.json"

def merge_job_cover_path(job: dict) -> Path | None:
    output_dir = Path(str(job.get("output_dir") or ""))
    for name in ["cover.png", "cover.jpg", "cover.jpeg", "thumbnail.png", "thumbnail.jpg", "thumbnail.jpeg"]:
        path = output_dir / name
        if path.exists():
            return path
    return None

def save_merge_cover_upload(job: dict, uploaded_file) -> Path:
    output_dir = Path(str(job.get("output_dir") or ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(uploaded_file.name or "")).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    target = output_dir / f"cover{suffix}"
    target.write_bytes(uploaded_file.getvalue())
    return target

def merge_upload_log_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.upload.log"

def merge_upload_proc_path(job_id: str) -> Path:
    return merge_jobs_dir() / f"{job_id}.upload.proc.json"

def merge_source_metadata_paths(item: dict) -> list[Path]:
    ep_path = Path(str(item.get("ep_path") or ""))
    if not ep_path.exists():
        return []
    candidates = [
        ep_path / "05_output" / "youtube_meta.json",
        ep_path / "07_publish" / "youtube_meta.json",
        ep_path / "youtube_meta.json",
    ]
    return [path for path in candidates if path.exists()]

def merge_preview_job_label(job: dict) -> str:
    status = get_schedule_status_label(str(job.get("status") or "queued"))
    created = str(job.get("created_at") or "").replace("T", " ")
    title = str(job.get("title") or job.get("id") or "").strip()
    return f"[{status}] {title} | {created}"

def merge_job_display_status(job: dict, state: dict | None = None) -> str:
    state = state or get_merge_job_state(str(job.get("id") or ""))
    if state.get("running"):
        return "running"
    status = str(job.get("status") or "queued").strip().lower()
    if status == "queued":
        schedule_text = str(job.get("schedule_at") or "").strip()
        try:
            if schedule_text and datetime.fromisoformat(schedule_text) < datetime.now() and not job.get("started_at"):
                return "missed"
        except Exception:
            pass
    return status

def merge_job_status_label(status_code: str) -> str:
    labels = {
        "queued": "等待中",
        "running": "執行中",
        "done": "完成",
        "error": "失敗",
        "missed": "排程未啟動",
    }
    return labels.get(str(status_code or "").strip().lower(), str(status_code or "—"))

def build_merge_segment_rows(job: dict) -> pd.DataFrame:
    rows = []
    offset = 0.0
    for idx, item in enumerate(job.get("items") or [], start=1):
        duration = merge_item_duration_seconds(item)
        end = offset + float(duration or 0.0)
        rows.append(
            {
                "順序": idx,
                "專案模式": item.get("profile_name", item.get("profile_id", "")),
                "集數": f"Ep{int(item.get('ep') or 0):02d}",
                "來源範圍": item.get("range", ""),
                "開始時間": seconds_to_label(offset),
                "結束時間": seconds_to_label(end) if duration is not None else "未知",
                "片段長度": seconds_to_label(duration) if duration is not None else "未知",
                "來源影片": item.get("video_path", ""),
            }
        )
        offset = end
    return pd.DataFrame(rows)

def parse_metadata_time_to_seconds(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", text)
    if not match:
        return None
    first = int(match.group(1))
    second = int(match.group(2))
    third = match.group(3)
    if third is None:
        return float(first * 60 + second)
    return float(first * 3600 + second * 60 + int(third))

def format_youtube_chapter_time(seconds_value: float) -> str:
    total = max(0, int(round(float(seconds_value))))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def parse_source_chapters_from_metadata(meta: dict) -> list[dict]:
    chapters = []
    raw_chapters = meta.get("chapters")
    if isinstance(raw_chapters, list):
        for item in raw_chapters:
            if not isinstance(item, dict):
                continue
            seconds = parse_metadata_time_to_seconds(str(item.get("time") or ""))
            title = str(item.get("title") or "").strip()
            if seconds is not None and title:
                chapters.append({"seconds": seconds, "title": title})
    if chapters:
        return chapters

    description = str(meta.get("description") or "")
    line_time_pattern = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[-:：]?\s*(.+?)\s*$")
    for raw_line in description.splitlines():
        line = raw_line.strip()
        match = line_time_pattern.match(line)
        if not match:
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        third = match.group(3)
        seconds = float(first * 60 + second) if third is None else float(first * 3600 + second * 60 + int(third))
        title = str(match.group(4) or "").strip(" -:：")
        if title:
            chapters.append({"seconds": seconds, "title": title})
    return chapters

def merge_source_metadata_bundle_signature(job: dict) -> tuple:
    sig = []
    for item in job.get("items") or []:
        selected_video = Path(selected_merge_video_path(item))
        metadata_paths = merge_source_metadata_paths(item)
        sig.append(
            (
                str(item.get("profile_id") or ""),
                int(item.get("ep") or 0),
                str(item.get("range") or ""),
                path_signature(selected_video),
                tuple((str(path), path_signature(path)) for path in metadata_paths),
            )
        )
    return tuple(sig)

@st.cache_data(show_spinner=False, max_entries=64)
def _merge_source_metadata_bundle_cached(job_json: str, signature: tuple) -> dict:
    job = json.loads(job_json)
    sources = []
    merged_chapters = []
    offset = 0.0
    total_words = 0
    for source_index, item in enumerate(job.get("items") or [], start=1):
        duration = merge_item_duration_seconds(item) or 0.0
        start_word = int(item.get("range", "0-0").split("-")[0] or 0) if "-" in str(item.get("range", "")) else 0
        end_word = int(item.get("range", "0-0").split("-")[-1] or 0) if "-" in str(item.get("range", "")) else 0
        if start_word and end_word and end_word >= start_word:
            total_words += end_word - start_word + 1

        source_meta = {}
        source_meta_path = None
        for path in merge_source_metadata_paths(item):
            source_meta = read_json_file(path) or {}
            source_meta_path = path
            if source_meta:
                break
        source_chapters = parse_source_chapters_from_metadata(source_meta)
        source_label = f"Ep{int(item.get('ep') or 0):02d} ({item.get('range', '')})"
        sources.append(
            {
                "order": source_index,
                "profile_name": item.get("profile_name", item.get("profile_id", "")),
                "episode": source_label,
                "offset_seconds": offset,
                "duration_seconds": duration,
                "metadata_path": str(source_meta_path or ""),
                "title": source_meta.get("title", ""),
                "summary": source_meta.get("summary", ""),
                "description": source_meta.get("description", ""),
                "tags": source_meta.get("tags", []),
                "chapter_count": len(source_chapters),
            }
        )
        if source_chapters:
            for chapter in source_chapters:
                chapter_seconds = offset + float(chapter.get("seconds") or 0.0)
                chapter_title = str(chapter.get("title") or "").strip()
                merged_chapters.append(
                    {
                        "time": format_youtube_chapter_time(chapter_seconds),
                        "title": f"{source_label} {chapter_title}",
                        "source_episode": source_label,
                        "source_time": format_youtube_chapter_time(float(chapter.get("seconds") or 0.0)),
                    }
                )
        else:
            merged_chapters.append(
                {
                    "time": format_youtube_chapter_time(offset),
                    "title": f"{source_label} 開始",
                    "source_episode": source_label,
                    "source_time": "00:00",
                }
            )
        offset += duration
    return {
        "sources": sources,
        "merged_chapters": merged_chapters,
        "total_words": total_words,
        "total_episodes": len(job.get("items") or []),
        "total_duration_seconds": offset,
    }

def merge_source_metadata_bundle(job: dict) -> dict:
    job_json = json.dumps(job, ensure_ascii=False, sort_keys=True)
    return _merge_source_metadata_bundle_cached(job_json, merge_source_metadata_bundle_signature(job))

def extract_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError("LLM response did not contain a JSON object")

def merge_metadata_prompt(job: dict, subtitle_text: str, source_bundle: dict) -> str:
    title = str(job.get("title") or "").strip()
    source_rows = []
    for idx, item in enumerate(job.get("items") or [], start=1):
        source_rows.append(
            f"{idx}. {item.get('profile_name', item.get('profile_id', ''))} "
            f"Ep{int(item.get('ep') or 0):02d} ({item.get('range', '')})"
        )
    sources_text = "\n".join(source_rows) or "無來源資訊"
    merged_chapters_text = "\n".join(
        f"{item.get('time')} {item.get('title')}"
        for item in source_bundle.get("merged_chapters", [])
    )
    source_meta_summary = []
    for source in source_bundle.get("sources", []):
        source_meta_summary.append(
            {
                "episode": source.get("episode", ""),
                "offset": format_youtube_chapter_time(source.get("offset_seconds", 0)),
                "metadata_path": source.get("metadata_path", ""),
                "tags": source.get("tags", []),
                "chapter_count": source.get("chapter_count", 0),
            }
        )
    return f"""你是 YouTube SEO 與影片上架 metadata 編輯。請根據合併後字幕，產生繁體中文 YouTube metadata。

重要規則：
- 不要重新產生 title。Title 已固定為：{title}
- 這是 {source_bundle.get("total_episodes", 0)} 集合併影片，總共 {source_bundle.get("total_words", 0)} 個單字。summary 必須明確寫出這個事實。
- 只產生 description、tags、summary。章節時間軸已由系統根據各集 metadata 與影片 offset 算好，不要自行改寫章節時間。
- description 要適合直接貼到 YouTube 說明欄，需包含簡短影片介紹、合併來源集數摘要，並可自然提到完整章節時間軸會附在後方。
- tags 請給 8 到 15 個繁體中文或英文關鍵字，不要包含 #。
- 回覆必須是 JSON，不要加 markdown。

來源集數：
{sources_text}

各集 metadata 摘要：
{json.dumps(source_meta_summary, ensure_ascii=False, indent=2)}

系統已計算完成的完整章節時間軸，description 不要漏掉這些來源範圍：
{merged_chapters_text}

合併後字幕 SRT：
{subtitle_text}

請輸出 JSON 格式：
{{
  "summary": "...",
  "description": "...",
  "tags": ["..."]
}}
"""

def generate_merge_metadata_with_gemini(job: dict, prompt: str) -> tuple[dict, str]:
    from google import genai
    from google.genai import types

    model = os.environ.get("CAP_MERGE_METADATA_GEMINI_MODEL", os.environ.get("CAP_TEXT_MODEL", "gemini-2.5-pro"))
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return extract_json_object(response.text), f"gemini:{model}"

def generate_merge_metadata_with_openai(job: dict, prompt: str) -> tuple[dict, str]:
    from openai import OpenAI

    model = os.environ.get("CAP_MERGE_METADATA_OPENAI_MODEL", os.environ.get("OPENAI_METADATA_MODEL", "gpt-4o-mini"))
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You generate valid JSON only. Do not wrap the response in markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )
    return extract_json_object(response.output_text), f"openai:{model}"

def generate_merge_metadata_with_nvidia(job: dict, prompt: str) -> tuple[dict, str]:
    scripts_dir = _root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response

    model = os.environ.get("CAP_MERGE_METADATA_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)
    response_text = nvidia_chat_response(
        prompt,
        model=model,
        system_prompt="You generate valid JSON only. Do not wrap the response in markdown.",
        temperature=0.2,
    )
    return extract_json_object(response_text), f"nvidia:{model}"

def generate_merge_metadata(job: dict, provider: str = "auto") -> dict:
    subtitle_path = merge_job_output_subtitle_path(job)
    if not subtitle_path.exists():
        raise FileNotFoundError(f"找不到合併字幕：{subtitle_path}")
    subtitle_text = subtitle_path.read_text(encoding="utf-8", errors="ignore")
    if not subtitle_text.strip():
        raise ValueError("合併字幕是空的，無法產生 metadata。")

    from dotenv import load_dotenv

    load_dotenv(_root() / ".env")
    source_bundle = merge_source_metadata_bundle(job)
    prompt = merge_metadata_prompt(job, subtitle_text, source_bundle)
    provider = str(provider or "auto").strip().lower()
    errors = []
    ai_data = {}
    generated_by = ""

    if provider in {"auto", "gemini"}:
        try:
            ai_data, generated_by = generate_merge_metadata_with_gemini(job, prompt)
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise

    if not ai_data and provider in {"auto", "openai"}:
        try:
            ai_data, generated_by = generate_merge_metadata_with_openai(job, prompt)
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            raise RuntimeError("Metadata generation failed: " + " | ".join(errors)) from exc
            """
            raise RuntimeError("Metadata 產生失敗；" + " | ".join(errors)) from exc

    """
    if not ai_data and provider == "nvidia":
        try:
            ai_data, generated_by = generate_merge_metadata_with_nvidia(job, prompt)
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            raise RuntimeError("Metadata generation failed: " + " | ".join(errors)) from exc

    if not ai_data:
        raise RuntimeError("Metadata generation failed: " + " | ".join(errors))
        """
        raise RuntimeError("Metadata 產生失敗；" + " | ".join(errors))

    """
    title = str(job.get("title") or "").strip()
    chapters = source_bundle.get("merged_chapters") or ai_data.get("chapters") or []
    summary = str(ai_data.get("summary") or "").strip()
    if source_bundle.get("total_episodes") and source_bundle.get("total_words"):
        required_summary_prefix = (
            f"本集是 {int(source_bundle.get('total_episodes') or 0)} 集合併的總複習，"
            f"完整整理 {int(source_bundle.get('total_words') or 0)} 個國中英文單字。"
        )
        if str(source_bundle.get("total_words")) not in summary:
            summary = required_summary_prefix + (" " + summary if summary else "")

    chapter_text = "\n".join(f"{item.get('time')} {item.get('title')}" for item in chapters)
    source_text = "\n".join(
        f"{idx}. {source.get('profile_name', '')} {source.get('episode', '')}"
        for idx, source in enumerate(source_bundle.get("sources", []), start=1)
    )
    description = str(ai_data.get("description") or "").strip()
    description_parts = []
    if summary:
        description_parts.append(summary)
    if description and description != summary:
        description_parts.append(description)
    if source_text:
        description_parts.append("合併來源集數：\n" + source_text)
    if chapter_text:
        description_parts.append("章節時間軸：\n" + chapter_text)

    meta = {
        "title": title,
        "description": "\n\n".join(description_parts).strip(),
        "tags": [str(tag).strip() for tag in (ai_data.get("tags") or []) if str(tag).strip()],
        "summary": summary,
        "chapters": chapters,
        "source_metadata": source_bundle.get("sources", []),
        "total_episodes": source_bundle.get("total_episodes", 0),
        "total_words": source_bundle.get("total_words", 0),
        "default_language": "zh-TW",
        "default_audio_language": "zh-TW",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_by": generated_by,
        "source_job_id": job.get("id", ""),
        "source_saved_config_id": job.get("saved_config_id", ""),
    }
    return meta

def write_merge_metadata(job: dict, meta: dict) -> Path:
    path = merge_job_metadata_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def build_merge_job_payload(title: str, items: list[dict], scheduled_at: datetime | None = None, saved_config: dict | None = None) -> tuple[str, dict]:
    now_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_ep = int(items[0].get("ep") or 0) if items else 0
    last_ep = int(items[-1].get("ep") or first_ep) if items else first_ep
    job_id = f"merge_{now_text}_ep{first_ep:02d}_{last_ep:02d}"
    output_dir = merge_output_root() / f"{now_text}_{slugify_filename(title)}"
    clean_items = []
    for idx, item in enumerate(items, start=1):
        item = normalize_merge_item(item, idx, default_include_outro=(idx == len(items)))
        clean_items.append(
            {
                "order": idx,
                "profile_id": item.get("profile_id"),
                "profile_name": item.get("profile_name"),
                "ep": int(item.get("ep") or 0),
                "range": item.get("range", ""),
                "ep_path": item.get("ep_path", ""),
                "video_path": item.get("video_path", ""),
                "video_path_no_outro": item.get("video_path_no_outro", ""),
                "video_path_with_outro": item.get("video_path_with_outro", ""),
                "include_outro": bool(item.get("include_outro")),
                "merge_video_path": item.get("merge_video_path", ""),
                "subtitle_path": item.get("subtitle_path", ""),
            }
        )
    payload = {
        "id": job_id,
        "title": title,
        "saved_config_id": (saved_config or {}).get("id", ""),
        "saved_config_updated_at": (saved_config or {}).get("updated_at", ""),
        "items": clean_items,
        "output_dir": str(output_dir),
        "log_path": str(merge_log_path(job_id)),
        "proc_path": str(merge_proc_path(job_id)),
        "proc_last_path": str(merge_proc_last_path(job_id)),
        "schedule_at": scheduled_at.isoformat(timespec="seconds") if scheduled_at else datetime.now().isoformat(timespec="seconds"),
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "ended_at": None,
        "last_message": "queued",
    }
    return job_id, payload

def write_merge_launcher(job_id: str) -> Path:
    launcher_path = merge_launcher_path(job_id)
    script_path = (_root() / "scripts" / "merge_episode_videos.py").resolve()
    job_path = merge_job_path(job_id).resolve()
    log_path = merge_log_path(job_id).resolve()
    cmd_parts = [str(Path(sys.executable).resolve()), "-u", str(script_path), "--job", str(job_path)]
    launcher_lines = [
        '$ErrorActionPreference = "Continue"',
        '$env:PYTHONUNBUFFERED = "1"',
        '$env:PYTHONIOENCODING = "utf-8"',
        '$env:PYTHONUTF8 = "1"',
        "& " + " ".join(json.dumps(part, ensure_ascii=False) for part in cmd_parts) + " *>> " + json.dumps(str(log_path), ensure_ascii=False),
        "exit $LASTEXITCODE",
        "",
    ]
    launcher_path.write_text("\n".join(launcher_lines), encoding="utf-8")
    return launcher_path

def start_merge_job_now(job_id: str) -> dict:
    log_path = merge_log_path(job_id)
    script_path = (_root() / "scripts" / "merge_episode_videos.py").resolve()
    job_path = merge_job_path(job_id).resolve()
    try:
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        log_file.write(f"\n==== START merge:{job_id} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            [str(Path(sys.executable).resolve()), "-u", str(script_path), "--job", str(job_path)],
            cwd=str(_root()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env=env,
        )
        log_file.close()
        return {"ok": True, "pid": process.pid, "log": str(log_path)}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "log": str(log_path)}

def schedule_merge_job(job_id: str, scheduled_at: datetime) -> dict:
    launcher_path = write_merge_launcher(job_id)
    task_name = f"CAP2000_Merge_{job_id}"
    register_script = (_root() / "scripts" / "register_schedule_task.ps1").resolve()
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
        scheduled_at.isoformat(timespec="seconds"),
        "-Description",
        f"CAP2000 episode merge {job_id}",
    ]
    result = subprocess.run(
        register_cmd,
        cwd=str(_root()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        return {"ok": False, "message": (result.stderr or result.stdout or "").strip()}
    job = read_merge_job(job_id)
    job.update({
        "task_name": task_name,
        "launcher_path": str(launcher_path),
        "last_message": f"scheduled for {scheduled_at.isoformat(sep=' ', timespec='minutes')}",
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    })
    write_merge_job(job_id, job)
    with open(merge_log_path(job_id), "a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n==== REGISTER merge:{job_id} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"====: {scheduled_at.isoformat(sep=' ', timespec='minutes')}\n"
        )
    return {"ok": True, "task_name": task_name, "log": str(merge_log_path(job_id))}

def render_episode_merge_workspace() -> None:
    st.header("集數合併")

    saved_configs = list_merge_saved_configs()
    saved_config_map = {str(config.get("id")): config for config in saved_configs if str(config.get("id", "")).strip()}
    saved_options = [""] + list(saved_config_map.keys())
    current_loaded = str(st.session_state.get("merge_loaded_config_id") or "")
    if current_loaded not in saved_options:
        current_loaded = ""
        st.session_state["merge_loaded_config_id"] = ""

    select_sig = int(st.session_state.get("merge_saved_config_select_sig", 0) or 0)
    saved_cols = st.columns([3, 1, 2])
    selected_saved_id = saved_cols[0].selectbox(
        "已儲存合併設定",
        saved_options,
        index=saved_options.index(current_loaded),
        key=f"merge_saved_config_select_{select_sig}",
        format_func=lambda config_id: "新增 / 空白設定" if not config_id else merge_saved_config_label(saved_config_map.get(config_id, {})),
    )
    if selected_saved_id != current_loaded:
        if selected_saved_id:
            selected_config = saved_config_map.get(str(selected_saved_id), {})
            st.session_state["merge_loaded_config_id"] = str(selected_config.get("id") or "")
            st.session_state["merge_title"] = str(selected_config.get("title") or "")
            st.session_state["merge_queue"] = list(selected_config.get("items") or [])
        else:
            st.session_state["merge_loaded_config_id"] = ""
            st.session_state["merge_title"] = ""
            st.session_state["merge_queue"] = []
        st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
        st.rerun()

    if saved_cols[1].button("新增空白設定", key="merge_new_config"):
        st.session_state["merge_loaded_config_id"] = ""
        st.session_state["merge_title"] = ""
        st.session_state["merge_queue"] = []
        st.session_state["merge_saved_config_select_sig"] = select_sig + 1
        st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
        st.rerun()
    loaded_label = merge_saved_config_label(saved_config_map.get(current_loaded, {})) if current_loaded else "目前為尚未儲存的新設定"
    saved_cols[2].caption(loaded_label)

    selected_saved_config = saved_config_map.get(current_loaded, {})
    manage_cols = st.columns([3, 1, 1, 2])
    rename_key = f"merge_rename_title_{current_loaded or 'new'}_{select_sig}"
    rename_title = manage_cols[0].text_input(
        "修改已儲存設定標題",
        value=str(selected_saved_config.get("title") or ""),
        key=rename_key,
        disabled=not current_loaded,
    )
    if manage_cols[1].button("修改標題", key="merge_update_saved_title", disabled=not current_loaded):
        if not str(rename_title or "").strip():
            st.error("請先輸入新的設定標題。")
        else:
            updated_config = update_merge_saved_config_title(current_loaded, rename_title)
            if updated_config:
                st.session_state["merge_title"] = str(updated_config.get("title") or "")
                st.session_state["merge_saved_config_select_sig"] = select_sig + 1
                st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
                st.success("已更新設定標題。")
                st.rerun()
            else:
                st.error("找不到要修改的已儲存設定。")
    if manage_cols[2].button("刪除設定", key="merge_delete_saved_config", disabled=not current_loaded):
        if delete_merge_saved_config(current_loaded):
            st.session_state["merge_loaded_config_id"] = ""
            st.session_state["merge_title"] = ""
            st.session_state["merge_queue"] = []
            st.session_state["merge_saved_config_select_sig"] = select_sig + 1
            st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
            st.success("已刪除合併設定。")
            st.rerun()
        else:
            st.error("找不到要刪除的已儲存設定。")
    manage_cols[3].caption("刪除只會移除合併設定，不會刪除已產出的影片、Metadata、封面或 Log。")

    merge_tab, preview_tab, metadata_tab, publish_tab, log_tab = st.tabs(["合併影片", "預覽", "Metadata", "發布", "Log"])

    with merge_tab:
        title = st.text_input("標題", key="merge_title")
        source_profiles = merge_source_profiles()
        if not source_profiles:
            st.info("目前沒有可選擇的來源專案模式。")
            return

        profile_options = list(source_profiles.keys())
        selected_profile = st.selectbox(
            "專案模式",
            profile_options,
            key="merge_source_profile",
            format_func=lambda pid: source_profiles.get(pid, {}).get("name", pid),
        )
        source_rows = list_merge_episode_sources(selected_profile)
        source_options = [row for row in source_rows if row.get("video_ready")]
        if not source_options:
            st.warning("此專案模式尚未找到可合併的 final video。")
        else:
            selected_idx = st.selectbox(
                "集數",
                list(range(len(source_options))),
                key="merge_source_episode",
                format_func=lambda idx: merge_source_label(source_options[int(idx)]),
            )
            add_cols = st.columns([1, 4])
            if add_cols[0].button("新增", key="merge_add_source"):
                queue = list(st.session_state.get("merge_queue", []))
                item = dict(source_options[int(selected_idx)])
                if any(row.get("video_path") == item.get("video_path") for row in queue):
                    st.warning("此影片已在待合併清單中。")
                else:
                    queue = [normalize_merge_item(row, idx, default_include_outro=False) for idx, row in enumerate(queue, start=1)]
                    item["include_outro"] = True
                    item = normalize_merge_item(item, len(queue) + 1, default_include_outro=True)
                    item["duration"] = merge_item_duration_seconds(item, probe_missing=True)
                    queue.append(item)
                    st.session_state["merge_queue"] = queue
                    st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
                    st.success("已加入待合併清單。")
                    st.rerun()
            add_cols[1].caption(f"影片：{source_options[int(selected_idx)].get('video_path', '')}")

        queue = list(st.session_state.get("merge_queue", []))
        st.subheader("待合併清單")
        if not queue:
            st.caption("尚未加入任何影片。")
        else:
            table_rows = []
            queue = [normalize_merge_item(item, idx, default_include_outro=(idx == len(queue))) for idx, item in enumerate(queue, start=1)]
            for idx, item in enumerate(queue, start=1):
                item_duration = merge_item_duration_seconds(item, probe_missing=False)
                table_rows.append(
                    {
                        "順序": idx,
                        "移除": False,
                        "專案模式": item.get("profile_name", item.get("profile_id", "")),
                        "集數": f"Ep{int(item.get('ep') or 0):02d}",
                        "範圍": item.get("range", ""),
                        "時間長度": seconds_to_label(item_duration) if item_duration is not None else "未知",
                        "字幕": "有" if item.get("subtitle_path") else "無",
                        "影片路徑": item.get("video_path", ""),
                    }
                )
            queue_df = pd.DataFrame(table_rows)
            queue_df.insert(2, "合併片尾", [bool(item.get("include_outro")) for item in queue])
            queue_df.insert(3, "實際合併影片", [selected_merge_video_path(item) for item in queue])
            edited = st.data_editor(
                queue_df,
                use_container_width=True,
                hide_index=True,
                key=f"merge_queue_editor_{int(st.session_state.get('merge_editor_sig', 0) or 0)}",
                column_config={
                    "合併片尾": st.column_config.CheckboxColumn("合併片尾"),
                    "實際合併影片": st.column_config.TextColumn("實際合併影片", disabled=True, width="large"),
                    "順序": st.column_config.NumberColumn("順序", min_value=1, step=1),
                    "移除": st.column_config.CheckboxColumn("移除"),
                    "影片路徑": st.column_config.TextColumn("影片路徑", disabled=True, width="large"),
                },
                disabled=["專案模式", "集數", "範圍", "時間長度", "字幕", "影片路徑"],
            )
            preview_queue = queue_from_editor(queue, edited)
            total_duration, missing_duration_count = merge_queue_duration_seconds(preview_queue, probe_missing=False)
            duration_cols = st.columns([1.2, 1.2, 3.6])
            duration_cols[0].metric("合併後時間長度", seconds_to_label(total_duration))
            duration_cols[1].metric("待合併影片數", len(preview_queue))
            if missing_duration_count:
                duration_cols[2].warning(f"有 {missing_duration_count} 支影片無法讀取時間長度，總長度可能低估。")
            else:
                duration_cols[2].caption("時間長度會依目前表格順序與移除勾選即時更新。")
            action_cols = st.columns([1.2, 1.2, 1.2, 3])
            if action_cols[0].button("套用清單調整", key="merge_apply_queue"):
                st.session_state["merge_queue"] = preview_queue
                st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
                st.success("已更新待合併清單。")
                st.rerun()
            if action_cols[1].button("清空清單", key="merge_clear_queue"):
                st.session_state["merge_queue"] = []
                st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
                st.rerun()
            if action_cols[2].button("儲存", key="merge_save_config"):
                save_queue = preview_queue
                if not title.strip():
                    st.error("請先填寫標題。")
                elif not save_queue:
                    st.error("待合併清單至少需要 1 支影片才能儲存。")
                else:
                    saved = get_or_write_merge_saved_config(title.strip(), save_queue, st.session_state.get("merge_loaded_config_id", ""))
                    st.session_state["merge_loaded_config_id"] = str(saved.get("id") or "")
                    st.session_state["merge_queue"] = list(saved.get("items") or [])
                    st.session_state["merge_saved_config_select_sig"] = int(st.session_state.get("merge_saved_config_select_sig", 0) or 0) + 1
                    st.session_state["merge_editor_sig"] = int(st.session_state.get("merge_editor_sig", 0) or 0) + 1
                    st.success(f"已儲存設定：{saved.get('title')}")
                    st.rerun()
            action_cols[3].caption("可直接修改「順序」欄位，或勾選「移除」後套用；合併與排程會先儲存目前設定再執行。")

            st.divider()
            now_dt = datetime.now()
            schedule_cols = st.columns([1.2, 1.2, 1.2, 1.2, 2.4])
            with schedule_cols[0]:
                run_now = st.button("合併", type="primary", key="merge_run_now")
            with schedule_cols[1]:
                run_date = st.date_input("排程日期", min_value=now_dt.date(), key="merge_schedule_date")
            with schedule_cols[2]:
                run_time = st.time_input("排程時間", key="merge_schedule_time")
            with schedule_cols[3]:
                create_schedule = st.button("設定執行排程", key="merge_create_schedule")
            schedule_cols[4].caption("至少需要 2 支影片。執行期間會要求 Windows 保持喚醒；若使用排程，電腦仍不可關機。")

            if run_now or create_schedule:
                current_queue = preview_queue
                if len(current_queue) < 2:
                    st.error("待合併清單至少需要 2 支影片。")
                elif not title.strip():
                    st.error("請先填寫標題。")
                elif create_schedule and datetime.combine(run_date, run_time) <= now_dt:
                    st.error("排程時間不能設定為過去時間。")
                else:
                    scheduled_at = datetime.combine(run_date, run_time) if create_schedule else None
                    saved = get_or_write_merge_saved_config(title.strip(), current_queue, st.session_state.get("merge_loaded_config_id", ""))
                    st.session_state["merge_loaded_config_id"] = str(saved.get("id") or "")
                    st.session_state["merge_queue"] = list(saved.get("items") or [])
                    st.session_state["merge_saved_config_select_sig"] = int(st.session_state.get("merge_saved_config_select_sig", 0) or 0) + 1
                    job_id, payload = build_merge_job_payload(title.strip(), list(saved.get("items") or []), scheduled_at=scheduled_at, saved_config=saved)
                    write_merge_job(job_id, payload)
                    if run_now:
                        res = start_merge_job_now(job_id)
                        if res.get("ok"):
                            st.success(f"已儲存設定並開始合併：{job_id}")
                        else:
                            st.error(f"合併啟動失敗：{res.get('message')}")
                    else:
                        res = schedule_merge_job(job_id, scheduled_at)
                        if res.get("ok"):
                            st.success(f"已儲存設定並建立合併排程：{scheduled_at.isoformat(sep=' ', timespec='minutes')}")
                        else:
                            st.error(f"排程建立失敗：{res.get('message')}")

            st.subheader("合併 / 排程執行 Log")
            selected_merge_config_id = str(st.session_state.get("merge_loaded_config_id") or "")
            merge_log_jobs = list_merge_jobs_for_config(selected_merge_config_id)[:8] if selected_merge_config_id else []
            if not merge_log_jobs:
                st.caption("尚無合併或排程執行紀錄。")
            else:
                if st.button("刷新合併執行 Log", key="merge_run_log_refresh"):
                    st.rerun()
                for idx, job in enumerate(merge_log_jobs):
                    job_id = str(job.get("id") or "")
                    state = get_merge_job_state(job_id)
                    display_status = merge_job_display_status(job, state)
                    title_text = f"[{merge_job_status_label(display_status)}] {job.get('title', '')} | {job_id}"
                    with st.expander(title_text, expanded=bool(state.get("running")) or idx == 0):
                        log_cols = st.columns(5)
                        log_cols[0].metric("狀態", merge_job_status_label(display_status))
                        log_cols[1].metric("建立時間", str(job.get("created_at") or "").replace("T", " ") or "—")
                        log_cols[2].metric("排程時間", str(job.get("schedule_at") or "").replace("T", " ") or "—")
                        log_cols[3].metric("開始時間", str(job.get("started_at") or "").replace("T", " ") or "—")
                        log_cols[4].metric("完成時間", str(job.get("ended_at") or "").replace("T", " ") or "—")
                        st.caption(f"Last Message: {job.get('last_message', '')}")
                        if display_status == "missed":
                            st.warning("排程時間已過，但沒有開始時間。這通常代表 Windows 工作排程沒有成功啟動 launcher。新版已改用 ASCII job id/task name，請重新建立排程。")
                        log_path = merge_log_path(job_id)
                        if log_path.exists():
                            st.download_button(
                                "下載合併 Log",
                                data=log_path.read_bytes(),
                                file_name=log_path.name,
                                key=f"merge_run_log_download_{job_id}_{idx}",
                            )
                            st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:], language="text")
                        else:
                            st.caption("尚無 Log 檔。")

    with preview_tab:
        selected_preview_config_id = str(st.session_state.get("merge_loaded_config_id") or "")
        if selected_preview_config_id:
            st.caption(f"目前依上方已儲存合併設定顯示預覽：{merge_saved_config_label(saved_config_map.get(selected_preview_config_id, {}))}")
        else:
            st.caption("目前未選擇已儲存合併設定，顯示全部已完成合併結果。")

        preview_jobs = []
        if selected_preview_config_id:
            preview_jobs = [
                job for job in list_merge_jobs_for_config(selected_preview_config_id)
                if str(job.get("status") or "").lower() == "done" and merge_job_output_video_path(job).exists()
            ]
        if not preview_jobs:
            st.info("尚無可預覽的合併結果。請先完成一次合併。")
        else:
            selected_preview_idx = st.selectbox(
                "選擇合併結果",
                list(range(len(preview_jobs))),
                key="merge_preview_job",
                format_func=lambda idx: merge_preview_job_label(preview_jobs[int(idx)]),
            )
            preview_job = preview_jobs[int(selected_preview_idx)]
            video_path = merge_job_output_video_path(preview_job)
            subtitle_path = merge_job_output_subtitle_path(preview_job)
            subtitle_entries = parse_srt_entries(subtitle_path)
            video_duration = media_duration_seconds(str(video_path))
            subtitle_end = max([safe_float(row.get("end", 0)) for row in subtitle_entries] + [0.0])
            expected_duration, expected_missing = merge_queue_duration_seconds(list(preview_job.get("items") or []))

            metric_cols = st.columns(4)
            metric_cols[0].metric("影片長度", seconds_to_label(video_duration) if video_duration is not None else "未知")
            metric_cols[1].metric("字幕尾端", seconds_to_label(subtitle_end))
            metric_cols[2].metric("字幕數", len(subtitle_entries))
            if video_duration is not None:
                metric_cols[3].metric("影片 / 字幕差距", f"{video_duration - subtitle_end:+.2f}s")
            else:
                metric_cols[3].metric("預估清單長度", seconds_to_label(expected_duration))
            if expected_missing:
                st.warning(f"來源清單有 {expected_missing} 支影片無法讀取長度，來源區段結束時間可能不完整。")

            video_col, subtitle_col = st.columns([1.5, 1])
            with video_col:
                st.caption(f"合併影片：{video_path}")
                st.video(str(video_path))
            with subtitle_col:
                st.caption(f"合併字幕：{subtitle_path}")
                if subtitle_path.exists():
                    st.download_button(
                        "下載合併字幕",
                        data=subtitle_path.read_bytes(),
                        file_name=subtitle_path.name,
                        key=f"merge_preview_download_subtitle_{preview_job.get('id', '')}",
                    )
                    st.text_area(
                        "字幕檔內容",
                        value=read_text_file(subtitle_path, ""),
                        height=360,
                        key=f"merge_preview_srt_text_{preview_job.get('id', '')}",
                    )
                else:
                    st.warning("找不到合併字幕檔。")

            st.subheader("來源集數接續區段")
            segment_df = build_merge_segment_rows(preview_job)
            if segment_df.empty:
                st.caption("沒有來源集數資訊。")
            else:
                st.dataframe(segment_df, use_container_width=True, hide_index=True)

            st.subheader("字幕時間軸")
            if not subtitle_entries:
                st.caption("沒有可解析的字幕。")
            else:
                subtitle_df = pd.DataFrame(
                    [
                        {
                            "序號": row.get("index", ""),
                            "開始": seconds_to_label(row.get("start", 0)),
                            "結束": seconds_to_label(row.get("end", 0)),
                            "字幕": row.get("content", ""),
                        }
                        for row in subtitle_entries
                    ]
                )
                st.dataframe(
                    subtitle_df,
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    column_config={"字幕": st.column_config.TextColumn("字幕", width="large")},
                )

    with metadata_tab:
        selected_metadata_config_id = str(st.session_state.get("merge_loaded_config_id") or "")
        if selected_metadata_config_id:
            st.caption(f"目前依上方已儲存合併設定顯示 Metadata：{merge_saved_config_label(saved_config_map.get(selected_metadata_config_id, {}))}")
        else:
            st.caption("目前未選擇已儲存合併設定，顯示全部已完成合併結果。")

        metadata_jobs = []
        if selected_metadata_config_id:
            metadata_jobs = [
                job for job in list_merge_jobs_for_config(selected_metadata_config_id)
                if str(job.get("status") or "").lower() == "done" and merge_job_output_subtitle_path(job).exists()
            ]
        if not metadata_jobs:
            st.info("尚無可產生 Metadata 的合併結果。請先完成一次合併。")
        else:
            selected_meta_idx = st.selectbox(
                "選擇合併結果",
                list(range(len(metadata_jobs))),
                key="merge_metadata_job",
                format_func=lambda idx: merge_preview_job_label(metadata_jobs[int(idx)]),
            )
            metadata_job = metadata_jobs[int(selected_meta_idx)]
            metadata_path = merge_job_metadata_path(metadata_job)
            title_value = str(metadata_job.get("title") or "").strip()
            source_bundle_preview = merge_source_metadata_bundle(metadata_job)
            st.text_input("Title", value=title_value, disabled=True, key=f"merge_meta_title_{metadata_job.get('id', '')}")
            st.caption("Title 沿用合併設定，不會交給 LLM 重新產生。")
            source_metric_cols = st.columns(3)
            source_metric_cols[0].metric("來源集數", int(source_bundle_preview.get("total_episodes") or 0))
            source_metric_cols[1].metric("預估單字數", int(source_bundle_preview.get("total_words") or 0))
            source_metric_cols[2].metric("來源章節數", len(source_bundle_preview.get("merged_chapters") or []))
            with st.expander("將套用的來源章節時間軸", expanded=False):
                st.dataframe(
                    pd.DataFrame(source_bundle_preview.get("merged_chapters") or []),
                    use_container_width=True,
                    hide_index=True,
                )
            provider_choice = st.radio(
                "LLM Provider",
                ["auto", "openai", "gemini", "nvidia"],
                horizontal=True,
                key=f"merge_metadata_provider_{metadata_job.get('id', '')}",
                format_func=lambda value: {
                    "auto": "自動：Gemini 失敗改用 OpenAI",
                    "openai": "OpenAI",
                    "gemini": "Gemini",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )

            meta_action_cols = st.columns([1.2, 1.2, 3.6])
            if meta_action_cols[0].button("產生 Metadata", type="primary", key=f"merge_generate_meta_{metadata_job.get('id', '')}"):
                try:
                    with st.spinner("正在使用 LLM API 產生合併 Metadata..."):
                        meta = generate_merge_metadata(metadata_job, provider=provider_choice)
                        saved_path = write_merge_metadata(metadata_job, meta)
                    st.success(f"已產生 Metadata：{saved_path}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Metadata 產生失敗：{exc}")
            if metadata_path.exists():
                meta_action_cols[1].download_button(
                    "下載 Metadata",
                    data=metadata_path.read_bytes(),
                    file_name=metadata_path.name,
                    key=f"merge_download_meta_{metadata_job.get('id', '')}",
                )
            meta_action_cols[2].caption(f"輸出檔案：{metadata_path}")

            current_meta = read_json_file(metadata_path) or {}
            desc_default = str(current_meta.get("description") or "")
            tags_default = ", ".join([str(tag) for tag in (current_meta.get("tags") or [])])
            summary_default = str(current_meta.get("summary") or "")
            chapters_default = json.dumps(current_meta.get("chapters") or [], ensure_ascii=False, indent=2)

            with st.form(f"merge_metadata_form_{metadata_job.get('id', '')}_{int(metadata_path.stat().st_mtime) if metadata_path.exists() else 0}"):
                summary_val = st.text_area("Summary", value=summary_default, height=110)
                desc_val = st.text_area("Description", value=desc_default, height=300)
                tags_val = st.text_input("Tags", value=tags_default)
                chapters_val = st.text_area("Chapters JSON", value=chapters_default, height=180)
                save_meta = st.form_submit_button("儲存 Metadata")
            if save_meta:
                try:
                    chapters_data = json.loads(chapters_val or "[]")
                    tags_data = [tag.strip() for tag in tags_val.split(",") if tag.strip()]
                    meta = {
                        **current_meta,
                        "title": title_value,
                        "summary": summary_val.strip(),
                        "description": desc_val.strip(),
                        "tags": tags_data,
                        "chapters": chapters_data,
                        "default_language": current_meta.get("default_language") or "zh-TW",
                        "default_audio_language": current_meta.get("default_audio_language") or "zh-TW",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "source_job_id": metadata_job.get("id", ""),
                        "source_saved_config_id": metadata_job.get("saved_config_id", ""),
                    }
                    saved_path = write_merge_metadata(metadata_job, meta)
                    st.success(f"已儲存 Metadata：{saved_path}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"儲存 Metadata 失敗：{exc}")

            if current_meta:
                with st.expander("Metadata JSON", expanded=False):
                    st.json(current_meta)

    with publish_tab:
        selected_publish_config_id = str(st.session_state.get("merge_loaded_config_id") or "")
        if selected_publish_config_id:
            st.caption(f"目前依上方已儲存合併設定顯示發布項目：{merge_saved_config_label(saved_config_map.get(selected_publish_config_id, {}))}")
        else:
            st.caption("目前未選擇已儲存合併設定，顯示全部已完成合併結果。")

        publish_jobs = []
        if selected_publish_config_id:
            publish_jobs = [
                job for job in list_merge_jobs_for_config(selected_publish_config_id)
                if str(job.get("status") or "").lower() == "done" and merge_job_output_video_path(job).exists()
            ]
        if not publish_jobs:
            st.info("尚無可發布的合併結果。請先完成合併。")
        else:
            selected_publish_idx = st.selectbox(
                "選擇合併結果",
                list(range(len(publish_jobs))),
                key="merge_publish_job",
                format_func=lambda idx: merge_preview_job_label(publish_jobs[int(idx)]),
            )
            publish_job = publish_jobs[int(selected_publish_idx)]
            publish_job_id = str(publish_job.get("id") or "")
            video_path = merge_job_output_video_path(publish_job)
            subtitle_path = merge_job_output_subtitle_path(publish_job)
            metadata_path = merge_job_metadata_path(publish_job)
            cover_path = merge_job_cover_path(publish_job)
            upload_record_path = merge_job_upload_record_path(publish_job)
            upload_record = read_json_file(upload_record_path) or {}
            upload_state = get_merge_upload_state(publish_job_id)

            ready_cols = st.columns(5)
            ready_cols[0].metric("影片", "就緒" if video_path.exists() else "缺少")
            ready_cols[1].metric("字幕", "就緒" if subtitle_path.exists() else "缺少")
            ready_cols[2].metric("Metadata", "就緒" if metadata_path.exists() else "缺少")
            ready_cols[3].metric("封面", "就緒" if cover_path else "未設定")
            ready_cols[4].metric("上傳狀態", "執行中" if upload_state.get("running") else upload_record.get("status", "尚無"))
            st.caption(f"影片：{video_path}")
            st.caption(f"字幕：{subtitle_path}")
            st.caption(f"Metadata：{metadata_path}")
            st.caption(f"封面：{cover_path or '尚未上傳'}")

            cover_cols = st.columns([1.2, 2.4, 2.4])
            with cover_cols[0]:
                uploaded_cover = st.file_uploader(
                    "上傳封面",
                    type=["png", "jpg", "jpeg"],
                    key=f"merge_cover_upload_{publish_job_id}",
                )
                if uploaded_cover is not None:
                    upload_sig = f"{uploaded_cover.name}:{uploaded_cover.size}"
                    processed_key = f"merge_cover_upload_processed_{publish_job_id}"
                    if st.session_state.get(processed_key) != upload_sig:
                        try:
                            saved_cover = save_merge_cover_upload(publish_job, uploaded_cover)
                            st.session_state[processed_key] = upload_sig
                            st.success(f"已儲存封面：{saved_cover.name}")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"封面儲存失敗：{exc}")
            with cover_cols[1]:
                if cover_path and cover_path.exists():
                    st.image(str(cover_path), caption="目前封面", use_container_width=True)
                else:
                    st.caption("尚未上傳封面。")
            with cover_cols[2]:
                st.caption("封面會在上傳影片成功後設定到 YouTube。YouTube API 接受的縮圖大小上限為 2MB；若超過，系統會嘗試壓縮後再上傳。")

            if upload_record.get("video_id"):
                st.markdown(f"[開啟 YouTube 影片](https://www.youtube.com/watch?v={upload_record['video_id']})")

            if not metadata_path.exists():
                st.warning("發布前需要先在 Metadata 分頁產生並確認 youtube_meta.json。")

            token_sig, client_sig = youtube_auth_file_signatures()
            playlist_items, playlist_error = list_youtube_playlists_cached(token_sig, client_sig)
            playlist_options = [{"id": "", "title": "(不加入播放清單)", "label": "(不加入播放清單)"}] + playlist_items
            if playlist_error:
                st.error(f"YouTube 播放清單讀取失敗：{playlist_error}")
                if "invalid_grant" in playlist_error.lower() or "重新授權" in playlist_error:
                    if st.button("重新授權 YouTube", key=f"merge_reauth_youtube_{publish_job_id}"):
                        try:
                            reauthorize_youtube_for_app()
                            list_youtube_playlists_cached.clear()
                            st.success("YouTube 已重新授權。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"YouTube 重新授權失敗：{exc}")

            publish_defaults = {
                "privacy": str(upload_record.get("privacy") or "private"),
                "schedule_on": bool(upload_record.get("publish_at")),
                "publish_at": str(upload_record.get("publish_at") or ""),
                "playlist_id": str(upload_record.get("playlist_id") or ""),
            }
            privacy_key = f"merge_publish_privacy_{publish_job_id}"
            schedule_key = f"merge_publish_schedule_on_{publish_job_id}"
            date_key = f"merge_publish_date_{publish_job_id}"
            time_key = f"merge_publish_time_{publish_job_id}"
            playlist_key = f"merge_publish_playlist_{publish_job_id}"

            if privacy_key not in st.session_state:
                st.session_state[privacy_key] = publish_defaults["privacy"] if publish_defaults["privacy"] in {"private", "unlisted", "public"} else "private"
            if schedule_key not in st.session_state:
                st.session_state[schedule_key] = publish_defaults["schedule_on"]
            if playlist_key not in st.session_state:
                st.session_state[playlist_key] = resolve_default_playlist_id(playlist_options, publish_defaults["playlist_id"])

            privacy_val = st.selectbox("隱私設定", ["private", "unlisted", "public"], key=privacy_key)
            schedule_on = st.checkbox("排程發布", key=schedule_key)
            schedule_cols = st.columns(2)
            with schedule_cols[0]:
                publish_date = st.date_input("發布日期", key=date_key, disabled=not schedule_on)
            with schedule_cols[1]:
                publish_time = st.time_input("發布時間", key=time_key, disabled=not schedule_on)

            selected_playlist_id = st.selectbox(
                "播放清單",
                options=[item["id"] for item in playlist_options],
                format_func=lambda pid: next((item["label"] for item in playlist_options if item["id"] == pid), pid or "(不加入播放清單)"),
                key=playlist_key,
            )

            publish_at_text = ""
            if schedule_on:
                publish_at_text = datetime.combine(publish_date, publish_time).strftime("%Y-%m-%d %H:%M")
                st.caption(f"YouTube 排程發布時間：{publish_at_text} (Asia/Taipei)。排程發布會以 private 上傳並設定 publishAt。")
            elif privacy_val == "public":
                st.warning("選擇 public 會在上傳完成後立即公開。")

            upload_disabled = (
                upload_state.get("running")
                or not video_path.exists()
                or not metadata_path.exists()
            )
            action_cols = st.columns([1.2, 1.2, 3.6])
            if action_cols[0].button("上傳到 YouTube", type="primary", key=f"merge_upload_youtube_{publish_job_id}", disabled=upload_disabled):
                res = start_merge_upload_job(
                    publish_job,
                    privacy=privacy_val,
                    publish_at=publish_at_text,
                    playlist_id=selected_playlist_id.strip(),
                )
                if res.get("ok"):
                    st.success(f"已啟動合併影片上傳。log: {res.get('log')}")
                    st.rerun()
                else:
                    st.error(f"上傳啟動失敗：{res.get('message')}")
            if action_cols[1].button("重新整理發布狀態", key=f"merge_publish_refresh_{publish_job_id}"):
                st.rerun()
            action_cols[2].caption("上傳會在背景執行。若 YouTube token 失效，背景程序可能需要重新授權；請依 Log 提示處理。")

            if upload_record:
                with st.expander("上傳紀錄", expanded=True):
                    st.json(upload_record)

            upload_log = merge_upload_log_path(publish_job_id)
            if upload_log.exists():
                with st.expander("發布 Log", expanded=bool(upload_state.get("running"))):
                    st.download_button(
                        "下載發布 Log",
                        data=upload_log.read_bytes(),
                        file_name=upload_log.name,
                        key=f"merge_download_publish_log_{publish_job_id}",
                    )
                    st.code(upload_log.read_text(encoding="utf-8", errors="ignore")[-8000:], language="text")
            elif upload_state.get("running"):
                st.info(f"上傳背景執行中，PID {upload_state.get('pid')}，Log 尚未建立。")

    with log_tab:
        selected_log_config_id = str(st.session_state.get("merge_loaded_config_id") or "")
        if selected_log_config_id:
            st.caption(f"目前依上方已儲存合併設定篩選 Log：{merge_saved_config_label(saved_config_map.get(selected_log_config_id, {}))}")
        else:
            st.caption("目前未選擇已儲存合併設定，顯示全部 Log。")
        auto_refresh = st.toggle("自動刷新合併 Log", value=True, key="merge_log_auto_refresh")

        @st.fragment(run_every=5 if auto_refresh else None)
        def render_merge_logs():
            if not selected_log_config_id:
                st.caption("請先選擇已儲存合併設定。")
                return
            jobs = list_merge_jobs_for_config(selected_log_config_id)
            if not jobs:
                st.caption("尚無合併 Log。")
                return
            st.button("只刷新合併 Log", key="merge_log_refresh")
            for idx, job in enumerate(jobs[:12]):
                job_id = str(job.get("id", ""))
                state = get_merge_job_state(job_id)
                latest_job = state.get("job") or job
                status = "running" if state.get("running") else str(latest_job.get("status", "queued"))
                with st.expander(f"[{get_schedule_status_label(status)}] {latest_job.get('title', '')} | {job_id}", expanded=bool(state.get("running"))):
                    cols = st.columns(4)
                    cols[0].metric("狀態", get_schedule_status_label(status))
                    cols[1].metric("PID", state.get("pid") or "—")
                    cols[2].metric("建立時間", str(latest_job.get("created_at", "")).replace("T", " ") or "—")
                    cols[3].metric("排程時間", str(latest_job.get("schedule_at", "")).replace("T", " ") or "—")
                    if latest_job.get("saved_config_id"):
                        st.caption(f"儲存設定：{latest_job.get('saved_config_id')}")
                    st.caption(f"輸出資料夾：{latest_job.get('output_dir', '')}")
                    if latest_job.get("output_video"):
                        st.caption(f"合併影片：{latest_job.get('output_video')}")
                    if latest_job.get("output_subtitle"):
                        st.caption(f"合併字幕：{latest_job.get('output_subtitle')}")
                    log_path = merge_log_path(job_id)
                    if log_path.exists():
                        st.download_button("下載 Log", data=log_path.read_bytes(), file_name=log_path.name, key=f"merge_dl_log_{job_id}_{idx}")
                        st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:], language="text")
                    else:
                        st.caption("尚無 Log。")

        render_merge_logs()
