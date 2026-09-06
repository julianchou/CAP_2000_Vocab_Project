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
import streamlit.components.v1 as components

ROOT: Path | None = None


def _not_configured(*args, **kwargs):
    raise RuntimeError("Short generator section is not configured. Call configure_short_generator first.")


media_duration_seconds = _not_configured
seconds_to_label = _not_configured
youtube_auth_file_signatures = _not_configured
list_youtube_playlists_cached = _not_configured
reauthorize_youtube_for_app = _not_configured
resolve_default_playlist_id = _not_configured
asset_tag_badge = _not_configured
asset_tag_for_path = _not_configured
set_asset_tag_for_path = _not_configured
collect_asset_tag_options = _not_configured
filter_asset_paths_by_tag = _not_configured
ASSET_TAG_FILTER_ALL = "??"


def configure_short_generator(root: Path, deps: dict) -> None:
    global ROOT
    global media_duration_seconds, seconds_to_label
    global youtube_auth_file_signatures, list_youtube_playlists_cached, reauthorize_youtube_for_app
    global resolve_default_playlist_id
    global asset_tag_badge, asset_tag_for_path, set_asset_tag_for_path
    global collect_asset_tag_options, filter_asset_paths_by_tag, ASSET_TAG_FILTER_ALL

    ROOT = root
    media_duration_seconds = deps["media_duration_seconds"]
    seconds_to_label = deps["seconds_to_label"]
    youtube_auth_file_signatures = deps["youtube_auth_file_signatures"]
    list_youtube_playlists_cached = deps["list_youtube_playlists_cached"]
    reauthorize_youtube_for_app = deps["reauthorize_youtube_for_app"]
    resolve_default_playlist_id = deps["resolve_default_playlist_id"]
    asset_tag_badge = deps["asset_tag_badge"]
    asset_tag_for_path = deps["asset_tag_for_path"]
    set_asset_tag_for_path = deps["set_asset_tag_for_path"]
    collect_asset_tag_options = deps["collect_asset_tag_options"]
    filter_asset_paths_by_tag = deps["filter_asset_paths_by_tag"]
    ASSET_TAG_FILTER_ALL = deps["ASSET_TAG_FILTER_ALL"]


def _root() -> Path:
    if ROOT is None:
        raise RuntimeError("Short generator section is not configured. Call configure_short_generator first.")
    return ROOT


SHORT_BLANK_OPTION = "新增/空白設定"
SHORT_STATUS_OPTIONS = ["製作中", "已上傳"]
SHORT_EPISODE_DIRS = [
    "00_logs",
    "01_audio",
    "02_subtitles",
    "03_storyboards",
    "04_images",
    "05_output",
]

def short_workspace_root() -> Path:
    path = _root() / "workspaces" / "short_generator"
    path.mkdir(parents=True, exist_ok=True)
    return path

def short_programs_path() -> Path:
    return short_workspace_root() / "programs.json"

def short_slugify(name: str) -> str:
    text = re.sub(r"\s+", "_", str(name or "").strip())
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = text.strip("._")
    return text or "short_program"

def short_read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def short_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def short_load_programs() -> list[Dict]:
    payload = short_read_json(short_programs_path(), {"programs": []})
    programs = payload.get("programs", []) if isinstance(payload, dict) else []
    return [p for p in programs if isinstance(p, dict)]

def short_save_programs(programs: list[Dict]) -> None:
    short_write_json(short_programs_path(), {"programs": programs})

def short_program_dir(program: Dict) -> Path:
    return short_workspace_root() / short_slugify(program.get("slug") or program.get("name"))

def short_next_program_id(programs: list[Dict]) -> str:
    max_no = 0
    for program in programs:
        m = re.match(r"short_program_(\d+)$", str(program.get("id", "")))
        if m:
            max_no = max(max_no, int(m.group(1)))
    return f"short_program_{max_no + 1}"

def short_get_program(program_id: str) -> Dict | None:
    for program in short_load_programs():
        if program.get("id") == program_id:
            return program
    return None

def short_episode_dir(program: Dict, episode_no: int) -> Path:
    return short_program_dir(program) / f"Ep{int(episode_no):02d}"

def short_episode_json_path(ep_dir: Path) -> Path:
    return ep_dir / "short.json"

def short_metadata_path(ep_dir: Path) -> Path:
    return ep_dir / "05_output" / "short_metadata.json"

def short_storyboard_path(ep_dir: Path) -> Path:
    return ep_dir / "03_storyboards" / "storyboard.json"
def short_storyboard_meta_path(ep_dir: Path) -> Path:
    return ep_dir / "03_storyboards" / "storyboard_meta.json"

def short_preferred_subtitle_path(ep_dir: Path) -> Path | None:
    for path in [
        ep_dir / "02_subtitles" / "final.srt",
        ep_dir / "02_subtitles" / "reviewed.srt",
        ep_dir / "02_subtitles" / "whisper.srt",
    ]:
        if short_output_completed(path):
            return path
    return None

def short_storyboard_matches_current_subtitles(ep_dir: Path, storyboard_path: Path) -> tuple[bool, str]:
    subtitle_path = short_preferred_subtitle_path(ep_dir)
    if subtitle_path is None:
        return False, "No usable subtitle file found. Complete subtitles before generating storyboard images."

    meta = short_read_json(short_storyboard_meta_path(ep_dir), {})
    input_srt = str(meta.get("input_srt") or "").strip() if isinstance(meta, dict) else ""
    if input_srt:
        try:
            if Path(input_srt).resolve() == subtitle_path.resolve():
                if storyboard_path.exists() and subtitle_path.stat().st_mtime > storyboard_path.stat().st_mtime:
                    return False, f"{subtitle_path.name} is newer than storyboard.json. Re-run step 6 before step 7."
                return True, ""
        except Exception:
            if str(Path(input_srt)) == str(subtitle_path):
                if storyboard_path.exists() and subtitle_path.stat().st_mtime > storyboard_path.stat().st_mtime:
                    return False, f"{subtitle_path.name} is newer than storyboard.json. Re-run step 6 before step 7."
                return True, ""
        return (
            False,
            f"storyboard.json was generated from {Path(input_srt).name}, but the current subtitle source is {subtitle_path.name}. Re-run step 6 first.",
        )

    if subtitle_path.name == "final.srt" and storyboard_path.exists() and subtitle_path.stat().st_mtime > storyboard_path.stat().st_mtime:
        return False, "final.srt is newer than storyboard.json. Re-run step 6 before step 7."

    return True, ""

def short_compose_list_path(ep_dir: Path) -> Path:
    return ep_dir / "05_output" / "compose_list.json"

def short_preview_video_path(ep_dir: Path) -> Path:
    return ep_dir / "05_output" / "preview_short.mp4"

def short_final_video_path(ep_dir: Path) -> Path:
    return ep_dir / "05_output" / "final_short.mp4"

def short_upload_record_path(ep_dir: Path) -> Path:
    return ep_dir / "05_output" / "upload_record.json"

def short_publish_settings_path(ep_dir: Path) -> Path:
    return ep_dir / "05_output" / "publish_settings.json"

def short_publish_cover_path(ep_dir: Path) -> Path | None:
    for path in [
        ep_dir / "05_output" / "cover.png",
        ep_dir / "05_output" / "cover.jpg",
        ep_dir / "05_output" / "cover.jpeg",
        ep_dir / "04_images" / "storyboard" / "scene_001.png",
    ]:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None

def short_step_log_path(ep_dir: Path, step_no: str) -> Path:
    return ep_dir / "00_logs" / f"short_step_{str(step_no).replace('.', '_')}.log"

def short_step_proc_path(ep_dir: Path, step_no: str) -> Path:
    return ep_dir / "00_logs" / f"short_step_{str(step_no).replace('.', '_')}.proc.json"

def short_append_step_log(ep_dir: Path, step_no: str, message: str, level: str = "INFO") -> None:
    log_path = short_step_log_path(ep_dir, step_no)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] [{level}] {message}\n")

def short_log_run_marker(ep_dir: Path, step_no: str, label: str, status: str, level: str = "INFO") -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    short_append_step_log(ep_dir, step_no, f"===== RUN {status} {ts}：{label} =====", level=level)

def short_step_log_text(ep_dir: Path, step_no: str) -> str:
    log_path = short_step_log_path(ep_dir, step_no)
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="ignore")

def short_log_has_error(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["[error]", "traceback", "exception", "failed", "失敗", "錯誤"])

def short_is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        return f'"{pid}"' in result.stdout or f",{pid}," in result.stdout
    except Exception:
        return False

def short_step_state(ep_dir: Path, step_no: str) -> Dict:
    proc_path = short_step_proc_path(ep_dir, step_no)
    state = {"running": False, "pid": None, "started_at": "", "ended_at": ""}
    if not proc_path.exists():
        return state
    try:
        payload = json.loads(proc_path.read_text(encoding="utf-8"))
    except Exception:
        return state
    pid = int(payload.get("pid", 0) or 0)
    running = short_is_pid_running(pid)
    state.update({
        "running": running,
        "pid": pid,
        "started_at": payload.get("started_at", ""),
        "ended_at": payload.get("ended_at", ""),
    })
    if not running and not payload.get("ended_at"):
        payload["ended_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            proc_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            short_log_run_marker(ep_dir, step_no, str(payload.get("label") or f"步驟{step_no}"), "END")
            short_append_step_log(ep_dir, step_no, f"背景程序已結束，結束時間：{payload['ended_at']}。請檢查輸出檔與上方 stdout/stderr。")
        except Exception:
            pass
        state["ended_at"] = payload["ended_at"]
    return state

def short_rerun_when_step_finishes(ep_dir: Path, step_no: str, key_suffix: str = "") -> Dict:
    state = short_step_state(ep_dir, step_no)
    state_key = f"short_step_was_running_{ep_dir.name}_{str(step_no).replace('.', '_')}_{key_suffix}"
    was_running = bool(st.session_state.get(state_key, False))
    is_running = bool(state.get("running"))
    st.session_state[state_key] = is_running
    if was_running and not is_running:
        st.rerun(scope="app")
    return state

def short_output_completed(path: Path) -> bool:
    if path.is_dir():
        return path.exists() and any(path.iterdir())
    if not path.exists():
        return False
    if path.suffix.lower() in {".txt", ".srt", ".json", ".csv", ".md"}:
        return path.stat().st_size > 0
    return True


@st.cache_data(show_spinner=False)
def short_image_thumbnail_bytes(path_text: str, mtime_ns: int, max_width: int = 220) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageOps

    path = Path(path_text)
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return b""
    with Image.open(path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")
        width, height = image.size
        if width > max_width:
            new_height = max(1, int(height * (max_width / width)))
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=58, optimize=True)
        return buffer.getvalue()


def render_short_copy_error_button(text: str, key: str) -> None:
    payload = json.dumps(text)
    components.html(
        f"""
        <button id="{key}" style="padding:0.35rem 0.7rem;border:1px solid #bbb;border-radius:6px;background:white;cursor:pointer;">
          複製錯誤 Log
        </button>
        <span id="{key}_msg" style="margin-left:0.5rem;color:#666;font-size:0.85rem;"></span>
        <script>
        const btn = document.getElementById("{key}");
        const msg = document.getElementById("{key}_msg");
        btn.onclick = async () => {{
          await navigator.clipboard.writeText({payload});
          msg.textContent = "已複製";
          setTimeout(() => msg.textContent = "", 1600);
        }};
        </script>
        """,
        height=42,
    )

def short_render_step_log_panel(
    ep_dir: Path,
    step_no: str,
    label: str,
    expanded: bool = False,
    auto_expand_errors: bool = True,
    key_suffix: str = "",
) -> None:
    log_path = short_step_log_path(ep_dir, step_no)
    text = short_step_log_text(ep_dir, step_no)
    state = short_step_state(ep_dir, step_no)
    has_error = short_log_has_error(text)
    title = f"步驟{step_no} Log - {label}"
    if state.get("running"):
        title += f"（執行中 PID {state.get('pid')}）"
    if has_error:
        title += "（偵測到錯誤）"
    state_key = f"short_log_expanded_{ep_dir.name}_{str(step_no).replace('.', '_')}_{key_suffix}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    is_expanded = st.checkbox(
        title,
        key=state_key,
        help="勾選展開 Log；取消勾選收合。此狀態會在自動刷新後保留。",
    )
    if is_expanded:
        st.caption(str(log_path))
        if state.get("started_at"):
            st.caption(f"開始時間：{state.get('started_at')}")
        if state.get("ended_at"):
            st.caption(f"結束時間：{state.get('ended_at')}")
        if state.get("running"):
            st.info(f"背景執行中，PID {state.get('pid')}。自動刷新開啟時會持續更新這裡的輸出。")
        if text:
            st.code(text[-8000:], language="text")
            if has_error:
                st.text_area(
                    "錯誤內容，可手動複製",
                    value=text,
                    height=160,
                    key=f"short_error_copy_text_{ep_dir.name}_{step_no}_{key_suffix}",
                )
                render_short_copy_error_button(text, f"short_copy_error_{ep_dir.name}_{step_no}_{key_suffix}")
        else:
            st.caption("尚無 log。")

def short_run_logged_step(ep_dir: Path, step_no: str, label: str, action) -> bool:
    import traceback

    short_log_run_marker(ep_dir, step_no, label, "START")
    short_append_step_log(ep_dir, step_no, f"開始：{label}")
    try:
        action()
    except Exception as exc:
        short_append_step_log(ep_dir, step_no, f"失敗：{exc}", level="ERROR")
        short_append_step_log(ep_dir, step_no, traceback.format_exc(), level="ERROR")
        short_log_run_marker(ep_dir, step_no, label, "END", level="ERROR")
        st.error(f"步驟{step_no}執行失敗，請查看下方 Log。")
        return False
    short_append_step_log(ep_dir, step_no, f"完成：{label}")
    short_log_run_marker(ep_dir, step_no, label, "END")
    return True

def short_run_logged_script(ep_dir: Path, step_no: str, label: str, cmd: list[str]) -> bool:
    short_log_run_marker(ep_dir, step_no, label, "START")
    short_append_step_log(ep_dir, step_no, f"開始：{label}")
    short_append_step_log(ep_dir, step_no, "執行命令：" + " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CAP_WORKSPACE_ROOT"] = str(short_workspace_root())
    env["CAP_PROFILE_ID"] = "short_generator"
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_root()),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        short_append_step_log(ep_dir, step_no, f"啟動失敗：{exc}", level="ERROR")
        short_log_run_marker(ep_dir, step_no, label, "END", level="ERROR")
        st.error(f"步驟{step_no}啟動失敗，請查看下方 Log。")
        return False
    if result.stdout:
        short_append_step_log(ep_dir, step_no, "stdout:\n" + result.stdout.rstrip())
    if result.stderr:
        short_append_step_log(ep_dir, step_no, "stderr:\n" + result.stderr.rstrip(), level="ERROR" if result.returncode else "INFO")
    short_append_step_log(ep_dir, step_no, f"return_code={result.returncode}")
    if result.returncode != 0:
        short_append_step_log(ep_dir, step_no, f"失敗：{label}", level="ERROR")
        short_log_run_marker(ep_dir, step_no, label, "END", level="ERROR")
        st.error(f"步驟{step_no}執行失敗，請查看下方 Log。")
        return False
    short_append_step_log(ep_dir, step_no, f"完成：{label}")
    short_log_run_marker(ep_dir, step_no, label, "END")
    return True

def short_start_logged_script(ep_dir: Path, step_no: str, label: str, cmd: list[str]) -> bool:
    state = short_step_state(ep_dir, step_no)
    if state.get("running"):
        st.warning(f"步驟{step_no}仍在執行中，PID {state.get('pid')}。")
        return False

    log_path = short_step_log_path(ep_dir, step_no)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    short_log_run_marker(ep_dir, step_no, label, "START")
    short_append_step_log(ep_dir, step_no, f"開始背景執行：{label}")
    short_append_step_log(ep_dir, step_no, "執行命令：" + " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CAP_WORKSPACE_ROOT"] = str(short_workspace_root())
    env["CAP_PROFILE_ID"] = "short_generator"
    log_fh = None
    try:
        log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(_root()),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    except Exception as exc:
        short_append_step_log(ep_dir, step_no, f"啟動失敗：{exc}", level="ERROR")
        short_log_run_marker(ep_dir, step_no, label, "END", level="ERROR")
        st.error(f"步驟{step_no}啟動失敗，請查看下方 Log。")
        return False
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass

    short_step_proc_path(ep_dir, step_no).write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "label": label,
                "cmd": cmd,
                "ended_at": "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    st.session_state[
        f"short_step_was_running_{ep_dir.name}_{str(step_no).replace('.', '_')}_control"
    ] = True
    short_append_step_log(ep_dir, step_no, f"背景程序已啟動，PID {proc.pid}。")
    return True

def short_audio_files(ep_dir: Path) -> list[Path]:
    audio_dir = ep_dir / "01_audio"
    if not audio_dir.exists():
        return []
    return sorted([p for p in audio_dir.iterdir() if p.is_file()])

def short_list_episodes(program: Dict) -> list[Dict]:
    program_dir = short_program_dir(program)
    if not program_dir.exists():
        return []
    rows = []
    for ep_dir in sorted([p for p in program_dir.iterdir() if p.is_dir() and re.match(r"^Ep\d+$", p.name)]):
        try:
            episode_no = int(ep_dir.name.replace("Ep", ""))
        except ValueError:
            continue
        data = short_read_json(short_episode_json_path(ep_dir), {})
        meta = short_read_json(short_metadata_path(ep_dir), {})
        upload_record = short_read_json(short_upload_record_path(ep_dir), {})
        uploaded_marker = ep_dir / "05_output" / "uploaded.txt"
        upload_success = str(upload_record.get("status", "")).strip().lower() == "success" or uploaded_marker.exists()
        status = "已上傳" if upload_success else (data.get("status") or "製作中")
        rows.append({
            "episode_no": episode_no,
            "title": data.get("title") or meta.get("title") or f"Ep{episode_no:02d}",
            "status": status,
            "path": ep_dir,
            "updated_at": upload_record.get("ended_at") or data.get("updated_at", ""),
        })
    return rows

def short_next_episode_no(program: Dict) -> int:
    episodes = short_list_episodes(program)
    if not episodes:
        return 1
    return max(int(e["episode_no"]) for e in episodes) + 1

def short_ensure_episode(program: Dict, episode_no: int, title: str = "") -> Path:
    ep_dir = short_episode_dir(program, episode_no)
    for rel in SHORT_EPISODE_DIRS:
        (ep_dir / rel).mkdir(parents=True, exist_ok=True)
    data = short_read_json(short_episode_json_path(ep_dir), {})
    data.update({
        "program_id": program.get("id"),
        "program_name": program.get("name", ""),
        "episode_no": int(episode_no),
        "title": title or data.get("title") or f"{program.get('name', 'Short')} Ep{int(episode_no):02d}",
        "status": data.get("status") or "製作中",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    short_write_json(short_episode_json_path(ep_dir), data)
    return ep_dir

def short_save_episode_data(ep_dir: Path, updates: Dict) -> Dict:
    data = short_read_json(short_episode_json_path(ep_dir), {})
    data.update(updates)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    short_write_json(short_episode_json_path(ep_dir), data)
    return data

def render_short_management():
    st.header("Short 管理")
    programs = short_load_programs()
    notice = st.session_state.pop("short_program_notice", None)
    if notice:
        st.success(notice)

    with st.expander("新增 Short 節目", expanded=not programs):
        with st.form("short_create_program_form"):
            name = st.text_input("Short節目名稱", key="short_new_program_name")
            submitted = st.form_submit_button("新增節目")
        if submitted:
            clean_name = name.strip()
            if not clean_name:
                st.error("請輸入 Short 節目名稱。")
            elif any(p.get("name") == clean_name for p in programs):
                st.error("已有相同名稱的 Short 節目。")
            else:
                now = datetime.now().isoformat(timespec="seconds")
                slug_base = short_slugify(clean_name)
                slug = slug_base
                used = {short_slugify(p.get("slug") or p.get("name")) for p in programs}
                suffix = 2
                while slug in used:
                    slug = f"{slug_base}_{suffix}"
                    suffix += 1
                programs.append({
                    "id": short_next_program_id(programs),
                    "name": clean_name,
                    "slug": slug,
                    "created_at": now,
                    "updated_at": now,
                })
                short_save_programs(programs)
                short_program_dir(programs[-1]).mkdir(parents=True, exist_ok=True)
                st.session_state["short_program_notice"] = "Short 節目已新增。"
                st.rerun()

    if not programs:
        st.info("尚未建立 Short 節目。請先新增節目名稱。")
        return

    rows = []
    for program in programs:
        episodes = short_list_episodes(program)
        rows.append({
            "節目名稱": program.get("name", ""),
            "已生成集數": len(episodes),
            "製作中": sum(1 for ep in episodes if ep.get("status") == "製作中"),
            "已上傳": sum(1 for ep in episodes if ep.get("status") == "已上傳"),
            "最後更新": program.get("updated_at", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected_id = st.selectbox(
        "選擇要管理的 Short 節目",
        [p.get("id") for p in programs],
        format_func=lambda pid: next((p.get("name", pid) for p in programs if p.get("id") == pid), pid),
        key="short_manage_program",
    )
    selected = short_get_program(selected_id)
    if not selected:
        return

    col_rename, col_delete = st.columns([2, 1])
    with col_rename:
        with st.form(f"short_rename_{selected_id}"):
            new_name = st.text_input("修改short節目名稱", value=selected.get("name", ""))
            rename_submit = st.form_submit_button("儲存節目名稱")
        if rename_submit:
            clean_name = new_name.strip()
            if not clean_name:
                st.error("節目名稱不可空白。")
            elif any(p.get("id") != selected_id and p.get("name") == clean_name for p in programs):
                st.error("已有相同名稱的 Short 節目。")
            else:
                for program in programs:
                    if program.get("id") == selected_id:
                        program["name"] = clean_name
                        program["updated_at"] = datetime.now().isoformat(timespec="seconds")
                short_save_programs(programs)
                st.session_state["short_program_notice"] = "Short 節目名稱已更新。"
                st.rerun()
    with col_delete:
        st.caption("刪除只會從管理清單移除，不刪除既有檔案。")
        confirm = st.checkbox("確認刪除此節目", key=f"short_delete_confirm_{selected_id}")
        if st.button("刪除節目", key=f"short_delete_{selected_id}", disabled=not confirm):
            programs = [p for p in programs if p.get("id") != selected_id]
            short_save_programs(programs)
            st.session_state.pop("short_manage_program", None)
            st.session_state["short_program_notice"] = "Short 節目已從清單刪除。"
            st.rerun()

    episodes = short_list_episodes(selected)
    st.subheader("已生成集數及狀態")
    if episodes:
        st.dataframe(
            pd.DataFrame([{
                "集數": f"Ep{ep['episode_no']:02d}",
                "標題": ep.get("title", ""),
                "狀態": ep.get("status", "製作中"),
                "最後更新": ep.get("updated_at", ""),
                "資料夾": str(ep.get("path", "")),
            } for ep in episodes]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("此節目尚未生成任何 Short 集數。")

def render_short_control_steps(program: Dict, ep_dir: Path | None, selected_episode_no: int | None):
    st.subheader("控制步驟")
    is_new = ep_dir is None
    base_data = {} if is_new else short_read_json(short_episode_json_path(ep_dir), {})
    default_title = "" if is_new else base_data.get("title", "")
    title = st.text_input("步驟一「設定標題」", value=default_title, key=f"short_title_{program.get('id')}_{selected_episode_no or 'new'}")
    status_value = st.selectbox(
        "集數狀態",
        SHORT_STATUS_OPTIONS,
        index=SHORT_STATUS_OPTIONS.index(base_data.get("status", "製作中")) if base_data.get("status", "製作中") in SHORT_STATUS_OPTIONS else 0,
        key=f"short_status_{program.get('id')}_{selected_episode_no or 'new'}",
    )
    if st.button("儲存標題 / 建立集數", key=f"short_save_title_{program.get('id')}_{selected_episode_no or 'new'}"):
        episode_no = selected_episode_no or short_next_episode_no(program)
        ep_dir = short_ensure_episode(program, episode_no, title=title)
        ok = short_run_logged_step(
            ep_dir,
            "1",
            "設定標題",
            lambda: short_save_episode_data(ep_dir, {"title": title, "status": status_value}),
        )
        if ok:
            st.success(f"Ep{episode_no:02d} 已儲存。")
            st.rerun()

    if is_new:
        st.info("請先儲存標題建立新集數，再上傳語音或編輯後續內容。")
        return

    auto_refresh_logs = st.toggle(
        "自動刷新步驟 Log",
        value=True,
        key=f"short_log_auto_refresh_{ep_dir.name}",
        help="開啟後，Log 面板每 2 秒刷新一次，用來觀察後續接入背景任務後的執行進度。",
    )

    def render_inline_short_step_log(log_step_no: str, log_label: str):
        log_state = short_step_state(ep_dir, log_step_no)
        refresh_interval = 2 if (auto_refresh_logs and log_state.get("running")) else None

        @st.fragment(run_every=refresh_interval)
        def _render_inline_log():
            short_rerun_when_step_finishes(ep_dir, log_step_no, "control")
            short_render_step_log_panel(
                ep_dir,
                log_step_no,
                log_label,
                auto_expand_errors=(str(log_step_no) != "4"),
                key_suffix="control",
            )

        _render_inline_log()

    render_inline_short_step_log("1", "設定標題")

    st.divider()
    st.markdown("**步驟二「上傳語音」**")
    uploaded_audio = st.file_uploader(
        "上傳語音",
        type=["mp3", "m4a", "wav", "aac", "ogg"],
        key=f"short_audio_upload_{ep_dir.name}",
    )
    if uploaded_audio and st.button("儲存語音檔", key=f"short_audio_save_{ep_dir.name}"):
        audio_path = ep_dir / "01_audio" / uploaded_audio.name
        ok = short_run_logged_step(
            ep_dir,
            "2",
            "上傳語音",
            lambda: audio_path.write_bytes(uploaded_audio.getbuffer()),
        )
        if ok:
            st.success(f"語音已儲存：{audio_path.name}")
    audio_files = short_audio_files(ep_dir)
    if audio_files:
        audio_choice = st.selectbox("預覽語音", audio_files, format_func=lambda p: p.name, key=f"short_audio_preview_{ep_dir.name}")
        st.audio(str(audio_choice))
    else:
        st.caption("尚未上傳語音。")
    render_inline_short_step_log("2", "上傳語音")

    st.divider()
    steps = [
        ("3", "Whisper產生字幕", ep_dir / "02_subtitles" / "whisper.srt", "使用本地端 Whisper 模型將上傳語音轉成 SRT 字幕。"),
        ("4", "AI字幕校對", ep_dir / "02_subtitles" / "reviewed.srt", "使用選定 AI Provider 校對 Whisper 字幕並輸出 reviewed.srt。"),
        ("6", "AI 依語境分鏡規劃", short_storyboard_path(ep_dir), "使用選定 AI Provider 依字幕語境產生可編輯分鏡草稿。"),
        ("7", "AI 產生分鏡圖片", ep_dir / "04_images" / "storyboard_images_manifest.json", "依分鏡 prompt 產生 9:16 Short 圖片，並寫回分鏡 asset 欄位。"),
        ("8", "產生合成清單", short_compose_list_path(ep_dir), "依目前分鏡產生合成清單。"),
        ("9", "FFmpeg 影片合成", ep_dir / "05_output" / "final_short.mp4", "依合成清單用 FFmpeg 合成 9:16 Short 影片。"),
        ("10", "產生 Short metadata", short_metadata_path(ep_dir), "產生可於 Metadata 分頁修改的 metadata。"),
    ]
    short_subtitle_review_provider = st.session_state.get(f"short_subtitle_review_provider_{ep_dir.name}", "auto")
    short_storyboard_provider = st.session_state.get(f"short_storyboard_provider_{ep_dir.name}", "auto")
    short_storyboard_image_provider = st.session_state.get(f"short_storyboard_image_provider_{ep_dir.name}", "gemini")
    short_metadata_provider = st.session_state.get(f"short_metadata_provider_{ep_dir.name}", "auto")
    for step_no, label, output_path, caption in steps:
        step_state = short_step_state(ep_dir, step_no)
        cols = st.columns([1.2, 3, 1])
        cols[0].markdown(f"**步驟{step_no}「{label}」**")
        cols[1].caption(caption)
        exists = short_output_completed(output_path)
        if step_state.get("running"):
            cols[2].write("執行中")
        else:
            cols[2].write("已完成" if exists else "未產生")
        if step_no == "4":
            subtitle_review_providers = ["auto", "gemini", "openai", "nvidia"]
            short_subtitle_review_provider = st.selectbox(
                "步驟四 AI字幕校對 Provider",
                subtitle_review_providers,
                index=subtitle_review_providers.index(short_subtitle_review_provider)
                if short_subtitle_review_provider in subtitle_review_providers
                else 0,
                key=f"short_subtitle_review_provider_{ep_dir.name}",
                format_func=lambda value: {
                    "auto": "Gemini 額度不夠/失敗時自動轉 OpenAI",
                    "gemini": "Gemini",
                    "openai": "OpenAI",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )
        if step_no == "6":
            short_storyboard_provider = st.selectbox(
                "步驟六 AI分鏡規劃 Provider",
                ["auto", "gemini", "openai", "nvidia"],
                index=["auto", "gemini", "openai", "nvidia"].index(short_storyboard_provider)
                if short_storyboard_provider in ["auto", "gemini", "openai", "nvidia"]
                else 0,
                key=f"short_storyboard_provider_{ep_dir.name}",
                format_func=lambda value: {
                    "auto": "Gemini 額度不夠/失敗時自動轉 OpenAI",
                    "gemini": "Gemini",
                    "openai": "OpenAI",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )
        if step_no == "7":
            storyboard_image_providers = ["gemini", "openai", "nvidia"]
            short_storyboard_image_provider = st.selectbox(
                "步驟七 AI 產生分鏡圖片 Provider",
                storyboard_image_providers,
                index=storyboard_image_providers.index(short_storyboard_image_provider)
                if short_storyboard_image_provider in storyboard_image_providers
                else 0,
                key=f"short_storyboard_image_provider_{ep_dir.name}",
                format_func=lambda value: {
                    "gemini": "Gemini",
                    "openai": "OpenAI",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )
        if step_no == "10":
            short_metadata_provider = st.selectbox(
                "步驟十 Short metadata Provider",
                ["auto", "gemini", "openai", "nvidia"],
                index=["auto", "gemini", "openai", "nvidia"].index(short_metadata_provider)
                if short_metadata_provider in ["auto", "gemini", "openai", "nvidia"]
                else 0,
                key=f"short_metadata_provider_{ep_dir.name}",
                format_func=lambda value: {
                    "auto": "Gemini 額度不夠/失敗時自動轉 OpenAI",
                    "gemini": "Gemini",
                    "openai": "OpenAI",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )
        button_label = "執行中..." if step_state.get("running") else "產生/更新"
        if cols[2].button(
            button_label,
            key=f"short_step_{step_no}_{ep_dir.name}",
            disabled=bool(step_state.get("running")),
        ):
            if step_no == "3":
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_whisper_subtitles.py"),
                    "--ep-dir",
                    str(ep_dir),
                    "--output",
                    str(output_path),
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue
            if step_no == "4":
                source = ep_dir / "02_subtitles" / "whisper.srt"
                if not short_output_completed(source):
                    short_append_step_log(ep_dir, step_no, "找不到非空的 Whisper 字幕結果，無法執行 AI字幕校對。", level="ERROR")
                    st.error("請先完成步驟3，產生非空的 Whisper 字幕。")
                    continue
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_subtitle_reviewer.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output_path),
                    "--provider",
                    short_subtitle_review_provider,
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue
            if step_no == "6":
                source = short_preferred_subtitle_path(ep_dir)
                if source is None:
                    short_append_step_log(ep_dir, step_no, "找不到非空字幕，無法執行 AI 依語境分鏡規劃。", level="ERROR")
                    st.error("請先完成步驟4，或至少完成步驟3產生非空字幕。")
                    continue
                if source.name == "final.srt":
                    st.caption("Step 6 will use final.srt, including manual subtitle edits.")
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_storyboard_planner.py"),
                    "--input-srt",
                    str(source),
                    "--output",
                    str(output_path),
                    "--provider",
                    short_storyboard_provider,
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue
            if step_no == "7":
                storyboard_source = short_storyboard_path(ep_dir)
                if not short_output_completed(storyboard_source):
                    short_append_step_log(ep_dir, step_no, "找不到非空分鏡 storyboard.json，無法產生分鏡圖片。", level="ERROR")
                    st.error("請先完成步驟6，產生非空的 storyboard.json。")
                    continue
                storyboard_is_current, storyboard_message = short_storyboard_matches_current_subtitles(ep_dir, storyboard_source)
                if not storyboard_is_current:
                    short_append_step_log(ep_dir, step_no, storyboard_message, level="ERROR")
                    st.error(storyboard_message)
                    continue
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_generate_storyboard_images.py"),
                    "--ep-dir",
                    str(ep_dir),
                    "--storyboard",
                    str(storyboard_source),
                    "--manifest",
                    str(output_path),
                    "--provider",
                    short_storyboard_image_provider,
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue
            if step_no == "8":
                storyboard_source = short_storyboard_path(ep_dir)
                if not short_output_completed(storyboard_source):
                    short_append_step_log(ep_dir, step_no, "找不到非空分鏡 storyboard.json，無法產生合成清單。", level="ERROR")
                    st.error("請先完成步驟6，產生非空的 storyboard.json。")
                    continue
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_build_compose_list.py"),
                    "--ep-dir",
                    str(ep_dir),
                    "--storyboard",
                    str(storyboard_source),
                    "--output",
                    str(output_path),
                    "--preview-output",
                    str(short_preview_video_path(ep_dir)),
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue
            if step_no == "9":
                compose_source = short_compose_list_path(ep_dir)
                if not short_output_completed(compose_source):
                    short_append_step_log(ep_dir, step_no, "找不到非空合成清單 compose_list.json，無法執行 FFmpeg 合成。", level="ERROR")
                    st.error("請先完成步驟8，產生非空的 compose_list.json。")
                    continue
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_ffmpeg_compose.py"),
                    "--ep-dir",
                    str(ep_dir),
                    "--compose-list",
                    str(compose_source),
                    "--output",
                    str(output_path),
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue
            if step_no == "10":
                subtitle_source = ep_dir / "02_subtitles" / "final.srt"
                reviewed_source = ep_dir / "02_subtitles" / "reviewed.srt"
                whisper_source = ep_dir / "02_subtitles" / "whisper.srt"
                if not any(short_output_completed(path) for path in [subtitle_source, reviewed_source, whisper_source]):
                    short_append_step_log(ep_dir, step_no, "找不到非空字幕，無法產生 Short metadata。", level="ERROR")
                    st.error("請先完成字幕步驟，至少需有 final.srt、reviewed.srt 或 whisper.srt。")
                    continue
                cmd = [
                    sys.executable,
                    "-u",
                    str(_root() / "scripts" / "short_metadata_generator.py"),
                    "--ep-dir",
                    str(ep_dir),
                    "--output",
                    str(output_path),
                    "--provider",
                    short_metadata_provider,
                ]
                ok = short_start_logged_script(ep_dir, step_no, label, cmd)
                if ok:
                    st.success(f"步驟{step_no}已啟動，請看本步驟下方 Log。")
                    st.rerun()
                continue

            def run_short_placeholder_step():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("", encoding="utf-8")

            ok = short_run_logged_step(ep_dir, step_no, label, run_short_placeholder_step)
            if ok:
                st.success(f"步驟{step_no}已建立輸出。")
                st.rerun()
        render_inline_short_step_log(step_no, label)
        if step_no in {"3", "4"} and short_output_completed(output_path):
            with st.expander(f"查看步驟{step_no}字幕結果", expanded=False):
                st.code(output_path.read_text(encoding="utf-8", errors="ignore")[-12000:], language="text")
        if step_no == "6" and short_output_completed(output_path):
            with st.expander("查看步驟6分鏡規劃結果", expanded=False):
                storyboard_preview = short_read_json(output_path, [])
                if isinstance(storyboard_preview, list) and storyboard_preview:
                    st.dataframe(pd.DataFrame(storyboard_preview).head(20), use_container_width=True, hide_index=True)
                st.code(output_path.read_text(encoding="utf-8", errors="ignore")[:12000], language="json")
        if step_no == "7" and short_output_completed(output_path):
            with st.expander("查看步驟7分鏡圖片結果", expanded=False):
                manifest = short_read_json(output_path, {})
                items = manifest.get("items", []) if isinstance(manifest, dict) else []
                if isinstance(manifest, dict) and manifest.get("status") == "quota_exhausted":
                    st.error("Gemini/Imagen 額度或 monthly spending cap 已用盡，步驟7已停止批次產圖。")
                    st.caption(str(manifest.get("error", "")))
                elif isinstance(manifest, dict) and manifest.get("status") == "image_generation_failed":
                    st.error("Gemini/Imagen 已達上限，且 OpenAI 圖片 fallback 也失敗。")
                    st.caption(str(manifest.get("error", "")))
                elif isinstance(manifest, dict) and manifest.get("fallback") == "openai":
                    st.warning("Gemini/Imagen 已達上限，部分或全部圖片已自動改用 OpenAI 產生。")
                if items:
                    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                    preview_cols = st.columns(3)
                    for idx, item in enumerate(items[:9]):
                        image_path = ep_dir / str(item.get("path", ""))
                        if image_path.exists():
                            preview_cols[idx % 3].image(str(image_path), caption=f"Scene {item.get('scene', '')}", use_container_width=True)
                st.code(output_path.read_text(encoding="utf-8", errors="ignore")[:12000], language="json")
        if step_no == "4":
            source_path = output_path if short_output_completed(output_path) else (ep_dir / "02_subtitles" / "whisper.srt")
            subtitle_text = source_path.read_text(encoding="utf-8", errors="ignore") if short_output_completed(source_path) else ""
            manual_srt_key = f"short_step4_manual_srt_{ep_dir.name}"
            manual_source_key = f"{manual_srt_key}_source"
            manual_source_sig = (
                f"{source_path.resolve()}:{source_path.stat().st_mtime_ns}:{source_path.stat().st_size}"
                if short_output_completed(source_path)
                else "missing"
            )
            if st.session_state.get(manual_source_key) != manual_source_sig:
                st.session_state[manual_srt_key] = subtitle_text
                st.session_state[manual_source_key] = manual_source_sig
            with st.expander("手動調整字幕", expanded=not short_output_completed(output_path) and bool(subtitle_text)):
                st.caption("優先編輯 AI 校對後的 reviewed.srt；若尚未產生，會先載入 whisper.srt。儲存後會寫入 reviewed.srt。")
                review_audio_files = short_audio_files(ep_dir)
                if review_audio_files:
                    review_audio_choice = st.selectbox(
                        "校對用音效播放",
                        review_audio_files,
                        format_func=lambda p: p.name,
                        key=f"short_step4_review_audio_{ep_dir.name}",
                    )
                    st.audio(str(review_audio_choice))
                else:
                    st.caption("尚未上傳音效檔；請先在步驟二上傳語音/音效檔。")
                edited_reviewed_srt = st.text_area(
                    "字幕內容",
                    value=subtitle_text,
                    height=320,
                    key=manual_srt_key,
                )
                if st.button("儲存手動調整字幕", key=f"short_step4_save_manual_srt_{ep_dir.name}"):
                    def save_manual_reviewed_srt():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text(edited_reviewed_srt.strip() + "\n", encoding="utf-8")

                    ok = short_run_logged_step(ep_dir, "4", "手動調整字幕", save_manual_reviewed_srt)
                    if ok:
                        st.success("手動調整字幕已儲存到 reviewed.srt。")
                        st.rerun()

    st.divider()
    st.markdown("**步驟五「人工字幕確認」**")
    reviewed_path = ep_dir / "02_subtitles" / "reviewed.srt"
    final_path = ep_dir / "02_subtitles" / "final.srt"
    subtitle_text = final_path.read_text(encoding="utf-8", errors="ignore") if final_path.exists() else (
        reviewed_path.read_text(encoding="utf-8", errors="ignore") if reviewed_path.exists() else ""
    )
    edited_subtitles = st.text_area("手動修改字幕後存檔", value=subtitle_text, height=260, key=f"short_final_subtitles_{ep_dir.name}")
    if st.button("儲存人工確認字幕", key=f"short_save_final_subtitles_{ep_dir.name}"):
        def save_final_subtitles():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(edited_subtitles, encoding="utf-8")

        ok = short_run_logged_step(ep_dir, "5", "人工字幕確認", save_final_subtitles)
        if ok:
            st.success("人工確認字幕已儲存。")
    render_inline_short_step_log("5", "人工字幕確認")

def render_short_storyboard_editor(ep_dir: Path):
    st.subheader("修改分鏡/替換影音")
    storyboard_path = short_storyboard_path(ep_dir)
    storyboard = short_read_json(storyboard_path, [])
    if isinstance(storyboard, dict) and isinstance(storyboard.get("scenes"), list):
        storyboard = storyboard.get("scenes", [])
    if not isinstance(storyboard, list):
        storyboard = []
    if not storyboard:
        storyboard = [{
            "scene": 1,
            "start": "00:00:00,000",
            "end": "00:00:03,000",
            "start_seconds": 0,
            "end_seconds": 3,
            "subtitle": "",
            "summary": "",
            "prompt": "Vertical 9:16 short video scene",
            "image_prompt": "Vertical 9:16 short video scene",
            "asset": "",
            "animation_prompt": "",
            "animation_asset": "",
            "reason": "",
        }]
    storyboard_sig = "missing"
    if storyboard_path.exists():
        storyboard_sig = f"{int(storyboard_path.stat().st_mtime)}_{storyboard_path.stat().st_size}_{len(storyboard)}"
    st.caption(f"分鏡來源：{storyboard_path}")
    top_cols = st.columns([1, 1, 2])
    top_cols[0].metric("Scenes", len(storyboard))
    top_cols[1].metric("步驟6結果", "已產生" if short_output_completed(storyboard_path) else "尚未產生")
    if top_cols[2].button("重新載入步驟6分鏡結果", key=f"short_reload_storyboard_{ep_dir.name}_{storyboard_sig}"):
        st.rerun()
    meta_path = storyboard_path.with_name("storyboard_meta.json")
    if meta_path.exists():
        meta = short_read_json(meta_path, {})
        st.caption(
            f"Provider: {meta.get('provider_used', '') or meta.get('provider_requested', '')} | "
            f"Prompt: {meta.get('prompt_path', '')}"
        )

    def save_storyboard_rows(rows: list[dict], action_label: str) -> bool:
        for idx, row in enumerate(rows, 1):
            row["scene"] = idx
        return short_run_logged_step(ep_dir, "6", action_label, lambda: short_write_json(storyboard_path, rows))

    def scene_id(row: dict) -> str:
        return str(row.get("scene", row.get("scene_id", ""))).strip()

    def scene_label(row: dict) -> str:
        text = str(row.get("summary") or row.get("subtitle") or row.get("prompt") or "").strip()
        return f"Scene {scene_id(row)} | {text[:60]}"

    def resolve_asset_path(value: str) -> Path:
        raw = Path(str(value or "").strip())
        return raw if raw.is_absolute() else ep_dir / raw

    def rel_asset(path: Path) -> str:
        try:
            return path.resolve().relative_to(ep_dir.resolve()).as_posix()
        except Exception:
            return str(path)

    def scene_asset_candidates(row: dict, kind: str) -> list[Path]:
        raw_scene = scene_id(row)
        candidates = []
        if raw_scene:
            scene_tokens = [raw_scene]
            if str(raw_scene).isdigit():
                scene_no = int(raw_scene)
                scene_tokens.extend([f"{scene_no:03d}", str(scene_no)])
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"] if kind == "image" else ["*.mp4", "*.mov", "*.webm", "*.m4v"]
            base_dir = ep_dir / "04_images" / ("storyboard" if kind == "image" else "animations")
            for token in dict.fromkeys(scene_tokens):
                for pattern in patterns:
                    candidates.extend(sorted(base_dir.glob(f"scene_{token}{Path(pattern).suffix}")))
                    candidates.extend(sorted(base_dir.glob(f"scene_{token}_*{Path(pattern).suffix}")))
        valid = [path for path in candidates if path.exists() and path.is_file() and path.stat().st_size > 0]
        return sorted(
            valid,
            key=lambda path: (
                0 if "_upload_" in path.name.lower() else 1,
                -path.stat().st_mtime,
                path.name.lower(),
            ),
        )

    def fallback_scene_asset(row: dict, kind: str) -> Path | None:
        candidates = scene_asset_candidates(row, kind)
        return candidates[0] if candidates else None

    def collect_short_assets(kind: str) -> list[Path]:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"] if kind == "image" else ["*.mp4", "*.mov", "*.webm"]
        out = []
        seen = set()
        for pattern in patterns:
            for path in short_workspace_root().rglob(pattern):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(path)
        return sorted(out)

    def current_image(row: dict) -> Path | None:
        value = str(row.get("asset", "") or row.get("image_asset", "")).strip()
        path = resolve_asset_path(value) if value else None
        if path and path.exists():
            return path
        return fallback_scene_asset(row, "image")

    def current_animation(row: dict) -> Path | None:
        value = str(row.get("animation_asset", "") or row.get("animation_video_path", "")).strip()
        path = resolve_asset_path(value) if value else None
        if path and path.exists():
            return path
        return fallback_scene_asset(row, "animation")

    work_tab, edit_tab, assets_tab, table_tab = st.tabs(["單鏡工作台", "合併 / 切分", "素材 / Tag", "整張分鏡表"])

    with work_tab:
        scene_ids = [scene_id(row) or str(idx + 1) for idx, row in enumerate(storyboard)]
        scene_by_id = {sid: idx for idx, sid in enumerate(scene_ids)}
        scene_select_key = f"short_scene_select_{ep_dir.name}"
        if st.session_state.get(scene_select_key) not in scene_by_id:
            st.session_state[scene_select_key] = scene_ids[0]
        selected_scene_id = st.selectbox(
            "選擇 Scene",
            scene_ids,
            key=scene_select_key,
            format_func=lambda sid: scene_label(storyboard[scene_by_id.get(str(sid), 0)]),
        )
        selected_idx = scene_by_id.get(str(selected_scene_id), 0)
        selected = dict(storyboard[selected_idx])
        selected_scene = scene_id(selected) or str(selected_idx + 1)

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(f"**Scene {selected_scene}**")
            selected["summary"] = st.text_area("summary", value=str(selected.get("summary", "")), height=80, key=f"short_scene_summary_{ep_dir.name}_{selected_scene}_{storyboard_sig}")
            selected["subtitle"] = st.text_area("subtitle", value=str(selected.get("subtitle", "")), height=90, key=f"short_scene_subtitle_{ep_dir.name}_{selected_scene}_{storyboard_sig}")
            selected["prompt"] = st.text_area("image prompt", value=str(selected.get("prompt") or selected.get("image_prompt") or ""), height=150, key=f"short_scene_prompt_{ep_dir.name}_{selected_scene}_{storyboard_sig}")
            selected["image_prompt"] = selected["prompt"]
            selected["animation_prompt"] = st.text_area("animation prompt", value=str(selected.get("animation_prompt", "")), height=130, key=f"short_scene_anim_prompt_{ep_dir.name}_{selected_scene}_{storyboard_sig}")

            if st.button("儲存此 Scene", key=f"short_save_scene_{ep_dir.name}_{selected_scene}_{storyboard_sig}"):
                storyboard[selected_idx].update(selected)
                if save_storyboard_rows(storyboard, f"儲存 Scene {selected_scene}"):
                    st.success("Scene 已儲存。")
                    st.rerun()

            provider_cols = st.columns(2)
            scene_image_providers = ["gemini", "openai", "nvidia"]
            scene_animation_providers = ["gemini", "openai", "nvidia"]
            scene_image_provider = provider_cols[0].selectbox(
                "產生圖片 Provider",
                scene_image_providers,
                index=scene_image_providers.index("nvidia"),
                key=f"short_scene_image_provider_{ep_dir.name}_{selected_scene}",
                format_func=lambda value: {
                    "gemini": "Gemini",
                    "openai": "OpenAI",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )
            scene_animation_provider = provider_cols[1].selectbox(
                "產生動畫 Provider",
                scene_animation_providers,
                key=f"short_scene_animation_provider_{ep_dir.name}_{selected_scene}",
                format_func=lambda value: {
                    "gemini": "Gemini",
                    "openai": "OpenAI",
                    "nvidia": "NVIDIA",
                }.get(value, value),
            )
            if scene_animation_provider == "nvidia":
                st.caption("NVIDIA 動畫會使用 Stable Video Diffusion；此 endpoint 會將輸入圖縮放到 1024x576。")

            def storyboard_ready_for_scene_assets() -> bool:
                storyboard_is_current, storyboard_message = short_storyboard_matches_current_subtitles(ep_dir, storyboard_path)
                if not storyboard_is_current:
                    short_append_step_log(ep_dir, "7", storyboard_message, level="ERROR")
                    st.error(storyboard_message)
                    return False
                return True

            btn_cols = st.columns(4)
            if btn_cols[0].button("AI建議動畫 Prompt", key=f"short_suggest_anim_{ep_dir.name}_{selected_scene}"):
                storyboard[selected_idx].update(selected)
                save_storyboard_rows(storyboard, f"儲存 Scene {selected_scene} 後產生動畫 Prompt")
                cmd = [
                    sys.executable, "-u", str(_root() / "scripts" / "short_generate_animation_prompt.py"),
                    "--storyboard", str(storyboard_path),
                    "--scene-id", selected_scene,
                    "--provider", scene_animation_provider,
                ]
                if short_start_logged_script(ep_dir, "6", f"AI 建議 Scene {selected_scene} 動畫 Prompt", cmd):
                    st.success("動畫 Prompt 任務已啟動。")
                    st.rerun()
            if btn_cols[1].button("產生動畫", key=f"short_gen_anim_{ep_dir.name}_{selected_scene}"):
                cmd = [
                    sys.executable, "-u", str(_root() / "scripts" / "short_generate_animation.py"),
                    "--ep-dir", str(ep_dir),
                    "--storyboard", str(storyboard_path),
                    "--scene-id", selected_scene,
                    "--provider", scene_animation_provider,
                ]
                if short_start_logged_script(ep_dir, "7", f"產生 Scene {selected_scene} 動畫", cmd):
                    st.success("動畫任務已啟動。")
                    st.rerun()
            if btn_cols[2].button("重產圖片", key=f"short_regen_image_{ep_dir.name}_{selected_scene}"):
                if not storyboard_ready_for_scene_assets():
                    st.stop()
                cmd = [
                    sys.executable, "-u", str(_root() / "scripts" / "short_generate_storyboard_images.py"),
                    "--ep-dir", str(ep_dir),
                    "--storyboard", str(storyboard_path),
                    "--manifest", str(ep_dir / "04_images" / "storyboard_images_manifest.json"),
                    "--force",
                    "--scene-id", selected_scene,
                    "--provider", scene_image_provider,
                ]
                if short_start_logged_script(ep_dir, "7", f"重產 Scene {selected_scene} 圖片", cmd):
                    st.success("重產圖片任務已啟動。")
                    st.rerun()
            if btn_cols[3].button("重產圖並產動畫", key=f"short_regen_image_anim_{ep_dir.name}_{selected_scene}"):
                if not storyboard_ready_for_scene_assets():
                    st.stop()
                cmd = [
                    sys.executable, "-u", str(_root() / "scripts" / "short_regenerate_image_prompt_animation.py"),
                    "--ep-dir", str(ep_dir),
                    "--storyboard", str(storyboard_path),
                    "--manifest", str(ep_dir / "04_images" / "storyboard_images_manifest.json"),
                    "--scene-id", selected_scene,
                    "--image-provider", scene_image_provider,
                    "--animation-provider", scene_animation_provider,
                ]
                if short_start_logged_script(ep_dir, "7", f"重產 Scene {selected_scene} 圖片並產動畫", cmd):
                    st.success("重產圖並產動畫任務已啟動。")
                    st.rerun()

            st.markdown("**Scene 任務 Log**")
            short_render_step_log_panel(ep_dir, "6", "分鏡 / 動畫 Prompt", auto_expand_errors=False, key_suffix=f"scene_{selected_scene}")
            short_render_step_log_panel(ep_dir, "7", "圖片 / 動畫", auto_expand_errors=False, key_suffix=f"scene_{selected_scene}")

        with right:
            image_path = current_image(selected)
            animation_path = current_animation(selected)
            st.markdown("**目前素材預覽**")
            if image_path:
                st.image(str(image_path), caption=f"圖片：{image_path.name} | Tag: {asset_tag_badge(image_path)}", use_container_width=True)
            else:
                st.info("此 Scene 尚未指定圖片。")
            if animation_path:
                st.video(str(animation_path))
                st.caption(f"動畫：{animation_path.name} | Tag: {asset_tag_badge(animation_path)}")
            else:
                st.caption("此 Scene 尚未指定動畫。")

            uploaded_image = st.file_uploader("上傳/更換圖片", type=["png", "jpg", "jpeg", "webp"], key=f"short_upload_image_{ep_dir.name}_{selected_scene}")
            if uploaded_image and st.button("使用上傳圖片", key=f"short_use_upload_image_{ep_dir.name}_{selected_scene}"):
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_image.name)
                target = ep_dir / "04_images" / "storyboard" / f"scene_{selected_scene}_upload_{safe_name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(uploaded_image.getbuffer())
                storyboard[selected_idx]["asset"] = rel_asset(target)
                if save_storyboard_rows(storyboard, f"更換 Scene {selected_scene} 圖片"):
                    st.rerun()

            uploaded_animation = st.file_uploader("上傳/更換動畫", type=["mp4", "mov", "webm"], key=f"short_upload_animation_{ep_dir.name}_{selected_scene}")
            if uploaded_animation and st.button("使用上傳動畫", key=f"short_use_upload_animation_{ep_dir.name}_{selected_scene}"):
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_animation.name)
                target = ep_dir / "04_images" / "animations" / f"scene_{selected_scene}_upload_{safe_name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(uploaded_animation.getbuffer())
                rel = rel_asset(target)
                storyboard[selected_idx]["animation_asset"] = rel
                storyboard[selected_idx]["animation_video_path"] = rel
                if save_storyboard_rows(storyboard, f"更換 Scene {selected_scene} 動畫"):
                    st.rerun()

    with edit_tab:
        scene_options = [scene_label(row) for row in storyboard]
        st.markdown("**合併多個分鏡**")
        merge_cols = st.columns(3)
        merge_start = merge_cols[0].selectbox("起始 Scene", scene_options, key=f"short_merge_start_{ep_dir.name}_{storyboard_sig}")
        merge_end = merge_cols[1].selectbox("結束 Scene", scene_options, index=min(1, len(scene_options) - 1), key=f"short_merge_end_{ep_dir.name}_{storyboard_sig}")
        if merge_cols[2].button("合併選取 Scene", key=f"short_merge_apply_{ep_dir.name}_{storyboard_sig}"):
            start_idx = scene_options.index(merge_start)
            end_idx = scene_options.index(merge_end)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            merged_rows = storyboard[start_idx:end_idx + 1]
            merged = dict(merged_rows[0])
            merged["end"] = merged_rows[-1].get("end", merged.get("end", ""))
            merged["end_seconds"] = merged_rows[-1].get("end_seconds", merged.get("end_seconds", ""))
            merged["subtitle"] = " ".join(str(row.get("subtitle", "")).strip() for row in merged_rows if str(row.get("subtitle", "")).strip())
            merged["summary"] = " / ".join(str(row.get("summary", "")).strip() for row in merged_rows if str(row.get("summary", "")).strip())[:240]
            merged["reason"] = "手動合併多個分鏡"
            if save_storyboard_rows(storyboard[:start_idx] + [merged] + storyboard[end_idx + 1:], "合併多個分鏡"):
                st.rerun()

        st.divider()
        st.markdown("**一個分鏡切成多個**")
        split_cols = st.columns(3)
        split_label = split_cols[0].selectbox("要切分的 Scene", scene_options, key=f"short_split_scene_{ep_dir.name}_{storyboard_sig}")
        split_count = int(split_cols[1].number_input("切成幾段", min_value=2, max_value=8, value=2, step=1, key=f"short_split_count_{ep_dir.name}_{storyboard_sig}"))
        if split_cols[2].button("套用切分", key=f"short_split_apply_{ep_dir.name}_{storyboard_sig}"):
            idx = scene_options.index(split_label)
            source = storyboard[idx]
            start_sec = float(source.get("start_seconds", 0) or 0)
            end_sec = float(source.get("end_seconds", start_sec + split_count) or start_sec + split_count)
            duration = max((end_sec - start_sec) / split_count, 0.3)
            parts = []
            for part_idx in range(split_count):
                part = dict(source)
                part_start = start_sec + duration * part_idx
                part_end = end_sec if part_idx == split_count - 1 else part_start + duration
                part["start_seconds"] = round(part_start, 3)
                part["end_seconds"] = round(part_end, 3)
                part["start"] = seconds_to_label(part_start).replace(".", ",")
                part["end"] = seconds_to_label(part_end).replace(".", ",")
                part["summary"] = f"{source.get('summary', '')} Part {part_idx + 1}".strip()
                part["asset"] = source.get("asset", "") if part_idx == 0 else ""
                part["animation_asset"] = ""
                part["animation_video_path"] = ""
                part["reason"] = "手動切分分鏡"
                parts.append(part)
            if save_storyboard_rows(storyboard[:idx] + parts + storyboard[idx + 1:], "一個分鏡切成多個"):
                st.rerun()

    with assets_tab:
        image_assets = collect_short_assets("image")
        animation_assets = collect_short_assets("animation")
        all_assets = image_assets + animation_assets
        tag_options = collect_asset_tag_options(all_assets)
        selected_tag = st.selectbox(
            "依 Tag 篩選素材",
            tag_options,
            format_func=lambda value: {"__all__": "全部", "__untagged__": "未貼 tag"}.get(value, value),
            key=f"short_asset_tag_filter_{ep_dir.name}_{storyboard_sig}",
        )
        filtered_images = filter_asset_paths_by_tag(image_assets, selected_tag)
        filtered_animations = filter_asset_paths_by_tag(animation_assets, selected_tag)
        labels = [scene_label(row) for row in storyboard]
        selected_label = st.selectbox("套用到 Scene", labels, key=f"short_asset_scene_select_{ep_dir.name}_{storyboard_sig}")
        selected_idx = labels.index(selected_label)
        selected_scene = scene_id(storyboard[selected_idx])
        col_img, col_anim = st.columns(2)
        with col_img:
            st.caption("圖片素材")
            if filtered_images:
                chosen_image = st.selectbox("選擇圖片", [str(path) for path in filtered_images], format_func=lambda value: f"{Path(value).name} | Tag: {asset_tag_badge(Path(value))}", key=f"short_choose_image_asset_{ep_dir.name}_{storyboard_sig}")
                if st.button("套用圖片", key=f"short_apply_image_asset_{ep_dir.name}_{storyboard_sig}"):
                    storyboard[selected_idx]["asset"] = rel_asset(Path(chosen_image))
                    if save_storyboard_rows(storyboard, f"套用圖片到 Scene {selected_scene}"):
                        st.rerun()
            else:
                st.caption("沒有符合條件的圖片。")
        with col_anim:
            st.caption("動畫素材")
            if filtered_animations:
                chosen_animation = st.selectbox("選擇動畫", [str(path) for path in filtered_animations], format_func=lambda value: f"{Path(value).name} | Tag: {asset_tag_badge(Path(value))}", key=f"short_choose_animation_asset_{ep_dir.name}_{storyboard_sig}")
                if st.button("套用動畫", key=f"short_apply_animation_asset_{ep_dir.name}_{storyboard_sig}"):
                    rel = rel_asset(Path(chosen_animation))
                    storyboard[selected_idx]["animation_asset"] = rel
                    storyboard[selected_idx]["animation_video_path"] = rel
                    if save_storyboard_rows(storyboard, f"套用動畫到 Scene {selected_scene}"):
                        st.rerun()
            else:
                st.caption("沒有符合條件的動畫。")

        tag_target_options = [str(path) for path in all_assets]
        if tag_target_options:
            tag_target = st.selectbox("選擇要貼 Tag 的素材", tag_target_options, format_func=lambda value: f"{Path(value).name} | Tag: {asset_tag_badge(Path(value))}", key=f"short_tag_target_{ep_dir.name}_{storyboard_sig}")
            tag_text = st.text_input("Tag", value=asset_tag_for_path(tag_target), key=f"short_tag_text_{ep_dir.name}_{storyboard_sig}_{tag_target}")
            if st.button("儲存素材 Tag", key=f"short_save_asset_tag_{ep_dir.name}_{storyboard_sig}"):
                saved = set_asset_tag_for_path(tag_target, tag_text)
                if saved:
                    st.success(f"已更新素材 Tag：{saved}")
                    st.rerun()

    with table_tab:
        st.markdown("**分鏡預覽**")
        preview_cols = st.columns(5)
        for idx, row in enumerate(storyboard):
            scene_no = scene_id(row) or str(idx + 1)
            image_path = current_image(row)
            with preview_cols[idx % len(preview_cols)]:
                if image_path:
                    try:
                        thumbnail = short_image_thumbnail_bytes(str(image_path), image_path.stat().st_mtime_ns)
                    except Exception:
                        thumbnail = b""
                    if thumbnail:
                        st.image(thumbnail, caption=f"Scene {scene_no}", use_container_width=True)
                    else:
                        st.caption(f"Scene {scene_no}：縮圖讀取失敗")
                else:
                    st.caption(f"Scene {scene_no}：無圖片")
        st.divider()

        df = pd.DataFrame(storyboard)
        preferred_cols = ["scene", "start", "end", "subtitle", "summary", "prompt", "image_prompt", "asset", "animation_prompt", "animation_asset", "reason", "start_seconds", "end_seconds", "source_type"]
        ordered_cols = [col for col in preferred_cols if col in df.columns] + [col for col in df.columns if col not in preferred_cols]
        df = df[ordered_cols]
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key=f"short_storyboard_editor_{ep_dir.name}_{storyboard_sig}")
        if st.button("儲存整張分鏡表", key=f"short_save_storyboard_{ep_dir.name}"):
            if save_storyboard_rows(edited.fillna("").to_dict(orient="records"), "儲存整張分鏡表"):
                st.success("分鏡已儲存。")

    st.caption("替換影音可先填入 asset / animation_asset；執行「產生合成清單」會將目前分鏡寫入清單。")

def render_short_video_preview(ep_dir: Path):
    st.subheader("影片預覽")
    compose_path = short_compose_list_path(ep_dir)
    preview_video_path = short_preview_video_path(ep_dir)
    final_video_path = ep_dir / "05_output" / "final_short.mp4"

    if not short_output_completed(compose_path):
        st.info("尚未產生合成清單。請先執行步驟8「產生合成清單」。")
        return

    compose_payload = short_read_json(compose_path, {})
    if not isinstance(compose_payload, dict):
        st.error("合成清單格式無法讀取。")
        return

    items = compose_payload.get("items", []) if isinstance(compose_payload.get("items", []), list) else []
    warnings = compose_payload.get("warnings", []) if isinstance(compose_payload.get("warnings", []), list) else []
    timeline_rows = [
        {
            "scene": item.get("scene", idx + 1),
            "start": item.get("start", ""),
            "end": item.get("end", ""),
            "duration_seconds": item.get("duration_seconds", ""),
            "visual_type": item.get("visual_type", ""),
            "subtitle": item.get("subtitle", ""),
            "selected_visual": item.get("selected_visual", ""),
        }
        for idx, item in enumerate(items)
    ]

    subtitle_candidates = compose_payload.get("subtitle_candidates", [])
    subtitle_path = None
    if isinstance(subtitle_candidates, list):
        for candidate in subtitle_candidates:
            path = Path(str(candidate))
            if not path.is_absolute():
                path = ep_dir / path
            if short_output_completed(path):
                subtitle_path = path
                break

    def video_meta(path: Path) -> tuple[str, str]:
        if not short_output_completed(path):
            return "未產生", ""
        duration = media_duration_seconds(str(path))
        duration_label = seconds_to_label(duration) if duration is not None else "未知"
        size_mb = path.stat().st_size / 1024 / 1024
        return duration_label, f"{size_mb:.1f} MB"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes", compose_payload.get("scene_count", len(items)))
    c2.metric("Warnings", len(warnings))
    c3.metric("預覽長度", video_meta(preview_video_path)[0])
    c4.metric("正式影片", "已產生" if short_output_completed(final_video_path) else "未產生")

    if warnings:
        st.warning("合成清單有缺少素材或其他警告，請先在「修改分鏡/替換影音」補齊。")

    preview_tab, final_tab, data_tab = st.tabs(["合成前預覽", "正式影片", "資料檢查"])

    def render_player_panel(path: Path, empty_message: str, caption: str):
        player_col, side_col = st.columns([0.78, 1.42], gap="large")
        with player_col:
            if short_output_completed(path):
                st.video(str(path))
                st.caption(caption)
            else:
                st.info(empty_message)
        with side_col:
            duration_label, size_label = video_meta(path)
            info_cols = st.columns(3)
            info_cols[0].metric("長度", duration_label)
            info_cols[1].metric("大小", size_label or "-")
            info_cols[2].metric("比例", compose_payload.get("ratio", "9:16"))
            if path.exists():
                st.caption(str(path))
            if timeline_rows:
                st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True, height=330)

    with preview_tab:
        preview_state = short_step_state(ep_dir, "8_preview")
        preview_is_stale = (
            short_output_completed(preview_video_path)
            and preview_video_path.stat().st_mtime < compose_path.stat().st_mtime
        )
        action_cols = st.columns([1.1, 1, 2.4])
        if action_cols[0].button(
            "產生/更新預覽影片",
            key=f"short_generate_preview_video_{ep_dir.name}",
            disabled=bool(preview_state.get("running")) or bool(warnings),
        ):
            cmd = [
                sys.executable,
                "-u",
                str(_root() / "scripts" / "short_ffmpeg_compose.py"),
                "--ep-dir",
                str(ep_dir),
                "--compose-list",
                str(compose_path),
                "--output",
                str(preview_video_path),
            ]
            ok = short_start_logged_script(ep_dir, "8_preview", "合成前預覽影片", cmd)
            if ok:
                st.success("合成前預覽影片已啟動產生，請看下方 Log。")
                st.rerun()
        action_cols[1].write("執行中" if preview_state.get("running") else ("已產生" if short_output_completed(preview_video_path) else "未產生"))
        if warnings:
            action_cols[2].caption("合成清單仍有警告，請先補齊素材後再產生預覽影片。")
        elif preview_is_stale:
            action_cols[2].warning("預覽影片早於目前合成清單，建議重新產生。")
        else:
            action_cols[2].caption("預覽影片依步驟8合成清單產生，適合先檢查節奏、字幕與素材。")
        render_player_panel(
            preview_video_path,
            "尚未產生合成前預覽影片。執行步驟8後會自動產生，也可以在這裡手動更新。",
            "合成前預覽影片",
        )
        short_render_step_log_panel(ep_dir, "8_preview", "合成前預覽影片", auto_expand_errors=False, key_suffix="video_preview")

    with final_tab:
        render_player_panel(
            final_video_path,
            "尚未產生正式影片。請執行步驟9「FFmpeg 影片合成」。",
            "步驟9正式輸出影片",
        )

    with data_tab:
        if warnings:
            with st.expander("合成清單警告", expanded=True):
                st.dataframe(pd.DataFrame(warnings), use_container_width=True, hide_index=True)
        if subtitle_path:
            with st.expander("查看字幕檔", expanded=False):
                st.caption(str(subtitle_path))
                st.code(subtitle_path.read_text(encoding="utf-8", errors="ignore")[-12000:], language="text")
        with st.expander("查看 compose_list.json", expanded=False):
            st.code(compose_path.read_text(encoding="utf-8", errors="ignore")[:12000], language="json")

def render_short_metadata_editor(ep_dir: Path):
    st.subheader("Metadata")
    meta_path = short_metadata_path(ep_dir)
    ep_data = short_read_json(short_episode_json_path(ep_dir), {})
    metadata = short_read_json(meta_path, {})
    with st.form(f"short_metadata_form_{ep_dir.name}"):
        title = st.text_input("Title", value=metadata.get("title") or ep_data.get("title", ""))
        hook = st.text_input("Hook", value=metadata.get("hook", ""))
        description = st.text_area("Description", value=metadata.get("description", ""), height=180)
        hashtags = st.text_input("Hashtags", value=", ".join(metadata.get("hashtags", ["#Shorts"]) if isinstance(metadata.get("hashtags", []), list) else []))
        tags = st.text_input("Tags", value=", ".join(metadata.get("tags", []) if isinstance(metadata.get("tags", []), list) else []))
        pinned_comment = st.text_area("Pinned Comment", value=metadata.get("pinned_comment", ""), height=120)
        summary = st.text_area("Summary", value=metadata.get("summary", ""), height=120)
        privacy = st.selectbox("Privacy", ["private", "unlisted", "public"], index=["private", "unlisted", "public"].index(metadata.get("privacy", "private")) if metadata.get("privacy", "private") in ["private", "unlisted", "public"] else 0)
        submitted = st.form_submit_button("儲存 Metadata")
    if submitted:
        payload = {
            "title": title,
            "hook": hook,
            "description": description,
            "hashtags": [h.strip() for h in hashtags.split(",") if h.strip()],
            "tags": [tag.strip().lstrip("#") for tag in tags.split(",") if tag.strip()],
            "pinned_comment": pinned_comment,
            "summary": summary,
            "language": metadata.get("language", "zh-TW"),
            "privacy": privacy,
            "generated_by": metadata.get("generated_by", ""),
            "generated_at": metadata.get("generated_at", ""),
            "source": metadata.get("source", {}),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        ok = short_run_logged_step(
            ep_dir,
            "10",
            "修改 Short metadata",
            lambda: short_write_json(meta_path, payload),
        )
        if ok:
            st.success("Short metadata 已儲存。")
    if meta_path.exists():
        st.code(meta_path.read_text(encoding="utf-8", errors="ignore"), language="json")

def render_short_publish(ep_dir: Path):
    st.subheader("發佈")
    video_path = short_final_video_path(ep_dir)
    meta_path = short_metadata_path(ep_dir)
    subtitle_path = next((path for path in [
        ep_dir / "02_subtitles" / "final.srt",
        ep_dir / "02_subtitles" / "reviewed.srt",
        ep_dir / "02_subtitles" / "whisper.srt",
    ] if short_output_completed(path)), None)
    cover_path = short_publish_cover_path(ep_dir)
    upload_record_path = short_upload_record_path(ep_dir)
    upload_record = short_read_json(upload_record_path, {})
    upload_state = short_step_state(ep_dir, "publish")

    ready_cols = st.columns(5)
    ready_cols[0].metric("影片", "就緒" if short_output_completed(video_path) else "缺少")
    ready_cols[1].metric("Metadata", "就緒" if short_output_completed(meta_path) else "缺少")
    ready_cols[2].metric("字幕", "就緒" if subtitle_path else "缺少")
    ready_cols[3].metric("封面", "就緒" if cover_path else "未設定")
    ready_cols[4].metric("上傳狀態", "執行中" if upload_state.get("running") else upload_record.get("status", "尚無"))

    st.caption(f"影片：{video_path}")
    st.caption(f"Metadata：{meta_path}")
    st.caption(f"字幕：{subtitle_path or '尚未找到字幕'}")
    st.caption(f"封面：{cover_path or '預設會尋找 05_output/cover 或第一張分鏡圖'}")

    cover_cols = st.columns([1.1, 1, 2])
    with cover_cols[0]:
        uploaded_cover = st.file_uploader("上傳封面", type=["png", "jpg", "jpeg"], key=f"short_cover_upload_{ep_dir.name}")
        if uploaded_cover is not None:
            suffix = Path(uploaded_cover.name).suffix.lower() or ".png"
            save_path = ep_dir / "05_output" / f"cover{suffix}"
            upload_sig = f"{uploaded_cover.name}:{uploaded_cover.size}"
            processed_key = f"short_cover_upload_processed_{ep_dir.name}"
            if st.session_state.get(processed_key) != upload_sig:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(uploaded_cover.getbuffer())
                st.session_state[processed_key] = upload_sig
                st.success(f"已儲存封面：{save_path.name}")
                st.rerun()
    with cover_cols[1]:
        if cover_path and cover_path.exists():
            st.image(str(cover_path), caption="目前封面", use_container_width=True)
        else:
            st.caption("尚未設定封面。")
    with cover_cols[2]:
        st.caption("封面會在上傳影片成功後設定到 YouTube。若未上傳封面，會優先使用第一張分鏡圖。")

    if upload_record.get("video_id"):
        st.markdown(f"[開啟 YouTube 影片](https://www.youtube.com/watch?v={upload_record['video_id']})")

    if not short_output_completed(video_path):
        st.warning("發布前需要先完成步驟9，產生 final_short.mp4。")
    if not short_output_completed(meta_path):
        st.warning("發布前需要先完成步驟10，產生並確認 Short metadata。")

    token_sig, client_sig = youtube_auth_file_signatures()
    playlist_items, playlist_error = list_youtube_playlists_cached(token_sig, client_sig)
    playlist_options = [{"id": "", "title": "(不加入播放清單)", "label": "(不加入播放清單)"}] + playlist_items
    if playlist_error:
        st.error(f"YouTube 播放清單讀取失敗：{playlist_error}")
        if "invalid_grant" in playlist_error.lower() or "重新授權" in playlist_error:
            if st.button("重新授權 YouTube", key=f"short_reauth_youtube_{ep_dir.name}"):
                try:
                    reauthorize_youtube_for_app()
                    list_youtube_playlists_cached.clear()
                    st.success("YouTube 已重新授權。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"YouTube 重新授權失敗：{exc}")

    settings = short_read_json(short_publish_settings_path(ep_dir), {})
    privacy_key = f"short_publish_privacy_{ep_dir.name}"
    schedule_key = f"short_publish_schedule_on_{ep_dir.name}"
    date_key = f"short_publish_date_{ep_dir.name}"
    time_key = f"short_publish_time_{ep_dir.name}"
    playlist_key = f"short_publish_playlist_{ep_dir.name}"
    if privacy_key not in st.session_state:
        st.session_state[privacy_key] = str(settings.get("privacy") or upload_record.get("privacy") or "private")
    if schedule_key not in st.session_state:
        st.session_state[schedule_key] = bool(settings.get("schedule_on") or upload_record.get("publish_at"))
    if playlist_key not in st.session_state:
        st.session_state[playlist_key] = resolve_default_playlist_id(playlist_options, str(settings.get("playlist_id") or upload_record.get("playlist_id") or ""))

    privacy_val = st.selectbox("隱私設定", ["private", "unlisted", "public"], key=privacy_key)
    schedule_on = st.checkbox("排程發布", key=schedule_key)
    sched_cols = st.columns(2)
    with sched_cols[0]:
        publish_date = st.date_input("發布日期", key=date_key, disabled=not schedule_on)
    with sched_cols[1]:
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

    action_cols = st.columns([1.2, 1.2, 3])
    if action_cols[0].button("儲存發布設定", key=f"short_save_publish_settings_{ep_dir.name}"):
        short_write_json(short_publish_settings_path(ep_dir), {
            "privacy": privacy_val,
            "schedule_on": bool(schedule_on),
            "publish_at": publish_at_text,
            "playlist_id": selected_playlist_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        st.success("發布設定已儲存。")
    upload_disabled = upload_state.get("running") or not short_output_completed(video_path) or not short_output_completed(meta_path)
    if action_cols[1].button("上傳到 YouTube", type="primary", key=f"short_upload_youtube_{ep_dir.name}", disabled=upload_disabled):
        cmd = [
            sys.executable,
            "-u",
            str(_root() / "scripts" / "upload_short_to_youtube.py"),
            "--ep-dir",
            str(ep_dir),
            "--privacy",
            privacy_val,
        ]
        if publish_at_text:
            cmd.extend(["--publish_at", publish_at_text])
        if selected_playlist_id:
            cmd.extend(["--playlist_id", selected_playlist_id])
        if cover_path:
            cmd.extend(["--cover_path", str(cover_path)])
        ok = short_start_logged_script(ep_dir, "publish", "上傳 Short 到 YouTube", cmd)
        if ok:
            st.success("已啟動 YouTube 上傳，請查看發布 Log。")
            st.rerun()
    if action_cols[2].button("重新整理發布狀態", key=f"short_refresh_publish_{ep_dir.name}"):
        st.rerun()

    if upload_record:
        with st.expander("上傳紀錄", expanded=False):
            st.json(upload_record)
    short_render_step_log_panel(ep_dir, "publish", "上傳 Short 到 YouTube", auto_expand_errors=False, key_suffix="publish")

def render_short_factory():
    st.header("短影音工廠")
    programs = short_load_programs()
    if not programs:
        st.info("尚未建立 Short 節目。請先到「Short 管理」新增節目名稱。")
        return
    program_id = st.selectbox(
        "選擇 Short 節目名稱",
        [p.get("id") for p in programs],
        format_func=lambda pid: next((p.get("name", pid) for p in programs if p.get("id") == pid), pid),
        key="short_factory_program",
    )
    program = short_get_program(program_id)
    if not program:
        return
    episodes = short_list_episodes(program)
    episode_options = [SHORT_BLANK_OPTION] + [f"Ep{ep['episode_no']:02d} - {ep.get('title', '')}" for ep in episodes]
    episode_choice = st.selectbox("選擇集數", episode_options, key=f"short_factory_episode_{program_id}")
    selected_episode_no = None
    ep_dir = None
    if episode_choice != SHORT_BLANK_OPTION:
        m = re.match(r"Ep(\d+)", episode_choice)
        if m:
            selected_episode_no = int(m.group(1))
            ep_dir = short_episode_dir(program, selected_episode_no)

    if ep_dir:
        st.caption(f"Episode Path: {ep_dir}")
    else:
        st.caption("目前為新增/空白設定。")

    tab_control, tab_storyboard, tab_preview, tab_metadata, tab_publish = st.tabs(["控制步驟", "修改分鏡/替換影音", "影片預覽", "Metadata", "發佈"])
    with tab_control:
        render_short_control_steps(program, ep_dir, selected_episode_no)
    with tab_storyboard:
        if ep_dir:
            render_short_storyboard_editor(ep_dir)
        else:
            st.info("請先在「控制步驟」儲存標題建立新集數。")
    with tab_preview:
        if ep_dir:
            render_short_video_preview(ep_dir)
        else:
            st.info("請先在「控制步驟」儲存標題建立新集數。")
    with tab_metadata:
        if ep_dir:
            render_short_metadata_editor(ep_dir)
        else:
            st.info("請先建立新集數後再編輯 Metadata。")
    with tab_publish:
        if ep_dir:
            render_short_publish(ep_dir)
        else:
            st.info("請先建立新集數後再發佈。")

