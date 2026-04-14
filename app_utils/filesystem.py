import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

MANUAL_MARKER_TEMPLATE = "step{no}_manual_done.txt"

EP_DIR_PATTERN = re.compile(r"^Ep(?P<ep>\d{2})_(?P<start>\d{4})_(?P<end>\d{4})$")

STATUS_DEFAULT = {
    "1": "pending",
    "1.5": "pending",
    "2": "pending",
    "3": "pending",
    "4": "pending",
    "5": "pending",
    "6": "pending",
    "7": "pending",
}


def workspace_dir(root: Path, ws_override: str | None = None) -> Path:
    return root / (ws_override or "workspace")


def list_episode_dirs(root: Path, ws_override: str | None = None) -> List[Path]:
    ws = workspace_dir(root, ws_override)
    if not ws.exists():
        return []
    return sorted([p for p in ws.iterdir() if p.is_dir() and EP_DIR_PATTERN.match(p.name)])


def parse_episode_info(ep_path: Path) -> Dict:
    m = EP_DIR_PATTERN.match(ep_path.name)
    if not m:
        return {"ep": None, "start": None, "end": None, "path": ep_path}
    return {
        "ep": int(m.group("ep")),
        "start": int(m.group("start")),
        "end": int(m.group("end")),
        "path": ep_path,
    }


def episode_status_path(ep_path: Path) -> Path:
    return ep_path / "status.json"


def read_status(ep_path: Path) -> Dict:
    p = episode_status_path(ep_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stages": STATUS_DEFAULT.copy(), "updated_at": None}


def write_status(ep_path: Path, data: Dict) -> None:
    data = dict(data)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    episode_status_path(ep_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_logs_dir(ep_path: Path) -> Path:
    p = ep_path / "00_logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_file_for_stage(ep_path: Path, stage_id: str) -> Path:
    return ensure_logs_dir(ep_path) / f"stage_{stage_id.replace('.', '_')}.log"


def video_output_path(ep_path: Path) -> Path:
    return ep_path / "05_output" / "final_video.mp4"


def subtitles_raw_path(ep_path: Path) -> Path:
    return ep_path / "02_subtitles" / "notebooklm_audio.srt"


def subtitles_fixed_path(ep_path: Path) -> Path:
    return ep_path / "02_subtitles" / "notebooklm_audio_fixed.srt"


def audio_merged_path(ep_path: Path) -> Path:
    return ep_path / "01_audio" / "notebooklm_audio.m4a"


def images_dir(ep_path: Path) -> Path:
    return ep_path / "04_images"


def storyboard_csv_path(ep_path: Path) -> Path:
    return ep_path / "03_storyboards" / "storyboard.csv"


def inputs_txt_path(ep_path: Path) -> Path:\n    p1 = ep_path / "05_output" / "inputs.txt"\n    return p1 if p1.exists() else (ep_path / "inputs.txt")


def youtube_meta_paths(ep_path: Path) -> List[Path]:
    return [
        ep_path / "05_output" / "youtube_meta.json",
        ep_path / "youtube_meta.json",
    ]


def _has_any_image(ep_path: Path) -> bool:
    p = images_dir(ep_path)
    if not p.exists():
        return False
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        if any(p.glob(ext)):
            return True
    return False


def _file_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def infer_stage_statuses(ep_path: Path, prev: Optional[Dict] = None) -> Dict:
    """Infer stage statuses from filesystem. Preserve prev error/running if not completed."""
    st = STATUS_DEFAULT.copy()
    # Stage 1: images exist
    if _has_any_image(ep_path):
        st["1"] = "done"
    # Stage 1.5: merged audio exists
    if audio_merged_path(ep_path).exists():
        st["1.5"] = "done"
    # Stage 2: raw srt exists
    if subtitles_raw_path(ep_path).exists():
        st["2"] = "done"
    # Stage 3: fixed srt exists
    if subtitles_fixed_path(ep_path).exists():
        st["3"] = "done"
    '# Stage 4: ai review / storyboard exists
    for marker in [ep_path / "03_storyboards" / "ai_review.json", ep_path / "03_storyboards" / "ai_review_done.txt"]:
        if marker.exists():
            st["4"] = "done"
            break
    if st.get("4") != "done" and storyboard_csv_path(ep_path).exists():
        st["4"] = "done"
    # Stage 5: inputs.txt exists (either 05_output/inputs.txt or root inputs.txt) and non-empty
    if _file_nonempty(inputs_txt_path(ep_path)):
        st["5"] = "done"
    # Stage 6: final video exists' inputs.txt exists and non-empty OR storyboard.csv exists
    if _file_nonempty(inputs_txt_path(ep_path)) or storyboard_csv_path(ep_path).exists():
        st["5"] = "done"
    # Stage 6: final video exists
    if video_output_path(ep_path).exists():
        st["6"] = "done"
    # Stage 7: youtube meta or uploaded marker
    uploaded_markers = [ep_path / "05_output" / "uploaded.txt", ep_path / "05_output" / "upload.ok"]
    if any(p.exists() for p in youtube_meta_paths(ep_path)) or any(m.exists() for m in uploaded_markers):
        st["7"] = "done"

    # Overlay previous statuses (keep error/running if not already done)
    if prev and isinstance(prev, dict):
        p_st = (prev.get("stages") or {}) if "stages" in prev else prev
        for sid, pv in p_st.items():
            if st.get(sid) != "done" and pv in ("error", "running"):
                st[sid] = pv
    return st


def manual_marker_path(ep_path: Path, step_no: str) -> Path:
    return ensure_logs_dir(ep_path) / MANUAL_MARKER_TEMPLATE.format(no=str(step_no))


def set_manual_marker(ep_path: Path, step_no: str, done: bool) -> None:
    p = manual_marker_path(ep_path, step_no)
    if done:
        p.write_text("done", encoding="utf-8")
    else:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def has_manual_marker(ep_path: Path, step_no: str) -> bool:
    return manual_marker_path(ep_path, step_no).exists()

