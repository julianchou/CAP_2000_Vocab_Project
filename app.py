import os
import sys
from pathlib import Path
import json
import re
import time
import base64
import mimetypes
from typing import Dict
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from app_utils.filesystem import (
    list_episode_dirs,
    parse_episode_info,
    read_status,
    write_status,
    video_output_path,
    subtitles_fixed_path,
    audio_merged_path,
    images_dir,
    inputs_txt_path,
    infer_stage_statuses,
    manual_marker_path,
    set_manual_marker,
    has_manual_marker,
    log_file_for_stage,
    storyboard_csv_path,
    youtube_meta_paths,
)
from app_utils.pipeline import StageRunner, load_stage_config
import yaml
from app_utils.finops import load_cost_model, estimate_batch_cost, gating
from app_utils.ui_helpers import with_emoji
from app_utils.substeps import (
    load_substeps_config,
    evaluate_substeps,
    evaluate_substeps_debug,
)

ROOT = Path(__file__).resolve().parent

PROFILES_CFG = ROOT / "config" / "profiles.yaml"
SECTION_OPTIONS = [
    "📊 Dashboard",
    "⚙️ Pipeline Manager",
    "💰 FinOps Monitor",
    "🎬 Studio & Publisher",
]

def load_profiles():
    if PROFILES_CFG.exists():
        data = yaml.safe_load(PROFILES_CFG.read_text(encoding="utf-8")) or {}
        return data.get("profiles", [])
    return []

def get_profile_map():
    profs = load_profiles()
    return {p.get("id"): p for p in profs}

st.set_page_config(page_title="CAP 2000 AI Video CMS", layout="wide")

query_section = st.query_params.get("section")
if query_section in SECTION_OPTIONS and "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = query_section

st.sidebar.title("CAP 2000 控制台")
section = st.sidebar.radio("功能模組", SECTION_OPTIONS, key="nav_radio")
if st.query_params.get("section") != section:
    st.query_params["section"] = section

# Common data (profile-aware)
profiles = get_profile_map()
# Sidebar profile switch
profile_ids = list(profiles.keys()) or ["vocab"]
if "profile_id" not in st.session_state:
    st.session_state["profile_id"] = profile_ids[0]
choices = [profiles.get(pid, {"name": pid}).get("name", pid) + f" ({pid})" for pid in profile_ids]
idx_default = profile_ids.index(st.session_state["profile_id"]) if st.session_state["profile_id"] in profile_ids else 0
selected = st.sidebar.selectbox("專案環境", choices, index=idx_default, key="profile_select")
pid = selected.split("(")[-1].rstrip(")")
if pid in profiles and pid != st.session_state["profile_id"]:
    st.session_state["profile_id"] = pid
profile = profiles.get(st.session_state["profile_id"], {})
WS_ROOT = profile.get("workspace", "workspace")
PATH_STAGES = profile.get("stages")
PATH_SUBSTEPS = profile.get("substeps")
PATH_COSTS = profile.get("costs")
stage_cfg = load_stage_config(ROOT, PATH_STAGES)
runner = StageRunner(ROOT, PATH_STAGES, (ROOT / WS_ROOT))
cost_model = load_cost_model(ROOT, PATH_COSTS)
substeps_map = load_substeps_config(ROOT, PATH_SUBSTEPS)

# --- helpers ---

def scan_episodes():
    eps = []
    for p in list_episode_dirs(ROOT, WS_ROOT):
        info = parse_episode_info(p)
        raw_prev = read_status(p)
        computed = infer_stage_statuses(p, prev=raw_prev)
        eps.append({
            "Ep": f"Ep{info['ep']:02d}",
            "Range": f"{info['start']:04d}-{info['end']:04d}",
            "Current": computed,
            "_raw": info,
        })
    return eps


def stage_table(eps):
    rows = []
    for e in eps:
        stg = e["Current"]
        latest = None
        for sid in ["1", "1.5", "2", "3", "4", "5", "6", "7"]:
            s = stg.get(sid, "pending")
            if s in ("running", "error", "done"):
                latest = f"{sid}:{s}"
        rows.append({
            "Ep": e["Ep"],
            "Range": e["Range"],
            "1": with_emoji(stg.get("1", "pending")),
            "1.5": with_emoji(stg.get("1.5", "pending")),
            "2": with_emoji(stg.get("2", "pending")),
            "3": with_emoji(stg.get("3", "pending")),
            "4": with_emoji(stg.get("4", "pending")),
            "5": with_emoji(stg.get("5", "pending")),
            "6": with_emoji(stg.get("6", "pending")),
            "7": with_emoji(stg.get("7", "pending")),
            "Latest": latest or "—",
        })
    return pd.DataFrame(rows)


def compute_substeps_status(eps):
    """Return per-episode No.3–17 status using the same logic
    as the details panel (evaluate_substeps_debug)."""
    out = []
    for e in eps:
        info = e["_raw"]
        dbg = evaluate_substeps_debug(info["path"], substeps_map)
        status = {}
        for row in dbg:
            k = str(row.get("no"))
            if k and str(k).strip().isdigit():
                k = str(int(str(k)))
            status[k] = "done" if row.get("ok") else "pending"
        out.append({
            "Ep": e["Ep"],
            "Range": e["Range"],
            "status": status,
            "_path": str(info["path"]),
            "_rows": dbg,
        })
    return out

def substeps_table(eps):
    rows = []
    snapshot = compute_substeps_status(eps)
    for item in snapshot:
        row = {"Ep": item["Ep"], "Range": item["Range"]}
        for i in range(3, 18):
            k = str(i)
            row[k] = with_emoji(item["status"].get(k, "pending"))
        rows.append(row)
    return pd.DataFrame(rows)

def first_existing_path(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[0]) if paths else None

def read_json_file(path: Path | None):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def read_text_file(path: Path | None, default: str = "") -> str:
    if not path or not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return default

def save_text_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

def seconds_to_label(seconds_value) -> str:
    try:
        total = float(seconds_value)
    except Exception:
        return str(seconds_value)
    mins, secs = divmod(max(total, 0.0), 60)
    hours, mins = divmod(int(mins), 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:05.2f}"
    return f"{mins:02d}:{secs:05.2f}"

def parse_srt_timestamp_to_seconds(ts: str) -> float:
    hms, ms = ts.split(",")
    h, m, s = [int(x) for x in hms.split(":")]
    return h * 3600 + m * 60 + s + (int(ms) / 1000.0)

def parse_srt_entries(path: Path) -> list[Dict]:
    text = read_text_file(path, "")
    if not text.strip():
        return []
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    rows = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        idx_line = lines[0].strip()
        time_line = lines[1].strip()
        if "-->" not in time_line:
            continue
        start_ts, end_ts = [part.strip() for part in time_line.split("-->")]
        try:
            start_sec = parse_srt_timestamp_to_seconds(start_ts)
            end_sec = parse_srt_timestamp_to_seconds(end_ts)
        except Exception:
            continue
        rows.append({
            "index": idx_line,
            "start": start_sec,
            "end": end_sec,
            "start_label": start_ts.replace(",", "."),
            "end_label": end_ts.replace(",", "."),
            "content": "\n".join(lines[2:]).strip(),
        })
    return rows

def subtitles_for_scene(entries: list[Dict], start_sec, end_sec) -> list[Dict]:
    try:
        start_val = float(start_sec)
        end_val = float(end_sec)
    except Exception:
        return []
    return [row for row in entries if row["end"] > start_val and row["start"] < end_val]

def parse_tags_text(tags_text: str) -> list[str]:
    vals = []
    for raw in (tags_text or "").replace("\r", "\n").replace(",", "\n").split("\n"):
        item = raw.strip()
        if item:
            vals.append(item)
    return vals

def save_youtube_meta(ep_path: Path, title: str, description: str, tags_text: str, drive_link: str = "") -> Path:
    out_path = ep_path / "05_output" / "youtube_meta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title.strip(),
        "description": description,
        "tags": parse_tags_text(tags_text),
    }
    if drive_link.strip():
        payload["drive_link"] = drive_link.strip()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    return out_path

def storyboard_path_for(ep_path: Path) -> Path:
    return ep_path / "03_storyboards" / "storyboard.csv"

def read_storyboard_df(ep_path: Path) -> pd.DataFrame:
    p = storyboard_path_for(ep_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig").fillna("")
    return df

def flashcard_map_for(ep_path: Path) -> Dict[str, str]:
    folder = images_dir(ep_path) / "flashcards"
    items = {}
    if folder.exists():
        for p in sorted(folder.glob("*.*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                items[p.stem] = str(p.resolve()).replace("\\", "/")
    return items

def ai_scene_image_path(ep_path: Path, start_time) -> Path:
    start_token = str(start_time).strip()
    return images_dir(ep_path) / "ai_generated" / f"img_{start_token}.png"

def storyboard_scene_image_path(ep_path: Path, row: pd.Series | Dict) -> Path | None:
    source_type = str(row.get("source_type", "")).strip().upper()
    flashcards = flashcard_map_for(ep_path)
    if source_type == "FLASHCARD":
        word = str(row.get("flashcard_word", "")).strip()
        if word in flashcards:
            return Path(flashcards[word])
        custom_path = str(row.get("custom_image_path", "")).strip()
        return Path(custom_path) if custom_path else None

    custom_path = str(row.get("custom_image_path", "")).strip()
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    return ai_scene_image_path(ep_path, row.get("start_time", ""))

def normalize_storyboard_df(ep_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        "scene_id",
        "start_time",
        "end_time",
        "source_type",
        "flashcard_word",
        "custom_image_path",
        "image_prompt",
        "reason",
        "subtitle_reference",
    ]
    out = df.copy()
    for col in expected_cols:
        if col not in out.columns:
            out[col] = ""

    flashcards = flashcard_map_for(ep_path)
    out = out[expected_cols].fillna("")
    for idx in out.index:
        source_type = str(out.at[idx, "source_type"]).strip().upper() or "AI"
        out.at[idx, "source_type"] = source_type
        if source_type == "FLASHCARD":
            word = str(out.at[idx, "flashcard_word"]).strip()
            out.at[idx, "flashcard_word"] = word
            if word in flashcards:
                out.at[idx, "custom_image_path"] = flashcards[word]
            out.at[idx, "image_prompt"] = ""
    return out

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def file_to_data_uri(path: Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def build_episode_preview_payload(ep_path: Path, storyboard_df: pd.DataFrame, subtitle_entries: list[Dict]) -> Dict:
    image_cache: Dict[str, str] = {}
    scenes = []
    for _, row in storyboard_df.iterrows():
        image_path = storyboard_scene_image_path(ep_path, row)
        image_path_str = str(image_path) if image_path else ""
        if image_path_str not in image_cache:
            image_cache[image_path_str] = file_to_data_uri(image_path) if image_path_str else ""
        scenes.append({
            "scene_id": str(row.get("scene_id", "")),
            "start": safe_float(row.get("start_time", 0)),
            "end": safe_float(row.get("end_time", 0)),
            "source_type": str(row.get("source_type", "")).upper(),
            "flashcard_word": str(row.get("flashcard_word", "")),
            "reason": str(row.get("reason", "")),
            "image_prompt": str(row.get("image_prompt", "")),
            "subtitle_reference": str(row.get("subtitle_reference", "")),
            "image_path": image_path_str,
            "image_data_uri": image_cache[image_path_str],
        })
    subtitles = [{
        "index": str(row.get("index", "")),
        "start": safe_float(row.get("start", 0)),
        "end": safe_float(row.get("end", 0)),
        "content": str(row.get("content", "")),
    } for row in subtitle_entries]
    return {"scenes": scenes, "subtitles": subtitles}

def render_episode_preview_player(audio_path: Path, payload: Dict, height: int = 860) -> None:
    audio_data_uri = file_to_data_uri(audio_path)
    if not audio_data_uri:
        st.warning("找不到可播放的整集音訊。")
        return

    html = f"""
    <div id="cap-preview-root">
      <style>
        body {{
          margin: 0;
          font-family: 'Segoe UI', sans-serif;
          background: #f6f4ef;
          color: #1d1d1f;
        }}
        .cap-shell {{
          padding: 18px;
          display: grid;
          gap: 16px;
        }}
        .cap-card {{
          background: #fffdf8;
          border: 1px solid #d8cfbf;
          border-radius: 16px;
          box-shadow: 0 8px 24px rgba(0,0,0,0.06);
          overflow: hidden;
        }}
        .cap-main {{
          display: grid;
          grid-template-columns: minmax(320px, 1.35fr) minmax(300px, 1fr);
          gap: 16px;
        }}
        .cap-media {{
          padding: 16px;
        }}
        .cap-media img {{
          width: 100%;
          aspect-ratio: 16 / 9;
          object-fit: contain;
          background: #ebe4d6;
          border-radius: 14px;
          display: block;
        }}
        .cap-empty {{
          width: 100%;
          aspect-ratio: 16 / 9;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #ebe4d6;
          color: #7a6f5d;
          border-radius: 14px;
        }}
        .cap-side {{
          padding: 16px 16px 18px;
          display: grid;
          gap: 12px;
          align-content: start;
        }}
        .cap-audio {{
          width: 100%;
        }}
        .cap-meta {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }}
        .cap-chip {{
          padding: 10px 12px;
          border-radius: 12px;
          background: #f3ede2;
          border: 1px solid #e1d7c5;
        }}
        .cap-label {{
          font-size: 12px;
          color: #7b6f5d;
          margin-bottom: 4px;
        }}
        .cap-value {{
          font-size: 15px;
          font-weight: 600;
          white-space: pre-wrap;
          word-break: break-word;
        }}
        .cap-subtitle {{
          min-height: 120px;
          padding: 14px 16px;
          background: #1c1a17;
          color: #fffaf0;
          border-radius: 14px;
          line-height: 1.6;
          font-size: 22px;
          display: flex;
          align-items: center;
        }}
        .cap-hint {{
          color: #6a6052;
          font-size: 13px;
        }}
        .cap-timeline {{
          padding: 0 16px 16px;
          display: grid;
          gap: 10px;
        }}
        .cap-toolbar {{
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          align-items: center;
          padding: 16px 16px 0;
        }}
        .cap-toolbar button {{
          border: 1px solid #d7cdbd;
          background: #fff;
          color: #40382d;
          border-radius: 999px;
          padding: 8px 12px;
          cursor: pointer;
        }}
        .cap-toolbar button:hover {{
          background: #f5efe4;
        }}
        .cap-items {{
          max-height: 240px;
          overflow: auto;
          padding-right: 2px;
        }}
        .cap-item {{
          border: 1px solid #eadfcd;
          background: #fff;
          border-radius: 12px;
          padding: 10px 12px;
          cursor: pointer;
          margin-bottom: 8px;
        }}
        .cap-item.active {{
          border-color: #a56a1d;
          background: #fff2de;
        }}
        .cap-item-title {{
          font-weight: 700;
          margin-bottom: 4px;
        }}
        .cap-item-time {{
          color: #8a7a66;
          font-size: 12px;
          margin-bottom: 6px;
        }}
        .cap-item-text {{
          color: #40382d;
          white-space: pre-wrap;
          word-break: break-word;
        }}
        @media (max-width: 960px) {{
          .cap-main {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
      <div class="cap-shell">
        <div class="cap-card cap-main">
          <div class="cap-media">
            <img id="cap-scene-image" alt="scene image" />
            <div id="cap-scene-empty" class="cap-empty" style="display:none;">此時間點尚無圖片</div>
          </div>
          <div class="cap-side">
            <audio id="cap-audio" class="cap-audio" controls preload="metadata" src="{audio_data_uri}"></audio>
            <div class="cap-subtitle" id="cap-subtitle">按下播放後，這裡會同步顯示整集字幕。</div>
            <div class="cap-meta">
              <div class="cap-chip"><div class="cap-label">目前 Scene</div><div class="cap-value" id="cap-scene-id">-</div></div>
              <div class="cap-chip"><div class="cap-label">Scene 時間</div><div class="cap-value" id="cap-scene-time">-</div></div>
              <div class="cap-chip"><div class="cap-label">圖片來源</div><div class="cap-value" id="cap-source-type">-</div></div>
              <div class="cap-chip"><div class="cap-label">單字卡 / 圖片路徑</div><div class="cap-value" id="cap-image-ref">-</div></div>
            </div>
            <div class="cap-chip">
              <div class="cap-label">Reason</div>
              <div class="cap-value" id="cap-reason">-</div>
            </div>
            <div class="cap-chip">
              <div class="cap-label">Image Prompt</div>
              <div class="cap-value" id="cap-prompt">-</div>
            </div>
            <div class="cap-hint">整集播放時會依 storyboard.csv 切換圖片，依字幕時間軸切換字幕。下方可直接點 Scene 或字幕跳轉。</div>
          </div>
        </div>
        <div class="cap-card">
          <div class="cap-toolbar">
            <button type="button" id="cap-jump-prev-scene">上一個 Scene</button>
            <button type="button" id="cap-jump-next-scene">下一個 Scene</button>
            <button type="button" id="cap-jump-prev-sub">上一句字幕</button>
            <button type="button" id="cap-jump-next-sub">下一句字幕</button>
          </div>
          <div class="cap-timeline">
            <div>
              <div class="cap-label">Scene 時間軸</div>
              <div class="cap-items" id="cap-scene-list"></div>
            </div>
            <div>
              <div class="cap-label">字幕時間軸</div>
              <div class="cap-items" id="cap-subtitle-list"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script>
      const payload = {json.dumps(payload, ensure_ascii=False)};
      const scenes = payload.scenes || [];
      const subtitles = payload.subtitles || [];
      const audio = document.getElementById("cap-audio");
      const sceneImage = document.getElementById("cap-scene-image");
      const sceneEmpty = document.getElementById("cap-scene-empty");
      const subtitleBox = document.getElementById("cap-subtitle");
      const sceneIdEl = document.getElementById("cap-scene-id");
      const sceneTimeEl = document.getElementById("cap-scene-time");
      const sourceTypeEl = document.getElementById("cap-source-type");
      const imageRefEl = document.getElementById("cap-image-ref");
      const reasonEl = document.getElementById("cap-reason");
      const promptEl = document.getElementById("cap-prompt");
      const sceneList = document.getElementById("cap-scene-list");
      const subtitleList = document.getElementById("cap-subtitle-list");
      let activeSceneIndex = -1;
      let activeSubtitleIndex = -1;

      function fmtTime(sec) {{
        const total = Math.max(Number(sec || 0), 0);
        const hours = Math.floor(total / 3600);
        const mins = Math.floor((total % 3600) / 60);
        const secs = (total % 60).toFixed(2).padStart(5, "0");
        if (hours > 0) return `${{String(hours).padStart(2, "0")}}:${{String(mins).padStart(2, "0")}}:${{secs}}`;
        return `${{String(mins).padStart(2, "0")}}:${{secs}}`;
      }}

      function findActiveIndex(items, currentTime) {{
        if (!items.length) return -1;
        for (let i = 0; i < items.length; i += 1) {{
          const item = items[i];
          if (currentTime >= Number(item.start || 0) && currentTime < Number(item.end || 0)) return i;
        }}
        for (let i = items.length - 1; i >= 0; i -= 1) {{
          if (currentTime >= Number(items[i].start || 0)) return i;
        }}
        return -1;
      }}

      function setActive(listEl, selector, activeIndex) {{
        listEl.querySelectorAll(selector).forEach((node, idx) => {{
          node.classList.toggle("active", idx === activeIndex);
        }});
      }}

      function renderTimeline() {{
        sceneList.innerHTML = scenes.map((scene, idx) => `
          <div class="cap-item" data-scene-index="${{idx}}">
            <div class="cap-item-title">Scene ${{scene.scene_id || "-"}} | ${{scene.source_type || "-"}}</div>
            <div class="cap-item-time">${{fmtTime(scene.start)}} - ${{fmtTime(scene.end)}}</div>
            <div class="cap-item-text">${{scene.reason || scene.subtitle_reference || ""}}</div>
          </div>
        `).join("");
        subtitleList.innerHTML = subtitles.map((sub, idx) => `
          <div class="cap-item" data-sub-index="${{idx}}">
            <div class="cap-item-title">字幕 ${{sub.index || idx + 1}}</div>
            <div class="cap-item-time">${{fmtTime(sub.start)}} - ${{fmtTime(sub.end)}}</div>
            <div class="cap-item-text">${{sub.content || ""}}</div>
          </div>
        `).join("");

        sceneList.querySelectorAll("[data-scene-index]").forEach((node) => {{
          node.addEventListener("click", () => {{
            const idx = Number(node.getAttribute("data-scene-index"));
            audio.currentTime = Number(scenes[idx].start || 0);
            audio.play();
            syncPreview();
          }});
        }});
        subtitleList.querySelectorAll("[data-sub-index]").forEach((node) => {{
          node.addEventListener("click", () => {{
            const idx = Number(node.getAttribute("data-sub-index"));
            audio.currentTime = Number(subtitles[idx].start || 0);
            audio.play();
            syncPreview();
          }});
        }});
      }}

      function syncPreview() {{
        const currentTime = Number(audio.currentTime || 0);
        const sceneIdx = findActiveIndex(scenes, currentTime);
        const subtitleIdx = findActiveIndex(subtitles, currentTime);

        if (sceneIdx !== activeSceneIndex) {{
          activeSceneIndex = sceneIdx;
          const scene = scenes[sceneIdx];
          if (scene) {{
            sceneIdEl.textContent = scene.scene_id || "-";
            sceneTimeEl.textContent = `${{fmtTime(scene.start)}} - ${{fmtTime(scene.end)}}`;
            sourceTypeEl.textContent = scene.source_type || "-";
            imageRefEl.textContent = scene.source_type === "FLASHCARD"
              ? (scene.flashcard_word || scene.image_path || "-")
              : (scene.image_path || "-");
            reasonEl.textContent = scene.reason || "-";
            promptEl.textContent = scene.image_prompt || "-";
            if (scene.image_data_uri) {{
              sceneImage.src = scene.image_data_uri;
              sceneImage.style.display = "block";
              sceneEmpty.style.display = "none";
            }} else {{
              sceneImage.removeAttribute("src");
              sceneImage.style.display = "none";
              sceneEmpty.style.display = "flex";
            }}
          }} else {{
            sceneIdEl.textContent = "-";
            sceneTimeEl.textContent = "-";
            sourceTypeEl.textContent = "-";
            imageRefEl.textContent = "-";
            reasonEl.textContent = "-";
            promptEl.textContent = "-";
            sceneImage.removeAttribute("src");
            sceneImage.style.display = "none";
            sceneEmpty.style.display = "flex";
          }}
          setActive(sceneList, "[data-scene-index]", activeSceneIndex);
        }}

        if (subtitleIdx !== activeSubtitleIndex) {{
          activeSubtitleIndex = subtitleIdx;
          const subtitle = subtitles[subtitleIdx];
          subtitleBox.textContent = subtitle ? subtitle.content : "此時間點沒有字幕。";
          setActive(subtitleList, "[data-sub-index]", activeSubtitleIndex);
        }}
      }}

      function jumpToNeighbor(items, currentIndex, direction) {{
        if (!items.length) return;
        const nextIndex = Math.min(Math.max(currentIndex + direction, 0), items.length - 1);
        audio.currentTime = Number(items[nextIndex].start || 0);
        audio.play();
        syncPreview();
      }}

      document.getElementById("cap-jump-prev-scene").addEventListener("click", () => jumpToNeighbor(scenes, Math.max(activeSceneIndex, 0), -1));
      document.getElementById("cap-jump-next-scene").addEventListener("click", () => jumpToNeighbor(scenes, Math.max(activeSceneIndex, 0), 1));
      document.getElementById("cap-jump-prev-sub").addEventListener("click", () => jumpToNeighbor(subtitles, Math.max(activeSubtitleIndex, 0), -1));
      document.getElementById("cap-jump-next-sub").addEventListener("click", () => jumpToNeighbor(subtitles, Math.max(activeSubtitleIndex, 0), 1));

      renderTimeline();
      audio.addEventListener("timeupdate", syncPreview);
      audio.addEventListener("seeked", syncPreview);
      audio.addEventListener("loadedmetadata", syncPreview);
      syncPreview();
    </script>
    """
    components.html(html, height=height, scrolling=True)

def save_storyboard_df(ep_path: Path, df: pd.DataFrame) -> Path:
    p = storyboard_path_for(ep_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_storyboard_df(ep_path, df)
    normalized.to_csv(p, index=False, encoding="utf-8-sig")
    return p

def publisher_task_state(info: Dict, task_name: str) -> Dict:
    return runner.get_substep_state(info, f"pub_{task_name}")

def start_publisher_task(info: Dict, task_name: str, step_def: Dict) -> Dict:
    return runner.start_substep(info, f"pub_{task_name}", step_def)

# convenience
STAGE_IDS = ["1","1.5","2","3","4","5","6","7"]

# --- UI ---
if section == "📊 Dashboard":
    st.header("專案與進度總覽")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先執行 setup_cap_2000_project.py 建立結構。")
    else:
        # 主表：顯示 15 個子步驟（No.3–17）
        st.caption(f"規則載入：{len(substeps_map)} 項")
        st.dataframe(substeps_table(eps), use_container_width=True)
        # 規則 (debug)：顯示從 YAML 載入的子步驟鍵值
        with st.expander("子步驟規則 (debug)", expanded=False):
            st.write([{ "no": getattr(s, "no", None), "key": getattr(s, "key", None), "name": getattr(s, "name", None)} for s in substeps_map])

        # 附表：原 1–7 階段狀態
        with st.expander("階段總覽 (1–7)", expanded=False):
            st.dataframe(stage_table(eps), use_container_width=True)

        # 手動確認控制（No.7 / No.11 / No.13）
        with st.expander("手動確認工具 (僅 7 / 11 / 13)", expanded=False):
            eps_map = {f"Ep{e['_raw']['ep']:02d}": e for e in eps}
            ep_choice = st.selectbox("選擇集數以標記手動步驟", list(eps_map.keys()), key="manual_ep")
            target_m = eps_map[ep_choice]
            ep_info = target_m["_raw"]
            colA, colB, colC, colD = st.columns([1, 1, 1, 2])
            with colA:
                s7_cur = has_manual_marker(ep_info["path"], "7")
                s7_new = st.checkbox("Step 7 完成 (NotebookLM)", value=s7_cur, key="manual7")
            with colB:
                s11_cur = has_manual_marker(ep_info["path"], "11")
                s11_new = st.checkbox("Step 11 完成 (人工字幕)", value=s11_cur, key="manual11")
            with colC:
                s13_cur = has_manual_marker(ep_info["path"], "13")
                s13_new = st.checkbox("Step 13 完成 (人工微調)", value=s13_cur, key="manual13")
            with colD:
                if st.button("更新手動標記", key="manual_update"):
                    set_manual_marker(ep_info["path"], "7", s7_new)
                    set_manual_marker(ep_info["path"], "11", s11_new)
                    set_manual_marker(ep_info["path"], "13", s13_new)
                    st.success("已更新手動標記，請重新展開刷新表格或切換頁籤。")

        # 檢核詳情 (可選集數)
        with st.expander("檢核詳情 (No.3–17)", expanded=False):
            ep_opts = {f"{e['Ep']} ({e['Range']})": e for e in eps}
            keys_list = list(ep_opts.keys())
            default_idx = 0
            for idx, k in enumerate(keys_list):
                if k.startswith("Ep04 "):
                    default_idx = idx
                    break
            choice = st.selectbox("選擇要檢核的集數", keys_list, index=default_idx, key="detail_ep")
            target = ep_opts.get(choice)
            if target:
                info = target["_raw"]
                st.caption(f"Episode Path: {info['path']}")
                dbg = evaluate_substeps_debug(info["path"], substeps_map)
                rows = []
                for item in dbg:
                    pats = item.get("patterns", [])
                    nonempty_patterns = [p for p in pats if p.get("nonempty")]
                    rule = (item.get("rule") or "any").lower()
                    # Require Nonempty: 是否有任何 pattern 標註 nonempty:true
                    nonempty_required = len(nonempty_patterns) > 0
                    # Nonempty OK: 與子步驟 Rule 對齊
                    if not nonempty_patterns:
                        nonempty_ok = True
                    elif rule == "any":
                        nonempty_ok = any(p.get("satisfied") for p in nonempty_patterns)
                    else:
                        nonempty_ok = all(p.get("satisfied") for p in nonempty_patterns)
                    matched_count = sum(p.get("matched_count",0) for p in pats)
                    matched_paths = []
                    for p in pats:
                        matched_paths.extend(p.get("matched", []))
                    matched_paths = matched_paths[:10]
                    rows.append({
                        "No": item.get("no"),
                        "Name": item.get("name"),
                        "Rule": item.get("rule"),
                        "Step OK": "🟢" if item.get("ok") else "⚪",
                        "Require Nonempty": "✅" if nonempty_required else "",
                        "Nonempty OK": "✅" if nonempty_ok else "❌",
                        "Patterns": ", ".join([p.get("pattern","") for p in pats]),
                        "Matched Count": matched_count,
                        "Matched (up to 10)": "\n".join(matched_paths),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("找不到可檢核的集數。")

# --- Pipeline Manager ---
elif section == "⚙️ Pipeline Manager":
    st.header("工作流引擎與日誌 (Pipeline Manager)")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先建立或同步集數資料夾。")
    else:
        ep_map = {f"Ep{e['_raw']['ep']:02d} ({e['Range']})": e for e in eps}
        ep_choice = st.selectbox("選擇集數", list(ep_map.keys()), key="pm_ep")
        target = ep_map[ep_choice]
        info = target["_raw"]
        st.caption(f"Episode Path: {info['path']}")

        # 對應關係 (依 profile)
        def get_mapping(profile_id: str):
            mapping = {
                '1':  [3,4,5,6,7],
                '1.5':[8],
                '2':  [9],
                '3':  [10,11],
                '4':  [12,13,14],
                '5':  [15],
                '6':  [16],
                '7':  [17],
            }
            sub2stage = {}
            exec_mode = {}
            for sid, lst in mapping.items():
                for n in lst:
                    sub2stage[str(n)] = sid
            manual_only = {'7','11','13'}
            for n in range(3,18):
                ns = str(n)
                if ns in manual_only:
                    exec_mode[ns] = 'manual'
                elif ns in sub2stage:
                    exec_mode[ns] = 'auto'
                else:
                    exec_mode[ns] = 'na'
            return mapping, sub2stage, exec_mode
        stage2subs, sub2stage, exec_mode = get_mapping(st.session_state["profile_id"])
        stage_names = {str(s.get('id')): s.get('name','') for s in (stage_cfg.get('stages') or [])}

        tab_stage, tab_sub = st.tabs(["階段控制 (1–7)", "子步驟 (3–17)"])

        with tab_stage:
            cur = target["Current"]
            cols = st.columns(8)
            for i, sid in enumerate(STAGE_IDS):
                with cols[i]:
                    st.metric(label=f"Stage {sid}", value=cur.get(sid, "pending"))
            st.divider()
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                sid = st.selectbox("選擇要執行的階段", STAGE_IDS, key="pm_stage")
            with c2:
                run_btn = st.button("執行所選階段", key="pm_run")
            with c3:
                st.write("「執行」後可於下方日誌檢視輸出。")
            if run_btn:
                status = runner.run_stage(info, sid)
                st.success(f"已執行 Stage {sid}，目前狀態：{status['stages'].get(sid)}")
            with st.expander("終端機日誌面板 (Log Viewer)", expanded=False):
                log_tabs = st.tabs([f"Stage {sid}" for sid in STAGE_IDS])
                for sid, tab in zip(STAGE_IDS, log_tabs):
                    with tab:
                        lp = log_file_for_stage(info["path"], sid)
                        if lp.exists():
                            st.download_button("下載 log", data=lp.read_bytes(), file_name=lp.name, key=f"dl_{sid}")
                            st.code(lp.read_text(encoding="utf-8", errors="ignore")[-5000:])
                        else:
                            st.caption("尚無此階段的日誌。")

        with tab_sub:
            st.caption("在此直接執行 3–16（第17發布請到 Studio & Publisher）。")
            auto_refresh_sub = st.toggle(
                "自動刷新執行中的子步驟 Log",
                value=True,
                key="pm_sub_auto_refresh",
                help="有子步驟執行中時，每 2 秒自動更新一次狀態與 Log。",
            )
            any_running_sub = any(bool(runner.get_substep_state(info, str(n)).get("running")) for n in range(3, 18))
            refresh_interval = 2 if (auto_refresh_sub and any_running_sub) else None

            @st.fragment(run_every=refresh_interval)
            def render_pipeline_substeps():
                st.caption("只會刷新此面板，不會切回 Dashboard。")
                st.button("只刷新子步驟狀態", key="refresh_substeps_fragment")
                dbg = evaluate_substeps_debug(info["path"], substeps_map)
                by_no = {str(r.get('no')): r for r in dbg}
                for n in range(3,18):
                    ns = str(n)
                    row = by_no.get(ns, {})
                    pats = row.get('patterns', [])
                    nonempty_patterns = [p for p in pats if p.get('nonempty')]
                    rule = (row.get('rule') or 'any').lower()
                    nonempty_required = len(nonempty_patterns) > 0
                    if not nonempty_patterns:
                        nonempty_ok = True
                    elif rule == 'any':
                        nonempty_ok = any(p.get('satisfied') for p in nonempty_patterns)
                    else:
                        nonempty_ok = all(p.get('satisfied') for p in nonempty_patterns)
                    stage_for = sub2stage.get(ns)
                    run_spec = row.get('run') or {}
                    sub_state = runner.get_substep_state(info, ns)
                    log_path = Path(sub_state.get("log", ""))
                    is_running = bool(sub_state.get("running"))
                    cols = st.columns([0.6, 2.2, 2.0, 1.2, 2.6])
                    with cols[0]:
                        st.markdown(f"**No.{ns}**")
                    with cols[1]:
                        st.write(row.get('name',''))
                        if pats:
                            st.caption("Patterns: " + ", ".join([p.get('pattern','') for p in pats]))
                    with cols[2]:
                        if stage_for:
                            st.write(f"Stage {stage_for} {stage_names.get(stage_for,'')}")
                        else:
                            st.write('—')
                        st.caption("Step OK: " + ("🟢" if row.get('ok') else "⚪"))
                    with cols[3]:
                        st.write("Nonempty OK: " + ("✅" if nonempty_ok else ("❌" if nonempty_required else "")))
                    with cols[4]:
                        if ns == '17':
                            if st.button("前往 Studio & Publisher", key=f"goto_pub_{ns}"):
                                st.session_state['nav_radio'] = "🎬 Studio & Publisher"
                                st.query_params["section"] = "🎬 Studio & Publisher"
                                st.rerun()
                        elif run_spec.get('script'):
                            step_def = {
                                'name': row.get('name',''),
                                'type': run_spec.get('type','python'),
                                'script': run_spec.get('script'),
                                'args': run_spec.get('args') or [],
                            }
                            show_cmd = f"{step_def['type']} {step_def['script']} {' '.join(step_def['args'])}"
                            if st.button('執行中...' if is_running else '執行子步驟', key=f"run_sub_{ns}", disabled=is_running):
                                res = runner.start_substep(info, ns, step_def)
                                if res.get("ok"):
                                    st.success(f"子步驟 No.{ns} 已啟動。log: {res.get('log')}")
                                else:
                                    st.error(f"子步驟 No.{ns} 啟動失敗：{res.get('message')}")
                            st.caption("將執行：" + show_cmd)
                            if is_running:
                                st.caption(f"PID {sub_state.get('pid')} 執行中")
                        elif exec_mode.get(ns) == 'manual' and ns in ('7','11','13'):
                            done = has_manual_marker(info['path'], ns)
                            marker_path = manual_marker_path(info['path'], ns)
                            marker_sig = f"{int(marker_path.stat().st_mtime)}_{int(done)}" if marker_path.exists() else "0_0"
                            st.caption("目前狀態：" + ("已標記" if done else "未標記"))
                            if ns == "13":
                                if st.button("前往分鏡編輯", key="goto_storyboard_13"):
                                    st.session_state['nav_radio'] = "🎬 Studio & Publisher"
                                    st.query_params["section"] = "🎬 Studio & Publisher"
                                    st.session_state["studio_open_tab"] = "storyboard"
                                    st.rerun()
                            with st.form(key=f"man_form_{ns}_{marker_sig}"):
                                new_val = st.checkbox("標記完成", value=done, key=f"man_{ns}_{marker_sig}")
                                submitted = st.form_submit_button("更新標記")
                            if submitted:
                                set_manual_marker(info['path'], ns, new_val)
                                st.success("已更新手動標記。")
                                st.rerun()
                        else:
                            st.caption("無對應自動化")
                    if log_path.exists():
                        with st.expander("執行 Log" if is_running else "最近 Log", expanded=is_running):
                            st.download_button("下載 log", data=log_path.read_bytes(), file_name=log_path.name, key=f"dl_sub_{ns}")
                            st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-5000:])
                            if is_running:
                                st.caption("執行中。此面板會自動刷新；也可按上方按鈕只刷新子步驟狀態。")
                    st.divider()

            render_pipeline_substeps()
            st.caption("說明：\n- 手動步驟（7,11,13）可在此切換完成標記。\n- 其它子步驟按鈕會直接執行對應的 Python 腳本。\n- 第17步 (發布) 請到 Studio & Publisher。")

        with st.expander("1–7 與 3–17 對應關係", expanded=False):
            substep_name_map = {str(getattr(s, "no", "")): getattr(s, "name", "") for s in substeps_map}
            map_rows = []
            for sid in STAGE_IDS:
                subs = stage2subs.get(sid, [])
                chinese = stage_names.get(sid,'')
                subs_text = ", ".join([f"{x} " + substep_name_map.get(str(x), "") for x in subs])
                map_rows.append({"Stage": f"{sid} {chinese}", "Substeps": subs_text})
            st.table(pd.DataFrame(map_rows))# --- FinOps ---
elif section == "💰 FinOps Monitor":
    st.header("成本與配額控管 (FinOps Monitor)")
    st.write("此區塊保留，未變更先前行為。")

# --- Studio & Publisher ---
elif section == "🎬 Studio & Publisher":
    st.header("影音預覽與發布中樞 (Studio & Publisher)")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先建立或同步集數資料夾。")
    else:
        ep_map = {f"Ep{e['_raw']['ep']:02d} ({e['Range']})": e for e in eps}
        studio_choice = st.selectbox("選擇集數", list(ep_map.keys()), key="studio_ep")
        target = ep_map[studio_choice]
        info = target["_raw"]
        ep_path = info["path"]
        audio_path = audio_merged_path(ep_path)
        video_path = video_output_path(ep_path)
        cover_path = images_dir(ep_path) / "cover.png"
        subtitle_path = subtitles_fixed_path(ep_path)
        storyboard_path = storyboard_csv_path(ep_path)
        inputs_path = inputs_txt_path(ep_path)
        meta_path = first_existing_path(youtube_meta_paths(ep_path))
        meta_data = read_json_file(meta_path) or {}
        upload_record_path = ep_path / "05_output" / "upload_record.json"
        upload_record = read_json_file(upload_record_path) or {}
        meta_state = publisher_task_state(info, "meta")
        upload_state = publisher_task_state(info, "upload")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("影片", "就緒" if video_path.exists() else "缺少")
        c2.metric("封面", "就緒" if cover_path.exists() else "缺少")
        c3.metric("字幕", "就緒" if subtitle_path.exists() else "缺少")
        c4.metric("Metadata", "就緒" if meta_path and meta_path.exists() else "缺少")
        c5.metric("最近上傳", upload_record.get("status", "尚無"))
        st.caption(f"Episode Path: {ep_path}")

        tab_preview, tab_prerender, tab_storyboard, tab_meta, tab_publish, tab_logs = st.tabs(["預覽", "預渲染預覽", "No.13 分鏡", "Metadata", "發布", "Log"])
        if st.session_state.pop("studio_open_tab", None) == "storyboard":
            st.info("已切換到 Studio & Publisher，請使用「No.13 分鏡」頁籤進行修改。")

        with tab_preview:
            pv1, pv2 = st.columns([1, 1.6])
            with pv1:
                st.subheader("封面")
                if cover_path.exists():
                    st.image(str(cover_path), use_container_width=True)
                else:
                    st.caption("尚未找到封面圖。")
            with pv2:
                st.subheader("影片")
                if video_path.exists():
                    st.caption(f"{video_path.name} | {video_path.stat().st_size / (1024 * 1024):.1f} MB")
                    st.video(str(video_path))
                else:
                    st.caption("尚未找到 final_video.mp4。")
            st.divider()
            a1, a2 = st.columns([1, 1])
            with a1:
                st.subheader("字幕全文與編輯")
                st.caption(f"目標檔案：{subtitle_path}")
                subtitle_text = read_text_file(subtitle_path, "")
                subtitle_form_key = f"studio_subtitle_form_{info['ep']}_{st.session_state['profile_id']}"
                with st.form(subtitle_form_key):
                    subtitle_val = st.text_area(
                        "字幕內容",
                        value=subtitle_text,
                        height=520,
                    )
                    save_subtitle = st.form_submit_button("儲存字幕")
                if save_subtitle:
                    saved_path = save_text_file(subtitle_path, subtitle_val)
                    st.success(f"已儲存字幕：{saved_path}")
                    st.rerun()
                if subtitle_path.exists():
                    st.download_button(
                        "下載字幕",
                        data=subtitle_path.read_bytes(),
                        file_name=subtitle_path.name,
                        key=f"dl_subtitle_{info['ep']}_{st.session_state['profile_id']}",
                    )
                else:
                    st.caption("目前尚無校對字幕；可直接在上方貼上內容後儲存建立。")
            with a2:
                st.subheader("Metadata 摘要")
                if meta_data:
                    st.text_input("標題", value=meta_data.get("title", ""), key=f"preview_title_{info['ep']}", disabled=True)
                    st.text_area("描述", value=meta_data.get("description", ""), height=220, key=f"preview_desc_{info['ep']}", disabled=True)
                    st.text_area(
                        "Tags",
                        value="\n".join(meta_data.get("tags", [])),
                        height=120,
                        key=f"preview_tags_{info['ep']}",
                        disabled=True,
                    )
                else:
                    st.caption("尚未找到 youtube_meta.json。")

        with tab_prerender:
            st.subheader("整集預覽")
            ready_audio = audio_path.exists()
            ready_subtitle = subtitle_path.exists()
            ready_storyboard = storyboard_path.exists()
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("語音", "就緒" if ready_audio else "缺少")
            pc2.metric("字幕", "就緒" if ready_subtitle else "缺少")
            pc3.metric("分鏡", "就緒" if ready_storyboard else "缺少")
            if not (ready_audio and ready_subtitle and ready_storyboard):
                st.warning("整集預覽需要同時具備語音、字幕與分鏡表。")
            else:
                preview_storyboard_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
                subtitle_entries = parse_srt_entries(subtitle_path)
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric("Scene 數", len(preview_storyboard_df))
                pr2.metric("字幕數", len(subtitle_entries))
                pr3.metric(
                    "預覽時長",
                    seconds_to_label(
                        max(
                            [safe_float(row.get("end_time", 0)) for _, row in preview_storyboard_df.iterrows()] +
                            [safe_float(row.get("end", 0)) for row in subtitle_entries] +
                            [0.0]
                        )
                    ),
                )

                preview_payload = build_episode_preview_payload(ep_path, preview_storyboard_df, subtitle_entries)
                render_episode_preview_player(audio_path, preview_payload)

                with st.expander("完整字幕時間軸", expanded=False):
                    if subtitle_entries:
                        full_subtitle_df = pd.DataFrame([
                            {
                                "No": row["index"],
                                "Start": row["start_label"],
                                "End": row["end_label"],
                                "Content": row["content"],
                            }
                            for row in subtitle_entries
                        ])
                        st.dataframe(full_subtitle_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("目前沒有可解析的字幕。")

                with st.expander("完整分鏡表", expanded=False):
                    st.dataframe(preview_storyboard_df, use_container_width=True, hide_index=True)

        with tab_storyboard:
            st.subheader("No.13 修改分鏡 / 替換單字卡")
            storyboard_df = read_storyboard_df(ep_path)
            flashcards = flashcard_map_for(ep_path)
            if storyboard_df.empty:
                st.warning(f"找不到分鏡表：{storyboard_path_for(ep_path)}")
            else:
                storyboard_df = normalize_storyboard_df(ep_path, storyboard_df)
                done13 = has_manual_marker(ep_path, "13")
                top1, top2, top3 = st.columns([1.2, 1.2, 4])
                with top1:
                    st.write("No.13 狀態：" + ("✅ 已完成" if done13 else "⚪ 未完成"))
                with top2:
                    with st.form(key=f"storyboard_mark_13_{info['ep']}_{int(done13)}"):
                        mark13 = st.checkbox("標記 No.13 完成", value=done13, key=f"storyboard_mark13_{info['ep']}_{int(done13)}")
                        submit_mark13 = st.form_submit_button("更新 No.13 標記")
                    if submit_mark13:
                        set_manual_marker(ep_path, "13", mark13)
                        st.success("已更新 No.13 標記。")
                        st.rerun()
                with top3:
                    st.caption(f"分鏡檔：{storyboard_path_for(ep_path)}")

                scene_labels = []
                for _, r in storyboard_df.iterrows():
                    sid = str(r.get("scene_id", ""))
                    scene_labels.append(f"Scene {sid} | {str(r.get('source_type', '')).upper()} | {str(r.get('reason', ''))[:40]}")
                selected_scene_label = st.selectbox("選擇要修改的 Scene", scene_labels, key=f"storyboard_scene_{info['ep']}_{st.session_state['profile_id']}")
                selected_idx = scene_labels.index(selected_scene_label)
                selected_row = storyboard_df.iloc[selected_idx].copy()
                selected_scene_id = selected_row.get("scene_id")
                current_word = str(selected_row.get("flashcard_word", "")).strip()
                selected_scene_task = publisher_task_state(info, f"sceneimg_{selected_scene_id}")
                current_image_path = str(storyboard_scene_image_path(ep_path, selected_row) or "").strip()
                storyboard_auto_refresh = st.toggle(
                    "自動刷新單張重產圖 Log / 圖片",
                    value=True,
                    key=f"storyboard_auto_refresh_{info['ep']}_{st.session_state['profile_id']}",
                    help="Scene 單獨重產時，每 2 秒自動更新 log 與圖片。",
                )

                editor_left, editor_right = st.columns([1.1, 1.4])
                with editor_left:
                    st.caption(f"Scene ID: {selected_scene_id} | {selected_row.get('start_time')}s - {selected_row.get('end_time')}s")
                    st.caption(f"目前圖片路徑：{current_image_path or '無'}")
                    if current_image_path and Path(current_image_path).exists():
                        st.image(current_image_path, use_container_width=True)
                    else:
                        st.caption("目前沒有可預覽的圖片。")

                with editor_right:
                    source_key = f"storyboard_source_{info['ep']}_{selected_scene_id}"
                    flashcard_key = f"storyboard_flashcard_{info['ep']}_{selected_scene_id}"
                    custom_image_key = f"storyboard_custom_image_{info['ep']}_{selected_scene_id}"
                    prompt_key = f"storyboard_prompt_{info['ep']}_{selected_scene_id}"
                    reason_key = f"storyboard_reason_{info['ep']}_{selected_scene_id}"
                    if source_key not in st.session_state:
                        st.session_state[source_key] = str(selected_row.get("source_type", "")).upper() or "AI"
                    if flashcard_key not in st.session_state:
                        st.session_state[flashcard_key] = current_word if current_word in flashcards else ""
                    if custom_image_key not in st.session_state:
                        st.session_state[custom_image_key] = str(selected_row.get("custom_image_path", "")).strip()
                    if prompt_key not in st.session_state:
                        st.session_state[prompt_key] = str(selected_row.get("image_prompt", ""))
                    if reason_key not in st.session_state:
                        st.session_state[reason_key] = str(selected_row.get("reason", ""))

                    source_type_val = st.selectbox(
                        "source_type",
                        ["FLASHCARD", "AI"],
                        key=source_key,
                    )
                    flashcard_options = [""] + sorted(flashcards.keys())
                    if st.session_state[flashcard_key] not in flashcard_options:
                        st.session_state[flashcard_key] = ""
                    flashcard_word_val = st.selectbox(
                        "替換單字卡",
                        flashcard_options,
                        key=flashcard_key,
                        disabled=(source_type_val != "FLASHCARD"),
                    )
                    custom_image_val = st.text_input(
                        "自訂圖片路徑",
                        key=custom_image_key,
                        disabled=(source_type_val == "FLASHCARD"),
                    )
                    image_prompt_val = st.text_area(
                        "image_prompt",
                        height=140,
                        key=prompt_key,
                        disabled=(source_type_val == "FLASHCARD"),
                    )
                    reason_val = st.text_area("reason", height=100, key=reason_key)

                    action1, action2 = st.columns([1, 1.2])
                    with action1:
                        apply_scene = st.button("儲存此 Scene", key=f"storyboard_save_scene_{info['ep']}_{selected_scene_id}")
                    with action2:
                        regen_scene = st.button(
                            "重產此 AI 分鏡圖",
                            key=f"storyboard_regen_scene_{info['ep']}_{selected_scene_id}",
                            disabled=(source_type_val != "AI") or bool(selected_scene_task.get("running")),
                        )

                    if apply_scene:
                        latest_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
                        match_idx = latest_df.index[latest_df["scene_id"].astype(str) == str(selected_scene_id)]
                        if len(match_idx) == 0:
                            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                        else:
                            idx = match_idx[0]
                            latest_df.at[idx, "source_type"] = source_type_val
                            latest_df.at[idx, "reason"] = reason_val
                            if source_type_val == "FLASHCARD":
                                latest_df.at[idx, "flashcard_word"] = flashcard_word_val
                                latest_df.at[idx, "custom_image_path"] = flashcards.get(flashcard_word_val, "")
                                latest_df.at[idx, "image_prompt"] = ""
                            else:
                                latest_df.at[idx, "flashcard_word"] = ""
                                latest_df.at[idx, "custom_image_path"] = custom_image_val.strip()
                                latest_df.at[idx, "image_prompt"] = image_prompt_val
                            saved_storyboard = save_storyboard_df(ep_path, latest_df)
                            st.success(f"已儲存 Scene {selected_scene_id}：{saved_storyboard}")
                            st.rerun()
                    if regen_scene:
                        latest_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
                        match_idx = latest_df.index[latest_df["scene_id"].astype(str) == str(selected_scene_id)]
                        if len(match_idx) == 0:
                            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                        else:
                            idx = match_idx[0]
                            latest_df.at[idx, "source_type"] = "AI"
                            latest_df.at[idx, "flashcard_word"] = ""
                            latest_df.at[idx, "custom_image_path"] = st.session_state[custom_image_key].strip()
                            latest_df.at[idx, "image_prompt"] = st.session_state[prompt_key]
                            latest_df.at[idx, "reason"] = st.session_state[reason_key]
                            save_storyboard_df(ep_path, latest_df)
                            res = start_publisher_task(
                                info,
                                scene_task_name,
                                {
                                    "name": f"重產 Scene {selected_scene_id} 圖片",
                                    "type": "python",
                                    "script": "scripts/gen_matched_preview_v4.py",
                                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id)],
                                },
                            )
                            if res.get("ok"):
                                st.success(f"Scene {selected_scene_id} 重產任務已啟動。log: {res.get('log')}")
                            else:
                                st.error(f"Scene {selected_scene_id} 重產任務啟動失敗：{res.get('message')}")
                            st.rerun()

                scene_refresh_interval = 2 if (storyboard_auto_refresh and bool(selected_scene_task.get("running"))) else None

                @st.fragment(run_every=scene_refresh_interval)
                def render_storyboard_scene_status():
                    latest_storyboard_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
                    latest_match = latest_storyboard_df[latest_storyboard_df["scene_id"].astype(str) == str(selected_scene_id)]
                    latest_row = latest_match.iloc[0].copy() if not latest_match.empty else selected_row.copy()
                    latest_scene_state = publisher_task_state(info, f"sceneimg_{selected_scene_id}")
                    latest_scene_log = Path(latest_scene_state.get("log", ""))
                    latest_image_path = str(storyboard_scene_image_path(ep_path, latest_row) or "").strip()

                    status_col, log_col = st.columns([1, 1.5])
                    with status_col:
                        st.caption(f"最新圖片路徑：{latest_image_path or '無'}")
                        st.button("只刷新此 Scene 狀態", key=f"refresh_scene_status_{info['ep']}_{selected_scene_id}")
                        if latest_scene_state.get("running"):
                            st.info(f"Scene {selected_scene_id} 產圖中，PID {latest_scene_state.get('pid')}")
                        else:
                            st.caption("目前未在重產。")
                    with log_col:
                        if latest_scene_log.exists():
                            with st.expander("此 Scene 產圖 Log", expanded=bool(latest_scene_state.get("running"))):
                                st.code(latest_scene_log.read_text(encoding="utf-8", errors="ignore")[-4000:])
                                if latest_scene_state.get("running"):
                                    st.caption("執行中。此區會自動刷新。")
                    st.text_area(
                        "字幕參考",
                        value=str(latest_row.get("subtitle_reference", "")),
                        height=220,
                        key=f"storyboard_subref_{info['ep']}_{selected_scene_id}",
                        disabled=True,
                    )

                    st.divider()
                    st.subheader("每個分鏡的圖片總覽")
                    gallery_rows = [latest_storyboard_df.iloc[i:i + 3] for i in range(0, len(latest_storyboard_df), 3)]
                    for row_group in gallery_rows:
                        gallery_cols = st.columns(3)
                        for col_idx, (_, row) in enumerate(row_group.iterrows()):
                            img_path = storyboard_scene_image_path(ep_path, row)
                            with gallery_cols[col_idx]:
                                with st.container(border=True):
                                    st.caption(
                                        f"Scene {row.get('scene_id')} | {str(row.get('source_type', '')).upper()} | "
                                        f"{row.get('start_time')} - {row.get('end_time')}"
                                    )
                                    if img_path and img_path.exists():
                                        st.image(str(img_path), use_container_width=True)
                                    else:
                                        st.markdown(
                                            """
                                            <div style="
                                                height: 180px;
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                border: 1px dashed #999;
                                                border-radius: 8px;
                                                color: #666;
                                                background: rgba(127,127,127,0.06);
                                            ">
                                                尚無圖片
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    st.caption(str(row.get("reason", ""))[:80])

                render_storyboard_scene_status()

                st.divider()
                st.caption("整張分鏡表快速編輯")
                editable_df = storyboard_df.copy()
                edited_storyboard_df = st.data_editor(
                    editable_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    disabled=["scene_id", "subtitle_reference"],
                    column_config={
                        "scene_id": st.column_config.NumberColumn(),
                        "start_time": st.column_config.NumberColumn(format="%.2f"),
                        "end_time": st.column_config.NumberColumn(format="%.2f"),
                        "source_type": st.column_config.SelectboxColumn(options=["AI", "FLASHCARD"]),
                        "flashcard_word": st.column_config.TextColumn(),
                        "custom_image_path": st.column_config.TextColumn(width="large"),
                        "image_prompt": st.column_config.TextColumn(width="large"),
                        "reason": st.column_config.TextColumn(width="medium"),
                        "subtitle_reference": st.column_config.TextColumn(width="large"),
                    },
                    key=f"storyboard_editor_{info['ep']}_{st.session_state['profile_id']}",
                )
                if st.button("儲存整張分鏡表", key=f"save_storyboard_all_{info['ep']}_{st.session_state['profile_id']}"):
                    saved_storyboard = save_storyboard_df(ep_path, edited_storyboard_df)
                    st.success(f"已儲存分鏡表：{saved_storyboard}")
                    st.rerun()

        with tab_meta:
            st.subheader("YouTube Metadata")
            if st.button("重新整理 Metadata / Log", key="studio_refresh_meta"):
                st.rerun()
            if meta_state.get("running"):
                st.info(f"Metadata 生成中，PID {meta_state.get('pid')}")
            meta_key = f"studio_meta_form_ep_{info['ep']}_{st.session_state['profile_id']}"
            with st.form(meta_key):
                title_val = st.text_input("標題", value=meta_data.get("title", ""))
                desc_val = st.text_area("描述", value=meta_data.get("description", ""), height=260)
                tags_val = st.text_area("Tags", value="\n".join(meta_data.get("tags", [])), height=160)
                drive_val = st.text_input("Drive 連結", value=meta_data.get("drive_link", ""))
                save_meta = st.form_submit_button("儲存 Metadata")
            if save_meta:
                saved_path = save_youtube_meta(ep_path, title_val, desc_val, tags_val, drive_val)
                st.success(f"已儲存 Metadata：{saved_path}")
                st.rerun()

            meta_step = {
                "name": "產生 YouTube Metadata",
                "type": "python",
                "script": "scripts/generate_youtube_meta.py",
                "args": ["--ep", str(info["ep"])],
            }
            if st.button("產生 / 重產 Metadata", key="studio_run_meta", disabled=bool(meta_state.get("running"))):
                res = start_publisher_task(info, "meta", meta_step)
                st.session_state["studio_notice"] = {
                    "task": "meta",
                    "ok": bool(res.get("ok")),
                    "message": res.get("message", ""),
                    "log": res.get("log", ""),
                }
                st.rerun()

            meta_log = Path(meta_state.get("log", ""))
            notice = st.session_state.get("studio_notice")
            if isinstance(notice, dict) and notice.get("task") == "meta":
                if notice.get("ok"):
                    st.success(f"Metadata 任務已啟動。log: {notice.get('log')}")
                else:
                    st.error(f"Metadata 任務啟動失敗：{notice.get('message')}")
                st.session_state.pop("studio_notice", None)
            if meta_log.exists():
                with st.expander("Metadata Log", expanded=bool(meta_state.get("running"))):
                    st.download_button("下載 Metadata log", data=meta_log.read_bytes(), file_name=meta_log.name, key="dl_studio_meta_log")
                    st.code(meta_log.read_text(encoding="utf-8", errors="ignore")[-5000:])

        with tab_publish:
            st.subheader("YouTube 上傳")
            ready_to_upload = video_path.exists() and meta_path and meta_path.exists()
            if not ready_to_upload:
                st.warning("上傳前至少需要 final_video.mp4 與 youtube_meta.json。")
            if not subtitle_path.exists():
                st.caption("提醒：找不到校對字幕，上傳時會略過字幕。")
            upload_status = upload_record.get("status")
            if upload_status:
                st.caption(f"最近上傳狀態：{upload_status}")
            if upload_state.get("running"):
                st.info(f"上傳進行中，PID {upload_state.get('pid')}")

            privacy_default = upload_record.get("privacy", "private")
            privacy_opts = ["private", "unlisted", "public"]
            privacy_idx = privacy_opts.index(privacy_default) if privacy_default in privacy_opts else 0
            privacy_val = st.selectbox("隱私設定", privacy_opts, index=privacy_idx, key=f"studio_privacy_{info['ep']}")
            schedule_on = st.checkbox("排程發布", value=False, key=f"studio_schedule_{info['ep']}")
            sched_cols = st.columns(2)
            with sched_cols[0]:
                publish_date = st.date_input("發布日期", key=f"studio_publish_date_{info['ep']}", disabled=not schedule_on)
            with sched_cols[1]:
                publish_time = st.time_input("發布時間", key=f"studio_publish_time_{info['ep']}", disabled=not schedule_on)
            playlist_id = st.text_input("播放清單 ID（可留空）", value="", key=f"studio_playlist_{info['ep']}")

            upload_args = ["--ep", str(info["ep"]), "--privacy", privacy_val]
            if schedule_on:
                publish_at = f"{publish_date.isoformat()} {publish_time.strftime('%H:%M')}"
                upload_args += ["--publish_at", publish_at]
            if playlist_id.strip():
                upload_args += ["--playlist_id", playlist_id.strip()]
            upload_step = {
                "name": "上傳 YouTube",
                "type": "python",
                "script": "scripts/upload_to_youtube.py",
                "args": upload_args,
            }
            if st.button("執行上傳", key="studio_run_upload", disabled=(not ready_to_upload) or bool(upload_state.get("running"))):
                res = start_publisher_task(info, "upload", upload_step)
                st.session_state["studio_notice"] = {
                    "task": "upload",
                    "ok": bool(res.get("ok")),
                    "message": res.get("message", ""),
                    "log": res.get("log", ""),
                }
                st.rerun()

            notice = st.session_state.get("studio_notice")
            if isinstance(notice, dict) and notice.get("task") == "upload":
                if notice.get("ok"):
                    st.success(f"上傳任務已啟動。log: {notice.get('log')}")
                else:
                    st.error(f"上傳任務啟動失敗：{notice.get('message')}")
                st.session_state.pop("studio_notice", None)

            if upload_record:
                st.divider()
                st.json(upload_record)
                if upload_record.get("video_id"):
                    st.markdown(f"[開啟 YouTube 影片](https://www.youtube.com/watch?v={upload_record['video_id']})")

        with tab_logs:
            if st.button("重新整理發布 Log", key="studio_refresh_logs"):
                st.rerun()
            log_tabs = st.tabs(["Metadata", "Upload"])
            meta_log = Path(meta_state.get("log", ""))
            upload_log = Path(upload_state.get("log", ""))
            for label, log_path, running, dl_key in [
                ("Metadata", meta_log, bool(meta_state.get("running")), "dl_studio_meta_log_2"),
                ("Upload", upload_log, bool(upload_state.get("running")), "dl_studio_upload_log"),
            ]:
                with log_tabs[0] if label == "Metadata" else log_tabs[1]:
                    if log_path.exists():
                        st.download_button(f"下載 {label} log", data=log_path.read_bytes(), file_name=log_path.name, key=dl_key)
                        st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
                        if running:
                            st.caption("執行中。可按上方按鈕重新整理。")
                    else:
                        st.caption(f"尚無 {label} log。")


