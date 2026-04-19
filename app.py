import os
import sys
from pathlib import Path
import json
import re
import time
import base64
import mimetypes
import subprocess
from io import BytesIO
from datetime import datetime
from typing import Dict
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image, ImageFile

from app_utils.filesystem import (
    episode_dir_name,
    find_episode_dirs_by_ep,
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
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def normalize_substep_no(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.split(".")
    if all(part.isdigit() for part in parts):
        return ".".join(str(int(part)) for part in parts)
    return text


def substep_sort_key(value) -> tuple:
    text = normalize_substep_no(value)
    parts = text.split(".")
    if all(part.isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (1, (text,))


SUBSTEP_IDS = sorted(
    [normalize_substep_no(getattr(s, "no", "")) for s in substeps_map if normalize_substep_no(getattr(s, "no", ""))],
    key=substep_sort_key,
)
SUBSTEP_MIN = SUBSTEP_IDS[0] if SUBSTEP_IDS else "3"
SUBSTEP_MAX = SUBSTEP_IDS[-1] if SUBSTEP_IDS else "17"
VOCAB_REBASE_EP = 6
VOCAB_REBASE_START = 151
EPISODE_SCAFFOLD_DIRS = [
    "00_logs",
    "01_audio",
    "02_subtitles",
    "03_storyboards",
    "04_images",
    "05_output",
]
DEFAULT_CLOZE_PROMPT_TEMPLATE = """你是專業的英文考題編輯，請依據提供的 vocab_data，為本集每個單字各產生 1 題四選一克漏字題。

你必須嚴格遵守以下規則：
1. 請依 vocab_data 原本順序輸出，總題數必須剛好等於 {{VOCAB_COUNT}} 題。
2. 每題都必須對應到同一列 vocab 的 Word。
3. 請優先以該列的 English_Sentence 為基礎，把目標字詞替換成 `____`，形成 `blank_sentence`。
4. 如果句中需要詞形變化，例如第三人稱單數、過去式、現在分詞，正確選項請使用符合句子的詞形。
5. 干擾選項必須自然、合理、可辨識，不要亂造不存在或很奇怪的字。
6. explanation 請用繁體中文，簡短說明為什麼正確，並可順帶提示中文意思。
7. 只輸出 JSON，不要輸出任何額外文字。

請輸出 JSON 陣列。每個元素都必須包含：
- surface_word
- blank_sentence
- choice_A
- choice_B
- choice_C
- choice_D
- correct_option
- explanation

補充資訊：
- 集數：第 {{EPISODE}} 集
- 單字範圍：{{START_WORD}}-{{END_WORD}}
- 單字資料 JSON：
{{VOCAB_JSON}}
"""
DEFAULT_NOTEBOOKLM_PROMPT_TEMPLATE = """【節目設定】
這是一集《會考英文隨身聽》節目，主題是國中會考英文 2000 單字教學。
集數：第 {{EPISODE}} 集
單字範圍：{{START_WORD}} 到 {{END_WORD}}

【角色設定】
請用雙主持人對話方式撰寫節目摘要與導聽 prompt：
- Mary：溫柔、清楚、善於整理重點
- John：自然、口語、會補充例句與學習提醒

【節目流程要求】
1. 用自然口語的雙人對話介紹本集單字。
2. 依單字原始順序逐字帶出，不可漏字。
3. 每個單字都要包含詞性、中文意思、英文例句重點與中文提示。
4. 避免只是在念表格，要像節目內容，有銜接、有互動。
5. 結尾簡短鼓勵聽眾持續複習。

{{PREVIOUS_EPISODE_CLOZE_SECTION}}

{{NEXT_EPISODE_CLOZE_SECTION}}

【本集單字資料】
{{VOCAB_LIST}}
"""
DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE = """你是一位專業的影片字幕校對專家。請針對以下 SRT 內容進行修正，並務必遵守下列規則：

1. 單字修復與局部合併：
- 僅在單字被切斷時合併，例如同一個英文單字被拆成兩個字幕序號。
- 嚴禁因語意相近就合併不同序號。
- 嚴禁跨句、跨段任意合併。

2. 全面校對：
- 修正英文拼字錯誤。
- 根據上下文修正常見音近誤辨。
- 關鍵單字可視情況加上引號，但不要破壞 SRT 結構。

3. 時間軸與格式：
- 保持標準 SRT 格式。
- 若因修復切字而局部合併，請重新整理序號。
- 盡量讓相鄰字幕保留極短空隙。
- 只輸出純 SRT，不要輸出 Markdown、解說或任何額外文字。

【原始字幕內容】
{{SRT_CONTENT}}
"""
DEFAULT_STORYBOARD_PROMPT_TEMPLATE = """你是一位專業的英語教學影片分鏡導演。以下是一份英語教學 Podcast 的 SRT 字幕檔。
你的首要任務是：精準捕捉每一個「本集重點單字」出現的片段，並將畫面切換為對應圖卡；此外，也要處理節目開頭或結尾的克漏字卡片。

【本集重點單字清單】（請反覆核對字幕，絕不能漏掉任何一個字）
{{WORD_LIST}}

【圖卡與分鏡規劃規則】
1. 最高優先級：閃卡切換（FLASHCARD）
- 逐行檢查字幕。只要主持人在對話中提到、拼寫、解釋或舉例本集重點單字，就必須切出一個獨立分鏡。
- 該分鏡的 source_type 必須是 "FLASHCARD"。
- flashcard_word 必須精準填入清單中的單字拼法。
- 這類分鏡的 image_prompt 留空。

2. 克漏字題卡規則
{{CLOZE_CARD_RULES}}
- 只要有克漏字題卡需求，該題卡必須獨立成一個單獨分鏡，不可和一般 AI 串場畫面合併。
- 題目卡與答案卡都視為 FLASHCARD 分鏡，不可寫成 AI 分鏡。
- 主持人宣告「現在要出題」或「現在要公布答案」的串場，可以另外用 AI 分鏡；但真正顯示題卡的那一鏡一定要獨立。

3. 串場與故事畫面（AI）
- 只有在沒有講解重點單字，且也不是克漏字解題/出題段落時，才可設為 "AI"。
- AI 分鏡盡量控制在 10 到 30 秒之間，過長請拆分，過短可適度合併。
- AI 分鏡必須提供完整英文 image_prompt，並固定以 ", aspect ratio 16:9, cinematic wide shot" 作結。

4. 人物一致性規則
{{HOST_PROFILE_RULES}}

5. 時間連續性
- start_time 與 end_time 必須連續，不可留空隙，也不可重疊。
- 必須完整覆蓋從 0 秒到影片結束的全部內容。
- 每個分鏡都必須落在實際字幕時間範圍內，不可超出最後一筆字幕結束時間。

【輸出格式】
只輸出合法 JSON 陣列，不要輸出 Markdown 或任何說明文字。
每個元素至少包含：
- start_time
- end_time
- source_type
- image_prompt
- flashcard_word
- reason

【字幕內容】
{{SRT_CONTENT}}
"""
DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE = """你是一位專業的 YouTube SEO 專家與國中英文老師。
請根據以下提供的完整影片字幕檔（SRT），為影片生成 YouTube Metadata。

【任務要求】
1. 這是《會考英文隨身聽》的 2000 單字教學影片。
2. 請閱讀完整字幕，精準抓出本集教學單字，以及它們在影片中出現的精準時間（MM:SS）。
3. 必須將本集 {{START_WORD}} 到 {{END_WORD}} 號範圍內的重點單字完整列出，不可使用省略符號。
4. 請在 summary 結尾，用親切語氣提醒觀眾：頻道固定在每週日與每週三晚上 6 點上傳新影片。

【字幕檔完整內容】
{{SRT_CONTENT}}

【輸出要求】
請嚴格輸出 JSON，必須包含：
1. summary：大約 100 字摘要。
2. tags：15 個標籤。
3. words：陣列，每個元素包含 time、word、meaning。
"""

# --- helpers ---

def compute_rebased_episode_range(ep_num: int, words_per_episode: int) -> tuple[int, int]:
    if ep_num < VOCAB_REBASE_EP:
        raise ValueError(f"第 {VOCAB_REBASE_EP} 集起才會套用新的每集單字數規則。")
    if words_per_episode <= 0:
        raise ValueError("每集單字數必須大於 0。")
    start = VOCAB_REBASE_START + (ep_num - VOCAB_REBASE_EP) * words_per_episode
    end = start + words_per_episode - 1
    return start, end

def build_episode_generation_plan(start_ep: int, end_ep: int, words_per_episode: int) -> list[Dict]:
    if start_ep > end_ep:
        raise ValueError("起始集數不可大於結束集數。")
    plans = []
    for ep_num in range(start_ep, end_ep + 1):
        start_word, end_word = compute_rebased_episode_range(ep_num, words_per_episode)
        dir_name = episode_dir_name(ep_num, start_word, end_word)
        existing_dirs = find_episode_dirs_by_ep(ROOT, ep_num, WS_ROOT)
        exact_match = next((p for p in existing_dirs if p.name == dir_name), None)
        conflict_dirs = [p for p in existing_dirs if p.name != dir_name]
        if exact_match and not conflict_dirs:
            action = "沿用"
        elif exact_match and conflict_dirs:
            action = "沿用 + 備份衝突"
        elif conflict_dirs:
            action = "需備份舊資料夾"
        else:
            action = "建立"
        plans.append({
            "ep": ep_num,
            "start": start_word,
            "end": end_word,
            "words_per_episode": words_per_episode,
            "dir_name": dir_name,
            "target_path": ROOT / WS_ROOT / dir_name,
            "existing_dirs": existing_dirs,
            "exact_match": exact_match,
            "conflict_dirs": conflict_dirs,
            "action": action,
        })
    return plans

def generation_plan_table(plans: list[Dict]) -> pd.DataFrame:
    rows = []
    for plan in plans:
        rows.append({
            "Ep": f"Ep{plan['ep']:02d}",
            "Range": f"{plan['start']:04d}-{plan['end']:04d}",
            "每集單字數": plan["words_per_episode"],
            "目標資料夾": plan["dir_name"],
            "現有資料夾": "\n".join(p.name for p in plan["existing_dirs"]) or "—",
            "動作": plan["action"],
        })
    return pd.DataFrame(rows)

def backup_episode_dir(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}__bak_{stamp}")
    seq = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}__bak_{stamp}_{seq}")
        seq += 1
    path.rename(candidate)
    return candidate

def apply_episode_generation(plans: list[Dict], allow_backup_conflicts: bool) -> Dict:
    created = []
    reused = []
    backed_up = []
    conflicts = []
    for plan in plans:
        if plan["conflict_dirs"] and not allow_backup_conflicts:
            conflicts.append(plan)
            continue
        for old_dir in plan["conflict_dirs"]:
            backed_up.append({"from": old_dir, "to": backup_episode_dir(old_dir)})
        target_path = plan["target_path"]
        existed_before = target_path.exists()
        target_path.mkdir(parents=True, exist_ok=True)
        for folder_name in EPISODE_SCAFFOLD_DIRS:
            (target_path / folder_name).mkdir(parents=True, exist_ok=True)
        if existed_before:
            reused.append(target_path)
        else:
            created.append(target_path)
    return {
        "created": created,
        "reused": reused,
        "backed_up": backed_up,
        "conflicts": conflicts,
    }

def get_pipeline_mapping(profile_id: str):
    if profile_id == "vocab":
        mapping = {
            "1": ["3", "4", "5", "6", "7", "8", "9"],
            "1.5": ["10", "10.5"],
            "2": ["11"],
            "3": ["12", "13"],
            "4": ["14", "15", "16"],
            "5": ["17"],
            "6": ["18"],
            "7": ["19"],
        }
        manual_labels = {
            "9": "Step 9 完成 (NotebookLM)",
            "13": "Step 13 完成 (人工字幕)",
            "15": "Step 15 完成 (人工微調)",
        }
        publish_substep = "19"
        storyboard_manual_substep = "15"
    else:
        mapping = {
            "1": ["3", "4", "5", "6", "7"],
            "1.5": ["8"],
            "2": ["9"],
            "3": ["10", "11"],
            "4": ["12", "13", "14"],
            "5": ["15"],
            "6": ["16"],
            "7": ["17"],
        }
        manual_labels = {
            "7": "Step 7 完成 (NotebookLM)",
            "11": "Step 11 完成 (人工字幕)",
            "13": "Step 13 完成 (人工微調)",
        }
        publish_substep = "17"
        storyboard_manual_substep = "13"

    sub2stage = {}
    for sid, lst in mapping.items():
        for n in lst:
            sub2stage[normalize_substep_no(n)] = sid
    exec_mode = {}
    for ns in SUBSTEP_IDS:
        if ns in manual_labels:
            exec_mode[ns] = "manual"
        elif ns in sub2stage:
            exec_mode[ns] = "auto"
        else:
            exec_mode[ns] = "na"
    return mapping, sub2stage, exec_mode, manual_labels, publish_substep, storyboard_manual_substep

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
    """Return per-episode substep status using the same logic
    as the details panel (evaluate_substeps_debug)."""
    out = []
    for e in eps:
        info = e["_raw"]
        dbg = evaluate_substeps_debug(info["path"], substeps_map)
        status = {}
        for row in dbg:
            k = normalize_substep_no(row.get("no"))
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
        for k in SUBSTEP_IDS:
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

def cloze_prompt_path(ep_path: Path) -> Path:
    return ep_path / "03_storyboards" / "cloze_quiz_prompt.txt"

def ensure_cloze_prompt_file(ep_path: Path) -> Path:
    prompt_path = cloze_prompt_path(ep_path)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_CLOZE_PROMPT_TEMPLATE)
    return prompt_path

def cloze_question_csv_path(ep_path: Path) -> Path:
    return ep_path / "03_storyboards" / "cloze_questions.csv"

def cloze_question_json_path(ep_path: Path) -> Path:
    return ep_path / "03_storyboards" / "cloze_questions.json"

def cloze_question_preview_df(ep_path: Path) -> pd.DataFrame:
    csv_path = cloze_question_csv_path(ep_path)
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path).fillna("")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def render_text_output_expander(title: str, path: Path | None, *, text_label: str, key_prefix: str, expanded: bool = False, height: int = 320):
    if not path or not path.exists():
        return
    with st.expander(title, expanded=expanded):
        st.caption(f"輸出檔案：{path}")
        st.text_area(
            text_label,
            value=read_text_file(path, ""),
            height=height,
            key=f"{key_prefix}_text_output",
        )

def render_final_prompt_expander(title: str, prompt_text: str, *, key_prefix: str, expanded: bool = False, height: int = 360):
    with st.expander(title, expanded=expanded):
        st.caption("以下內容為目前 Prompt 模板套入變數後的最終送出內容預覽。")
        st.text_area(
            title,
            value=prompt_text,
            height=height,
            key=f"{key_prefix}_final_prompt",
        )

def render_dataframe_output_expander(title: str, df: pd.DataFrame, *, expanded: bool = False):
    if df.empty:
        return
    with st.expander(title, expanded=expanded):
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_cloze_output_expander(ep_path: Path, *, key_prefix: str, expanded: bool = False):
    csv_path = cloze_question_csv_path(ep_path)
    json_path = cloze_question_json_path(ep_path)
    preview_df = cloze_question_preview_df(ep_path)
    if not csv_path.exists() and not json_path.exists():
        return
    with st.expander("Step 5 輸出預覽", expanded=expanded):
        file_parts = []
        if csv_path.exists():
            file_parts.append(str(csv_path))
        if json_path.exists():
            file_parts.append(str(json_path))
        if file_parts:
            st.caption("輸出檔案：" + " | ".join(file_parts))
        tab_table, tab_json, tab_csv = st.tabs(["表格", "JSON", "CSV"])
        with tab_table:
            if preview_df.empty:
                st.caption("尚無可預覽題目。")
            else:
                preview_cols = [
                    c for c in [
                        "question_id",
                        "word",
                        "blank_sentence",
                        "choice_A",
                        "choice_B",
                        "choice_C",
                        "choice_D",
                        "correct_option",
                        "explanation",
                    ]
                    if c in preview_df.columns
                ]
                st.dataframe(
                    preview_df[preview_cols] if preview_cols else preview_df,
                    use_container_width=True,
                    hide_index=True,
                )
        with tab_json:
            if json_path.exists():
                st.text_area(
                    "cloze_questions.json",
                    value=read_text_file(json_path, ""),
                    height=320,
                    key=f"{key_prefix}_cloze_json",
                )
            else:
                st.caption("尚未產生 cloze_questions.json。")
        with tab_csv:
            if csv_path.exists():
                st.text_area(
                    "cloze_questions.csv",
                    value=read_text_file(csv_path, ""),
                    height=320,
                    key=f"{key_prefix}_cloze_csv",
                )
            else:
                st.caption("尚未產生 cloze_questions.csv。")

@st.cache_data(show_spinner=False, max_entries=256)
def load_preview_image_bytes(path_str: str, max_width: int = 640):
    path = Path(path_str)
    try:
        with Image.open(path) as img:
            img.load()
            preview = img.convert("RGB")
            if max_width > 0 and preview.width > max_width:
                new_height = max(1, int(preview.height * (max_width / preview.width)))
                preview = preview.resize((max_width, new_height), Image.Resampling.LANCZOS)
            buf = BytesIO()
            preview.save(buf, format="JPEG", quality=82, optimize=True)
            return buf.getvalue(), None
    except Exception as e:
        return None, str(e)

def render_safe_image_preview(
    path: Path | None,
    *,
    empty_message: str,
    broken_hint: str,
    caption_label: str,
    max_width: int = 640,
):
    if not path:
        st.caption(empty_message)
        return
    st.caption(f"{caption_label}：{path.name}")
    image_bytes, err = load_preview_image_bytes(str(path), max_width=max_width)
    if image_bytes is None:
        st.warning(f"{broken_hint}：{err}")
        st.caption(f"檔案位置：{path}")
        return
    st.image(image_bytes, use_container_width=True)

def render_thumbnail_card(path: Path, *, caption_text: str, current: bool = False, max_width: int = 420):
    with st.container(border=True):
        st.caption("目前使用中" if current else caption_text)
        render_safe_image_preview(
            path,
            empty_message="沒有可顯示的圖片。",
            broken_hint="圖片無法讀取",
            caption_label="預覽",
            max_width=max_width,
        )

def render_manual_subtitle_editor(ep_path: Path, step_no: str, *, key_prefix: str, expanded: bool = False):
    subtitle_path = subtitles_fixed_path(ep_path)
    file_text = read_text_file(subtitle_path, "")
    text_key = f"{key_prefix}_subtitle_text"
    loaded_key = f"{text_key}__loaded"
    marker_key = f"{key_prefix}_subtitle_done"
    sync_textarea_state(text_key, loaded_key, file_text)
    if marker_key not in st.session_state:
        st.session_state[marker_key] = has_manual_marker(ep_path, step_no)
    dirty = st.session_state.get(text_key, "") != file_text

    with st.expander(f"No.{step_no} 人工字幕校正", expanded=expanded):
        st.caption(f"字幕檔案：{subtitle_path}")
        if dirty:
            st.warning("目前字幕內容尚未存檔。")
        st.text_area(
            "字幕內容",
            key=text_key,
            height=360,
        )
        st.checkbox("同步更新完成標記", key=marker_key)
        action1, action2, action3 = st.columns([1, 1.2, 1.1])
        with action1:
            save_subtitle = st.button("儲存字幕", key=f"{key_prefix}_save_subtitle")
        with action2:
            save_and_mark = st.button("儲存字幕並更新標記", key=f"{key_prefix}_save_mark_subtitle")
        with action3:
            reload_subtitle = st.button("還原檔案內容", key=f"{key_prefix}_reload_subtitle")

        if save_subtitle:
            saved_path = save_text_file(subtitle_path, st.session_state.get(text_key, ""))
            st.session_state[loaded_key] = st.session_state.get(text_key, "")
            st.success(f"已儲存字幕：{saved_path}")
            st.rerun()
        if save_and_mark:
            saved_path = save_text_file(subtitle_path, st.session_state.get(text_key, ""))
            st.session_state[loaded_key] = st.session_state.get(text_key, "")
            set_manual_marker(ep_path, step_no, bool(st.session_state.get(marker_key)))
            st.success(f"已儲存字幕並更新 No.{step_no} 標記：{saved_path}")
            st.rerun()
        if reload_subtitle:
            st.session_state[text_key] = file_text
            st.session_state[loaded_key] = file_text
            st.session_state[marker_key] = has_manual_marker(ep_path, step_no)
            st.info("已重新載入目前字幕檔內容。")
            st.rerun()

        if subtitle_path.exists():
            st.download_button(
                "下載字幕",
                data=subtitle_path.read_bytes(),
                file_name=subtitle_path.name,
                key=f"{key_prefix}_download_subtitle",
            )
        else:
            st.caption("目前尚無校對字幕；可直接在上方貼上內容後儲存建立。")

def render_prompts_workspace(info: Dict, ep_path: Path, profile_id: str, subtitle_path: Path, meta_path: Path | None):
    st.subheader("Prompt 編輯與手動重跑")
    st.caption("可先修改各步驟 prompt、存檔，再手動執行對應子步驟。未存檔時，執行按鈕會鎖住。")

    notice = st.session_state.get("studio_prompt_notice")
    if isinstance(notice, dict):
        if notice.get("ok"):
            st.success(f"Step {notice.get('task')} 任務已啟動。log: {notice.get('log')}")
        else:
            st.error(f"Step {notice.get('task')} 任務啟動失敗：{notice.get('message')}")
        st.session_state.pop("studio_prompt_notice", None)

    def render_prompt_block(
        step_no: str,
        title: str,
        prompt_path: Path,
        default_template: str,
        step_def: Dict,
        *,
        help_text: str = "",
        expanded: bool = False,
        output_renderer = None,
        final_prompt_renderer = None,
    ):
        file_text = read_text_file(prompt_path, default_template)
        text_key = f"studio_prompt_{step_no}_{info['ep']}_{profile_id}"
        loaded_key = f"{text_key}__loaded"
        sync_textarea_state(text_key, loaded_key, file_text)
        current_text = st.session_state.get(text_key, "")
        dirty = current_text != file_text
        state = runner.get_substep_state(info, step_no)

        with st.expander(f"No.{step_no} {title}", expanded=expanded):
            top1, top2, top3 = st.columns([1, 1, 4])
            top1.metric("Prompt", "已建立" if prompt_path.exists() else "缺少")
            top2.metric("狀態", "執行中" if state.get("running") else "待命")
            top3.caption(f"Prompt 檔案：{prompt_path}")
            if dirty:
                st.warning("目前編輯內容尚未存檔。請先儲存，再執行這個步驟。")
            if help_text:
                st.caption(help_text)
            st.text_area(
                f"{title} Prompt",
                key=text_key,
                height=300,
            )
            action1, action2, action3 = st.columns([1, 1, 1.2])
            with action1:
                save_prompt = st.button("儲存 Prompt", key=f"save_prompt_{step_no}_{info['ep']}")
            with action2:
                reload_prompt = st.button("還原檔案內容", key=f"reload_prompt_{step_no}_{info['ep']}")
            with action3:
                run_prompt_step = st.button(
                    f"執行 Step {step_no}",
                    key=f"run_prompt_{step_no}_{info['ep']}",
                    disabled=bool(state.get("running")) or dirty or not current_text.strip(),
                )
            if save_prompt:
                saved_prompt_path = save_text_file(prompt_path, current_text)
                st.session_state[loaded_key] = current_text
                st.success(f"已儲存 Prompt：{saved_prompt_path}")
                st.rerun()
            if reload_prompt:
                st.session_state[text_key] = file_text
                st.session_state[loaded_key] = file_text
                st.info("已重新載入目前檔案內容。")
                st.rerun()
            if run_prompt_step:
                res = runner.start_substep(info, step_no, step_def)
                st.session_state["studio_prompt_notice"] = {
                    "task": step_no,
                    "ok": bool(res.get("ok")),
                    "message": res.get("message", ""),
                    "log": res.get("log", ""),
                }
                st.rerun()
            if state.get("running"):
                st.info(f"Step {step_no} 執行中，PID {state.get('pid')}")
            if final_prompt_renderer:
                try:
                    final_prompt_text = final_prompt_renderer(current_text)
                except Exception as e:
                    final_prompt_text = f"[render error] {e}"
                render_final_prompt_expander(
                    f"Step {step_no} 最終 Prompt 內容",
                    final_prompt_text,
                    key_prefix=f"prompt_render_{step_no}_{info['ep']}_{profile_id}",
                    expanded=False,
                    height=380,
                )
            if output_renderer:
                output_renderer()
            log_path = Path(state.get("log", ""))
            if log_path.exists():
                with st.expander(f"Step {step_no} Log", expanded=bool(state.get("running"))):
                    log_summary = prompt_log_summary(state, log_path)
                    meta1, meta2, meta3 = st.columns(3)
                    meta1.metric("開始時間", log_summary["start"])
                    meta2.metric("結束時間", log_summary["end"])
                    meta3.metric("耗時", log_summary["elapsed"])
                    st.caption(f"Log 檔案：{log_path}")
                    st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-5000:])
            return state, log_path, dirty

    step5_def = {
        "name": "產生克漏字題目檔",
        "type": "python",
        "script": "scripts/generate_cloze_quiz.py",
        "args": ["--ep", str(info["ep"])],
    }
    step8_def = {
        "name": "產生語音摘要 Prompt",
        "type": "python",
        "script": "scripts/generate_notebooklm_prompt.py",
        "args": ["--ep", str(info["ep"])],
    }
    step12_def = {
        "name": "AI 字幕校對",
        "type": "python",
        "script": "scripts/llm_subtitle_reviewer.py",
        "args": ["--ep", str(info["ep"])],
    }
    step14_def = {
        "name": "AI 依語境分鏡規劃",
        "type": "python",
        "script": "scripts/llm_director.py",
        "args": ["--ep", str(info["ep"])],
    }
    step19_def = {
        "name": "產生 YouTube Metadata",
        "type": "python",
        "script": "scripts/generate_youtube_meta.py",
        "args": ["--ep", str(info["ep"])],
    }

    prompt_path = ensure_cloze_prompt_file(ep_path)
    render_prompt_block(
        "5",
        "克漏字題目生成",
        prompt_path,
        DEFAULT_CLOZE_PROMPT_TEMPLATE,
        step5_def,
        help_text="Step 5 會讀這份 prompt，產生 cloze_questions.csv / json。",
        expanded=True,
        output_renderer=lambda: render_cloze_output_expander(
            ep_path,
            key_prefix=f"studio_step5_{info['ep']}_{profile_id}",
            expanded=False,
        ),
        final_prompt_renderer=lambda template_text: render_step5_prompt_preview(template_text, ep_path),
    )

    notebook_prompt_path = ensure_notebooklm_prompt_template_file(ep_path)
    render_prompt_block(
        "8",
        "產生語音摘要 Prompt",
        notebook_prompt_path,
        DEFAULT_NOTEBOOKLM_PROMPT_TEMPLATE,
        step8_def,
        help_text="Step 8 會將這份 template 渲染成 notebooklm_prompt_epXX.txt。",
        output_renderer=lambda: render_text_output_expander(
            "Step 8 輸出預覽",
            notebooklm_prompt_output_path(ep_path, info["ep"]),
            text_label="NotebookLM Prompt 輸出",
            key_prefix=f"studio_step8_{info['ep']}_{profile_id}",
            expanded=False,
            height=320,
        ),
        final_prompt_renderer=lambda template_text: render_step8_prompt_preview(template_text, ep_path),
    )

    subtitle_prompt = ensure_subtitle_review_prompt_file(ep_path)
    render_prompt_block(
        "12",
        "AI 字幕校對",
        subtitle_prompt,
        DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE,
        step12_def,
        help_text="Step 12 會讀取原始字幕，依這份 prompt 產生校對後的 notebooklm_audio_fixed.srt。",
        output_renderer=lambda: render_text_output_expander(
            "Step 12 輸出預覽",
            subtitle_path,
            text_label="校對字幕",
            key_prefix=f"studio_step12_{info['ep']}_{profile_id}",
            expanded=False,
            height=260,
        ) if subtitle_path.exists() else None,
        final_prompt_renderer=lambda template_text: render_step12_prompt_preview(
            template_text,
            ep_path / "02_subtitles" / "notebooklm_audio.srt",
        ),
    )

    storyboard_prompt = ensure_storyboard_prompt_file(ep_path)
    render_prompt_block(
        "14",
        "AI 依語境分鏡規劃",
        storyboard_prompt,
        DEFAULT_STORYBOARD_PROMPT_TEMPLATE,
        step14_def,
        help_text="Step 14 會依字幕與單字表生成 storyboard.csv。",
        output_renderer=lambda: render_dataframe_output_expander(
            "Step 14 輸出預覽",
            read_storyboard_df(ep_path),
            expanded=False,
        ),
        final_prompt_renderer=lambda template_text: render_step14_prompt_preview(template_text, ep_path, subtitle_path),
    )

    meta_prompt = ensure_youtube_meta_prompt_file(ep_path)
    render_prompt_block(
        "19",
        "產生 YouTube Metadata",
        meta_prompt,
        DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE,
        step19_def,
        help_text="Step 19 會讀取校對字幕與集數範圍，產生 youtube_meta.json。",
        output_renderer=lambda: render_text_output_expander(
            "Step 19 輸出預覽",
            first_existing_path(youtube_meta_paths(ep_path)),
            text_label="YouTube Metadata JSON",
            key_prefix=f"studio_step19_{info['ep']}_{profile_id}",
            expanded=False,
            height=260,
        ) if first_existing_path(youtube_meta_paths(ep_path)) and first_existing_path(youtube_meta_paths(ep_path)).exists() else None,
        final_prompt_renderer=lambda template_text: render_step19_prompt_preview(template_text, ep_path, subtitle_path),
    )

def render_storyboard_workspace(info: Dict, ep_path: Path, profile_id: str):
    st.subheader("No.13 修改分鏡 / 替換單字卡")
    storyboard_df = read_storyboard_df(ep_path)
    flashcards = flashcard_map_for(ep_path)
    if storyboard_df.empty:
        st.warning(f"找不到分鏡表：{storyboard_path_for(ep_path)}")
        return

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

    scene_option_labels = [
        f"Scene {str(r.get('scene_id', ''))} | {seconds_to_label(r.get('start_time', 0))} - "
        f"{seconds_to_label(r.get('end_time', 0))} | {str(r.get('reason', ''))[:36]}"
        for _, r in storyboard_df.iterrows()
    ]
    scene_option_map = dict(zip(scene_option_labels, storyboard_df["scene_id"].astype(str).tolist()))

    with st.container(border=True):
        st.markdown("**結構調整**")
        st.caption("這裡處理分鏡結構本身。合併或切分後，Scene 會重新編號；動畫影片綁定會清空，避免沿用錯的場次。")
        merge_tab, split_tab = st.tabs(["合併多個分鏡", "一個分鏡切成多個"])

        with merge_tab:
            merge_cols = st.columns([1.4, 1.4, 1.6])
            with merge_cols[0]:
                merge_start_label = st.selectbox(
                    "起始 Scene",
                    scene_option_labels,
                    key=f"storyboard_merge_start_{info['ep']}_{profile_id}",
                )
            with merge_cols[1]:
                merge_end_label = st.selectbox(
                    "結束 Scene",
                    scene_option_labels,
                    index=min(1, len(scene_option_labels) - 1),
                    key=f"storyboard_merge_end_{info['ep']}_{profile_id}",
                )
            with merge_cols[2]:
                merge_carry_mode = st.radio(
                    "合併後保留哪一個素材設定",
                    ["first", "last", "clear_ai"],
                    format_func=lambda x: {
                        "first": "保留第一個 Scene",
                        "last": "保留最後一個 Scene",
                        "clear_ai": "清空並改成 AI Scene",
                    }[x],
                    horizontal=True,
                    key=f"storyboard_merge_mode_{info['ep']}_{profile_id}",
                )
            merge_start_id = scene_option_map[merge_start_label]
            merge_end_id = scene_option_map[merge_end_label]
            merge_scene_ids = storyboard_df["scene_id"].astype(str).tolist()
            merge_start_pos = merge_scene_ids.index(str(merge_start_id))
            merge_end_pos = merge_scene_ids.index(str(merge_end_id))
            if merge_start_pos > merge_end_pos:
                merge_start_pos, merge_end_pos = merge_end_pos, merge_start_pos
            merge_preview_df = storyboard_df.iloc[merge_start_pos:merge_end_pos + 1].copy()
            default_merge_reason = _storyboard_join_text(merge_preview_df["reason"].tolist(), sep=" / ")
            merge_reason_key = f"storyboard_merge_reason_{info['ep']}_{profile_id}_{merge_start_pos}_{merge_end_pos}"
            if merge_reason_key not in st.session_state:
                st.session_state[merge_reason_key] = default_merge_reason
            merge_summary1, merge_summary2, merge_summary3 = st.columns(3)
            merge_summary1.metric("選取 Scene 數", len(merge_preview_df))
            merge_summary2.metric(
                "合併後時長",
                f"{safe_float(merge_preview_df['end_time'].iloc[-1]) - safe_float(merge_preview_df['start_time'].iloc[0]):.2f}s" if not merge_preview_df.empty else "0s",
            )
            merge_summary3.metric(
                "保留素材",
                {
                    "first": f"Scene {merge_preview_df.iloc[0]['scene_id']}" if not merge_preview_df.empty else "—",
                    "last": f"Scene {merge_preview_df.iloc[-1]['scene_id']}" if not merge_preview_df.empty else "—",
                    "clear_ai": "清空為 AI",
                }[merge_carry_mode],
            )
            st.text_area(
                "合併後 reason",
                key=merge_reason_key,
                height=90,
                help="預設會把多個 Scene 的 reason 串起來，你也可以在這裡改成較乾淨的描述。",
            )
            with st.expander("合併預覽", expanded=True):
                st.dataframe(
                    merge_preview_df[["scene_id", "start_time", "end_time", "source_type", "flashcard_word", "reason"]],
                    use_container_width=True,
                    hide_index=True,
                )
            merge_disabled = len(merge_preview_df) < 2
            if merge_disabled:
                st.warning("至少要選取 2 個連續 Scene 才能合併。")
            if st.button("合併選取 Scene", key=f"storyboard_merge_apply_{info['ep']}_{profile_id}", disabled=merge_disabled):
                saved_storyboard, merged_df = merge_storyboard_scenes(
                    ep_path,
                    merge_start_id,
                    merge_end_id,
                    carry_mode=merge_carry_mode,
                    merged_reason=st.session_state.get(merge_reason_key, ""),
                )
                if saved_storyboard is None:
                    st.error("合併失敗，請重新整理後再試。")
                else:
                    st.success(f"已合併 Scene {merge_start_id} 到 Scene {merge_end_id}：{saved_storyboard}")
                    st.rerun()

        with split_tab:
            split_top1, split_top2 = st.columns([1.6, 1])
            with split_top1:
                split_scene_label = st.selectbox(
                    "要切分的 Scene",
                    scene_option_labels,
                    key=f"storyboard_split_scene_{info['ep']}_{profile_id}",
                )
            with split_top2:
                split_count = int(
                    st.number_input(
                        "切成幾段",
                        min_value=2,
                        max_value=8,
                        value=2,
                        step=1,
                        key=f"storyboard_split_count_{info['ep']}_{profile_id}",
                    )
                )
            split_scene_id = scene_option_map[split_scene_label]
            split_source_row = storyboard_df[storyboard_df["scene_id"].astype(str) == str(split_scene_id)].iloc[0].copy()
            split_draft_df = build_split_scene_draft(ep_path, split_source_row, split_count)
            split_editor_key = f"storyboard_split_editor_{info['ep']}_{profile_id}_{split_scene_id}_{split_count}"
            st.caption("預設先等分時間。你可以直接改每段的開始/結束時間與 reason，再套用。")
            edited_split_df = st.data_editor(
                split_draft_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "start_time": st.column_config.NumberColumn("開始", format="%.3f"),
                    "end_time": st.column_config.NumberColumn("結束", format="%.3f"),
                    "source_type": st.column_config.SelectboxColumn("素材來源", options=["AI", "FLASHCARD"]),
                    "flashcard_word": st.column_config.TextColumn("單字卡"),
                    "custom_image_path": st.column_config.TextColumn("圖片路徑", width="large"),
                    "image_prompt": st.column_config.TextColumn("圖片 Prompt", width="large"),
                    "animation_prompt": st.column_config.TextColumn("動畫 Prompt", width="large"),
                    "reason": st.column_config.TextColumn("說明", width="large"),
                    "subtitle_reference": st.column_config.TextColumn("字幕參考", width="large"),
                },
                key=split_editor_key,
            )
            split_times = [
                (safe_float(row["start_time"]), safe_float(row["end_time"]))
                for _, row in edited_split_df.iterrows()
            ]
            split_valid = True
            split_error = ""
            for idx, (seg_start, seg_end) in enumerate(split_times):
                if seg_end <= seg_start:
                    split_valid = False
                    split_error = f"第 {idx + 1} 段的結束時間必須大於開始時間。"
                    break
                if idx > 0 and seg_start < split_times[idx - 1][1]:
                    split_valid = False
                    split_error = f"第 {idx + 1} 段與前一段時間重疊。"
                    break
            split_stats1, split_stats2, split_stats3 = st.columns(3)
            split_stats1.metric("原始時長", f"{safe_float(split_source_row.get('end_time')) - safe_float(split_source_row.get('start_time')):.2f}s")
            split_stats2.metric("切分後段數", len(edited_split_df))
            split_stats3.metric("首尾範圍", f"{safe_float(edited_split_df.iloc[0]['start_time']):.2f}s → {safe_float(edited_split_df.iloc[-1]['end_time']):.2f}s")
            if not split_valid:
                st.error(split_error)
            else:
                st.info("時間檢查通過。套用後會用這些段落取代原本的單一 Scene。")
            if st.button("套用切分", key=f"storyboard_split_apply_{info['ep']}_{profile_id}", disabled=not split_valid):
                saved_storyboard, split_df = split_storyboard_scene(ep_path, split_scene_id, edited_split_df)
                if saved_storyboard is None:
                    st.error("切分失敗，請重新整理後再試。")
                else:
                    st.success(f"已將 Scene {split_scene_id} 切成 {len(edited_split_df)} 段：{saved_storyboard}")
                    st.rerun()

    st.divider()
    st.markdown("**單一 Scene 編輯**")

    scene_choice_ids = storyboard_df["scene_id"].astype(str).tolist()
    scene_choice_map = {
        str(r.get("scene_id", "")): (
            f"Scene {str(r.get('scene_id', ''))} | "
            f"{str(r.get('source_type', '')).upper()} | "
            f"{str(r.get('reason', ''))[:40]}"
        )
        for _, r in storyboard_df.iterrows()
    }
    scene_select_key = f"storyboard_scene_{info['ep']}_{profile_id}"
    if st.session_state.get(scene_select_key) not in scene_choice_ids:
        st.session_state[scene_select_key] = scene_choice_ids[0]
    selected_scene_value = st.selectbox(
        "選擇要修改的 Scene",
        scene_choice_ids,
        key=scene_select_key,
        format_func=lambda sid: scene_choice_map.get(str(sid), f"Scene {sid}"),
    )
    selected_idx = scene_choice_ids.index(str(selected_scene_value))
    selected_row = storyboard_df.iloc[selected_idx].copy()
    selected_scene_id = selected_row.get("scene_id")
    current_word = str(selected_row.get("flashcard_word", "")).strip()
    scene_image_task_name = f"sceneimg_{selected_scene_id}"
    scene_anim_prompt_task_name = f"sceneanimprompt_{selected_scene_id}"
    scene_anim_video_task_name = f"sceneanimvideo_{selected_scene_id}"
    selected_scene_task = publisher_task_state(info, scene_image_task_name)
    selected_anim_prompt_task = publisher_task_state(info, scene_anim_prompt_task_name)
    selected_anim_video_task = publisher_task_state(info, scene_anim_video_task_name)
    current_image_path = str(storyboard_scene_image_path(ep_path, selected_row) or "").strip()
    current_animation_path = str(storyboard_scene_animation_path(ep_path, selected_row) or "").strip()
    image_candidates = scene_image_candidate_paths(ep_path, selected_row)
    animation_candidates = scene_animation_candidate_paths(ep_path, selected_row)
    storyboard_auto_refresh = st.toggle(
        "自動刷新 Scene 產圖 / 動畫 Log",
        value=True,
        key=f"storyboard_auto_refresh_{info['ep']}_{profile_id}",
        help="Scene 單獨產圖、動畫 Prompt、動畫生成執行時，每 2 秒自動更新狀態與 Log。",
    )

    source_key = f"storyboard_source_{info['ep']}_{selected_scene_id}"
    flashcard_key = f"storyboard_flashcard_{info['ep']}_{selected_scene_id}"
    custom_image_key = f"storyboard_custom_image_{info['ep']}_{selected_scene_id}"
    prompt_key = f"storyboard_prompt_{info['ep']}_{selected_scene_id}"
    animation_prompt_key = f"storyboard_animation_prompt_{info['ep']}_{selected_scene_id}"
    animation_prompt_loaded_key = f"{animation_prompt_key}__loaded"
    asset_mode_key = f"storyboard_asset_mode_{info['ep']}_{selected_scene_id}"
    image_choice_key = f"storyboard_image_choice_{info['ep']}_{selected_scene_id}"
    animation_choice_key = f"storyboard_animation_choice_{info['ep']}_{selected_scene_id}"
    reason_key = f"storyboard_reason_{info['ep']}_{selected_scene_id}"
    if source_key not in st.session_state:
        st.session_state[source_key] = str(selected_row.get("source_type", "")).upper() or "AI"
    if flashcard_key not in st.session_state:
        st.session_state[flashcard_key] = current_word if current_word in flashcards else ""
    if custom_image_key not in st.session_state:
        st.session_state[custom_image_key] = str(selected_row.get("custom_image_path", "")).strip()
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = str(selected_row.get("image_prompt", ""))
    if animation_prompt_key not in st.session_state:
        st.session_state[animation_prompt_key] = str(selected_row.get("animation_prompt", ""))
    if animation_prompt_loaded_key not in st.session_state:
        st.session_state[animation_prompt_loaded_key] = str(selected_row.get("animation_prompt", ""))
    latest_animation_prompt_from_row = str(selected_row.get("animation_prompt", ""))
    if (
        st.session_state.get(animation_prompt_loaded_key) != latest_animation_prompt_from_row
        and not bool(selected_anim_prompt_task.get("running"))
    ):
        st.session_state[animation_prompt_key] = latest_animation_prompt_from_row
        st.session_state[animation_prompt_loaded_key] = latest_animation_prompt_from_row
    if reason_key not in st.session_state:
        st.session_state[reason_key] = str(selected_row.get("reason", ""))

    image_choice_map = {
        asset_choice_label(p, ai_scene_image_path(ep_path, selected_row.get("start_time", "")), "圖片"): str(p)
        for p in image_candidates
    }
    image_choice_labels = ["(使用自訂圖片路徑 / 預設原始圖)"] + list(image_choice_map.keys())
    current_image_choice_label = next(
        (label for label, path in image_choice_map.items() if path == current_image_path),
        image_choice_labels[0],
    )
    if st.session_state.get(image_choice_key) not in image_choice_labels:
        st.session_state[image_choice_key] = current_image_choice_label

    animation_choice_map = {
        asset_choice_label(p, ai_scene_animation_path(ep_path, selected_row), "動畫"): str(p)
        for p in animation_candidates
    }
    animation_choice_labels = ["(不使用動畫)"] + list(animation_choice_map.keys())
    current_animation_choice_label = next(
        (label for label, path in animation_choice_map.items() if path == current_animation_path),
        animation_choice_labels[0],
    )
    if st.session_state.get(animation_choice_key) not in animation_choice_labels:
        st.session_state[animation_choice_key] = current_animation_choice_label
    available_asset_modes = ["圖片"] + (["動畫"] if animation_candidates else [])
    current_asset_mode = "動畫" if current_animation_path and animation_candidates else "圖片"
    if st.session_state.get(asset_mode_key) not in available_asset_modes:
        st.session_state[asset_mode_key] = current_asset_mode

    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Scene", str(selected_scene_id))
    meta2.metric("開始", seconds_to_label(selected_row.get("start_time", 0)))
    meta3.metric("結束", seconds_to_label(selected_row.get("end_time", 0)))
    meta4.metric("目前素材", "動畫" if current_animation_path else "圖片")

    st.subheader("目前素材")
    if current_animation_path and Path(current_animation_path).exists():
        st.caption(f"目前使用動畫：{current_animation_path}")
        st.video(current_animation_path)
    elif current_image_path and Path(current_image_path).exists():
        render_safe_image_preview(
            Path(current_image_path),
            empty_message="目前沒有已選定的素材。",
            broken_hint="目前素材圖片無法讀取",
            caption_label="目前使用圖片",
            max_width=720,
        )
    else:
        st.caption("目前沒有已選定的素材。")

    with st.expander(f"候選圖片（{len(image_candidates)}）", expanded=False):
        if image_candidates:
            image_rows = [image_candidates[i:i + 3] for i in range(0, len(image_candidates), 3)]
            for row_group in image_rows:
                cols = st.columns(3)
                for col_idx, candidate_path in enumerate(row_group):
                    with cols[col_idx]:
                        render_thumbnail_card(
                            candidate_path,
                            caption_text=candidate_path.name,
                            current=(str(candidate_path) == current_image_path),
                            max_width=360,
                        )
        else:
            st.caption("尚無候選圖片。")
    with st.expander(f"候選動畫（{len(animation_candidates)}）", expanded=False):
        if animation_candidates:
            animation_rows = [animation_candidates[i:i + 2] for i in range(0, len(animation_candidates), 2)]
            for row_group in animation_rows:
                cols = st.columns(2)
                for col_idx, candidate_path in enumerate(row_group):
                    with cols[col_idx]:
                        with st.container(border=True):
                            st.caption("目前使用中" if str(candidate_path) == current_animation_path else candidate_path.name)
                            st.video(str(candidate_path))
        else:
            st.caption("尚無候選動畫。")

    st.subheader("Scene 設定")
    basic1, basic2, basic3 = st.columns([1, 1.2, 1.4])
    with basic1:
        source_type_val = st.selectbox(
            "source_type",
            ["FLASHCARD", "AI"],
            key=source_key,
        )
        asset_mode_val = st.radio(
            "此分鏡使用素材",
            available_asset_modes,
            key=asset_mode_key,
            horizontal=True,
        )
    with basic2:
        flashcard_options = [""] + sorted(flashcards.keys())
        if current_word and current_word not in flashcard_options:
            flashcard_options.append(current_word)
        if st.session_state[flashcard_key] not in flashcard_options:
            st.session_state[flashcard_key] = ""
        flashcard_word_val = st.selectbox(
            "替換單字卡",
            flashcard_options,
            key=flashcard_key,
            disabled=(source_type_val != "FLASHCARD"),
        )
        selected_image_choice_label = st.selectbox(
            "選擇使用圖片版本",
            image_choice_labels,
            key=image_choice_key,
            disabled=(source_type_val == "FLASHCARD") or (asset_mode_val != "圖片"),
        )
    with basic3:
        custom_image_val = st.text_input(
            "自訂圖片路徑",
            key=custom_image_key,
            disabled=(source_type_val == "FLASHCARD"),
        )
        selected_animation_choice_label = st.selectbox(
            "選擇使用動畫版本",
            animation_choice_labels,
            key=animation_choice_key,
            disabled=(asset_mode_val != "動畫"),
        )

    image_prompt_val = st.text_area(
        "image_prompt",
        height=140,
        key=prompt_key,
        disabled=(source_type_val == "FLASHCARD"),
    )
    animation_prompt_val = st.text_area(
        "animation_prompt",
        height=140,
        key=animation_prompt_key,
    )
    reason_val = st.text_area("reason", height=100, key=reason_key)

    selected_image_choice_path = image_choice_map.get(selected_image_choice_label, "")
    selected_animation_choice_path = animation_choice_map.get(selected_animation_choice_label, "")

    action1, action2, action3, action4 = st.columns([1, 1.1, 1.2, 1.1])
    with action1:
        apply_scene = st.button("儲存此 Scene", key=f"storyboard_save_scene_{info['ep']}_{selected_scene_id}")
    with action2:
        regen_scene = st.button(
            "重產此 AI 分鏡圖",
            key=f"storyboard_regen_scene_{info['ep']}_{selected_scene_id}",
            disabled=(source_type_val != "AI") or bool(selected_scene_task.get("running")),
        )
    with action3:
        suggest_animation_prompt = st.button(
            "AI 建議動畫 Prompt",
            key=f"storyboard_suggest_anim_prompt_{info['ep']}_{selected_scene_id}",
            disabled=bool(selected_anim_prompt_task.get("running")),
        )
    with action4:
        generate_animation = st.button(
            "產生動畫",
            key=f"storyboard_generate_animation_{info['ep']}_{selected_scene_id}",
            disabled=bool(selected_scene_task.get("running")) or bool(selected_anim_video_task.get("running")),
        )

    if apply_scene:
        chosen_ai_image_path = selected_image_choice_path or custom_image_val.strip()
        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
        scene_updates = {
            "source_type": source_type_val,
            "reason": reason_val,
            "animation_prompt": animation_prompt_val,
            "animation_video_path": chosen_animation_path,
        }
        if source_type_val == "FLASHCARD":
            scene_updates["flashcard_word"] = flashcard_word_val
            scene_updates["custom_image_path"] = resolve_flashcard_custom_image(
                flashcards,
                flashcard_word_val,
                str(selected_row.get("custom_image_path", "")),
            )
            scene_updates["image_prompt"] = ""
        else:
            scene_updates["flashcard_word"] = ""
            scene_updates["custom_image_path"] = chosen_ai_image_path
            scene_updates["image_prompt"] = image_prompt_val
        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
        if saved_storyboard is None or idx is None:
            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
        else:
            st.success(f"已儲存 Scene {selected_scene_id}：{saved_storyboard}")
            st.rerun()
    if regen_scene:
        chosen_ai_image_path = selected_image_choice_path or st.session_state[custom_image_key].strip()
        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
        saved_storyboard, latest_df, idx = update_storyboard_scene(
            ep_path,
            selected_scene_id,
            {
                "source_type": "AI",
                "flashcard_word": "",
                "custom_image_path": chosen_ai_image_path,
                "image_prompt": st.session_state[prompt_key],
                "animation_prompt": st.session_state[animation_prompt_key],
                "animation_video_path": chosen_animation_path,
                "reason": st.session_state[reason_key],
            },
        )
        if saved_storyboard is None or idx is None:
            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
        else:
            res = start_publisher_task(
                info,
                scene_image_task_name,
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
    if suggest_animation_prompt:
        chosen_ai_image_path = selected_image_choice_path or custom_image_val.strip()
        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
        scene_updates = {
            "source_type": source_type_val,
            "reason": reason_val,
            "animation_prompt": animation_prompt_val,
            "animation_video_path": chosen_animation_path,
        }
        if source_type_val == "FLASHCARD":
            scene_updates["flashcard_word"] = flashcard_word_val
            scene_updates["custom_image_path"] = resolve_flashcard_custom_image(
                flashcards,
                flashcard_word_val,
                str(selected_row.get("custom_image_path", "")),
            )
            scene_updates["image_prompt"] = ""
        else:
            scene_updates["flashcard_word"] = ""
            scene_updates["custom_image_path"] = chosen_ai_image_path
            scene_updates["image_prompt"] = image_prompt_val
        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
        if saved_storyboard is None or idx is None:
            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
        else:
            res = start_publisher_task(
                info,
                scene_anim_prompt_task_name,
                {
                    "name": f"AI 建議 Scene {selected_scene_id} 動畫 Prompt",
                    "type": "python",
                    "script": "scripts/generate_animation_prompt.py",
                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id)],
                },
            )
            if res.get("ok"):
                st.success(f"Scene {selected_scene_id} 動畫 Prompt 任務已啟動。log: {res.get('log')}")
            else:
                st.error(f"Scene {selected_scene_id} 動畫 Prompt 任務啟動失敗：{res.get('message')}")
            st.rerun()
    if generate_animation:
        chosen_ai_image_path = selected_image_choice_path or custom_image_val.strip()
        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
        scene_updates = {
            "source_type": source_type_val,
            "reason": reason_val,
            "animation_prompt": animation_prompt_val,
            "animation_video_path": chosen_animation_path,
        }
        if source_type_val == "FLASHCARD":
            scene_updates["flashcard_word"] = flashcard_word_val
            scene_updates["custom_image_path"] = resolve_flashcard_custom_image(
                flashcards,
                flashcard_word_val,
                str(selected_row.get("custom_image_path", "")),
            )
            scene_updates["image_prompt"] = ""
        else:
            scene_updates["flashcard_word"] = ""
            scene_updates["custom_image_path"] = chosen_ai_image_path
            scene_updates["image_prompt"] = image_prompt_val
        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
        if saved_storyboard is None or idx is None:
            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
        elif not current_image_path and not str(storyboard_scene_image_path(ep_path, latest_df.iloc[idx]) or "").strip():
            st.error("目前沒有可用圖片，請先確認圖卡或 AI 圖已就緒。")
        elif not st.session_state[animation_prompt_key].strip():
            st.error("animation_prompt 為空，請先產生或編輯動畫 Prompt。")
        else:
            res = start_publisher_task(
                info,
                scene_anim_video_task_name,
                {
                    "name": f"產生 Scene {selected_scene_id} 動畫",
                    "type": "python",
                    "script": "scripts/generate_scene_animation.py",
                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id)],
                },
            )
            if res.get("ok"):
                st.success(f"Scene {selected_scene_id} 動畫任務已啟動。log: {res.get('log')}")
            else:
                st.error(f"Scene {selected_scene_id} 動畫任務啟動失敗：{res.get('message')}")
            st.rerun()

    any_scene_worker_running = any(
        bool(task.get("running"))
        for task in [selected_scene_task, selected_anim_prompt_task, selected_anim_video_task]
    )
    scene_running_prev_key = f"storyboard_running_prev_{info['ep']}_{selected_scene_id}"
    st.session_state[scene_running_prev_key] = any_scene_worker_running
    scene_refresh_interval = 2 if (storyboard_auto_refresh and any_scene_worker_running) else None

    @st.fragment(run_every=scene_refresh_interval)
    def render_storyboard_scene_status():
        latest_storyboard_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
        latest_match = latest_storyboard_df[latest_storyboard_df["scene_id"].astype(str) == str(selected_scene_id)]
        latest_row = latest_match.iloc[0].copy() if not latest_match.empty else selected_row.copy()
        latest_scene_state = publisher_task_state(info, f"sceneimg_{selected_scene_id}")
        latest_anim_prompt_state = publisher_task_state(info, f"sceneanimprompt_{selected_scene_id}")
        latest_anim_video_state = publisher_task_state(info, f"sceneanimvideo_{selected_scene_id}")
        latest_scene_log = Path(latest_scene_state.get("log", ""))
        latest_anim_prompt_log = Path(latest_anim_prompt_state.get("log", ""))
        latest_anim_video_log = Path(latest_anim_video_state.get("log", ""))
        latest_image_path = str(storyboard_scene_image_path(ep_path, latest_row) or "").strip()
        latest_animation_path = str(storyboard_scene_animation_path(ep_path, latest_row) or "").strip()
        latest_image_candidates = scene_image_candidate_paths(ep_path, latest_row)
        latest_animation_candidates = scene_animation_candidate_paths(ep_path, latest_row)
        newest_animation_candidate = newest_existing_path(latest_animation_candidates)
        latest_any_running = any([
            latest_scene_state.get("running"),
            latest_anim_prompt_state.get("running"),
            latest_anim_video_state.get("running"),
        ])
        if st.session_state.get(scene_running_prev_key) and not latest_any_running:
            st.session_state[scene_running_prev_key] = False
            st.rerun()
        st.session_state[scene_running_prev_key] = latest_any_running

        st.subheader("Scene 執行狀態")
        status_top1, status_top2, status_top3, status_top4 = st.columns([1, 1, 1, 1.2])
        status_top1.metric("產圖任務", "執行中" if latest_scene_state.get("running") else ("已完成" if latest_scene_log.exists() else "尚未執行"))
        status_top2.metric("動畫 Prompt", "執行中" if latest_anim_prompt_state.get("running") else ("已完成" if latest_anim_prompt_log.exists() else "尚未執行"))
        status_top3.metric("動畫生成", "執行中" if latest_anim_video_state.get("running") else ("已生成候選" if newest_animation_candidate else "尚未執行"))
        status_top4.metric("候選動畫數", len(latest_animation_candidates))
        st.button("只刷新此 Scene 狀態", key=f"refresh_scene_status_{info['ep']}_{selected_scene_id}")
        st.caption(f"最新圖片路徑：{latest_image_path or '無'}")
        st.caption(f"最新動畫路徑：{latest_animation_path or '無'}")
        if newest_animation_candidate:
            st.caption(f"最新生成候選動畫：{newest_animation_candidate}")
        if latest_scene_state.get("running"):
            st.info(f"Scene {selected_scene_id} 產圖中，PID {latest_scene_state.get('pid')}")
        if latest_anim_prompt_state.get("running"):
            st.info(f"Scene {selected_scene_id} 動畫 Prompt 產生中，PID {latest_anim_prompt_state.get('pid')}")
        if latest_anim_video_state.get("running"):
            st.info(f"Scene {selected_scene_id} 動畫生成中，PID {latest_anim_video_state.get('pid')}")
        if not latest_any_running:
            st.caption("目前未在執行 Scene 相關任務。")

        if latest_animation_path and Path(latest_animation_path).exists():
            with st.expander("目前使用中的動畫", expanded=False):
                st.video(latest_animation_path)
        if newest_animation_candidate and str(newest_animation_candidate) != latest_animation_path:
            with st.expander("最新生成的候選動畫", expanded=False):
                st.video(str(newest_animation_candidate))
                st.caption("若要讓這支動畫成為此分鏡正式素材，請在上方『選擇使用動畫版本』選取後，再按『儲存此 Scene』。")

        if latest_scene_log.exists():
            with st.expander("此 Scene 產圖 Log", expanded=bool(latest_scene_state.get("running"))):
                st.code(latest_scene_log.read_text(encoding="utf-8", errors="ignore")[-4000:])
                if latest_scene_state.get("running"):
                    st.caption("執行中。此區會自動刷新。")
        if latest_anim_prompt_log.exists():
            with st.expander("此 Scene 動畫 Prompt Log", expanded=bool(latest_anim_prompt_state.get("running"))):
                st.code(latest_anim_prompt_log.read_text(encoding="utf-8", errors="ignore")[-4000:])
                if latest_anim_prompt_state.get("running"):
                    st.caption("執行中。此區會自動刷新。")
        if latest_anim_video_log.exists():
            anim_log_text = latest_anim_video_log.read_text(encoding="utf-8", errors="ignore")
            with st.expander("此 Scene 動畫生成 Log", expanded=True):
                st.code(anim_log_text[-6000:])
                if latest_anim_video_state.get("running"):
                    st.caption("執行中。此區會自動刷新。")
                if "STEP 1/4" in anim_log_text:
                    st.caption("進度提示：請看 `STEP 1/4` 到 `STEP 4/4`，以及 `DONE animation generated`。")
                if newest_animation_candidate:
                    st.caption(f"最新候選動畫檔：{newest_animation_candidate.name}")
        st.text_area(
            "字幕參考",
            value=str(latest_row.get("subtitle_reference", "")),
            height=220,
            key=f"storyboard_subref_{info['ep']}_{selected_scene_id}",
            disabled=True,
        )

        with st.expander(f"此 Scene 候選圖片（{len(latest_image_candidates)}）", expanded=False):
            if latest_image_candidates:
                image_groups = [latest_image_candidates[i:i + 3] for i in range(0, len(latest_image_candidates), 3)]
                for group in image_groups:
                    cols = st.columns(3)
                    for col_idx, candidate_path in enumerate(group):
                        with cols[col_idx]:
                            render_thumbnail_card(
                                candidate_path,
                                caption_text=candidate_path.name,
                                current=(str(candidate_path) == latest_image_path),
                                max_width=320,
                            )
            else:
                st.caption("尚無候選圖片。")

        with st.expander(f"此 Scene 候選動畫（{len(latest_animation_candidates)}）", expanded=False):
            if latest_animation_candidates:
                animation_groups = [latest_animation_candidates[i:i + 2] for i in range(0, len(latest_animation_candidates), 2)]
                for group in animation_groups:
                    cols = st.columns(2)
                    for col_idx, candidate_path in enumerate(group):
                        with cols[col_idx]:
                            with st.container(border=True):
                                st.caption("目前使用中" if str(candidate_path) == latest_animation_path else candidate_path.name)
                                st.video(str(candidate_path))
            else:
                st.caption("尚無候選動畫。")

        with st.expander(f"每個分鏡的圖片總覽（{len(latest_storyboard_df)}）", expanded=False):
            gallery_rows = [latest_storyboard_df.iloc[i:i + 3] for i in range(0, len(latest_storyboard_df), 3)]
            for row_group in gallery_rows:
                gallery_cols = st.columns(3)
                for col_idx, (_, row) in enumerate(row_group.iterrows()):
                    img_path = storyboard_scene_image_path(ep_path, row)
                    anim_path = storyboard_scene_animation_path(ep_path, row)
                    with gallery_cols[col_idx]:
                        with st.container(border=True):
                            st.caption(
                                f"Scene {row.get('scene_id')} | {str(row.get('source_type', '')).upper()} | "
                                f"{row.get('start_time')} - {row.get('end_time')}"
                            )
                            if anim_path and anim_path.exists():
                                st.caption("已生成動畫")
                            if img_path and img_path.exists():
                                render_safe_image_preview(
                                    img_path,
                                    empty_message="尚無圖片",
                                    broken_hint="分鏡圖片無法讀取",
                                    caption_label="分鏡預覽",
                                    max_width=280,
                                )
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
            "animation_prompt": st.column_config.TextColumn(width="large"),
            "animation_video_path": st.column_config.TextColumn(width="large"),
            "reason": st.column_config.TextColumn(width="medium"),
            "subtitle_reference": st.column_config.TextColumn(width="large"),
        },
        key=f"storyboard_editor_{info['ep']}_{profile_id}",
    )
    if st.button("儲存整張分鏡表", key=f"save_storyboard_all_{info['ep']}_{profile_id}"):
        saved_storyboard = save_storyboard_df(ep_path, edited_storyboard_df)
        st.success(f"已儲存分鏡表：{saved_storyboard}")
        st.rerun()

def notebooklm_prompt_template_path(ep_path: Path) -> Path:
    return ep_path / "notebooklm_prompt_template.txt"

def ensure_notebooklm_prompt_template_file(ep_path: Path) -> Path:
    prompt_path = notebooklm_prompt_template_path(ep_path)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_NOTEBOOKLM_PROMPT_TEMPLATE)
    return prompt_path

def notebooklm_prompt_output_path(ep_path: Path, ep_num: int) -> Path:
    return ep_path / f"notebooklm_prompt_ep{ep_num:02d}.txt"

def subtitle_review_prompt_path(ep_path: Path) -> Path:
    return ep_path / "02_subtitles" / "subtitle_review_prompt.txt"

def ensure_subtitle_review_prompt_file(ep_path: Path) -> Path:
    prompt_path = subtitle_review_prompt_path(ep_path)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE)
    return prompt_path

def storyboard_prompt_path(ep_path: Path) -> Path:
    return ep_path / "03_storyboards" / "storyboard_prompt.txt"

def ensure_storyboard_prompt_file(ep_path: Path) -> Path:
    prompt_path = storyboard_prompt_path(ep_path)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_STORYBOARD_PROMPT_TEMPLATE)
    return prompt_path

def youtube_meta_prompt_path(ep_path: Path) -> Path:
    return ep_path / "05_output" / "youtube_meta_prompt.txt"

def ensure_youtube_meta_prompt_file(ep_path: Path) -> Path:
    prompt_path = youtube_meta_prompt_path(ep_path)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE)
    return prompt_path

def sync_textarea_state(text_key: str, loaded_key: str, file_text: str) -> None:
    if text_key not in st.session_state:
        st.session_state[text_key] = file_text
        st.session_state[loaded_key] = file_text
    elif st.session_state.get(loaded_key) != file_text:
        st.session_state[text_key] = file_text
        st.session_state[loaded_key] = file_text

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

def format_wallclock_ts(value) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"

def format_elapsed_seconds(started_at, ended_at = None) -> str:
    try:
        start_val = float(started_at)
    except Exception:
        return "—"
    end_source = ended_at if ended_at is not None else time.time()
    try:
        end_val = float(end_source)
    except Exception:
        return "—"
    elapsed = max(end_val - start_val, 0.0)
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    mins, secs = divmod(elapsed, 60)
    if mins < 60:
        return f"{int(mins)}m {secs:.1f}s"
    hours, mins = divmod(mins, 60)
    return f"{int(hours)}h {int(mins)}m {secs:.1f}s"

def prompt_log_summary(state: Dict, log_path: Path) -> Dict[str, str]:
    started_at = state.get("started_at")
    ended_at = state.get("ended_at")
    if ended_at is None and log_path.exists() and not state.get("running"):
        ended_at = log_path.stat().st_mtime
    return {
        "start": format_wallclock_ts(started_at),
        "end": "執行中" if state.get("running") else format_wallclock_ts(ended_at),
        "elapsed": format_elapsed_seconds(started_at, None if state.get("running") else ended_at),
    }

def load_vocab_df_for_prompt(ep_path: Path) -> pd.DataFrame:
    vocab_path = ep_path / "03_storyboards" / "vocab_data.csv"
    if not vocab_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(vocab_path).fillna("")
    except Exception:
        return pd.DataFrame()

def load_first_cloze_question_for_prompt(ep_path: Path) -> Dict | None:
    json_path = cloze_question_json_path(ep_path)
    csv_path = cloze_question_csv_path(ep_path)
    if json_path.exists():
        payload = read_json_file(json_path)
        if isinstance(payload, list) and payload:
            return payload[0]
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path).fillna("")
            if not df.empty:
                return df.iloc[0].to_dict()
        except Exception:
            return None
    return None

def find_episode_path_by_number(ep_num: int) -> Path | None:
    matches = find_episode_dirs_by_ep(ROOT, ep_num, WS_ROOT)
    return matches[0] if matches else None

def build_vocab_list_text_for_prompt(vocab_df: pd.DataFrame) -> str:
    if vocab_df.empty:
        return ""
    lines = []
    for _, row in vocab_df.iterrows():
        lines.append(f"- 單字：{row.get('Word', '')} ({row.get('POS', '')}) {row.get('Meaning', '')}")
        lines.append(f"  > 英文例句：{row.get('English_Sentence', '')}")
        lines.append(f"  > 中文翻譯：{row.get('Chinese_Translation', '')}")
        lines.append("")
    return "\n".join(lines).strip()

def build_previous_cloze_section_for_prompt(ep_num: int) -> str:
    if ep_num <= 1:
        return ""
    prev_ep_path = find_episode_path_by_number(ep_num - 1)
    if not prev_ep_path:
        return ""
    question = load_first_cloze_question_for_prompt(prev_ep_path)
    if not question:
        return ""
    answer_surface = question.get("correct_surface_word", question.get("surface_word", ""))
    return (
        "【前一集克漏字解答安排】\n"
        "如果前一集節目結束前有出克漏字題，請在本集開頭先提醒同學：這題是上一集節目結束前留給大家作答的題目。\n"
        "- 請完整念出題目與四個選項。\n"
        "- 接著公布正確答案，並詳細說明為什麼這個答案正確。\n"
        "- 解說時請結合句意、詞性、中文意思與常見混淆點。\n"
        "- 語氣要像節目主持人在揭曉答案，不要像在念考卷解析。\n\n"
        "前一集題目資料：\n"
        f"- 來自第 {ep_num - 1} 集的第 1 題\n"
        f"- 題目：{question.get('blank_sentence', '')}\n"
        f"- A：{question.get('choice_A', '')}\n"
        f"- B：{question.get('choice_B', '')}\n"
        f"- C：{question.get('choice_C', '')}\n"
        f"- D：{question.get('choice_D', '')}\n"
        f"- 正確答案：{question.get('correct_option', '')}. {answer_surface}\n"
        f"- 解說重點：{question.get('explanation', '')}"
    )

def build_next_cloze_section_for_prompt(ep_path: Path, ep_num: int) -> str:
    question = load_first_cloze_question_for_prompt(ep_path)
    if not question:
        return ""
    return (
        "【本集結尾克漏字預告安排】\n"
        "在節目結束前，請從本集克漏字題庫取出第 1 題當作節目尾聲的小測驗。\n"
        "- 請明確說明：這題是今天節目結束前要留給同學作答的題目，答案會在下一集開頭公布。\n"
        "- 請把題目和四個選項完整念出來。\n"
        "- 不要直接公布正確答案。\n"
        "- 結尾要提醒同學先自己想想看，下一集會正式揭曉並詳細解析。\n\n"
        "本集要出的題目資料：\n"
        f"- 來自第 {ep_num} 集的第 1 題\n"
        f"- 題目：{question.get('blank_sentence', '')}\n"
        f"- A：{question.get('choice_A', '')}\n"
        f"- B：{question.get('choice_B', '')}\n"
        f"- C：{question.get('choice_C', '')}\n"
        f"- D：{question.get('choice_D', '')}"
    )

def build_storyboard_cloze_rules_for_prompt(ep_path: Path, ep_num: int) -> str:
    rules = []
    prev_ep_path = find_episode_path_by_number(ep_num - 1) if ep_num > 1 else None
    if prev_ep_path and load_first_cloze_question_for_prompt(prev_ep_path):
        rules.append(
            '- If the previous episode has a cloze answer card, reserve one FLASHCARD scene for it with source_type set to "FLASHCARD" and flashcard_word set to "__PREV_CLOZE_Q1_ANSWER__".'
        )
        rules.append("- Use that special token only once, for the opening answer-reveal segment of the episode.")
    if load_first_cloze_question_for_prompt(ep_path):
        rules.append(
            '- If this episode has a cloze question card, reserve one FLASHCARD scene for it with source_type set to "FLASHCARD" and flashcard_word set to "__CURRENT_CLOZE_Q1_QUESTION__".'
        )
        rules.append("- Use that special token only once, for the closing teaser question segment of the episode.")
    if not rules:
        rules.append("- No special cloze-card flashcard scenes are required for this episode.")
    return "\n".join(rules)

def build_storyboard_host_rules_for_prompt() -> str:
    host_profile_path = ROOT / "core" / "assets" / "host_profiles.json"
    payload = read_json_file(host_profile_path) or {}
    style = str(payload.get("style_config", "")).strip()
    characters = payload.get("characters") or {}
    john_desc = str((characters.get("john") or {}).get("description", "")).strip()
    mary_desc = str((characters.get("mary") or {}).get("description", "")).strip()
    if not john_desc or not mary_desc:
        return ""
    rules = [
        "Host consistency rules for AI scenes:",
        f"- Overall style: {style}" if style else "- Keep one stable visual style for host scenes.",
        f"- John: {john_desc}",
        f"- Mary: {mary_desc}",
        "- If an AI scene includes the podcast hosts, presenters, or generic on-screen narrators, use John and Mary instead of anonymous people.",
        "- When both hosts appear together, default to John on the left and Mary on the right unless the action clearly needs a different arrangement.",
        "- Put the host descriptions directly inside image_prompt instead of generic phrases like 'a man and a woman' or 'podcast hosts'.",
        "- If a scene is purely object-based, abstract, or clearly requires another role, do not force the hosts into that scene.",
    ]
    return "\n".join(rules)

def render_step5_prompt_preview(template_text: str, ep_path: Path) -> str:
    ep_info = parse_episode_info(ep_path)
    vocab_df = load_vocab_df_for_prompt(ep_path)
    vocab_json = vocab_df.to_json(orient="records", force_ascii=False, indent=2) if not vocab_df.empty else "[]"
    return (
        template_text
        .replace("{{EPISODE}}", str(ep_info.get("ep") or ""))
        .replace("{{START_WORD}}", str(ep_info.get("start") or ""))
        .replace("{{END_WORD}}", str(ep_info.get("end") or ""))
        .replace("{{VOCAB_COUNT}}", str(len(vocab_df)))
        .replace("{{VOCAB_JSON}}", vocab_json)
    )

def render_step8_prompt_preview(template_text: str, ep_path: Path) -> str:
    ep_info = parse_episode_info(ep_path)
    vocab_df = load_vocab_df_for_prompt(ep_path)
    previous_section = build_previous_cloze_section_for_prompt(int(ep_info.get("ep") or 0))
    next_section = build_next_cloze_section_for_prompt(ep_path, int(ep_info.get("ep") or 0))
    rendered = (
        template_text
        .replace("{{EPISODE}}", str(ep_info.get("ep") or ""))
        .replace("{{START_WORD}}", str(ep_info.get("start") or ""))
        .replace("{{END_WORD}}", str(ep_info.get("end") or ""))
        .replace("{{VOCAB_LIST}}", build_vocab_list_text_for_prompt(vocab_df))
        .replace("{{PREVIOUS_EPISODE_CLOZE_SECTION}}", previous_section.strip())
        .replace("{{NEXT_EPISODE_CLOZE_SECTION}}", next_section.strip())
    )
    fallback_sections = []
    if "{{PREVIOUS_EPISODE_CLOZE_SECTION}}" not in template_text and previous_section.strip():
        fallback_sections.append(previous_section.strip())
    if "{{NEXT_EPISODE_CLOZE_SECTION}}" not in template_text and next_section.strip():
        fallback_sections.append(next_section.strip())
    if fallback_sections:
        rendered += "\n\n【額外節目安排】\n" + "\n\n".join(fallback_sections)
    return rendered.strip() + "\n"

def render_step12_prompt_preview(template_text: str, subtitle_path: Path) -> str:
    return template_text.replace("{{SRT_CONTENT}}", read_text_file(subtitle_path, ""))

def render_step14_prompt_preview(template_text: str, ep_path: Path, subtitle_path: Path) -> str:
    vocab_df = load_vocab_df_for_prompt(ep_path)
    word_list = vocab_df["Word"].astype(str).str.strip().tolist() if not vocab_df.empty and "Word" in vocab_df.columns else []
    cloze_rules = build_storyboard_cloze_rules_for_prompt(ep_path, int(parse_episode_info(ep_path).get("ep") or 0))
    host_rules = build_storyboard_host_rules_for_prompt()
    rendered = (
        template_text
        .replace("{{WORD_LIST}}", json.dumps(word_list, ensure_ascii=False, indent=2))
        .replace("{{CLOZE_CARD_RULES}}", cloze_rules)
        .replace("{{HOST_PROFILE_RULES}}", host_rules)
        .replace("{{SRT_CONTENT}}", read_text_file(subtitle_path, ""))
    )
    if "{{CLOZE_CARD_RULES}}" not in template_text and cloze_rules.strip():
        rendered += "\n\nCloze card rules:\n" + cloze_rules.strip()
    if "{{HOST_PROFILE_RULES}}" not in template_text and host_rules.strip():
        rendered += "\n\nHost profile rules:\n" + host_rules.strip()
    rendered += (
        "\n\nTimeline rules:\n"
        "- Every scene must stay within the actual SRT timeline.\n"
        "- Do not invent extra intro/outro scenes after the last subtitle ends.\n"
        "- If the episode ends at the last subtitle, the storyboard must also end there.\n"
    )
    return rendered

def render_step19_prompt_preview(template_text: str, ep_path: Path, subtitle_path: Path) -> str:
    ep_info = parse_episode_info(ep_path)
    return (
        template_text
        .replace("{{START_WORD}}", str(ep_info.get("start") or ""))
        .replace("{{END_WORD}}", str(ep_info.get("end") or ""))
        .replace("{{SRT_CONTENT}}", read_text_file(subtitle_path, ""))
    )

def flashcard_map_for(ep_path: Path) -> Dict[str, str]:
    folder = images_dir(ep_path) / "flashcards"
    items = {}
    if folder.exists():
        for p in sorted(folder.glob("*.*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                items[p.stem] = str(p.resolve()).replace("\\", "/")
    return items

def resolve_flashcard_custom_image(flashcards: Dict[str, str], flashcard_word: str, fallback_path: str = "") -> str:
    word = str(flashcard_word or "").strip()
    if not word:
        return ""
    if word in flashcards:
        return flashcards[word]
    return str(fallback_path or "").strip()

def ai_scene_image_path(ep_path: Path, start_time) -> Path:
    start_token = str(start_time).strip()
    return images_dir(ep_path) / "ai_generated" / f"img_{start_token}.png"

def animation_dir_for(ep_path: Path) -> Path:
    return images_dir(ep_path) / "animations"

def ai_scene_animation_path(ep_path: Path, row: pd.Series | Dict) -> Path:
    scene_id = str(row.get("scene_id", "")).strip()
    if scene_id:
        return animation_dir_for(ep_path) / f"scene_{scene_id}.mp4"
    start_token = str(row.get("start_time", "")).strip()
    return animation_dir_for(ep_path) / f"scene_{start_token}.mp4"

def _dedupe_existing_paths(paths: list[Path | None]) -> list[Path]:
    items = []
    seen = set()
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        if p.exists():
            items.append(p)
            seen.add(key)
    return items

def scene_image_candidate_paths(ep_path: Path, row: pd.Series | Dict) -> list[Path]:
    source_type = str(row.get("source_type", "")).strip().upper()
    if source_type == "FLASHCARD":
        return _dedupe_existing_paths([storyboard_scene_image_path(ep_path, row)])

    base_path = ai_scene_image_path(ep_path, row.get("start_time", ""))
    variant_paths = sorted(base_path.parent.glob(f"{base_path.stem}__v*{base_path.suffix}"))
    custom_path = str(row.get("custom_image_path", "")).strip()
    return _dedupe_existing_paths(
        ([Path(custom_path)] if custom_path else []) + [base_path] + variant_paths
    )

def scene_animation_candidate_paths(ep_path: Path, row: pd.Series | Dict) -> list[Path]:
    base_path = ai_scene_animation_path(ep_path, row)
    variant_paths = sorted(base_path.parent.glob(f"{base_path.stem}__v*{base_path.suffix}"))
    custom_path = str(row.get("animation_video_path", "")).strip()
    return _dedupe_existing_paths(
        ([Path(custom_path)] if custom_path else []) + [base_path] + variant_paths
    )

def asset_choice_label(path: Path, base_path: Path | None, kind: str) -> str:
    if base_path and path.resolve() == base_path.resolve():
        return f"原始{kind} | {path.name}"
    m = re.search(r"__v(\d+)$", path.stem)
    if m:
        return f"{kind}變體 v{m.group(1)} | {path.name}"
    return f"自訂{kind} | {path.name}"

def newest_existing_path(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)

def storyboard_scene_animation_path(ep_path: Path, row: pd.Series | Dict) -> Path | None:
    custom_path = str(row.get("animation_video_path", "")).strip()
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    auto_path = ai_scene_animation_path(ep_path, row)
    return auto_path if auto_path.exists() else None

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
        "animation_prompt",
        "animation_video_path",
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
        anim_path = str(out.at[idx, "animation_video_path"]).strip()
        if anim_path:
            out.at[idx, "animation_video_path"] = str(Path(anim_path)).replace("\\", "/")
    return out

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

@st.cache_data(show_spinner=False, max_entries=128)
def media_duration_seconds(path_str: str) -> float | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except Exception:
        return None
    duration = safe_float((result.stdout or "").strip(), -1.0)
    return duration if duration >= 0 else None

def file_to_data_uri(path: Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def path_to_file_uri(path: Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""

@st.cache_data(show_spinner=False, max_entries=64)
def video_to_data_uri(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return file_to_data_uri(path)

@st.cache_data(show_spinner=False, max_entries=256)
def image_to_preview_data_uri(path_str: str, max_width: int = 960, quality: int = 78) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    try:
        with Image.open(path) as img:
            img.load()
            preview = img.convert("RGB")
            if max_width > 0 and preview.width > max_width:
                new_height = max(1, int(preview.height * (max_width / preview.width)))
                preview = preview.resize((max_width, new_height), Image.Resampling.LANCZOS)
            buf = BytesIO()
            preview.save(buf, format="JPEG", quality=quality, optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

def build_episode_preview_payload(ep_path: Path, storyboard_df: pd.DataFrame, subtitle_entries: list[Dict]) -> Dict:
    image_cache: Dict[str, str] = {}
    video_cache: Dict[str, str] = {}
    animation_paths = []
    for _, row in storyboard_df.iterrows():
        animation_path = storyboard_scene_animation_path(ep_path, row)
        if animation_path and animation_path.exists():
            animation_paths.append(animation_path)
    unique_animation_paths = {str(p.resolve()): p for p in animation_paths}
    total_animation_bytes = sum(p.stat().st_size for p in unique_animation_paths.values())
    inline_animation_limit = 25 * 1024 * 1024
    allow_inline_animations = total_animation_bytes <= inline_animation_limit
    scenes = []
    for _, row in storyboard_df.iterrows():
        image_path = storyboard_scene_image_path(ep_path, row)
        animation_path = storyboard_scene_animation_path(ep_path, row)
        image_path_str = str(image_path) if image_path else ""
        animation_path_str = str(animation_path) if animation_path else ""
        animation_data_uri = ""
        if allow_inline_animations and animation_path_str:
            if animation_path_str not in video_cache:
                video_cache[animation_path_str] = video_to_data_uri(animation_path_str)
            animation_data_uri = video_cache[animation_path_str]
        if image_path_str not in image_cache:
            image_cache[image_path_str] = image_to_preview_data_uri(image_path_str) if image_path_str else ""
        scenes.append({
            "scene_id": str(row.get("scene_id", "")),
            "start": safe_float(row.get("start_time", 0)),
            "end": safe_float(row.get("end_time", 0)),
            "source_type": str(row.get("source_type", "")).upper(),
            "flashcard_word": str(row.get("flashcard_word", "")),
            "reason": str(row.get("reason", "")),
            "image_prompt": str(row.get("image_prompt", "")),
            "animation_prompt": str(row.get("animation_prompt", "")),
            "subtitle_reference": str(row.get("subtitle_reference", "")),
            "image_path": image_path_str,
            "image_data_uri": image_cache[image_path_str],
            "animation_path": animation_path_str,
            "animation_data_uri": animation_data_uri,
            "asset_mode": "ANIMATION" if animation_path_str else "IMAGE",
        })
    subtitles = [{
        "index": str(row.get("index", "")),
        "start": safe_float(row.get("start", 0)),
        "end": safe_float(row.get("end", 0)),
        "content": str(row.get("content", "")),
    } for row in subtitle_entries]
    return {
        "scenes": scenes,
        "subtitles": subtitles,
        "allow_inline_animations": allow_inline_animations,
        "total_animation_bytes": total_animation_bytes,
    }

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
        .cap-media video {{
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
            <video id="cap-scene-video" muted playsinline loop style="display:none;"></video>
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
      const sceneVideo = document.getElementById("cap-scene-video");
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
      let activeSceneVideoSrc = "";

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
            sourceTypeEl.textContent = `${{scene.source_type || "-"}} / ${{scene.asset_mode || "-"}}`;
            imageRefEl.textContent = scene.asset_mode === "ANIMATION"
              ? (scene.animation_path || "-")
              : (scene.source_type === "FLASHCARD" ? (scene.flashcard_word || scene.image_path || "-") : (scene.image_path || "-"));
            reasonEl.textContent = scene.reason || "-";
            promptEl.textContent = scene.asset_mode === "ANIMATION"
              ? (scene.animation_prompt || scene.image_prompt || "-")
              : (scene.image_prompt || "-");
            if (scene.asset_mode === "ANIMATION" && scene.animation_file_uri) {{
              if (activeSceneVideoSrc !== scene.animation_file_uri) {{
                sceneVideo.src = scene.animation_file_uri;
                activeSceneVideoSrc = scene.animation_file_uri;
              }}
              const sceneOffset = Math.max(currentTime - Number(scene.start || 0), 0);
              const syncSceneVideo = () => {{
                if (sceneVideo.duration && Number.isFinite(sceneVideo.duration) && sceneVideo.duration > 0) {{
                  const targetTime = sceneOffset % sceneVideo.duration;
                  if (Math.abs((sceneVideo.currentTime || 0) - targetTime) > 0.4) {{
                    sceneVideo.currentTime = targetTime;
                  }}
                }}
              }};
              if (sceneVideo.readyState >= 1) {{
                syncSceneVideo();
              }} else {{
                sceneVideo.onloadedmetadata = () => syncSceneVideo();
              }}
              sceneVideo.style.display = "block";
              sceneImage.removeAttribute("src");
              sceneImage.style.display = "none";
              sceneEmpty.style.display = "none";
              sceneVideo.play().catch(() => null);
            }} else if (scene.image_data_uri) {{
              if (!sceneVideo.paused) sceneVideo.pause();
              sceneVideo.removeAttribute("src");
              activeSceneVideoSrc = "";
              sceneVideo.style.display = "none";
              sceneImage.src = scene.image_data_uri;
              sceneImage.style.display = "block";
              sceneEmpty.style.display = "none";
            }} else {{
              if (!sceneVideo.paused) sceneVideo.pause();
              sceneVideo.removeAttribute("src");
              activeSceneVideoSrc = "";
              sceneVideo.style.display = "none";
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
            if (!sceneVideo.paused) sceneVideo.pause();
            sceneVideo.removeAttribute("src");
            activeSceneVideoSrc = "";
            sceneVideo.style.display = "none";
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

        const activeScene = scenes[activeSceneIndex];
        if (activeScene && activeScene.asset_mode === "ANIMATION" && sceneVideo.src) {{
          const activeOffset = Math.max(currentTime - Number(activeScene.start || 0), 0);
          if (sceneVideo.duration && Number.isFinite(sceneVideo.duration) && sceneVideo.duration > 0) {{
            const targetTime = activeOffset % sceneVideo.duration;
            if (Math.abs((sceneVideo.currentTime || 0) - targetTime) > 0.4) {{
              sceneVideo.currentTime = targetTime;
            }}
          }}
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
      audio.addEventListener("pause", () => {{
        if (!sceneVideo.paused) sceneVideo.pause();
      }});
      audio.addEventListener("play", () => {{
        if (sceneVideo.src) sceneVideo.play().catch(() => null);
      }});
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

def renumber_storyboard_scenes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["scene_id"] = [str(i + 1) for i in range(len(out))]
    return out

def _storyboard_join_text(values, *, sep: str = " | ") -> str:
    items = []
    seen = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
    return sep.join(items)

def subtitle_reference_from_entries(entries: list[Dict]) -> str:
    parts = []
    for row in entries:
        start_label = str(row.get("start_label", "")).strip()
        end_label = str(row.get("end_label", "")).strip()
        content = str(row.get("content", "")).replace("\n", " ").strip()
        if not content:
            continue
        if start_label and end_label:
            parts.append(f"[{start_label}-->{end_label}] {content}")
        else:
            parts.append(content)
    return " | ".join(parts)

def scene_subtitle_reference_for_window(ep_path: Path, start_sec, end_sec) -> str:
    subtitle_path = subtitles_fixed_path(ep_path)
    subtitle_entries = parse_srt_entries(subtitle_path)
    if not subtitle_entries:
        return ""
    matched = subtitles_for_scene(subtitle_entries, start_sec, end_sec)
    return subtitle_reference_from_entries(matched)

def _storyboard_structural_defaults(ep_path: Path, row: pd.Series | Dict) -> Dict:
    source_type = str(row.get("source_type", "")).strip().upper() or "AI"
    image_path = storyboard_scene_image_path(ep_path, row)
    custom_image_path = str(image_path).replace("\\", "/") if image_path else ""
    if source_type == "FLASHCARD":
        return {
            "source_type": "FLASHCARD",
            "flashcard_word": str(row.get("flashcard_word", "")).strip(),
            "custom_image_path": custom_image_path or str(row.get("custom_image_path", "")).strip(),
            "image_prompt": "",
            "animation_prompt": str(row.get("animation_prompt", "")).strip(),
            "animation_video_path": "",
        }
    return {
        "source_type": "AI",
        "flashcard_word": "",
        "custom_image_path": custom_image_path,
        "image_prompt": str(row.get("image_prompt", "")).strip(),
        "animation_prompt": str(row.get("animation_prompt", "")).strip(),
        "animation_video_path": "",
    }

def merge_storyboard_scenes(
    ep_path: Path,
    start_scene_id,
    end_scene_id,
    *,
    carry_mode: str = "first",
    merged_reason: str = "",
) -> tuple[Path | None, pd.DataFrame | None]:
    latest_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
    if latest_df.empty:
        return None, None

    order = latest_df.reset_index(drop=True)
    start_match = order.index[order["scene_id"].astype(str) == str(start_scene_id)]
    end_match = order.index[order["scene_id"].astype(str) == str(end_scene_id)]
    if len(start_match) == 0 or len(end_match) == 0:
        return None, latest_df
    start_idx = int(start_match[0])
    end_idx = int(end_match[0])
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    selected = order.iloc[start_idx:end_idx + 1].copy()
    if len(selected) < 2:
        return None, latest_df

    if carry_mode == "last":
        base_row = selected.iloc[-1]
    elif carry_mode == "clear_ai":
        base_row = pd.Series({
            "source_type": "AI",
            "flashcard_word": "",
            "custom_image_path": "",
            "image_prompt": "",
            "animation_prompt": "",
            "animation_video_path": "",
        })
    else:
        base_row = selected.iloc[0]

    merged_row = {
        "scene_id": str(selected.iloc[0].get("scene_id", "")),
        "start_time": safe_float(selected["start_time"].iloc[0]),
        "end_time": safe_float(selected["end_time"].iloc[-1]),
        "reason": merged_reason.strip() or _storyboard_join_text(selected["reason"].tolist(), sep=" / "),
        "subtitle_reference": "",
    }
    merged_row.update(_storyboard_structural_defaults(ep_path, base_row))
    merged_row["subtitle_reference"] = (
        scene_subtitle_reference_for_window(ep_path, merged_row["start_time"], merged_row["end_time"])
        or _storyboard_join_text(selected["subtitle_reference"].tolist(), sep=" | ")
    )

    before = order.iloc[:start_idx].copy()
    after = order.iloc[end_idx + 1:].copy()
    merged_df = pd.concat([before, pd.DataFrame([merged_row]), after], ignore_index=True)
    merged_df = renumber_storyboard_scenes(merged_df)
    saved_storyboard = save_storyboard_df(ep_path, merged_df)
    return saved_storyboard, merged_df

def build_split_scene_draft(ep_path: Path, row: pd.Series | Dict, split_count: int) -> pd.DataFrame:
    split_count = max(2, int(split_count))
    start_time = safe_float(row.get("start_time", 0.0))
    end_time = safe_float(row.get("end_time", start_time))
    duration = max(end_time - start_time, 0.01)
    step = duration / split_count
    base_reason = str(row.get("reason", "")).strip()
    structural = _storyboard_structural_defaults(ep_path, row)
    rows = []
    for idx in range(split_count):
        seg_start = round(start_time + (step * idx), 3)
        seg_end = round(end_time if idx == split_count - 1 else (start_time + (step * (idx + 1))), 3)
        item = {
            "scene_id": "",
            "start_time": seg_start,
            "end_time": seg_end,
            "reason": f"{base_reason}（段 {idx + 1}/{split_count}）" if base_reason else f"切分段 {idx + 1}/{split_count}",
            "subtitle_reference": scene_subtitle_reference_for_window(ep_path, seg_start, seg_end),
        }
        item.update(structural)
        item["animation_video_path"] = ""
        rows.append(item)
    return pd.DataFrame(rows)

def split_storyboard_scene(
    ep_path: Path,
    scene_id,
    segments_df: pd.DataFrame,
) -> tuple[Path | None, pd.DataFrame | None]:
    latest_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
    if latest_df.empty:
        return None, None
    match_idx = latest_df.index[latest_df["scene_id"].astype(str) == str(scene_id)]
    if len(match_idx) == 0:
        return None, latest_df
    idx = int(match_idx[0])
    cleaned_segments = normalize_storyboard_df(ep_path, segments_df.copy())
    cleaned_segments["start_time"] = cleaned_segments["start_time"].apply(lambda v: round(safe_float(v), 3))
    cleaned_segments["end_time"] = cleaned_segments["end_time"].apply(lambda v: round(safe_float(v), 3))
    cleaned_segments = cleaned_segments.sort_values(["start_time", "end_time"]).reset_index(drop=True)
    cleaned_segments["subtitle_reference"] = cleaned_segments.apply(
        lambda row: scene_subtitle_reference_for_window(ep_path, row["start_time"], row["end_time"])
        or str(row.get("subtitle_reference", "")).strip(),
        axis=1,
    )
    before = latest_df.iloc[:idx].copy()
    after = latest_df.iloc[idx + 1:].copy()
    merged_df = pd.concat([before, cleaned_segments, after], ignore_index=True)
    merged_df = renumber_storyboard_scenes(merged_df)
    saved_storyboard = save_storyboard_df(ep_path, merged_df)
    return saved_storyboard, merged_df

def update_storyboard_scene(ep_path: Path, scene_id, updates: Dict) -> tuple[Path | None, pd.DataFrame | None, int | None]:
    latest_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
    if latest_df.empty:
        return None, None, None
    match_idx = latest_df.index[latest_df["scene_id"].astype(str) == str(scene_id)]
    if len(match_idx) == 0:
        return None, latest_df, None
    idx = match_idx[0]
    for key, value in updates.items():
        if key in latest_df.columns:
            latest_df.at[idx, key] = value
    saved_storyboard = save_storyboard_df(ep_path, latest_df)
    return saved_storyboard, latest_df, idx

def publisher_task_state(info: Dict, task_name: str) -> Dict:
    return runner.get_substep_state(info, f"pub_{task_name}")

def start_publisher_task(info: Dict, task_name: str, step_def: Dict) -> Dict:
    return runner.start_substep(info, f"pub_{task_name}", step_def)

# convenience
STAGE_IDS = ["1","1.5","2","3","4","5","6","7"]

# --- UI ---
if section == "📊 Dashboard":
    st.header("專案與進度總覽")
    if st.session_state["profile_id"] == "vocab":
        with st.expander("第 6 集起單字集數建立工具", expanded=True):
            st.caption("第 1–5 集維持既有每集 30 字。第 6 集起，系統會以第 151 字為起點，依你輸入的每集單字數連續推算區間。")
            notice = st.session_state.pop("episode_generation_notice", None)
            if isinstance(notice, dict):
                if notice.get("level") == "success":
                    st.success(notice.get("message", ""))
                elif notice.get("level") == "warning":
                    st.warning(notice.get("message", ""))
                elif notice.get("level") == "error":
                    st.error(notice.get("message", ""))

            single_tab, batch_tab = st.tabs(["產生單集", "產生多集"])

            with single_tab:
                single_ep = int(st.number_input("產生集數", min_value=VOCAB_REBASE_EP, value=VOCAB_REBASE_EP, step=1, key="gen_single_ep"))
                single_words = int(st.number_input("單集單字數", min_value=1, value=10, step=1, key="gen_single_words"))
                single_allow_backup = st.checkbox(
                    "若該集已有不同區間資料夾，先備份舊資料夾再重建",
                    value=False,
                    key="gen_single_allow_backup",
                )
                single_plans = build_episode_generation_plan(single_ep, single_ep, single_words)
                st.dataframe(generation_plan_table(single_plans), use_container_width=True, hide_index=True)
                if single_plans[0]["conflict_dirs"] and not single_allow_backup:
                    st.warning("此集已有不同區間的既有資料夾。若要套用新區間，請勾選備份舊資料夾。")
                if st.button("建立 / 更新單集", key="gen_single_submit"):
                    result = apply_episode_generation(single_plans, single_allow_backup)
                    if result["conflicts"]:
                        blocked_eps = ", ".join(f"Ep{p['ep']:02d}" for p in result["conflicts"])
                        st.session_state["episode_generation_notice"] = {
                            "level": "error",
                            "message": f"{blocked_eps} 有既有區間衝突，尚未變更。請勾選備份舊資料夾後再執行。",
                        }
                    else:
                        parts = []
                        if result["created"]:
                            parts.append(f"新建 {len(result['created'])} 集")
                        if result["reused"]:
                            parts.append(f"沿用 {len(result['reused'])} 集")
                        if result["backed_up"]:
                            parts.append(f"備份舊資料夾 {len(result['backed_up'])} 個")
                        st.session_state["episode_generation_notice"] = {
                            "level": "success",
                            "message": "；".join(parts) or "單集設定已完成。",
                        }
                    st.rerun()

            with batch_tab:
                batch_start_ep = int(st.number_input("產生起始集數", min_value=VOCAB_REBASE_EP, value=VOCAB_REBASE_EP, step=1, key="gen_batch_start"))
                batch_end_ep = int(st.number_input("產生結束集數", min_value=batch_start_ep, value=batch_start_ep + 4, step=1, key="gen_batch_end"))
                batch_words = int(st.number_input("每集單字數", min_value=1, value=10, step=1, key="gen_batch_words"))
                batch_allow_backup = st.checkbox(
                    "若既有集數資料夾區間不同，先備份舊資料夾再重建",
                    value=False,
                    key="gen_batch_allow_backup",
                )
                batch_plans = build_episode_generation_plan(batch_start_ep, batch_end_ep, batch_words)
                st.dataframe(generation_plan_table(batch_plans), use_container_width=True, hide_index=True)
                batch_conflicts = [plan for plan in batch_plans if plan["conflict_dirs"]]
                if batch_conflicts and not batch_allow_backup:
                    st.warning(f"目前有 {len(batch_conflicts)} 集存在區間衝突。若要套用新規則，請勾選備份舊資料夾。")
                if st.button("建立 / 更新多集", key="gen_batch_submit"):
                    result = apply_episode_generation(batch_plans, batch_allow_backup)
                    if result["conflicts"]:
                        blocked_eps = ", ".join(f"Ep{p['ep']:02d}" for p in result["conflicts"])
                        st.session_state["episode_generation_notice"] = {
                            "level": "error",
                            "message": f"以下集數因區間衝突而未變更：{blocked_eps}。請勾選備份舊資料夾後再執行。",
                        }
                    else:
                        parts = []
                        if result["created"]:
                            parts.append(f"新建 {len(result['created'])} 集")
                        if result["reused"]:
                            parts.append(f"沿用 {len(result['reused'])} 集")
                        if result["backed_up"]:
                            parts.append(f"備份舊資料夾 {len(result['backed_up'])} 個")
                        st.session_state["episode_generation_notice"] = {
                            "level": "success",
                            "message": "；".join(parts) or "多集設定已完成。",
                        }
                    st.rerun()

    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先執行 setup_cap_2000_project.py 建立結構。")
    else:
        _, _, _, manual_labels, _, _ = get_pipeline_mapping(st.session_state["profile_id"])
        manual_step_ids = list(manual_labels.keys())
        manual_step_text = " / ".join(manual_step_ids)

        # 主表：顯示目前 profile 的全部子步驟
        st.caption(f"規則載入：{len(substeps_map)} 項")
        st.dataframe(substeps_table(eps), use_container_width=True)
        # 規則 (debug)：顯示從 YAML 載入的子步驟鍵值
        with st.expander("子步驟規則 (debug)", expanded=False):
            st.write([{ "no": getattr(s, "no", None), "key": getattr(s, "key", None), "name": getattr(s, "name", None)} for s in substeps_map])

        # 附表：原 1–7 階段狀態
        with st.expander("階段總覽 (1–7)", expanded=False):
            st.dataframe(stage_table(eps), use_container_width=True)

        # 手動確認控制
        with st.expander(f"手動確認工具 (僅 {manual_step_text})", expanded=False):
            eps_map = {f"Ep{e['_raw']['ep']:02d}": e for e in eps}
            ep_choice = st.selectbox("選擇集數以標記手動步驟", list(eps_map.keys()), key="manual_ep")
            target_m = eps_map[ep_choice]
            ep_info = target_m["_raw"]
            marker_cols = st.columns([1] * len(manual_step_ids) + [2])
            updated_vals = {}
            for idx, step_no in enumerate(manual_step_ids):
                with marker_cols[idx]:
                    cur = has_manual_marker(ep_info["path"], step_no)
                    updated_vals[step_no] = st.checkbox(manual_labels[step_no], value=cur, key=f"manual{step_no}")
            with marker_cols[-1]:
                if st.button("更新手動標記", key="manual_update"):
                    for step_no, val in updated_vals.items():
                        set_manual_marker(ep_info["path"], step_no, val)
                    st.success("已更新手動標記，請重新展開刷新表格或切換頁籤。")

        # 檢核詳情 (可選集數)
        with st.expander(f"檢核詳情 (No.{SUBSTEP_MIN}–{SUBSTEP_MAX})", expanded=False):
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

        stage2subs, sub2stage, exec_mode, manual_labels, publish_substep, storyboard_manual_substep = get_pipeline_mapping(st.session_state["profile_id"])
        stage_names = {str(s.get('id')): s.get('name','') for s in (stage_cfg.get('stages') or [])}
        auto_substep_ids = [s for s in SUBSTEP_IDS if s != publish_substep]
        max_auto_substep = auto_substep_ids[-1] if auto_substep_ids else SUBSTEP_MAX

        if st.session_state["profile_id"] == "vocab":
            tab_stage, tab_sub, tab_prompts_pm, tab_storyboard_pm = st.tabs(
                ["階段控制 (1–7)", f"子步驟 ({SUBSTEP_MIN}–{SUBSTEP_MAX})", "Prompts", "修改分鏡/替換單字卡"]
            )
        else:
            tab_stage, tab_sub = st.tabs(["階段控制 (1–7)", f"子步驟 ({SUBSTEP_MIN}–{SUBSTEP_MAX})"])
            tab_prompts_pm = None
            tab_storyboard_pm = None
        if st.session_state.pop("pipeline_open_tab", None) == "storyboard":
            st.info("請使用「修改分鏡/替換單字卡」分頁進行調整。")

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
            st.caption(f"在此直接執行 {SUBSTEP_MIN}–{max_auto_substep}（第{publish_substep}發布請到 Studio & Publisher）。")
            subtitle_manual_substep = "13" if st.session_state["profile_id"] == "vocab" else "11"
            auto_refresh_sub = st.toggle(
                "自動刷新執行中的子步驟 Log",
                value=True,
                key="pm_sub_auto_refresh",
                help="有子步驟執行中時，每 2 秒自動更新一次狀態與 Log。",
            )
            any_running_sub = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in SUBSTEP_IDS)
            refresh_interval = 2 if (auto_refresh_sub and any_running_sub) else None

            @st.fragment(run_every=refresh_interval)
            def render_pipeline_substeps():
                st.caption("只會刷新此面板，不會切回 Dashboard。")
                st.button("只刷新子步驟狀態", key="refresh_substeps_fragment")
                dbg = evaluate_substeps_debug(info["path"], substeps_map)
                by_no = {str(r.get('no')): r for r in dbg}
                for ns in SUBSTEP_IDS:
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
                        if ns == publish_substep:
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
                        elif exec_mode.get(ns) == 'manual' and ns in manual_labels:
                            done = has_manual_marker(info['path'], ns)
                            marker_path = manual_marker_path(info['path'], ns)
                            marker_sig = f"{int(marker_path.stat().st_mtime)}_{int(done)}" if marker_path.exists() else "0_0"
                            st.caption("目前狀態：" + ("已標記" if done else "未標記"))
                            if ns == storyboard_manual_substep:
                                if st.button("前往修改分鏡/替換單字卡", key="goto_storyboard_13"):
                                    st.session_state['nav_radio'] = "⚙️ Pipeline Manager"
                                    st.query_params["section"] = "⚙️ Pipeline Manager"
                                    st.session_state["pipeline_open_tab"] = "storyboard"
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
                    if ns == "5":
                        render_cloze_output_expander(
                            info["path"],
                            key_prefix=f"pm_step5_{info['ep']}_{st.session_state['profile_id']}",
                            expanded=False,
                        )
                    elif ns == "8":
                        render_text_output_expander(
                            "Step 8 輸出預覽",
                            notebooklm_prompt_output_path(info["path"], info["ep"]),
                            text_label="NotebookLM Prompt 輸出",
                            key_prefix=f"pm_step8_{info['ep']}_{st.session_state['profile_id']}",
                            expanded=False,
                            height=320,
                        )
                    elif ns == subtitle_manual_substep:
                        render_manual_subtitle_editor(
                            info["path"],
                            ns,
                            key_prefix=f"pm_step{ns}_{info['ep']}_{st.session_state['profile_id']}",
                            expanded=False,
                        )
                    st.divider()

            render_pipeline_substeps()
            st.caption(
                "說明：\n"
                f"- 手動步驟（{', '.join(manual_labels.keys())}）可在此切換完成標記。\n"
                "- 其它子步驟按鈕會直接執行對應的 Python 腳本。\n"
                f"- 第{publish_substep}步 (發布) 請到 Studio & Publisher。"
            )

        if tab_prompts_pm is not None:
            with tab_prompts_pm:
                render_prompts_workspace(
                    info,
                    info["path"],
                    st.session_state["profile_id"],
                    subtitles_fixed_path(info["path"]),
                    first_existing_path(youtube_meta_paths(info["path"])),
                )
        if tab_storyboard_pm is not None:
            with tab_storyboard_pm:
                render_storyboard_workspace(info, info["path"], st.session_state["profile_id"])

        with st.expander(f"1–7 與 {SUBSTEP_MIN}–{SUBSTEP_MAX} 對應關係", expanded=False):
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

        profile_id = st.session_state["profile_id"]
        tab_preview, tab_prerender, tab_meta, tab_publish, tab_logs = st.tabs(
            ["預覽", "預渲染預覽", "Metadata", "發布", "Log"]
        )

        with tab_preview:
            pv1, pv2 = st.columns([1, 1.6])
            with pv1:
                st.subheader("封面")
                if cover_path.exists():
                    render_safe_image_preview(
                        cover_path,
                        empty_message="尚未找到封面圖。",
                        broken_hint="封面圖無法讀取",
                        caption_label="封面圖",
                        max_width=900,
                    )
                else:
                    st.caption("尚未找到封面圖。")
            with pv2:
                st.subheader("影片")
                if video_path.exists():
                    st.caption(f"{video_path.name} | {video_path.stat().st_size / (1024 * 1024):.1f} MB")
                    with st.expander("展開影片預覽", expanded=False):
                        st.video(str(video_path))
                else:
                    st.caption("尚未找到 final_video.mp4。")
            st.caption("此頁為輕量預覽：封面使用壓縮預覽，影片改為手動展開載入。")
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
                storyboard_end_sec = max(
                    [safe_float(row.get("end_time", 0)) for _, row in preview_storyboard_df.iterrows()] + [0.0]
                )
                subtitle_end_sec = max(
                    [safe_float(row.get("end", 0)) for row in subtitle_entries] + [0.0]
                )
                audio_duration_sec = media_duration_seconds(str(audio_path))
                preview_duration_sec = audio_duration_sec if audio_duration_sec is not None else max(
                    storyboard_end_sec,
                    subtitle_end_sec,
                    0.0,
                )
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric("Scene 數", len(preview_storyboard_df))
                pr2.metric("字幕數", len(subtitle_entries))
                pr3.metric(
                    "預覽時長",
                    seconds_to_label(preview_duration_sec),
                )
                if audio_duration_sec is not None:
                    overruns = []
                    if storyboard_end_sec > audio_duration_sec + 0.5:
                        overruns.append(f"分鏡表尾端 {seconds_to_label(storyboard_end_sec)}")
                    if subtitle_end_sec > audio_duration_sec + 0.5:
                        overruns.append(f"字幕尾端 {seconds_to_label(subtitle_end_sec)}")
                    if overruns:
                        st.warning(
                            "音訊實際長度為 "
                            f"{seconds_to_label(audio_duration_sec)}；"
                            + "、".join(overruns)
                            + " 已超出音訊範圍。預渲染預覽已改以音訊長度為準。"
                        )

                preview_payload = build_episode_preview_payload(ep_path, preview_storyboard_df, subtitle_entries)
                render_episode_preview_player(audio_path, preview_payload)
                st.caption("預渲染預覽為輕量模式：使用壓縮後分鏡縮圖同步音訊，不內嵌每個 Scene 的動畫影片，避免頁面訊息量過大。")

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

        if False:  # storyboard workspace moved to Pipeline Manager
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
                scene_image_task_name = f"sceneimg_{selected_scene_id}"
                scene_anim_prompt_task_name = f"sceneanimprompt_{selected_scene_id}"
                scene_anim_video_task_name = f"sceneanimvideo_{selected_scene_id}"
                selected_scene_task = publisher_task_state(info, scene_image_task_name)
                selected_anim_prompt_task = publisher_task_state(info, scene_anim_prompt_task_name)
                selected_anim_video_task = publisher_task_state(info, scene_anim_video_task_name)
                current_image_path = str(storyboard_scene_image_path(ep_path, selected_row) or "").strip()
                current_animation_path = str(storyboard_scene_animation_path(ep_path, selected_row) or "").strip()
                image_candidates = scene_image_candidate_paths(ep_path, selected_row)
                animation_candidates = scene_animation_candidate_paths(ep_path, selected_row)
                storyboard_auto_refresh = st.toggle(
                    "自動刷新 Scene 產圖 / 動畫 Log",
                    value=True,
                    key=f"storyboard_auto_refresh_{info['ep']}_{st.session_state['profile_id']}",
                    help="Scene 單獨產圖、動畫 Prompt、動畫生成執行時，每 2 秒自動更新狀態與 Log。",
                )

                source_key = f"storyboard_source_{info['ep']}_{selected_scene_id}"
                flashcard_key = f"storyboard_flashcard_{info['ep']}_{selected_scene_id}"
                custom_image_key = f"storyboard_custom_image_{info['ep']}_{selected_scene_id}"
                prompt_key = f"storyboard_prompt_{info['ep']}_{selected_scene_id}"
                animation_prompt_key = f"storyboard_animation_prompt_{info['ep']}_{selected_scene_id}"
                animation_prompt_loaded_key = f"{animation_prompt_key}__loaded"
                asset_mode_key = f"storyboard_asset_mode_{info['ep']}_{selected_scene_id}"
                image_choice_key = f"storyboard_image_choice_{info['ep']}_{selected_scene_id}"
                animation_choice_key = f"storyboard_animation_choice_{info['ep']}_{selected_scene_id}"
                reason_key = f"storyboard_reason_{info['ep']}_{selected_scene_id}"
                if source_key not in st.session_state:
                    st.session_state[source_key] = str(selected_row.get("source_type", "")).upper() or "AI"
                if flashcard_key not in st.session_state:
                    st.session_state[flashcard_key] = current_word if current_word in flashcards else ""
                if custom_image_key not in st.session_state:
                    st.session_state[custom_image_key] = str(selected_row.get("custom_image_path", "")).strip()
                if prompt_key not in st.session_state:
                    st.session_state[prompt_key] = str(selected_row.get("image_prompt", ""))
                if animation_prompt_key not in st.session_state:
                    st.session_state[animation_prompt_key] = str(selected_row.get("animation_prompt", ""))
                if animation_prompt_loaded_key not in st.session_state:
                    st.session_state[animation_prompt_loaded_key] = str(selected_row.get("animation_prompt", ""))
                latest_animation_prompt_from_row = str(selected_row.get("animation_prompt", ""))
                if (
                    st.session_state.get(animation_prompt_loaded_key) != latest_animation_prompt_from_row
                    and not bool(selected_anim_prompt_task.get("running"))
                ):
                    st.session_state[animation_prompt_key] = latest_animation_prompt_from_row
                    st.session_state[animation_prompt_loaded_key] = latest_animation_prompt_from_row
                if reason_key not in st.session_state:
                    st.session_state[reason_key] = str(selected_row.get("reason", ""))

                image_choice_map = {
                    asset_choice_label(p, ai_scene_image_path(ep_path, selected_row.get("start_time", "")), "圖片"): str(p)
                    for p in image_candidates
                }
                image_choice_labels = ["(使用自訂圖片路徑 / 預設原始圖)"] + list(image_choice_map.keys())
                current_image_choice_label = next(
                    (label for label, path in image_choice_map.items() if path == current_image_path),
                    image_choice_labels[0],
                )
                if st.session_state.get(image_choice_key) not in image_choice_labels:
                    st.session_state[image_choice_key] = current_image_choice_label

                animation_choice_map = {
                    asset_choice_label(p, ai_scene_animation_path(ep_path, selected_row), "動畫"): str(p)
                    for p in animation_candidates
                }
                animation_choice_labels = ["(不使用動畫)"] + list(animation_choice_map.keys())
                current_animation_choice_label = next(
                    (label for label, path in animation_choice_map.items() if path == current_animation_path),
                    animation_choice_labels[0],
                )
                if st.session_state.get(animation_choice_key) not in animation_choice_labels:
                    st.session_state[animation_choice_key] = current_animation_choice_label
                available_asset_modes = ["圖片"] + (["動畫"] if animation_candidates else [])
                current_asset_mode = "動畫" if current_animation_path and animation_candidates else "圖片"
                if st.session_state.get(asset_mode_key) not in available_asset_modes:
                    st.session_state[asset_mode_key] = current_asset_mode

                meta1, meta2, meta3, meta4 = st.columns(4)
                meta1.metric("Scene", str(selected_scene_id))
                meta2.metric("開始", seconds_to_label(selected_row.get("start_time", 0)))
                meta3.metric("結束", seconds_to_label(selected_row.get("end_time", 0)))
                meta4.metric("目前素材", "動畫" if current_animation_path else "圖片")

                preview_tab_current, preview_tab_image, preview_tab_animation = st.tabs(["目前素材", "候選圖片", "候選動畫"])
                with preview_tab_current:
                    if current_animation_path and Path(current_animation_path).exists():
                        st.caption(f"目前使用動畫：{current_animation_path}")
                        st.video(current_animation_path)
                    elif current_image_path and Path(current_image_path).exists():
                        st.caption(f"目前使用圖片：{current_image_path}")
                        st.image(current_image_path, use_container_width=True)
                    else:
                        st.caption("目前沒有已選定的素材。")
                with preview_tab_image:
                    if image_candidates:
                        image_rows = [image_candidates[i:i + 3] for i in range(0, len(image_candidates), 3)]
                        for row_group in image_rows:
                            cols = st.columns(3)
                            for col_idx, candidate_path in enumerate(row_group):
                                with cols[col_idx]:
                                    with st.container(border=True):
                                        st.caption(candidate_path.name)
                                        st.image(str(candidate_path), use_container_width=True)
                    else:
                        st.caption("尚無候選圖片。")
                with preview_tab_animation:
                    if animation_candidates:
                        animation_rows = [animation_candidates[i:i + 2] for i in range(0, len(animation_candidates), 2)]
                        for row_group in animation_rows:
                            cols = st.columns(2)
                            for col_idx, candidate_path in enumerate(row_group):
                                with cols[col_idx]:
                                    with st.container(border=True):
                                        st.caption(candidate_path.name)
                                        st.video(str(candidate_path))
                    else:
                        st.caption("尚無候選動畫。")

                st.subheader("Scene 設定")
                basic1, basic2, basic3 = st.columns([1, 1.2, 1.4])
                with basic1:
                    source_type_val = st.selectbox(
                        "source_type",
                        ["FLASHCARD", "AI"],
                        key=source_key,
                    )
                    asset_mode_val = st.radio(
                        "此分鏡使用素材",
                        available_asset_modes,
                        key=asset_mode_key,
                        horizontal=True,
                    )
                with basic2:
                    flashcard_options = [""] + sorted(flashcards.keys())
                    if current_word and current_word not in flashcard_options:
                        flashcard_options.append(current_word)
                    if st.session_state[flashcard_key] not in flashcard_options:
                        st.session_state[flashcard_key] = ""
                    flashcard_word_val = st.selectbox(
                        "替換單字卡",
                        flashcard_options,
                        key=flashcard_key,
                        disabled=(source_type_val != "FLASHCARD"),
                    )
                    selected_image_choice_label = st.selectbox(
                        "選擇使用圖片版本",
                        image_choice_labels,
                        key=image_choice_key,
                        disabled=(source_type_val == "FLASHCARD") or (asset_mode_val != "圖片"),
                    )
                with basic3:
                    custom_image_val = st.text_input(
                        "自訂圖片路徑",
                        key=custom_image_key,
                        disabled=(source_type_val == "FLASHCARD"),
                    )
                    selected_animation_choice_label = st.selectbox(
                        "選擇使用動畫版本",
                        animation_choice_labels,
                        key=animation_choice_key,
                        disabled=(asset_mode_val != "動畫"),
                    )

                image_prompt_val = st.text_area(
                    "image_prompt",
                    height=140,
                    key=prompt_key,
                    disabled=(source_type_val == "FLASHCARD"),
                )
                animation_prompt_val = st.text_area(
                    "animation_prompt",
                    height=140,
                    key=animation_prompt_key,
                )
                reason_val = st.text_area("reason", height=100, key=reason_key)

                selected_image_choice_path = image_choice_map.get(selected_image_choice_label, "")
                selected_animation_choice_path = animation_choice_map.get(selected_animation_choice_label, "")

                action1, action2, action3, action4 = st.columns([1, 1.1, 1.2, 1.1])
                with action1:
                    apply_scene = st.button("儲存此 Scene", key=f"storyboard_save_scene_{info['ep']}_{selected_scene_id}")
                with action2:
                    regen_scene = st.button(
                        "重產此 AI 分鏡圖",
                        key=f"storyboard_regen_scene_{info['ep']}_{selected_scene_id}",
                        disabled=(source_type_val != "AI") or bool(selected_scene_task.get("running")),
                    )
                with action3:
                    suggest_animation_prompt = st.button(
                        "AI 建議動畫 Prompt",
                        key=f"storyboard_suggest_anim_prompt_{info['ep']}_{selected_scene_id}",
                        disabled=bool(selected_anim_prompt_task.get("running")),
                    )
                with action4:
                    generate_animation = st.button(
                        "產生動畫",
                        key=f"storyboard_generate_animation_{info['ep']}_{selected_scene_id}",
                        disabled=bool(selected_scene_task.get("running")) or bool(selected_anim_video_task.get("running")),
                    )

                    if apply_scene:
                        chosen_ai_image_path = selected_image_choice_path or custom_image_val.strip()
                        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
                        scene_updates = {
                            "source_type": source_type_val,
                            "reason": reason_val,
                            "animation_prompt": animation_prompt_val,
                            "animation_video_path": chosen_animation_path,
                        }
                        if source_type_val == "FLASHCARD":
                            scene_updates["flashcard_word"] = flashcard_word_val
                            scene_updates["custom_image_path"] = resolve_flashcard_custom_image(
                                flashcards,
                                flashcard_word_val,
                                str(selected_row.get("custom_image_path", "")),
                            )
                            scene_updates["image_prompt"] = ""
                        else:
                            scene_updates["flashcard_word"] = ""
                            scene_updates["custom_image_path"] = chosen_ai_image_path
                            scene_updates["image_prompt"] = image_prompt_val
                        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
                        if saved_storyboard is None or idx is None:
                            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                        else:
                            st.success(f"已儲存 Scene {selected_scene_id}：{saved_storyboard}")
                            st.rerun()
                    if regen_scene:
                        chosen_ai_image_path = selected_image_choice_path or st.session_state[custom_image_key].strip()
                        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
                        saved_storyboard, latest_df, idx = update_storyboard_scene(
                            ep_path,
                            selected_scene_id,
                            {
                                "source_type": "AI",
                                "flashcard_word": "",
                                "custom_image_path": chosen_ai_image_path,
                                "image_prompt": st.session_state[prompt_key],
                                "animation_prompt": st.session_state[animation_prompt_key],
                                "animation_video_path": chosen_animation_path,
                                "reason": st.session_state[reason_key],
                            },
                        )
                        if saved_storyboard is None or idx is None:
                            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                        else:
                            res = start_publisher_task(
                                info,
                                scene_image_task_name,
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
                    if suggest_animation_prompt:
                        chosen_ai_image_path = selected_image_choice_path or custom_image_val.strip()
                        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
                        scene_updates = {
                            "source_type": source_type_val,
                            "reason": reason_val,
                            "animation_prompt": animation_prompt_val,
                            "animation_video_path": chosen_animation_path,
                        }
                        if source_type_val == "FLASHCARD":
                            scene_updates["flashcard_word"] = flashcard_word_val
                            scene_updates["custom_image_path"] = resolve_flashcard_custom_image(
                                flashcards,
                                flashcard_word_val,
                                str(selected_row.get("custom_image_path", "")),
                            )
                            scene_updates["image_prompt"] = ""
                        else:
                            scene_updates["flashcard_word"] = ""
                            scene_updates["custom_image_path"] = chosen_ai_image_path
                            scene_updates["image_prompt"] = image_prompt_val
                        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
                        if saved_storyboard is None or idx is None:
                            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                        else:
                            res = start_publisher_task(
                                info,
                                scene_anim_prompt_task_name,
                                {
                                    "name": f"AI 建議 Scene {selected_scene_id} 動畫 Prompt",
                                    "type": "python",
                                    "script": "scripts/generate_animation_prompt.py",
                                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id)],
                                },
                            )
                            if res.get("ok"):
                                st.success(f"Scene {selected_scene_id} 動畫 Prompt 任務已啟動。log: {res.get('log')}")
                            else:
                                st.error(f"Scene {selected_scene_id} 動畫 Prompt 任務啟動失敗：{res.get('message')}")
                            st.rerun()
                    if generate_animation:
                        chosen_ai_image_path = selected_image_choice_path or custom_image_val.strip()
                        chosen_animation_path = selected_animation_choice_path if asset_mode_val == "動畫" else ""
                        scene_updates = {
                            "source_type": source_type_val,
                            "reason": reason_val,
                            "animation_prompt": animation_prompt_val,
                            "animation_video_path": chosen_animation_path,
                        }
                        if source_type_val == "FLASHCARD":
                            scene_updates["flashcard_word"] = flashcard_word_val
                            scene_updates["custom_image_path"] = resolve_flashcard_custom_image(
                                flashcards,
                                flashcard_word_val,
                                str(selected_row.get("custom_image_path", "")),
                            )
                            scene_updates["image_prompt"] = ""
                        else:
                            scene_updates["flashcard_word"] = ""
                            scene_updates["custom_image_path"] = chosen_ai_image_path
                            scene_updates["image_prompt"] = image_prompt_val
                        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
                        if saved_storyboard is None or idx is None:
                            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                        elif not current_image_path and not str(storyboard_scene_image_path(ep_path, latest_df.iloc[idx]) or "").strip():
                            st.error("目前沒有可用圖片，請先確認圖卡或 AI 圖已就緒。")
                        elif not st.session_state[animation_prompt_key].strip():
                            st.error("animation_prompt 為空，請先產生或編輯動畫 Prompt。")
                        else:
                            res = start_publisher_task(
                                info,
                                scene_anim_video_task_name,
                                {
                                    "name": f"產生 Scene {selected_scene_id} 動畫",
                                    "type": "python",
                                    "script": "scripts/generate_scene_animation.py",
                                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id)],
                                },
                            )
                            if res.get("ok"):
                                st.success(f"Scene {selected_scene_id} 動畫任務已啟動。log: {res.get('log')}")
                            else:
                                st.error(f"Scene {selected_scene_id} 動畫任務啟動失敗：{res.get('message')}")
                            st.rerun()

                any_scene_worker_running = any(
                    bool(task.get("running"))
                    for task in [selected_scene_task, selected_anim_prompt_task, selected_anim_video_task]
                )
                scene_running_prev_key = f"storyboard_running_prev_{info['ep']}_{selected_scene_id}"
                st.session_state[scene_running_prev_key] = any_scene_worker_running
                scene_refresh_interval = 2 if (storyboard_auto_refresh and any_scene_worker_running) else None

                @st.fragment(run_every=scene_refresh_interval)
                def render_storyboard_scene_status():
                    latest_storyboard_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
                    latest_match = latest_storyboard_df[latest_storyboard_df["scene_id"].astype(str) == str(selected_scene_id)]
                    latest_row = latest_match.iloc[0].copy() if not latest_match.empty else selected_row.copy()
                    latest_scene_state = publisher_task_state(info, f"sceneimg_{selected_scene_id}")
                    latest_anim_prompt_state = publisher_task_state(info, f"sceneanimprompt_{selected_scene_id}")
                    latest_anim_video_state = publisher_task_state(info, f"sceneanimvideo_{selected_scene_id}")
                    latest_scene_log = Path(latest_scene_state.get("log", ""))
                    latest_anim_prompt_log = Path(latest_anim_prompt_state.get("log", ""))
                    latest_anim_video_log = Path(latest_anim_video_state.get("log", ""))
                    latest_image_path = str(storyboard_scene_image_path(ep_path, latest_row) or "").strip()
                    latest_animation_path = str(storyboard_scene_animation_path(ep_path, latest_row) or "").strip()
                    latest_image_candidates = scene_image_candidate_paths(ep_path, latest_row)
                    latest_animation_candidates = scene_animation_candidate_paths(ep_path, latest_row)
                    newest_animation_candidate = newest_existing_path(latest_animation_candidates)
                    latest_any_running = any([
                        latest_scene_state.get("running"),
                        latest_anim_prompt_state.get("running"),
                        latest_anim_video_state.get("running"),
                    ])
                    if st.session_state.get(scene_running_prev_key) and not latest_any_running:
                        st.session_state[scene_running_prev_key] = False
                        st.rerun()
                    st.session_state[scene_running_prev_key] = latest_any_running

                    st.subheader("Scene 執行狀態")
                    status_top1, status_top2, status_top3, status_top4 = st.columns([1, 1, 1, 1.2])
                    status_top1.metric("產圖任務", "執行中" if latest_scene_state.get("running") else ("已完成" if latest_scene_log.exists() else "尚未執行"))
                    status_top2.metric("動畫 Prompt", "執行中" if latest_anim_prompt_state.get("running") else ("已完成" if latest_anim_prompt_log.exists() else "尚未執行"))
                    status_top3.metric("動畫生成", "執行中" if latest_anim_video_state.get("running") else ("已生成候選" if newest_animation_candidate else "尚未執行"))
                    status_top4.metric("候選動畫數", len(latest_animation_candidates))
                    st.button("只刷新此 Scene 狀態", key=f"refresh_scene_status_{info['ep']}_{selected_scene_id}")
                    st.caption(f"最新圖片路徑：{latest_image_path or '無'}")
                    st.caption(f"最新動畫路徑：{latest_animation_path or '無'}")
                    if newest_animation_candidate:
                        st.caption(f"最新生成候選動畫：{newest_animation_candidate}")
                    if latest_scene_state.get("running"):
                        st.info(f"Scene {selected_scene_id} 產圖中，PID {latest_scene_state.get('pid')}")
                    if latest_anim_prompt_state.get("running"):
                        st.info(f"Scene {selected_scene_id} 動畫 Prompt 產生中，PID {latest_anim_prompt_state.get('pid')}")
                    if latest_anim_video_state.get("running"):
                        st.info(f"Scene {selected_scene_id} 動畫生成中，PID {latest_anim_video_state.get('pid')}")
                    if not latest_any_running:
                        st.caption("目前未在執行 Scene 相關任務。")

                    if latest_animation_path and Path(latest_animation_path).exists():
                        st.caption("目前使用中的動畫")
                        st.video(latest_animation_path)
                    if newest_animation_candidate and str(newest_animation_candidate) != latest_animation_path:
                        st.caption("最新生成的候選動畫")
                        st.video(str(newest_animation_candidate))
                        st.caption("若要讓這支動畫成為此分鏡正式素材，請在上方『選擇使用動畫版本』選取後，再按『儲存此 Scene』。")

                    if latest_scene_log.exists():
                        with st.expander("此 Scene 產圖 Log", expanded=bool(latest_scene_state.get("running"))):
                            st.code(latest_scene_log.read_text(encoding="utf-8", errors="ignore")[-4000:])
                            if latest_scene_state.get("running"):
                                st.caption("執行中。此區會自動刷新。")
                    if latest_anim_prompt_log.exists():
                        with st.expander("此 Scene 動畫 Prompt Log", expanded=bool(latest_anim_prompt_state.get("running"))):
                            st.code(latest_anim_prompt_log.read_text(encoding="utf-8", errors="ignore")[-4000:])
                            if latest_anim_prompt_state.get("running"):
                                st.caption("執行中。此區會自動刷新。")
                    if latest_anim_video_log.exists():
                        anim_log_text = latest_anim_video_log.read_text(encoding="utf-8", errors="ignore")
                        with st.expander("此 Scene 動畫生成 Log", expanded=True):
                            st.code(anim_log_text[-6000:])
                            if latest_anim_video_state.get("running"):
                                st.caption("執行中。此區會自動刷新。")
                            if "STEP 1/4" in anim_log_text:
                                st.caption("進度提示：請看 `STEP 1/4` 到 `STEP 4/4`，以及 `DONE animation generated`。")
                            if newest_animation_candidate:
                                st.caption(f"最新候選動畫檔：{newest_animation_candidate.name}")
                    st.text_area(
                        "字幕參考",
                        value=str(latest_row.get("subtitle_reference", "")),
                        height=220,
                        key=f"storyboard_subref_{info['ep']}_{selected_scene_id}",
                        disabled=True,
                    )

                    st.divider()
                    st.subheader("此 Scene 候選圖片")
                    if latest_image_candidates:
                        image_groups = [latest_image_candidates[i:i + 3] for i in range(0, len(latest_image_candidates), 3)]
                        for group in image_groups:
                            cols = st.columns(3)
                            for col_idx, candidate_path in enumerate(group):
                                with cols[col_idx]:
                                    with st.container(border=True):
                                        if str(candidate_path) == latest_image_path:
                                            st.caption("目前使用中")
                                        else:
                                            st.caption(candidate_path.name)
                                        st.image(str(candidate_path), use_container_width=True)
                    else:
                        st.caption("尚無候選圖片。")

                    st.subheader("此 Scene 候選動畫")
                    if latest_animation_candidates:
                        animation_groups = [latest_animation_candidates[i:i + 2] for i in range(0, len(latest_animation_candidates), 2)]
                        for group in animation_groups:
                            cols = st.columns(2)
                            for col_idx, candidate_path in enumerate(group):
                                with cols[col_idx]:
                                    with st.container(border=True):
                                        if str(candidate_path) == latest_animation_path:
                                            st.caption("目前使用中")
                                        else:
                                            st.caption(candidate_path.name)
                                        st.video(str(candidate_path))
                    else:
                        st.caption("尚無候選動畫。")

                    st.divider()
                    st.subheader("每個分鏡的圖片總覽")
                    gallery_rows = [latest_storyboard_df.iloc[i:i + 3] for i in range(0, len(latest_storyboard_df), 3)]
                    for row_group in gallery_rows:
                        gallery_cols = st.columns(3)
                        for col_idx, (_, row) in enumerate(row_group.iterrows()):
                            img_path = storyboard_scene_image_path(ep_path, row)
                            anim_path = storyboard_scene_animation_path(ep_path, row)
                            with gallery_cols[col_idx]:
                                with st.container(border=True):
                                    st.caption(
                                        f"Scene {row.get('scene_id')} | {str(row.get('source_type', '')).upper()} | "
                                        f"{row.get('start_time')} - {row.get('end_time')}"
                                    )
                                    if anim_path and anim_path.exists():
                                        st.caption("已生成動畫")
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
                        "animation_prompt": st.column_config.TextColumn(width="large"),
                        "animation_video_path": st.column_config.TextColumn(width="large"),
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


