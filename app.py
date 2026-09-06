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
from datetime import date, datetime
from typing import Dict
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image, ImageFile
from dotenv import load_dotenv

from app_utils.filesystem import (
    episode_dir_name,
    find_episode_dirs_by_ep,
    list_episode_dirs,
    parse_episode_info,
    read_status,
    write_status,
    video_output_path,
    subtitles_raw_path,
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
    _file_nonempty,
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
from app_utils.asset_tags import (
    ASSET_TAG_FILTER_ALL as SHARED_ASSET_TAG_FILTER_ALL,
    ASSET_TAG_FILTER_UNTAGGED as SHARED_ASSET_TAG_FILTER_UNTAGGED,
    asset_tag_badge as shared_asset_tag_badge,
    asset_tag_for_path as shared_asset_tag_for_path,
    asset_tags_json_path as shared_asset_tags_json_path,
    asset_tag_storage_key as shared_asset_tag_storage_key,
    collect_asset_tag_options as shared_collect_asset_tag_options,
    episode_path_for_asset as shared_episode_path_for_asset,
    filter_asset_paths_by_tag as shared_filter_asset_paths_by_tag,
    read_episode_asset_tags as shared_read_episode_asset_tags,
    set_asset_tag_for_path as shared_set_asset_tag_for_path,
    write_episode_asset_tags as shared_write_episode_asset_tags,
)
from app_utils.schedules import (
    cleanup_schedule_history,
    delete_schedule_job,
    get_schedule_job_state,
    list_schedule_jobs,
    schedule_job_log_path,
    start_schedule_job,
    start_schedule_job_now,
    stop_schedule_job,
    write_schedule_job,
)
from app_utils.power import wake_timer_policy
from app_sections.youtube_ai import (
    configure_youtube_ai,
    render_youtube_ai_dashboard,
    render_youtube_ai_pipeline_manager,
    render_youtube_ai_settings,
)
from app_sections.episode_merge import (
    configure_episode_merge,
    render_episode_merge_workspace,
)
from app_sections.short_generator import (
    configure_short_generator,
    render_short_factory,
    render_short_management,
)
from app_sections.story import (
    configure_story_section,
    STORY_OPENAI_TTS_VOICES,
    story_outline_options_path,
    story_selected_json_path,
    story_selected_md_path,
    story_outline_json_path,
    story_outline_md_path,
    story_script_json_path,
    story_script_md_path,
    story_tts_text_path,
    story_characters_json_path,
    story_locations_json_path,
    story_tts_lines_json_path,
    story_tts_lines_csv_path,
    story_voice_cast_json_path,
    story_voice_segments_dir,
    story_voice_previews_dir,
    story_merged_audio_path,
    story_subtitles_srt_path,
    story_subtitles_json_path,
    story_storyboard_csv_path,
    story_storyboard_json_path,
    default_story_voice_cast,
    load_story_outline_options,
    render_story_summary,
    render_story_outline_result,
    render_story_script_result,
    render_story_characters_result,
    render_story_locations_result,
    render_story_tts_lines_result,
    has_openai_api_key_configured,
    render_story_voice_workspace,
    render_story_subtitles_result,
    render_story_storyboard_result,
    render_story_selection_confirm,
    render_story_inline_confirm_if_needed,
    run_story_outline_selection,
    render_story_outline_selection_panel,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROFILES_CFG = ROOT / "config" / "profiles.yaml"
SECTION_OPTIONS = [
    "📊 Dashboard",
    "⚙️ Pipeline Manager",
    "⏰ 排程執行",
    "🧩 Prompt Sources",
    "💰 FinOps Monitor",
    "🎬 Studio & Publisher",
]
PROFILE_SECTION_OPTIONS = {
    "youtube_ai": [
        "📊 Dashboard",
        "⚙️ Settings",
        "🧩 Pipeline Manager",
    ],
    "episode_merge": [
        "集數合併",
    ],
    "short_generator": [
        "Short 管理",
        "短影音工廠",
    ],
    "single_video": [
        "📊 Dashboard",
        "⚙️ Pipeline Manager",
        "🎬 Studio & Publisher",
    ],
}

def file_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return 0, 0


@st.cache_data(show_spinner=False)
def _load_profiles_cached(config_path: str, signature: tuple[int, int]) -> list[dict]:
    _ = signature
    path = Path(config_path)
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("profiles", [])
    return []


def load_profiles():
    if PROFILES_CFG.exists():
        return _load_profiles_cached(str(PROFILES_CFG), file_signature(PROFILES_CFG))
    return _load_profiles_cached(str(PROFILES_CFG), (0, 0))

def get_profile_map():
    profs = load_profiles()
    return {p.get("id"): p for p in profs}

def section_options_for_profile(profile_id: str) -> list[str]:
    return PROFILE_SECTION_OPTIONS.get(profile_id, SECTION_OPTIONS)

st.set_page_config(page_title="CAP 2000 AI Video CMS", layout="wide")

st.sidebar.title("CAP 2000 控制台")

# Common data (profile-aware)
profiles = get_profile_map()
profile_ids = list(profiles.keys()) or ["vocab"]
if "profile_id" not in st.session_state:
    st.session_state["profile_id"] = profile_ids[0]
if st.session_state["profile_id"] not in profile_ids:
    st.session_state["profile_id"] = profile_ids[0]

def profile_id_for_section(section_label: str) -> str | None:
    for pid in profile_ids:
        if section_label in section_options_for_profile(pid):
            return pid
    return None

query_profile = st.query_params.get("profile")
query_section_initial = st.query_params.get("section")
profile_resolved_from_query = False
if query_profile in profile_ids and query_profile != st.session_state["profile_id"]:
    st.session_state["profile_id"] = query_profile
    profile_resolved_from_query = True
elif query_section_initial:
    query_section_profile = profile_id_for_section(query_section_initial)
    if (
        query_section_profile
        and query_section_initial not in section_options_for_profile(st.session_state["profile_id"])
    ):
        st.session_state["profile_id"] = query_section_profile
        profile_resolved_from_query = True

def profile_label(profile_id: str) -> str:
    return profiles.get(profile_id, {"name": profile_id}).get("name", profile_id) + f" ({profile_id})"

if st.session_state.get("profile_select") not in profile_ids or profile_resolved_from_query:
    st.session_state["profile_select"] = st.session_state["profile_id"]
selected_profile_id = st.sidebar.selectbox(
    "專案環境",
    profile_ids,
    index=profile_ids.index(st.session_state["profile_id"]),
    key="profile_select",
    format_func=profile_label,
)
if selected_profile_id != st.session_state["profile_id"]:
    st.session_state["profile_id"] = selected_profile_id
    next_sections = section_options_for_profile(selected_profile_id)
    st.session_state["nav_radio"] = next_sections[0]
    st.query_params["profile"] = selected_profile_id
    st.query_params["section"] = next_sections[0]
    st.rerun()

section_options = section_options_for_profile(st.session_state["profile_id"])
query_section = st.query_params.get("section")
if query_section in section_options and "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = query_section
if st.session_state.get("nav_radio") not in section_options:
    st.session_state["nav_radio"] = section_options[0]
section = st.sidebar.radio("功能模組", section_options, key="nav_radio")
if st.query_params.get("profile") != st.session_state["profile_id"]:
    st.query_params["profile"] = st.session_state["profile_id"]
if st.query_params.get("section") != section:
    st.query_params["section"] = section

profile = profiles.get(st.session_state["profile_id"], {})
WS_ROOT = profile.get("workspace", "workspace")
PATH_STAGES = profile.get("stages")
PATH_SUBSTEPS = profile.get("substeps")
PATH_COSTS = profile.get("costs")
stage_cfg = load_stage_config(ROOT, PATH_STAGES)
runner = StageRunner(ROOT, PATH_STAGES, (ROOT / WS_ROOT), profile_id=st.session_state["profile_id"])
configure_youtube_ai(ROOT, WS_ROOT, runner)
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
DEFAULT_EPISODE_SCAFFOLD_DIRS = [
    "00_logs",
    "01_audio",
    "02_subtitles",
    "03_storyboards",
    "04_images",
    "05_output",
]
STORY_EPISODE_SCAFFOLD_DIRS = [
    "00_logs",
    "01_preproduction",
    "02_story",
    "03_characters_scenes",
    "04_audio_subtitles",
    "04_audio_subtitles/voice_segments",
    "05_storyboards",
    "05_storyboards/images",
    "05_storyboards/animations",
    "06_video",
    "07_publish",
]
DEFAULT_CLOZE_PROMPT_TEMPLATE = """你是專業的英文考題編輯，請依據提供的 vocab_data，為本集每個單字各產生 1 題四選一克漏字題。

你必須嚴格遵守以下規則：
1. 請依 vocab_data 原本順序輸出，總題數必須剛好等於 {{VOCAB_COUNT}} 題。
2. 每題都必須對應到同一列 vocab 的 Word，不可串到別的單字。
3. `blank_sentence` 必須直接使用該列的 `English_Sentence`，只把目標字詞替換成 `____`；不可改寫成別句，不可借用其他單字的句子。
4. `surface_word` 必須是該列目標字在原句中實際出現的詞形；若原句有詞形變化，正確答案也必須用同一詞形。
5. 四個選項中只能有一個正確答案。其餘三個干擾選項必須在這個句子裡明顯不通順、不合語意，不能出現兩個以上都合理的答案。
6. 不可重用其他題目的句子、答案、解釋或選項。
7. explanation 請用繁體中文，簡短說明為什麼正確，並指出其他選項不適合這句。
8. 只輸出 JSON，不要輸出任何額外文字。

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
DEFAULT_COVER_IMAGE_PROMPT_TEMPLATE = """請為《會考英文隨身聽》設計一張全新的 YouTube 封面圖。

【封面目標】
- 用可愛、吸睛、適合國中英文學習頻道的風格，設計 16:9 橫式封面。
- John 與 Mary 必須同時出現在封面中，並保持角色一致性。
- 請根據本集單字挑選 4 到 6 個最適合視覺化的元素融入畫面，不要把 10 個單字全部塞成一堆文字。
- 畫面要簡潔、集中、容易一眼看懂，不要過度擁擠。
- 適度加入「第 {{EPISODE}} 集」的視覺提示，但避免太多小字。
- 不要做成真實照片，請維持可愛 3D 卡通、明亮、討喜、適合兒少教育內容的質感。

【本集單字資料】
{{VOCAB_LIST}}

【John & Mary 角色設定 JSON】
{{HOST_PROFILE_JSON}}

【額外要求】
- 如果有動物、腳踏車、生日、biology 等元素，優先以童趣、可愛、正向的方式呈現。
- 避免暴力、驚悚、陰暗、過度寫實、複雜背景、凌亂排版。
- 最終請直接輸出一段可拿來產圖的完整 prompt 內容，不要解說，不要分點，不要加 Markdown。
"""
DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE = """你是一位專業的影片字幕校對專家。請針對以下 SRT 內容進行修正，並務必遵守下列規則：

0. 參考資料：
- 「No.8 產生語音摘要 Prompt」是本集 NotebookLM 語音內容的規劃與重點摘要，可用來判斷主題、單字、例句、段落順序與中英文脈絡。
- 參考資料只用於校對文字，不可把未出現在字幕音訊中的內容新增進 SRT。
- 若原始字幕與參考資料衝突，優先保留原始字幕的實際口語內容與時間軸。

1. 單字修復與局部合併：
- 不要合併、刪除、拆分或重排任何字幕 cue。
- No.12 的 LLM 只負責校正每個 cue 內的文字；英文斷句合併會由系統後處理完成。
- 嚴禁因語意相近就合併不同序號。
- 嚴禁跨句、跨段任意合併。

2. 時間軸保護：
- 必須保留每一筆原始字幕的序號、開始時間與結束時間。
- 輸出 cue 數量必須與原始 SRT 完全相同。
- 不可任意改動分鐘、秒數或順序。
- 不可讓字幕突然跳到很後面的時間。
- 不可遺漏大段中間字幕內容。

3. 全面校對：
- 修正英文拼字錯誤。
- 根據上下文修正常見音近誤辨。
- 關鍵單字可視情況加上引號，但不要破壞 SRT 結構。
- 英文句尾不可使用中文句號「。」；請改成英文句號「.」。

4. 中文斷句與空格：
- 修正中文被不自然拆字、斷詞或插入空格的情況，例如「這 是 一 個」應校正為「這是一個」。
- 中文標點前不可有空格，中文標點後通常不需要空格。
- 可依中文語意補上逗號、句號、問號或驚嘆號，讓句子更自然，但不可改寫成不同意思。
- 不要把英文單字與中文強行黏在一起；中英混排可保留必要空格。

5. 時間軸與格式：
- 保持標準 SRT 格式。
- 每一個字幕 cue 必須固定為：序號一行、時間軸一行、字幕文字一到多行、空白行。時間軸那一行只能有時間，不可把字幕文字接在時間軸後面。
- 盡量讓相鄰字幕保留極短空隙。
- 只輸出純 SRT，不要輸出 Markdown、解說或任何額外文字。

【No.8 產生語音摘要 Prompt 參考】
{{NOTEBOOKLM_PROMPT}}

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
- 你可以閱讀並使用中文字幕脈絡來理解畫面；但輸出的 image_prompt 必須是完整英文，不可直接保留中文、日文、韓文或非英文字幕原文。請把中文脈絡轉寫成英文畫面描述，並固定以 ", aspect ratio 16:9, cinematic wide shot" 作結。

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
DEFAULT_SINGLE_VIDEO_STORYBOARD_PROMPT_TEMPLATE = """你是影片分鏡導演。請根據「固定 Scene 區塊」替 YouTube 影片規劃分鏡內容。

請輸出 JSON 陣列，每個物件必須包含：
scene_id, start_time, end_time, source_type, flashcard_word, custom_image_path, image_prompt, animation_prompt, animation_video_path, reason, subtitle_reference

規則：
1. 必須逐一處理「固定 Scene 區塊」，輸出數量、scene_id 與區塊完全一致，不可新增、刪除、合併或重排。
2. source_type 一律填 AI。
3. flashcard_word 一律留空。
4. start_time、end_time、subtitle_reference 必須原樣複製對應的固定 Scene 區塊。
5. image_prompt 必須只根據同一個 Scene 區塊的 subtitle_reference 產生，不可使用其他 Scene 的內容。
6. image_prompt 請用英文，描述明確主體、場景、動作、情緒、風格與 16:9 構圖，不要產生任何可讀文字。
7. animation_prompt 請用英文，描述輕微鏡頭或角色動作。
8. reason 請用中文簡短說明此畫面如何對應該 Scene 的字幕語境。
9. custom_image_path 與 animation_video_path 留空。
10. 只輸出 JSON，不要加 Markdown 或任何額外說明。

影片標題：{{YOUTUBE_TITLE}}

固定 Scene 區塊：
{{SCENE_BLOCKS}}
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

def get_episode_scaffold_dirs(profile_id: str) -> list[str]:
    if profile_id == "story":
        return STORY_EPISODE_SCAFFOLD_DIRS
    return DEFAULT_EPISODE_SCAFFOLD_DIRS

def single_video_info_path(ep_path: Path) -> Path:
    return ep_path / "video_info.json"

def load_single_video_info(ep_path: Path) -> Dict:
    payload = read_json_file(single_video_info_path(ep_path)) or {}
    return payload if isinstance(payload, dict) else {}

def save_single_video_info(ep_path: Path, youtube_title: str) -> Path:
    payload = load_single_video_info(ep_path)
    payload["youtube_title"] = str(youtube_title or "").strip()
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    target = single_video_info_path(ep_path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

def single_video_title(ep_path: Path) -> str:
    info = load_single_video_info(ep_path)
    title = str(info.get("youtube_title") or "").strip()
    if title:
        return title
    meta = read_json_file(first_existing_path(youtube_meta_paths(ep_path))) or {}
    if isinstance(meta, dict):
        title = str(meta.get("title") or "").strip()
    return title

def single_video_option_label(item: Dict) -> str:
    raw = item.get("_raw", {})
    ep_path = raw.get("path")
    title = single_video_title(ep_path) if isinstance(ep_path, Path) else ""
    base = f"Ep{int(raw.get('ep') or 0):02d}"
    return f"{base} | {title}" if title else f"{base} | 未命名影片"

def build_single_video_generation_plan(start_ep: int, end_ep: int, title: str = "") -> list[Dict]:
    if start_ep > end_ep:
        raise ValueError("起始集數不可大於結束集數。")
    plans = []
    for ep_num in range(start_ep, end_ep + 1):
        dir_name = episode_dir_name(ep_num, 0, 0)
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
            "start": 0,
            "end": 0,
            "dir_name": dir_name,
            "target_path": ROOT / WS_ROOT / dir_name,
            "existing_dirs": existing_dirs,
            "exact_match": exact_match,
            "conflict_dirs": conflict_dirs,
            "action": action,
            "youtube_title": title if start_ep == end_ep else "",
        })
    return plans

def single_video_generation_plan_table(plans: list[Dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "影片": f"Ep{plan['ep']:02d}",
            "YouTube Title": plan.get("youtube_title") or "—",
            "目標資料夾": plan["dir_name"],
            "現有資料夾": "\n".join(p.name for p in plan["existing_dirs"]) or "—",
            "動作": plan["action"],
        }
        for plan in plans
    ])

def single_video_prompt_scene_count(timeline_end: float, subtitle_count: int) -> int:
    configured = str(os.getenv("CAP_SINGLE_VIDEO_SCENE_COUNT", "")).strip()
    if configured:
        try:
            return max(1, min(int(configured), max(subtitle_count, 1)))
        except ValueError:
            pass
    target_seconds = max(safe_float(os.getenv("CAP_SINGLE_VIDEO_TARGET_SCENE_SECONDS"), 18.0), 5.0)
    estimated = max(1, round(max(timeline_end, 0.2) / target_seconds))
    return max(1, min(estimated, max(subtitle_count, 1), 80))

def build_single_video_prompt_scene_blocks(srt_path: Path) -> list[Dict]:
    entries = parse_srt_entries(srt_path)
    if not entries:
        return []
    timeline_end = max(safe_float(row.get("end", 0), 0) for row in entries)
    scene_count = single_video_prompt_scene_count(timeline_end, len(entries))
    blocks = []
    total = len(entries)
    for idx in range(scene_count):
        start_idx = int(idx * total / scene_count)
        end_idx = int((idx + 1) * total / scene_count) - 1
        end_idx = max(start_idx, min(end_idx, total - 1))
        chunk = entries[start_idx: end_idx + 1]
        blocks.append({
            "scene_id": idx + 1,
            "start_time": round(safe_float(chunk[0].get("start", 0), 0), 3),
            "end_time": round(safe_float(chunk[-1].get("end", 0), 0), 3),
            "subtitle_reference": " | ".join(
                re.sub(r"\s+", " ", str(row.get("content") or "")).strip()
                for row in chunk
                if str(row.get("content") or "").strip()
            ),
        })
    return blocks

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

def apply_episode_generation(plans: list[Dict], allow_backup_conflicts: bool, scaffold_dirs: list[str] | None = None) -> Dict:
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
        for folder_name in (scaffold_dirs or DEFAULT_EPISODE_SCAFFOLD_DIRS):
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

def build_story_episode_generation_plan(start_ep: int, end_ep: int) -> list[Dict]:
    if start_ep > end_ep:
        raise ValueError("起始集數不可大於結束集數。")
    plans = []
    for ep_num in range(start_ep, end_ep + 1):
        dir_name = episode_dir_name(ep_num, 0, 0)
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
            "start": 0,
            "end": 0,
            "dir_name": dir_name,
            "target_path": ROOT / WS_ROOT / dir_name,
            "existing_dirs": existing_dirs,
            "exact_match": exact_match,
            "conflict_dirs": conflict_dirs,
            "action": action,
        })
    return plans

def story_generation_plan_table(plans: list[Dict]) -> pd.DataFrame:
    rows = []
    for plan in plans:
        rows.append({
            "Ep": f"Ep{plan['ep']:02d}",
            "目標資料夾": plan["dir_name"],
            "現有資料夾": "\n".join(p.name for p in plan["existing_dirs"]) or "—",
            "動作": plan["action"],
        })
    return pd.DataFrame(rows)

def get_pipeline_mapping(profile_id: str):
    if profile_id == "vocab":
        mapping = {
            "1": ["3", "4", "5", "6", "7", "8", "9", "10"],
            "2": ["10.5", "11", "12", "13"],
            "3": ["14", "15"],
            "4": ["16"],
            "5": ["17", "18", "19", "20"],
            "6": ["21"],
        }
        manual_labels = {
            "9": "Step 9 完成 (NotebookLM)",
            "13": "Step 13 完成 (人工字幕)",
            "15": "Step 15 完成 (人工微調)",
        }
        publish_substep = "21"
        storyboard_manual_substep = "15"
    elif profile_id == "story":
        mapping = {
            "1": ["1.1", "1.2"],
            "2": ["2.1", "2.2"],
            "3": ["3.1", "3.2"],
            "4": ["4.1", "4.2", "4.3", "4.4"],
            "5": ["5.1", "5.2", "5.3"],
            "6": ["6.1"],
            "7": ["7.1", "7.2"],
        }
        manual_labels = {}
        publish_substep = "7.2"
        storyboard_manual_substep = "5.1"
    elif profile_id == "single_video":
        mapping = {
            "1": ["1", "2", "3", "4"],
            "2": ["5"],
            "3": ["6"],
            "4": ["7", "8", "9", "10"],
            "5": ["11"],
        }
        manual_labels = {
            "4": "Step 4 完成 (人工字幕)",
        }
        publish_substep = "11"
        storyboard_manual_substep = ""
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

def get_stage_ids_for_profile(profile_id: str) -> list[str]:
    if profile_id == "vocab":
        return ["1", "2", "3", "4", "5", "6"]
    if profile_id == "story":
        return ["1", "2", "3", "4", "5", "6", "7"]
    if profile_id == "single_video":
        return ["1", "2", "3", "4", "5"]
    if profile_id == "episode_merge":
        return []
    return ["1", "1.5", "2", "3", "4", "5", "6", "7"]

def get_schedule_stage_options(profile_id: str) -> tuple[list[str], dict[str, str]]:
    stage_ids = get_stage_ids_for_profile(profile_id)
    stage_names = {str(s.get("id")): s.get("name", "") for s in (stage_cfg.get("stages") or [])}
    options = list(stage_ids)
    if profile_id == "vocab":
        options.append("4.1")
        stage_names["4.1"] = "將所有 AI 圖片轉成動畫"
    return options, stage_names

def get_stage_range_label(stage_ids: list[str]) -> str:
    if not stage_ids:
        return ""
    return f"{stage_ids[0]}–{stage_ids[-1]}"

def scan_episodes():
    eps = []
    for p in list_episode_dirs(ROOT, WS_ROOT):
        info = parse_episode_info(p)
        raw_prev = read_status(p)
        computed = infer_stage_statuses_for_profile(p, st.session_state["profile_id"], prev=raw_prev)
        eps.append({
            "Ep": f"Ep{info['ep']:02d}",
            "Range": f"{info['start']:04d}-{info['end']:04d}",
            "Current": computed,
            "_raw": info,
        })
    return eps

def next_episode_number_default(min_ep: int = 1) -> int:
    eps = [
        int(parse_episode_info(p).get("ep") or 0)
        for p in list_episode_dirs(ROOT, WS_ROOT)
    ]
    current_max = max(eps or [min_ep - 1])
    return max(min_ep, current_max + 1)

def infer_stage_statuses_for_profile(ep_path: Path, profile_id: str, prev: Dict | None = None) -> Dict:
    if profile_id == "story":
        stage_ids = get_stage_ids_for_profile(profile_id)
        st = {sid: "pending" for sid in stage_ids}
        if (
            (
                _file_nonempty(ep_path / "01_preproduction" / "story_outline_options.json")
                or _file_nonempty(ep_path / "01_preproduction" / "story_brainstorm_options.json")
            )
            and _file_nonempty(story_selected_json_path(ep_path))
            and _file_nonempty(story_selected_md_path(ep_path))
        ):
            st["1"] = "done"
        if (
            _file_nonempty(ep_path / "02_story" / "story_outline.md")
            and _file_nonempty(ep_path / "02_story" / "story_script.md")
        ):
            st["2"] = "done"
        if (
            _file_nonempty(ep_path / "03_characters_scenes" / "characters.json")
            and _file_nonempty(ep_path / "03_characters_scenes" / "locations.json")
        ):
            st["3"] = "done"
        if (
            _file_nonempty(ep_path / "04_audio_subtitles" / "tts_lines.csv")
            and _file_nonempty(ep_path / "04_audio_subtitles" / "story_audio.m4a")
            and _file_nonempty(ep_path / "04_audio_subtitles" / "story_subtitles.srt")
        ):
            st["4"] = "done"
        if (
            _file_nonempty(ep_path / "05_storyboards" / "storyboard.csv")
            and any((ep_path / "05_storyboards" / "images").glob("*.*"))
            and any((ep_path / "05_storyboards" / "animations").glob("*.*"))
        ):
            st["5"] = "done"
        if _file_nonempty(ep_path / "06_video" / "final_video.mp4"):
            st["6"] = "done"
        if (
            _file_nonempty(ep_path / "07_publish" / "cover.png")
            and _file_nonempty(ep_path / "07_publish" / "youtube_meta.json")
        ):
            st["7"] = "done"
        if prev and isinstance(prev, dict):
            p_st = (prev.get("stages") or {}) if "stages" in prev else prev
            for sid, pv in p_st.items():
                if sid in st and st.get(sid) != "done" and pv in ("error", "running"):
                    st[sid] = pv
        return st

    if profile_id == "single_video":
        stage_ids = get_stage_ids_for_profile(profile_id)
        st = {sid: "pending" for sid in stage_ids}
        if subtitles_fixed_path(ep_path).exists():
            st["1"] = "done"
        if storyboard_csv_path(ep_path).exists():
            st["2"] = "done"
        preview_image_dirs = [
            ep_path / "04_images" / "preview_images",
            ep_path / "04_images" / "ai_generated",
        ]
        if any(p.exists() and any(p.iterdir()) for p in preview_image_dirs):
            st["3"] = "done"
        inputs_ready = _file_nonempty(inputs_txt_path(ep_path))
        final_video = video_output_path(ep_path).exists()
        meta_exists = any(p.exists() for p in youtube_meta_paths(ep_path))
        if inputs_ready and final_video and meta_exists:
            st["4"] = "done"
        upload_record = read_json_file(ep_path / "05_output" / "upload_record.json") or {}
        uploaded_markers = [ep_path / "05_output" / "uploaded.txt", ep_path / "05_output" / "upload.ok"]
        if (
            (isinstance(upload_record, dict) and str(upload_record.get("status", "")).strip().lower() == "success")
            or any(m.exists() for m in uploaded_markers)
        ):
            st["5"] = "done"
        if prev and isinstance(prev, dict):
            p_st = (prev.get("stages") or {}) if "stages" in prev else prev
            for sid, pv in p_st.items():
                if sid in st and st.get(sid) != "done" and pv in ("error", "running"):
                    st[sid] = pv
        return st

    if profile_id != "vocab":
        return infer_stage_statuses(ep_path, prev=prev)

    stage_ids = get_stage_ids_for_profile(profile_id)
    st = {sid: "pending" for sid in stage_ids}

    stage1_ready = (
        _file_nonempty(storyboard_csv_path(ep_path).parent / "vocab_data.csv")
        and notebooklm_prompt_output_path(ep_path, int(parse_episode_info(ep_path).get("ep") or 0)).exists()
    )
    if stage1_ready:
        st["1"] = "done"

    if subtitles_fixed_path(ep_path).exists():
        st["2"] = "done"

    if storyboard_csv_path(ep_path).exists():
        st["3"] = "done"

    preview_image_dirs = [
        ep_path / "04_images" / "preview_images",
        ep_path / "04_images" / "ai_generated",
    ]
    if any(p.exists() and any(p.iterdir()) for p in preview_image_dirs):
        st["4"] = "done"

    inputs_ready = _file_nonempty(inputs_txt_path(ep_path))
    final_video = video_output_path(ep_path).exists()
    meta_exists = any(p.exists() for p in youtube_meta_paths(ep_path))
    if inputs_ready and final_video and meta_exists:
        st["5"] = "done"

    upload_record = read_json_file(ep_path / "05_output" / "upload_record.json") or {}
    uploaded_markers = [ep_path / "05_output" / "uploaded.txt", ep_path / "05_output" / "upload.ok"]
    if (
        (isinstance(upload_record, dict) and str(upload_record.get("status", "")).strip().lower() == "success")
        or any(m.exists() for m in uploaded_markers)
    ):
        st["6"] = "done"

    if prev and isinstance(prev, dict):
        p_st = (prev.get("stages") or {}) if "stages" in prev else prev
        for sid, pv in p_st.items():
            if sid in st and st.get(sid) != "done" and pv in ("error", "running"):
                st[sid] = pv
    return st


def stage_table(eps):
    stage_ids = get_stage_ids_for_profile(st.session_state["profile_id"])
    rows = []
    for e in eps:
        stg = e["Current"]
        latest = None
        for sid in stage_ids:
            s = stg.get(sid, "pending")
            if s in ("running", "error", "done"):
                latest = f"{sid}:{s}"
        row = {
            "Ep": e["Ep"],
            "Range": e["Range"],
        }
        if st.session_state["profile_id"] == "single_video":
            row["YouTube Title"] = single_video_title(e["_raw"]["path"]) or "—"
        for sid in stage_ids:
            row[sid] = with_emoji(stg.get(sid, "pending"))
        row["Latest"] = latest or "—"
        rows.append(row)
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
        if st.session_state["profile_id"] == "single_video":
            row["YouTube Title"] = single_video_title(Path(item["_path"])) or "—"
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

VOCAB_TEXT_LLM_STEP_DEFAULTS = {
    "3": "auto",
    "5": "auto",
    "12": "auto",
    "14": "auto",
    "20": "auto",
}
VOCAB_TEXT_LLM_PROVIDER_LABELS = {
    "auto": "自動",
    "openai": "OpenAI / ChatGPT",
    "gemini": "Gemini",
    "nvidia": "NVIDIA",
}
VOCAB_TEXT_LLM_STEP_PROVIDER_LABELS = {
    "12": {
        "auto": "自動：NVIDIA → OpenAI → Gemini",
    },
    "14": {
        "auto": "自動：NVIDIA → OpenAI → Gemini",
    },
}
VOCAB_IMAGE_PROVIDER_LABELS = {
    "auto": "自動：Gemini → OpenAI → NVIDIA",
    "gemini": "Gemini / Imagen",
    "openai": "OpenAI / ChatGPT",
    "nvidia": "NVIDIA NIM",
}
VOCAB_IMAGE_MODEL_OPTIONS = {
    "gemini": ["imagen-4.0-generate-001"],
    "openai": ["gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"],
    "nvidia": ["black-forest-labs/flux.2-klein-4b", "flux.2-klein-4b"],
}
VOCAB_ANIMATION_PROVIDER_LABELS = {
    "gemini": "Gemini / Veo",
    "openai": "OpenAI / Sora",
}
VOCAB_ANIMATION_MODEL_OPTIONS = {
    "gemini": ["veo-2.0-generate-001"],
    "openai": ["sora-2", "sora-2-pro"],
}
VOCAB_TEXT_LLM_OPENAI_MODEL_DEFAULTS = {
    "14": "gpt-5.2",
}
VOCAB_TEXT_LLM_MODEL_DEFAULTS = {
    "3": {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    },
    "5": {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    },
    "12": {
        "gemini": "gemini-2.5-pro",
        "openai": "gpt-4o-mini",
        "nvidia": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    },
    "14": {
        "gemini": "gemini-2.5-pro",
        "openai": "gpt-5.2",
        "nvidia": "stepfun-ai/step-3-5-flash",
    },
    "20": {
        "gemini": "gemini-2.5-pro",
        "openai": "gpt-4o-mini",
        "nvidia": "meta/llama-3.3-70b-instruct",
    },
}
VOCAB_TEXT_LLM_MODEL_ENV_KEYS = {
    "3": {
        "gemini": "CAP_VOCAB_GEMINI_MODEL",
        "openai": "CAP_VOCAB_OPENAI_MODEL",
        "nvidia": "CAP_VOCAB_NVIDIA_MODEL",
    },
    "5": {
        "gemini": "CAP_CLOZE_MODEL",
        "openai": "CAP_CLOZE_OPENAI_MODEL",
        "nvidia": "CAP_CLOZE_NVIDIA_MODEL",
    },
    "12": {
        "gemini": "CAP_SUBTITLE_REVIEW_GEMINI_MODEL",
        "openai": "CAP_SUBTITLE_REVIEW_OPENAI_MODEL",
        "nvidia": "CAP_SUBTITLE_REVIEW_NVIDIA_MODEL",
    },
    "14": {
        "gemini": "CAP_STORYBOARD_GEMINI_MODEL",
        "openai": "CAP_STORYBOARD_OPENAI_MODEL",
        "nvidia": "CAP_STORYBOARD_NVIDIA_MODEL",
    },
    "20": {
        "gemini": "CAP_YOUTUBE_META_GEMINI_MODEL",
        "openai": "CAP_YOUTUBE_META_OPENAI_MODEL",
        "nvidia": "CAP_YOUTUBE_META_NVIDIA_MODEL",
    },
}
VOCAB_TEXT_LLM_EXTRA_MODEL_DEFAULTS = {
    "12_nvidia_fallback": {
        "nvidia": "nvidia/llama-3.3-nemotron-super-49b-v1,nvidia/llama-3.1-nemotron-nano-8b-v1",
    },
    "5_review": {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
    },
}
VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS = {
    "5_review": "auto",
}
VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS = {
    "auto": "Auto",
    "gemini": "Gemini",
    "openai": "OpenAI / ChatGPT",
}
VOCAB_TEXT_LLM_EXTRA_PROVIDER_ENV_KEYS = {
    "5_review": "CAP_CLOZE_REVIEW_PROVIDER",
}
VOCAB_TEXT_LLM_EXTRA_MODEL_ENV_KEYS = {
    "12_nvidia_fallback": {
        "nvidia": "CAP_SUBTITLE_REVIEW_NVIDIA_FALLBACK_MODEL",
    },
    "5_review": {
        "gemini": "CAP_CLOZE_REVIEW_MODEL",
        "openai": "CAP_CLOZE_OPENAI_REVIEW_MODEL",
    },
}
VOCAB_IMAGE_PROVIDER_DEFAULTS = {
    "16": "auto",
    "18": "auto",
}
VOCAB_IMAGE_MODEL_DEFAULTS = {
    "gemini": "imagen-4.0-generate-001",
    "openai": "gpt-image-1.5",
    "nvidia": "black-forest-labs/flux.2-klein-4b",
}
VOCAB_IMAGE_MODEL_STEP_DEFAULTS = {
    "16": dict(VOCAB_IMAGE_MODEL_DEFAULTS),
    "18": dict(VOCAB_IMAGE_MODEL_DEFAULTS),
}
VOCAB_ANIMATION_PROVIDER_DEFAULT = "gemini"
VOCAB_ANIMATION_MODEL_DEFAULTS = {
    "gemini": "veo-2.0-generate-001",
    "openai": "sora-2",
}


def profile_llm_provider_settings_path(profile_id: str) -> Path:
    return ROOT / "config" / str(profile_id or "vocab") / "llm_provider_settings.json"


def load_profile_llm_provider_settings(profile_id: str) -> dict:
    settings = dict(VOCAB_TEXT_LLM_STEP_DEFAULTS)
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        providers = payload.get("vocab_text_llm_providers", payload)
        if isinstance(providers, dict):
            for step_no, provider in providers.items():
                step_key = str(step_no)
                provider_key = str(provider or "").strip().lower()
                if step_key in settings and provider_key in VOCAB_TEXT_LLM_PROVIDER_LABELS:
                    settings[step_key] = provider_key
    return settings


def default_vocab_text_llm_model_settings() -> dict:
    return {
        step_no: dict(provider_models)
        for step_no, provider_models in VOCAB_TEXT_LLM_MODEL_DEFAULTS.items()
    }


def default_vocab_text_llm_extra_model_settings() -> dict:
    return {
        call_id: dict(provider_models)
        for call_id, provider_models in VOCAB_TEXT_LLM_EXTRA_MODEL_DEFAULTS.items()
    }


def default_vocab_text_llm_extra_provider_settings() -> dict:
    return dict(VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS)


def vocab_text_model_for_step(model_settings: dict | None, step_no: str, provider: str) -> str:
    step_key = str(step_no)
    provider_key = str(provider or "").strip().lower()
    if provider_key == "auto":
        provider_key = "nvidia" if step_key in {"12", "14"} else "gemini"
    step_models = (model_settings or {}).get(step_key, {})
    if not isinstance(step_models, dict):
        step_models = {}
    default_models = VOCAB_TEXT_LLM_MODEL_DEFAULTS.get(step_key, {})
    return str(step_models.get(provider_key) or default_models.get(provider_key) or "").strip()


def vocab_text_extra_model_for_call(model_settings: dict | None, call_id: str, provider: str) -> str:
    call_key = str(call_id)
    provider_key = str(provider or "").strip().lower()
    call_models = (model_settings or {}).get(call_key, {})
    if not isinstance(call_models, dict):
        call_models = {}
    default_models = VOCAB_TEXT_LLM_EXTRA_MODEL_DEFAULTS.get(call_key, {})
    return str(call_models.get(provider_key) or default_models.get(provider_key) or "").strip()


def vocab_text_model_summary(model_settings: dict | None, step_no: str, provider: str) -> str:
    step_key = str(step_no)
    provider_key = str(provider or "").strip().lower()
    if provider_key == "auto":
        fallback_order = ("nvidia", "openai", "gemini") if step_key in {"12", "14"} else ("gemini", "openai", "nvidia")
        parts = [
            f"{p}: {vocab_text_model_for_step(model_settings, step_no, p)}"
            for p in fallback_order
        ]
        return "auto fallback: " + " → ".join(parts)
    return f"{provider_key}: {vocab_text_model_for_step(model_settings, step_no, provider_key)}"


def vocab_text_provider_label(value: str, step_no: str | None = None) -> str:
    step_labels = VOCAB_TEXT_LLM_STEP_PROVIDER_LABELS.get(str(step_no or ""), {})
    return step_labels.get(value) or VOCAB_TEXT_LLM_PROVIDER_LABELS.get(value, value)


def load_profile_llm_extra_model_settings(profile_id: str) -> dict:
    settings = default_vocab_text_llm_extra_model_settings()
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        models = payload.get("vocab_text_llm_extra_models", {})
        if isinstance(models, dict):
            for call_id, provider_models in models.items():
                call_key = str(call_id)
                if call_key not in settings or not isinstance(provider_models, dict):
                    continue
                for provider, model in provider_models.items():
                    provider_key = str(provider or "").strip().lower()
                    model_name = str(model or "").strip()
                    if provider_key in settings[call_key] and model_name:
                        settings[call_key][provider_key] = model_name
    return settings


def load_profile_llm_extra_provider_settings(profile_id: str) -> dict:
    settings = default_vocab_text_llm_extra_provider_settings()
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        providers = payload.get("vocab_text_llm_extra_providers", {})
        if isinstance(providers, dict):
            for call_id, provider in providers.items():
                call_key = str(call_id)
                provider_key = str(provider or "").strip().lower()
                if call_key in settings and provider_key in VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS:
                    settings[call_key] = provider_key
    return settings


def load_profile_llm_model_settings(profile_id: str) -> dict:
    settings = default_vocab_text_llm_model_settings()
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        models = payload.get("vocab_text_llm_models", {})
        if isinstance(models, dict):
            for step_no, provider_models in models.items():
                step_key = str(step_no)
                if step_key not in settings or not isinstance(provider_models, dict):
                    continue
                for provider, model in provider_models.items():
                    provider_key = str(provider or "").strip().lower()
                    model_name = str(model or "").strip()
                    if provider_key in settings[step_key] and model_name:
                        settings[step_key][provider_key] = model_name
        legacy_openai_models = payload.get("vocab_text_llm_openai_models", {})
        if isinstance(legacy_openai_models, dict):
            for step_no, model in legacy_openai_models.items():
                step_key = str(step_no)
                model_name = str(model or "").strip()
                if step_key in settings and model_name:
                    settings[step_key]["openai"] = model_name
    return settings


def load_profile_image_provider_settings(profile_id: str) -> dict:
    settings = dict(VOCAB_IMAGE_PROVIDER_DEFAULTS)
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        providers = payload.get("vocab_image_providers", {})
        if isinstance(providers, dict):
            for step_no, provider in providers.items():
                step_key = str(step_no)
                provider_key = str(provider or "").strip().lower()
                if step_key in settings and provider_key in VOCAB_IMAGE_PROVIDER_LABELS:
                    settings[step_key] = provider_key
    return settings


def load_profile_image_model_settings(profile_id: str) -> dict:
    settings = dict(VOCAB_IMAGE_MODEL_DEFAULTS)
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        models = payload.get("vocab_image_models", {})
        if isinstance(models, dict):
            for provider, model in models.items():
                provider_key = str(provider or "").strip().lower()
                model_name = str(model or "").strip()
                if provider_key in settings and model_name:
                    settings[provider_key] = model_name
    return settings


def default_vocab_image_model_step_settings() -> dict:
    return {
        step_no: dict(provider_models)
        for step_no, provider_models in VOCAB_IMAGE_MODEL_STEP_DEFAULTS.items()
    }


def load_profile_image_model_step_settings(profile_id: str) -> dict:
    settings = default_vocab_image_model_step_settings()
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    legacy_models = load_profile_image_model_settings(profile_id)
    for step_models in settings.values():
        for provider, model in legacy_models.items():
            provider_key = str(provider or "").strip().lower()
            model_name = str(model or "").strip()
            if provider_key in step_models and model_name:
                step_models[provider_key] = model_name
    if isinstance(payload, dict):
        by_step = payload.get("vocab_image_models_by_step", {})
        if isinstance(by_step, dict):
            for step_no, provider_models in by_step.items():
                step_key = str(step_no)
                if step_key not in settings or not isinstance(provider_models, dict):
                    continue
                for provider, model in provider_models.items():
                    provider_key = str(provider or "").strip().lower()
                    model_name = str(model or "").strip()
                    if provider_key in settings[step_key] and model_name:
                        settings[step_key][provider_key] = model_name
    return settings


def vocab_image_model_for_step(model_settings_by_step: dict | None, step_no: str, provider: str) -> str:
    step_key = str(step_no)
    provider_key = str(provider or "").strip().lower()
    step_models = (model_settings_by_step or {}).get(step_key, {})
    if not isinstance(step_models, dict):
        step_models = {}
    default_models = VOCAB_IMAGE_MODEL_STEP_DEFAULTS.get(step_key, VOCAB_IMAGE_MODEL_DEFAULTS)
    return str(step_models.get(provider_key) or default_models.get(provider_key) or "").strip()


def vocab_image_model_summary(image_model_settings: dict | None, provider: str) -> str:
    image_model_settings = image_model_settings or {}
    models = {
        provider_key: str(image_model_settings.get(provider_key) or default_model)
        for provider_key, default_model in VOCAB_IMAGE_MODEL_DEFAULTS.items()
    }
    provider_key = str(provider or "").strip().lower()
    if provider_key == "auto":
        return "auto fallback: gemini: {gemini} → openai: {openai} → nvidia: {nvidia}".format(**models)
    if provider_key in models:
        return f"{provider_key}: {models[provider_key]}"
    return " / ".join(f"{key}: {value}" for key, value in models.items())


def load_profile_animation_provider_settings(profile_id: str) -> dict:
    settings = {
        "provider": VOCAB_ANIMATION_PROVIDER_DEFAULT,
        "models": dict(VOCAB_ANIMATION_MODEL_DEFAULTS),
    }
    payload = read_json_file(profile_llm_provider_settings_path(profile_id)) or {}
    if isinstance(payload, dict):
        provider = str(payload.get("vocab_animation_provider") or "").strip().lower()
        if provider in VOCAB_ANIMATION_PROVIDER_LABELS:
            settings["provider"] = provider
        models = payload.get("vocab_animation_models", {})
        if isinstance(models, dict):
            for provider_key, model in models.items():
                provider_key = str(provider_key or "").strip().lower()
                model_name = str(model or "").strip()
                if provider_key in settings["models"] and model_name:
                    settings["models"][provider_key] = model_name
    return settings


def save_profile_llm_provider_settings(
    profile_id: str,
    settings: dict,
    model_settings: dict | None = None,
    image_provider_settings: dict | None = None,
    image_model_settings: dict | None = None,
    animation_settings: dict | None = None,
    extra_model_settings: dict | None = None,
    extra_provider_settings: dict | None = None,
    image_model_step_settings: dict | None = None,
) -> Path:
    path = profile_llm_provider_settings_path(profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_json_file(path) or {}
    if model_settings is None and isinstance(existing, dict):
        model_settings = existing.get("vocab_text_llm_models", existing.get("vocab_text_llm_openai_models", {}))
    if extra_model_settings is None and isinstance(existing, dict):
        extra_model_settings = existing.get("vocab_text_llm_extra_models", {})
    if extra_provider_settings is None and isinstance(existing, dict):
        extra_provider_settings = existing.get("vocab_text_llm_extra_providers", {})
    if image_provider_settings is None and isinstance(existing, dict):
        image_provider_settings = existing.get("vocab_image_providers", {})
    if image_model_settings is None and isinstance(existing, dict):
        image_model_settings = existing.get("vocab_image_models", {})
    if image_model_step_settings is None and isinstance(existing, dict):
        image_model_step_settings = existing.get("vocab_image_models_by_step", {})
    if animation_settings is None and isinstance(existing, dict):
        animation_settings = {
            "provider": existing.get("vocab_animation_provider", VOCAB_ANIMATION_PROVIDER_DEFAULT),
            "models": existing.get("vocab_animation_models", {}),
        }
    model_settings = model_settings or {}
    extra_model_settings = extra_model_settings or {}
    extra_provider_settings = extra_provider_settings or {}
    image_provider_settings = image_provider_settings or {}
    image_model_settings = image_model_settings or {}
    image_model_step_settings = image_model_step_settings or {}
    animation_settings = animation_settings or {}
    animation_provider = str(animation_settings.get("provider") or VOCAB_ANIMATION_PROVIDER_DEFAULT)
    animation_models = animation_settings.get("models") if isinstance(animation_settings.get("models"), dict) else {}
    default_text_models = default_vocab_text_llm_model_settings()
    normalized_text_models = {}
    for step_no, provider_defaults in default_text_models.items():
        incoming_step = model_settings.get(step_no, {}) if isinstance(model_settings, dict) else {}
        if not isinstance(incoming_step, dict):
            incoming_step = {"openai": incoming_step}
        normalized_text_models[step_no] = {
            provider: str(incoming_step.get(provider) or default_model)
            for provider, default_model in provider_defaults.items()
        }
    default_extra_models = default_vocab_text_llm_extra_model_settings()
    normalized_extra_models = {}
    for call_id, provider_defaults in default_extra_models.items():
        incoming_call = extra_model_settings.get(call_id, {}) if isinstance(extra_model_settings, dict) else {}
        if not isinstance(incoming_call, dict):
            incoming_call = {}
        normalized_extra_models[call_id] = {
            provider: str(incoming_call.get(provider) or default_model)
            for provider, default_model in provider_defaults.items()
        }
    payload = {
        "vocab_text_llm_providers": {
            step_no: str(settings.get(step_no) or default_provider)
            for step_no, default_provider in VOCAB_TEXT_LLM_STEP_DEFAULTS.items()
        },
        "vocab_text_llm_models": normalized_text_models,
        "vocab_text_llm_extra_providers": {
            call_id: str(extra_provider_settings.get(call_id) or default_provider)
            for call_id, default_provider in VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS.items()
        },
        "vocab_text_llm_extra_models": normalized_extra_models,
        "vocab_text_llm_openai_models": {
            step_no: str(normalized_text_models.get(step_no, {}).get("openai") or default_model)
            for step_no, default_model in VOCAB_TEXT_LLM_OPENAI_MODEL_DEFAULTS.items()
        },
        "vocab_image_providers": {
            step_no: str(image_provider_settings.get(step_no) or default_provider)
            for step_no, default_provider in VOCAB_IMAGE_PROVIDER_DEFAULTS.items()
        },
        "vocab_image_models": {
            provider: str(image_model_settings.get(provider) or default_model)
            for provider, default_model in VOCAB_IMAGE_MODEL_DEFAULTS.items()
        },
        "vocab_image_models_by_step": {
            step_no: {
                provider: str(
                    (image_model_step_settings.get(step_no, {}) if isinstance(image_model_step_settings.get(step_no, {}), dict) else {}).get(provider)
                    or image_model_settings.get(provider)
                    or default_model
                )
                for provider, default_model in provider_defaults.items()
            }
            for step_no, provider_defaults in VOCAB_IMAGE_MODEL_STEP_DEFAULTS.items()
        },
        "vocab_animation_provider": (
            animation_provider if animation_provider in VOCAB_ANIMATION_PROVIDER_LABELS else VOCAB_ANIMATION_PROVIDER_DEFAULT
        ),
        "vocab_animation_models": {
            provider: str(animation_models.get(provider) or default_model)
            for provider, default_model in VOCAB_ANIMATION_MODEL_DEFAULTS.items()
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_vocab_llm_runtime_settings(
    step_no: str,
    step_def: dict,
    provider: str,
    model_settings: dict | None = None,
    extra_model_settings: dict | None = None,
    extra_provider_settings: dict | None = None,
) -> dict:
    updated = dict(step_def)
    args_list = list(updated.get("args") or [])
    if "--provider" in args_list:
        provider_idx = args_list.index("--provider")
        if provider_idx + 1 < len(args_list):
            args_list[provider_idx + 1] = provider
        else:
            args_list.append(provider)
    else:
        args_list.extend(["--provider", provider])
    updated["args"] = args_list

    step_key = str(step_no)
    model_env_keys = VOCAB_TEXT_LLM_MODEL_ENV_KEYS.get(step_key, {})
    if model_env_keys:
        env = dict(updated.get("env") or {})
        for model_provider, env_key in model_env_keys.items():
            model_name = vocab_text_model_for_step(model_settings, step_key, model_provider)
            if model_name:
                env[env_key] = model_name
        if step_key == "5":
            review_provider = str((extra_provider_settings or {}).get("5_review") or VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS["5_review"]).strip().lower()
            if review_provider in VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS:
                env[VOCAB_TEXT_LLM_EXTRA_PROVIDER_ENV_KEYS["5_review"]] = review_provider
            for model_provider, env_key in VOCAB_TEXT_LLM_EXTRA_MODEL_ENV_KEYS["5_review"].items():
                model_name = vocab_text_extra_model_for_call(extra_model_settings, "5_review", model_provider)
                if model_name:
                    env[env_key] = model_name
        if step_key == "12":
            for model_provider, env_key in VOCAB_TEXT_LLM_EXTRA_MODEL_ENV_KEYS["12_nvidia_fallback"].items():
                model_name = vocab_text_extra_model_for_call(extra_model_settings, "12_nvidia_fallback", model_provider)
                if model_name:
                    env[env_key] = model_name
        updated["env"] = env
    return updated


def apply_provider_arg(step_def: dict, provider: str) -> dict:
    updated = dict(step_def)
    args_list = list(updated.get("args") or [])
    if "--provider" in args_list:
        provider_idx = args_list.index("--provider")
        if provider_idx + 1 < len(args_list):
            args_list[provider_idx + 1] = provider
        else:
            args_list.append(provider)
    else:
        args_list.extend(["--provider", provider])
    updated["args"] = args_list
    return updated


def build_storyboard_media_env(image_models: dict | None, animation_settings: dict | None) -> dict:
    image_models = image_models or {}
    animation_settings = animation_settings or {}
    animation_models = animation_settings.get("models") if isinstance(animation_settings.get("models"), dict) else {}
    animation_provider = str(animation_settings.get("provider") or VOCAB_ANIMATION_PROVIDER_DEFAULT)
    return {
        "CAP_STORYBOARD_GEMINI_IMAGE_MODEL": str(image_models.get("gemini") or VOCAB_IMAGE_MODEL_DEFAULTS["gemini"]),
        "CAP_STORYBOARD_OPENAI_IMAGE_MODEL": str(image_models.get("openai") or VOCAB_IMAGE_MODEL_DEFAULTS["openai"]),
        "CAP_STORYBOARD_NVIDIA_IMAGE_MODEL": str(image_models.get("nvidia") or VOCAB_IMAGE_MODEL_DEFAULTS["nvidia"]),
        "CAP_ANIMATION_PROVIDER": animation_provider if animation_provider in VOCAB_ANIMATION_PROVIDER_LABELS else VOCAB_ANIMATION_PROVIDER_DEFAULT,
        "CAP_ANIMATION_MODEL": str(animation_models.get("gemini") or VOCAB_ANIMATION_MODEL_DEFAULTS["gemini"]),
        "CAP_OPENAI_VIDEO_MODEL": str(animation_models.get("openai") or VOCAB_ANIMATION_MODEL_DEFAULTS["openai"]),
    }

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

def get_stage_current_step(ep_path: Path, stage_id: str) -> str:
    log_path = log_file_for_stage(ep_path, stage_id)
    if not log_path.exists():
        return ""
    text = read_text_file(log_path, "")
    if not text:
        return ""
    run_pattern = re.compile(rf"^==== RUN {re.escape(str(stage_id))} \| (.+?) \| .+$", re.MULTILINE)
    done_pattern = re.compile(rf"^==== (?:OK|FAIL|SKIP) {re.escape(str(stage_id))} \| (.+?) \| .+$", re.MULTILINE)
    runs = [m.group(1).strip() for m in run_pattern.finditer(text)]
    dones = [m.group(1).strip() for m in done_pattern.finditer(text)]
    if not runs:
        return ""
    active_runs = []
    done_counts: dict[str, int] = {}
    for name in dones:
        done_counts[name] = done_counts.get(name, 0) + 1
    for name in runs:
        remain = done_counts.get(name, 0)
        if remain > 0:
            done_counts[name] = remain - 1
            continue
        active_runs.append(name)
    for name in reversed(active_runs):
        if name.startswith("No."):
            return name
    return ""

def format_episode_batch_label(episodes: list[int]) -> str:
    items = sorted([int(x) for x in episodes if str(x).strip()])
    if not items:
        return "—"
    if len(items) == 1:
        return f"Ep{items[0]:02d}"
    return f"Ep{items[0]:02d}–Ep{items[-1]:02d} ({len(items)} 集)"

def build_schedule_job_id(stage_id: str, episodes: list[int]) -> str:
    now_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    ordered_episodes = sorted([int(x) for x in episodes]) if episodes else []
    first_ep = int(ordered_episodes[0]) if ordered_episodes else 0
    last_ep = int(ordered_episodes[-1]) if ordered_episodes else first_ep
    return f"sched_stage{str(stage_id).replace('.', '_')}_ep{first_ep:02d}_{last_ep:02d}_{now_text}"

def get_schedule_display_status(job: Dict, state: Dict) -> str:
    meta_status = str(job.get("status", "queued")).strip().lower() or "queued"
    if state.get("running"):
        return "running"
    if meta_status == "queued":
        return "waiting"
    if meta_status == "running":
        return "stopped"
    return meta_status

def get_schedule_status_label(status_code: str) -> str:
    labels = {
        "waiting": "等待中",
        "running": "執行中",
        "done": "執行完成",
        "partial": "部分完成",
        "error": "執行錯誤",
        "stopped": "已停止",
        "queued": "等待中",
    }
    return labels.get(str(status_code or "").strip().lower(), str(status_code or "—"))

def render_schedule_workspace(eps):
    st.caption("建立背景排程 job。建立後不需要停留在此分頁；job 會在指定時間自動開始，並依序處理單集或多集。")
    st.caption("為了能在半夜準時執行，新的排程會建立 Windows 排程任務，並嘗試在指定時間喚醒電腦執行；電腦仍不可關機。")
    wake_policy = wake_timer_policy()
    if wake_policy.get("available"):
        ac_label = wake_policy.get("ac_label", "未知")
        dc_label = wake_policy.get("dc_label", "未知")
        if not wake_policy.get("ac_enabled") or not wake_policy.get("dc_enabled"):
            st.warning(
                "Windows 電源計畫可能阻止睡眠喚醒："
                f"允許喚醒計時器 AC={ac_label}、DC={dc_label}。"
                "若排程時電腦使用電池或 Modern Standby 進入低電源狀態，可能會延到手動喚醒後才執行。"
            )
        else:
            st.caption(f"Windows 喚醒計時器：AC={ac_label}、DC={dc_label}。")
    else:
        st.caption(f"無法檢查 Windows 喚醒計時器：{wake_policy.get('message', 'unknown')}")
    if st.session_state["profile_id"] == "vocab":
        st.caption("`4.1 將所有 AI 圖片轉成動畫` 會沿用「修改分鏡 / 替換單字卡」的批次動畫流程，並把成功產出的動畫回寫成該 Scene 的預設素材。")

    stage_options, stage_names = get_schedule_stage_options(st.session_state["profile_id"])
    available_eps = sorted(
        (int(e["_raw"]["ep"]) for e in eps if e.get("_raw", {}).get("ep") is not None),
        reverse=True,
    )
    if not available_eps:
        st.info("尚未發現可排程的集數。")
        return

    ep_label_map = {ep_no: f"Ep{ep_no:02d}" for ep_no in available_eps}
    schedule_mode = st.radio(
        "排程範圍",
        ["單集", "多集範圍"],
        horizontal=True,
        key="pm_schedule_mode",
    )
    schedule_stage = st.selectbox(
        "選擇要排程的 Stage",
        stage_options,
        key="pm_schedule_stage",
        format_func=lambda sid: f"Stage {sid} {stage_names.get(str(sid), '')}".strip(),
    )

    if schedule_mode == "單集":
        selected_ep = st.selectbox(
            "選擇集數",
            available_eps,
            key="pm_schedule_single_ep",
            format_func=lambda ep_no: ep_label_map.get(ep_no, str(ep_no)),
        )
        scheduled_episodes = [int(selected_ep)]
    else:
        range_cols = st.columns(2)
        with range_cols[0]:
            start_ep = st.selectbox(
                "起始集數",
                available_eps,
                key="pm_schedule_start_ep",
                format_func=lambda ep_no: ep_label_map.get(ep_no, str(ep_no)),
            )
        with range_cols[1]:
            end_ep = st.selectbox(
                "結束集數",
                available_eps,
                index=0,
                key="pm_schedule_end_ep",
                format_func=lambda ep_no: ep_label_map.get(ep_no, str(ep_no)),
            )
        low_ep = min(int(start_ep), int(end_ep))
        high_ep = max(int(start_ep), int(end_ep))
        scheduled_episodes = sorted(ep_no for ep_no in available_eps if low_ep <= ep_no <= high_ep)
        st.caption(f"將依序處理：{format_episode_batch_label(scheduled_episodes)}")

    schedule_cols = st.columns(4)
    now_dt = datetime.now()
    today = now_dt.date()
    saved_run_date = st.session_state.get("pm_schedule_date")
    if isinstance(saved_run_date, datetime):
        saved_run_date = saved_run_date.date()
    if isinstance(saved_run_date, date) and saved_run_date < today:
        st.session_state["pm_schedule_date"] = today
    with schedule_cols[0]:
        run_date = st.date_input("執行日期", min_value=today, key="pm_schedule_date")
    with schedule_cols[1]:
        run_time = st.time_input("執行時間", key="pm_schedule_time")
    with schedule_cols[2]:
        create_schedule = st.button("建立排程", key="pm_schedule_create")
    with schedule_cols[3]:
        run_now = st.button("立即執行", key="pm_schedule_run_now")

    if create_schedule:
        if not scheduled_episodes:
            st.error("沒有可排程的集數。")
        else:
            scheduled_at = datetime.combine(run_date, run_time)
            if scheduled_at <= now_dt:
                st.error("排程時間不能設定為過去時間，請選擇晚於現在的時間。")
            else:
                job_id = build_schedule_job_id(schedule_stage, scheduled_episodes)
                stage_label = f"Stage {schedule_stage} {stage_names.get(str(schedule_stage), '')}".strip()
                job_payload = {
                    "id": job_id,
                    "profile_id": st.session_state["profile_id"],
                    "stage_id": str(schedule_stage),
                    "stage_name": stage_names.get(str(schedule_stage), ""),
                    "episodes": scheduled_episodes,
                    "schedule_at": scheduled_at.isoformat(timespec="seconds"),
                    "status": "queued",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "started_at": None,
                    "ended_at": None,
                    "current_episode": None,
                    "completed_episodes": [],
                    "partial_episodes": [],
                    "failed_episodes": [],
                    "last_message": f"queued for {scheduled_at.isoformat(sep=' ', timespec='minutes')}",
                }
                write_schedule_job(ROOT, st.session_state["profile_id"], job_id, job_payload)
                res = start_schedule_job(
                    ROOT,
                    st.session_state["profile_id"],
                    (ROOT / WS_ROOT).resolve(),
                    PATH_STAGES,
                    job_id,
                    f"排程執行 {stage_label}",
                )
                if res.get("ok"):
                    success_text = (
                        f"已建立排程：{stage_label} | {format_episode_batch_label(scheduled_episodes)} | "
                        f"{scheduled_at.isoformat(sep=' ', timespec='minutes')}"
                    )
                    if res.get("wake_to_run", True):
                        st.success(success_text)
                    else:
                        st.warning(success_text + "。此排程已退回一般 Task Scheduler 建立方式，無法保證睡眠時自動喚醒。")
                else:
                    st.error(f"排程啟動失敗：{res.get('message')}")

    if run_now:
        if not scheduled_episodes:
            st.error("沒有可執行的集數。")
        else:
            started_at = datetime.now()
            job_id = build_schedule_job_id(schedule_stage, scheduled_episodes)
            stage_label = f"Stage {schedule_stage} {stage_names.get(str(schedule_stage), '')}".strip()
            job_payload = {
                "id": job_id,
                "profile_id": st.session_state["profile_id"],
                "stage_id": str(schedule_stage),
                "stage_name": stage_names.get(str(schedule_stage), ""),
                "episodes": scheduled_episodes,
                "schedule_at": started_at.isoformat(timespec="seconds"),
                "status": "queued",
                "created_at": started_at.isoformat(timespec="seconds"),
                "started_at": None,
                "ended_at": None,
                "current_episode": None,
                "completed_episodes": [],
                "partial_episodes": [],
                "failed_episodes": [],
                "last_message": "starting immediately",
            }
            write_schedule_job(ROOT, st.session_state["profile_id"], job_id, job_payload)
            res = start_schedule_job_now(
                ROOT,
                st.session_state["profile_id"],
                (ROOT / WS_ROOT).resolve(),
                PATH_STAGES,
                job_id,
                f"立即執行 {stage_label}",
            )
            if res.get("ok"):
                st.success(
                    f"已開始背景執行：{stage_label} | {format_episode_batch_label(scheduled_episodes)}"
                )
            else:
                st.error(f"立即執行失敗：{res.get('message')}")

    auto_refresh_schedule = st.toggle(
        "自動刷新排程 Log",
        value=True,
        key="pm_schedule_auto_refresh",
        help="有排程 job 等待或執行中時，每 5 秒自動刷新一次。",
    )

    @st.fragment(run_every=5 if auto_refresh_schedule else None)
    def render_schedule_jobs_panel():
        jobs = list_schedule_jobs(ROOT, st.session_state["profile_id"])
        if not jobs:
            st.caption("尚無排程 job。")
            return

        top_actions = st.columns([1.2, 1.6, 4])
        top_actions[0].button("只刷新排程狀態", key="refresh_schedule_jobs_fragment")
        if top_actions[1].button("清除歷史記錄", key="cleanup_schedule_history"):
            res = cleanup_schedule_history(ROOT, st.session_state["profile_id"])
            deleted_count = len(res.get("deleted_ids") or [])
            skipped_count = len(res.get("skipped_ids") or [])
            error_count = len(res.get("errors") or [])
            if error_count:
                st.error(f"清除完成，但有 {error_count} 筆失敗。")
            else:
                st.success(f"已清除 {deleted_count} 筆歷史記錄。")
            if skipped_count:
                st.caption(f"略過 {skipped_count} 筆執行中的排程。")
            st.rerun()
        top_actions[2].caption("清除歷史記錄會刪除非執行中的排程 metadata 與 log，保留執行中的排程。")
        for idx, job in enumerate(jobs[:12]):
            job_id = str(job.get("id", ""))
            state = get_schedule_job_state(ROOT, st.session_state["profile_id"], job_id)
            display_status = get_schedule_display_status(job, state)
            display_status_label = get_schedule_status_label(display_status)
            episodes_text = format_episode_batch_label([int(x) for x in (job.get("episodes") or [])])
            scheduled_at_text = str(job.get("schedule_at", "")).replace("T", " ")
            title = f"[{display_status_label}] Stage {job.get('stage_id')} {job.get('stage_name', '')} | {episodes_text}"
            with st.expander(title, expanded=bool(state.get("running"))):
                meta_cols = st.columns(4)
                meta_cols[0].metric("狀態", display_status_label)
                meta_cols[1].metric("排程時間", scheduled_at_text or "—")
                meta_cols[2].metric("目前集數", job.get("current_episode") or "—")
                meta_cols[3].metric("PID", state.get("pid") or "—")
                task_mode = str(job.get("task_mode", "")).strip() or "—"
                wake_label = "可喚醒" if bool(job.get("wake_to_run", False)) else "一般"
                st.caption(f"排程模式：{task_mode} | 喚醒能力：{wake_label}")
                action_cols = st.columns([1.2, 1.2, 3.6])
                delete_disabled = display_status == "running"
                if action_cols[0].button(
                    "刪除排程",
                    key=f"delete_sched_{job_id or 'unknown'}_{idx}",
                    disabled=delete_disabled,
                ):
                    res = delete_schedule_job(ROOT, st.session_state["profile_id"], job_id)
                    if res.get("ok"):
                        st.success(f"已刪除排程：{job_id}")
                        st.rerun()
                    else:
                        st.error(f"刪除失敗：{res.get('message')}")
                if action_cols[1].button(
                    "中斷執行",
                    key=f"stop_sched_{job_id or 'unknown'}_{idx}",
                    disabled=display_status != "running",
                ):
                    res = stop_schedule_job(ROOT, st.session_state["profile_id"], job_id)
                    if res.get("ok"):
                        st.success(f"已中斷排程：{job_id}")
                        st.rerun()
                    else:
                        st.error(f"中斷失敗：{res.get('message')}")
                if delete_disabled:
                    action_cols[2].caption("執行中的排程需先中斷；成功中斷後即可刪除。")
                else:
                    action_cols[2].caption("可刪除等待中、執行完成、執行錯誤、已停止的排程。")
                st.caption(f"Job ID: {job_id}")
                st.caption(f"Last Message: {job.get('last_message', '')}")
                st.caption(
                    f"完成：{', '.join([f'Ep{int(x):02d}' for x in (job.get('completed_episodes') or [])]) or '—'} | "
                    f"部分完成：{', '.join([f'Ep{int(x):02d}' for x in (job.get('partial_episodes') or [])]) or '—'} | "
                    f"失敗：{', '.join([f'Ep{int(x):02d}' for x in (job.get('failed_episodes') or [])]) or '—'}"
                )
                log_path = Path(schedule_job_log_path(ROOT, st.session_state["profile_id"], job_id))
                if log_path.exists():
                    st.download_button(
                        "下載排程 log",
                        data=log_path.read_bytes(),
                        file_name=log_path.name,
                        key=f"dl_sched_{job_id or 'unknown'}_{idx}",
                    )
                    st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
                else:
                    st.caption("尚無排程 log。")

    render_schedule_jobs_panel()

def shared_prompt_dir(profile_id: str) -> Path:
    return ROOT / "config" / str(profile_id or "default").strip() / "prompts"

def cloze_prompt_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "cloze_quiz_prompt.txt"

def ensure_cloze_prompt_file(profile_id: str) -> Path:
    prompt_path = cloze_prompt_path(profile_id)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_CLOZE_PROMPT_TEMPLATE)
    return prompt_path

def cover_image_prompt_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "cover_image_prompt.txt"

def ensure_cover_image_prompt_file(profile_id: str) -> Path:
    prompt_path = cover_image_prompt_path(profile_id)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_COVER_IMAGE_PROMPT_TEMPLATE)
    return prompt_path

def cover_prompt_mode_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "cover_prompt_mode.txt"

def ensure_cover_prompt_mode_file(profile_id: str) -> Path:
    mode_path = cover_prompt_mode_path(profile_id)
    if not mode_path.exists():
        mode_path.parent.mkdir(parents=True, exist_ok=True)
        mode_path.write_text("direct\n", encoding="utf-8")
    return mode_path

def read_cover_prompt_mode(profile_id: str) -> str:
    mode_path = ensure_cover_prompt_mode_file(profile_id)
    value = str(read_text_file(mode_path, "direct")).strip().lower()
    return value if value in {"direct", "gemini"} else "direct"

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

def cover_prompt_output_path(ep_path: Path, ep_num: int) -> Path:
    return images_dir(ep_path) / f"cover_prompt_ep{ep_num:02d}.txt"

def cover_candidate_output_path(ep_path: Path) -> Path:
    return images_dir(ep_path) / "cover_cute.png"

def render_copy_button(text: str, *, key: str, label: str = "快速複製") -> None:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    payload = json.dumps(str(text or ""))
    components.html(
        f"""
        <div style="display:flex;align-items:center;gap:8px;">
          <button id="btn_{safe_key}" style="padding:0.3rem 0.75rem;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer;">
            {label}
          </button>
          <span id="status_{safe_key}" style="font-size:0.9rem;color:#0f766e;"></span>
        </div>
        <script>
          const btn = document.getElementById("btn_{safe_key}");
          const status = document.getElementById("status_{safe_key}");
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText({payload});
              status.textContent = "已複製";
            }} catch (err) {{
              status.textContent = "複製失敗";
            }}
            setTimeout(() => {{ status.textContent = ""; }}, 1500);
          }});
        </script>
        """,
        height=44,
    )

def render_text_output_expander(title: str, path: Path | None, *, text_label: str, key_prefix: str, expanded: bool = False, height: int = 320):
    if not path or not path.exists():
        return
    file_text = read_text_file(path, "")
    text_key = f"{key_prefix}_text_output"
    loaded_key = f"{text_key}__loaded"
    sync_textarea_state(text_key, loaded_key, file_text)
    with st.expander(title, expanded=expanded):
        st.caption(f"輸出檔案：{path}")
        render_copy_button(file_text, key=f"{key_prefix}_copy_output", label="快速複製輸出")
        st.text_area(
            text_label,
            key=text_key,
            height=height,
        )


def render_subtitle_compare_panel(ep_path: Path, *, key_prefix: str, expanded: bool = False):
    raw_path = subtitles_raw_path(ep_path)
    fixed_path = subtitles_fixed_path(ep_path)
    raw_text = read_text_file(raw_path, "")
    fixed_text = read_text_file(fixed_path, "")

    with st.expander("No.11 / No.12 字幕比對", expanded=expanded):
        st.caption("No.11 原始字幕保留在 notebooklm_audio.srt；No.12 校對後字幕輸出到 notebooklm_audio_fixed.srt。")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("No.11 Whisper 原始字幕")
            st.caption(str(raw_path))
            if raw_path.exists():
                st.download_button(
                    "下載 No.11 原始字幕",
                    data=raw_path.read_bytes(),
                    file_name=raw_path.name,
                    key=f"{key_prefix}_download_raw_srt",
                )
            st.text_area(
                "原始字幕內容",
                value=raw_text,
                height=420,
                key=f"{key_prefix}_raw_srt",
                disabled=True,
            )
        with c2:
            st.subheader("No.12 AI 校對字幕")
            st.caption(str(fixed_path))
            if fixed_path.exists():
                st.download_button(
                    "下載 No.12 校對字幕",
                    data=fixed_path.read_bytes(),
                    file_name=fixed_path.name,
                    key=f"{key_prefix}_download_fixed_srt",
                )
            st.text_area(
                "校對字幕內容",
                value=fixed_text,
                height=420,
                key=f"{key_prefix}_fixed_srt",
                disabled=True,
            )
        if not raw_path.exists():
            st.warning("尚未找到 No.11 原始字幕，請先執行 No.11 Whisper 產生字幕。")
        if not fixed_path.exists():
            st.caption("尚未找到 No.12 校對字幕；執行 No.12 後即可比對。")


def render_final_prompt_expander(title: str, prompt_text: str, *, key_prefix: str, expanded: bool = False, height: int = 360):
    text_key = f"{key_prefix}_final_prompt"
    loaded_key = f"{text_key}__loaded"
    sync_textarea_state(text_key, loaded_key, prompt_text)
    with st.expander(title, expanded=expanded):
        st.caption("以下內容為目前 Prompt 模板套入變數後的最終送出內容預覽。")
        render_copy_button(prompt_text, key=f"{key_prefix}_copy_final", label="快速複製最終 Prompt")
        st.text_area(
            title,
            key=text_key,
            height=height,
        )

def build_prompt_source_rows(profile_id: str) -> list[Dict]:
    if profile_id == "single_video":
        return [
            {
                "Step": "0",
                "用途": "AI 封面圖 Prompt",
                "Prompt 檔案": str(cover_image_prompt_path(profile_id)),
                "缺檔時預設": "DEFAULT_COVER_IMAGE_PROMPT_TEMPLATE",
                "主要資料來源": "video_info.json + 字幕",
            },
            {
                "Step": "3",
                "用途": "AI 字幕校對",
                "Prompt 檔案": str(subtitle_review_prompt_path(profile_id)),
                "缺檔時預設": "DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE",
                "主要資料來源": "02_subtitles/notebooklm_audio.srt",
            },
            {
                "Step": "5",
                "用途": "AI 分鏡規劃",
                "Prompt 檔案": str(storyboard_prompt_path(profile_id)),
                "缺檔時預設": "DEFAULT_STORYBOARD_PROMPT_TEMPLATE",
                "主要資料來源": "校對字幕 + YouTube Title",
            },
            {
                "Step": "10",
                "用途": "YouTube Metadata",
                "Prompt 檔案": str(youtube_meta_prompt_path(profile_id)),
                "缺檔時預設": "DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE",
                "主要資料來源": "字幕 + YouTube Title",
            },
        ]
    return [
        {
            "Step": "5",
            "用途": "克漏字題目生成",
            "Prompt 檔案": str(cloze_prompt_path(profile_id)),
            "缺檔時預設": "DEFAULT_CLOZE_PROMPT_TEMPLATE",
            "主要資料來源": "03_storyboards/vocab_data.csv",
        },
        {
            "Step": "7.1",
            "用途": "AI 封面圖 Prompt",
            "Prompt 檔案": str(cover_image_prompt_path(profile_id)),
            "缺檔時預設": "DEFAULT_COVER_IMAGE_PROMPT_TEMPLATE",
            "主要資料來源": "03_storyboards/vocab_data.csv + core/assets/host_profiles.json",
        },
        {
            "Step": "8",
            "用途": "NotebookLM 節目摘要 Prompt",
            "Prompt 檔案": str(notebooklm_prompt_template_path(profile_id)),
            "缺檔時預設": "DEFAULT_NOTEBOOKLM_PROMPT_TEMPLATE",
            "主要資料來源": "03_storyboards/vocab_data.csv + 前/本集 cloze 題",
        },
        {
            "Step": "12",
            "用途": "AI 字幕校對",
            "Prompt 檔案": str(subtitle_review_prompt_path(profile_id)),
            "缺檔時預設": "DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE",
            "主要資料來源": "02_subtitles/notebooklm_audio.srt",
        },
        {
            "Step": "14",
            "用途": "AI 分鏡規劃",
            "Prompt 檔案": str(storyboard_prompt_path(profile_id)),
            "缺檔時預設": "DEFAULT_STORYBOARD_PROMPT_TEMPLATE",
            "主要資料來源": "storyboard vocab + 校對字幕 + host/cloze 規則",
        },
        {
            "Step": "20",
            "用途": "YouTube Metadata",
            "Prompt 檔案": str(youtube_meta_prompt_path(profile_id)),
            "缺檔時預設": "DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE",
            "主要資料來源": "字幕 + 集數範圍",
        },
    ]

def render_prompt_source_summary(profile_id: str):
    st.caption("每個步驟會優先讀取目前 profile 的共用 Prompt 檔；如果檔案不存在，系統會先用內建預設模板建立。")
    st.caption("舊的單集 prompt 檔不再是主要來源；現在調整一次，會影響同一個 profile 後續所有集數。")
    st.dataframe(pd.DataFrame(build_prompt_source_rows(profile_id)), use_container_width=True, hide_index=True)

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

def _preview_image_signature(path_str: str) -> tuple[int, int]:
    path = Path(path_str)
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size

@st.cache_data(show_spinner=False, max_entries=256)
def _load_preview_image_bytes_cached(
    path_str: str,
    file_mtime_ns: int,
    file_size: int,
    max_width: int = 640,
):
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

def load_preview_image_bytes(path_str: str, max_width: int = 640):
    path = Path(path_str)
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        signature = _preview_image_signature(path_str)
    except Exception as e:
        return None, str(e)
    return _load_preview_image_bytes_cached(
        str(path),
        signature[0],
        signature[1],
        max_width=max_width,
    )

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
    audio_path = audio_merged_path(ep_path)
    file_text = read_text_file(subtitle_path, "")
    text_key = f"{key_prefix}_subtitle_text"
    loaded_key = f"{text_key}__loaded"
    marker_key = f"{key_prefix}_subtitle_done"
    sync_textarea_state(text_key, loaded_key, file_text)
    if marker_key not in st.session_state:
        st.session_state[marker_key] = has_manual_marker(ep_path, step_no)
    dirty = st.session_state.get(text_key, "") != file_text

    with st.expander(f"No.{step_no} 人工字幕校正", expanded=expanded):
        st.caption(f"語音檔案：{audio_path}")
        if audio_path.exists():
            st.audio(str(audio_path))
        else:
            upload_hint = "No.1 上傳語音音訊" if st.session_state.get("profile_id") == "single_video" else "No.10 上傳語音音訊與 No.10.5 合併片頭音訊"
            st.warning(f"尚未找到語音檔，請先完成 {upload_hint}。")
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

        render_subtitle_compare_panel(ep_path, key_prefix=f"{key_prefix}_compare", expanded=False)

        if subtitle_path.exists():
            st.download_button(
                "下載字幕",
                data=subtitle_path.read_bytes(),
                file_name=subtitle_path.name,
                key=f"{key_prefix}_download_subtitle",
            )
        else:
            st.caption("目前尚無校對字幕；可直接在上方貼上內容後儲存建立。")

def render_prompts_workspace(
    info: Dict,
    ep_path: Path,
    profile_id: str,
    subtitle_path: Path,
    meta_path: Path | None,
    *,
    show_header: bool = True,
):
    if show_header:
        st.subheader("Prompt 編輯與手動重跑")
        st.caption("這裡編輯的是目前 profile 共用的 prompt 模板。可先修改、存檔，再手動執行對應子步驟。未存檔時，執行按鈕會鎖住。")

    notice = st.session_state.get("studio_prompt_notice")
    if isinstance(notice, dict):
        if notice.get("ok"):
            st.success(f"Step {notice.get('task')} 任務已啟動。log: {notice.get('log')}")
        else:
            st.error(f"Step {notice.get('task')} 任務啟動失敗：{notice.get('message')}")
        st.session_state.pop("studio_prompt_notice", None)

    prompt_auto_refresh = st.toggle(
        "自動刷新執行中的 Prompt Log",
        value=True,
        key=f"prompt_auto_refresh_{info['ep']}_{profile_id}_{int(show_header)}",
        help="有 Prompt 相關子步驟執行中時，每 2 秒自動更新一次狀態與 Log。",
    )

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
        extra_renderer = None,
    ):
        file_text = read_text_file(prompt_path, default_template)
        text_key = f"studio_prompt_{step_no}_{info['ep']}_{profile_id}"
        loaded_key = f"{text_key}__loaded"
        sync_textarea_state(text_key, loaded_key, file_text)
        current_text = st.session_state.get(text_key, "")
        dirty = current_text != file_text
        state = runner.get_substep_state(info, step_no)

        with st.expander(f"No.{step_no} {title}", expanded=bool(state.get("running")) or expanded):
            top1, top2, top3 = st.columns([1, 1, 4])
            top1.metric("Prompt", "已建立" if prompt_path.exists() else "缺少")
            top2.metric("狀態", "執行中" if state.get("running") else "待命")
            top3.caption(f"Prompt 檔案：{prompt_path}")
            if dirty:
                st.warning("目前編輯內容尚未存檔。請先儲存，再執行這個步驟。")
            if help_text:
                st.caption(help_text)
            if extra_renderer:
                extra_renderer()
            render_copy_button(current_text, key=f"copy_prompt_{step_no}_{info['ep']}", label="快速複製目前 Prompt")
            st.text_area(
                f"{title} Prompt",
                key=text_key,
                height=300,
            )
            action1, action2, action3, action4 = st.columns([1, 1, 1, 1.2])
            with action1:
                save_prompt = st.button("儲存 Prompt", key=f"save_prompt_{step_no}_{info['ep']}")
            with action2:
                reload_prompt = st.button("還原檔案內容", key=f"reload_prompt_{step_no}_{info['ep']}")
            with action3:
                load_default_prompt = st.button("載入系統預設", key=f"default_prompt_{step_no}_{info['ep']}")
            with action4:
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
            if load_default_prompt:
                st.session_state[text_key] = default_template
                st.info("已載入系統預設模板，尚未存檔。")
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

    if profile_id == "single_video":
        title = single_video_title(ep_path)
        raw_subtitle_path = ep_path / "02_subtitles" / "notebooklm_audio.srt"
        fixed_subtitle_path = subtitles_fixed_path(ep_path)

        def render_single_prompt_preview(template_text: str, srt_path: Path | None = None) -> str:
            selected_srt_path = srt_path or fixed_subtitle_path
            scene_blocks = build_single_video_prompt_scene_blocks(selected_srt_path)
            return (
                str(template_text or "")
                .replace("{{EPISODE}}", str(info["ep"]))
                .replace("{{YOUTUBE_TITLE}}", title or "")
                .replace("{{SRT_CONTENT}}", read_text_file(selected_srt_path, ""))
                .replace("{{SCENE_BLOCKS}}", json.dumps(scene_blocks, ensure_ascii=False, indent=2))
            )

        single_step_defs = {
            "0": {
                "name": "AI 封面圖 Prompt",
                "type": "python",
                "script": "scripts/generate_ai_cover.py",
                "args": ["--ep", str(info["ep"])],
            },
            "3": {
                "name": "AI 字幕校對",
                "type": "python",
                "script": "scripts/llm_subtitle_reviewer.py",
                "args": ["--ep", str(info["ep"]), "--provider", "openai"],
            },
            "5": {
                "name": "AI 依語境分鏡規劃",
                "type": "python",
                "script": "scripts/llm_director.py",
                "args": ["--ep", str(info["ep"]), "--provider", "openai"],
            },
            "10": {
                "name": "產生 YouTube Metadata",
                "type": "python",
                "script": "scripts/generate_youtube_meta.py",
                "args": ["--ep", str(info["ep"])],
            },
        }
        prompt_step_ids = list(single_step_defs.keys())
        any_prompt_running = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in prompt_step_ids)
        prompt_refresh_interval = 2 if (prompt_auto_refresh and any_prompt_running) else None
        prompt_running_prev_key = f"prompt_running_prev_{info['ep']}_{profile_id}_{int(show_header)}"
        if any_prompt_running:
            st.session_state[prompt_running_prev_key] = True

        @st.fragment(run_every=prompt_refresh_interval)
        def render_single_prompt_panels():
            st.caption("只會刷新此面板，不會切換影片或跳離目前頁面。")
            st.button("只刷新 Prompt 狀態", key=f"refresh_prompt_blocks_{info['ep']}_{profile_id}_{int(show_header)}")
            latest_prompt_running = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in prompt_step_ids)
            if st.session_state.get(prompt_running_prev_key) and not latest_prompt_running:
                st.session_state[prompt_running_prev_key] = False
                st.rerun()
            st.session_state[prompt_running_prev_key] = latest_prompt_running

            render_prompt_block(
                "0",
                "AI 封面圖 Prompt",
                ensure_cover_image_prompt_file(profile_id),
                DEFAULT_COVER_IMAGE_PROMPT_TEMPLATE,
                single_step_defs["0"],
                help_text="No.0 放置封面圖 Prompt；不列入子步驟 1–10 的完成追蹤。",
                output_renderer=lambda: render_text_output_expander(
                    "No.0 輸出 Prompt",
                    cover_prompt_output_path(ep_path, info["ep"]),
                    text_label="Cover Prompt 輸出",
                    key_prefix=f"single_step0_prompt_{info['ep']}_{profile_id}",
                    expanded=False,
                    height=260,
                ),
                final_prompt_renderer=lambda template_text: render_single_prompt_preview(template_text, fixed_subtitle_path),
            )
            render_prompt_block(
                "3",
                "AI 字幕校對",
                ensure_subtitle_review_prompt_file(profile_id),
                DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE,
                single_step_defs["3"],
                help_text="No.3 會讀取 Whisper 原始字幕，產生校對後的 notebooklm_audio_fixed.srt。",
                output_renderer=lambda: render_text_output_expander(
                    "No.3 輸出預覽",
                    fixed_subtitle_path,
                    text_label="校對字幕",
                    key_prefix=f"single_step3_{info['ep']}_{profile_id}",
                    expanded=False,
                    height=260,
                ) if fixed_subtitle_path.exists() else None,
                final_prompt_renderer=lambda template_text: render_single_prompt_preview(template_text, raw_subtitle_path),
            )
            render_prompt_block(
                "5",
                "AI 依語境分鏡規劃",
                ensure_storyboard_prompt_file(profile_id),
                DEFAULT_STORYBOARD_PROMPT_TEMPLATE,
                single_step_defs["5"],
                help_text="No.5 會依校對字幕與 YouTube Title 生成 storyboard.csv。",
                output_renderer=lambda: render_dataframe_output_expander(
                    "No.5 輸出預覽",
                    read_storyboard_df(ep_path),
                    expanded=False,
                ),
                final_prompt_renderer=lambda template_text: render_single_prompt_preview(template_text, fixed_subtitle_path),
            )
            render_prompt_block(
                "10",
                "產生 YouTube Metadata",
                ensure_youtube_meta_prompt_file(profile_id),
                DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE,
                single_step_defs["10"],
                help_text="No.10 會讀取校對字幕與 YouTube Title，產生 youtube_meta.json。",
                output_renderer=lambda: render_text_output_expander(
                    "No.10 輸出預覽",
                    first_existing_path(youtube_meta_paths(ep_path)),
                    text_label="YouTube Metadata JSON",
                    key_prefix=f"single_step10_{info['ep']}_{profile_id}",
                    expanded=False,
                    height=260,
                ) if first_existing_path(youtube_meta_paths(ep_path)) and first_existing_path(youtube_meta_paths(ep_path)).exists() else None,
                final_prompt_renderer=lambda template_text: render_single_prompt_preview(template_text, fixed_subtitle_path),
            )

        render_single_prompt_panels()
        return

    step5_def = {
        "name": "產生克漏字題目檔",
        "type": "python",
        "script": "scripts/generate_cloze_quiz.py",
        "args": ["--ep", str(info["ep"])],
    }
    step7_1_def = {
        "name": "AI 封面圖 Prompt",
        "type": "python",
        "script": "scripts/generate_ai_cover.py",
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
    step20_def = {
        "name": "產生 YouTube Metadata",
        "type": "python",
        "script": "scripts/generate_youtube_meta.py",
        "args": ["--ep", str(info["ep"])],
    }
    prompt_step_ids = ["5", "7.1", "8", "12", "14", "20"]
    any_prompt_running = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in prompt_step_ids)
    prompt_refresh_interval = 2 if (prompt_auto_refresh and any_prompt_running) else None
    prompt_running_prev_key = f"prompt_running_prev_{info['ep']}_{profile_id}_{int(show_header)}"
    if any_prompt_running:
        st.session_state[prompt_running_prev_key] = True

    @st.fragment(run_every=prompt_refresh_interval)
    def render_prompt_panels():
        st.caption("只會刷新此面板，不會切換集數或跳離目前頁面。")
        st.button("只刷新 Prompt 狀態", key=f"refresh_prompt_blocks_{info['ep']}_{profile_id}_{int(show_header)}")
        latest_prompt_running = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in prompt_step_ids)
        if st.session_state.get(prompt_running_prev_key) and not latest_prompt_running:
            st.session_state[prompt_running_prev_key] = False
            st.rerun()
        st.session_state[prompt_running_prev_key] = latest_prompt_running
        prompt_provider_settings = load_profile_llm_provider_settings(profile_id) if profile_id == "vocab" else {}
        prompt_model_settings = load_profile_llm_model_settings(profile_id) if profile_id == "vocab" else {}
        prompt_extra_provider_settings = load_profile_llm_extra_provider_settings(profile_id) if profile_id == "vocab" else {}
        prompt_extra_model_settings = load_profile_llm_extra_model_settings(profile_id) if profile_id == "vocab" else {}
        step14_provider_key = f"prompt_step14_provider_{info['ep']}_{profile_id}"
        step14_model_keys = {
            provider: f"prompt_step14_{provider}_model_{info['ep']}_{profile_id}"
            for provider in ("gemini", "openai", "nvidia")
        }

        def render_step14_llm_controls():
            if profile_id != "vocab":
                return
            provider_options = list(VOCAB_TEXT_LLM_PROVIDER_LABELS.keys())
            saved_provider = str(prompt_provider_settings.get("14") or VOCAB_TEXT_LLM_STEP_DEFAULTS["14"])
            if saved_provider not in provider_options:
                saved_provider = "openai"
            provider_choice = st.selectbox(
                "Step 14 文字 LLM Provider",
                provider_options,
                index=provider_options.index(saved_provider),
                key=step14_provider_key,
                format_func=lambda value: vocab_text_provider_label(value, "14"),
            )
            model_choices = {}
            model_cols = st.columns(3)
            for col, model_provider in zip(model_cols, ("gemini", "openai", "nvidia")):
                with col:
                    model_choices[model_provider] = st.text_input(
                        f"Step 14 {model_provider} model",
                        value=vocab_text_model_for_step(prompt_model_settings, "14", model_provider),
                        key=step14_model_keys[model_provider],
                        help=f"執行時注入 {VOCAB_TEXT_LLM_MODEL_ENV_KEYS['14'][model_provider]}。",
                    ).strip()
            st.caption("目前執行 model：" + vocab_text_model_summary({"14": model_choices}, "14", provider_choice))
            if st.button("儲存 Step 14 LLM 設定", key=f"save_prompt_step14_llm_{info['ep']}_{profile_id}"):
                prompt_provider_settings["14"] = provider_choice
                prompt_model_settings["14"] = {
                    provider: model_choices.get(provider) or VOCAB_TEXT_LLM_MODEL_DEFAULTS["14"][provider]
                    for provider in ("gemini", "openai", "nvidia")
                }
                saved_path = save_profile_llm_provider_settings(profile_id, prompt_provider_settings, prompt_model_settings)
                st.success(f"Step 14 LLM 設定已儲存：{saved_path}")
                st.rerun()

        def render_step5_review_model_controls():
            if profile_id != "vocab":
                return
            st.caption("No.5 題目生成 model 與 review model 分開設定。")
            provider_options = list(VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS.keys())
            saved_review_provider = str(prompt_extra_provider_settings.get("5_review") or VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS["5_review"])
            if saved_review_provider not in provider_options:
                saved_review_provider = "auto"
            review_provider = st.selectbox(
                "Step 5 review provider",
                provider_options,
                index=provider_options.index(saved_review_provider),
                key=f"prompt_step5_review_provider_{info['ep']}_{profile_id}",
                format_func=lambda value: VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS.get(value, value),
                help=f"執行 review 時注入 {VOCAB_TEXT_LLM_EXTRA_PROVIDER_ENV_KEYS['5_review']}。",
            )
            review_models = {}
            review_cols = st.columns(2)
            for col, model_provider in zip(review_cols, ("gemini", "openai")):
                with col:
                    review_models[model_provider] = st.text_input(
                        f"Step 5 review {model_provider} model",
                        value=vocab_text_extra_model_for_call(prompt_extra_model_settings, "5_review", model_provider),
                        key=f"prompt_step5_review_{model_provider}_model_{info['ep']}_{profile_id}",
                        help=f"執行 review 時注入 {VOCAB_TEXT_LLM_EXTRA_MODEL_ENV_KEYS['5_review'][model_provider]}。",
                    ).strip()
            if st.button("儲存 Step 5 Review 模型", key=f"save_prompt_step5_review_models_{info['ep']}_{profile_id}"):
                prompt_extra_provider_settings["5_review"] = review_provider
                prompt_extra_model_settings["5_review"] = {
                    provider: review_models.get(provider) or VOCAB_TEXT_LLM_EXTRA_MODEL_DEFAULTS["5_review"][provider]
                    for provider in ("gemini", "openai")
                }
                saved_path = save_profile_llm_provider_settings(
                    profile_id,
                    prompt_provider_settings,
                    prompt_model_settings,
                    extra_model_settings=prompt_extra_model_settings,
                    extra_provider_settings=prompt_extra_provider_settings,
                )
                st.success(f"Step 5 Review 模型已儲存：{saved_path}")
                st.rerun()

        def render_step12_extra_model_controls():
            if profile_id != "vocab":
                return
            st.caption("No.12 NVIDIA fallback model 可用逗號分隔多個模型，會依序重試；不跟主字幕校對 model 共用。")
            fallback_model = st.text_input(
                "Step 12 NVIDIA fallback model",
                value=vocab_text_extra_model_for_call(prompt_extra_model_settings, "12_nvidia_fallback", "nvidia"),
                key=f"prompt_step12_nvidia_fallback_model_{info['ep']}_{profile_id}",
                help=f"執行 fallback retry 時注入 {VOCAB_TEXT_LLM_EXTRA_MODEL_ENV_KEYS['12_nvidia_fallback']['nvidia']}。",
            ).strip()
            if st.button("儲存 Step 12 Fallback 模型", key=f"save_prompt_step12_fallback_model_{info['ep']}_{profile_id}"):
                prompt_extra_model_settings["12_nvidia_fallback"] = {
                    "nvidia": fallback_model or VOCAB_TEXT_LLM_EXTRA_MODEL_DEFAULTS["12_nvidia_fallback"]["nvidia"]
                }
                saved_path = save_profile_llm_provider_settings(
                    profile_id,
                    prompt_provider_settings,
                    prompt_model_settings,
                    extra_model_settings=prompt_extra_model_settings,
                )
                st.success(f"Step 12 Fallback 模型已儲存：{saved_path}")
                st.rerun()

        prompt_path = ensure_cloze_prompt_file(profile_id)
        render_prompt_block(
            "5",
            "克漏字題目生成",
            prompt_path,
            DEFAULT_CLOZE_PROMPT_TEMPLATE,
            step5_def,
            help_text="Step 5 會讀這份 prompt，產生 cloze_questions.csv / json。",
            expanded=False,
            output_renderer=lambda: render_cloze_output_expander(
                ep_path,
                key_prefix=f"studio_step5_{info['ep']}_{profile_id}",
                expanded=False,
            ),
            extra_renderer=render_step5_review_model_controls,
            final_prompt_renderer=lambda template_text: render_step5_prompt_preview(template_text, ep_path),
        )

        cover_prompt_path = ensure_cover_image_prompt_file(profile_id)
        cover_prompt_mode_file = ensure_cover_prompt_mode_file(profile_id)
        cover_mode_labels = {
            "direct": "直接輸出模板",
            "gemini": "Gemini 改寫",
        }
        def render_cover_step_output():
            render_text_output_expander(
                "Step 7.1 輸出 Prompt",
                cover_prompt_output_path(ep_path, info["ep"]),
                text_label="Cover Prompt 輸出",
                key_prefix=f"studio_step7_1_prompt_{info['ep']}_{profile_id}",
                expanded=False,
                height=260,
            )

        def render_cover_mode_controls():
            mode_key = f"studio_cover_prompt_mode_{info['ep']}_{profile_id}"
            stored_mode = read_cover_prompt_mode(profile_id)
            if st.session_state.get(mode_key) not in cover_mode_labels:
                st.session_state[mode_key] = stored_mode
            selected_mode = st.radio(
                "Prompt 產生模式",
                options=["direct", "gemini"],
                horizontal=True,
                key=mode_key,
                format_func=lambda value: cover_mode_labels.get(value, value),
            )
            if selected_mode != stored_mode:
                save_text_file(cover_prompt_mode_file, selected_mode + "\n")
                st.caption(f"目前設定已切換為：{cover_mode_labels[selected_mode]}")
            st.caption(
                "預設為「直接輸出模板」：只做 placeholder 代入，不呼叫文字 API。切到「Gemini 改寫」時，才會把渲染後內容送到 Gemini 產生最終封面 prompt。"
            )

        render_prompt_block(
            "7.1",
            "AI 封面圖 Prompt",
            cover_prompt_path,
            DEFAULT_COVER_IMAGE_PROMPT_TEMPLATE,
            step7_1_def,
            help_text="Step 7.1 不直接呼叫產圖 API。你可以選擇只輸出模板，或先把模板渲染結果送到 Gemini 文字模型改寫後再輸出。",
            output_renderer=render_cover_step_output,
            extra_renderer=render_cover_mode_controls,
        )

        notebook_prompt_path = ensure_notebooklm_prompt_template_file(profile_id)
        render_prompt_block(
            "8",
            "產生語音摘要 Prompt",
            notebook_prompt_path,
            DEFAULT_NOTEBOOKLM_PROMPT_TEMPLATE,
            step8_def,
            help_text="Step 8 會根據這份模板產生最終語音摘要 Prompt。輸出檔內容可直接複製貼上使用。",
            output_renderer=lambda: render_text_output_expander(
                "Step 8 輸出 Prompt",
                notebooklm_prompt_output_path(ep_path, info["ep"]),
                text_label="語音摘要 Prompt",
                key_prefix=f"studio_step8_{info['ep']}_{profile_id}",
                expanded=False,
                height=320,
            ),
        )

        subtitle_prompt = ensure_subtitle_review_prompt_file(profile_id)
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
            extra_renderer=render_step12_extra_model_controls,
            final_prompt_renderer=lambda template_text: render_step12_prompt_preview(
                template_text,
                ep_path,
                ep_path / "02_subtitles" / "notebooklm_audio.srt",
            ),
        )

        storyboard_prompt = ensure_storyboard_prompt_file(profile_id)
        step14_runtime_def = step14_def
        if profile_id == "vocab":
            runtime_provider = str(
                st.session_state.get(step14_provider_key)
                or prompt_provider_settings.get("14")
                or VOCAB_TEXT_LLM_STEP_DEFAULTS["14"]
            )
            runtime_models = dict(prompt_model_settings)
            runtime_models["14"] = {
                model_provider: str(
                    st.session_state.get(step14_model_keys[model_provider])
                    or vocab_text_model_for_step(prompt_model_settings, "14", model_provider)
                ).strip()
                for model_provider in ("gemini", "openai", "nvidia")
            }
            step14_runtime_def = apply_vocab_llm_runtime_settings(
                "14",
                step14_def,
                runtime_provider,
                runtime_models,
            )
        render_prompt_block(
            "14",
            "AI 依語境分鏡規劃",
            storyboard_prompt,
            DEFAULT_STORYBOARD_PROMPT_TEMPLATE,
            step14_runtime_def,
            help_text="Step 14 會依字幕與單字表生成 storyboard.csv。",
            output_renderer=lambda: render_dataframe_output_expander(
                "Step 14 輸出預覽",
                read_storyboard_df(ep_path),
                expanded=False,
            ),
            extra_renderer=render_step14_llm_controls,
            final_prompt_renderer=lambda template_text: render_step14_prompt_preview(template_text, ep_path, subtitle_path),
        )

        meta_prompt = ensure_youtube_meta_prompt_file(profile_id)
        render_prompt_block(
            "20",
            "產生 YouTube Metadata",
            meta_prompt,
            DEFAULT_YOUTUBE_META_PROMPT_TEMPLATE,
            step20_def,
            help_text="Step 20 會讀取校對字幕與集數範圍，產生 youtube_meta.json。",
            output_renderer=lambda: render_text_output_expander(
                "Step 20 輸出預覽",
                first_existing_path(youtube_meta_paths(ep_path)),
                text_label="YouTube Metadata JSON",
                key_prefix=f"studio_step20_{info['ep']}_{profile_id}",
                expanded=False,
                height=260,
            ) if first_existing_path(youtube_meta_paths(ep_path)) and first_existing_path(youtube_meta_paths(ep_path)).exists() else None,
            final_prompt_renderer=lambda template_text: render_step19_prompt_preview(template_text, ep_path, subtitle_path),
        )

    render_prompt_panels()

def render_vocab_audio_upload_control(ep_path: Path, ns: str, is_done: bool):
    audio_path = audio_merged_path(ep_path)
    upload_key = f"pm_vocab_audio_upload_{ep_path.name}_{ns}"
    uploaded_audio = st.file_uploader(
        "上傳語音",
        type=["m4a", "mp3", "wav", "aac", "ogg"],
        key=upload_key,
        help="後續步驟固定讀取 01_audio/notebooklm_audio.m4a；上傳後會自動存成此檔名。",
    )
    if audio_path.exists():
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        st.caption(f"目前檔案：{audio_path.name} ({size_mb:.2f} MB)")
        st.audio(str(audio_path))
    elif is_done:
        st.caption("狀態顯示已完成，但找不到固定音訊檔。請重新上傳。")
    else:
        st.caption("尚未上傳固定音訊檔。")

    if uploaded_audio is not None:
        st.caption(f"將儲存為：{audio_path.relative_to(ep_path)}")
        if st.button("儲存語音檔", key=f"pm_vocab_audio_save_{ep_path.name}_{ns}"):
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(uploaded_audio.getbuffer())

            intro_marker = ep_path / "00_logs" / "step10_5_intro_merged.txt"
            marker_was_present = intro_marker.exists()
            if marker_was_present:
                intro_marker.unlink()

            log_path = log_file_for_stage(ep_path, f"sub{ns}")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\n==== UPLOAD sub{ns} | 上傳語音音訊 | "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                lf.write(f"source_name={uploaded_audio.name}\n")
                lf.write(f"saved_as={audio_path}\n")
                lf.write(f"bytes={audio_path.stat().st_size}\n")
                lf.write(f"intro_marker_cleared={marker_was_present}\n")
            st.success("語音已上傳並改名為 notebooklm_audio.m4a。No.10.5 片頭合併標記已重置。")
            st.rerun()


def vocab_outro_dir(ep_path: Path) -> Path:
    return ep_path / "05_output" / "outro"


def vocab_outro_config_path(ep_path: Path) -> Path:
    return vocab_outro_dir(ep_path) / "outro_config.json"


def load_vocab_outro_config(ep_path: Path) -> Dict:
    config = read_json_file(vocab_outro_config_path(ep_path)) or {}
    return config if isinstance(config, dict) else {}


def find_vocab_outro_media(ep_path: Path) -> Path | None:
    outro_dir = vocab_outro_dir(ep_path)
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".m4v"):
        candidate = outro_dir / f"outro_media{ext}"
        if candidate.exists():
            return candidate
    return None


def find_vocab_outro_audio(ep_path: Path) -> Path | None:
    outro_dir = vocab_outro_dir(ep_path)
    for ext in (".m4a", ".mp3", ".wav", ".aac", ".ogg"):
        candidate = outro_dir / f"outro_audio{ext}"
        if candidate.exists():
            return candidate
    return None


def save_uploaded_file(uploaded_file, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded_file.getbuffer())


def render_vocab_outro_control(ep_path: Path, ns: str, is_done: bool):
    config = load_vocab_outro_config(ep_path)
    media_path = find_vocab_outro_media(ep_path)
    audio_path = find_vocab_outro_audio(ep_path)
    current_duration = float(config.get("duration_seconds") or 5.0)
    if current_duration <= 0:
        current_duration = 5.0

    st.caption("可選步驟；未上傳片尾媒體時，No.18 產生合成清單會略過片尾。")
    duration_seconds = st.number_input(
        "片尾長度（秒）",
        min_value=0.5,
        max_value=120.0,
        value=current_duration,
        step=0.5,
        key=f"pm_vocab_outro_duration_{ep_path.name}_{ns}",
    )
    uploaded_media = st.file_uploader(
        "上傳片尾圖片或影像",
        type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm", "m4v"],
        key=f"pm_vocab_outro_media_{ep_path.name}_{ns}",
    )
    uploaded_audio = st.file_uploader(
        "上傳片尾音效",
        type=["m4a", "mp3", "wav", "aac", "ogg"],
        key=f"pm_vocab_outro_audio_{ep_path.name}_{ns}",
    )

    if media_path:
        size_mb = media_path.stat().st_size / (1024 * 1024)
        st.caption(f"目前片尾媒體：{media_path.name} ({size_mb:.2f} MB)")
        if media_path.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}:
            st.video(str(media_path))
        else:
            st.image(str(media_path))
    elif is_done:
        st.caption("狀態顯示已完成，但找不到片尾媒體。請重新上傳。")
    else:
        st.caption("目前未設定片尾媒體。")

    if audio_path:
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        st.caption(f"目前片尾音效：{audio_path.name} ({size_mb:.2f} MB)")
        st.audio(str(audio_path))
    else:
        st.caption("片尾音效未設定；合成時會使用靜音片尾。")

    if st.button("儲存片尾設定", key=f"pm_vocab_outro_save_{ep_path.name}_{ns}"):
        outro_dir = vocab_outro_dir(ep_path)
        outro_dir.mkdir(parents=True, exist_ok=True)
        saved_media_path = media_path
        saved_audio_path = audio_path

        if uploaded_media is not None:
            media_suffix = Path(uploaded_media.name).suffix.lower()
            for old_media in outro_dir.glob("outro_media.*"):
                old_media.unlink(missing_ok=True)
            saved_media_path = outro_dir / f"outro_media{media_suffix}"
            save_uploaded_file(uploaded_media, saved_media_path)

        if uploaded_audio is not None:
            audio_suffix = Path(uploaded_audio.name).suffix.lower()
            for old_audio in outro_dir.glob("outro_audio.*"):
                old_audio.unlink(missing_ok=True)
            saved_audio_path = outro_dir / f"outro_audio{audio_suffix}"
            save_uploaded_file(uploaded_audio, saved_audio_path)

        config = {
            "duration_seconds": float(duration_seconds),
            "media_path": str(saved_media_path.relative_to(ep_path)).replace("\\", "/") if saved_media_path else "",
            "audio_path": str(saved_audio_path.relative_to(ep_path)).replace("\\", "/") if saved_audio_path else "",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        vocab_outro_config_path(ep_path).write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_path = log_file_for_stage(ep_path, f"sub{ns}")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"\n==== SAVE sub{ns} | 片尾製作 | "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            lf.write(json.dumps(config, ensure_ascii=False) + "\n")
        st.success("片尾設定已儲存。")
        st.rerun()


def render_prerender_workspace(ep_path: Path):
    st.subheader("整集預覽")
    if st.session_state.get("profile_id") == "story":
        audio_path = story_merged_audio_path(ep_path)
        subtitle_path = story_subtitles_srt_path(ep_path)
        storyboard_path = story_storyboard_csv_path(ep_path)
    else:
        audio_path = audio_merged_path(ep_path)
        subtitle_path = subtitles_fixed_path(ep_path)
        storyboard_path = storyboard_csv_path(ep_path)
    ready_audio = audio_path.exists()
    ready_subtitle = subtitle_path.exists()
    ready_storyboard = storyboard_path.exists()
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("語音", "就緒" if ready_audio else "缺少")
    pc2.metric("字幕", "就緒" if ready_subtitle else "缺少")
    pc3.metric("分鏡", "就緒" if ready_storyboard else "缺少")
    if not (ready_audio and ready_subtitle and ready_storyboard):
        st.warning("整集預覽需要同時具備語音、字幕與分鏡表。")
        return

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
    pr3.metric("預覽時長", seconds_to_label(preview_duration_sec))
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

    render_subtitle_compare_panel(
        ep_path,
        key_prefix=f"prerender_subtitle_compare_{parse_episode_info(ep_path).get('ep')}_{st.session_state.get('profile_id', '')}",
        expanded=False,
    )

    with st.expander("完整分鏡表", expanded=False):
        st.dataframe(preview_storyboard_df, use_container_width=True, hide_index=True)

def render_storyboard_workspace(info: Dict, ep_path: Path, profile_id: str):
    is_story_mode = profile_id == "story"
    no_flashcard_mode = profile_id in ("story", "single_video")
    if profile_id == "story":
        st.subheader("No.5.1 修改分鏡")
    elif profile_id == "single_video":
        st.subheader("修改分鏡 / 替換圖片")
    else:
        st.subheader("No.13 修改分鏡 / 替換單字卡")
    storyboard_path = storyboard_path_for(ep_path)
    storyboard_sig = storyboard_file_signature(ep_path)
    storyboard_key_suffix = f"{info['ep']}_{profile_id}_{storyboard_sig}"
    storyboard_df = read_storyboard_df(ep_path)
    flashcards = {} if no_flashcard_mode else flashcard_map_for(ep_path)
    if storyboard_df.empty:
        st.warning(f"找不到分鏡表：{storyboard_path}")
        return

    storyboard_df = normalize_storyboard_df(ep_path, storyboard_df)
    if profile_id in ("story", "single_video"):
        st.caption(f"分鏡檔：{storyboard_path}")
    else:
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
            st.caption(f"分鏡檔：{storyboard_path}")
            st.caption(f"已載入版本：{storyboard_sig}")
            if st.button("重新載入分鏡結果", key=f"reload_storyboard_{storyboard_key_suffix}"):
                st.rerun()

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
                merge_cols = ["scene_id", "start_time", "end_time", "source_type", "reason"]
                if not no_flashcard_mode:
                    merge_cols.insert(4, "flashcard_word")
                st.dataframe(merge_preview_df[merge_cols], use_container_width=True, hide_index=True)
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
            split_editor_df = split_draft_df.drop(columns=["flashcard_word"], errors="ignore") if no_flashcard_mode else split_draft_df
            split_editor_key = f"storyboard_split_editor_{info['ep']}_{profile_id}_{split_scene_id}_{split_count}"
            st.caption("預設先等分時間。你可以直接改每段的開始/結束時間與 reason，再套用。")
            edited_split_df = st.data_editor(
                split_editor_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "start_time": st.column_config.NumberColumn("開始", format="%.3f"),
                    "end_time": st.column_config.NumberColumn("結束", format="%.3f"),
                    "source_type": st.column_config.SelectboxColumn("素材來源", options=["AI"] if no_flashcard_mode else ["AI", "FLASHCARD"]),
                    "flashcard_word": st.column_config.TextColumn("單字卡", disabled=no_flashcard_mode),
                    "custom_image_path": st.column_config.TextColumn("圖片路徑", width="large"),
                    "image_prompt": st.column_config.TextColumn("圖片 Prompt", width="large"),
                    "animation_prompt": st.column_config.TextColumn("動畫 Prompt", width="large"),
                    "reason": st.column_config.TextColumn("說明", width="large"),
                    "subtitle_reference": st.column_config.TextColumn("字幕參考", width="large"),
                },
                key=split_editor_key,
            )
            if no_flashcard_mode:
                edited_split_df["source_type"] = "AI"
                edited_split_df["flashcard_word"] = ""
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
    st.markdown("**批次動畫**")
    ai_scene_rows = [
        row.copy()
        for _, row in storyboard_df.iterrows()
        if str(row.get("source_type", "")).strip().upper() == "AI"
    ]
    ai_scene_total = len(ai_scene_rows)
    ai_scene_with_image = sum(1 for row in ai_scene_rows if storyboard_scene_image_path(ep_path, row))
    ai_scene_using_animation = sum(1 for row in ai_scene_rows if storyboard_scene_animation_path(ep_path, row))
    ai_scene_batch_ready = sum(
        1
        for row in ai_scene_rows
        if storyboard_scene_image_path(ep_path, row) and not storyboard_scene_animation_path(ep_path, row)
    )
    batch1, batch2, batch3, batch4, batch5 = st.columns([1, 1, 1, 1, 1.8])
    batch1.metric("AI Scene", ai_scene_total)
    batch2.metric("有圖片", ai_scene_with_image)
    batch3.metric("已使用動畫", ai_scene_using_animation)
    batch4.metric("可批次處理", ai_scene_batch_ready)
    batch_animation_task_name = "sceneanimall"
    batch_animation_task = publisher_task_state(info, batch_animation_task_name)
    with batch5:
        batch_generate_animations = st.button(
            "將所有 AI 圖片轉成動畫",
            key=f"storyboard_batch_generate_animation_{info['ep']}_{profile_id}",
            disabled=(ai_scene_batch_ready == 0) or bool(batch_animation_task.get("running")),
            use_container_width=True,
        )
        if batch_animation_task.get("running"):
            st.caption(f"批次任務執行中，PID {batch_animation_task.get('pid')}")
    st.caption(
        "只處理 `source_type=AI` 的 Scene。批次任務會先替每個 Scene 產生並儲存 `animation_prompt`，"
        "再接著生成動畫；已經使用動畫的 Scene 會自動略過。若缺圖、Prompt 生成失敗或動畫 API 失敗，會寫入 batch log 後繼續下一個 Scene。"
    )
    media_image_provider_settings = load_profile_image_provider_settings(profile_id) if profile_id == "vocab" else dict(VOCAB_IMAGE_PROVIDER_DEFAULTS)
    media_image_model_settings = load_profile_image_model_settings(profile_id) if profile_id == "vocab" else dict(VOCAB_IMAGE_MODEL_DEFAULTS)
    media_image_model_step_settings = load_profile_image_model_step_settings(profile_id) if profile_id == "vocab" else default_vocab_image_model_step_settings()
    media_animation_settings = load_profile_animation_provider_settings(profile_id) if profile_id == "vocab" else {
        "provider": VOCAB_ANIMATION_PROVIDER_DEFAULT,
        "models": dict(VOCAB_ANIMATION_MODEL_DEFAULTS),
    }
    with st.expander("AI 產圖 / 動畫 Provider", expanded=False):
        provider_cols = st.columns(2)
        with provider_cols[0]:
            image_provider_options = list(VOCAB_IMAGE_PROVIDER_LABELS.keys())
            selected_image_provider = st.selectbox(
                "圖片 Provider",
                image_provider_options,
                index=image_provider_options.index(str(media_image_provider_settings.get("16") or "auto")),
                key=f"storyboard_image_provider_{info['ep']}_{profile_id}",
                format_func=lambda value: VOCAB_IMAGE_PROVIDER_LABELS.get(value, value),
            )
            if selected_image_provider == "nvidia" and not os.environ.get("CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL") and not os.environ.get("NVIDIA_IMAGE_BASE_URL"):
                st.warning("NVIDIA 圖片 Provider 需要先在 .env 設定 CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL 或 NVIDIA_IMAGE_BASE_URL。")
            image_model_values = {}
            for provider_key, options in VOCAB_IMAGE_MODEL_OPTIONS.items():
                current_model = vocab_image_model_for_step(media_image_model_step_settings, "16", provider_key)
                if current_model not in options:
                    options = [current_model] + options
                image_model_values[provider_key] = st.selectbox(
                    f"{VOCAB_IMAGE_PROVIDER_LABELS[provider_key]} 圖片模型",
                    options,
                    index=options.index(current_model),
                    key=f"storyboard_image_model_{provider_key}_{info['ep']}_{profile_id}",
                )
        with provider_cols[1]:
            animation_provider_options = list(VOCAB_ANIMATION_PROVIDER_LABELS.keys())
            selected_animation_provider = st.selectbox(
                "動畫 Provider",
                animation_provider_options,
                index=animation_provider_options.index(str(media_animation_settings.get("provider") or VOCAB_ANIMATION_PROVIDER_DEFAULT)),
                key=f"storyboard_animation_provider_{info['ep']}_{profile_id}",
                format_func=lambda value: VOCAB_ANIMATION_PROVIDER_LABELS.get(value, value),
            )
            animation_model_settings = dict(media_animation_settings.get("models") or {})
            animation_model_values = {}
            for provider_key, options in VOCAB_ANIMATION_MODEL_OPTIONS.items():
                current_model = str(animation_model_settings.get(provider_key) or VOCAB_ANIMATION_MODEL_DEFAULTS[provider_key])
                if current_model not in options:
                    options = [current_model] + options
                animation_model_values[provider_key] = st.selectbox(
                    f"{VOCAB_ANIMATION_PROVIDER_LABELS[provider_key]} 動畫模型",
                    options,
                    index=options.index(current_model),
                    key=f"storyboard_animation_model_{provider_key}_{info['ep']}_{profile_id}",
                )
            st.caption("NVIDIA 目前只接了圖片 NIM；官方 Visual GenAI NIM 文件沒有同等的通用影片生成 API，因此動畫 Provider 暫不提供 NVIDIA。")
        if st.button("儲存 AI 產圖 / 動畫 Provider", key=f"save_storyboard_media_provider_{info['ep']}_{profile_id}"):
            if profile_id == "vocab":
                media_image_provider_settings["16"] = selected_image_provider
                media_animation_settings = {"provider": selected_animation_provider, "models": animation_model_values}
                saved_path = save_profile_llm_provider_settings(
                    profile_id,
                    load_profile_llm_provider_settings(profile_id),
                    load_profile_llm_model_settings(profile_id),
                    media_image_provider_settings,
                    image_model_values,
                    media_animation_settings,
                    image_model_step_settings={
                        **media_image_model_step_settings,
                        "16": image_model_values,
                    },
                )
                st.success(f"已儲存 Provider 設定：{saved_path}")
                st.rerun()
            else:
                st.info("此 Provider 設定目前只套用於 vocab profile。")
    current_media_env = build_storyboard_media_env(image_model_values, {"provider": selected_animation_provider, "models": animation_model_values})
    if batch_animation_task.get("log"):
        st.caption(f"批次 log：{batch_animation_task.get('log')}")
    if batch_generate_animations:
        res = start_publisher_task(
            info,
            batch_animation_task_name,
            {
                "name": "將所有 AI 圖片轉成動畫",
                "type": "python",
                "script": "scripts/generate_all_scene_animations.py",
                "args": ["--ep", str(info["ep"]), "--provider", selected_animation_provider],
                "env": current_media_env,
            },
        )
        if res.get("ok"):
            st.success(f"批次動畫任務已啟動。log: {res.get('log')}")
        else:
            st.error(f"批次動畫任務啟動失敗：{res.get('message')}")
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
    scene_image_and_animation_task_name = f"sceneimganim_{selected_scene_id}"
    batch_animation_task_name = "sceneanimall"
    selected_scene_task = publisher_task_state(info, scene_image_task_name)
    selected_anim_prompt_task = publisher_task_state(info, scene_anim_prompt_task_name)
    selected_anim_video_task = publisher_task_state(info, scene_anim_video_task_name)
    selected_img_anim_task = publisher_task_state(info, scene_image_and_animation_task_name)
    batch_animation_task = publisher_task_state(info, batch_animation_task_name)
    current_image_path = str(storyboard_scene_image_path(ep_path, selected_row) or "").strip()
    current_animation_path = str(storyboard_scene_animation_path(ep_path, selected_row) or "").strip()
    image_candidates = scene_image_candidate_paths(ep_path, selected_row)
    animation_candidates = scene_animation_candidate_paths(ep_path, selected_row)
    other_image_candidates = other_episode_image_candidate_paths(ep_path)
    other_animation_candidates = other_episode_animation_candidate_paths(ep_path)
    storyboard_auto_refresh = st.toggle(
        "自動刷新 Scene 產圖 / 動畫 Log",
        value=True,
        key=f"storyboard_auto_refresh_{storyboard_key_suffix}",
        help="Scene 單獨產圖、動畫 Prompt、動畫生成執行時，每 2 秒自動更新狀態與 Log。",
    )

    source_key = f"storyboard_source_{storyboard_key_suffix}_{selected_scene_id}"
    flashcard_key = f"storyboard_flashcard_{storyboard_key_suffix}_{selected_scene_id}"
    custom_image_key = f"storyboard_custom_image_{storyboard_key_suffix}_{selected_scene_id}"
    prompt_key = f"storyboard_prompt_{storyboard_key_suffix}_{selected_scene_id}"
    animation_prompt_key = f"storyboard_animation_prompt_{storyboard_key_suffix}_{selected_scene_id}"
    animation_prompt_loaded_key = f"{animation_prompt_key}__loaded"
    asset_mode_key = f"storyboard_asset_mode_{storyboard_key_suffix}_{selected_scene_id}"
    image_choice_key = f"storyboard_image_choice_{storyboard_key_suffix}_{selected_scene_id}"
    animation_choice_key = f"storyboard_animation_choice_{storyboard_key_suffix}_{selected_scene_id}"
    asset_tag_filter_key = f"storyboard_asset_tag_filter_{storyboard_key_suffix}_{selected_scene_id}"
    asset_tag_target_key = f"storyboard_asset_tag_target_{storyboard_key_suffix}_{selected_scene_id}"
    asset_tag_input_key = f"storyboard_asset_tag_input_{storyboard_key_suffix}_{selected_scene_id}"
    asset_tag_loaded_key = f"{asset_tag_input_key}__loaded"
    reason_key = f"storyboard_reason_{storyboard_key_suffix}_{selected_scene_id}"
    if source_key not in st.session_state:
        st.session_state[source_key] = "AI" if no_flashcard_mode else (str(selected_row.get("source_type", "")).upper() or "AI")
    if flashcard_key not in st.session_state:
        st.session_state[flashcard_key] = current_word if current_word in flashcards else ""
    if no_flashcard_mode:
        st.session_state[source_key] = "AI"
        st.session_state[flashcard_key] = ""
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

    image_base_path = ai_scene_image_path(ep_path, selected_row.get("start_time", ""))
    animation_base_path = ai_scene_animation_path(ep_path, selected_row)
    all_asset_candidates = _dedupe_existing_paths(
        image_candidates + other_image_candidates + animation_candidates + other_animation_candidates
    )
    asset_tag_filter_options = collect_asset_tag_options(all_asset_candidates)
    if st.session_state.get(asset_tag_filter_key) not in asset_tag_filter_options:
        st.session_state[asset_tag_filter_key] = ASSET_TAG_FILTER_ALL
    selected_asset_tag_filter = st.session_state.get(asset_tag_filter_key, ASSET_TAG_FILTER_ALL)

    filtered_image_candidates = filter_asset_paths_by_tag(image_candidates, selected_asset_tag_filter)
    filtered_other_image_candidates = filter_asset_paths_by_tag(other_image_candidates, selected_asset_tag_filter)
    filtered_animation_candidates = filter_asset_paths_by_tag(animation_candidates, selected_asset_tag_filter)
    filtered_other_animation_candidates = filter_asset_paths_by_tag(other_animation_candidates, selected_asset_tag_filter)

    tag_target_entries: list[tuple[str, Path, Path | None]] = []
    tag_target_entries.extend([("圖片", path, image_base_path) for path in image_candidates])
    tag_target_entries.extend([("圖片", path, None) for path in other_image_candidates])
    tag_target_entries.extend([("動畫", path, animation_base_path) for path in animation_candidates])
    tag_target_entries.extend([("動畫", path, None) for path in other_animation_candidates])
    tag_target_options = [str(path) for _, path, _ in tag_target_entries]
    tag_target_labels = {
        str(path): f"[{kind}] {asset_label_with_tag(ep_path, path, kind, base_path=base_path)}"
        for kind, path, base_path in tag_target_entries
    }
    default_tag_target = current_animation_path or current_image_path or (tag_target_options[0] if tag_target_options else "")
    if st.session_state.get(asset_tag_target_key) not in tag_target_options:
        st.session_state[asset_tag_target_key] = default_tag_target
    selected_tag_target_path = st.session_state.get(asset_tag_target_key, "")
    if st.session_state.get(asset_tag_loaded_key) != selected_tag_target_path:
        st.session_state[asset_tag_input_key] = asset_tag_for_path(selected_tag_target_path)
        st.session_state[asset_tag_loaded_key] = selected_tag_target_path

    image_choice_map = build_asset_choice_map(
        ep_path,
        filtered_image_candidates + filtered_other_image_candidates,
        "圖片",
        base_path=image_base_path,
    )
    image_choice_labels = ["(使用自訂圖片路徑 / 預設原始圖)"] + list(image_choice_map.keys())
    current_image_choice_label = next(
        (label for label, path in image_choice_map.items() if path == current_image_path),
        image_choice_labels[0],
    )
    if st.session_state.get(image_choice_key) not in image_choice_labels:
        st.session_state[image_choice_key] = current_image_choice_label

    animation_choice_map = build_asset_choice_map(
        ep_path,
        filtered_animation_candidates + filtered_other_animation_candidates,
        "動畫",
        base_path=animation_base_path,
    )
    animation_choice_labels = ["(不使用動畫)"] + list(animation_choice_map.keys())
    current_animation_choice_label = next(
        (label for label, path in animation_choice_map.items() if path == current_animation_path),
        animation_choice_labels[0],
    )
    if st.session_state.get(animation_choice_key) not in animation_choice_labels:
        st.session_state[animation_choice_key] = current_animation_choice_label
    available_asset_modes = ["圖片"] + (["動畫"] if (animation_candidates or other_animation_candidates or current_animation_path) else [])
    current_asset_mode = "動畫" if current_animation_path else "圖片"
    if st.session_state.get(asset_mode_key) not in available_asset_modes:
        st.session_state[asset_mode_key] = current_asset_mode

    def replace_scene_with_other_image(candidate_path: Path) -> None:
        chosen_path = str(candidate_path)
        chosen_label = next(
            (label for label, path in image_choice_map.items() if path == chosen_path),
            image_choice_labels[0],
        )
        scene_updates = {
            "source_type": "AI",
            "flashcard_word": "",
            "custom_image_path": chosen_path,
            "image_prompt": st.session_state.get(prompt_key, str(selected_row.get("image_prompt", ""))),
            "animation_prompt": st.session_state.get(animation_prompt_key, str(selected_row.get("animation_prompt", ""))),
            "animation_video_path": "",
            "reason": st.session_state.get(reason_key, str(selected_row.get("reason", ""))),
        }
        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
        if saved_storyboard is None or idx is None:
            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
            return
        st.session_state[source_key] = "AI"
        st.session_state[flashcard_key] = ""
        st.session_state[asset_mode_key] = "圖片"
        st.session_state[custom_image_key] = chosen_path
        st.session_state[image_choice_key] = chosen_label
        st.session_state[animation_choice_key] = animation_choice_labels[0]
        st.success(f"已替換 Scene {selected_scene_id} 圖片：{candidate_path.name}")
        st.rerun()

    def replace_scene_with_other_animation(candidate_path: Path) -> None:
        chosen_path = str(candidate_path)
        chosen_label = next(
            (label for label, path in animation_choice_map.items() if path == chosen_path),
            animation_choice_labels[0],
        )
        preserved_image_path = current_image_path or str(selected_row.get("custom_image_path", "")).strip()
        scene_updates = {
            "source_type": "AI",
            "flashcard_word": "",
            "custom_image_path": preserved_image_path,
            "image_prompt": st.session_state.get(prompt_key, str(selected_row.get("image_prompt", ""))),
            "animation_prompt": st.session_state.get(animation_prompt_key, str(selected_row.get("animation_prompt", ""))),
            "animation_video_path": chosen_path,
            "reason": st.session_state.get(reason_key, str(selected_row.get("reason", ""))),
        }
        saved_storyboard, latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
        if saved_storyboard is None or idx is None:
            st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
            return
        st.session_state[source_key] = "AI"
        st.session_state[flashcard_key] = ""
        st.session_state[asset_mode_key] = "動畫"
        st.session_state[custom_image_key] = preserved_image_path
        st.session_state[animation_choice_key] = chosen_label
        st.success(f"已替換 Scene {selected_scene_id} 動畫：{candidate_path.name}")
        st.rerun()

    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Scene", str(selected_scene_id))
    meta2.metric("開始", seconds_to_label(selected_row.get("start_time", 0)))
    meta3.metric("結束", seconds_to_label(selected_row.get("end_time", 0)))
    meta4.metric("目前素材", "動畫" if current_animation_path else "圖片")

    st.subheader("目前素材")
    if current_animation_path and Path(current_animation_path).exists():
        st.caption(f"目前使用動畫：{current_animation_path}")
        st.caption(f"目前 Tag：{asset_tag_badge(Path(current_animation_path))}")
        st.video(current_animation_path)
    elif current_image_path and Path(current_image_path).exists():
        st.caption(f"目前 Tag：{asset_tag_badge(Path(current_image_path))}")
        render_safe_image_preview(
            Path(current_image_path),
            empty_message="目前沒有已選定的素材。",
            broken_hint="目前素材圖片無法讀取",
            caption_label="目前使用圖片",
            max_width=720,
        )
    else:
        st.caption("目前沒有已選定的素材。")

    with st.expander("上傳圖片到此 Scene", expanded=False):
        uploaded_scene_image = st.file_uploader(
            "上傳圖片",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"storyboard_upload_image_{storyboard_key_suffix}_{selected_scene_id}",
        )
        if uploaded_scene_image is not None:
            suffix = Path(uploaded_scene_image.name).suffix.lower() or ".png"
            if suffix == ".jpeg":
                suffix = ".jpg"
            target_image_path = ep_path / "04_images" / "uploads" / f"scene_{str(selected_scene_id).zfill(3)}{suffix}"
            st.caption(f"將儲存為：{target_image_path}")
            if st.button("儲存並套用上傳圖片", key=f"storyboard_apply_uploaded_image_{storyboard_key_suffix}_{selected_scene_id}"):
                save_uploaded_file(uploaded_scene_image, target_image_path)
                scene_updates = {
                    "source_type": "AI",
                    "flashcard_word": "",
                    "custom_image_path": str(target_image_path),
                    "image_prompt": st.session_state.get(prompt_key, str(selected_row.get("image_prompt", ""))),
                    "animation_prompt": st.session_state.get(animation_prompt_key, str(selected_row.get("animation_prompt", ""))),
                    "animation_video_path": "",
                    "reason": st.session_state.get(reason_key, str(selected_row.get("reason", ""))),
                }
                saved_storyboard, _latest_df, idx = update_storyboard_scene(ep_path, selected_scene_id, scene_updates)
                if saved_storyboard is None or idx is None:
                    st.error("找不到對應 scene，可能已被修改。請重新整理後再試。")
                else:
                    st.session_state[source_key] = "AI"
                    st.session_state[flashcard_key] = ""
                    st.session_state[asset_mode_key] = "圖片"
                    st.session_state[custom_image_key] = str(target_image_path)
                    st.success(f"已上傳並套用 Scene {selected_scene_id} 圖片：{saved_storyboard}")
                    st.rerun()

    st.subheader("素材 Tag")
    st.caption("先為素材設定單一 Tag，再用同一個 Tag 篩選下方圖片與動畫；選好素材後仍需按「儲存此 Scene」。")
    tag_top1, tag_top2 = st.columns([1, 1.6])
    with tag_top1:
        selected_asset_tag_filter = st.selectbox(
            "Tag 篩選（單選）",
            asset_tag_filter_options,
            key=asset_tag_filter_key,
            format_func=lambda value: {
                ASSET_TAG_FILTER_ALL: "全部",
                ASSET_TAG_FILTER_UNTAGGED: "未設定 Tag",
            }.get(value, value),
        )
    with tag_top2:
        if tag_target_options:
            selected_tag_target_value = st.selectbox(
                "設定素材 Tag",
                tag_target_options,
                key=asset_tag_target_key,
                format_func=lambda value: tag_target_labels.get(value, value),
            )
        else:
            selected_tag_target_value = ""
            st.text_input("設定素材 Tag", value="目前沒有可標記素材", disabled=True)
    if selected_tag_target_value and st.session_state.get(asset_tag_loaded_key) != selected_tag_target_value:
        st.session_state[asset_tag_input_key] = asset_tag_for_path(selected_tag_target_value)
        st.session_state[asset_tag_loaded_key] = selected_tag_target_value
    tag_bottom1, tag_bottom2 = st.columns([1.6, 0.6])
    with tag_bottom1:
        st.text_input(
            "素材 Tag",
            key=asset_tag_input_key,
            disabled=not bool(tag_target_options),
            help="留空後按儲存，會清除此素材的 Tag。",
        )
    with tag_bottom2:
        save_asset_tag = st.button(
            "儲存素材 Tag",
            key=f"storyboard_save_asset_tag_{info['ep']}_{selected_scene_id}",
            disabled=not bool(tag_target_options),
            use_container_width=True,
        )
    if save_asset_tag:
        saved_tag_path = set_asset_tag_for_path(selected_tag_target_value, st.session_state.get(asset_tag_input_key, ""))
        if saved_tag_path is None:
            st.error("找不到此素材所屬集數，無法儲存 Tag。")
        else:
            st.success(f"已更新素材 Tag：{saved_tag_path}")
            st.rerun()

    with st.expander(f"候選圖片（{len(filtered_image_candidates)} / {len(image_candidates)}）", expanded=False):
        if filtered_image_candidates:
            image_rows = [filtered_image_candidates[i:i + 3] for i in range(0, len(filtered_image_candidates), 3)]
            for row_group in image_rows:
                cols = st.columns(3)
                for col_idx, candidate_path in enumerate(row_group):
                    with cols[col_idx]:
                        render_thumbnail_card(
                            candidate_path,
                            caption_text=f"{candidate_path.name} | Tag: {asset_tag_badge(candidate_path)}",
                            current=(str(candidate_path) == current_image_path),
                            max_width=360,
                        )
        else:
            st.caption("目前 Tag 篩選下沒有候選圖片。")
    with st.expander(f"其它集數圖片（{len(filtered_other_image_candidates)} / {len(other_image_candidates)}）", expanded=False):
        if filtered_other_image_candidates:
            indexed_images = list(enumerate(filtered_other_image_candidates))
            image_rows = [indexed_images[i:i + 3] for i in range(0, len(indexed_images), 3)]
            for row_group in image_rows:
                cols = st.columns(3)
                for col_idx, item in enumerate(row_group):
                    asset_idx, candidate_path = item
                    with cols[col_idx]:
                        asset_ep_path = episode_path_for_asset(candidate_path)
                        asset_info = parse_episode_info(asset_ep_path) if asset_ep_path else {}
                        caption_text = (
                            f"Ep{int(asset_info.get('ep')):02d} | {candidate_path.name}"
                            if asset_info.get("ep") is not None
                            else candidate_path.name
                        )
                        with st.container(border=True):
                            st.caption("目前使用中" if str(candidate_path) == current_image_path else f"{caption_text} | Tag: {asset_tag_badge(candidate_path)}")
                            render_safe_image_preview(
                                candidate_path,
                                empty_message="沒有可顯示的圖片。",
                                broken_hint="圖片無法讀取",
                                caption_label="預覽",
                                max_width=360,
                            )
                            if st.button(
                                "替換此分鏡圖片",
                                key=f"storyboard_replace_other_image_{info['ep']}_{selected_scene_id}_{asset_idx}",
                                use_container_width=True,
                            ):
                                replace_scene_with_other_image(candidate_path)
        else:
            st.caption("目前 Tag 篩選下沒有其它集數圖片。")
    with st.expander(f"候選動畫（{len(filtered_animation_candidates)} / {len(animation_candidates)}）", expanded=False):
        if filtered_animation_candidates:
            animation_rows = [filtered_animation_candidates[i:i + 2] for i in range(0, len(filtered_animation_candidates), 2)]
            for row_group in animation_rows:
                cols = st.columns(2)
                for col_idx, candidate_path in enumerate(row_group):
                    with cols[col_idx]:
                        with st.container(border=True):
                            st.caption(
                                "目前使用中"
                                if str(candidate_path) == current_animation_path
                                else f"{candidate_path.name} | Tag: {asset_tag_badge(candidate_path)}"
                            )
                            st.video(str(candidate_path))
        else:
            st.caption("目前 Tag 篩選下沒有候選動畫。")
    with st.expander(f"其它集數動畫（{len(filtered_other_animation_candidates)} / {len(other_animation_candidates)}）", expanded=False):
        if filtered_other_animation_candidates:
            indexed_animations = list(enumerate(filtered_other_animation_candidates))
            episode_animation_groups: dict[str, list[tuple[int, Path]]] = {}
            episode_group_labels: dict[str, str] = {}
            animation_caption_map: dict[int, str] = {}
            for asset_idx, candidate_path in indexed_animations:
                asset_ep_path = episode_path_for_asset(candidate_path)
                asset_info = parse_episode_info(asset_ep_path) if asset_ep_path else {}
                if asset_info.get("ep") is not None:
                    episode_key = f"ep_{int(asset_info.get('ep')):02d}"
                    episode_label = f"Ep{int(asset_info.get('ep')):02d}"
                else:
                    episode_key = str(asset_ep_path or candidate_path.parent)
                    episode_label = asset_ep_path.name if asset_ep_path else "未知集數"
                episode_animation_groups.setdefault(episode_key, []).append((asset_idx, candidate_path))
                episode_group_labels[episode_key] = episode_label
                caption_text = f"{episode_label} | {candidate_path.name}"
                prefix = "目前使用中 | " if str(candidate_path) == current_animation_path else ""
                animation_caption_map[asset_idx] = f"{prefix}{caption_text} | Tag: {asset_tag_badge(candidate_path)}"

            episode_options = list(episode_animation_groups.keys())
            selected_animation_episode_key = st.selectbox(
                "選擇其它集數動畫",
                episode_options,
                key=f"storyboard_other_animation_episode_{info['ep']}_{selected_scene_id}",
                format_func=lambda value: f"{episode_group_labels.get(value, value)}（{len(episode_animation_groups.get(value, []))} 支）",
                help="一次載入一集的動畫，避免跨集 85 支影片同時重載造成黑畫面。",
            )
            selected_episode_animations = episode_animation_groups.get(selected_animation_episode_key, [])
            st.caption(
                f"顯示 {episode_group_labels.get(selected_animation_episode_key, '此集')} 的 "
                f"{len(selected_episode_animations)} 支動畫。"
            )
            animation_rows = [selected_episode_animations[i:i + 4] for i in range(0, len(selected_episode_animations), 4)]
            for row_group in animation_rows:
                cols = st.columns(4)
                for col_idx, item in enumerate(row_group):
                    asset_idx, candidate_path = item
                    with cols[col_idx]:
                        with st.container(border=True):
                            st.caption(animation_caption_map.get(asset_idx, candidate_path.name))
                            st.video(str(candidate_path))
                            if st.button(
                                "替換此分鏡動畫",
                                key=f"storyboard_replace_other_animation_{info['ep']}_{selected_scene_id}_{asset_idx}",
                                use_container_width=True,
                            ):
                                replace_scene_with_other_animation(candidate_path)
        else:
            st.caption("目前 Tag 篩選下沒有其它集數動畫。")

    st.subheader("Scene 設定")
    basic1, basic2, basic3 = st.columns([1, 1.2, 1.4])
    with basic1:
        if no_flashcard_mode:
            source_type_val = "AI"
            st.text_input("source_type", value="AI", disabled=True, key=f"{source_key}_story_fixed")
        else:
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
        if no_flashcard_mode:
            flashcard_word_val = ""
        else:
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

    action1, action2, action3, action4, action5 = st.columns([1, 1.05, 1.15, 1.05, 1.25])
    with action1:
        apply_scene = st.button("儲存此 Scene", key=f"storyboard_save_scene_{info['ep']}_{selected_scene_id}")
    with action2:
        regen_scene = st.button(
            "重產此 AI 分鏡圖",
            key=f"storyboard_regen_scene_{info['ep']}_{selected_scene_id}",
            disabled=(source_type_val != "AI") or bool(selected_scene_task.get("running")) or bool(selected_img_anim_task.get("running")),
        )
    with action3:
        suggest_animation_prompt = st.button(
            "AI 建議動畫 Prompt",
            key=f"storyboard_suggest_anim_prompt_{info['ep']}_{selected_scene_id}",
            disabled=bool(selected_anim_prompt_task.get("running")) or bool(selected_img_anim_task.get("running")),
        )
    with action4:
        generate_animation = st.button(
            "產生動畫",
            key=f"storyboard_generate_animation_{info['ep']}_{selected_scene_id}",
            disabled=bool(selected_scene_task.get("running")) or bool(selected_anim_video_task.get("running")) or bool(selected_img_anim_task.get("running")),
        )
    with action5:
        generate_image_and_animation = st.button(
            "重產圖並產動畫",
            key=f"storyboard_generate_image_and_animation_{info['ep']}_{selected_scene_id}",
            disabled=(
                source_type_val != "AI"
                or bool(selected_scene_task.get("running"))
                or bool(selected_anim_video_task.get("running"))
                or bool(selected_img_anim_task.get("running"))
            ),
            help="使用目前的 image_prompt 先重產這個 Scene 的圖片，再直接接著產生動畫。",
        )

    if apply_scene:
        current_custom_image_path = str(selected_row.get("custom_image_path", "")).strip()
        custom_image_path_input = custom_image_val.strip()
        custom_image_changed = bool(custom_image_path_input) and custom_image_path_input != current_custom_image_path
        chosen_ai_image_path = (
            custom_image_path_input
            if custom_image_changed
            else (selected_image_choice_path or custom_image_path_input)
        )
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
                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id), "--provider", selected_image_provider],
                    "env": current_media_env,
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
                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id), "--provider", "auto"],
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
                    "args": ["--ep", str(info["ep"]), "--scene_id", str(selected_scene_id), "--provider", selected_animation_provider],
                    "env": current_media_env,
                },
            )
            if res.get("ok"):
                st.success(f"Scene {selected_scene_id} 動畫任務已啟動。log: {res.get('log')}")
            else:
                st.error(f"Scene {selected_scene_id} 動畫任務啟動失敗：{res.get('message')}")
            st.rerun()
    if generate_image_and_animation:
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
        elif not st.session_state[prompt_key].strip():
            st.error("image_prompt 為空，請先確認這個 AI Scene 的產圖提示。")
        elif not st.session_state[animation_prompt_key].strip():
            st.error("animation_prompt 為空，請先產生或編輯動畫 Prompt。")
        else:
            res = start_publisher_task(
                info,
                scene_image_and_animation_task_name,
                {
                    "name": f"重產 Scene {selected_scene_id} 圖片並產生動畫",
                    "type": "python",
                    "script": "scripts/generate_scene_image_and_animation.py",
                    "args": [
                        "--ep",
                        str(info["ep"]),
                        "--scene_id",
                        str(selected_scene_id),
                        "--provider",
                        selected_image_provider,
                        "--animation_provider",
                        selected_animation_provider,
                    ],
                    "env": current_media_env,
                },
            )
            if res.get("ok"):
                st.success(f"Scene {selected_scene_id} 圖片+動畫任務已啟動。log: {res.get('log')}")
            else:
                st.error(f"Scene {selected_scene_id} 圖片+動畫任務啟動失敗：{res.get('message')}")
            st.rerun()

    any_scene_worker_running = any(
        bool(task.get("running"))
        for task in [selected_scene_task, selected_anim_prompt_task, selected_anim_video_task, selected_img_anim_task, batch_animation_task]
    )
    scene_refresh_interval = 2 if (storyboard_auto_refresh and any_scene_worker_running) else None

    @st.fragment(run_every=scene_refresh_interval)
    def render_storyboard_scene_status():
        latest_storyboard_df = normalize_storyboard_df(ep_path, read_storyboard_df(ep_path))
        latest_match = latest_storyboard_df[latest_storyboard_df["scene_id"].astype(str) == str(selected_scene_id)]
        latest_row = latest_match.iloc[0].copy() if not latest_match.empty else selected_row.copy()
        latest_scene_state = publisher_task_state(info, f"sceneimg_{selected_scene_id}")
        latest_anim_prompt_state = publisher_task_state(info, f"sceneanimprompt_{selected_scene_id}")
        latest_anim_video_state = publisher_task_state(info, f"sceneanimvideo_{selected_scene_id}")
        latest_img_anim_state = publisher_task_state(info, scene_image_and_animation_task_name)
        latest_batch_animation_state = publisher_task_state(info, batch_animation_task_name)
        latest_scene_log = Path(latest_scene_state.get("log", ""))
        latest_anim_prompt_log = Path(latest_anim_prompt_state.get("log", ""))
        latest_anim_video_log = Path(latest_anim_video_state.get("log", ""))
        latest_img_anim_log = Path(latest_img_anim_state.get("log", ""))
        latest_batch_animation_log = Path(latest_batch_animation_state.get("log", ""))
        latest_image_path = str(storyboard_scene_image_path(ep_path, latest_row) or "").strip()
        latest_animation_path = str(storyboard_scene_animation_path(ep_path, latest_row) or "").strip()
        latest_image_candidates = scene_image_candidate_paths(ep_path, latest_row)
        latest_animation_candidates = scene_animation_candidate_paths(ep_path, latest_row)
        newest_animation_candidate = newest_existing_path(latest_animation_candidates)
        latest_any_running = any([
            latest_scene_state.get("running"),
            latest_anim_prompt_state.get("running"),
            latest_anim_video_state.get("running"),
            latest_img_anim_state.get("running"),
            latest_batch_animation_state.get("running"),
        ])

        st.subheader("Scene 執行狀態")
        status_top1, status_top2, status_top3, status_top4, status_top5, status_top6 = st.columns([1, 1, 1, 1.2, 1.1, 1.2])
        status_top1.metric("產圖任務", "執行中" if latest_scene_state.get("running") else ("已完成" if latest_scene_log.exists() else "尚未執行"))
        status_top2.metric("動畫 Prompt", "執行中" if latest_anim_prompt_state.get("running") else ("已完成" if latest_anim_prompt_log.exists() else "尚未執行"))
        status_top3.metric("動畫生成", "執行中" if latest_anim_video_state.get("running") else ("已生成候選" if newest_animation_candidate else "尚未執行"))
        status_top4.metric("候選動畫數", len(latest_animation_candidates))
        status_top5.metric("批次動畫", "執行中" if latest_batch_animation_state.get("running") else ("已完成" if latest_batch_animation_log.exists() else "尚未執行"))
        status_top6.metric("圖+動畫", "執行中" if latest_img_anim_state.get("running") else ("已完成" if latest_img_anim_log.exists() else "尚未執行"))
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
        if latest_img_anim_state.get("running"):
            st.info(f"Scene {selected_scene_id} 圖片+動畫任務執行中，PID {latest_img_anim_state.get('pid')}")
        if latest_batch_animation_state.get("running"):
            st.info(f"整批 AI 圖片轉動畫執行中，PID {latest_batch_animation_state.get('pid')}")
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
        if latest_img_anim_log.exists():
            img_anim_log_text = latest_img_anim_log.read_text(encoding="utf-8", errors="ignore")
            with st.expander("此 Scene 圖片+動畫 Log", expanded=bool(latest_img_anim_state.get("running"))):
                st.code(img_anim_log_text[-6000:])
                if latest_img_anim_state.get("running"):
                    st.caption("執行中。此區會自動刷新。")
        if latest_batch_animation_log.exists():
            batch_log_text = latest_batch_animation_log.read_text(encoding="utf-8", errors="ignore")
            with st.expander("整批 AI 圖片轉動畫 Log", expanded=bool(latest_batch_animation_state.get("running"))):
                st.code(batch_log_text[-6000:])
                if latest_batch_animation_state.get("running"):
                    st.caption("執行中。此區會自動刷新。")
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
    if no_flashcard_mode:
        editable_df = editable_df.drop(columns=["flashcard_word"], errors="ignore")
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
            "source_type": st.column_config.SelectboxColumn(options=["AI"] if no_flashcard_mode else ["AI", "FLASHCARD"]),
            "flashcard_word": st.column_config.TextColumn(),
            "custom_image_path": st.column_config.TextColumn(width="large"),
            "image_prompt": st.column_config.TextColumn(width="large"),
            "animation_prompt": st.column_config.TextColumn(width="large"),
            "animation_video_path": st.column_config.TextColumn(width="large"),
            "reason": st.column_config.TextColumn(width="medium"),
            "subtitle_reference": st.column_config.TextColumn(width="large"),
        },
        key=f"storyboard_editor_{storyboard_key_suffix}",
    )
    if st.button("儲存整張分鏡表", key=f"save_storyboard_all_{storyboard_key_suffix}"):
        if no_flashcard_mode:
            edited_storyboard_df["source_type"] = "AI"
            edited_storyboard_df["flashcard_word"] = ""
        saved_storyboard = save_storyboard_df(ep_path, edited_storyboard_df)
        st.success(f"已儲存分鏡表：{saved_storyboard}")
        st.rerun()

def notebooklm_prompt_template_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "notebooklm_prompt_template.txt"

def ensure_notebooklm_prompt_template_file(profile_id: str) -> Path:
    prompt_path = notebooklm_prompt_template_path(profile_id)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_NOTEBOOKLM_PROMPT_TEMPLATE)
    return prompt_path

def notebooklm_prompt_output_path(ep_path: Path, ep_num: int) -> Path:
    return ep_path / f"notebooklm_prompt_ep{ep_num:02d}.txt"

def subtitle_review_prompt_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "subtitle_review_prompt.txt"

def ensure_subtitle_review_prompt_file(profile_id: str) -> Path:
    prompt_path = subtitle_review_prompt_path(profile_id)
    if not prompt_path.exists():
        save_text_file(prompt_path, DEFAULT_SUBTITLE_REVIEW_PROMPT_TEMPLATE)
    return prompt_path

def storyboard_prompt_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "storyboard_prompt.txt"

def ensure_storyboard_prompt_file(profile_id: str) -> Path:
    prompt_path = storyboard_prompt_path(profile_id)
    if not prompt_path.exists():
        default_template = (
            DEFAULT_SINGLE_VIDEO_STORYBOARD_PROMPT_TEMPLATE
            if profile_id == "single_video"
            else DEFAULT_STORYBOARD_PROMPT_TEMPLATE
        )
        save_text_file(prompt_path, default_template)
    return prompt_path

def youtube_meta_prompt_path(profile_id: str) -> Path:
    return shared_prompt_dir(profile_id) / "youtube_meta_prompt.txt"

def ensure_youtube_meta_prompt_file(profile_id: str) -> Path:
    prompt_path = youtube_meta_prompt_path(profile_id)
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
    elif file_text.strip() and not str(st.session_state.get(text_key, "")).strip():
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

def publish_settings_path(ep_path: Path) -> Path:
    return ep_path / "05_output" / "publish_settings.json"

def load_publish_settings(ep_path: Path | None) -> Dict:
    if not ep_path:
        return {}
    path = publish_settings_path(ep_path)
    payload = read_json_file(path)
    return payload if isinstance(payload, dict) else {}

def save_publish_settings(ep_path: Path, settings: Dict) -> Path:
    path = publish_settings_path(ep_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def post_publish_settings_path(ep_path: Path) -> Path:
    return ep_path / "05_output" / "post_publish_settings.json"


def load_post_publish_settings(ep_path: Path | None) -> Dict:
    if not ep_path:
        return {}
    payload = read_json_file(post_publish_settings_path(ep_path))
    return payload if isinstance(payload, dict) else {}


def save_post_publish_settings(ep_path: Path, settings: Dict) -> Path:
    path = post_publish_settings_path(ep_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_cloze_question_rows(ep_path: Path) -> list[Dict]:
    json_path = cloze_question_json_path(ep_path)
    csv_path = cloze_question_csv_path(ep_path)
    rows = []
    if json_path.exists():
        payload = read_json_file(json_path)
        if isinstance(payload, list):
            rows = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict) and isinstance(payload.get("questions"), list):
            rows = [item for item in payload.get("questions", []) if isinstance(item, dict)]
    if not rows and csv_path.exists():
        try:
            rows = pd.read_csv(csv_path).fillna("").to_dict("records")
        except Exception:
            rows = []
    return rows


def cloze_question_id(row: Dict, fallback_index: int) -> str:
    raw = str(row.get("question_id", "") or "").strip()
    return raw or f"Q{fallback_index:02d}"


def format_cloze_question_for_post(row: Dict, fallback_index: int) -> str:
    qid = cloze_question_id(row, fallback_index)
    sentence = str(row.get("blank_sentence", "") or row.get("question", "") or "").strip()
    choices = []
    for opt in ["A", "B", "C", "D"]:
        choice_text = str(row.get(f"choice_{opt}", "") or row.get(opt, "") or "").strip()
        if choice_text:
            choices.append(f"{opt}. {choice_text}")
    word_hint = str(row.get("word", "") or row.get("correct_word", "") or "").strip()
    header = f"{qid}"
    if word_hint:
        header += f" | {word_hint}"
    parts = [header]
    if sentence:
        parts.append(sentence)
    parts.extend(choices)
    return "\n".join(parts)


def build_cloze_post_text(ep_num: int, selected_rows: list[Dict]) -> str:
    blocks = [f"Ep{ep_num:02d} 克漏字小測驗"]
    for idx, row in enumerate(selected_rows, start=1):
        blocks.append(format_cloze_question_for_post(row, idx))
    blocks.append("把你的答案留言在下方，下一集公布解析。")
    return "\n\n".join(block for block in blocks if block.strip())


def publish_video_candidates(ep_path: Path) -> list[dict]:
    output_dir = ep_path / "05_output"
    candidates = [
        ("final_video_with_outro.mp4", "含片尾版"),
        ("final_video_no_outro.mp4", "無片尾版"),
        ("final_video.mp4", "相容預設版"),
    ]
    rows = []
    seen = set()
    for filename, label in candidates:
        if filename in seen:
            continue
        seen.add(filename)
        path = output_dir / filename
        if path.exists():
            rows.append({"filename": filename, "label": label, "path": path})
    return rows


def default_publish_video_name(ep_path: Path, publish_defaults: Dict) -> str:
    saved_name = Path(str(publish_defaults.get("video_name", "") or "")).name
    candidates = publish_video_candidates(ep_path)
    candidate_names = [item["filename"] for item in candidates]
    if saved_name and saved_name in candidate_names:
        return saved_name
    if "final_video_with_outro.mp4" in candidate_names:
        return "final_video_with_outro.mp4"
    if "final_video_no_outro.mp4" in candidate_names:
        return "final_video_no_outro.mp4"
    return "final_video.mp4"

def parse_publish_at_value(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    except Exception:
        return None

def studio_default_publish_settings(ep_path: Path, ep_num: int, upload_record: Dict) -> Dict:
    current_saved = load_publish_settings(ep_path)
    if current_saved:
        return current_saved

    current_from_record = {
        "privacy": str(upload_record.get("privacy", "")).strip(),
        "publish_at": str(upload_record.get("publish_at", "")).strip(),
        "playlist_id": str(upload_record.get("playlist_id", "")).strip(),
        "cover_name": str(upload_record.get("cover_name", "")).strip(),
        "video_name": str(upload_record.get("video_name", "")).strip(),
    }
    if any(current_from_record.values()):
        return {
            "privacy": current_from_record["privacy"] or "private",
            "schedule_on": bool(current_from_record["publish_at"]),
            "publish_at": current_from_record["publish_at"],
            "playlist_id": current_from_record["playlist_id"],
            "cover_name": current_from_record["cover_name"] or "cover.png",
            "video_name": current_from_record["video_name"] or "",
        }

    prev_ep_path = find_episode_path_by_number(ep_num - 1) if ep_num > 1 else None
    prev_saved = load_publish_settings(prev_ep_path)
    if prev_saved:
        return prev_saved

    prev_record = read_json_file((prev_ep_path / "05_output" / "upload_record.json")) if prev_ep_path else {}
    if isinstance(prev_record, dict) and prev_record:
        return {
            "privacy": str(prev_record.get("privacy", "")).strip() or "private",
            "schedule_on": bool(str(prev_record.get("publish_at", "")).strip()),
            "publish_at": str(prev_record.get("publish_at", "")).strip(),
            "playlist_id": str(prev_record.get("playlist_id", "")).strip(),
            "cover_name": str(prev_record.get("cover_name", "")).strip() or "cover.png",
            "video_name": str(prev_record.get("video_name", "")).strip(),
        }

    return {
        "privacy": "private",
        "schedule_on": False,
        "publish_at": "",
        "playlist_id": "",
        "cover_name": "cover.png",
        "video_name": "",
    }

def resolve_default_playlist_id(playlist_options: list[Dict], preferred_playlist_id: str) -> str:
    preferred = str(preferred_playlist_id or "").strip()
    playlist_ids = [str(item.get("id", "")).strip() for item in playlist_options]
    if preferred and preferred in playlist_ids:
        return preferred
    for item in playlist_options:
        playlist_id = str(item.get("id", "")).strip()
        if playlist_id:
            return playlist_id
    return playlist_options[0]["id"] if playlist_options else ""

def youtube_auth_file_signatures() -> tuple[int, int]:
    token_path = ROOT / "token_youtube.pickle"
    client_path = ROOT / "client_secret.json"
    token_sig = token_path.stat().st_mtime_ns if token_path.exists() else 0
    client_sig = client_path.stat().st_mtime_ns if client_path.exists() else 0
    return token_sig, client_sig

def reauthorize_youtube_for_app() -> None:
    import pickle
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = ROOT / "token_youtube.pickle"
    client_path = ROOT / "client_secret.json"
    scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

    if not client_path.exists():
        raise FileNotFoundError(f"找不到 client_secret.json：{client_path}")

    try:
        if token_path.exists():
            token_path.unlink()
    except Exception as e:
        raise RuntimeError(f"刪除舊的 YouTube token 失敗：{e}")

    flow = InstalledAppFlow.from_client_secrets_file(str(client_path), scopes)
    creds = flow.run_local_server(port=0)
    with open(token_path, "wb") as token_fp:
        pickle.dump(creds, token_fp)

def load_youtube_service_for_app():
    import pickle
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    from googleapiclient.discovery import build

    token_path = ROOT / "token_youtube.pickle"
    client_path = ROOT / "client_secret.json"
    if not client_path.exists():
        raise FileNotFoundError(f"找不到 client_secret.json：{client_path}")

    creds = None
    if token_path.exists():
        try:
            with open(token_path, "rb") as token_fp:
                creds = pickle.load(token_fp)
        except Exception as e:
            raise RuntimeError(f"讀取 YouTube token 失敗：{e}")

    if not creds:
        raise RuntimeError("尚未完成 YouTube OAuth 授權，請先執行一次上傳流程。")

    if not creds.valid:
        if creds.expired and getattr(creds, "refresh_token", None):
            try:
                creds.refresh(Request())
                with open(token_path, "wb") as token_fp:
                    pickle.dump(creds, token_fp)
            except RefreshError as e:
                raise RuntimeError(f"YouTube token refresh 失敗：{e}。請重新授權 YouTube。")
        else:
            raise RuntimeError("YouTube token 無效，請重新授權。")

    return build("youtube", "v3", credentials=creds)

@st.cache_data(show_spinner=False, ttl=300)
def list_youtube_playlists_cached(token_sig: int, client_sig: int):
    _ = (token_sig, client_sig)
    try:
        youtube = load_youtube_service_for_app()
        results = []
        page_token = None
        while True:
            resp = youtube.playlists().list(
                part="snippet",
                mine=True,
                maxResults=50,
                pageToken=page_token,
            ).execute()
            for item in resp.get("items", []):
                playlist_id = str(item.get("id", "")).strip()
                title = str((item.get("snippet") or {}).get("title", "")).strip()
                if playlist_id and title:
                    results.append({
                        "id": playlist_id,
                        "title": title,
                        "label": f"{title} ({playlist_id})",
                    })
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return results, ""
    except Exception as e:
        return [], str(e)

def storyboard_path_for(ep_path: Path) -> Path:
    if st.session_state.get("profile_id") == "story":
        return ep_path / "05_storyboards" / "storyboard.csv"
    return ep_path / "03_storyboards" / "storyboard.csv"


@st.cache_data(show_spinner=False, max_entries=128)
def _read_storyboard_df_cached(path_str: str, signature: tuple[int, int]) -> pd.DataFrame:
    _ = signature
    return pd.read_csv(path_str, encoding="utf-8-sig").fillna("")


def read_storyboard_df(ep_path: Path) -> pd.DataFrame:
    p = storyboard_path_for(ep_path)
    if not p.exists():
        return pd.DataFrame()
    return _read_storyboard_df_cached(str(p), file_signature(p))

def storyboard_file_signature(ep_path: Path) -> str:
    p = storyboard_path_for(ep_path)
    if not p.exists():
        return "missing"
    try:
        stat = p.stat()
        return f"{int(stat.st_mtime)}_{stat.st_size}"
    except Exception:
        return "unknown"

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
        "- Keep the hosts as supporting figures unless the subtitle clearly requires them to be the only main subject.",
        "- Prefer wide shot, medium-long shot, side angle, or over-the-shoulder staging; avoid portrait framing, close-ups, and large front-facing faces.",
        "- In quiz, explanation, or infographic scenes, keep the board, text, or teaching object as the main visual focus and place the hosts smaller near the edges.",
        "- Preserve host identity mainly through outfit, silhouette, color palette, and left/right placement instead of detailed facial close-ups.",
        "- If a scene is purely object-based, abstract, or clearly requires another role, do not force the hosts into that scene.",
    ]
    return "\n".join(rules)

def host_profile_json_text_for_prompt() -> str:
    host_profile_path = ROOT / "core" / "assets" / "host_profiles.json"
    payload = read_json_file(host_profile_path) or {}
    return json.dumps(payload, ensure_ascii=False, indent=2) if payload else "{}"

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

def render_step7_1_prompt_preview(template_text: str, ep_path: Path) -> str:
    ep_info = parse_episode_info(ep_path)
    vocab_df = load_vocab_df_for_prompt(ep_path)
    return (
        template_text
        .replace("{{EPISODE}}", str(ep_info.get("ep") or ""))
        .replace("{{VOCAB_LIST}}", build_vocab_list_text_for_prompt(vocab_df))
        .replace("{{HOST_PROFILE_JSON}}", host_profile_json_text_for_prompt())
    ).strip() + "\n"

def render_step12_prompt_preview(template_text: str, ep_path: Path, subtitle_path: Path) -> str:
    ep_num = int(parse_episode_info(ep_path).get("ep") or 0)
    notebook_prompt = read_text_file(notebooklm_prompt_output_path(ep_path, ep_num), "").strip()
    rendered = (
        template_text
        .replace("{{SRT_CONTENT}}", read_text_file(subtitle_path, ""))
        .replace("{{NOTEBOOKLM_PROMPT}}", notebook_prompt or "（未找到 No.8 output，請僅依原始字幕內容校對。）")
    )
    if "{{NOTEBOOKLM_PROMPT}}" not in template_text and notebook_prompt:
        rendered += "\n\n【No.8 產生語音摘要 Prompt 參考】\n" + notebook_prompt + "\n"
    return rendered

def render_step14_prompt_preview(template_text: str, ep_path: Path, subtitle_path: Path) -> str:
    vocab_df = load_vocab_df_for_prompt(ep_path)
    word_list = vocab_df["Word"].astype(str).str.strip().tolist() if not vocab_df.empty and "Word" in vocab_df.columns else []
    cloze_rules = build_storyboard_cloze_rules_for_prompt(ep_path, int(parse_episode_info(ep_path).get("ep") or 0))
    host_rules = build_storyboard_host_rules_for_prompt()
    subtitle_text = read_text_file(
        subtitle_path,
        f"（尚未找到 No.12 校對字幕：{subtitle_path}。請先完成 No.12，No.14 才能產生分鏡。）",
    )
    rendered = (
        template_text
        .replace("{{WORD_LIST}}", json.dumps(word_list, ensure_ascii=False, indent=2))
        .replace("{{CLOZE_CARD_RULES}}", cloze_rules)
        .replace("{{HOST_PROFILE_RULES}}", host_rules)
        .replace("{{SRT_CONTENT}}", subtitle_text)
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
    if st.session_state.get("profile_id") == "story":
        return ep_path / "05_storyboards" / "images" / "ai_generated" / f"img_{start_token}.png"
    return images_dir(ep_path) / "ai_generated" / f"img_{start_token}.png"

def animation_dir_for(ep_path: Path) -> Path:
    if st.session_state.get("profile_id") == "story":
        return ep_path / "05_storyboards" / "animations"
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

def episode_path_for_asset(asset_path: Path) -> Path | None:
    return shared_episode_path_for_asset(asset_path)

def other_episode_image_candidate_paths(current_ep_path: Path, limit: int = 120) -> list[Path]:
    candidates: list[Path] = []
    for episode_path in reversed(list_episode_dirs(ROOT, WS_ROOT)):
        if episode_path == current_ep_path:
            continue
        for folder in [
            episode_path / "04_images" / "ai_generated",
            episode_path / "04_images" / "preview_images",
        ]:
            if not folder.exists():
                continue
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                candidates.extend(sorted(folder.glob(pattern), reverse=True))
    return _dedupe_existing_paths(candidates)[:limit]

def other_episode_animation_candidate_paths(current_ep_path: Path, limit: int = 120) -> list[Path]:
    candidates: list[Path] = []
    for episode_path in reversed(list_episode_dirs(ROOT, WS_ROOT)):
        if episode_path == current_ep_path:
            continue
        folder = episode_path / "04_images" / "animations"
        if not folder.exists():
            continue
        for pattern in ("*.mp4", "*.mov", "*.webm", "*.m4v"):
            candidates.extend(sorted(folder.glob(pattern), reverse=True))
    return _dedupe_existing_paths(candidates)[:limit]

ASSET_TAG_FILTER_ALL = SHARED_ASSET_TAG_FILTER_ALL
ASSET_TAG_FILTER_UNTAGGED = SHARED_ASSET_TAG_FILTER_UNTAGGED

def asset_tags_json_path(ep_path: Path) -> Path:
    return shared_asset_tags_json_path(ep_path)

def asset_tag_storage_key(asset_path: Path, ep_path: Path | None = None) -> str:
    return shared_asset_tag_storage_key(asset_path, ep_path)

def read_episode_asset_tags(ep_path: Path) -> dict[str, str]:
    return shared_read_episode_asset_tags(ep_path)

def write_episode_asset_tags(ep_path: Path, tags: dict[str, str]) -> Path:
    return shared_write_episode_asset_tags(ep_path, tags)

def asset_tag_for_path(asset_path: Path | str | None) -> str:
    return shared_asset_tag_for_path(asset_path)

def set_asset_tag_for_path(asset_path: Path | str, tag_text: str) -> Path | None:
    return shared_set_asset_tag_for_path(asset_path, tag_text)

def collect_asset_tag_options(paths: list[Path]) -> list[str]:
    return shared_collect_asset_tag_options(paths)

def filter_asset_paths_by_tag(paths: list[Path], selected_tag: str) -> list[Path]:
    return shared_filter_asset_paths_by_tag(paths, selected_tag)

def asset_tag_badge(path: Path | None) -> str:
    return shared_asset_tag_badge(path)

def cross_episode_asset_choice_label(current_ep_path: Path, path: Path, kind: str) -> str:
    asset_ep_path = episode_path_for_asset(path)
    if not asset_ep_path or asset_ep_path == current_ep_path:
        return asset_choice_label(path, None, kind)
    info = parse_episode_info(asset_ep_path)
    ep_text = f"Ep{int(info.get('ep')):02d}" if info.get("ep") is not None else asset_ep_path.name
    rel_text = path.relative_to(asset_ep_path).as_posix()
    return f"其它集 {ep_text} | {rel_text}"

def asset_choice_label(path: Path, base_path: Path | None, kind: str) -> str:
    if base_path and path.resolve() == base_path.resolve():
        return f"原始{kind} | {path.name}"
    m = re.search(r"__v(\d+)$", path.stem)
    if m:
        return f"{kind}變體 v{m.group(1)} | {path.name}"
    return f"自訂{kind} | {path.name}"

def asset_label_with_tag(current_ep_path: Path, path: Path, kind: str, base_path: Path | None = None) -> str:
    asset_ep_path = episode_path_for_asset(path)
    if asset_ep_path and asset_ep_path != current_ep_path:
        base_label = cross_episode_asset_choice_label(current_ep_path, path, kind)
    else:
        base_label = asset_choice_label(path, base_path, kind)
    return f"{base_label} | Tag: {asset_tag_badge(path)}"

def build_asset_choice_map(
    current_ep_path: Path,
    paths: list[Path],
    kind: str,
    base_path: Path | None = None,
) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for path in paths:
        base_label = asset_label_with_tag(current_ep_path, path, kind, base_path=base_path)
        label = base_label
        suffix = 2
        while label in label_map and label_map[label] != str(path):
            label = f"{base_label} ({suffix})"
            suffix += 1
        label_map[label] = str(path)
    return label_map

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
        "paragraph_id",
        "location_id",
        "location_name",
        "characters",
        "summary",
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

@st.cache_data(show_spinner=False, max_entries=1024)
def _media_duration_seconds_cached(path_str: str, signature: tuple[int, int]) -> float | None:
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

def media_duration_seconds(path_str: str) -> float | None:
    path = Path(path_str)
    return _media_duration_seconds_cached(str(path), file_signature(path))

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
def _image_to_preview_data_uri_cached(
    path_str: str,
    file_mtime_ns: int,
    file_size: int,
    max_width: int = 960,
    quality: int = 78,
) -> str:
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

def image_to_preview_data_uri(path_str: str, max_width: int = 960, quality: int = 78) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    try:
        file_mtime_ns, file_size = _preview_image_signature(path_str)
    except Exception:
        return ""
    return _image_to_preview_data_uri_cached(
        str(path),
        file_mtime_ns,
        file_size,
        max_width=max_width,
        quality=quality,
    )

def build_episode_preview_payload(ep_path: Path, storyboard_df: pd.DataFrame, subtitle_entries: list[Dict]) -> Dict:
    image_cache: Dict[str, str] = {}
    video_cache: Dict[str, str] = {}
    sorted_storyboard_df = storyboard_df.copy()
    if not sorted_storyboard_df.empty:
        sorted_storyboard_df["_preview_start"] = sorted_storyboard_df["start_time"].apply(lambda v: safe_float(v, 0))
        sorted_storyboard_df["_preview_end"] = sorted_storyboard_df["end_time"].apply(lambda v: safe_float(v, 0))
        sorted_storyboard_df["_preview_scene"] = sorted_storyboard_df["scene_id"].apply(lambda v: safe_float(v, 0))
        sorted_storyboard_df = sorted_storyboard_df.sort_values(
            ["_preview_start", "_preview_end", "_preview_scene"],
            kind="stable",
        ).drop(columns=["_preview_start", "_preview_end", "_preview_scene"])
    animation_paths = []
    for _, row in sorted_storyboard_df.iterrows():
        animation_path = storyboard_scene_animation_path(ep_path, row)
        if animation_path and animation_path.exists():
            animation_paths.append(animation_path)
    unique_animation_paths = {str(p.resolve()): p for p in animation_paths}
    total_animation_bytes = sum(p.stat().st_size for p in unique_animation_paths.values())
    inline_animation_limit = 25 * 1024 * 1024
    allow_inline_animations = total_animation_bytes <= inline_animation_limit
    scenes = []
    for _, row in sorted_storyboard_df.iterrows():
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
            "animation_file_uri": path_to_file_uri(Path(animation_path_str)) if animation_path_str else "",
            "asset_mode": "ANIMATION" if animation_path_str else "IMAGE",
        })
    outro_config = load_vocab_outro_config(ep_path)
    outro_media_path = find_vocab_outro_media(ep_path)
    outro_audio_path = find_vocab_outro_audio(ep_path)
    outro_duration = safe_float(outro_config.get("duration_seconds", 0), 0)
    if outro_media_path and outro_media_path.exists() and outro_duration > 0:
        outro_start = max(
            [safe_float(scene.get("end", 0)) for scene in scenes]
            + [safe_float(row.get("end", 0)) for row in subtitle_entries]
            + [0.0]
        )
        media_suffix = outro_media_path.suffix.lower()
        is_video = media_suffix in {".mp4", ".mov", ".webm", ".m4v"}
        outro_image_data_uri = ""
        outro_animation_data_uri = ""
        outro_animation_file_uri = ""
        if is_video:
            if outro_media_path.stat().st_size <= inline_animation_limit:
                outro_animation_data_uri = video_to_data_uri(str(outro_media_path))
            outro_animation_file_uri = path_to_file_uri(outro_media_path)
        else:
            outro_image_data_uri = image_to_preview_data_uri(str(outro_media_path))
        scenes.append({
            "scene_id": "片尾",
            "start": outro_start,
            "end": outro_start + outro_duration,
            "source_type": "OUTRO",
            "flashcard_word": "",
            "reason": "片尾製作",
            "image_prompt": "",
            "animation_prompt": "",
            "subtitle_reference": "",
            "image_path": str(outro_media_path) if not is_video else "",
            "image_data_uri": outro_image_data_uri,
            "animation_path": str(outro_media_path) if is_video else "",
            "animation_data_uri": outro_animation_data_uri,
            "animation_file_uri": outro_animation_file_uri,
            "asset_mode": "ANIMATION" if is_video else "IMAGE",
            "is_outro": True,
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
        "outro": {
            "enabled": bool(outro_media_path and outro_media_path.exists() and outro_duration > 0),
            "audio_data_uri": file_to_data_uri(outro_audio_path) if outro_audio_path else "",
            "duration": outro_duration,
        },
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
            <audio id="cap-outro-audio" preload="metadata" src=""></audio>
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
      const outro = payload.outro || {{}};
      const audio = document.getElementById("cap-audio");
      const outroAudio = document.getElementById("cap-outro-audio");
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
      let isOutroPlayback = false;
      if (outro.audio_data_uri) outroAudio.src = outro.audio_data_uri;

      function outroScene() {{
        return scenes.find((scene) => scene && scene.is_outro);
      }}

      function previewTime() {{
        const scene = outroScene();
        if (isOutroPlayback && scene) {{
          return Number(scene.start || 0) + Number(outroAudio.currentTime || 0);
        }}
        return Number(audio.currentTime || 0);
      }}

      function playOutro(offsetSeconds = 0) {{
        const scene = outroScene();
        if (!scene) return;
        isOutroPlayback = true;
        audio.pause();
        const duration = Number(scene.end || 0) - Number(scene.start || 0);
        const target = Math.min(Math.max(Number(offsetSeconds || 0), 0), Math.max(duration, 0));
        if (outroAudio.src) {{
          outroAudio.currentTime = target;
          outroAudio.play().catch(() => null);
        }} else {{
          outroAudio.currentTime = target;
        }}
        syncPreview();
      }}

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
        let latestCompletedIndex = -1;
        let latestCompletedEnd = -Infinity;
        for (let i = 0; i < items.length; i += 1) {{
          const item = items[i];
          const start = Number(item.start || 0);
          const end = Number(item.end || 0);
          if (currentTime >= start && currentTime < end) return i;
          if (currentTime >= end && end > latestCompletedEnd) {{
            latestCompletedIndex = i;
            latestCompletedEnd = end;
          }}
        }}
        for (let i = 0; i < items.length; i += 1) {{
          if (currentTime < Number(items[i].start || 0)) return i;
        }}
        return latestCompletedIndex;
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
            const scene = scenes[idx];
            if (scene && scene.is_outro) {{
              playOutro(0);
            }} else {{
              isOutroPlayback = false;
              outroAudio.pause();
              audio.currentTime = Number(scene.start || 0);
              audio.play();
            }}
            syncPreview();
          }});
        }});
        subtitleList.querySelectorAll("[data-sub-index]").forEach((node) => {{
          node.addEventListener("click", () => {{
            const idx = Number(node.getAttribute("data-sub-index"));
            isOutroPlayback = false;
            outroAudio.pause();
            audio.currentTime = Number(subtitles[idx].start || 0);
            audio.play();
            syncPreview();
          }});
        }});
      }}

      function syncPreview() {{
        const currentTime = previewTime();
        const sceneIdx = findActiveIndex(scenes, currentTime);
        const activeSceneForSubtitle = scenes[sceneIdx];
        const subtitleIdx = activeSceneForSubtitle && activeSceneForSubtitle.is_outro ? -1 : findActiveIndex(subtitles, currentTime);

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
            const sceneVideoSrc = scene.animation_data_uri || scene.animation_file_uri || "";
            if (scene.asset_mode === "ANIMATION" && sceneVideoSrc) {{
              if (activeSceneVideoSrc !== sceneVideoSrc) {{
                sceneVideo.src = sceneVideoSrc;
                activeSceneVideoSrc = sceneVideoSrc;
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
        const item = items[nextIndex];
        if (item && item.is_outro) {{
          playOutro(0);
        }} else {{
          isOutroPlayback = false;
          outroAudio.pause();
          audio.currentTime = Number(item.start || 0);
          audio.play();
        }}
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
      audio.addEventListener("ended", () => {{
        const scene = outroScene();
        if (scene) playOutro(0);
      }});
      audio.addEventListener("pause", () => {{
        if (!sceneVideo.paused) sceneVideo.pause();
      }});
      audio.addEventListener("play", () => {{
        if (isOutroPlayback) {{
          isOutroPlayback = false;
          outroAudio.pause();
        }}
        if (sceneVideo.src) sceneVideo.play().catch(() => null);
      }});
      outroAudio.addEventListener("timeupdate", syncPreview);
      outroAudio.addEventListener("ended", () => {{
        isOutroPlayback = false;
        syncPreview();
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

configure_story_section(
    ROOT,
    WS_ROOT,
    runner,
    {
        "read_json_file": read_json_file,
        "read_text_file": read_text_file,
        "parse_srt_entries": parse_srt_entries,
        "seconds_to_label": seconds_to_label,
    },
)

configure_short_generator(
    ROOT,
    {
        "media_duration_seconds": media_duration_seconds,
        "seconds_to_label": seconds_to_label,
        "youtube_auth_file_signatures": youtube_auth_file_signatures,
        "list_youtube_playlists_cached": list_youtube_playlists_cached,
        "reauthorize_youtube_for_app": reauthorize_youtube_for_app,
        "resolve_default_playlist_id": resolve_default_playlist_id,
        "asset_tag_badge": asset_tag_badge,
        "asset_tag_for_path": asset_tag_for_path,
        "set_asset_tag_for_path": set_asset_tag_for_path,
        "collect_asset_tag_options": collect_asset_tag_options,
        "filter_asset_paths_by_tag": filter_asset_paths_by_tag,
        "ASSET_TAG_FILTER_ALL": ASSET_TAG_FILTER_ALL,
    },
)

configure_episode_merge(
    ROOT,
    profiles,
    {
        "story_subtitles_srt_path": story_subtitles_srt_path,
        "media_duration_seconds": media_duration_seconds,
        "seconds_to_label": seconds_to_label,
        "get_schedule_status_label": get_schedule_status_label,
        "read_json_file": read_json_file,
        "parse_srt_entries": parse_srt_entries,
        "safe_float": safe_float,
        "read_text_file": read_text_file,
        "youtube_auth_file_signatures": youtube_auth_file_signatures,
        "list_youtube_playlists_cached": list_youtube_playlists_cached,
        "reauthorize_youtube_for_app": reauthorize_youtube_for_app,
        "resolve_default_playlist_id": resolve_default_playlist_id,
    },
)

# convenience
STAGE_IDS = get_stage_ids_for_profile(st.session_state["profile_id"])
STAGE_RANGE_LABEL = get_stage_range_label(STAGE_IDS)

# --- UI ---
if section == "Short 管理":
    render_short_management()

elif section == "短影音工廠":
    render_short_factory()

elif section == "集數合併":
    render_episode_merge_workspace()

elif section == "📊 Dashboard":
    st.header("專案與進度總覽")
    if st.session_state["profile_id"] == "vocab":
        next_ep_default = next_episode_number_default(VOCAB_REBASE_EP)
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
                single_ep = int(st.number_input("產生集數", min_value=VOCAB_REBASE_EP, value=next_ep_default, step=1, key="gen_single_ep"))
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
                batch_start_ep = int(st.number_input("產生起始集數", min_value=VOCAB_REBASE_EP, value=next_ep_default, step=1, key="gen_batch_start"))
                batch_end_ep = int(st.number_input("產生結束集數", min_value=batch_start_ep, value=max(next_ep_default, batch_start_ep), step=1, key="gen_batch_end"))
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

    elif st.session_state["profile_id"] == "story":
        next_ep_default = next_episode_number_default(1)
        with st.expander("故事集數建立工具", expanded=True):
            st.caption("故事生成模式使用 EpNN_0000_0000 作為集數資料夾，並建立 01_preproduction 到 07_publish 的故事製作目錄。")
            notice = st.session_state.pop("story_episode_generation_notice", None)
            if isinstance(notice, dict):
                if notice.get("level") == "success":
                    st.success(notice.get("message", ""))
                elif notice.get("level") == "warning":
                    st.warning(notice.get("message", ""))
                elif notice.get("level") == "error":
                    st.error(notice.get("message", ""))

            single_tab, batch_tab = st.tabs(["產生單集", "產生多集"])

            with single_tab:
                single_ep = int(st.number_input("產生集數", min_value=1, value=next_ep_default, step=1, key="story_gen_single_ep"))
                single_allow_backup = st.checkbox(
                    "若該集已有不同資料夾，先備份舊資料夾再重建",
                    value=False,
                    key="story_gen_single_allow_backup",
                )
                single_plans = build_story_episode_generation_plan(single_ep, single_ep)
                st.dataframe(story_generation_plan_table(single_plans), use_container_width=True, hide_index=True)
                if single_plans[0]["conflict_dirs"] and not single_allow_backup:
                    st.warning("此集已有不同命名的既有資料夾。若要套用故事模式資料夾，請勾選備份舊資料夾。")
                if st.button("建立 / 更新故事單集", key="story_gen_single_submit"):
                    result = apply_episode_generation(single_plans, single_allow_backup, get_episode_scaffold_dirs("story"))
                    if result["conflicts"]:
                        blocked_eps = ", ".join(f"Ep{p['ep']:02d}" for p in result["conflicts"])
                        st.session_state["story_episode_generation_notice"] = {
                            "level": "error",
                            "message": f"{blocked_eps} 有既有資料夾衝突，尚未變更。請勾選備份舊資料夾後再執行。",
                        }
                    else:
                        parts = []
                        if result["created"]:
                            parts.append(f"新建 {len(result['created'])} 集")
                        if result["reused"]:
                            parts.append(f"沿用 {len(result['reused'])} 集")
                        if result["backed_up"]:
                            parts.append(f"備份舊資料夾 {len(result['backed_up'])} 個")
                        st.session_state["story_episode_generation_notice"] = {
                            "level": "success",
                            "message": "；".join(parts) or "故事單集設定已完成。",
                        }
                    st.rerun()

            with batch_tab:
                batch_start_ep = int(st.number_input("產生起始集數", min_value=1, value=next_ep_default, step=1, key="story_gen_batch_start"))
                batch_end_ep = int(st.number_input("產生結束集數", min_value=batch_start_ep, value=max(next_ep_default, batch_start_ep), step=1, key="story_gen_batch_end"))
                batch_allow_backup = st.checkbox(
                    "若既有集數資料夾命名不同，先備份舊資料夾再重建",
                    value=False,
                    key="story_gen_batch_allow_backup",
                )
                batch_plans = build_story_episode_generation_plan(batch_start_ep, batch_end_ep)
                st.dataframe(story_generation_plan_table(batch_plans), use_container_width=True, hide_index=True)
                batch_conflicts = [plan for plan in batch_plans if plan["conflict_dirs"]]
                if batch_conflicts and not batch_allow_backup:
                    st.warning(f"目前有 {len(batch_conflicts)} 集存在資料夾衝突。若要套用故事模式，請勾選備份舊資料夾。")
                if st.button("建立 / 更新故事多集", key="story_gen_batch_submit"):
                    result = apply_episode_generation(batch_plans, batch_allow_backup, get_episode_scaffold_dirs("story"))
                    if result["conflicts"]:
                        blocked_eps = ", ".join(f"Ep{p['ep']:02d}" for p in result["conflicts"])
                        st.session_state["story_episode_generation_notice"] = {
                            "level": "error",
                            "message": f"以下集數因資料夾衝突而未變更：{blocked_eps}。請勾選備份舊資料夾後再執行。",
                        }
                    else:
                        parts = []
                        if result["created"]:
                            parts.append(f"新建 {len(result['created'])} 集")
                        if result["reused"]:
                            parts.append(f"沿用 {len(result['reused'])} 集")
                        if result["backed_up"]:
                            parts.append(f"備份舊資料夾 {len(result['backed_up'])} 個")
                        st.session_state["story_episode_generation_notice"] = {
                            "level": "success",
                            "message": "；".join(parts) or "故事多集設定已完成。",
                        }
                    st.rerun()

    elif st.session_state["profile_id"] == "single_video":
        next_ep_default = next_episode_number_default(1)
        with st.expander("新增 YouTube Title", expanded=True):
            st.caption("單一影片生成器使用 EpNN_0000_0000 資料夾；YouTube Title 會儲存在該影片的 video_info.json，並用於 Pipeline 下拉選單與 Prompt。")
            notice = st.session_state.pop("single_video_generation_notice", None)
            if isinstance(notice, dict):
                if notice.get("level") == "success":
                    st.success(notice.get("message", ""))
                elif notice.get("level") == "warning":
                    st.warning(notice.get("message", ""))
                elif notice.get("level") == "error":
                    st.error(notice.get("message", ""))

            single_ep = int(st.number_input("影片編號", min_value=1, value=next_ep_default, step=1, key="single_video_gen_ep"))
            youtube_title = st.text_input("YouTube Title", key="single_video_youtube_title")
            single_allow_backup = st.checkbox(
                "若該編號已有不同資料夾，先備份舊資料夾再重建",
                value=False,
                key="single_video_allow_backup",
            )
            single_plans = build_single_video_generation_plan(single_ep, single_ep, youtube_title)
            st.dataframe(single_video_generation_plan_table(single_plans), use_container_width=True, hide_index=True)
            if single_plans[0]["conflict_dirs"] and not single_allow_backup:
                st.warning("此編號已有不同命名的既有資料夾。若要套用單一影片資料夾，請勾選備份舊資料夾。")
            if st.button("建立 / 更新影片", key="single_video_gen_submit"):
                if not youtube_title.strip():
                    st.session_state["single_video_generation_notice"] = {
                        "level": "error",
                        "message": "請先輸入 YouTube Title。",
                    }
                    st.rerun()
                result = apply_episode_generation(single_plans, single_allow_backup, get_episode_scaffold_dirs("single_video"))
                if result["conflicts"]:
                    blocked_eps = ", ".join(f"Ep{p['ep']:02d}" for p in result["conflicts"])
                    st.session_state["single_video_generation_notice"] = {
                        "level": "error",
                        "message": f"{blocked_eps} 有既有資料夾衝突，尚未變更。請勾選備份舊資料夾後再執行。",
                    }
                else:
                    target_path = single_plans[0]["target_path"]
                    info_path = save_single_video_info(target_path, youtube_title)
                    parts = []
                    if result["created"]:
                        parts.append("新建影片")
                    if result["reused"]:
                        parts.append("更新影片")
                    if result["backed_up"]:
                        parts.append(f"備份舊資料夾 {len(result['backed_up'])} 個")
                    parts.append(f"已儲存 Title：{info_path}")
                    st.session_state["single_video_generation_notice"] = {
                        "level": "success",
                        "message": "；".join(parts),
                    }
                st.rerun()

    elif st.session_state["profile_id"] == "youtube_ai":
        render_youtube_ai_dashboard()

    if st.session_state["profile_id"] == "youtube_ai":
        st.stop()

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

        # 附表：目前 profile 的階段狀態
        with st.expander(f"階段總覽 ({STAGE_RANGE_LABEL})", expanded=False):
            st.dataframe(stage_table(eps), use_container_width=True)

        render_story_outline_selection_panel(eps)

        # 手動確認控制
        if manual_step_ids:
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

elif section == "⚙️ Settings":
    st.header("設定 (Settings)")
    if st.session_state["profile_id"] == "youtube_ai":
        render_youtube_ai_settings()
        st.stop()
    st.info("此專案環境尚未提供 Settings 模組。")

# --- Pipeline Manager ---
elif section in ("⚙️ Pipeline Manager", "🧩 Pipeline Manager"):
    st.header("工作流引擎與日誌 (Pipeline Manager)")
    if st.session_state["profile_id"] == "youtube_ai":
        render_youtube_ai_pipeline_manager()
        st.stop()
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先建立或同步集數資料夾。")
    else:
        if st.session_state["profile_id"] == "single_video":
            ep_map = {single_video_option_label(e): e for e in eps}
            ep_select_label = "選擇要編輯的影片"
        else:
            ep_map = {f"Ep{e['_raw']['ep']:02d} ({e['Range']})": e for e in eps}
            ep_select_label = "選擇集數"
        ep_options = list(ep_map.keys())
        ep_choice = st.selectbox(ep_select_label, ep_options, index=max(len(ep_options) - 1, 0), key="pm_ep")
        target = ep_map[ep_choice]
        info = target["_raw"]
        st.caption(f"Episode Path: {info['path']}")
        if st.session_state["profile_id"] == "single_video":
            title_cols = st.columns([2.4, 0.8])
            current_title = single_video_title(info["path"])
            title_key = f"single_video_pm_title_{info['ep']}"
            title_cols[0].text_input("YouTube Title", value=current_title, key=title_key)
            if title_cols[1].button("儲存 Title", key=f"single_video_pm_save_title_{info['ep']}", use_container_width=True):
                saved_path = save_single_video_info(info["path"], st.session_state.get(title_key, ""))
                st.success(f"已儲存 Title：{saved_path}")
                st.rerun()
        if st.session_state["profile_id"] == "story":
            render_story_summary(read_json_file(story_selected_json_path(info["path"])) or {})
            render_story_outline_result(info["path"])
            render_story_script_result(info["path"])
            render_story_characters_result(info["path"])
            render_story_locations_result(info["path"])
            render_story_tts_lines_result(info["path"])
            render_story_subtitles_result(info["path"])
            render_story_storyboard_result(info["path"])

        stage2subs, sub2stage, exec_mode, manual_labels, publish_substep, storyboard_manual_substep = get_pipeline_mapping(st.session_state["profile_id"])
        stage_names = {str(s.get('id')): s.get('name','') for s in (stage_cfg.get('stages') or [])}
        auto_substep_ids = [s for s in SUBSTEP_IDS if s != publish_substep]
        max_auto_substep = auto_substep_ids[-1] if auto_substep_ids else SUBSTEP_MAX

        if st.session_state["profile_id"] in ("vocab", "single_video"):
            storyboard_tab_name = "修改分鏡/替換單字卡" if st.session_state["profile_id"] == "vocab" else "修改分鏡/替換圖片"
            tab_stage, tab_sub, tab_prerender_pm, tab_prompts_pm, tab_storyboard_pm = st.tabs(
                [f"階段控制 ({STAGE_RANGE_LABEL})", f"子步驟 ({SUBSTEP_MIN}–{SUBSTEP_MAX})", "預渲染預覽", "Prompts", storyboard_tab_name]
            )
            tab_story_voice_pm = None
        elif st.session_state["profile_id"] == "story":
            tab_stage, tab_sub, tab_story_voice_pm, tab_prerender_pm, tab_storyboard_pm = st.tabs(
                [f"階段控制 ({STAGE_RANGE_LABEL})", f"子步驟 ({SUBSTEP_MIN}–{SUBSTEP_MAX})", "語音設定 / 4.2", "預渲染預覽", "修改分鏡"]
            )
            tab_prompts_pm = None
        else:
            tab_stage, tab_sub, tab_prerender_pm = st.tabs(
                [f"階段控制 ({STAGE_RANGE_LABEL})", f"子步驟 ({SUBSTEP_MIN}–{SUBSTEP_MAX})", "預渲染預覽"]
            )
            tab_story_voice_pm = None
            tab_prompts_pm = None
            tab_storyboard_pm = None
        if st.session_state.pop("pipeline_open_tab", None) == "storyboard":
            if st.session_state["profile_id"] == "story":
                storyboard_tab_label = "修改分鏡"
            elif st.session_state["profile_id"] == "single_video":
                storyboard_tab_label = "修改分鏡/替換圖片"
            else:
                storyboard_tab_label = "修改分鏡/替換單字卡"
            st.info(f"請使用「{storyboard_tab_label}」分頁進行調整。")

        with tab_stage:
            auto_refresh_stage = st.toggle(
                "自動刷新執行中的 Stage Log",
                value=True,
                key="pm_stage_auto_refresh",
                help="有 Stage 執行中時，每 2 秒自動更新一次狀態與 Log。",
            )
            selected_stage = st.selectbox("選擇要執行的階段", STAGE_IDS, key="pm_stage")
            any_running_stage = any(bool(runner.get_stage_state(info, stage_no).get("running")) for stage_no in STAGE_IDS)
            stage_refresh_interval = 2 if (auto_refresh_stage and any_running_stage) else None
            stage_running_prev_key = f"pm_stage_running_prev_{info['ep']}_{st.session_state['profile_id']}"
            if any_running_stage:
                st.session_state[stage_running_prev_key] = True

            @st.fragment(run_every=stage_refresh_interval)
            def render_stage_panel():
                latest_stage_running = any(bool(runner.get_stage_state(info, stage_no).get("running")) for stage_no in STAGE_IDS)
                if st.session_state.get(stage_running_prev_key) and not latest_stage_running:
                    st.session_state[stage_running_prev_key] = False
                    st.rerun()
                st.session_state[stage_running_prev_key] = latest_stage_running
                saved_status = read_status(info["path"])
                saved_stage_statuses = (saved_status.get("stages") or {}) if isinstance(saved_status, dict) else {}
                cur = infer_stage_statuses_for_profile(info["path"], st.session_state["profile_id"], prev=None)
                stage_states = {stage_no: runner.get_stage_state(info, stage_no) for stage_no in STAGE_IDS}

                cols = st.columns(max(len(STAGE_IDS), 1))
                for i, stage_no in enumerate(STAGE_IDS):
                    with cols[i]:
                        label = cur.get(stage_no, "pending")
                        if saved_stage_statuses.get(stage_no) == "error" and label != "done":
                            label = "error"
                        if stage_states[stage_no].get("running"):
                            label = "running"
                        st.metric(label=f"Stage {stage_no}", value=label)

                st.divider()
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    st.caption(f"目前選擇：Stage {selected_stage}")
                with c2:
                    selected_state = stage_states.get(selected_stage, {})
                    run_btn = st.button(
                        "執行中..." if selected_state.get("running") else "執行所選階段",
                        key="pm_run",
                        disabled=bool(selected_state.get("running")),
                    )
                with c3:
                    if selected_state.get("running"):
                        current_step = get_stage_current_step(info["path"], selected_stage)
                        if current_step:
                            st.write(
                                f"Stage {selected_stage} 背景執行中，PID {selected_state.get('pid')}，目前步驟：{current_step}"
                            )
                        else:
                            st.write(f"Stage {selected_stage} 背景執行中，PID {selected_state.get('pid')}")
                    else:
                        st.write("「執行」後可於下方日誌檢視輸出。")

                if run_btn:
                    res = runner.start_stage(info, selected_stage)
                    if res.get("ok"):
                        st.success(f"Stage {selected_stage} 已啟動。log: {res.get('log')}")
                        st.rerun()
                    else:
                        st.error(f"Stage {selected_stage} 啟動失敗：{res.get('message')}")

            @st.fragment(run_every=stage_refresh_interval)
            def render_stage_logs():
                latest_stage_running = any(bool(runner.get_stage_state(info, stage_no).get("running")) for stage_no in STAGE_IDS)
                if st.session_state.get(stage_running_prev_key) and not latest_stage_running:
                    st.session_state[stage_running_prev_key] = False
                    st.rerun()
                st.session_state[stage_running_prev_key] = latest_stage_running
                stage_states = {stage_no: runner.get_stage_state(info, stage_no) for stage_no in STAGE_IDS}
                st.caption("Stage Log 會在有任務執行中時自動刷新。")
                st.button("只刷新 Stage Log", key="refresh_stage_logs_fragment")
                with st.expander("終端機日誌面板 (Log Viewer)", expanded=False):
                    log_tabs = st.tabs([f"Stage {stage_no}" for stage_no in STAGE_IDS])
                    for stage_no, tab in zip(STAGE_IDS, log_tabs):
                        with tab:
                            lp = log_file_for_stage(info["path"], stage_no)
                            state = stage_states.get(stage_no, {})
                            if state.get("running"):
                                st.caption(f"背景執行中，PID {state.get('pid')}")
                            if lp.exists():
                                st.download_button("下載 log", data=lp.read_bytes(), file_name=lp.name, key=f"dl_{stage_no}")
                                st.code(lp.read_text(encoding="utf-8", errors="ignore")[-5000:])
                            else:
                                st.caption("尚無此階段的日誌。")

            render_stage_panel()
            render_stage_logs()

        with tab_sub:
            st.caption(f"在此直接執行 {SUBSTEP_MIN}–{max_auto_substep}（第{publish_substep}發布請到 Studio & Publisher）。")
            if st.session_state["profile_id"] == "vocab":
                subtitle_manual_substep = "13"
            elif st.session_state["profile_id"] == "single_video":
                subtitle_manual_substep = "4"
            else:
                subtitle_manual_substep = "11"
            auto_refresh_sub = st.toggle(
                "自動刷新執行中的子步驟 Log",
                value=True,
                key="pm_sub_auto_refresh",
                help="有子步驟執行中時，每 2 秒自動更新一次狀態與 Log。",
            )
            vocab_provider_settings = {}
            vocab_model_settings = {}
            vocab_extra_provider_settings = {}
            vocab_extra_model_settings = {}
            vocab_image_provider_settings = {}
            vocab_image_model_settings = {}
            vocab_image_model_step_settings = {}
            vocab_animation_settings = {}
            if st.session_state["profile_id"] == "vocab":
                vocab_provider_settings = load_profile_llm_provider_settings("vocab")
                vocab_model_settings = load_profile_llm_model_settings("vocab")
                vocab_extra_provider_settings = load_profile_llm_extra_provider_settings("vocab")
                vocab_extra_model_settings = load_profile_llm_extra_model_settings("vocab")
                vocab_image_provider_settings = load_profile_image_provider_settings("vocab")
                vocab_image_model_settings = load_profile_image_model_settings("vocab")
                vocab_image_model_step_settings = load_profile_image_model_step_settings("vocab")
                vocab_animation_settings = load_profile_animation_provider_settings("vocab")
                st.caption(
                    "文字 LLM Provider/Model 與 No.16 圖片 Provider 會在各步驟列中分開設定並儲存；各步驟可依任務使用不同 model。"
                )
            vocab_text_llm_scripts = {
                "scripts/generate_vocab_content.py",
                "scripts/generate_cloze_quiz.py",
                "scripts/llm_subtitle_reviewer.py",
                "scripts/llm_director.py",
                "scripts/generate_youtube_meta.py",
            }
            vocab_image_provider_scripts = {
                "scripts/gen_matched_preview_v4.py",
            }
            any_running_sub = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in SUBSTEP_IDS)
            refresh_interval = 2 if (auto_refresh_sub and any_running_sub) else None
            sub_running_prev_key = f"pm_sub_running_prev_{info['ep']}_{st.session_state['profile_id']}"
            if any_running_sub:
                st.session_state[sub_running_prev_key] = True
            if not any_running_sub:
                st.caption("目前沒有子步驟執行中；此面板不會自動刷新。")

            @st.fragment(run_every=refresh_interval)
            def render_pipeline_substeps():
                st.caption("只會刷新此面板，不會切回 Dashboard。")
                st.button("只刷新子步驟狀態", key="refresh_substeps_fragment")
                latest_sub_running = any(bool(runner.get_substep_state(info, ns).get("running")) for ns in SUBSTEP_IDS)
                if st.session_state.get(sub_running_prev_key) and not latest_sub_running:
                    st.session_state[sub_running_prev_key] = False
                    st.rerun()
                st.session_state[sub_running_prev_key] = latest_sub_running
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
                        if (
                            (st.session_state["profile_id"] == "vocab" and ns == "10")
                            or (st.session_state["profile_id"] == "single_video" and ns == "1")
                        ):
                            render_vocab_audio_upload_control(info["path"], ns, bool(row.get("ok")))
                        elif (
                            (st.session_state["profile_id"] == "vocab" and ns == "17")
                            or (st.session_state["profile_id"] == "single_video" and ns == "7")
                        ):
                            render_vocab_outro_control(info["path"], ns, bool(row.get("ok")))
                        elif ns == publish_substep:
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
                            script_name = str(step_def.get("script") or "").replace("\\", "/")
                            if st.session_state["profile_id"] == "vocab" and script_name in vocab_text_llm_scripts:
                                provider_options = list(VOCAB_TEXT_LLM_PROVIDER_LABELS.keys())
                                saved_provider = str(
                                    vocab_provider_settings.get(ns)
                                    or VOCAB_TEXT_LLM_STEP_DEFAULTS.get(ns)
                                    or "auto"
                                )
                                if saved_provider not in provider_options:
                                    saved_provider = "auto"
                                provider_choice = st.selectbox(
                                    "文字 LLM Provider",
                                    provider_options,
                                    index=provider_options.index(saved_provider),
                                    key=f"pm_vocab_llm_provider_{ns}",
                                    format_func=lambda value, step_no=ns: vocab_text_provider_label(value, step_no),
                                    help=f"只影響 Pipeline Manager 直接執行 No.{ns}。Stage 與排程依各自設定執行。",
                                )
                                current_step_models = {}
                                for model_provider in ("gemini", "openai", "nvidia"):
                                    current_step_models[model_provider] = st.text_input(
                                        f"{model_provider} model",
                                        value=vocab_text_model_for_step(vocab_model_settings, ns, model_provider),
                                        key=f"pm_vocab_{model_provider}_model_{ns}",
                                            help=f"執行 No.{ns} 時注入 {VOCAB_TEXT_LLM_MODEL_ENV_KEYS.get(ns, {}).get(model_provider, '對應環境變數')}。",
                                    ).strip()
                                st.caption("目前執行 model：" + vocab_text_model_summary({ns: current_step_models}, ns, provider_choice))
                                current_extra_models = {}
                                extra_call_id = ""
                                extra_model_providers = ()
                                if ns == "5":
                                    extra_call_id = "5_review"
                                    extra_model_providers = ("gemini", "openai")
                                    st.caption("No.5 review LLM 是獨立呼叫，model 不跟題目生成共用。")
                                elif ns == "12":
                                    extra_call_id = "12_nvidia_fallback"
                                    extra_model_providers = ("nvidia",)
                                    st.caption("No.12 NVIDIA fallback 可用逗號分隔多個模型，會依序重試；model 不跟主 review 共用。")
                                if extra_call_id:
                                    if extra_call_id in VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS:
                                        extra_provider_options = list(VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS.keys())
                                        saved_extra_provider = str(
                                            vocab_extra_provider_settings.get(extra_call_id)
                                            or VOCAB_TEXT_LLM_EXTRA_PROVIDER_DEFAULTS[extra_call_id]
                                        )
                                        if saved_extra_provider not in extra_provider_options:
                                            saved_extra_provider = "auto"
                                        current_extra_provider = st.selectbox(
                                            f"{extra_call_id} provider",
                                            extra_provider_options,
                                            index=extra_provider_options.index(saved_extra_provider),
                                            key=f"pm_vocab_{extra_call_id}_provider_{ns}",
                                            format_func=lambda value: VOCAB_TEXT_LLM_EXTRA_PROVIDER_LABELS.get(value, value),
                                            help=f"執行 No.{ns} 額外 LLM 呼叫時注入 {VOCAB_TEXT_LLM_EXTRA_PROVIDER_ENV_KEYS[extra_call_id]}。",
                                        )
                                    else:
                                        current_extra_provider = ""
                                    extra_cols = st.columns(len(extra_model_providers))
                                    for extra_col, model_provider in zip(extra_cols, extra_model_providers):
                                        with extra_col:
                                            current_extra_models[model_provider] = st.text_input(
                                                f"{extra_call_id} {model_provider} model",
                                                value=vocab_text_extra_model_for_call(vocab_extra_model_settings, extra_call_id, model_provider),
                                                key=f"pm_vocab_{extra_call_id}_{model_provider}_model_{ns}",
                                                help=f"執行 No.{ns} 額外 LLM 呼叫時注入 {VOCAB_TEXT_LLM_EXTRA_MODEL_ENV_KEYS[extra_call_id][model_provider]}。",
                                            ).strip()
                                save_cols = st.columns([1, 1, 1])
                                if save_cols[0].button("儲存設定", key=f"save_vocab_llm_provider_{ns}"):
                                    vocab_provider_settings[ns] = provider_choice
                                    saved_path = save_profile_llm_provider_settings("vocab", vocab_provider_settings, vocab_model_settings)
                                    st.success(f"No.{ns} Provider 已儲存：{vocab_text_provider_label(provider_choice, ns)}")
                                    st.caption(f"設定檔：{saved_path}")
                                if save_cols[1].button("儲存模型", key=f"save_vocab_llm_model_{ns}"):
                                    vocab_model_settings[ns] = {
                                        model_provider: current_step_models.get(model_provider) or VOCAB_TEXT_LLM_MODEL_DEFAULTS.get(ns, {}).get(model_provider, "")
                                        for model_provider in ("gemini", "openai", "nvidia")
                                    }
                                    saved_path = save_profile_llm_provider_settings("vocab", vocab_provider_settings, vocab_model_settings)
                                    st.success(f"No.{ns} Model 已儲存。")
                                    st.caption(f"設定檔：{saved_path}")
                                if extra_call_id and save_cols[2].button("儲存額外呼叫模型", key=f"save_vocab_llm_extra_model_{ns}"):
                                    if current_extra_provider:
                                        vocab_extra_provider_settings[extra_call_id] = current_extra_provider
                                    vocab_extra_model_settings[extra_call_id] = {
                                        model_provider: current_extra_models.get(model_provider) or VOCAB_TEXT_LLM_EXTRA_MODEL_DEFAULTS[extra_call_id][model_provider]
                                        for model_provider in extra_model_providers
                                    }
                                    saved_path = save_profile_llm_provider_settings(
                                        "vocab",
                                        vocab_provider_settings,
                                        vocab_model_settings,
                                        extra_model_settings=vocab_extra_model_settings,
                                        extra_provider_settings=vocab_extra_provider_settings,
                                    )
                                    st.success(f"No.{ns} 額外呼叫 Model 已儲存。")
                                    st.caption(f"設定檔：{saved_path}")
                                runtime_model_settings = dict(vocab_model_settings)
                                runtime_model_settings[ns] = current_step_models
                                runtime_extra_model_settings = dict(vocab_extra_model_settings)
                                runtime_extra_provider_settings = dict(vocab_extra_provider_settings)
                                if extra_call_id and current_extra_models:
                                    runtime_extra_model_settings[extra_call_id] = current_extra_models
                                if extra_call_id and current_extra_provider:
                                    runtime_extra_provider_settings[extra_call_id] = current_extra_provider
                                step_def = apply_vocab_llm_runtime_settings(
                                    ns,
                                    step_def,
                                    provider_choice,
                                    runtime_model_settings,
                                    runtime_extra_model_settings,
                                    runtime_extra_provider_settings,
                                )
                            elif st.session_state["profile_id"] == "vocab" and script_name in vocab_image_provider_scripts:
                                provider_options = list(VOCAB_IMAGE_PROVIDER_LABELS.keys())
                                saved_provider = str(
                                    vocab_image_provider_settings.get(ns)
                                    or VOCAB_IMAGE_PROVIDER_DEFAULTS.get(ns)
                                    or "auto"
                                )
                                if saved_provider not in provider_options:
                                    saved_provider = "auto"
                                provider_choice = st.selectbox(
                                    "圖片 Provider",
                                    provider_options,
                                    index=provider_options.index(saved_provider),
                                    key=f"pm_vocab_image_provider_{ns}",
                                    format_func=lambda value: VOCAB_IMAGE_PROVIDER_LABELS.get(value, value),
                                    help=(
                                        f"No.{ns} 使用 gen_matched_preview_v4.py。auto 會先用 Gemini/Imagen，失敗後改用 OpenAI，再改用 NVIDIA；"
                                        "NVIDIA 需要設定 CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL 與 NVIDIA_API_KEY。"
                                    ),
                                )
                                current_image_model_values = {}
                                for image_provider_key, options in VOCAB_IMAGE_MODEL_OPTIONS.items():
                                    current_model = vocab_image_model_for_step(vocab_image_model_step_settings, ns, image_provider_key)
                                    model_options = list(options)
                                    if current_model not in model_options:
                                        model_options = [current_model] + model_options
                                    current_image_model_values[image_provider_key] = st.selectbox(
                                        f"{image_provider_key} image model",
                                        model_options,
                                        index=model_options.index(current_model),
                                        key=f"pm_vocab_image_model_{image_provider_key}_{ns}",
                                        help=f"執行 No.{ns} 時注入 {build_storyboard_media_env({image_provider_key: current_model}, vocab_animation_settings).get('CAP_STORYBOARD_' + image_provider_key.upper() + '_IMAGE_MODEL', '對應環境變數')}。",
                                    )
                                st.caption("目前執行圖片 model：" + vocab_image_model_summary(current_image_model_values, provider_choice))
                                if provider_choice == "nvidia" and not os.environ.get("CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL") and not os.environ.get("NVIDIA_IMAGE_BASE_URL"):
                                    st.warning("NVIDIA 圖片 Provider 尚未設定 endpoint；執行前請先在 .env 補 CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL 或 NVIDIA_IMAGE_BASE_URL。")
                                image_save_cols = st.columns([1, 1])
                                if image_save_cols[0].button("儲存圖片 Provider", key=f"save_vocab_image_provider_{ns}"):
                                    vocab_image_provider_settings[ns] = provider_choice
                                    saved_path = save_profile_llm_provider_settings(
                                        "vocab",
                                        vocab_provider_settings,
                                        vocab_model_settings,
                                        vocab_image_provider_settings,
                                        vocab_image_model_settings,
                                        vocab_animation_settings,
                                    )
                                    st.success(f"No.{ns} 圖片 Provider 已儲存：{VOCAB_IMAGE_PROVIDER_LABELS.get(provider_choice, provider_choice)}")
                                    st.caption(f"設定檔：{saved_path}")
                                if image_save_cols[1].button("儲存圖片 Models", key=f"save_vocab_image_models_{ns}"):
                                    vocab_image_model_step_settings[ns] = dict(current_image_model_values)
                                    saved_path = save_profile_llm_provider_settings(
                                        "vocab",
                                        vocab_provider_settings,
                                        vocab_model_settings,
                                        vocab_image_provider_settings,
                                        vocab_image_model_settings,
                                        vocab_animation_settings,
                                        image_model_step_settings=vocab_image_model_step_settings,
                                    )
                                    st.success("圖片 Models 已儲存。")
                                    st.caption(f"設定檔：{saved_path}")
                                step_def = apply_provider_arg(step_def, provider_choice)
                                step_env = dict(step_def.get("env") or {})
                                step_env.update(build_storyboard_media_env(current_image_model_values, vocab_animation_settings))
                                step_def["env"] = step_env
                            show_cmd = f"{step_def['type']} {step_def['script']} {' '.join(step_def['args'])}"
                            if st.button('執行中...' if is_running else '執行子步驟', key=f"run_sub_{ns}", disabled=is_running):
                                res = runner.start_substep(info, ns, step_def)
                                if res.get("ok"):
                                    st.success(f"子步驟 No.{ns} 已啟動。log: {res.get('log')}")
                                    st.rerun()
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
                                goto_storyboard_label = "前往修改分鏡" if st.session_state["profile_id"] == "story" else "前往修改分鏡/替換單字卡"
                                if st.button(goto_storyboard_label, key="goto_storyboard_13"):
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

        any_pipeline_running_for_output_tabs = (
            any(bool(runner.get_stage_state(info, stage_no).get("running")) for stage_no in STAGE_IDS)
            or any(bool(runner.get_substep_state(info, ns).get("running")) for ns in SUBSTEP_IDS)
        )
        output_tab_refresh_interval = 2 if any_pipeline_running_for_output_tabs else None
        output_tabs_running_prev_key = f"pm_output_tabs_running_prev_{info['ep']}_{st.session_state['profile_id']}"
        if any_pipeline_running_for_output_tabs:
            st.session_state[output_tabs_running_prev_key] = True

        def stop_output_tab_refresh_when_done():
            latest_running = (
                any(bool(runner.get_stage_state(info, stage_no).get("running")) for stage_no in STAGE_IDS)
                or any(bool(runner.get_substep_state(info, ns).get("running")) for ns in SUBSTEP_IDS)
            )
            if st.session_state.get(output_tabs_running_prev_key) and not latest_running:
                st.session_state[output_tabs_running_prev_key] = False
                st.rerun()
            st.session_state[output_tabs_running_prev_key] = latest_running

        with tab_prerender_pm:
            @st.fragment(run_every=output_tab_refresh_interval)
            def render_prerender_pm_fragment():
                stop_output_tab_refresh_when_done()
                st.button("重新載入最新預覽", key=f"refresh_prerender_pm_{info['ep']}_{st.session_state['profile_id']}")
                render_prerender_workspace(info["path"])

            render_prerender_pm_fragment()

        if tab_story_voice_pm is not None:
            with tab_story_voice_pm:
                render_story_voice_workspace(info, info["path"])

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
                @st.fragment(run_every=output_tab_refresh_interval)
                def render_storyboard_pm_fragment():
                    stop_output_tab_refresh_when_done()
                    label = "重新載入最新分鏡"
                    if st.session_state["profile_id"] == "vocab":
                        label = "重新載入最新分鏡/單字卡"
                    st.button(label, key=f"refresh_storyboard_pm_{info['ep']}_{st.session_state['profile_id']}")
                    render_storyboard_workspace(info, info["path"], st.session_state["profile_id"])

                render_storyboard_pm_fragment()

        with st.expander(f"{STAGE_RANGE_LABEL} 與 {SUBSTEP_MIN}–{SUBSTEP_MAX} 對應關係", expanded=False):
            substep_name_map = {str(getattr(s, "no", "")): getattr(s, "name", "") for s in substeps_map}
            map_rows = []
            for sid in STAGE_IDS:
                subs = stage2subs.get(sid, [])
                chinese = stage_names.get(sid,'')
                subs_text = ", ".join([f"{x} " + substep_name_map.get(str(x), "") for x in subs])
                map_rows.append({"Stage": f"{sid} {chinese}", "Substeps": subs_text})
            st.table(pd.DataFrame(map_rows))

elif section == "⏰ 排程執行":
    st.header("排程執行")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先建立或同步集數資料夾。")
    else:
        render_schedule_workspace(eps)

elif section == "🧩 Prompt Sources":
    st.header("Prompt 來源與模板編輯")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先建立或同步集數資料夾。")
    else:
        ep_map = {f"Ep{e['_raw']['ep']:02d} ({e['Range']})": e for e in eps}
        prompt_options = list(ep_map.keys())
        prompt_choice = st.selectbox("選擇集數", prompt_options, index=max(len(prompt_options) - 1, 0), key="prompt_sources_ep")
        target = ep_map[prompt_choice]
        info = target["_raw"]
        ep_path = info["path"]
        subtitle_path = subtitles_fixed_path(ep_path)
        meta_path = first_existing_path(youtube_meta_paths(ep_path))
        st.caption(f"Episode Path: {ep_path}")
        st.info("這一頁調整的是目前 profile 共用的 Prompt 模板。修改一次，之後同一模式下的所有集數都會共用。")
        render_prompt_source_summary(st.session_state["profile_id"])
        st.divider()
        render_prompts_workspace(
            info,
            ep_path,
            st.session_state["profile_id"],
            subtitle_path,
            meta_path,
            show_header=False,
        )

# --- FinOps ---
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
        studio_options = list(ep_map.keys())
        studio_choice = st.selectbox("選擇集數", studio_options, index=max(len(studio_options) - 1, 0), key="studio_ep")
        target = ep_map[studio_choice]
        info = target["_raw"]
        ep_path = info["path"]
        audio_path = audio_merged_path(ep_path)
        video_path = video_output_path(ep_path)
        video_candidates = publish_video_candidates(ep_path)
        cover_path = images_dir(ep_path) / "cover.png"
        cover_cute_path = images_dir(ep_path) / "cover_cute.png"
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
        c1.metric("影片", "就緒" if video_candidates or video_path.exists() else "缺少")
        c2.metric("封面", "就緒" if cover_path.exists() else "缺少")
        c3.metric("字幕", "就緒" if subtitle_path.exists() else "缺少")
        c4.metric("Metadata", "就緒" if meta_path and meta_path.exists() else "缺少")
        c5.metric("最近上傳", upload_record.get("status", "尚無"))
        st.caption(f"Episode Path: {ep_path}")

        profile_id = st.session_state["profile_id"]
        tab_preview, tab_meta, tab_publish, tab_posts, tab_logs = st.tabs(
            ["預覽", "Metadata", "發布 Videos", "發布 Posts", "Log"]
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
                if cover_cute_path.exists():
                    st.caption("AI 候選封面")
                    render_safe_image_preview(
                        cover_cute_path,
                        empty_message="尚未找到 AI 候選封面。",
                        broken_hint="AI 候選封面無法讀取",
                        caption_label="cover_cute",
                        max_width=900,
                    )
            with pv2:
                st.subheader("影片")
                preview_video_options = [item["filename"] for item in video_candidates]
                if not preview_video_options and video_path.exists():
                    preview_video_options = [video_path.name]
                preview_video_label_map = {
                    item["filename"]: (
                        f"{item['label']} - {item['filename']} "
                        f"({item['path'].stat().st_size / (1024 * 1024):.1f} MB)"
                    )
                    for item in video_candidates
                }
                default_preview_video = (
                    "final_video_with_outro.mp4"
                    if "final_video_with_outro.mp4" in preview_video_options
                    else (preview_video_options[0] if preview_video_options else "")
                )
                if preview_video_options:
                    preview_video_name = st.selectbox(
                        "預覽影片版本",
                        preview_video_options,
                        index=preview_video_options.index(default_preview_video),
                        format_func=lambda name: preview_video_label_map.get(name, name),
                        key=f"studio_preview_video_name_{info['ep']}_{st.session_state['profile_id']}",
                    )
                    preview_video_path = ep_path / "05_output" / Path(preview_video_name).name
                else:
                    preview_video_path = video_path
                if preview_video_path.exists():
                    st.caption(f"{preview_video_path.name} | {preview_video_path.stat().st_size / (1024 * 1024):.1f} MB")
                    if video_candidates:
                        st.caption("可用版本：" + "、".join(f"{item['label']}({item['filename']})" for item in video_candidates))
                    with st.expander("展開影片預覽", expanded=False):
                        st.video(str(preview_video_path))
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
                render_subtitle_compare_panel(
                    ep_path,
                    key_prefix=f"studio_subtitle_compare_{info['ep']}_{st.session_state['profile_id']}",
                    expanded=False,
                )
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
                        current_custom_image_path = str(selected_row.get("custom_image_path", "")).strip()
                        custom_image_path_input = custom_image_val.strip()
                        custom_image_changed = bool(custom_image_path_input) and custom_image_path_input != current_custom_image_path
                        chosen_ai_image_path = (
                            custom_image_path_input
                            if custom_image_changed
                            else (selected_image_choice_path or custom_image_path_input)
                        )
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
            meta_running = bool(meta_state.get("running"))
            video_candidates = publish_video_candidates(ep_path)
            ready_to_upload = bool(video_candidates and meta_path and meta_path.exists() and not meta_running)
            if not ready_to_upload:
                if meta_running:
                    st.warning("Metadata 仍在生成中。上傳只會使用既有 youtube_meta.json，請等待 Metadata 任務完成後再上傳。")
                else:
                    st.warning("上傳前至少需要可上傳影片與 youtube_meta.json。")
            if not subtitle_path.exists():
                st.caption("提醒：找不到校對字幕，上傳時會略過字幕。")
            upload_status = upload_record.get("status")
            if upload_status:
                st.caption(f"最近上傳狀態：{upload_status}")
            if upload_state.get("running"):
                st.info(f"上傳進行中，PID {upload_state.get('pid')}")
            token_sig, client_sig = youtube_auth_file_signatures()
            playlist_items, playlist_error = list_youtube_playlists_cached(token_sig, client_sig)
            playlist_options = [{"id": "", "title": "(不加入播放清單)", "label": "(不加入播放清單)"}] + playlist_items
            if playlist_error:
                st.error(f"YouTube 播放清單讀取失敗：{playlist_error}")
                if "invalid_grant" in playlist_error.lower() or "重新授權" in playlist_error:
                    if st.button("重新授權 YouTube", key=f"reauth_youtube_{info['ep']}"):
                        try:
                            reauthorize_youtube_for_app()
                            list_youtube_playlists_cached.clear()
                            st.success("YouTube 重新授權完成，正在重新載入播放清單。")
                            st.rerun()
                        except Exception as e:
                            st.error(f"YouTube 重新授權失敗：{e}")

            publish_defaults = studio_default_publish_settings(ep_path, int(info["ep"]), upload_record)
            publish_state_marker_key = f"studio_publish_loaded_{profile_id}"
            publish_state_marker = f"{profile_id}:{info['ep']}"

            privacy_key = f"studio_privacy_{info['ep']}"
            schedule_key = f"studio_schedule_{info['ep']}"
            publish_date_key = f"studio_publish_date_{info['ep']}"
            publish_time_key = f"studio_publish_time_{info['ep']}"
            playlist_key = f"studio_playlist_{info['ep']}"
            cover_key = f"studio_cover_name_{info['ep']}"
            video_name_key = f"studio_video_name_{info['ep']}"

            publish_dt = parse_publish_at_value(str(publish_defaults.get("publish_at", "")))
            default_playlist_id = str(publish_defaults.get("playlist_id", "")).strip()
            default_playlist_id = resolve_default_playlist_id(playlist_options, default_playlist_id)
            default_video_name = default_publish_video_name(ep_path, publish_defaults)

            if st.session_state.get(publish_state_marker_key) != publish_state_marker:
                st.session_state[privacy_key] = str(publish_defaults.get("privacy", "private") or "private")
                st.session_state[schedule_key] = bool(publish_defaults.get("schedule_on")) or bool(publish_dt)
                st.session_state[publish_date_key] = (publish_dt or datetime.now()).date()
                st.session_state[publish_time_key] = (publish_dt or datetime.now().replace(second=0, microsecond=0)).time()
                st.session_state[playlist_key] = default_playlist_id
                st.session_state[cover_key] = str(publish_defaults.get("cover_name", "cover.png") or "cover.png")
                st.session_state[video_name_key] = default_video_name
                st.session_state[publish_state_marker_key] = publish_state_marker

            privacy_opts = ["private", "unlisted", "public"]
            privacy_val = st.selectbox("隱私設定", privacy_opts, key=privacy_key)
            schedule_on = st.checkbox("排程發布", key=schedule_key)
            sched_cols = st.columns(2)
            with sched_cols[0]:
                publish_date = st.date_input("發布日期", key=publish_date_key, disabled=not schedule_on)
            with sched_cols[1]:
                publish_time = st.time_input("發布時間", key=publish_time_key, disabled=not schedule_on)

            selected_playlist_id = st.selectbox(
                "播放清單",
                options=[item["id"] for item in playlist_options],
                format_func=lambda pid: next((item["label"] for item in playlist_options if item["id"] == pid), pid or "(不加入播放清單)"),
                key=playlist_key,
            )

            cover_name_val = st.selectbox(
                "正式封面檔名",
                ["cover.png", "cover_cute.png"],
                key=cover_key,
            )
            video_options = [item["filename"] for item in video_candidates] or ["final_video.mp4"]
            current_video_name = st.session_state.get(video_name_key, default_video_name)
            if current_video_name not in video_options:
                current_video_name = video_options[0]
                st.session_state[video_name_key] = current_video_name
            video_label_map = {
                item["filename"]: (
                    f"{item['label']} - {item['filename']} "
                    f"({item['path'].stat().st_size / (1024 * 1024):.1f} MB)"
                )
                for item in video_candidates
            }
            video_name_val = st.selectbox(
                "上傳影片版本",
                video_options,
                index=video_options.index(current_video_name),
                format_func=lambda name: video_label_map.get(name, name),
                key=video_name_key,
            )
            selected_upload_video_path = ep_path / "05_output" / Path(video_name_val).name
            if selected_upload_video_path.exists():
                st.caption(f"將上傳：{selected_upload_video_path}")

            publish_at = f"{publish_date.isoformat()} {publish_time.strftime('%H:%M')}" if schedule_on else ""
            save_cols = st.columns([1, 1.4, 2.2])
            with save_cols[0]:
                save_publish = st.button("儲存發布設定", key=f"studio_save_publish_{info['ep']}")
            with save_cols[1]:
                if save_publish:
                    saved_publish_path = save_publish_settings(
                        ep_path,
                        {
                            "privacy": privacy_val,
                            "schedule_on": bool(schedule_on),
                            "publish_at": publish_at,
                            "playlist_id": selected_playlist_id,
                            "cover_name": cover_name_val,
                            "video_name": video_name_val,
                        },
                    )
                    st.success(f"已儲存發布設定：{saved_publish_path}")
                    st.rerun()
            with save_cols[2]:
                if selected_playlist_id:
                    selected_playlist_label = next((item["label"] for item in playlist_options if item["id"] == selected_playlist_id), selected_playlist_id)
                    st.caption(f"目前播放清單：{selected_playlist_label}")
                else:
                    st.caption("目前不加入播放清單。")

            upload_args = [
                "--ep",
                str(info["ep"]),
                "--privacy",
                privacy_val,
                "--cover_name",
                cover_name_val,
                "--video_name",
                Path(video_name_val).name,
            ]

            if schedule_on:
                upload_args += ["--publish_at", publish_at]

            if selected_playlist_id.strip():
                upload_args += ["--playlist_id", selected_playlist_id.strip()]
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

        with tab_posts:
            st.subheader("YouTube Posts 草稿與排程")
            st.info("YouTube Data API 目前沒有公開的 Community Posts 建立/排程 endpoint。此分頁先整理 Posts 內容與發布時程，供手動貼到 YouTube Studio。")
            cloze_rows = load_cloze_question_rows(ep_path)
            saved_post_settings = load_post_publish_settings(ep_path)
            if not cloze_rows:
                st.warning("尚未找到克漏字題目。請先執行 No.5 產生 cloze_questions.csv / json。")
            else:
                saved_selected_ids = {
                    str(item).strip()
                    for item in saved_post_settings.get("selected_question_ids", [])
                    if str(item).strip()
                }
                if not saved_selected_ids and cloze_rows:
                    saved_selected_ids = {cloze_question_id(cloze_rows[0], 1)}

                post_table_rows = []
                for idx, row in enumerate(cloze_rows, start=1):
                    qid = cloze_question_id(row, idx)
                    post_table_rows.append({
                        "發布": qid in saved_selected_ids,
                        "題號": qid,
                        "單字": str(row.get("word", "") or row.get("correct_word", "") or "").strip(),
                        "題目": str(row.get("blank_sentence", "") or row.get("question", "") or "").strip(),
                        "A": str(row.get("choice_A", "") or row.get("A", "") or "").strip(),
                        "B": str(row.get("choice_B", "") or row.get("B", "") or "").strip(),
                        "C": str(row.get("choice_C", "") or row.get("C", "") or "").strip(),
                        "D": str(row.get("choice_D", "") or row.get("D", "") or "").strip(),
                        "答案": str(row.get("correct_option", "") or "").strip(),
                    })

                edited_posts_df = st.data_editor(
                    pd.DataFrame(post_table_rows),
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    disabled=["題號", "單字", "題目", "A", "B", "C", "D", "答案"],
                    column_config={
                        "發布": st.column_config.CheckboxColumn("發布"),
                        "題目": st.column_config.TextColumn(width="large"),
                    },
                    key=f"studio_posts_questions_{info['ep']}_{profile_id}",
                )

                selected_ids = [
                    str(row.get("題號", "")).strip()
                    for row in edited_posts_df.to_dict("records")
                    if bool(row.get("發布")) and str(row.get("題號", "")).strip()
                ]
                selected_cloze_rows = [
                    row
                    for idx, row in enumerate(cloze_rows, start=1)
                    if cloze_question_id(row, idx) in set(selected_ids)
                ]

                post_publish_dt = parse_publish_at_value(str(saved_post_settings.get("publish_at", "")))
                posts_marker_key = f"studio_posts_loaded_{profile_id}"
                posts_marker = f"{profile_id}:{info['ep']}"
                posts_schedule_key = f"studio_posts_schedule_{info['ep']}"
                posts_date_key = f"studio_posts_date_{info['ep']}"
                posts_time_key = f"studio_posts_time_{info['ep']}"
                posts_text_key = f"studio_posts_text_{info['ep']}"

                generated_post_text = build_cloze_post_text(int(info["ep"]), selected_cloze_rows)
                if st.session_state.get(posts_marker_key) != posts_marker:
                    st.session_state[posts_schedule_key] = bool(saved_post_settings.get("schedule_on")) or bool(post_publish_dt)
                    st.session_state[posts_date_key] = (post_publish_dt or datetime.now()).date()
                    st.session_state[posts_time_key] = (post_publish_dt or datetime.now().replace(second=0, microsecond=0)).time()
                    st.session_state[posts_text_key] = str(saved_post_settings.get("post_text", "") or generated_post_text)
                    st.session_state[posts_marker_key] = posts_marker

                if st.button("用勾選題目重建貼文", key=f"studio_posts_rebuild_{info['ep']}"):
                    st.session_state[posts_text_key] = generated_post_text
                    st.rerun()

                posts_schedule_on = st.checkbox("排程發布 Posts", key=posts_schedule_key)
                posts_sched_cols = st.columns(2)
                with posts_sched_cols[0]:
                    posts_publish_date = st.date_input("Posts 發布日期", key=posts_date_key, disabled=not posts_schedule_on)
                with posts_sched_cols[1]:
                    posts_publish_time = st.time_input("Posts 發布時間", key=posts_time_key, disabled=not posts_schedule_on)

                st.text_area(
                    "Posts 文字草稿",
                    key=posts_text_key,
                    height=260,
                    help="可直接複製到 YouTube Studio 的 Community Post。",
                )
                render_copy_button(
                    st.session_state.get(posts_text_key, ""),
                    key=f"studio_posts_copy_{info['ep']}_{profile_id}",
                    label="快速複製 Posts 草稿",
                )

                posts_publish_at = (
                    f"{posts_publish_date.isoformat()} {posts_publish_time.strftime('%H:%M')}"
                    if posts_schedule_on else ""
                )
                posts_save_cols = st.columns([1, 2.6])
                with posts_save_cols[0]:
                    save_posts = st.button("儲存 Posts 設定", key=f"studio_save_posts_{info['ep']}")
                with posts_save_cols[1]:
                    st.caption(f"已選 {len(selected_ids)} 題；發布時間以 Asia/Taipei 記錄。")
                if save_posts:
                    saved_posts_path = save_post_publish_settings(
                        ep_path,
                        {
                            "platform": "youtube_community_posts",
                            "api_available": False,
                            "api_note": "YouTube Data API has no public endpoint for creating or scheduling Community Posts.",
                            "schedule_on": bool(posts_schedule_on),
                            "publish_at": posts_publish_at,
                            "selected_question_ids": selected_ids,
                            "post_text": st.session_state.get(posts_text_key, ""),
                        },
                    )
                    st.success(f"已儲存 Posts 設定：{saved_posts_path}")
                    st.rerun()

                if saved_post_settings:
                    st.divider()
                    st.caption(f"目前儲存檔：{post_publish_settings_path(ep_path)}")
                    st.json(saved_post_settings)

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
