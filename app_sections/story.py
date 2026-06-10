import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict
import pandas as pd
import streamlit as st

ROOT: Path | None = None
WS_ROOT: str = "workspace"
runner = None


def _not_configured(*args, **kwargs):
    raise RuntimeError("Story section is not configured. Call configure_story_section first.")


read_json_file = _not_configured
read_text_file = _not_configured
parse_srt_entries = _not_configured
seconds_to_label = _not_configured


def configure_story_section(root: Path, workspace_root: str, stage_runner, deps: dict) -> None:
    global ROOT, WS_ROOT, runner
    global read_json_file, read_text_file, parse_srt_entries, seconds_to_label

    ROOT = root
    WS_ROOT = workspace_root
    runner = stage_runner
    read_json_file = deps["read_json_file"]
    read_text_file = deps["read_text_file"]
    parse_srt_entries = deps["parse_srt_entries"]
    seconds_to_label = deps["seconds_to_label"]


def _root() -> Path:
    if ROOT is None:
        raise RuntimeError("Story section is not configured. Call configure_story_section first.")
    return ROOT


def _runner():
    if runner is None:
        raise RuntimeError("Story section is not configured. Call configure_story_section first.")
    return runner


def story_outline_options_path(ep_path: Path) -> Path:
    return ep_path / "01_preproduction" / "story_outline_options.json"

def story_selected_json_path(ep_path: Path) -> Path:
    return ep_path / "01_preproduction" / "selected_story.json"

def story_selected_md_path(ep_path: Path) -> Path:
    return ep_path / "01_preproduction" / "selected_story.md"

def story_outline_json_path(ep_path: Path) -> Path:
    return ep_path / "02_story" / "story_outline.json"

def story_outline_md_path(ep_path: Path) -> Path:
    return ep_path / "02_story" / "story_outline.md"

def story_script_json_path(ep_path: Path) -> Path:
    return ep_path / "02_story" / "story_script.json"

def story_script_md_path(ep_path: Path) -> Path:
    return ep_path / "02_story" / "story_script.md"

def story_tts_text_path(ep_path: Path) -> Path:
    return ep_path / "02_story" / "story_text_for_tts.txt"

def story_characters_json_path(ep_path: Path) -> Path:
    return ep_path / "03_characters_scenes" / "characters.json"

def story_locations_json_path(ep_path: Path) -> Path:
    return ep_path / "03_characters_scenes" / "locations.json"

def story_tts_lines_json_path(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "tts_lines.json"

def story_tts_lines_csv_path(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "tts_lines.csv"

def story_voice_cast_json_path(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "voice_cast.json"

def story_voice_segments_dir(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "voice_segments"

def story_voice_previews_dir(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "voice_previews"

def story_merged_audio_path(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "story_audio.m4a"

def story_subtitles_srt_path(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "story_subtitles.srt"

def story_subtitles_json_path(ep_path: Path) -> Path:
    return ep_path / "04_audio_subtitles" / "story_subtitles.json"

def story_storyboard_csv_path(ep_path: Path) -> Path:
    return ep_path / "05_storyboards" / "storyboard.csv"

def story_storyboard_json_path(ep_path: Path) -> Path:
    return ep_path / "05_storyboards" / "storyboard.json"

STORY_OPENAI_TTS_VOICES = [
    "marin",
    "cedar",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]

def default_story_voice_cast(tts_data: dict) -> dict:
    speakers = []
    for line in tts_data.get("lines") or []:
        speaker = str(line.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    voice_cycle = ["marin", "coral", "fable", "nova", "sage", "shimmer", "verse", "cedar"]
    voices = {}
    for idx, speaker in enumerate(speakers):
        if speaker == "旁白":
            voice = "marin"
            instructions = "用溫暖、清楚、慢速的繁體中文說故事，像10歲以下小朋友的睡前童話旁白。"
        else:
            voice = voice_cycle[idx % len(voice_cycle)]
            instructions = f"用適合兒童童話角色「{speaker}」的繁體中文聲音說話，語氣自然、清楚、情緒明確，避免誇張尖銳。"
        voices[speaker] = {
            "provider": "openai",
            "voice": voice,
            "instructions": instructions,
        }
    return {
        "provider": "openai",
        "model": "gpt-4o-mini-tts",
        "response_format": "mp3",
        "voice_segments_dir": "04_audio_subtitles/voice_segments",
        "pause_ms": {
            "default": 350,
            "comma": 250,
            "sentence": 500,
            "question": 700,
            "paragraph": 900,
        },
        "voices": voices,
    }

def load_story_outline_options(ep_path: Path) -> list[dict]:
    data = read_json_file(story_outline_options_path(ep_path))
    if data is None:
        data = read_json_file(ep_path / "01_preproduction" / "story_brainstorm_options.json")
    if isinstance(data, dict):
        return list(data.get("options") or [])
    if isinstance(data, list):
        return data
    return []

def render_story_summary(story: dict, *, title: str = "已選定故事") -> None:
    if not story:
        st.warning("尚未選定本次故事。請先回 Dashboard 的「故事大綱選擇」確認一個故事。")
        return
    st.info(
        f"{title}：{story.get('id', '')}｜{story.get('title', '')}\n\n"
        f"類型：{story.get('genre', '')}｜目標年齡：{story.get('target_age', '')}\n\n"
        f"故事一句話：{story.get('logline', '')}\n\n"
        f"正向寓意：{story.get('moral', '')}"
    )

def render_story_outline_result(ep_path: Path) -> None:
    outline = read_json_file(story_outline_json_path(ep_path)) or {}
    outline_md = read_text_file(story_outline_md_path(ep_path), "")
    if not outline and not outline_md:
        st.warning("尚未建立故事大綱及段落。請執行 Stage 2 或子步驟 2.1。")
        return

    st.subheader("2.1 建立故事大綱及段落")
    if outline:
        st.info(
            f"故事：{outline.get('title', '')}\n\n"
            f"目標年齡：{outline.get('target_age', '')}｜調性：{outline.get('tone', '')}\n\n"
            f"故事一句話：{outline.get('logline', '')}\n\n"
            f"正向寓意：{outline.get('moral', '')}"
        )
        rows = []
        for paragraph in outline.get("paragraphs") or []:
            rows.append(
                {
                    "段落": paragraph.get("paragraph_id", ""),
                    "標題": paragraph.get("title", ""),
                    "目的": paragraph.get("purpose", ""),
                    "劇情": paragraph.get("plot", ""),
                    "旁白重點": paragraph.get("narration_focus", ""),
                    "對話重點": paragraph.get("dialogue_focus", ""),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("查看 story_outline.md", expanded=not bool(outline)):
        st.markdown(outline_md or "story_outline.md 尚未建立。")

def render_story_script_result(ep_path: Path) -> None:
    script_data = read_json_file(story_script_json_path(ep_path)) or {}
    script_md = read_text_file(story_script_md_path(ep_path), "")
    tts_text = read_text_file(story_tts_text_path(ep_path), "")
    st.subheader("2.2 產出各段落旁白與角色對話")
    if not script_data and not script_md:
        st.warning("尚未產出各段落旁白與角色對話。請執行 Stage 2 或子步驟 2.2。")
        return

    if script_data:
        st.info(
            f"故事：{script_data.get('title', '')}\n\n"
            f"目標年齡：{script_data.get('target_age', '')}｜調性：{script_data.get('tone', '')}\n\n"
            f"正向寓意：{script_data.get('moral', '')}"
        )
        for paragraph in script_data.get("paragraphs") or []:
            label = f"{paragraph.get('paragraph_id', '')} {paragraph.get('title', '')}".strip()
            with st.expander(label or "段落", expanded=False):
                storybook_text = str(paragraph.get("storybook_text", "")).strip()
                if storybook_text:
                    st.markdown("**可直接生成語音的正文**")
                    for block in storybook_text.split("\n\n"):
                        if block.strip():
                            st.write(block.strip())
                st.markdown("**旁白**")
                for line in paragraph.get("narration") or []:
                    st.write(f"- {line}")
                st.markdown("**角色對話**")
                dialogue_rows = []
                for item in paragraph.get("dialogues") or []:
                    dialogue_rows.append(
                        {
                            "角色": item.get("speaker", ""),
                            "台詞": item.get("text", ""),
                            "語氣": item.get("voice_hint", ""),
                        }
                    )
                if dialogue_rows:
                    st.dataframe(pd.DataFrame(dialogue_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("此段尚無角色對話。")
                source_plot = paragraph.get("source_plot", "")
                if source_plot:
                    st.caption(f"來源劇情：{source_plot}")
    with st.expander("查看 story_script.md", expanded=False):
        st.markdown(script_md or "story_script.md 尚未建立。")
    with st.expander("查看 story_text_for_tts.txt", expanded=False):
        st.text_area(
            "完整語音正文",
            value=tts_text,
            height=360,
            key=f"story_tts_text_{str(ep_path)}",
            disabled=True,
        )

def render_story_characters_result(ep_path: Path) -> None:
    data = read_json_file(story_characters_json_path(ep_path)) or {}
    st.subheader("3.1 角色人物塑造")
    if not data:
        st.warning("尚未產出角色人物 JSON。請執行 Stage 3 或子步驟 3.1。")
        return
    visual_system = data.get("visual_system") or {}
    st.info(
        f"故事：{data.get('story_title', '')}\n\n"
        f"目標年齡：{data.get('target_age', '')}\n\n"
        f"整體風格：{visual_system.get('global_style', '')}\n\n"
        f"一致性規則：{visual_system.get('continuity_rule', '')}"
    )
    for character in data.get("characters") or []:
        label = f"{character.get('name', '')}｜{character.get('role', '')}｜{character.get('character_type', '')}"
        with st.expander(label, expanded=False):
            personality = character.get("personality") or {}
            visual = character.get("visual_design") or {}
            image_gen = character.get("image_generation") or {}
            st.write(f"角色 ID：{character.get('character_id', '')}")
            st.write(f"核心特質：{'、'.join(personality.get('core_traits') or [])}")
            st.write(f"情緒弧線：{personality.get('emotional_arc', '')}")
            st.write(f"聲音風格：{personality.get('voice_style', '')}")
            st.write(f"視覺風格：{visual.get('style', '')}")
            st.write(f"色彩：{'、'.join(visual.get('color_palette') or [])}")
            st.write(f"預設表情：{visual.get('default_expression', '')}")
            st.write(f"一致性標籤：{'、'.join(visual.get('consistency_tags') or [])}")
            st.markdown("**AI 人物產圖 Prompt**")
            st.code(image_gen.get("prompt_en", ""), language="text")
            st.markdown("**Negative Prompt**")
            st.code(image_gen.get("negative_prompt_en", ""), language="text")
    with st.expander("查看 characters.json", expanded=False):
        st.json(data)

def render_story_locations_result(ep_path: Path) -> None:
    data = read_json_file(story_locations_json_path(ep_path)) or {}
    st.subheader("3.2 場景塑造")
    if not data:
        st.warning("尚未產出場景 JSON。請執行 Stage 3 或子步驟 3.2。")
        return
    scene_world = data.get("scene_world") or {}
    st.info(
        f"故事：{data.get('story_title', '')}\n\n"
        f"目標年齡：{data.get('target_age', '')}\n\n"
        f"整體場景風格：{scene_world.get('global_style', '')}\n\n"
        f"一致性規則：{scene_world.get('continuity_rule', '')}"
    )
    for location in data.get("locations") or []:
        label = f"{location.get('name', '')}｜{location.get('story_role', '')}"
        with st.expander(label, expanded=False):
            visual = location.get("visual_design") or {}
            image_gen = location.get("image_generation") or {}
            st.write(f"場景 ID：{location.get('location_id', '')}")
            st.write(f"對應段落：{'、'.join(location.get('linked_paragraph_ids') or [])}")
            st.write(f"對應角色：{'、'.join(location.get('linked_characters') or [])}")
            st.write(f"視覺風格：{visual.get('style', '')}")
            st.write(f"情緒氛圍：{visual.get('mood', '')}")
            st.write(f"時間光線：{visual.get('time_of_day', '')}")
            st.write(f"色彩：{'、'.join(visual.get('color_palette') or [])}")
            st.write(f"關鍵道具/元素：{'、'.join(visual.get('key_props') or [])}")
            st.write(f"構圖備註：{visual.get('composition_notes', '')}")
            st.write(f"一致性標籤：{'、'.join(visual.get('continuity_tags') or [])}")
            st.markdown("**AI 場景產圖 Prompt**")
            st.code(image_gen.get("prompt_en", ""), language="text")
            st.markdown("**Negative Prompt**")
            st.code(image_gen.get("negative_prompt_en", ""), language="text")
    with st.expander("查看 locations.json", expanded=False):
        st.json(data)

def render_story_tts_lines_result(ep_path: Path) -> None:
    data = read_json_file(story_tts_lines_json_path(ep_path)) or {}
    st.subheader("4.1 依內容及角色產生各段句及對話列表")
    if not data:
        st.warning("尚未產出 TTS 對話列表。請執行 Stage 4 或子步驟 4.1。")
        return

    lines = data.get("lines") or []
    st.info(
        f"故事：{data.get('story_title', '')}\n\n"
        f"總句數：{data.get('line_count', len(lines))}｜"
        f"旁白：{data.get('narration_count', 0)}｜"
        f"角色對話：{data.get('dialogue_count', 0)}"
    )

    rows = []
    for item in lines:
        rows.append(
            {
                "序號": item.get("sequence", ""),
                "段落": item.get("paragraph_id", ""),
                "段落標題": item.get("paragraph_title", ""),
                "類型": "旁白" if item.get("line_type") == "narration" else "角色對話",
                "角色": item.get("speaker", ""),
                "角色ID": item.get("character_id", ""),
                "聲音風格": item.get("voice_style", ""),
                "語氣": item.get("voice_hint", ""),
                "語音內容": item.get("text", ""),
                "字數": item.get("text_length", ""),
                "狀態": item.get("tts_status", ""),
                "音檔": item.get("audio_path", ""),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("tts_lines.json 目前沒有可顯示的句子。")

    csv_path = story_tts_lines_csv_path(ep_path)
    if csv_path.exists():
        st.caption(f"CSV：{csv_path}")
    with st.expander("查看 tts_lines.json", expanded=False):
        st.json(data)

def has_openai_api_key_configured() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    env_path = _root() / ".env"
    if not env_path.exists():
        return False
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
                return True
    except Exception:
        return False
    return False

def render_story_voice_workspace(info: Dict, ep_path: Path) -> None:
    st.subheader("語音設定 / No.4.2")
    tts_data = read_json_file(story_tts_lines_json_path(ep_path)) or {}
    if not tts_data:
        st.warning("尚未建立 4.1 TTS 對話列表。請先執行 No.4.1。")
        return

    cast_path = story_voice_cast_json_path(ep_path)
    voice_cast = read_json_file(cast_path) or default_story_voice_cast(tts_data)
    voice_cast.setdefault("voices", {})
    voice_cast.setdefault("pause_ms", {})

    speakers = []
    for line in tts_data.get("lines") or []:
        speaker = str(line.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    for speaker in speakers:
        voice_cast["voices"].setdefault(
            speaker,
            default_story_voice_cast({"lines": [{"speaker": speaker}]}).get("voices", {}).get(speaker, {}),
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TTS 句數", int(tts_data.get("line_count", len(tts_data.get("lines") or []))))
    m2.metric("角色/旁白", len(speakers))
    m3.metric("已產生片段", len(list(story_voice_segments_dir(ep_path).glob("*.*"))) if story_voice_segments_dir(ep_path).exists() else 0)
    m4.metric("合併音檔", "已產生" if story_merged_audio_path(ep_path).exists() else "尚未")

    if not has_openai_api_key_configured():
        st.warning("尚未偵測到 OPENAI_API_KEY。儲存設定可以先做，但執行 4.2 前需要在環境變數或 .env 設定 OPENAI_API_KEY。")

    with st.form(f"story_voice_cast_form_{str(ep_path)}"):
        st.markdown("**基本設定**")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            provider = st.selectbox(
                "TTS Provider",
                ["openai"],
                index=0,
                key=f"voice_provider_{str(ep_path)}",
            )
        with c2:
            model = st.text_input(
                "模型",
                value=str(voice_cast.get("model", "gpt-4o-mini-tts")),
                key=f"voice_model_{str(ep_path)}",
            )
        with c3:
            response_format = st.selectbox(
                "輸出格式",
                ["mp3"],
                index=0,
                key=f"voice_format_{str(ep_path)}",
                help="目前合併流程固定使用 mp3 片段，再輸出為 story_audio.m4a。",
            )

        st.markdown("**自然停頓設定（毫秒）**")
        pause_cfg = voice_cast.get("pause_ms") or {}
        p1, p2, p3, p4, p5 = st.columns(5)
        pause_default = p1.number_input("一般", min_value=0, max_value=3000, value=int(pause_cfg.get("default", 350)), step=50)
        pause_comma = p2.number_input("逗號", min_value=0, max_value=3000, value=int(pause_cfg.get("comma", 250)), step=50)
        pause_sentence = p3.number_input("句號/驚嘆", min_value=0, max_value=3000, value=int(pause_cfg.get("sentence", 500)), step=50)
        pause_question = p4.number_input("問句", min_value=0, max_value=3000, value=int(pause_cfg.get("question", 700)), step=50)
        pause_paragraph = p5.number_input("段落切換", min_value=0, max_value=5000, value=int(pause_cfg.get("paragraph", 900)), step=50)

        st.markdown("**角色聲音設定**")
        new_voices = {}
        for speaker in speakers:
            cfg = voice_cast.get("voices", {}).get(speaker, {})
            with st.expander(speaker, expanded=(speaker == "旁白")):
                vc1, vc2 = st.columns([1, 3])
                current_voice = str(cfg.get("voice", "marin"))
                with vc1:
                    selected_voice = st.selectbox(
                        "聲音",
                        STORY_OPENAI_TTS_VOICES,
                        index=STORY_OPENAI_TTS_VOICES.index(current_voice) if current_voice in STORY_OPENAI_TTS_VOICES else 0,
                        key=f"voice_select_{str(ep_path)}_{speaker}",
                    )
                with vc2:
                    instructions = st.text_area(
                        "聲音指令",
                        value=str(cfg.get("instructions", "")),
                        height=100,
                        key=f"voice_instructions_{str(ep_path)}_{speaker}",
                    )
                sample_line = next((line for line in tts_data.get("lines") or [] if line.get("speaker") == speaker), {})
                sample_text = str(sample_line.get("text", "")).strip()
                if sample_text:
                    st.caption(f"範例台詞：{sample_text[:120]}")
                new_voices[speaker] = {
                    "provider": provider,
                    "voice": selected_voice,
                    "instructions": instructions,
                }

        save_cast = st.form_submit_button("儲存語音設定")

    if save_cast:
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "provider": provider,
            "model": model,
            "response_format": response_format,
            "voice_segments_dir": "04_audio_subtitles/voice_segments",
            "pause_ms": {
                "default": int(pause_default),
                "comma": int(pause_comma),
                "sentence": int(pause_sentence),
                "question": int(pause_question),
                "paragraph": int(pause_paragraph),
            },
            "voices": new_voices,
        }
        cast_path.parent.mkdir(parents=True, exist_ok=True)
        cast_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"已儲存：{cast_path}")
        st.rerun()

    st.divider()
    st.markdown("**預聽角色聲音**")
    preview_cols = st.columns([1, 3, 1])
    with preview_cols[0]:
        preview_speaker = st.selectbox(
            "角色",
            speakers or ["旁白"],
            key=f"story_preview_speaker_{str(ep_path)}",
        )
    sample_line = next((line for line in tts_data.get("lines") or [] if line.get("speaker") == preview_speaker), {})
    default_preview_text = str(sample_line.get("text", "")).strip() or "你好，今天我們要一起走進一個溫暖又神奇的童話故事。"
    with preview_cols[1]:
        preview_text = st.text_area(
            "預聽台詞",
            value=default_preview_text,
            height=90,
            key=f"story_preview_text_{str(ep_path)}_{preview_speaker}",
        )
    preview_request_path = ep_path / "04_audio_subtitles" / "voice_preview_request.json"
    preview_state = _runner().get_substep_state(info, "4.2_preview")
    with preview_cols[2]:
        st.write("")
        st.write("")
        if st.button(
            "產生預聽",
            disabled=bool(preview_state.get("running")) or not bool(str(preview_text).strip()),
            key=f"story_preview_btn_{info['ep']}",
        ):
            preview_request_path.parent.mkdir(parents=True, exist_ok=True)
            preview_request_path.write_text(
                json.dumps(
                    {
                        "speaker": preview_speaker,
                        "text": str(preview_text).strip(),
                        "voice_hint": "預聽聲音",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            res = _runner().start_substep(
                info,
                "4.2_preview",
                {
                    "name": "No.4.2 預聽角色聲音",
                    "type": "python",
                    "script": "scripts/story_generate_tts_audio.py",
                    "args": ["--ep", "{ep}", "--preview-request", "04_audio_subtitles/voice_preview_request.json"],
                },
            )
            if res.get("ok"):
                st.success(f"已啟動預聽：{res.get('log')}")
            else:
                st.error(f"預聽啟動失敗：{res.get('message')}")

    if preview_state.get("running"):
        st.info(f"預聽產生中，PID {preview_state.get('pid')}。")
    preview_dir = story_voice_previews_dir(ep_path)
    preview_file = preview_dir / f"{''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in preview_speaker).strip('_') or 'line'}_preview.mp3"
    preview_request = read_json_file(preview_request_path) or {}
    request_audio = preview_request.get("audio_path")
    if request_audio:
        candidate = ep_path / str(request_audio)
        if candidate.exists():
            preview_file = candidate
    if preview_file.exists():
        st.audio(str(preview_file))
        st.caption(f"預聽音檔：{preview_file}")

    st.divider()
    st.markdown("**執行語音產生**")
    state_42 = _runner().get_substep_state(info, "4.2")
    state_43 = _runner().get_substep_state(info, "4.3")
    state_44 = _runner().get_substep_state(info, "4.4")
    state_combo = _runner().get_substep_state(info, "4.2_merge")
    any_running = bool(state_42.get("running") or state_43.get("running") or state_44.get("running") or state_combo.get("running"))
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1.4])
    with b1:
        if st.button("執行 4.2 產生語音片段", disabled=any_running, key=f"run_story_42_{info['ep']}"):
            res = _runner().start_substep(
                info,
                "4.2",
                {"name": "No.4.2 呼叫文字轉語音API", "type": "python", "script": "scripts/story_generate_tts_audio.py", "args": ["--ep", "{ep}"]},
            )
            if res.get("ok"):
                st.success(f"已啟動 4.2：{res.get('log')}")
            else:
                st.error(f"4.2 啟動失敗：{res.get('message')}")
    with b2:
        if st.button("執行 4.3 合併語音", disabled=any_running, key=f"run_story_43_{info['ep']}"):
            res = _runner().start_substep(
                info,
                "4.3",
                {"name": "No.4.3 將語音合併", "type": "python", "script": "scripts/story_merge_audio.py", "args": ["--ep", "{ep}"]},
            )
            if res.get("ok"):
                st.success(f"已啟動 4.3：{res.get('log')}")
            else:
                st.error(f"4.3 啟動失敗：{res.get('message')}")
    with b3:
        if st.button("執行 4.4 產生字幕", disabled=any_running, key=f"run_story_44_{info['ep']}"):
            res = _runner().start_substep(
                info,
                "4.4",
                {"name": "No.4.4 語音轉字幕", "type": "python", "script": "scripts/story_generate_subtitles.py", "args": ["--ep", "{ep}"]},
            )
            if res.get("ok"):
                st.success(f"已啟動 4.4：{res.get('log')}")
            else:
                st.error(f"4.4 啟動失敗：{res.get('message')}")
    with b4:
        if st.button("產生語音並合併", disabled=any_running, type="primary", key=f"run_story_42_merge_{info['ep']}"):
            res = _runner().start_substep(
                info,
                "4.2_merge",
                {
                    "name": "No.4.2 產生語音並合併",
                    "type": "python",
                    "script": "scripts/story_generate_tts_audio.py",
                    "args": ["--ep", "{ep}", "--merge-after"],
                },
            )
            if res.get("ok"):
                st.success(f"已啟動語音產生與合併：{res.get('log')}")
            else:
                st.error(f"啟動失敗：{res.get('message')}")

    for label, state in [("4.2", state_42), ("4.3", state_43), ("4.4", state_44), ("整合執行", state_combo)]:
        if state.get("running"):
            st.info(f"{label} 執行中，PID {state.get('pid')}。")

    merged = story_merged_audio_path(ep_path)
    if merged.exists():
        st.audio(str(merged))
        st.caption(f"合併音檔：{merged}")
    subtitles_path = story_subtitles_srt_path(ep_path)
    if subtitles_path.exists():
        st.caption(f"字幕檔：{subtitles_path}")
    with st.expander("查看 voice_cast.json", expanded=False):
        st.json(read_json_file(cast_path) or voice_cast)

def render_story_subtitles_result(ep_path: Path) -> None:
    srt_path = story_subtitles_srt_path(ep_path)
    data = read_json_file(story_subtitles_json_path(ep_path)) or {}
    st.subheader("4.4 語音轉字幕")
    if not srt_path.exists() and not data:
        st.warning("尚未產出故事字幕。請先完成 4.2 語音片段、4.3 合併語音，再執行 No.4.4。")
        return

    srt_text = read_text_file(srt_path, "")
    subtitles = data.get("subtitles") or []
    c1, c2, c3 = st.columns(3)
    c1.metric("字幕數", int(data.get("subtitle_count", len(subtitles))))
    c2.metric("來源", data.get("source", "tts_segments"))
    c3.metric("估計長度", seconds_to_label(float(data.get("duration_seconds", 0) or 0)))

    rows = []
    for idx, item in enumerate(subtitles, 1):
        rows.append(
            {
                "No": idx,
                "Start": seconds_to_label(float(item.get("start", 0) or 0)),
                "End": seconds_to_label(float(item.get("end", 0) or 0)),
                "角色": item.get("speaker", ""),
                "字幕": item.get("text", ""),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif srt_path.exists():
        parsed = parse_srt_entries(srt_path)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "No": item.get("index", ""),
                        "Start": item.get("start_label", ""),
                        "End": item.get("end_label", ""),
                        "字幕": item.get("content", ""),
                    }
                    for item in parsed
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("查看 story_subtitles.srt", expanded=False):
        st.text_area(
            "SRT",
            value=srt_text,
            height=360,
            key=f"story_subtitles_srt_{str(ep_path)}",
            disabled=True,
        )
    if data:
        with st.expander("查看 story_subtitles.json", expanded=False):
            st.json(data)

def render_story_storyboard_result(ep_path: Path) -> None:
    csv_path = story_storyboard_csv_path(ep_path)
    json_path = story_storyboard_json_path(ep_path)
    data = read_json_file(json_path) or {}
    st.subheader("5.1 字幕轉分鏡")
    if not csv_path.exists() and not data:
        st.warning("尚未產出故事分鏡。請先完成 4.4 字幕，再執行 No.5.1。")
        return

    df = pd.DataFrame()
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
        except Exception:
            df = pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    scene_count = int(data.get("scene_count", len(df) if not df.empty else 0))
    c1.metric("Scene 數", scene_count)
    if not df.empty and "end_time" in df.columns:
        c2.metric("分鏡尾端", seconds_to_label(float(pd.to_numeric(df["end_time"], errors="coerce").fillna(0).max())))
    else:
        c2.metric("分鏡尾端", "—")
    c3.metric("來源", data.get("source", "story_subtitles"))

    if not df.empty:
        display_cols = [
            col for col in [
                "scene_id",
                "start_time",
                "end_time",
                "paragraph_id",
                "location_name",
                "characters",
                "summary",
                "reason",
            ]
            if col in df.columns
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        with st.expander("查看各 Scene 產圖 Prompt", expanded=False):
            for _, row in df.iterrows():
                label = f"Scene {row.get('scene_id', '')}｜{row.get('location_name', '')}｜{seconds_to_label(float(row.get('start_time', 0) or 0))}-{seconds_to_label(float(row.get('end_time', 0) or 0))}"
                st.markdown(f"**{label}**")
                st.write(f"角色：{row.get('characters', '')}")
                st.write(f"字幕摘要：{row.get('summary', '')}")
                st.code(str(row.get("image_prompt", "")), language="text")
                subref = str(row.get("subtitle_reference", "")).strip()
                if subref:
                    st.caption(subref)
                st.divider()
    else:
        st.caption("storyboard.csv 無法讀取或目前沒有分鏡資料。")

    if data:
        with st.expander("查看 storyboard.json", expanded=False):
            st.json(data)

def render_story_selection_confirm(info: Dict, chosen: dict) -> None:
    if not chosen:
        return

    def do_select() -> None:
        ok, message = run_story_outline_selection(info, str(chosen.get("id", "")))
        if ok:
            st.session_state["story_select_notice"] = {
                "level": "success",
                "message": f"已選定本次故事：{chosen.get('id', '')}｜{chosen.get('title', '')}",
            }
            if message:
                st.session_state["story_select_notice"]["detail"] = message
            st.rerun()
        st.error(f"選擇故事失敗：{message or 'unknown error'}")

    if hasattr(st, "dialog"):
        @st.dialog("確認選擇這個故事")
        def confirm_dialog():
            render_story_summary(chosen, title="準備選定")
            st.warning("確認後，後續階段會以這個故事作為本集故事基底。")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("確認選擇", key=f"story_confirm_yes_{info['ep']}", type="primary"):
                    do_select()
            with c2:
                if st.button("取消", key=f"story_confirm_no_{info['ep']}"):
                    st.rerun()
        confirm_dialog()
    else:
        pending_key = f"story_confirm_pending_{info['ep']}"
        st.session_state[pending_key] = str(chosen.get("id", ""))

def render_story_inline_confirm_if_needed(info: Dict, options: list[dict]) -> None:
    pending_key = f"story_confirm_pending_{info['ep']}"
    pending_id = st.session_state.get(pending_key)
    if not pending_id:
        return
    chosen = next((item for item in options if str(item.get("id")) == str(pending_id)), {})
    if not chosen:
        st.session_state.pop(pending_key, None)
        return
    with st.container(border=True):
        render_story_summary(chosen, title="準備選定")
        st.warning("請再次確認。確認後，後續階段會以這個故事作為本集故事基底。")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("確認選擇", key=f"story_inline_confirm_yes_{info['ep']}", type="primary"):
                ok, message = run_story_outline_selection(info, pending_id)
                if ok:
                    st.session_state.pop(pending_key, None)
                    st.session_state["story_select_notice"] = {
                        "level": "success",
                        "message": f"已選定本次故事：{chosen.get('id', '')}｜{chosen.get('title', '')}",
                    }
                    if message:
                        st.session_state["story_select_notice"]["detail"] = message
                    st.rerun()
                st.error(f"選擇故事失敗：{message or 'unknown error'}")
        with c2:
            if st.button("取消", key=f"story_inline_confirm_no_{info['ep']}"):
                st.session_state.pop(pending_key, None)
                st.rerun()

def run_story_outline_selection(ep_info: Dict, option_id: str) -> tuple[bool, str]:
    script_path = _root() / "scripts" / "story_select_outline.py"
    env = dict(os.environ)
    env["CAP_WORKSPACE_ROOT"] = str((_root() / WS_ROOT).resolve())
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--ep",
        str(ep_info["ep"]),
        "--option-id",
        str(option_id),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_root()),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    message = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, message

def render_story_outline_selection_panel(eps):
    if st.session_state["profile_id"] != "story":
        return
    st.divider()
    st.subheader("故事大綱選擇")
    notice = st.session_state.pop("story_select_notice", None)
    if isinstance(notice, dict):
        if notice.get("level") == "success":
            st.success(notice.get("message", ""))
        elif notice.get("level") == "error":
            st.error(notice.get("message", ""))
        if notice.get("detail"):
            st.caption(notice.get("detail"))
    ep_opts = {f"{e['Ep']} ({e['Range']})": e for e in eps}
    if not ep_opts:
        st.caption("尚未建立故事集數。")
        return
    choice = st.selectbox("選擇故事集數", list(ep_opts.keys()), index=max(len(ep_opts) - 1, 0), key="story_select_ep")
    info = ep_opts[choice]["_raw"]
    options = load_story_outline_options(info["path"])
    selected = read_json_file(story_selected_json_path(info["path"])) or {}
    if selected:
        render_story_summary(selected, title="目前已選定")
    if not options:
        st.info("尚未產生故事大綱候選。請先在 Pipeline Manager 執行 Stage 1 或子步驟 1.2。")
        return

    selected_id = str(selected.get("id", ""))
    option_ids = [str(item.get("id", "")) for item in options]
    default_idx = option_ids.index(selected_id) if selected_id in option_ids else 0
    option_id = st.radio(
        "選擇本次故事",
        option_ids,
        index=default_idx,
        format_func=lambda oid: next(
            (
                f"{item.get('id')}｜{item.get('title', '')}｜{item.get('genre', '')}"
                for item in options
                if str(item.get("id")) == str(oid)
            ),
            oid,
        ),
        key=f"story_outline_choice_{info['ep']}",
    )
    chosen = next((item for item in options if str(item.get("id")) == str(option_id)), {})
    if chosen:
        st.markdown(f"**故事一句話：** {chosen.get('logline', '')}")
        st.markdown(f"**核心衝突：** {chosen.get('conflict', '')}")
        outline = chosen.get("three_act_outline") or []
        if outline:
            st.markdown("**三幕大綱：**")
            for idx, line in enumerate(outline, start=1):
                st.write(f"{idx}. {line}")
    if st.button("確認選擇這個故事", key=f"story_select_submit_{info['ep']}"):
        render_story_selection_confirm(info, chosen)
    render_story_inline_confirm_if_needed(info, options)
