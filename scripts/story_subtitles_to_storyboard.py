import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()
client = genai.Client()

BASE_DIR = Path(__file__).resolve().parent.parent
TEXT_MODEL = os.environ.get("CAP_STORY_STORYBOARD_MODEL", os.environ.get("CAP_TEXT_MODEL", "gemini-2.5-pro"))

DEFAULT_PROMPT_TEMPLATE = """你是一位專業的兒童童話影片分鏡導演。請根據故事字幕、角色設定與場景設定，進行 AI 依語境分鏡規劃。

【影片類型】
- 10歲以下小朋友適合觀看的童話故事影片。
- 畫面要溫暖、明亮、童趣、安心，不可驚悚、暴力、黑暗、過度寫實。

【核心任務】
1. 依字幕語境切分分鏡，每個分鏡必須是一個清楚的故事畫面。
2. 每個分鏡長度不可超過 {{MAX_SECONDS}} 秒，因為後續每段動畫最多只能產生 {{MAX_SECONDS}} 秒。
3. 分鏡必須完整覆蓋字幕時間軸，不可創造超出字幕尾端的額外畫面。
4. 分鏡應優先依故事事件、角色動作、場景轉換、情緒變化切分。
5. 每個分鏡都要指定對應角色與場景，並產生可供 AI 生圖使用的英文 image_prompt。

【時間軸規則】
- start_time 與 end_time 請使用秒數，例如 12.5。
- end_time 必須大於 start_time。
- 每個分鏡 duration = end_time - start_time 必須 <= {{MAX_SECONDS}}。
- 不可重疊；下一個分鏡的 start_time 應接續前一個 end_time。
- 若單一語意段太長，請拆成多個連續分鏡。

【童話視覺規則】
- image_prompt 必須是英文。
- image_prompt 必須包含：storybook illustration、child-friendly、warm、bright、aspect ratio 16:9, cinematic wide shot。
- 角色外觀要沿用 characters.json。
- 場景要沿用 locations.json。
- 不要產生文字卡、字幕卡、英文字卡或 UI 畫面。

【輸出格式】
只輸出合法 JSON 陣列，不要 Markdown，不要說明文字。
每個元素必須包含：
- start_time
- end_time
- source_type，固定填 "AI"
- paragraph_id
- location_id
- location_name
- characters，角色名稱陣列
- summary，繁體中文，簡短描述此分鏡劇情
- image_prompt，英文生圖 prompt
- reason，繁體中文，說明為什麼這裡需要切成一個分鏡

【故事字幕 JSON】
{{SUBTITLES_JSON}}

【角色設定 JSON】
{{CHARACTERS_JSON}}

【場景設定 JSON】
{{LOCATIONS_JSON}}
"""


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def read_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


CLEAN_STORYBOARD_PROMPT_TEMPLATE = """你是故事生成模式的分鏡導演。請依照故事字幕、No.3.1 角色人物塑造、No.3.2 場景塑造，以及 No.4.2 使用者預聽台詞/完整 TTS 台詞脈絡，規劃故事分鏡。

重要原則：
1. 這是故事生成模式，不是單字卡或詞彙模式。不要建立 FLASHCARD 分鏡，不要參考單字清單，不要輸出 flashcard_word。
2. 每個分鏡都必須服務故事情節、角色行動、情緒變化與場景連續性。
3. 每個分鏡長度不得超過 {{MAX_SECONDS}} 秒；時間必須連續覆蓋字幕時間軸，不可重疊。
4. image_prompt 必須根據以下資訊產生：
   - No.3.1 characters.json 的角色外觀、性格、服裝、聲音/語氣設定。
   - No.3.2 locations.json 的場景設計、色彩、道具與段落連結。
   - No.4.2 的預聽台詞 voice_preview_request.json，以及 tts_lines.json 的實際台詞、speaker、voice_hint。
   - 目前分鏡所覆蓋的字幕內容與段落 paragraph_id。
5. image_prompt 必須是英文，描述適合兒童故事影片的具體畫面，不要只重述字幕。
6. image_prompt 必須包含角色名稱/外觀、場景名稱/視覺細節、當下動作或情緒，並包含 storybook illustration、child-friendly、warm、bright、aspect ratio 16:9, cinematic wide shot。
7. 不要產生恐怖、暴力、危險、成人或不適合 10 歲以下兒童的畫面。

請只輸出 JSON array，不要 Markdown，不要補充說明。每個 item 必須包含：
- start_time
- end_time
- source_type：固定為 "AI"
- paragraph_id
- location_id
- location_name
- characters：角色名稱陣列
- summary：此分鏡的故事畫面摘要
- image_prompt：英文生圖 prompt
- reason：簡短說明此分鏡如何對應字幕、角色與場景

故事字幕 JSON：
{{SUBTITLES_JSON}}

No.3.1 角色人物塑造 characters.json：
{{CHARACTERS_JSON}}

No.3.2 場景塑造 locations.json：
{{LOCATIONS_JSON}}

No.4.1/4.2 完整 TTS 台詞 tts_lines.json：
{{TTS_LINES_JSON}}

No.4.2 使用者預聽台詞 voice_preview_request.json：
{{VOICE_PREVIEW_JSON}}
"""


def prompt_template_path() -> Path:
    return BASE_DIR / "config" / "story" / "prompts" / "storyboard_prompt.txt"


def ensure_prompt_template() -> Path:
    path = prompt_template_path()
    should_write = not path.exists()
    if not should_write:
        current = path.read_text(encoding="utf-8", errors="ignore")
        should_write = any(
            token in current
            for token in ["{{WORD_LIST}}", "{{SRT_CONTENT}}", "{{CLOZE_CARD_RULES}}", "{{HOST_PROFILE_RULES}}"]
        ) or "{{TTS_LINES_JSON}}" not in current
    if should_write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CLEAN_STORYBOARD_PROMPT_TEMPLATE, encoding="utf-8")
    return path


def safe_join(values) -> str:
    result = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return "、".join(result)


def parse_model_json(text: str) -> list[dict]:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("model response must be a JSON array")
    return [item for item in data if isinstance(item, dict)]


def compact_subtitles(subtitles_data: dict) -> list[dict]:
    rows = []
    for item in subtitles_data.get("subtitles") or []:
        rows.append(
            {
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "paragraph_id": item.get("paragraph_id", ""),
                "speaker": item.get("speaker", ""),
                "text": item.get("text", ""),
            }
        )
    return rows


def compact_tts_lines(tts_data: dict) -> list[dict]:
    rows = []
    for item in tts_data.get("lines") or []:
        rows.append(
            {
                "line_id": item.get("line_id", ""),
                "sequence": item.get("sequence", ""),
                "paragraph_id": item.get("paragraph_id", ""),
                "line_type": item.get("line_type", ""),
                "speaker": item.get("speaker", ""),
                "voice_style": item.get("voice_style", ""),
                "voice_hint": item.get("voice_hint", ""),
                "text": item.get("text", ""),
            }
        )
    return rows


def split_subtitle_item(item: dict, max_seconds: float) -> list[dict]:
    start_t = float(item.get("start", 0) or 0)
    end_t = float(item.get("end", 0) or 0)
    text = str(item.get("text", "")).strip()
    if end_t <= start_t or end_t - start_t <= max_seconds + 0.001 or not text:
        return [item]
    total_duration = end_t - start_t
    total_chars = max(len(text), 1)
    pieces = []
    cursor = start_t
    char_cursor = 0
    while cursor < end_t - 0.05 and char_cursor < len(text):
        part_end = min(cursor + max_seconds, end_t)
        ratio = (part_end - start_t) / total_duration
        split_at = max(int(round(total_chars * ratio)), char_cursor + 1)
        if part_end < end_t:
            window_start = max(char_cursor + 1, split_at - 8)
            window_end = min(len(text), split_at + 8)
            candidates = [pos + 1 for pos in range(window_start, window_end) if text[pos:pos + 1] in "，。！？、；：,.!?"]
            if candidates:
                split_at = min(candidates, key=lambda pos: abs(pos - split_at))
        else:
            split_at = len(text)
        split_at = min(max(split_at, char_cursor + 1), len(text))
        part = dict(item)
        part["start"] = round(cursor, 3)
        part["end"] = round(part_end, 3)
        part["text"] = text[char_cursor:split_at].strip()
        if part["text"]:
            pieces.append(part)
        cursor = part_end
        char_cursor = split_at
    return pieces or [item]


def enforce_subtitle_max_duration(subtitles: list[dict], max_seconds: float) -> list[dict]:
    out = []
    for item in subtitles:
        out.extend(split_subtitle_item(item, max_seconds))
    return out


def render_prompt(
    template: str,
    subtitles_data: dict,
    characters_data: dict,
    locations_data: dict,
    tts_lines_data: dict,
    voice_preview_data: dict,
    max_seconds: float,
) -> str:
    subtitles_payload = {
        "story_title": subtitles_data.get("story_title", ""),
        "duration_seconds": subtitles_data.get("duration_seconds", 0),
        "subtitles": compact_subtitles(subtitles_data),
    }
    tts_payload = {
        "story_title": tts_lines_data.get("story_title", ""),
        "lines": compact_tts_lines(tts_lines_data),
    }
    return (
        template
        .replace("{{MAX_SECONDS}}", f"{max_seconds:g}")
        .replace("{{SUBTITLES_JSON}}", json.dumps(subtitles_payload, ensure_ascii=False, indent=2))
        .replace("{{CHARACTERS_JSON}}", json.dumps(characters_data, ensure_ascii=False, indent=2))
        .replace("{{LOCATIONS_JSON}}", json.dumps(locations_data, ensure_ascii=False, indent=2))
        .replace("{{TTS_LINES_JSON}}", json.dumps(tts_payload, ensure_ascii=False, indent=2))
        .replace("{{VOICE_PREVIEW_JSON}}", json.dumps(voice_preview_data, ensure_ascii=False, indent=2))
    )


def timeline_end(subtitles_data: dict) -> float:
    ends = []
    for item in subtitles_data.get("subtitles") or []:
        try:
            ends.append(float(item.get("end", 0) or 0))
        except Exception:
            pass
    return max(ends) if ends else 0.0


def subtitle_refs(subtitles: list[dict], start_t: float, end_t: float) -> str:
    refs = []
    for item in subtitles:
        sub_start = float(item.get("start", 0) or 0)
        sub_end = float(item.get("end", 0) or 0)
        if sub_end <= start_t or sub_start >= end_t:
            continue
        refs.append(f"[{sub_start:.3f}-{sub_end:.3f}] {item.get('text', '')}")
    return " | ".join(refs)


def character_lookup(characters_data: dict) -> dict:
    lookup = {}
    for item in characters_data.get("characters") or []:
        name = str(item.get("name", "")).strip()
        if name:
            lookup[name] = item
    return lookup


def location_lookup(locations_data: dict) -> tuple[dict, dict]:
    by_paragraph = {}
    by_id = {}
    for item in locations_data.get("locations") or []:
        location_id = str(item.get("location_id", "")).strip()
        if location_id:
            by_id[location_id] = item
        for paragraph_id in item.get("linked_paragraph_ids") or []:
            by_paragraph[str(paragraph_id).strip()] = item
    return by_paragraph, by_id


def build_fallback_prompt(scene: dict, location: dict) -> str:
    location_name = str(location.get("name", "")).strip() or str(scene.get("location_name", "")).strip() or "fairy-tale story setting"
    visual = location.get("visual_design") or {}
    props = ", ".join(visual.get("key_props") or [])
    palette = ", ".join(visual.get("color_palette") or [])
    characters = safe_join(scene.get("characters") or [])
    return (
        "A warm, bright, child-friendly fairy-tale storybook illustration. "
        f"Scene location: {location_name}. "
        f"Characters: {characters or 'cute fairy-tale characters'}. "
        f"Story action: {scene.get('summary', '')}. "
        f"Environment details: {props}. Color palette: {palette}. "
        "gentle, safe for children under 10, no scary or violent content, "
        "aspect ratio 16:9, cinematic wide shot"
    )


def fallback_storyboard(subtitles_data: dict, characters_data: dict, locations_data: dict, max_seconds: float, reason: str) -> dict:
    subtitles = enforce_subtitle_max_duration(compact_subtitles(subtitles_data), max_seconds)
    locations_by_paragraph, locations_by_id = location_lookup(locations_data)
    scenes = []
    current = []
    for item in subtitles:
        if current:
            current_start = float(current[0].get("start", 0) or 0)
            item_end = float(item.get("end", 0) or 0)
            if item_end - current_start > max_seconds:
                scenes.append(build_fallback_scene(current, locations_by_paragraph, locations_by_id, max_seconds))
                current = []
        current.append(item)
    if current:
        scenes.append(build_fallback_scene(current, locations_by_paragraph, locations_by_id, max_seconds))

    for idx, scene in enumerate(scenes, 1):
        scene["scene_id"] = idx
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "fallback_context_storyboard",
        "model": "local_fallback",
        "fallback_reason": reason,
        "story_title": subtitles_data.get("story_title", ""),
        "scene_count": len(scenes),
        "max_seconds": max_seconds,
        "scenes": scenes,
    }


def build_fallback_scene(current: list[dict], locations_by_paragraph: dict, locations_by_id: dict, max_seconds: float) -> dict:
    paragraph_id = str(current[0].get("paragraph_id", "")).strip()
    location = locations_by_paragraph.get(paragraph_id) or next(iter(locations_by_id.values()), {})
    speakers = [
        str(row.get("speaker", "")).strip()
        for row in current
        if str(row.get("speaker", "")).strip() and str(row.get("speaker", "")).strip() != "旁白"
    ]
    text = "".join(str(row.get("text", "")).strip() for row in current)
    scene = {
        "scene_id": 0,
        "start_time": round(float(current[0].get("start", 0) or 0), 3),
        "end_time": round(float(current[-1].get("end", 0) or 0), 3),
        "source_type": "AI",
        "flashcard_word": "",
        "custom_image_path": "",
        "paragraph_id": paragraph_id,
        "location_id": location.get("location_id", ""),
        "location_name": location.get("name", ""),
        "characters": sorted(set(speakers)),
        "summary": text[:180],
        "reason": f"Gemini 無法完成時使用本地語境切分；每段控制在 {max_seconds:g} 秒以內。",
        "subtitle_reference": subtitle_refs(current, float(current[0].get("start", 0) or 0), float(current[-1].get("end", 0) or 0)),
        "animation_prompt": "",
    }
    scene["image_prompt"] = build_fallback_prompt(scene, location)
    return scene


def infer_scene_fields(scene: dict, subtitles: list[dict], start_t: float, end_t: float) -> dict:
    overlapping = [
        item for item in subtitles
        if float(item.get("end", 0) or 0) > start_t and float(item.get("start", 0) or 0) < end_t
    ]
    if not str(scene.get("paragraph_id", "")).strip() and overlapping:
        scene["paragraph_id"] = overlapping[0].get("paragraph_id", "")
    if not scene.get("characters"):
        speakers = [
            str(item.get("speaker", "")).strip()
            for item in overlapping
            if str(item.get("speaker", "")).strip() and str(item.get("speaker", "")).strip() != "旁白"
        ]
        scene["characters"] = sorted(set(speakers))
    if not str(scene.get("summary", "")).strip():
        scene["summary"] = "".join(str(item.get("text", "")) for item in overlapping)[:180]
    return scene


def split_long_scene(scene: dict, max_seconds: float) -> list[dict]:
    start_t = float(scene["start_time"])
    end_t = float(scene["end_time"])
    if end_t - start_t <= max_seconds + 0.001:
        return [scene]
    out = []
    cursor = start_t
    while cursor < end_t - 0.05:
        part = dict(scene)
        part["start_time"] = round(cursor, 3)
        part["end_time"] = round(min(cursor + max_seconds, end_t), 3)
        part["reason"] = (str(scene.get("reason", "")).strip() + "；因單一分鏡長度上限拆分。").strip("；")
        out.append(part)
        cursor = float(part["end_time"])
    return out


def normalize_scenes(raw_scenes: list[dict], subtitles_data: dict, max_seconds: float) -> list[dict]:
    subtitles = compact_subtitles(subtitles_data)
    end_limit = timeline_end(subtitles_data)
    normalized = []
    previous_end = 0.0

    for raw in raw_scenes:
        try:
            start_t = float(raw.get("start_time", previous_end) or previous_end)
            end_t = float(raw.get("end_time", start_t) or start_t)
        except Exception:
            continue
        start_t = max(start_t, previous_end)
        if end_limit > 0:
            start_t = min(start_t, end_limit)
            end_t = min(end_t, end_limit)
        if end_t <= start_t + 0.05:
            continue
        scene = dict(raw)
        scene["start_time"] = round(start_t, 3)
        scene["end_time"] = round(end_t, 3)
        scene["source_type"] = "AI"
        scene["flashcard_word"] = ""
        scene["custom_image_path"] = ""
        scene = infer_scene_fields(scene, subtitles, start_t, end_t)
        for part in split_long_scene(scene, max_seconds):
            part_start = float(part["start_time"])
            part_end = float(part["end_time"])
            part["subtitle_reference"] = subtitle_refs(subtitles, part_start, part_end)
            normalized.append(part)
            previous_end = part_end

    if end_limit > 0 and (not normalized or float(normalized[-1]["end_time"]) < end_limit - 0.05):
        cursor = float(normalized[-1]["end_time"]) if normalized else 0.0
        while cursor < end_limit - 0.05:
            end_t = min(cursor + max_seconds, end_limit)
            fallback = infer_scene_fields(
                {
                    "start_time": round(cursor, 3),
                    "end_time": round(end_t, 3),
                    "source_type": "AI",
                    "flashcard_word": "",
                    "custom_image_path": "",
                    "paragraph_id": "",
                    "location_id": "",
                    "location_name": "",
                    "characters": [],
                    "summary": "",
                    "image_prompt": "",
                    "reason": "補齊 AI 分鏡未覆蓋的字幕時間軸。",
                },
                subtitles,
                cursor,
                end_t,
            )
            fallback["subtitle_reference"] = subtitle_refs(subtitles, cursor, end_t)
            normalized.append(fallback)
            cursor = end_t

    for idx, scene in enumerate(normalized, 1):
        scene["scene_id"] = idx
        scene["source_type"] = "AI"
        scene.setdefault("flashcard_word", "")
        scene.setdefault("custom_image_path", "")
        scene.setdefault("paragraph_id", "")
        scene.setdefault("location_id", "")
        scene.setdefault("location_name", "")
        scene.setdefault("characters", [])
        scene.setdefault("summary", "")
        scene.setdefault("reason", "")
        scene.setdefault("subtitle_reference", "")
        scene.setdefault("animation_prompt", "")
        scene["image_prompt"] = ensure_image_prompt(scene)
    return normalized


def ensure_image_prompt(scene: dict) -> str:
    prompt = str(scene.get("image_prompt", "")).strip()
    if not prompt:
        prompt = (
            "A warm, bright, child-friendly fairy-tale storybook illustration. "
            f"Scene summary: {scene.get('summary', '')}. "
            f"Location: {scene.get('location_name', '')}. "
            f"Characters: {safe_join(scene.get('characters') or [])}. "
            "storybook illustration, warm, bright, gentle, suitable for children under 10, "
            "aspect ratio 16:9, cinematic wide shot"
        )
    required = ["storybook illustration", "child-friendly", "warm", "bright", "aspect ratio 16:9", "cinematic wide shot"]
    lower = prompt.lower()
    missing = [term for term in required if term not in lower]
    if missing:
        prompt = prompt.rstrip(" .") + ". " + ", ".join(missing)
    return prompt


def write_storyboard_csv(path: Path, scenes: list[dict]) -> None:
    fieldnames = [
        "scene_id",
        "start_time",
        "end_time",
        "source_type",
        "flashcard_word",
        "custom_image_path",
        "image_prompt",
        "reason",
        "subtitle_reference",
        "paragraph_id",
        "location_id",
        "location_name",
        "characters",
        "summary",
        "animation_prompt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            row = dict(scene)
            row["characters"] = safe_join(row.get("characters") or [])
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def generate_ai_storyboard(ep_path: Path, max_seconds: float) -> dict:
    subtitles_data = read_json(ep_path / "04_audio_subtitles" / "story_subtitles.json")
    characters_data = read_json(ep_path / "03_characters_scenes" / "characters.json")
    locations_data = read_json(ep_path / "03_characters_scenes" / "locations.json")
    tts_lines_data = read_json_optional(ep_path / "04_audio_subtitles" / "tts_lines.json")
    voice_preview_data = read_json_optional(ep_path / "04_audio_subtitles" / "voice_preview_request.json")
    if not subtitles_data.get("subtitles"):
        raise RuntimeError("story_subtitles.json has no subtitles. Run No.4.4 first.")

    template_path = ensure_prompt_template()
    prompt = render_prompt(
        template_path.read_text(encoding="utf-8"),
        subtitles_data,
        characters_data,
        locations_data,
        tts_lines_data,
        voice_preview_data,
        max_seconds,
    )
    prompt_out = ep_path / "05_storyboards" / "storyboard_prompt.txt"
    prompt_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_out.write_text(prompt, encoding="utf-8")

    raw_path = ep_path / "05_storyboards" / "storyboard_ai_response.json"
    try:
        print(f"Calling {TEXT_MODEL} for story storyboard planning...")
        response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
        raw_text = getattr(response, "text", "") or ""
        raw_path.write_text(raw_text, encoding="utf-8")
    except Exception as exc:
        message = str(exc)
        print(f"[WARN] Gemini storyboard planning failed; using local fallback. reason={message}")
        raw_path.write_text(
            json.dumps({"error": message, "fallback": "local_context_storyboard"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return fallback_storyboard(subtitles_data, characters_data, locations_data, max_seconds, message)

    try:
        raw_scenes = parse_model_json(raw_text)
    except Exception as exc:
        message = f"invalid model JSON: {exc}"
        print(f"[WARN] {message}; using local fallback.")
        return fallback_storyboard(subtitles_data, characters_data, locations_data, max_seconds, message)
    scenes = normalize_scenes(raw_scenes, subtitles_data, max_seconds)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "ai_context_storyboard",
        "model": TEXT_MODEL,
        "prompt_template": str(template_path),
        "story_title": subtitles_data.get("story_title", ""),
        "scene_count": len(scenes),
        "max_seconds": max_seconds,
        "scenes": scenes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode No.5.1: AI context storyboard planning from subtitles.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    parser.add_argument("--max-seconds", type=float, default=8.0)
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    max_seconds = min(max(args.max_seconds, 1.0), 8.0)
    payload = generate_ai_storyboard(ep_path, max_seconds=max_seconds)
    out_dir = ep_path / "05_storyboards"
    csv_path = out_dir / "storyboard.csv"
    json_path = out_dir / "storyboard.json"
    write_storyboard_csv(csv_path, payload["scenes"])
    write_json(json_path, payload)
    print(f"storyboard_csv={csv_path}")
    print(f"storyboard_json={json_path}")
    print(f"scene_count={payload['scene_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
