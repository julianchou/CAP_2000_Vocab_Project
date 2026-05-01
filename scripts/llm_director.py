import argparse
import csv
import json
import os
import re
from pathlib import Path

import pandas as pd
import srt
from dotenv import load_dotenv
from google import genai

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client()

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
MODEL_NAME = "gemini-2.5-pro"
HOST_PROFILE_PATH = base_dir / "core" / "assets" / "host_profiles.json"

def detect_profile_id() -> str:
    profile_id = str(os.environ.get("CAP_PROFILE_ID", "")).strip()
    if profile_id:
        return profile_id
    if workspace_dir.parent.name == "workspaces" and workspace_dir.name:
        return workspace_dir.name
    return "default"

def shared_prompt_path() -> Path:
    return base_dir / "config" / detect_profile_id() / "prompts" / "storyboard_prompt.txt"

PREV_CLOZE_ANSWER_TOKEN = "__PREV_CLOZE_Q1_ANSWER__"
CURRENT_CLOZE_QUESTION_TOKEN = "__CURRENT_CLOZE_Q1_QUESTION__"

DEFAULT_PROMPT_TEMPLATE = """你是一位專業的英語教學影片分鏡導演。以下是一份英語教學 Podcast 的 SRT 字幕檔。
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


def prompt_template_path_for(_target_folder: Path) -> Path:
    return shared_prompt_path()


def ensure_prompt_template(target_folder: Path) -> Path:
    prompt_path = prompt_template_path_for(target_folder)
    if not prompt_path.exists():
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def load_host_profiles() -> dict:
    if not HOST_PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(HOST_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_host_profile_rules(host_profiles: dict) -> str:
    if not host_profiles:
        return ""
    style = str(host_profiles.get("style_config", "")).strip()
    characters = host_profiles.get("characters") or {}
    john = characters.get("john") or {}
    mary = characters.get("mary") or {}
    john_desc = str(john.get("description", "")).strip()
    mary_desc = str(mary.get("description", "")).strip()
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
        "- Preserve host identity through consistent hairstyle, glasses, outfit, silhouette, color palette, and left/right placement instead of detailed facial close-ups.",
        "- Keep Mary's long blonde pigtails clearly visible whenever Mary appears.",
        "- Keep John's messy brown hair, green hoodie, and gaming headset consistent whenever John appears.",
        "- Avoid direct front-facing eye contact to camera; prefer three-quarter view or side angle for host scenes.",
        "- If a scene is purely object-based, abstract, or clearly requires another role, do not force the hosts into that scene.",
    ]
    return "\n".join(rules)


def load_questions(storyboard_dir: Path) -> list[dict]:
    json_path = storyboard_dir / "cloze_questions.json"
    csv_path = storyboard_dir / "cloze_questions.csv"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    if csv_path.exists():
        return pd.read_csv(csv_path).fillna("").to_dict(orient="records")
    return []


def find_first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_cloze_question_card(target_folder: Path) -> Path | None:
    questions = load_questions(target_folder / "03_storyboards")
    if not questions:
        return None
    q = questions[0]
    safe_word = str(q.get("word", "")).replace(" ", "_").replace("/", "_")
    stem = f"{q.get('question_id', 'Q01')}_{safe_word}"
    return find_first_existing_path([
        target_folder / "04_images" / "cloze_cards" / "questions" / f"{stem}.png",
    ])


def first_cloze_answer_card(target_folder: Path) -> Path | None:
    questions = load_questions(target_folder / "03_storyboards")
    if not questions:
        return None
    q = questions[0]
    safe_word = str(q.get("word", "")).replace(" ", "_").replace("/", "_")
    stem = f"{q.get('question_id', 'Q01')}_{safe_word}_answer"
    return find_first_existing_path([
        target_folder / "04_images" / "cloze_cards" / "answers" / f"{stem}.png",
    ])


def build_cloze_card_rules(ep_num: int, target_folder: Path) -> tuple[str, dict[str, str]]:
    rules = []
    asset_map: dict[str, str] = {}

    if ep_num > 1:
        try:
            prev_folder, _prev_start, _prev_end = resolve_episode_range(workspace_dir, ep_num - 1)
        except Exception:
            prev_folder = None
        if prev_folder:
            prev_answer_card = first_cloze_answer_card(prev_folder)
            if prev_answer_card:
                asset_map[PREV_CLOZE_ANSWER_TOKEN] = str(prev_answer_card.resolve()).replace("\\", "/")
                rules.append(
                    f'- If the previous episode has a cloze answer card, reserve one FLASHCARD scene for it with '
                    f'source_type set to "FLASHCARD" and flashcard_word set to "{PREV_CLOZE_ANSWER_TOKEN}".'
                )
                rules.append(
                    "- Use that special token only once, for the opening answer-reveal segment of the episode."
                )

    current_question_card = first_cloze_question_card(target_folder)
    if current_question_card:
        asset_map[CURRENT_CLOZE_QUESTION_TOKEN] = str(current_question_card.resolve()).replace("\\", "/")
        rules.append(
            f'- If this episode has a cloze question card, reserve one FLASHCARD scene for it with '
            f'source_type set to "FLASHCARD" and flashcard_word set to "{CURRENT_CLOZE_QUESTION_TOKEN}".'
        )
        rules.append(
            "- Use that special token only once, for the closing teaser question segment of the episode."
        )

    if not rules:
        rules.append("- No special cloze-card flashcard scenes are required for this episode.")

    return "\n".join(rules), asset_map


def render_prompt(
    template_text: str,
    word_list: list[str],
    srt_text: str,
    cloze_card_rules: str,
    host_profile_rules: str,
) -> str:
    rendered = (
        template_text
        .replace("{{WORD_LIST}}", json.dumps(word_list, ensure_ascii=False, indent=2))
        .replace("{{CLOZE_CARD_RULES}}", cloze_card_rules)
        .replace("{{HOST_PROFILE_RULES}}", host_profile_rules)
        .replace("{{SRT_CONTENT}}", srt_text)
    )
    if "{{CLOZE_CARD_RULES}}" not in template_text and cloze_card_rules.strip():
        rendered += "\n\nCloze card rules:\n" + cloze_card_rules.strip()
    if "{{HOST_PROFILE_RULES}}" not in template_text and host_profile_rules.strip():
        rendered += "\n\nHost profile rules:\n" + host_profile_rules.strip()
    rendered += (
        "\n\nTimeline rules:\n"
        "- Every scene must stay within the actual SRT timeline.\n"
        "- Do not invent extra intro/outro scenes after the last subtitle ends.\n"
        "- If the episode ends at the last subtitle, the storyboard must also end there.\n"
    )
    return rendered


def resolve_flashcard_image_path(images_folder: Path, flashcard_word: str, special_asset_map: dict[str, str]) -> str:
    token = str(flashcard_word or "").strip()
    if token in special_asset_map:
        return special_asset_map[token]
    safe_word = token.replace(" ", "_").replace("?", "")
    flashcard_path = images_folder / "flashcards" / f"{safe_word}.png"
    return str(flashcard_path.resolve()).replace("\\", "/")


def strip_visual_suffix(prompt: str) -> str:
    text = str(prompt or "").strip()
    text = re.sub(r",?\s*aspect ratio 16:9, cinematic wide shot\.?\s*$", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.")


def scene_needs_host_consistency(scene: dict) -> bool:
    haystack = " ".join(
        [
            str(scene.get("image_prompt", "")),
            str(scene.get("reason", "")),
            str(scene.get("subtitle_reference", "")),
        ]
    ).lower()
    keywords = [
        "host",
        "hosts",
        "podcast",
        "studio",
        "john",
        "mary",
        "recording",
        "microphone",
        "audience",
        "goodbye",
    ]
    return any(keyword in haystack for keyword in keywords)


def compact_identity_description(description: str) -> str:
    text = str(description or "").strip().rstrip(".")
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def apply_host_profiles_to_prompt(image_prompt: str, host_profiles: dict) -> str:
    if not host_profiles:
        return image_prompt
    characters = host_profiles.get("characters") or {}
    john_desc = str((characters.get("john") or {}).get("description", "")).strip()
    mary_desc = str((characters.get("mary") or {}).get("description", "")).strip()
    style = str(host_profiles.get("style_config", "")).strip()
    if not john_desc or not mary_desc:
        return image_prompt
    action_prompt = strip_visual_suffix(image_prompt)
    john_identity = compact_identity_description(john_desc)
    mary_identity = compact_identity_description(mary_desc)
    prefix_parts = []
    if style:
        prefix_parts.append(style)
    prefix_parts.append("Educational wide shot composition.")
    prefix_parts.append("The main teaching object or educational action should be the visual focus.")
    prefix_parts.append("If the recurring hosts appear, keep them as small supporting figures near the left and right edges.")
    prefix_parts.append("Avoid portrait framing, centered character posters, close facial emphasis, direct eye contact to camera, and large front-facing faces.")
    prefix_parts.append("Prefer medium-long shot, side angle, three-quarter view, or over-the-shoulder staging.")
    prefix_parts.append("Preserve host identity through hairstyle, glasses, outfit, silhouette, and left/right placement rather than close facial detail.")
    if john_identity:
        prefix_parts.append(f"John design reference: {john_identity}.")
    if mary_identity:
        prefix_parts.append(f"Mary design reference: {mary_identity}.")
        if "pigtail" in mary_identity.lower():
            prefix_parts.append("Keep Mary's long blonde pigtails clearly visible whenever she appears.")
    if "messy brown hair" in john_identity.lower():
        prefix_parts.append("Keep John's messy brown hair clearly readable whenever he appears.")
    prefix_parts.append(f"Scene content: {action_prompt}")
    return " ".join(prefix_parts).strip() + " aspect ratio 16:9, cinematic wide shot"


def parse_scene_time(value, floor_seconds: float | None = None) -> float:
    if isinstance(value, (int, float)):
        candidate = float(value)
        if floor_seconds is not None:
            while candidate < floor_seconds:
                candidate += 60.0
        return candidate

    text = str(value or "").strip()
    if not text:
        raise ValueError("empty scene time")

    try:
        candidate = float(text)
        if floor_seconds is not None:
            while candidate < floor_seconds:
                candidate += 60.0
        return candidate
    except ValueError:
        pass

    normalized = text.replace(",", ".")
    if ":" in normalized:
        parts = normalized.split(":")
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            candidate = hours * 3600 + minutes * 60 + seconds
            if floor_seconds is not None:
                while candidate < floor_seconds:
                    candidate += 60.0
            return candidate
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            candidate = minutes * 60 + seconds
            if floor_seconds is not None:
                while candidate < floor_seconds:
                    candidate += 60.0
            return candidate

    raise ValueError(f"invalid scene time: {value}")


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_scene_times(scene: dict, previous_end: float = 0.0) -> tuple[float, float]:
    start_t = parse_scene_time(scene.get("start_time"), previous_end)
    end_t = parse_scene_time(scene.get("end_time"), start_t)
    if end_t < start_t:
        raise ValueError(f"end_time earlier than start_time: {scene}")
    scene["start_time"] = round(start_t, 3)
    scene["end_time"] = round(end_t, 3)
    return scene["start_time"], scene["end_time"]


def clamp_scene_to_timeline(
    scene: dict,
    timeline_end: float,
    previous_end: float = 0.0,
    min_duration: float = 0.05,
) -> tuple[float, float] | None:
    start_t, end_t = normalize_scene_times(scene, previous_end)
    if start_t >= timeline_end - min_duration:
        return None
    clipped_end = min(end_t, timeline_end)
    if clipped_end <= start_t + min_duration:
        return None
    scene["start_time"] = round(start_t, 3)
    scene["end_time"] = round(clipped_end, 3)
    return scene["start_time"], scene["end_time"]


def collect_scene_subtitles(subs: list, start_t: float, end_t: float) -> str:
    scene_details = []
    for sub in subs:
        sub_start = sub.start.total_seconds()
        sub_end = sub.end.total_seconds()
        if sub_end <= start_t or sub_start >= end_t:
            continue
        t_start = str(sub.start).split(",")[0]
        t_end = str(sub.end).split(",")[0]
        clean_content = sub.content.replace("\n", " ")
        scene_details.append(f"[{t_start}-->{t_end}] {clean_content}")
    return " | ".join(scene_details)


def generate_storyboard(ep_num: int):
    target_folder, _start_word, _end_word = resolve_episode_range(workspace_dir, ep_num)
    srt_folder = target_folder / "02_subtitles"
    storyboard_folder = target_folder / "03_storyboards"
    images_folder = target_folder / "04_images"
    srt_file = next(srt_folder.glob("*.srt"), None)
    vocab_csv = storyboard_folder / "vocab_data.csv"
    output_csv = storyboard_folder / "storyboard.csv"

    if not srt_file:
        print(f"Missing SRT file in: {srt_folder}")
        return
    if not vocab_csv.exists():
        print(f"Missing vocab_data.csv: {vocab_csv}")
        return

    print("Loading vocab data and subtitle file...")
    df_vocab = pd.read_csv(vocab_csv).fillna("")
    word_list = df_vocab["Word"].astype(str).str.strip().tolist()
    srt_text = srt_file.read_text(encoding="utf-8")
    subs = list(srt.parse(srt_text))
    if not subs:
        print(f"SRT file has no subtitle entries: {srt_file}")
        return
    timeline_end = max(sub.end.total_seconds() for sub in subs)
    cloze_card_rules, special_asset_map = build_cloze_card_rules(ep_num, target_folder)
    host_profiles = load_host_profiles()
    host_profile_rules = build_host_profile_rules(host_profiles)

    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = render_prompt(prompt_template, word_list, srt_text, cloze_card_rules, host_profile_rules)

    print(f"Calling {MODEL_NAME} to generate storyboard scenes...")
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        cleaned_json_str = response.text.replace("```json", "").replace("```", "").strip()
        storyboard_data = json.loads(cleaned_json_str)

        previous_end = 0.0
        normalized_scenes = []
        trimmed_count = 0
        dropped_count = 0
        for scene in storyboard_data:
            original_end = safe_float(scene.get("end_time"), 0.0)
            clamped = clamp_scene_to_timeline(scene, timeline_end, previous_end)
            if clamped is None:
                dropped_count += 1
                continue
            start_t, end_t = clamped
            if original_end > end_t + 0.001:
                trimmed_count += 1

            scene["subtitle_reference"] = collect_scene_subtitles(subs, start_t, end_t)
            scene["flashcard_word"] = str(scene.get("flashcard_word", "")).strip()

            if str(scene.get("source_type", "")).strip().upper() == "FLASHCARD" and scene["flashcard_word"]:
                scene["custom_image_path"] = resolve_flashcard_image_path(
                    images_folder,
                    scene["flashcard_word"],
                    special_asset_map,
                )
                scene["image_prompt"] = ""
                scene["source_type"] = "FLASHCARD"
            else:
                scene["custom_image_path"] = ""
                scene["source_type"] = "AI"
                scene["flashcard_word"] = ""
                if scene_needs_host_consistency(scene):
                    scene["image_prompt"] = apply_host_profiles_to_prompt(scene.get("image_prompt", ""), host_profiles)
            previous_end = end_t
            normalized_scenes.append(scene)

        for i, scene in enumerate(normalized_scenes, start=1):
            scene["scene_id"] = i
        storyboard_data = normalized_scenes

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
        ]
        with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for scene in storyboard_data:
                writer.writerow(scene)

        print(f"Storyboard saved to: {output_csv}")
        if trimmed_count:
            print(f"Trimmed {trimmed_count} scene(s) to fit the SRT timeline end {timeline_end:.3f}s.")
        if dropped_count:
            print(f"Dropped {dropped_count} scene(s) that started beyond the SRT timeline.")
        if special_asset_map:
            print("Special flashcard assets:")
            for token, path in special_asset_map.items():
                print(f"- {token}: {path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="Episode number")
    args = parser.parse_args()
    generate_storyboard(args.ep)

