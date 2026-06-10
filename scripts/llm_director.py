import argparse
import csv
import json
import os
import re
from pathlib import Path

import pandas as pd
import srt
from dotenv import load_dotenv

from episode_range_utils import resolve_episode_range
from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response


load_dotenv()

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
GEMINI_MODEL_NAME = os.getenv("CAP_STORYBOARD_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL_NAME = os.getenv("CAP_STORYBOARD_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-5.2"))
NVIDIA_MODEL_NAME = os.getenv(
    "CAP_STORYBOARD_NVIDIA_MODEL",
    os.getenv("NVIDIA_STORYBOARD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
)
VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
HOST_PROFILE_PATH = base_dir / "core" / "assets" / "host_profiles.json"


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_STORYBOARD_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "openai"


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "spending cap" in text.lower() or "quota" in text.lower()


def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_text(text: str) -> str:
    text = strip_json_fence(text)
    if not text:
        return text
    if text[0] in "[{":
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def normalize_storyboard_payload(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("scenes", "storyboard", "storyboard_data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("model response must be a JSON array")


def normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def flashcard_matches_subtitle(flashcard_word: str, subtitle_reference: str) -> bool:
    word_key = normalize_match_text(flashcard_word)
    if not word_key:
        return False
    return word_key in normalize_match_text(subtitle_reference)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def ensure_visual_suffix(prompt: str) -> str:
    text = strip_visual_suffix(str(prompt or ""))
    if not text:
        text = "Cute 3D cartoon educational vocabulary video scene"
    return text.strip(" ,.") + ", aspect ratio 16:9, cinematic wide shot"


def english_fragments_from_subtitle_reference(subtitle_reference: str, max_chars: int = 190) -> str:
    parts = []
    for segment in str(subtitle_reference or "").split("|"):
        text = re.sub(r"\[[^\]]+\]", "", segment).strip()
        text = re.sub(r"[^A-Za-z0-9 ,.'!?;:()/-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" ,.;")
        if text and re.search(r"[A-Za-z]", text):
            parts.append(text)
    joined = "; ".join(parts)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1].rstrip(" ,.;") + "."
    return joined


def english_scene_focus(scene: dict) -> str:
    prompt_text = strip_visual_suffix(str(scene.get("image_prompt", "")))
    if prompt_text and not contains_cjk(prompt_text):
        return prompt_text
    reason = str(scene.get("reason", "")).strip()
    if reason and not contains_cjk(reason):
        return reason
    english_subtitles = english_fragments_from_subtitle_reference(str(scene.get("subtitle_reference", "")))
    if english_subtitles:
        return f"visual focus on the English learning idea: {english_subtitles}"
    return "visual focus on a classroom vocabulary learning transition, with simple educational props and a clear teaching moment"


def sanitize_image_prompt(scene: dict) -> str:
    prompt = str(scene.get("image_prompt", "")).strip()
    if prompt and not contains_cjk(prompt):
        return ensure_visual_suffix(prompt)
    focus = english_scene_focus(scene)
    return ensure_visual_suffix(
        "Cute 3D cartoon educational vocabulary video scene, two recurring hosts in a classroom podcast studio, "
        f"{focus}"
    )


def subtitle_text_matches_word(word: str, text: str) -> bool:
    word_key = normalize_match_text(word)
    text_key = normalize_match_text(text)
    if not word_key or not text_key:
        return False
    if word_key in text_key:
        return True
    # Subtitle review/Whisper sometimes splits English words in bilingual cues:
    # "det ect", "des k", etc. Keep this bounded to avoid broad fuzzy matches.
    if len(word_key) >= 4:
        compact_text = re.sub(r"[^a-z0-9]+", "", str(text or "").lower())
        return word_key in compact_text
    return False


def build_word_occurrence_windows(subs: list, words: list[str], padding: float = 0.18) -> dict[str, list[tuple[float, float]]]:
    windows: dict[str, list[tuple[float, float]]] = {}
    unique_words = []
    seen = set()
    for word in words:
        clean_word = str(word or "").strip()
        key = normalize_word_key(clean_word)
        if clean_word and key not in seen:
            unique_words.append(clean_word)
            seen.add(key)

    for word in unique_words:
        word_key = normalize_word_key(word)
        matches = []
        for sub in subs:
            if subtitle_text_matches_word(word, sub.content):
                matches.append(
                    (
                        max(0.0, sub.start.total_seconds() - padding),
                        sub.end.total_seconds() + padding,
                    )
                )
        merged: list[tuple[float, float]] = []
        for start_t, end_t in matches:
            if not merged or start_t > merged[-1][1] + 0.35:
                merged.append((start_t, end_t))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_t))
        if merged:
            windows[word_key] = merged
    return windows


def closest_word_window(
    word_windows: dict[str, list[tuple[float, float]]],
    word: str,
    preferred_start: float,
) -> tuple[float, float] | None:
    windows = word_windows.get(normalize_word_key(word)) or []
    if not windows:
        return None
    return min(windows, key=lambda item: abs(float(item[0]) - preferred_start))


def align_flashcard_scenes_to_subtitles(scenes: list[dict], word_windows: dict[str, list[tuple[float, float]]]) -> int:
    aligned = 0
    for scene in scenes:
        if str(scene.get("source_type", "")).strip().upper() != "FLASHCARD":
            continue
        word = str(scene.get("flashcard_word", "")).strip()
        if not word:
            continue
        start_t = safe_float(scene.get("start_time"), 0.0)
        end_t = safe_float(scene.get("end_time"), start_t)
        if closest_word_window(word_windows, word, start_t) is None:
            continue
        current_reference = str(scene.get("subtitle_reference", ""))
        if current_reference and flashcard_matches_subtitle(word, current_reference):
            continue
        window = closest_word_window(word_windows, word, start_t)
        if not window:
            continue
        new_start, new_end = window
        if abs(new_start - start_t) > 0.001 or abs(new_end - end_t) > 0.001:
            scene["start_time"] = round(new_start, 3)
            scene["end_time"] = round(new_end, 3)
            scene["reason"] = (
                str(scene.get("reason", "")).strip()
                + f" | aligned_to_subtitle_word:{word}"
            ).strip(" |")
            aligned += 1
    if aligned:
        scenes.sort(key=lambda item: (safe_float(item.get("start_time"), 0.0), safe_float(item.get("end_time"), 0.0)))
    return aligned


def fallback_ai_image_prompt(scene: dict) -> str:
    subtitle_reference = english_fragments_from_subtitle_reference(str(scene.get("subtitle_reference", "")))
    if not subtitle_reference:
        subtitle_reference = "a vocabulary learning transition moment in an educational podcast"
    return (
        "Cute 3D cartoon educational vocabulary video scene, two recurring hosts in a classroom podcast studio, "
        "visual focus on the learning idea from this subtitle segment: "
        f"{subtitle_reference}, aspect ratio 16:9, cinematic wide shot"
    )


def write_json_file(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_storyboard_csv(path: Path, scenes: list[dict]) -> Path:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            writer.writerow({key: scene.get(key, "") for key in fieldnames})
    return path


def save_rejected_storyboard_response(provider: str, raw: str) -> Path:
    debug_dir = base_dir / "runtime" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"storyboard_{provider}_rejected_response.json.txt"
    debug_path.write_text(str(raw or ""), encoding="utf-8")
    return debug_path


def generate_storyboard_json_with_gemini(prompt: str) -> str:
    from google import genai

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print(f"Calling Gemini to generate storyboard scenes: {GEMINI_MODEL_NAME}")
    response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    return getattr(response, "text", "") or ""


def generate_storyboard_json_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"Calling OpenAI/ChatGPT to generate storyboard scenes: {OPENAI_MODEL_NAME}")
    response = OpenAI().responses.create(
        model=OPENAI_MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "Return strict JSON only. The response must be a JSON array and must not include Markdown fences or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def generate_storyboard_json_with_nvidia(prompt: str) -> str:
    print(f"Calling NVIDIA to generate storyboard scenes: {NVIDIA_MODEL_NAME}")
    return nvidia_chat_response(
        prompt,
        model=NVIDIA_MODEL_NAME,
        system_prompt="Return strict JSON only. The response must be a JSON array and must not include Markdown fences or explanations.",
        temperature=0.2,
    )


def generate_storyboard_json(prompt: str, provider: str) -> tuple[list[dict], str]:
    provider = normalize_provider(provider)
    errors: list[str] = []
    if provider in {"auto", "nvidia"}:
        try:
            raw = generate_storyboard_json_with_nvidia(prompt)
            try:
                return normalize_storyboard_payload(json.loads(extract_json_text(raw))), f"nvidia:{NVIDIA_MODEL_NAME}"
            except Exception as parse_exc:
                debug_path = save_rejected_storyboard_response("nvidia", raw)
                raise ValueError(f"{parse_exc}; raw response saved to {debug_path}") from parse_exc
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            if provider == "nvidia":
                raise RuntimeError("Storyboard generation failed: " + " | ".join(errors)) from exc
            print(f"NVIDIA storyboard generation failed; falling back to OpenAI/ChatGPT: {exc}")

    if provider in {"auto", "openai"}:
        try:
            raw = generate_storyboard_json_with_openai(prompt)
            return normalize_storyboard_payload(json.loads(extract_json_text(raw))), f"openai:{OPENAI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            if provider == "openai":
                raise RuntimeError("Storyboard generation failed: " + " | ".join(errors)) from exc
            print(f"OpenAI storyboard generation failed; falling back to Gemini: {exc}")

    if provider in {"auto", "gemini"}:
        try:
            raw = generate_storyboard_json_with_gemini(prompt)
            return normalize_storyboard_payload(json.loads(extract_json_text(raw))), f"gemini:{GEMINI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            raise RuntimeError("Storyboard generation failed: " + " | ".join(errors)) from exc

    raise RuntimeError("Storyboard generation failed: " + " | ".join(errors))

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


def normalize_word_key(word: str) -> str:
    return str(word or "").strip().lower()


def safe_filename_token(value: str) -> str:
    token = str(value or "").strip()
    token = token.replace(" ", "_").replace("?", "")
    token = token.replace("/", "_").replace("\\", "_")
    token = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-", "."})
    return token or "card"


def safe_pos_token(value: str) -> str:
    token = safe_filename_token(str(value or "").replace(".", ""))
    return token or "pos"


def flashcard_stems_for_vocab_df(df_vocab: pd.DataFrame) -> list[tuple[str, str]]:
    word_counts = df_vocab["Word"].astype(str).str.strip().str.lower().value_counts().to_dict()
    stems: list[tuple[str, str]] = []
    for row_number, (_idx, row) in enumerate(df_vocab.iterrows(), start=1):
        word = str(row.get("Word", "")).strip()
        if not word:
            continue
        base_stem = safe_filename_token(word)
        if word_counts.get(normalize_word_key(word), 0) > 1:
            pos_stem = safe_pos_token(row.get("POS", ""))
            stem = f"{base_stem}__{row_number:02d}_{pos_stem}"
        else:
            stem = base_stem
        stems.append((normalize_word_key(word), stem))
    return stems


def build_flashcard_asset_queues(df_vocab: pd.DataFrame) -> dict[str, list[str]]:
    queues: dict[str, list[str]] = {}
    for word_key, stem in flashcard_stems_for_vocab_df(df_vocab):
        queues.setdefault(word_key, []).append(stem)
    return queues


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


def resolve_flashcard_image_path(
    images_folder: Path,
    flashcard_word: str,
    special_asset_map: dict[str, str],
    flashcard_asset_queues: dict[str, list[str]],
) -> str:
    token = str(flashcard_word or "").strip()
    if token in special_asset_map:
        return special_asset_map[token]
    word_key = normalize_word_key(token)
    stems = flashcard_asset_queues.get(word_key) or []
    safe_word = stems.pop(0) if stems else safe_filename_token(token)
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
) -> tuple[float, float, bool] | None:
    original_start = safe_float(scene.get("start_time"), previous_end)
    original_end = safe_float(scene.get("end_time"), original_start)
    start_t, end_t = normalize_scene_times(scene, previous_end)
    desired_duration = max(end_t - start_t, min_duration)

    repaired = False
    if start_t < previous_end:
        start_t = previous_end
        repaired = True
    elif start_t > previous_end + 0.001:
        start_t = previous_end
        repaired = True

    if start_t >= timeline_end - min_duration:
        if previous_end < timeline_end - min_duration:
            start_t = previous_end
            repaired = True
        else:
            return None

    clipped_end = min(max(end_t, start_t + desired_duration), timeline_end)
    if clipped_end <= start_t + min_duration:
        return None

    if original_start > timeline_end or original_end > timeline_end or abs(start_t - original_start) > 0.001:
        repaired = True
    scene["start_time"] = round(start_t, 3)
    scene["end_time"] = round(clipped_end, 3)
    return scene["start_time"], scene["end_time"], repaired


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


def subtitle_window_text(subs: list, start_t: float, end_t: float) -> str:
    parts = []
    for sub in subs:
        sub_start = sub.start.total_seconds()
        sub_end = sub.end.total_seconds()
        if sub_end <= start_t or sub_start >= end_t:
            continue
        parts.append(sub.content.replace("\n", " "))
    return " ".join(parts).strip()


def select_word_window(
    windows: list[tuple[float, float]],
    *,
    after_seconds: float,
    before_seconds: float | None = None,
    used_windows: list[tuple[float, float]] | None = None,
) -> tuple[float, float] | None:
    if not windows:
        return None
    used_windows = used_windows or []
    candidates = []
    for start_t, end_t in windows:
        if before_seconds is not None and start_t >= before_seconds - 0.001:
            continue
        if end_t < after_seconds - 1.0:
            continue
        candidates.append((start_t, end_t))
    non_overlapping = []
    for start_t, end_t in candidates:
        overlaps = any(not (end_t <= used_start + 0.001 or start_t >= used_end - 0.001) for used_start, used_end in used_windows)
        if not overlaps:
            non_overlapping.append((start_t, end_t))
    if non_overlapping:
        return min(non_overlapping, key=lambda item: (item[0], item[1]))
    if candidates:
        return min(candidates, key=lambda item: (item[0], item[1]))
    return None


def build_storyboard_word_anchors(
    df_vocab: pd.DataFrame,
    subs: list,
    word_windows: dict[str, list[tuple[float, float]]],
) -> list[dict]:
    anchors = []
    prev_answer = previous_answer_window(subs)
    quiz_window = cloze_question_window(subs)
    search_start = prev_answer[1] if prev_answer else 0.0
    quiz_start = quiz_window[0] if quiz_window else None
    used_windows: list[tuple[float, float]] = []
    for row_number, (_idx, row) in enumerate(df_vocab.iterrows(), start=1):
        word = str(row.get("Word", "")).strip()
        if not word:
            continue
        window = select_word_window(
            word_windows.get(normalize_word_key(word)) or [],
            after_seconds=search_start,
            before_seconds=quiz_start,
            used_windows=used_windows,
        )
        if not window:
            continue
        start_t, end_t = window
        used_windows.append((start_t, end_t))
        anchors.append(
            {
                "order": row_number,
                "word": word,
                "pos": str(row.get("POS", "")).strip(),
                "meaning": str(row.get("Meaning", "")).strip(),
                "english_sentence": str(row.get("English_Sentence", "")).strip(),
                "start_time": round(start_t, 3),
                "end_time": round(end_t, 3),
                "subtitle_reference": collect_scene_subtitles(subs, start_t, end_t),
            }
        )
    anchors.sort(key=lambda item: (float(item["start_time"]), int(item["order"])))
    return anchors


def cloze_question_window(subs: list) -> tuple[float, float] | None:
    hits = []
    keywords = ("blank", "選項", "題目", "下一集", "揭曉", "做答")
    for sub in subs:
        text = str(sub.content or "").lower()
        if any(keyword.lower() in text for keyword in keywords):
            hits.append((sub.start.total_seconds(), sub.end.total_seconds()))
    if not hits:
        return None
    start_t = hits[-1][0]
    end_t = hits[-1][1]
    for hit_start, hit_end in reversed(hits[:-1]):
        if start_t - hit_end <= 45.0:
            start_t = hit_start
            end_t = max(end_t, hit_end)
        else:
            break
    return max(0.0, start_t), end_t


def previous_answer_window(subs: list) -> tuple[float, float] | None:
    hits = []
    keywords = ("上一集", "正確答案", "答案", "解答")
    for sub in subs:
        text = str(sub.content or "")
        if any(keyword in text for keyword in keywords):
            hits.append((sub.start.total_seconds(), sub.end.total_seconds()))
    if not hits:
        return None
    start_t = hits[0][0]
    end_t = hits[0][1]
    for hit_start, hit_end in hits[1:]:
        if hit_start - end_t <= 12.0:
            end_t = max(end_t, hit_end)
        else:
            break
    return max(0.0, start_t), end_t


def new_ai_scene(start_t: float, end_t: float, reason: str, subs: list) -> dict:
    return {
        "start_time": round(start_t, 3),
        "end_time": round(end_t, 3),
        "source_type": "AI",
        "flashcard_word": "",
        "custom_image_path": "",
        "image_prompt": "",
        "reason": reason,
        "subtitle_reference": collect_scene_subtitles(subs, start_t, end_t),
    }


def new_flashcard_scene(
    start_t: float,
    end_t: float,
    flashcard_word: str,
    reason: str,
    subs: list,
    images_folder: Path,
    special_asset_map: dict[str, str],
    flashcard_asset_queues: dict[str, list[str]],
) -> dict:
    return {
        "start_time": round(start_t, 3),
        "end_time": round(end_t, 3),
        "source_type": "FLASHCARD",
        "flashcard_word": flashcard_word,
        "custom_image_path": resolve_flashcard_image_path(images_folder, flashcard_word, special_asset_map, flashcard_asset_queues),
        "image_prompt": "",
        "reason": reason,
        "subtitle_reference": collect_scene_subtitles(subs, start_t, end_t),
    }


def build_storyboard_skeleton(
    anchors: list[dict],
    subs: list,
    timeline_end: float,
    images_folder: Path,
    special_asset_map: dict[str, str],
    flashcard_asset_queues: dict[str, list[str]],
) -> list[dict]:
    scenes: list[dict] = []
    flash_scenes: list[dict] = []

    prev_window = previous_answer_window(subs) if PREV_CLOZE_ANSWER_TOKEN in special_asset_map else None
    if prev_window:
        flash_scenes.append(
            new_flashcard_scene(
                prev_window[0],
                prev_window[1],
                PREV_CLOZE_ANSWER_TOKEN,
                "Previous episode cloze answer card.",
                subs,
                images_folder,
                special_asset_map,
                flashcard_asset_queues,
            )
        )

    for anchor in anchors:
        word = str(anchor.get("word", "")).strip()
        flash_scenes.append(
            new_flashcard_scene(
                float(anchor["start_time"]),
                float(anchor["end_time"]),
                word,
                f"Flashcard for vocabulary word '{word}'.",
                subs,
                images_folder,
                special_asset_map,
                flashcard_asset_queues,
            )
        )

    question_window = cloze_question_window(subs) if CURRENT_CLOZE_QUESTION_TOKEN in special_asset_map else None
    if question_window:
        flash_scenes.append(
            new_flashcard_scene(
                question_window[0],
                question_window[1],
                CURRENT_CLOZE_QUESTION_TOKEN,
                "Current episode cloze question card.",
                subs,
                images_folder,
                special_asset_map,
                flashcard_asset_queues,
            )
        )

    flash_scenes.sort(key=lambda item: (float(item["start_time"]), float(item["end_time"])))
    for idx, flash_scene in enumerate(flash_scenes[:-1]):
        word = str(flash_scene.get("flashcard_word", "")).strip()
        if word == CURRENT_CLOZE_QUESTION_TOKEN:
            continue
        next_start = max(0.0, float(flash_scenes[idx + 1]["start_time"]))
        if next_start > float(flash_scene["end_time"]) + 0.05:
            flash_scene["end_time"] = round(min(next_start, timeline_end), 3)
            if word.startswith("__"):
                flash_scene["reason"] = str(flash_scene.get("reason", "")).rstrip(".") + "; extended through the card explanation."
            else:
                flash_scene["reason"] = str(flash_scene.get("reason", "")).rstrip(".") + "; extended through the vocabulary explanation."
            flash_scene["subtitle_reference"] = collect_scene_subtitles(
                subs,
                float(flash_scene["start_time"]),
                float(flash_scene["end_time"]),
            )
    cursor = 0.0
    min_gap = 0.2
    for flash_scene in flash_scenes:
        flash_start = max(0.0, float(flash_scene["start_time"]))
        flash_end = min(timeline_end, max(flash_start + 0.05, float(flash_scene["end_time"])))
        if flash_start > cursor + min_gap:
            scenes.append(new_ai_scene(cursor, flash_start, "AI transition between required flashcard segments.", subs))
        if flash_end <= cursor + 0.05:
            flash_end = min(timeline_end, cursor + 0.05)
        flash_scene["start_time"] = round(max(cursor, flash_start), 3)
        flash_scene["end_time"] = round(flash_end, 3)
        flash_scene["subtitle_reference"] = collect_scene_subtitles(subs, flash_scene["start_time"], flash_scene["end_time"])
        scenes.append(flash_scene)
        cursor = max(cursor, flash_end)
    if cursor < timeline_end - min_gap:
        scenes.append(new_ai_scene(cursor, timeline_end, "AI closing transition and episode wrap-up.", subs))
    for idx, scene in enumerate(scenes, start=1):
        scene["scene_id"] = idx
    return scenes


def ai_enrichment_prompt(scenes: list[dict], host_profile_rules: str) -> str:
    payload = [
        {
            "scene_id": scene.get("scene_id"),
            "start_time": scene.get("start_time"),
            "end_time": scene.get("end_time"),
            "source_type": scene.get("source_type"),
            "reason": scene.get("reason", ""),
            "subtitle_reference": scene.get("subtitle_reference", ""),
        }
        for scene in scenes
        if str(scene.get("source_type", "")).upper() == "AI"
    ]
    return f"""You are enriching a deterministic storyboard skeleton for an English vocabulary teaching video.
Do not add, remove, merge, split, or retime scenes.
Only return JSON for AI scenes. FLASHCARD scenes are intentionally omitted and must not be changed.

For each input AI scene, return:
- scene_id: same integer
- reason: Traditional Chinese, concise explanation of this AI transition
- image_prompt: English only. Use Chinese subtitles as context, but translate the visual idea into English. Do not include Chinese/Japanese/Korean text in image_prompt. End exactly with ", aspect ratio 16:9, cinematic wide shot".

Host/style rules:
{host_profile_rules or "- Use the recurring podcast hosts only when visually appropriate."}

AI scenes to enrich:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def generate_ai_scene_enrichment(prompt: str, provider: str) -> tuple[list[dict], str]:
    scenes, provider_used = generate_storyboard_json(prompt, provider)
    return scenes, provider_used


def apply_ai_enrichment(skeleton: list[dict], enriched_items: list[dict], host_profiles: dict) -> int:
    by_id = {}
    for item in enriched_items:
        try:
            by_id[int(item.get("scene_id"))] = item
        except Exception:
            continue
    updated = 0
    for scene in skeleton:
        if str(scene.get("source_type", "")).upper() != "AI":
            continue
        item = by_id.get(int(scene.get("scene_id", 0) or 0))
        if item:
            scene["reason"] = str(item.get("reason") or scene.get("reason") or "").strip()
            scene["image_prompt"] = str(item.get("image_prompt") or "").strip()
        if not str(scene.get("image_prompt", "")).strip():
            scene["image_prompt"] = fallback_ai_image_prompt(scene)
        if scene_needs_host_consistency(scene):
            scene["image_prompt"] = apply_host_profiles_to_prompt(scene.get("image_prompt", ""), host_profiles)
        scene["image_prompt"] = sanitize_image_prompt(scene)
        updated += 1
    return updated


def validate_storyboard_skeleton(scenes: list[dict], word_list: list[str]) -> list[str]:
    warnings = []
    present_words = {
        normalize_word_key(scene.get("flashcard_word", ""))
        for scene in scenes
        if str(scene.get("source_type", "")).upper() == "FLASHCARD"
    }
    for word in word_list:
        word_key = normalize_word_key(word)
        if word_key and word_key not in present_words:
            warnings.append(f"missing flashcard scene for word: {word}")
    for scene in scenes:
        if str(scene.get("source_type", "")).upper() == "AI" and contains_cjk(scene.get("image_prompt", "")):
            warnings.append(f"AI scene {scene.get('scene_id')} image_prompt still contains CJK text")
    return warnings


def generate_storyboard(ep_num: int, provider: str = "auto") -> bool:
    target_folder, _start_word, _end_word = resolve_episode_range(workspace_dir, ep_num)
    srt_folder = target_folder / "02_subtitles"
    storyboard_folder = target_folder / "03_storyboards"
    images_folder = target_folder / "04_images"
    srt_file = srt_folder / "notebooklm_audio_fixed.srt"
    vocab_csv = storyboard_folder / "vocab_data.csv"
    output_csv = storyboard_folder / "storyboard.csv"
    anchors_json = storyboard_folder / "storyboard_word_anchors.json"
    skeleton_csv = storyboard_folder / "storyboard_skeleton.csv"
    validation_json = storyboard_folder / "storyboard_validation.json"

    if not srt_file.exists():
        print(f"Missing fixed subtitle file: {srt_file}")
        return False
    if not vocab_csv.exists():
        print(f"Missing vocab_data.csv: {vocab_csv}")
        return False

    print(f"Loading vocab data and fixed subtitle file: {srt_file}")
    df_vocab = pd.read_csv(vocab_csv).fillna("")
    word_list = df_vocab["Word"].astype(str).str.strip().tolist()
    flashcard_asset_queues = build_flashcard_asset_queues(df_vocab)
    srt_text = srt_file.read_text(encoding="utf-8")
    subs = list(srt.parse(srt_text))
    if not subs:
        print(f"SRT file has no subtitle entries: {srt_file}")
        return False
    word_windows = build_word_occurrence_windows(subs, word_list)
    timeline_end = max(sub.end.total_seconds() for sub in subs)
    cloze_card_rules, special_asset_map = build_cloze_card_rules(ep_num, target_folder)
    host_profiles = load_host_profiles()
    host_profile_rules = build_host_profile_rules(host_profiles)

    provider = normalize_provider(provider)
    print(
        f"storyboard_provider={provider} "
        f"gemini_model={GEMINI_MODEL_NAME} openai_model={OPENAI_MODEL_NAME} "
        f"nvidia_model={NVIDIA_MODEL_NAME}"
    )
    try:
        print("14.1 Anchoring vocabulary words from SRT...")
        anchors = build_storyboard_word_anchors(df_vocab, subs, word_windows)
        write_json_file(anchors_json, anchors)
        print(f"Anchors saved to: {anchors_json} ({len(anchors)}/{len(word_list)} words)")

        print("14.2 Building deterministic storyboard skeleton...")
        skeleton = build_storyboard_skeleton(
            anchors,
            subs,
            timeline_end,
            images_folder,
            special_asset_map,
            flashcard_asset_queues,
        )
        write_storyboard_csv(skeleton_csv, skeleton)
        print(f"Skeleton saved to: {skeleton_csv} ({len(skeleton)} scenes)")

        print("14.3 Enriching AI scenes with LLM prompts/reasons...")
        provider_used = "local:fallback"
        try:
            enrichment_prompt = ai_enrichment_prompt(skeleton, host_profile_rules)
            enriched_items, provider_used = generate_ai_scene_enrichment(enrichment_prompt, provider)
            enriched_count = apply_ai_enrichment(skeleton, enriched_items, host_profiles)
            print(f"AI enrichment provider used: {provider_used}; enriched {enriched_count} AI scene(s).")
        except Exception as enrich_exc:
            print(f"WARN AI enrichment failed; using deterministic fallback image prompts. raw_error={enrich_exc}")
            enriched_count = apply_ai_enrichment(skeleton, [], host_profiles)
            print(f"Fallback enriched {enriched_count} AI scene(s).")

        print("14.4 Validating storyboard...")
        validation_warnings = validate_storyboard_skeleton(skeleton, word_list)
        write_json_file(
            validation_json,
            {
                "provider_used": provider_used,
                "anchor_count": len(anchors),
                "expected_word_count": len([word for word in word_list if str(word).strip()]),
                "scene_count": len(skeleton),
                "warnings": validation_warnings,
            },
        )
        if validation_warnings:
            for warning in validation_warnings:
                print(f"WARN {warning}")

        for i, scene in enumerate(skeleton, start=1):
            scene["scene_id"] = i
        storyboard_data = skeleton
        write_storyboard_csv(output_csv, storyboard_data)

        print(f"Storyboard saved to: {output_csv}")
        if special_asset_map:
            print("Special flashcard assets:")
            for token, path in special_asset_map.items():
                print(f"- {token}: {path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="Episode number")
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_PROVIDERS),
        default=os.getenv("CAP_STORYBOARD_PROVIDER", "auto"),
        help="LLM provider for storyboard generation. Use openai for ChatGPT.",
    )
    args = parser.parse_args()
    raise SystemExit(0 if generate_storyboard(args.ep, args.provider) else 1)

