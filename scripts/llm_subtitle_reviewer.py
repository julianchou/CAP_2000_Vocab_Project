import argparse
from datetime import timedelta
import os
from pathlib import Path
import re

import srt
from dotenv import load_dotenv

from episode_range_utils import resolve_episode_range
from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response


load_dotenv()

base_dir = Path(__file__).resolve().parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
GEMINI_MODEL_NAME = os.getenv("CAP_SUBTITLE_REVIEW_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL_NAME = os.getenv("CAP_SUBTITLE_REVIEW_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL_NAME = os.getenv("CAP_SUBTITLE_REVIEW_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)
NVIDIA_FALLBACK_MODEL_NAME = os.getenv(
    "CAP_SUBTITLE_REVIEW_NVIDIA_FALLBACK_MODEL",
    os.getenv("CAP_STORYBOARD_NVIDIA_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
)
VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_SUBTITLE_REVIEW_PROVIDER") or "openai").strip().lower()
    return provider if provider in VALID_PROVIDERS else "openai"


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "spending cap" in text.lower() or "quota" in text.lower()


def is_nvidia_retryable_timeout(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text or "504" in text


def strip_srt_fence(text: str) -> str:
    return str(text or "").replace("```srt", "").replace("```", "").strip()


def review_with_gemini(prompt: str) -> str:
    from google import genai

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print(f"Calling Gemini for subtitle review: {GEMINI_MODEL_NAME}")
    response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    return strip_srt_fence(getattr(response, "text", "") or "")


def review_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"Calling OpenAI/ChatGPT for subtitle review: {OPENAI_MODEL_NAME}")
    response = OpenAI().responses.create(
        model=OPENAI_MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "Return only valid SRT subtitle text. Do not include Markdown fences or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return strip_srt_fence(response.output_text)


def review_with_nvidia(prompt: str) -> str:
    print(f"Calling NVIDIA for subtitle review: {NVIDIA_MODEL_NAME}")
    try:
        raw = nvidia_chat_response(
            prompt,
            model=NVIDIA_MODEL_NAME,
            system_prompt="Return only valid SRT subtitle text. Do not include Markdown fences or explanations.",
            temperature=0.2,
        )
    except Exception as exc:
        fallback_model = str(NVIDIA_FALLBACK_MODEL_NAME or "").strip()
        if (
            not fallback_model
            or fallback_model == NVIDIA_MODEL_NAME
            or not is_nvidia_retryable_timeout(exc)
        ):
            raise
        print(
            "NVIDIA subtitle review timed out; "
            f"retrying with fallback model: {fallback_model}"
        )
        raw = nvidia_chat_response(
            prompt,
            model=fallback_model,
            system_prompt="Return only valid SRT subtitle text. Do not include Markdown fences or explanations.",
            temperature=0.2,
        )
    return strip_srt_fence(raw)

def detect_profile_id() -> str:
    profile_id = str(os.environ.get("CAP_PROFILE_ID", "")).strip()
    if profile_id:
        return profile_id
    if workspace_dir.parent.name == "workspaces" and workspace_dir.name:
        return workspace_dir.name
    return "default"

def shared_prompt_path() -> Path:
    return base_dir / "config" / detect_profile_id() / "prompts" / "subtitle_review_prompt.txt"

DEFAULT_PROMPT_TEMPLATE = """你是一位專業的影片字幕校對專家。請針對以下 SRT 內容進行修正，並務必遵守下列規則：

1. 單字修復與局部合併：
- 僅在單字被切斷時合併，例如同一個英文單字被拆成兩個字幕序號。
- 嚴禁因語意相近就合併不同段落。
- 嚴禁跨句、跨段任意合併。

2. 時間軸保護：
- 除非你真的合併了相鄰字幕，否則必須保留原本的開始時間與結束時間。
- 不可任意改動分鐘、秒數或順序。
- 不可讓字幕突然跳到很後面的時間。
- 不可遺漏大段中間字幕內容。
- 校對後的最後一筆字幕結束時間，必須與原始 SRT 大致一致。

3. 全面校對：
- 修正英文拼字錯誤。
- 根據上下文修正常見音近誤辨。
- 關鍵單字可視情況加上引號，但不要破壞 SRT 結構。

4. 格式要求：
- 保持標準 SRT 格式。
- 若因修復切字而局部合併，請重新整理序號。
- 只輸出純 SRT，不要輸出 Markdown、解說或任何額外文字。

【原始字幕內容】
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


def render_prompt(template_text: str, raw_srt_content: str) -> str:
    return template_text.replace("{{SRT_CONTENT}}", raw_srt_content)


def parse_srt_or_raise(srt_text: str):
    items = list(srt.parse(srt_text))
    if not items:
        raise ValueError("parsed subtitle list is empty")
    return items


def validate_cleaned_srt(raw_srt_content: str, cleaned_srt: str) -> tuple[bool, str]:
    try:
        raw_items = parse_srt_or_raise(raw_srt_content)
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception as e:
        return False, f"SRT parse failed: {e}"

    raw_last_end = raw_items[-1].end.total_seconds()
    cleaned_last_end = cleaned_items[-1].end.total_seconds()
    if abs(cleaned_last_end - raw_last_end) > 5.0:
        return False, (
            f"timeline end drift too large: raw={raw_last_end:.3f}s "
            f"cleaned={cleaned_last_end:.3f}s"
        )

    raw_max_gap = 0.0
    for prev_item, next_item in zip(raw_items, raw_items[1:]):
        raw_max_gap = max(raw_max_gap, next_item.start.total_seconds() - prev_item.end.total_seconds())

    cleaned_max_gap = 0.0
    previous_end = -1.0
    for item in cleaned_items:
        start_sec = item.start.total_seconds()
        end_sec = item.end.total_seconds()
        if end_sec <= start_sec:
            return False, f"non-positive subtitle duration at index {item.index}"
        if previous_end >= 0:
            if start_sec < previous_end:
                return False, f"subtitle timeline overlaps at index {item.index}"
            cleaned_max_gap = max(cleaned_max_gap, start_sec - previous_end)
        previous_end = end_sec

    if cleaned_max_gap > max(raw_max_gap + 5.0, 20.0):
        return False, (
            f"unexpected large subtitle gap: raw_max_gap={raw_max_gap:.3f}s "
            f"cleaned_max_gap={cleaned_max_gap:.3f}s"
        )

    return True, "ok"


TIMECODE_RE = re.compile(r"(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}),(?P<millis>\d{3})")


def repair_short_video_hour_rollover(raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    """Repair LLM outputs that misrender 00:10:xx as 01:00:xx for short videos."""
    try:
        raw_items = parse_srt_or_raise(raw_srt_content)
    except Exception:
        return cleaned_srt, None

    raw_last_end = raw_items[-1].end.total_seconds()
    if raw_last_end >= 3600:
        return cleaned_srt, None

    replacements = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        millis = int(match.group("millis"))
        if hours == 0 or minutes >= 10:
            return match.group(0)

        repaired_minutes = hours * 10 + minutes
        if repaired_minutes >= 60:
            return match.group(0)

        replacements += 1
        return f"00:{repaired_minutes:02d}:{seconds:02d},{millis:03d}"

    repaired_srt = TIMECODE_RE.sub(repl, cleaned_srt)
    if replacements == 0:
        return cleaned_srt, None

    return repaired_srt, f"repaired {replacements} timestamp(s) with short-video rollover normalization"


def repair_non_positive_durations(raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    """Repair cues whose start time is at or after their end time."""
    try:
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    repairs = 0
    for item in cleaned_items:
        if item.end > item.start:
            continue

        drift_seconds = item.start.total_seconds() - item.end.total_seconds()
        minute_steps = max(int(drift_seconds // 60) + 1, 1)
        candidate_start = item.start - timedelta(minutes=minute_steps)
        if candidate_start < timedelta(0) or candidate_start >= item.end:
            return cleaned_srt, None

        item.start = candidate_start
        repairs += 1

    if repairs == 0:
        return cleaned_srt, None

    repaired_srt = srt.compose(cleaned_items)
    return repaired_srt, f"repaired {repairs} non-positive subtitle duration(s)"


def repair_timeline_overlaps(raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    """Repair overlapping subtitle cues without altering the subtitle text."""
    try:
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    repairs = 0
    for previous_item, current_item in zip(cleaned_items, cleaned_items[1:]):
        if current_item.start >= previous_item.end:
            continue

        if current_item.start > previous_item.start:
            previous_item.end = current_item.start
            repairs += 1
            continue

        if current_item.end > previous_item.end:
            current_item.start = previous_item.end
            repairs += 1
            continue

        return cleaned_srt, None

    for item in cleaned_items:
        if item.end <= item.start:
            item.end = item.start + timedelta(milliseconds=20)

    if repairs == 0:
        return cleaned_srt, None

    repaired_srt = srt.compose(cleaned_items)
    return repaired_srt, f"repaired {repairs} overlapping timestamp boundary/boundaries"


def repair_timestamps_from_raw_alignment(raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    """Keep LLM text edits but restore timestamps when cue alignment is unchanged."""
    try:
        raw_items = parse_srt_or_raise(raw_srt_content)
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    if len(raw_items) != len(cleaned_items):
        return cleaned_srt, None

    changed_timestamps = 0
    for raw_item, cleaned_item in zip(raw_items, cleaned_items):
        if cleaned_item.start != raw_item.start or cleaned_item.end != raw_item.end:
            changed_timestamps += 1
        cleaned_item.index = raw_item.index
        cleaned_item.start = raw_item.start
        cleaned_item.end = raw_item.end

    if changed_timestamps == 0:
        return cleaned_srt, None

    repaired_srt = srt.compose(cleaned_items)
    return repaired_srt, f"restored timestamps for {changed_timestamps} aligned subtitle cue(s)"


def fallback_to_raw_srt(raw_srt_content: str, _cleaned_srt: str) -> tuple[str, str | None]:
    """Use the original SRT when the LLM changed cue structure too much to repair safely."""
    ok, message = validate_cleaned_srt(raw_srt_content, raw_srt_content)
    if not ok:
        return _cleaned_srt, None
    return raw_srt_content.strip(), "used original SRT as safe fallback after unrecoverable LLM timeline drift"


def validate_or_repair_llm_srt(raw_srt_content: str, cleaned_srt: str, output_srt: Path) -> tuple[str, str]:
    ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
    if not ok:
        repaired_srt, repair_note = repair_short_video_hour_rollover(raw_srt_content, cleaned_srt)
        if repair_note:
            cleaned_srt = repaired_srt
            ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
            if ok:
                message = f"ok ({repair_note})"
    if not ok:
        repaired_srt, repair_note = repair_non_positive_durations(raw_srt_content, cleaned_srt)
        if repair_note:
            cleaned_srt = repaired_srt
            ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
            if ok:
                message = f"ok ({repair_note})"
    if not ok:
        repaired_srt, repair_note = repair_timeline_overlaps(raw_srt_content, cleaned_srt)
        if repair_note:
            cleaned_srt = repaired_srt
            ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
            if ok:
                message = f"ok ({repair_note})"
    if not ok:
        repaired_srt, repair_note = repair_timestamps_from_raw_alignment(raw_srt_content, cleaned_srt)
        if repair_note:
            cleaned_srt = repaired_srt
            ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
            if ok:
                message = f"ok ({repair_note})"
    if not ok:
        rejected_path = output_srt.with_suffix(".rejected.srt")
        rejected_path.write_text(cleaned_srt, encoding="utf-8")
        repaired_srt, repair_note = fallback_to_raw_srt(raw_srt_content, cleaned_srt)
        if repair_note:
            cleaned_srt = repaired_srt
            ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
            if ok:
                message = f"ok ({repair_note}; rejected copy saved to {rejected_path})"
        if not ok:
            raise ValueError(
                f"AI subtitle output rejected: {message}. rejected copy saved to {rejected_path}"
            )
    return cleaned_srt, message


def call_subtitle_review_llm(prompt: str, provider: str) -> tuple[str, str]:
    provider = normalize_provider(provider)
    errors: list[str] = []
    if provider in {"auto", "gemini"}:
        try:
            return review_with_gemini(prompt), f"gemini:{GEMINI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print("Gemini quota/spending cap reached; falling back to OpenAI/ChatGPT.")
            else:
                print(f"Gemini subtitle review failed; falling back to OpenAI/ChatGPT: {exc}")

    if provider in {"auto", "openai"}:
        try:
            return review_with_openai(prompt), f"openai:{OPENAI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            raise RuntimeError("Subtitle review failed: " + " | ".join(errors)) from exc

    if provider == "nvidia":
        try:
            return review_with_nvidia(prompt), f"nvidia:{NVIDIA_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            raise RuntimeError("Subtitle review failed: " + " | ".join(errors)) from exc

    raise RuntimeError("Subtitle review failed: " + " | ".join(errors))


def review_subtitles_with_llm(ep_num: int, provider: str = "openai") -> bool:
    target_folder, _start_word, _end_word = resolve_episode_range(workspace_dir, ep_num)
    input_srt = target_folder / "02_subtitles" / "notebooklm_audio.srt"
    output_srt = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"

    if not input_srt.exists():
        print(f"Missing raw subtitle file: {input_srt}")
        return False

    print(f"Reviewing subtitles for episode {ep_num}...")
    raw_srt_content = input_srt.read_text(encoding="utf-8")
    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = render_prompt(prompt_template, raw_srt_content)

    provider = normalize_provider(provider)
    print(
        f"subtitle_review_provider={provider} "
        f"gemini_model={GEMINI_MODEL_NAME} openai_model={OPENAI_MODEL_NAME} "
        f"nvidia_model={NVIDIA_MODEL_NAME} "
        f"nvidia_fallback_model={NVIDIA_FALLBACK_MODEL_NAME}"
    )
    try:
        cleaned_srt, provider_used = call_subtitle_review_llm(prompt, provider)
        cleaned_srt, message = validate_or_repair_llm_srt(raw_srt_content, cleaned_srt, output_srt)
        output_srt.write_text(cleaned_srt, encoding="utf-8")
        print(f"Subtitle review provider used: {provider_used}")
        print(f"Subtitle review validation: {message}")
        print(f"Subtitle review completed: {output_srt}")
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
        default=os.getenv("CAP_SUBTITLE_REVIEW_PROVIDER", "openai"),
        help="LLM provider for subtitle review. Use openai for ChatGPT.",
    )
    args = parser.parse_args()
    raise SystemExit(0 if review_subtitles_with_llm(args.ep, args.provider) else 1)
