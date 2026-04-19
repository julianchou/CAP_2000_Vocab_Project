import argparse
import os
from pathlib import Path

import srt
from dotenv import load_dotenv
from google import genai

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client()

base_dir = Path(__file__).resolve().parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
MODEL_NAME = "gemini-2.5-pro"

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


def prompt_template_path_for(target_folder: Path) -> Path:
    return target_folder / "02_subtitles" / "subtitle_review_prompt.txt"


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


def review_subtitles_with_llm(ep_num: int):
    target_folder, _start_word, _end_word = resolve_episode_range(workspace_dir, ep_num)
    input_srt = target_folder / "02_subtitles" / "notebooklm_audio.srt"
    output_srt = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"

    if not input_srt.exists():
        print(f"Missing raw subtitle file: {input_srt}")
        return

    print(f"Reviewing subtitles for episode {ep_num}...")
    raw_srt_content = input_srt.read_text(encoding="utf-8")
    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = render_prompt(prompt_template, raw_srt_content)

    print(f"Calling {MODEL_NAME} for subtitle review...")
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        cleaned_srt = response.text.replace("```srt", "").replace("```", "").strip()
        ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
        if not ok:
            rejected_path = output_srt.with_suffix(".rejected.srt")
            rejected_path.write_text(cleaned_srt, encoding="utf-8")
            raise ValueError(
                f"AI subtitle output rejected: {message}. rejected copy saved to {rejected_path}"
            )
        output_srt.write_text(cleaned_srt, encoding="utf-8")
        print(f"Subtitle review completed: {output_srt}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="Episode number")
    args = parser.parse_args()
    review_subtitles_with_llm(args.ep)
