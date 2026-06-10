import argparse
import csv
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
NVIDIA_MODEL_NAME = os.getenv(
    "CAP_SUBTITLE_REVIEW_NVIDIA_MODEL",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
)
NVIDIA_FALLBACK_MODEL_NAME = os.getenv(
    "CAP_SUBTITLE_REVIEW_NVIDIA_FALLBACK_MODEL",
    "nvidia/llama-3.3-nemotron-super-49b-v1,nvidia/llama-3.1-nemotron-nano-8b-v1",
)
VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
SUBTITLE_REVIEW_CHUNK_CUES = max(int(os.getenv("CAP_SUBTITLE_REVIEW_CHUNK_CUES", "90") or "90"), 20)
SUBTITLE_REVIEW_CHUNK_OVERLAP = max(int(os.getenv("CAP_SUBTITLE_REVIEW_CHUNK_OVERLAP", "0") or "0"), 0)


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_SUBTITLE_REVIEW_PROVIDER") or "auto").strip().lower()
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


def review_with_nvidia(prompt: str) -> tuple[str, str]:
    print(f"Calling NVIDIA for subtitle review: {NVIDIA_MODEL_NAME}")
    model_order = [NVIDIA_MODEL_NAME]
    for fallback_model in str(NVIDIA_FALLBACK_MODEL_NAME or "").split(","):
        clean_model = fallback_model.strip()
        if clean_model and clean_model not in model_order:
            model_order.append(clean_model)

    errors = []
    for selected_model in model_order:
        try:
            if selected_model != NVIDIA_MODEL_NAME:
                print(f"Retrying NVIDIA subtitle review with fallback model: {selected_model}")
            raw = nvidia_chat_response(
                prompt,
                model=selected_model,
                system_prompt="Return only valid SRT subtitle text. Do not include Markdown fences or explanations.",
                temperature=0.2,
            )
            return strip_srt_fence(raw), selected_model
        except Exception as exc:
            errors.append(f"{selected_model}: {exc}")
            if selected_model == model_order[-1]:
                raise RuntimeError("NVIDIA subtitle review failed: " + " | ".join(errors)) from exc
            if is_nvidia_retryable_timeout(exc):
                print(f"NVIDIA subtitle review timed out for {selected_model}; trying next model.")
            else:
                print(f"NVIDIA subtitle review failed for {selected_model}; trying next model: {exc}")

    raise RuntimeError("NVIDIA subtitle review failed: " + " | ".join(errors))

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

0. 參考資料：
- 「No.8 產生語音摘要 Prompt」是本集 NotebookLM 語音內容的規劃與重點摘要，可用來判斷主題、單字、例句、段落順序與中英文脈絡。
- 參考資料只用於校對文字，不可把未出現在字幕音訊中的內容新增進 SRT。
- 若原始字幕與參考資料衝突，優先保留原始字幕的實際口語內容與時間軸。

1. 單字修復與局部合併：
- 不要合併、刪除、拆分或重排任何字幕 cue。
- No.12 的 LLM 只負責校正每個 cue 內的文字；英文斷句合併會由系統後處理完成。
- 嚴禁因語意相近就合併不同段落或不同句子。

2. 時間軸保護：
- 必須保留每一筆原始字幕的序號、開始時間與結束時間。
- 輸出 cue 數量必須與原始 SRT 完全相同。
- 不可任意改動分鐘、秒數或順序。
- 不可讓字幕突然跳到很後面的時間。
- 不可遺漏大段中間字幕內容。
- 校對後的最後一筆字幕結束時間，必須與原始 SRT 大致一致。

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

5. 格式要求：
- 保持標準 SRT 格式。
- 每一個字幕 cue 必須固定為：序號一行、時間軸一行、字幕文字一到多行、空白行。時間軸那一行只能有時間，不可把字幕文字接在時間軸後面。
- 若因修復切字而局部合併，請重新整理序號。
- 只輸出純 SRT，不要輸出 Markdown、解說或任何額外文字。

【No.8 產生語音摘要 Prompt 參考】
{{NOTEBOOKLM_PROMPT}}

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


def notebooklm_prompt_output_path(target_folder: Path, ep_num: int) -> Path:
    return target_folder / f"notebooklm_prompt_ep{ep_num:02d}.txt"


def read_notebooklm_prompt_context(target_folder: Path, ep_num: int) -> str:
    prompt_path = notebooklm_prompt_output_path(target_folder, ep_num)
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8", errors="ignore").strip()


def render_prompt(template_text: str, raw_srt_content: str, notebooklm_prompt: str = "") -> str:
    rendered = (
        template_text
        .replace("{{SRT_CONTENT}}", raw_srt_content)
        .replace("{{NOTEBOOKLM_PROMPT}}", notebooklm_prompt or "（未找到 No.8 output，請僅依原始字幕內容校對。）")
    )
    if "{{NOTEBOOKLM_PROMPT}}" not in template_text and notebooklm_prompt:
        rendered = (
            rendered
            + "\n\n【No.8 產生語音摘要 Prompt 參考】\n"
            + notebooklm_prompt
            + "\n"
        )
    return rendered


def compose_reindexed_srt(items: list[srt.Subtitle], start_index: int = 1) -> str:
    chunk_items = []
    for offset, item in enumerate(items):
        chunk_items.append(
            srt.Subtitle(
                index=start_index + offset,
                start=item.start,
                end=item.end,
                content=item.content,
                proprietary=item.proprietary,
            )
        )
    return srt.compose(chunk_items).strip() + "\n"


def split_subtitle_items(items: list[srt.Subtitle], chunk_size: int, overlap: int = 0) -> list[list[srt.Subtitle]]:
    if not items:
        return []
    clean_chunk_size = max(int(chunk_size or 0), 20)
    clean_overlap = max(min(int(overlap or 0), clean_chunk_size - 1), 0)
    chunks = []
    start = 0
    while start < len(items):
        end = min(start + clean_chunk_size, len(items))
        chunks.append(items[start:end])
        if end >= len(items):
            break
        start = end - clean_overlap
    return chunks


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

    for expected_index, item in enumerate(cleaned_items, start=1):
        if item.index != expected_index:
            return False, f"subtitle index sequence is not normalized at item {expected_index}: got {item.index}"

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
TIMECODE_LINE_RE = re.compile(
    r"^\s*(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+(?P<text>.*?))?\s*$"
)


def normalize_srt_structure(srt_text: str) -> tuple[str, str | None]:
    """Rebuild SRT blocks when an LLM puts cue text on timestamp lines or mixes indices."""
    normalized_text = re.sub(
        r"(?m)^\s*\d+\s*[\.)]\s*(?=\d{2}:\d{2}:\d{2},\d{3}\s+-->)",
        "",
        str(srt_text or "").replace("\ufeff", ""),
    )
    lines = normalized_text.splitlines()
    cues: list[srt.Subtitle] = []
    current_start = None
    current_end = None
    current_text: list[str] = []
    repairs = 0

    def flush_current() -> None:
        nonlocal current_start, current_end, current_text
        if current_start is None or current_end is None:
            current_text = []
            return
        content = "\n".join(line.strip() for line in current_text if line.strip()).strip()
        cues.append(
            srt.Subtitle(
                index=len(cues) + 1,
                start=current_start,
                end=current_end,
                content=content,
            )
        )
        current_start = None
        current_end = None
        current_text = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        match = TIMECODE_LINE_RE.match(stripped)
        if match:
            flush_current()
            current_start = srt.srt_timestamp_to_timedelta(match.group("start"))
            current_end = srt.srt_timestamp_to_timedelta(match.group("end"))
            inline_text = (match.group("text") or "").strip()
            if inline_text:
                current_text.append(inline_text)
                repairs += 1
            continue

        if not stripped:
            flush_current()
            continue

        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if stripped.isdigit() and TIMECODE_LINE_RE.match(next_line):
            continue

        if current_start is not None:
            current_text.append(stripped)

    flush_current()

    if not cues:
        return srt_text, None

    for index, item in enumerate(cues, start=1):
        item.index = index

    normalized = srt.compose(cues).strip() + "\n"
    if normalized.strip() == str(srt_text or "").strip() and repairs == 0:
        return srt_text, None
    return normalized, f"normalized malformed SRT structure for {len(cues)} cue(s)"


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


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
ENGLISH_SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]*$')


def normalize_cjk_spacing_text(text: str) -> str:
    text = re.sub(
        r"(?<=[\u3400-\u4dbf\u4e00-\u9fff])[\t ]+(?=[\u3400-\u4dbf\u4e00-\u9fff])",
        "",
        str(text or ""),
    )
    text = re.sub(r"[\t ]+([，。！？；：、])", r"\1", text)
    text = re.sub(r"([，。！？；：、])[\t ]+(?=[\u3400-\u4dbf\u4e00-\u9fff])", r"\1", text)
    text = re.sub(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff])[\t ]+([）】』」》])", r"\1", text)
    text = re.sub(r"([（【『「《])[\t ]+(?=[\u3400-\u4dbf\u4e00-\u9fff])", r"\1", text)
    return text


def repair_cjk_spacing(_raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    try:
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    repairs = 0
    for item in cleaned_items:
        if not CJK_RE.search(item.content):
            continue
        normalized = normalize_cjk_spacing_text(item.content)
        if normalized != item.content:
            item.content = normalized
            repairs += 1

    if repairs == 0:
        return cleaned_srt, None

    return srt.compose(cleaned_items), f"normalized Chinese spacing in {repairs} subtitle cue(s)"


def normalize_english_punctuation_text(text: str) -> str:
    if LATIN_RE.search(text) and not CJK_RE.search(text):
        text = re.sub(r"。+$", ".", text)
    return text


def is_english_fragment_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped or CJK_RE.search(stripped) or not LATIN_RE.search(stripped):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9 '\",;:\-()]+[.!?。]?", stripped))


def has_english_sentence_end(text: str) -> bool:
    stripped = normalize_english_punctuation_text(str(text or "").strip())
    return bool(ENGLISH_SENTENCE_END_RE.search(stripped))


def should_merge_english_fragment(current_group: list, next_item) -> bool:
    if not current_group or not is_english_fragment_text(next_item.content):
        return False
    if has_english_sentence_end(current_group[-1].content):
        return False
    gap = next_item.start.total_seconds() - current_group[-1].end.total_seconds()
    if gap > 0.35:
        return False
    merged_text = " ".join(item.content.strip() for item in [*current_group, next_item])
    if len(merged_text) > 120:
        return False
    duration = next_item.end.total_seconds() - current_group[0].start.total_seconds()
    if duration > 8.0:
        return False
    first_char = next_item.content.strip()[:1]
    return bool(first_char and (first_char.islower() or len(current_group) >= 2))


def repair_english_sentence_fragments(_raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    try:
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    normalized_punctuation = 0
    for item in cleaned_items:
        normalized = normalize_english_punctuation_text(item.content)
        if normalized != item.content:
            item.content = normalized
            normalized_punctuation += 1

    if normalized_punctuation == 0:
        return cleaned_srt, None

    return (
        srt.compose(cleaned_items),
        f"normalized {normalized_punctuation} English punctuation mark(s)",
    )


DEFAULT_SUBTITLE_TERM_CANDIDATES = {
    "Dining Room",
    "Dinner",
    "Difficulty",
    "Dinosaur",
    "delete",
    "appear",
    "appeared",
    "built",
    "disease",
    "ease",
    "honest",
    "invented",
    "jumped",
    "scientist",
    "scientists",
    "slept",
}
SPACED_ASCII_TERM_RE = re.compile(r"(?<![A-Za-z])([A-Za-z]+(?:\s+[A-Za-z]+)+)(?![A-Za-z])")
COMMON_SUBTITLE_TYPO_REPLACEMENTS = {
    "Scientist discovered": "Scientists discovered",
    "加上this就是": "加上dis就是",
    "菜窯": "菜餚",
    "sh h": "sh",
}


def normalize_subtitle_term_key(text: str) -> str:
    return re.sub(r"[^A-Za-z]+", "", str(text or "")).lower()


def add_subtitle_terms_from_text(term_candidates: set[str], text: str) -> None:
    for token in re.findall(r"[A-Za-z][A-Za-z'-]*", str(text or "")):
        cleaned = token.strip("'\".,!?;:()[]{}")
        if len(cleaned) >= 3:
            term_candidates.add(cleaned)


def subtitle_term_candidates_for(target_folder: Path) -> set[str]:
    term_candidates = set(DEFAULT_SUBTITLE_TERM_CANDIDATES)
    csv_paths = [
        target_folder / "03_storyboards" / "vocab_data.csv",
        target_folder / "03_storyboards" / "cloze_questions.csv",
    ]
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    for value in row.values():
                        add_subtitle_terms_from_text(term_candidates, value)
        except Exception:
            continue
    for prompt_path in sorted(target_folder.glob("notebooklm_prompt_ep*.txt")):
        try:
            add_subtitle_terms_from_text(term_candidates, prompt_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
    return term_candidates


def repair_spaced_known_terms(
    _raw_srt_content: str,
    cleaned_srt: str,
    term_candidates: set[str] | None = None,
) -> tuple[str, str | None]:
    try:
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    term_map = {
        normalize_subtitle_term_key(term): term
        for term in (term_candidates or DEFAULT_SUBTITLE_TERM_CANDIDATES)
        if len(normalize_subtitle_term_key(term)) >= 3
    }
    if not term_map:
        return cleaned_srt, None

    repairs = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal repairs
        original = match.group(1)
        chunks = original.split()
        repaired_chunks: list[str] = []
        idx = 0
        while idx < len(chunks):
            replacement = None
            replacement_end = idx
            for end in range(len(chunks), idx + 1, -1):
                candidate_key = normalize_subtitle_term_key("".join(chunks[idx:end]))
                replacement = term_map.get(candidate_key)
                if replacement:
                    replacement_end = end
                    break
            if replacement and replacement_end > idx + 1:
                if chunks[idx][:1].isupper() and replacement[:1].islower():
                    replacement = replacement[:1].upper() + replacement[1:]
                repaired_chunks.append(replacement)
                repairs += 1
                idx = replacement_end
            else:
                repaired_chunks.append(chunks[idx])
                idx += 1
        return " ".join(repaired_chunks)

    for item in cleaned_items:
        repaired = SPACED_ASCII_TERM_RE.sub(repl, item.content)
        if repaired != item.content:
            item.content = repaired

    if repairs == 0:
        return cleaned_srt, None

    return srt.compose(cleaned_items), f"repaired {repairs} spaced English term(s)"


def repair_common_subtitle_typos(_raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    try:
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    repairs = 0
    for item in cleaned_items:
        repaired = item.content
        for source, replacement in COMMON_SUBTITLE_TYPO_REPLACEMENTS.items():
            repaired = repaired.replace(source, replacement)
        if repaired != item.content:
            item.content = repaired
            repairs += 1

    if repairs == 0:
        return cleaned_srt, None

    return srt.compose(cleaned_items), f"repaired {repairs} common subtitle typo(s)"


ALIGNMENT_TEXT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_alignment_text(text: str) -> str:
    return ALIGNMENT_TEXT_RE.sub("", str(text or "")).lower()


def detect_adjacent_text_shift(raw_items: list, cleaned_items: list) -> str | None:
    """Detect cue text that moved into the previous timestamp while cue count stayed valid."""
    if len(raw_items) != len(cleaned_items):
        return None

    raw_texts = [normalize_alignment_text(item.content) for item in raw_items]
    cleaned_texts = [normalize_alignment_text(item.content) for item in cleaned_items]

    for idx in range(len(raw_texts) - 2):
        current_raw = raw_texts[idx]
        next_raw = raw_texts[idx + 1]
        following_raw = raw_texts[idx + 2]
        current_cleaned = cleaned_texts[idx]
        next_cleaned = cleaned_texts[idx + 1]

        if min(len(current_raw), len(next_raw), len(following_raw)) < 3:
            continue
        if not (CJK_RE.search(raw_items[idx].content) or CJK_RE.search(raw_items[idx + 1].content)):
            continue

        current_absorbed_next = current_raw in current_cleaned and next_raw in current_cleaned
        next_shifted_forward = next_raw not in next_cleaned and following_raw in next_cleaned
        if current_absorbed_next and next_shifted_forward:
            return (
                f"cue {raw_items[idx].index} absorbed cue {raw_items[idx + 1].index} text; "
                f"cue {cleaned_items[idx + 1].index} then shifted forward"
            )

    return None


def fallback_to_raw_when_alignment_changed(
    raw_srt_content: str,
    cleaned_srt: str,
    *,
    allow_count_change: bool = False,
) -> tuple[str, str | None]:
    try:
        raw_items = parse_srt_or_raise(raw_srt_content)
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    if len(raw_items) != len(cleaned_items):
        if allow_count_change:
            return cleaned_srt, None
        return (
            raw_srt_content.strip(),
            f"LLM changed cue count raw={len(raw_items)} cleaned={len(cleaned_items)}; used raw SRT timing as safe base",
        )

    shift_message = detect_adjacent_text_shift(raw_items, cleaned_items)
    if not shift_message:
        return cleaned_srt, None

    return (
        raw_srt_content.strip(),
        f"LLM shifted adjacent cue text ({shift_message}); used raw SRT timing as safe base",
    )


def fallback_to_raw_srt(raw_srt_content: str, _cleaned_srt: str) -> tuple[str, str | None]:
    """Use the original SRT when the LLM changed cue structure too much to repair safely."""
    ok, message = validate_cleaned_srt(raw_srt_content, raw_srt_content)
    if not ok:
        return _cleaned_srt, None
    return raw_srt_content.strip(), "used original SRT as safe fallback after unrecoverable LLM timeline drift"


def validate_or_repair_llm_srt(
    raw_srt_content: str,
    cleaned_srt: str,
    output_srt: Path,
    term_candidates: set[str] | None = None,
) -> tuple[str, str]:
    cleaned_srt, structure_note = normalize_srt_structure(cleaned_srt)
    cleaned_srt, alignment_note = fallback_to_raw_when_alignment_changed(raw_srt_content, cleaned_srt)
    cleaned_srt, cjk_note = repair_cjk_spacing(raw_srt_content, cleaned_srt)
    cleaned_srt, term_note = repair_spaced_known_terms(raw_srt_content, cleaned_srt, term_candidates)
    cleaned_srt, typo_note = repair_common_subtitle_typos(raw_srt_content, cleaned_srt)
    cleaned_srt, english_note = repair_english_sentence_fragments(raw_srt_content, cleaned_srt)
    cleaned_srt, post_repair_alignment_note = fallback_to_raw_when_alignment_changed(raw_srt_content, cleaned_srt)
    ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
    repair_notes = [
        note
        for note in [structure_note, alignment_note, cjk_note, term_note, typo_note, english_note, post_repair_alignment_note]
        if note
    ]
    if ok and repair_notes:
        message = f"ok ({'; '.join(repair_notes)})"
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
        raise ValueError(
            f"AI subtitle output rejected: {message}. rejected copy saved to {rejected_path}"
        )
    return cleaned_srt, message


def call_subtitle_review_llm(prompt: str, provider: str) -> tuple[str, str]:
    provider = normalize_provider(provider)
    errors: list[str] = []
    if provider in {"auto", "nvidia"}:
        try:
            cleaned_srt, selected_model = review_with_nvidia(prompt)
            return cleaned_srt, f"nvidia:{selected_model}"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            if provider == "nvidia":
                raise RuntimeError("Subtitle review failed: " + " | ".join(errors)) from exc
            print(f"NVIDIA subtitle review failed; falling back to OpenAI/ChatGPT: {exc}")

    if provider in {"auto", "openai"}:
        try:
            return review_with_openai(prompt), f"openai:{OPENAI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            if provider == "openai":
                raise RuntimeError("Subtitle review failed: " + " | ".join(errors)) from exc
            print(f"OpenAI subtitle review failed; falling back to Gemini: {exc}")

    if provider in {"auto", "gemini"}:
        try:
            return review_with_gemini(prompt), f"gemini:{GEMINI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            raise RuntimeError("Subtitle review failed: " + " | ".join(errors)) from exc

    raise RuntimeError("Subtitle review failed: " + " | ".join(errors))


def subtitle_review_provider_sequence(provider: str) -> list[str]:
    provider = normalize_provider(provider)
    if provider == "auto":
        return ["nvidia", "openai", "gemini"]
    return [provider]


def review_subtitles_chunked(
    *,
    raw_srt_content: str,
    prompt_template: str,
    notebooklm_prompt: str,
    output_srt: Path,
    term_candidates: set[str],
    provider_candidate: str,
    chunk_size: int = SUBTITLE_REVIEW_CHUNK_CUES,
    chunk_overlap: int = SUBTITLE_REVIEW_CHUNK_OVERLAP,
) -> tuple[str, str, str]:
    raw_items = parse_srt_or_raise(raw_srt_content)
    chunks = split_subtitle_items(raw_items, chunk_size, chunk_overlap)
    if len(chunks) <= 1:
        prompt = render_prompt(prompt_template, raw_srt_content, notebooklm_prompt)
        cleaned_srt, provider_used = call_subtitle_review_llm(prompt, provider_candidate)
        cleaned_srt, message = validate_or_repair_llm_srt(
            raw_srt_content,
            cleaned_srt,
            output_srt,
            term_candidates,
        )
        return cleaned_srt, provider_used, message

    print(
        "Subtitle review chunk mode: "
        f"provider={provider_candidate} chunks={len(chunks)} "
        f"chunk_cues={chunk_size} overlap={chunk_overlap}"
    )
    final_items: list[srt.Subtitle] = []
    notes: list[str] = []
    provider_used_values: list[str] = []
    seen_indexes: set[int] = set()

    for chunk_index, original_chunk in enumerate(chunks, start=1):
        original_start = int(original_chunk[0].index)
        original_end = int(original_chunk[-1].index)
        chunk_raw_srt = compose_reindexed_srt(original_chunk, start_index=1)
        chunk_prompt = render_prompt(prompt_template, chunk_raw_srt, notebooklm_prompt)
        print(
            f"Subtitle review chunk {chunk_index}/{len(chunks)} "
            f"raw_cues={original_start}-{original_end} provider={provider_candidate}"
        )
        try:
            cleaned_chunk_srt, provider_used = call_subtitle_review_llm(chunk_prompt, provider_candidate)
            cleaned_chunk_srt, chunk_message = validate_or_repair_llm_srt(
                chunk_raw_srt,
                cleaned_chunk_srt,
                output_srt,
                term_candidates,
            )
            provider_used_values.append(provider_used)
        except Exception as exc:
            if provider_candidate != "nvidia":
                raise
            cleaned_chunk_srt, repair_message = validate_or_repair_llm_srt(
                chunk_raw_srt,
                chunk_raw_srt,
                output_srt,
                term_candidates,
            )
            chunk_message = f"chunk raw fallback after NVIDIA failure: {exc}; {repair_message}"
            provider_used_values.append("nvidia:chunk-raw-fallback")
            print(f"Subtitle review chunk {chunk_index} used raw fallback: {exc}")

        cleaned_items = parse_srt_or_raise(cleaned_chunk_srt)
        if len(cleaned_items) != len(original_chunk):
            cleaned_items = parse_srt_or_raise(chunk_raw_srt)
            chunk_message = (
                f"{chunk_message}; chunk cue count changed "
                f"cleaned={len(cleaned_items)} raw={len(original_chunk)}; used raw chunk"
            )

        for cleaned_item, original_item in zip(cleaned_items, original_chunk):
            original_index = int(original_item.index)
            if original_index in seen_indexes:
                continue
            cleaned_item.index = original_index
            cleaned_item.start = original_item.start
            cleaned_item.end = original_item.end
            final_items.append(cleaned_item)
            seen_indexes.add(original_index)
        notes.append(f"chunk {chunk_index}: {chunk_message}")

    final_items.sort(key=lambda item: int(item.index))
    for expected_index, item in enumerate(final_items, start=1):
        item.index = expected_index
    final_srt = srt.compose(final_items).strip() + "\n"
    ok, message = validate_cleaned_srt(raw_srt_content, final_srt)
    if not ok:
        raise ValueError(f"chunked subtitle review output rejected: {message}")
    provider_used_summary = provider_used_values[0] if len(set(provider_used_values)) == 1 else "mixed:" + ",".join(sorted(set(provider_used_values)))
    return final_srt, provider_used_summary, f"ok (chunked {len(chunks)} chunk(s); " + "; ".join(notes) + ")"


def review_subtitles_with_llm(ep_num: int, provider: str = "auto") -> bool:
    target_folder, _start_word, _end_word = resolve_episode_range(workspace_dir, ep_num)
    input_srt = target_folder / "02_subtitles" / "notebooklm_audio.srt"
    output_srt = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"

    if not input_srt.exists():
        print(f"Missing raw subtitle file: {input_srt}")
        return False

    print(f"Reviewing subtitles for episode {ep_num}...")
    raw_srt_content = input_srt.read_text(encoding="utf-8")
    notebooklm_prompt = read_notebooklm_prompt_context(target_folder, ep_num)
    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    if notebooklm_prompt:
        print(f"Loaded No.8 NotebookLM prompt context: {notebooklm_prompt_output_path(target_folder, ep_num)}")
    else:
        print("No.8 NotebookLM prompt context not found; reviewing subtitles with SRT only.")

    provider = normalize_provider(provider)
    print(
        f"subtitle_review_provider={provider} "
        f"gemini_model={GEMINI_MODEL_NAME} openai_model={OPENAI_MODEL_NAME} "
        f"nvidia_model={NVIDIA_MODEL_NAME} "
        f"nvidia_fallback_model={NVIDIA_FALLBACK_MODEL_NAME}"
    )
    term_candidates = subtitle_term_candidates_for(target_folder)
    errors: list[str] = []
    provider_sequence = subtitle_review_provider_sequence(provider)
    for provider_candidate in provider_sequence:
        try:
            cleaned_srt, provider_used, message = review_subtitles_chunked(
                raw_srt_content=raw_srt_content,
                prompt_template=prompt_template,
                notebooklm_prompt=notebooklm_prompt,
                output_srt=output_srt,
                term_candidates=term_candidates,
                provider_candidate=provider_candidate,
            )
            output_srt.write_text(cleaned_srt, encoding="utf-8")
            print(f"Subtitle review provider used: {provider_used}")
            print(f"Subtitle review validation: {message}")
            print(f"Subtitle review completed: {output_srt}")
            return True
        except Exception as e:
            errors.append(f"{provider_candidate}: {e}")
            if provider != "auto":
                print(f"Error: {e}")
                return False
            remaining = provider_sequence[provider_sequence.index(provider_candidate) + 1:]
            if remaining:
                print(
                    f"Auto subtitle review provider {provider_candidate} failed; "
                    f"falling back to {remaining[0]}: {e}"
                )

    if provider == "auto":
        try:
            raw_fallback, message = fallback_to_raw_srt(raw_srt_content, "")
            output_srt.write_text(raw_fallback.strip() + "\n", encoding="utf-8")
            print("Subtitle review provider used: raw-fallback")
            print(
                "Subtitle review validation: "
                f"{message}; all AI providers failed: {' | '.join(errors)}"
            )
            print(f"Subtitle review completed with raw fallback: {output_srt}")
            return True
        except Exception as e:
            errors.append(f"raw-fallback: {e}")

    print("Error: Subtitle review failed: " + " | ".join(errors))
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="Episode number")
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_PROVIDERS),
        default=os.getenv("CAP_SUBTITLE_REVIEW_PROVIDER", "auto"),
        help="LLM provider for subtitle review. Use openai for ChatGPT.",
    )
    args = parser.parse_args()
    raise SystemExit(0 if review_subtitles_with_llm(args.ep, args.provider) else 1)
