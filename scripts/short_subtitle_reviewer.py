import argparse
import os
import re
from datetime import timedelta
from pathlib import Path

import srt
from dotenv import load_dotenv


VALID_PROVIDERS = {"auto", "gemini", "openai"}
GEMINI_MODEL = os.getenv("CAP_SHORT_SUBTITLE_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL = os.getenv("CAP_SHORT_SUBTITLE_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))


def normalize_provider(value: str | None) -> str:
    provider = str(value or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def strip_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:srt|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_srt_or_raise(srt_text: str) -> list[srt.Subtitle]:
    items = list(srt.parse(srt_text))
    if not items:
        raise ValueError("parsed subtitle list is empty")
    return items


def validate_cleaned_srt(raw_srt_content: str, cleaned_srt: str) -> tuple[bool, str]:
    try:
        raw_items = parse_srt_or_raise(raw_srt_content)
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception as exc:
        return False, f"SRT parse failed: {exc}"

    raw_last_end = raw_items[-1].end.total_seconds()
    cleaned_last_end = cleaned_items[-1].end.total_seconds()
    if abs(cleaned_last_end - raw_last_end) > 5.0:
        return False, f"timeline end drift too large: raw={raw_last_end:.3f}s cleaned={cleaned_last_end:.3f}s"

    previous_end = -1.0
    for item in cleaned_items:
        start_sec = item.start.total_seconds()
        end_sec = item.end.total_seconds()
        if end_sec <= start_sec:
            return False, f"non-positive subtitle duration at index {item.index}"
        if previous_end >= 0 and start_sec < previous_end:
            return False, f"subtitle timeline overlaps at index {item.index}"
        previous_end = end_sec

    return True, "ok"


def restore_timestamps_if_same_count(raw_srt_content: str, cleaned_srt: str) -> tuple[str, str | None]:
    try:
        raw_items = parse_srt_or_raise(raw_srt_content)
        cleaned_items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None
    if len(raw_items) != len(cleaned_items):
        return cleaned_srt, None
    for raw_item, cleaned_item in zip(raw_items, cleaned_items):
        cleaned_item.index = raw_item.index
        cleaned_item.start = raw_item.start
        cleaned_item.end = raw_item.end
    return srt.compose(cleaned_items), "restored original timestamps"


def fix_non_positive_durations(cleaned_srt: str) -> tuple[str, str | None]:
    try:
        items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None
    repairs = 0
    for item in items:
        if item.end <= item.start:
            item.end = item.start + timedelta(milliseconds=300)
            repairs += 1
    if not repairs:
        return cleaned_srt, None
    return srt.compose(items), f"repaired {repairs} non-positive duration(s)"


def repair_timeline_overlaps(cleaned_srt: str) -> tuple[str, str | None]:
    try:
        items = parse_srt_or_raise(cleaned_srt)
    except Exception:
        return cleaned_srt, None

    repairs = 0
    for previous_item, current_item in zip(items, items[1:]):
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

    for item in items:
        if item.end <= item.start:
            item.end = item.start + timedelta(milliseconds=300)

    if not repairs:
        return cleaned_srt, None
    return srt.compose(items), f"repaired {repairs} overlapping timestamp boundary/boundaries"


def force_safe_timeline(srt_content: str, min_duration_ms: int = 300) -> tuple[str, str | None]:
    try:
        items = parse_srt_or_raise(srt_content)
    except Exception:
        return srt_content, None

    repairs = 0
    min_delta = timedelta(milliseconds=max(int(min_duration_ms), 50))
    previous_end = timedelta(0)
    for idx, item in enumerate(items):
        item.index = idx + 1
        if item.start < previous_end:
            item.start = previous_end
            repairs += 1
        if item.end <= item.start:
            item.end = item.start + min_delta
            repairs += 1
        previous_end = item.end

    if not repairs:
        return srt_content, None
    return srt.compose(items), f"force-repaired {repairs} unsafe timestamp boundary/boundaries"


def repair_and_validate(raw_srt_content: str, cleaned_srt: str, rejected_path: Path) -> str:
    cleaned_srt = strip_fences(cleaned_srt)
    ok, message = validate_cleaned_srt(raw_srt_content, cleaned_srt)
    if ok:
        print(f"[INFO] validation={message}", flush=True)
        return cleaned_srt

    repaired, note = restore_timestamps_if_same_count(raw_srt_content, cleaned_srt)
    if note:
        ok, message = validate_cleaned_srt(raw_srt_content, repaired)
        if ok:
            print(f"[INFO] validation=ok ({note})", flush=True)
            return repaired

    repaired, note = fix_non_positive_durations(cleaned_srt)
    if note:
        ok, message = validate_cleaned_srt(raw_srt_content, repaired)
        if ok:
            print(f"[INFO] validation=ok ({note})", flush=True)
            return repaired

    repaired, note = repair_timeline_overlaps(cleaned_srt)
    if note:
        ok, message = validate_cleaned_srt(raw_srt_content, repaired)
        if ok:
            print(f"[INFO] validation=ok ({note})", flush=True)
            return repaired

    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.write_text(cleaned_srt, encoding="utf-8")
    ok, raw_message = validate_cleaned_srt(raw_srt_content, raw_srt_content)
    if ok:
        print(
            f"[WARN] AI subtitle output rejected: {message}. rejected copy saved to {rejected_path}. "
            "Using original Whisper SRT as safe fallback.",
            flush=True,
        )
        return raw_srt_content.strip()
    repaired_raw, raw_repair_note = force_safe_timeline(raw_srt_content)
    if raw_repair_note:
        ok, repaired_raw_message = validate_cleaned_srt(repaired_raw, repaired_raw)
        if ok:
            print(
                f"[WARN] AI subtitle output rejected: {message}. rejected copy saved to {rejected_path}. "
                f"Raw SRT fallback was invalid ({raw_message}); using repaired raw SRT fallback ({raw_repair_note}).",
                flush=True,
            )
            return repaired_raw.strip()
        raw_message = f"{raw_message}; repaired raw fallback invalid: {repaired_raw_message}"
    raise ValueError(
        f"AI subtitle output rejected: {message}. rejected copy saved to {rejected_path}. "
        f"Raw SRT fallback is invalid: {raw_message}"
    )


def build_prompt(raw_srt_content: str) -> str:
    return f"""
You are a professional subtitle proofreader for short-form videos.

Task:
1. Correct recognition errors, punctuation, spacing, and wording.
2. Keep the original SRT structure, cue order, and timestamps as much as possible.
3. Do not add commentary, Markdown, or code fences.
4. Output valid SRT only.
5. Use Traditional Chinese when the source is Chinese.

SRT:
{raw_srt_content}
""".strip()


def review_with_gemini(prompt: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    client = genai.Client(api_key=api_key)
    print(f"[INFO] provider=gemini model={GEMINI_MODEL}", flush=True)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return str(getattr(response, "text", "") or "")


def review_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"[INFO] provider=openai model={OPENAI_MODEL}", flush=True)
    response = OpenAI().responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You proofread subtitles and output valid SRT only."},
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota" in text or "spending cap" in text or "rate limit" in text


def review_subtitles(input_srt: Path, output_srt: Path, provider: str) -> Path:
    load_dotenv()
    if not input_srt.exists() or input_srt.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty input SRT: {input_srt}")

    raw_srt_content = input_srt.read_text(encoding="utf-8")
    parse_srt_or_raise(raw_srt_content)
    prompt = build_prompt(raw_srt_content)
    provider = normalize_provider(provider)
    errors = []

    cleaned = ""
    if provider in {"auto", "gemini"}:
        try:
            cleaned = review_with_gemini(prompt)
        except Exception as exc:
            message = str(exc)
            errors.append(f"Gemini: {message}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print("[WARN] Gemini quota/spending cap/rate limit exhausted; switching to OpenAI.", flush=True)
            else:
                print(f"[WARN] Gemini failed; switching to OpenAI: {message}", flush=True)

    if not cleaned and provider in {"auto", "openai"}:
        try:
            cleaned = review_with_openai(prompt)
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            raise RuntimeError("; ".join(errors)) from exc

    if not cleaned:
        raise RuntimeError("; ".join(errors) or "AI provider returned empty subtitle text")

    cleaned = repair_and_validate(raw_srt_content, cleaned, output_srt.with_suffix(".rejected.srt"))
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text(cleaned.strip() + "\n", encoding="utf-8")
    print(f"[INFO] output_srt={output_srt}", flush=True)
    print(f"[INFO] written_bytes={output_srt.stat().st_size}", flush=True)
    return output_srt


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 4: AI subtitle proofreading.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default=os.getenv("CAP_SHORT_SUBTITLE_REVIEW_PROVIDER", "auto"))
    args = parser.parse_args()
    review_subtitles(Path(args.input).resolve(), Path(args.output).resolve(), args.provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
