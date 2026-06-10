import argparse
import json
import os
import re
from pathlib import Path

import srt
from dotenv import load_dotenv

from llm_provider_utils import nvidia_chat_response

VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
GEMINI_MODEL = os.getenv("CAP_SHORT_STORYBOARD_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL = os.getenv("CAP_SHORT_STORYBOARD_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL = os.getenv("CAP_SHORT_STORYBOARD_NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")
NVIDIA_FALLBACK_MODEL = os.getenv(
    "CAP_SHORT_STORYBOARD_NVIDIA_FALLBACK_MODEL",
    os.getenv("CAP_SUBTITLE_REVIEW_NVIDIA_FALLBACK_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1"),
)


def normalize_provider(value: str | None) -> str:
    provider = str(value or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def seconds_to_srt_time(seconds: float) -> str:
    millis = int(round(max(float(seconds), 0.0) * 1000))
    hours, rem = divmod(millis, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def read_srt_entries(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty SRT: {path}")
    items = list(srt.parse(path.read_text(encoding="utf-8")))
    if not items:
        raise ValueError(f"SRT contains no entries: {path}")
    rows = []
    for item in items:
        rows.append(
            {
                "index": int(item.index),
                "start": round(item.start.total_seconds(), 3),
                "end": round(item.end.total_seconds(), 3),
                "text": str(item.content or "").strip(),
            }
        )
    return rows


def strip_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_array(text: str) -> list[dict]:
    cleaned = strip_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        data = json.loads(match.group(0))
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        raise ValueError("model response must be a JSON array or object with scenes array")
    return [item for item in data if isinstance(item, dict)]


def read_existing_storyboard(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def resolve_asset_path(ep_dir: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else ep_dir / path


def scene_number(row: dict, fallback: int) -> int:
    try:
        return int(row.get("scene", row.get("scene_id", fallback)) or fallback)
    except Exception:
        return fallback


def row_seconds(row: dict, start_key: str, fallback: float) -> float:
    try:
        value = row.get(start_key)
        if value is None and start_key == "start_seconds":
            value = row.get("start_time")
        if value is None and start_key == "end_seconds":
            value = row.get("end_time")
        return float(value if value is not None else fallback)
    except Exception:
        return fallback


def is_standard_scene_asset(path: Path, scene_no: int) -> bool:
    stem = path.stem.lower()
    return stem in {f"scene_{scene_no:03d}", f"scene_{scene_no}"}


def is_manual_asset_value(ep_dir: Path, row: dict, key: str, scene_no: int) -> bool:
    path = resolve_asset_path(ep_dir, str(row.get(key, "") or ""))
    if not path or not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    if "_upload_" in path.name.lower():
        return True
    return not is_standard_scene_asset(path, scene_no)


def preserved_asset_fields(ep_dir: Path, row: dict, scene_no: int) -> dict:
    preserved = {}
    for key in ("asset", "image_asset", "animation_asset", "animation_video_path"):
        if is_manual_asset_value(ep_dir, row, key, scene_no):
            preserved[key] = str(row.get(key, "") or "").strip()
    return preserved


def overlap_seconds(left: dict, right: dict) -> float:
    left_start = row_seconds(left, "start_seconds", 0.0)
    left_end = row_seconds(left, "end_seconds", left_start)
    right_start = row_seconds(right, "start_seconds", 0.0)
    right_end = row_seconds(right, "end_seconds", right_start)
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def carry_manual_assets(existing_rows: list[dict], scenes: list[dict], output_json: Path) -> int:
    ep_dir = output_json.parent.parent
    candidates = []
    for idx, row in enumerate(existing_rows, 1):
        no = scene_number(row, idx)
        fields = preserved_asset_fields(ep_dir, row, no)
        if fields:
            candidates.append({"idx": idx, "scene_no": no, "row": row, "fields": fields})

    used = set()
    preserved_count = 0
    for idx, scene in enumerate(scenes, 1):
        scene_no = scene_number(scene, idx)
        match = next((item for item in candidates if item["idx"] not in used and item["scene_no"] == scene_no), None)
        if not match:
            scored = [
                (overlap_seconds(scene, item["row"]), item)
                for item in candidates
                if item["idx"] not in used
            ]
            scored = [item for item in scored if item[0] > 0]
            match = max(scored, key=lambda item: item[0])[1] if scored else None
        if not match:
            continue
        scene.update(match["fields"])
        used.add(match["idx"])
        preserved_count += 1
    return preserved_count


def scene_subtitle_reference(entries: list[dict], start: float, end: float) -> str:
    parts = []
    for item in entries:
        if float(item["end"]) <= start or float(item["start"]) >= end:
            continue
        parts.append(str(item.get("text", "")).strip())
    return " ".join(part for part in parts if part).strip()


def default_prompt(summary: str) -> str:
    summary = str(summary or "").strip() or "A visually clear educational short video moment"
    return (
        f"{summary}. Vertical 9:16 short video frame, engaging educational visual, "
        "clear subject, cinematic lighting, crisp details, no subtitles embedded in image."
    )


def normalize_scenes(raw_scenes: list[dict], entries: list[dict], max_seconds: float) -> list[dict]:
    timeline_start = float(entries[0]["start"])
    timeline_end = float(entries[-1]["end"])
    previous_end = timeline_start
    scenes = []
    for raw in raw_scenes:
        try:
            start = float(raw.get("start", raw.get("start_time", previous_end)) or previous_end)
            end = float(raw.get("end", raw.get("end_time", start + max_seconds)) or start + max_seconds)
        except Exception:
            start, end = previous_end, previous_end + max_seconds
        start = max(start, previous_end, timeline_start)
        end = min(max(end, start + 0.3), timeline_end)
        if end <= start:
            continue
        subtitle = str(raw.get("subtitle") or raw.get("subtitle_reference") or "").strip()
        if not subtitle:
            subtitle = scene_subtitle_reference(entries, start, end)
        summary = str(raw.get("summary") or raw.get("description") or subtitle).strip()
        prompt = str(raw.get("prompt") or raw.get("image_prompt") or "").strip()
        if not prompt:
            prompt = default_prompt(summary)
        if "9:16" not in prompt and "vertical" not in prompt.lower():
            prompt = prompt.rstrip(" .") + ". Vertical 9:16 short video frame."
        scenes.append(
            {
                "scene": len(scenes) + 1,
                "start": seconds_to_srt_time(start),
                "end": seconds_to_srt_time(end),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "source_type": "AI",
                "subtitle": subtitle,
                "summary": summary,
                "prompt": prompt,
                "image_prompt": prompt,
                "asset": str(raw.get("asset", "") or ""),
                "reason": str(raw.get("reason", "") or "AI planned from subtitle context"),
            }
        )
        previous_end = end
        if previous_end >= timeline_end:
            break
    return scenes


def fallback_scenes(entries: list[dict], max_seconds: float, reason: str) -> list[dict]:
    scenes = []
    current = []
    start = float(entries[0]["start"])
    end = start
    for item in entries:
        item_end = float(item["end"])
        if current and item_end - start > max_seconds:
            text = " ".join(str(x["text"]).strip() for x in current if str(x["text"]).strip())
            scenes.append(
                {
                    "start": start,
                    "end": end,
                    "subtitle": text,
                    "summary": text[:160],
                    "prompt": default_prompt(text[:160]),
                    "reason": reason,
                }
            )
            current = []
            start = float(item["start"])
        current.append(item)
        end = item_end
    if current:
        text = " ".join(str(x["text"]).strip() for x in current if str(x["text"]).strip())
        scenes.append(
            {
                "start": start,
                "end": end,
                "subtitle": text,
                "summary": text[:160],
                "prompt": default_prompt(text[:160]),
                "reason": reason,
            }
        )
    return normalize_scenes(scenes, entries, max_seconds)


def build_prompt(entries: list[dict], max_seconds: float) -> str:
    compact = [
        {
            "index": item["index"],
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
        }
        for item in entries
    ]
    return f"""
You are a short-form video storyboard director.

Plan visual scenes from the subtitle context.

Rules:
1. Output JSON only. No Markdown and no explanations.
2. Output a JSON array. Each item must include: start, end, subtitle, summary, prompt, reason.
3. start and end are seconds. Keep them within the subtitle timeline.
4. Scenes must be sequential and must not overlap.
5. Prefer scenes of 2 to {max_seconds:g} seconds.
6. prompt must be English and suitable for image generation.
7. Every prompt must describe a vertical 9:16 Short visual. Do not ask to render subtitle text inside the image.
8. Use the subtitle context to choose the visual subject, emotion, action, and setting.

Subtitles JSON:
{json.dumps(compact, ensure_ascii=False)}
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
            {"role": "system", "content": "You plan short video storyboards and output JSON arrays only."},
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def review_with_nvidia(prompt: str, model: str | None = None) -> str:
    selected_model = model or NVIDIA_MODEL
    print(f"[INFO] provider=nvidia model={selected_model}", flush=True)
    return nvidia_chat_response(
        prompt,
        model=selected_model,
        system_prompt="You plan short video storyboards and output JSON arrays only.",
    )


def review_with_nvidia_fallback(prompt: str) -> str:
    models = [NVIDIA_MODEL]
    if NVIDIA_FALLBACK_MODEL and NVIDIA_FALLBACK_MODEL not in models:
        models.append(NVIDIA_FALLBACK_MODEL)

    errors = []
    for index, model in enumerate(models):
        try:
            return review_with_nvidia(prompt, model=model)
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            if index + 1 < len(models):
                print(f"[WARN] NVIDIA storyboard planning failed; trying fallback model: {exc}", flush=True)

    raise RuntimeError("; ".join(errors))


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota" in text or "spending cap" in text or "rate limit" in text


def plan_storyboard(input_srt: Path, output_json: Path, provider: str, max_seconds: float) -> Path:
    load_dotenv()
    entries = read_srt_entries(input_srt)
    existing_scenes = read_existing_storyboard(output_json)
    prompt = build_prompt(entries, max_seconds=max_seconds)
    prompt_path = output_json.with_name("storyboard_prompt.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    provider = normalize_provider(provider)
    errors = []

    raw_text = ""
    provider_used = ""
    if provider in {"auto", "gemini"}:
        try:
            raw_text = review_with_gemini(prompt)
            provider_used = "gemini"
        except Exception as exc:
            message = str(exc)
            errors.append(f"Gemini: {message}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print("[WARN] Gemini quota/spending cap/rate limit exhausted; switching to OpenAI.", flush=True)
            else:
                print(f"[WARN] Gemini storyboard planning failed; switching to OpenAI: {message}", flush=True)

    if not raw_text and provider in {"auto", "openai"}:
        try:
            raw_text = review_with_openai(prompt)
            provider_used = "openai"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            raise RuntimeError("; ".join(errors)) from exc

    if not raw_text and provider == "nvidia":
        try:
            raw_text = review_with_nvidia_fallback(prompt)
            provider_used = "nvidia"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            print(f"[WARN] NVIDIA storyboard planning failed; using local fallback scenes: {exc}", flush=True)

    raw_path = output_json.with_name("storyboard_ai_response.txt")
    if raw_text:
        raw_path.write_text(raw_text, encoding="utf-8")

    try:
        scenes = normalize_scenes(parse_json_array(raw_text), entries, max_seconds=max_seconds)
    except Exception as exc:
        if raw_text:
            print(f"[WARN] AI storyboard response could not be parsed safely: {exc}. Using local fallback scenes.", flush=True)
        else:
            print("[WARN] AI storyboard response is empty. Using local fallback scenes.", flush=True)
        scenes = fallback_scenes(entries, max_seconds=max_seconds, reason=f"local fallback after AI planning issue: {exc if raw_text else 'empty response'}")
        provider_used = provider_used or "fallback"

    if not scenes:
        scenes = fallback_scenes(entries, max_seconds=max_seconds, reason="local fallback because no normalized scenes were produced")
        provider_used = provider_used or "fallback"

    preserved_asset_count = carry_manual_assets(existing_scenes, scenes, output_json)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "provider_requested": provider,
        "provider_used": provider_used,
        "scene_count": len(scenes),
        "input_srt": str(input_srt),
        "output_json": str(output_json),
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_path) if raw_text else "",
        "errors": errors,
        "preserved_manual_asset_count": preserved_asset_count,
    }
    output_json.with_name("storyboard_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] provider_used={provider_used}", flush=True)
    print(f"[INFO] scene_count={len(scenes)}", flush=True)
    print(f"[INFO] preserved_manual_asset_count={preserved_asset_count}", flush=True)
    print(f"[INFO] output_json={output_json}", flush=True)
    return output_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 6: AI context storyboard planning.")
    parser.add_argument("--input-srt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default=os.getenv("CAP_SHORT_STORYBOARD_PROVIDER", "auto"))
    parser.add_argument("--max-seconds", type=float, default=float(os.getenv("CAP_SHORT_STORYBOARD_MAX_SECONDS", "5")))
    args = parser.parse_args()
    plan_storyboard(
        input_srt=Path(args.input_srt).resolve(),
        output_json=Path(args.output).resolve(),
        provider=args.provider,
        max_seconds=max(float(args.max_seconds), 1.0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
