import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response

VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
GEMINI_MODEL = os.getenv("CAP_SHORT_METADATA_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL = os.getenv("CAP_SHORT_METADATA_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL = os.getenv("CAP_SHORT_METADATA_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)


def normalize_provider(value: str | None) -> str:
    provider = str(value or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def read_text(path: Path, limit: int = 20000) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    return text[:limit]


def read_json(path: Path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def strip_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict:
    cleaned = strip_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("metadata response must be a JSON object")
    return data


def compact_storyboard(storyboard) -> list[dict]:
    if isinstance(storyboard, dict) and isinstance(storyboard.get("scenes"), list):
        storyboard = storyboard["scenes"]
    if not isinstance(storyboard, list):
        return []
    rows = []
    for item in storyboard:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "scene": item.get("scene", item.get("scene_id", "")),
                "start": item.get("start", ""),
                "end": item.get("end", ""),
                "subtitle": str(item.get("subtitle", "") or "")[:220],
                "summary": str(item.get("summary", "") or "")[:220],
            }
        )
    return rows


def compact_compose(compose) -> dict:
    if not isinstance(compose, dict):
        return {}
    return {
        "ratio": compose.get("ratio", "9:16"),
        "scene_count": compose.get("scene_count", 0),
        "audio": compose.get("audio", ""),
        "audio_duration_seconds": compose.get("audio_duration_seconds"),
        "output_video": compose.get("output_video", "05_output/final_short.mp4"),
        "warnings": compose.get("warnings", []),
    }


def build_prompt(ep_dir: Path, title: str, subtitle_text: str, storyboard_rows: list[dict], compose_info: dict) -> str:
    return f"""
你是 YouTube Shorts SEO metadata 編輯，請根據短影音內容產生繁體中文 metadata。

輸出規則：
1. 只輸出 JSON，不要 Markdown，不要解釋。
2. JSON 欄位必須包含：
   - title: 字串，適合 YouTube Shorts，上限 70 字，語氣有吸引力但不誇大。
   - description: 字串，2 到 5 行，說明本 Short 內容，包含一行 CTA。
   - hashtags: 字串陣列，8 到 15 個，必須包含 "#Shorts"。
   - tags: 字串陣列，10 到 25 個，不要加 #。
   - hook: 字串，短影音開頭鉤子。
   - pinned_comment: 字串，適合置頂留言。
   - summary: 字串，內容摘要。
   - language: 固定 "zh-TW"。
   - privacy: 固定 "private"。
3. 若內容涉及英文教學，title/description 可混合少量英文關鍵字。
4. 不要捏造影片中沒有的承諾或連結。

集數標題：
{title}

合成資訊 JSON：
{json.dumps(compose_info, ensure_ascii=False)}

分鏡摘要 JSON：
{json.dumps(storyboard_rows, ensure_ascii=False)}

字幕內容：
{subtitle_text[:12000]}
""".strip()


def generate_with_gemini(prompt: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    print(f"[INFO] provider=gemini model={GEMINI_MODEL}", flush=True)
    response = genai.Client(api_key=api_key).models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return str(getattr(response, "text", "") or "")


def generate_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"[INFO] provider=openai model={OPENAI_MODEL}", flush=True)
    response = OpenAI().responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You generate YouTube Shorts metadata and output JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def generate_with_nvidia(prompt: str) -> str:
    print(f"[INFO] provider=nvidia model={NVIDIA_MODEL}", flush=True)
    return nvidia_chat_response(
        prompt,
        model=NVIDIA_MODEL,
        system_prompt="You generate YouTube Shorts metadata and output JSON only.",
    )


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota" in text or "spending cap" in text or "rate limit" in text or "429" in text


def normalize_metadata(data: dict, fallback_title: str, provider_used: str) -> dict:
    title = str(data.get("title") or fallback_title or "Short").strip()[:90]
    description = str(data.get("description") or "").strip()
    hashtags = data.get("hashtags") if isinstance(data.get("hashtags"), list) else []
    hashtags = [str(item).strip() for item in hashtags if str(item).strip()]
    if "#Shorts" not in hashtags:
        hashtags.insert(0, "#Shorts")
    tags = data.get("tags") if isinstance(data.get("tags"), list) else []
    tags = [str(item).strip().lstrip("#") for item in tags if str(item).strip()]
    return {
        "title": title,
        "description": description,
        "hashtags": hashtags[:20],
        "tags": tags[:30],
        "hook": str(data.get("hook") or "").strip(),
        "pinned_comment": str(data.get("pinned_comment") or "").strip(),
        "summary": str(data.get("summary") or "").strip(),
        "language": "zh-TW",
        "privacy": str(data.get("privacy") or "private").strip() or "private",
        "generated_by": provider_used,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def generate_metadata(ep_dir: Path, output_path: Path, provider: str) -> Path:
    load_dotenv()
    provider = normalize_provider(provider)
    ep_data = read_json(ep_dir / "episode.json", {})
    fallback_title = str(ep_data.get("title") or ep_dir.name).strip()
    subtitle_path = ep_dir / "02_subtitles" / "final.srt"
    if not subtitle_path.exists():
        subtitle_path = ep_dir / "02_subtitles" / "reviewed.srt"
    if not subtitle_path.exists():
        subtitle_path = ep_dir / "02_subtitles" / "whisper.srt"
    subtitle_text = read_text(subtitle_path)
    if not subtitle_text:
        raise FileNotFoundError("找不到非空字幕，無法產生 Short metadata。")

    storyboard = read_json(ep_dir / "03_storyboards" / "storyboard.json", [])
    compose = read_json(ep_dir / "05_output" / "compose_list.json", {})
    prompt = build_prompt(ep_dir, fallback_title, subtitle_text, compact_storyboard(storyboard), compact_compose(compose))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_name("short_metadata_prompt.txt").write_text(prompt, encoding="utf-8")

    raw_text = ""
    provider_used = ""
    errors = []
    if provider in {"auto", "gemini"}:
        try:
            raw_text = generate_with_gemini(prompt)
            provider_used = "gemini"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print("[WARN] Gemini quota/spending cap/rate limit exhausted; switching to OpenAI.", flush=True)
            else:
                print(f"[WARN] Gemini metadata generation failed; switching to OpenAI: {exc}", flush=True)

    if not raw_text and provider in {"auto", "openai"}:
        try:
            raw_text = generate_with_openai(prompt)
            provider_used = "openai"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            raise RuntimeError("; ".join(errors)) from exc

    if not raw_text and provider == "nvidia":
        try:
            raw_text = generate_with_nvidia(prompt)
            provider_used = "nvidia"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            raise RuntimeError("; ".join(errors)) from exc

    if not raw_text:
        raise RuntimeError("; ".join(errors) or "metadata generation returned empty response")

    output_path.with_name("short_metadata_ai_response.txt").write_text(raw_text, encoding="utf-8")
    metadata = normalize_metadata(extract_json_object(raw_text), fallback_title, provider_used)
    metadata["source"] = {
        "episode_dir": str(ep_dir),
        "subtitle": str(subtitle_path),
        "storyboard": str(ep_dir / "03_storyboards" / "storyboard.json"),
        "compose_list": str(ep_dir / "05_output" / "compose_list.json"),
    }
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] provider_used={provider_used}", flush=True)
    print(f"[INFO] output_metadata={output_path}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 10: generate YouTube Shorts metadata.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default=os.getenv("CAP_SHORT_METADATA_PROVIDER", "auto"))
    args = parser.parse_args()
    generate_metadata(
        ep_dir=Path(args.ep_dir).resolve(),
        output_path=Path(args.output).resolve(),
        provider=args.provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
