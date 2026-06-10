import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response

VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
TEXT_MODEL = os.getenv("CAP_SHORT_ANIMATION_PROMPT_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-flash"))
OPENAI_MODEL = os.getenv("CAP_SHORT_ANIMATION_PROMPT_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL = os.getenv("CAP_SHORT_ANIMATION_PROMPT_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)


def normalize_provider(value: str | None) -> str:
    provider = str(value or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def read_storyboard(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        raise ValueError("storyboard must be a JSON array")
    return [item for item in data if isinstance(item, dict)]


def write_storyboard(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def scene_matches(row: dict, scene_id: str) -> bool:
    return str(row.get("scene", row.get("scene_id", ""))).strip() == str(scene_id).strip()


def clean_prompt(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'`")
    if len(text.split()) > 90:
        text = " ".join(text.split()[:90]).rstrip(",.;:") + "."
    return text


def fallback_prompt(scene: dict) -> str:
    hint = str(scene.get("summary") or scene.get("subtitle") or scene.get("prompt") or "the educational short scene").strip()
    return clean_prompt(
        f"Subtle cinematic motion for {hint}: gentle camera drift, soft parallax, small natural movement, "
        "stable composition, friendly educational mood, smooth lighting, no text changes, preserve the original image."
    )


def build_request(scene: dict) -> str:
    return f"""
Write one safe image-to-video animation prompt for a vertical 9:16 Short scene.

Rules:
- Output one English sentence only.
- Under 90 words.
- Preserve the image subject, layout, colors, and educational context.
- Describe subtle motion, camera movement, atmosphere, and continuity.
- Do not mention violence, fear, danger, injury, alarms, or distress.

Scene:
summary: {scene.get("summary", "")}
subtitle: {scene.get("subtitle", "")}
image_prompt: {scene.get("image_prompt") or scene.get("prompt", "")}
""".strip()


def generate_with_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    print(f"[INFO] provider=gemini model={TEXT_MODEL}")
    response = genai.Client(api_key=api_key).models.generate_content(model=TEXT_MODEL, contents=prompt)
    return str(getattr(response, "text", "") or "")


def generate_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"[INFO] provider=openai model={OPENAI_MODEL}")
    response = OpenAI().responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "Write one concise image-to-video animation prompt. Return one sentence only."},
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def generate_with_nvidia(prompt: str) -> str:
    print(f"[INFO] provider=nvidia model={NVIDIA_MODEL}")
    return nvidia_chat_response(
        prompt,
        model=NVIDIA_MODEL,
        system_prompt="Write one concise image-to-video animation prompt. Return one sentence only.",
    )


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota" in text or "spending cap" in text or "rate limit" in text or "429" in text


def call_prompt_model(prompt: str, provider: str) -> tuple[str, str]:
    provider = normalize_provider(provider)
    errors: list[str] = []
    if provider in {"auto", "gemini"}:
        try:
            return generate_with_gemini(prompt), "gemini"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print(f"[WARN] Gemini quota/rate limit reached; switching to OpenAI. raw_error={exc}", flush=True)
            else:
                print(f"[WARN] Gemini failed; switching to OpenAI. raw_error={exc}", flush=True)
    if provider in {"auto", "openai"}:
        try:
            return generate_with_openai(prompt), "openai"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            if provider == "openai":
                raise
    if provider in {"auto", "nvidia"}:
        try:
            return generate_with_nvidia(prompt), "nvidia"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            if provider == "nvidia":
                raise
    raise RuntimeError("Animation prompt generation failed: " + " | ".join(errors))


def generate_prompt(storyboard_path: Path, scene_id: str, provider: str = "auto") -> None:
    load_dotenv()
    rows = read_storyboard(storyboard_path)
    target = next((row for row in rows if scene_matches(row, scene_id)), None)
    if not target:
        raise ValueError(f"scene not found: {scene_id}")

    raw_prompt, provider_used = call_prompt_model(build_request(target), provider)
    prompt = clean_prompt(raw_prompt)
    if not prompt:
        prompt = fallback_prompt(target)
        print("[WARN] empty model response; used fallback prompt")
    target["animation_prompt"] = prompt
    write_storyboard(storyboard_path, rows)
    print(f"[INFO] saved_storyboard={storyboard_path}")
    print(f"[INFO] provider_used={provider_used}")
    print(f"[INFO] animation_prompt={prompt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator: suggest animation prompt for one scene.")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--provider", default=os.getenv("CAP_SHORT_ANIMATION_PROMPT_PROVIDER", "auto"), choices=sorted(VALID_PROVIDERS))
    args = parser.parse_args()
    generate_prompt(Path(args.storyboard).resolve(), args.scene_id, args.provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
