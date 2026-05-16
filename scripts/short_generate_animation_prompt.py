import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


TEXT_MODEL = os.getenv("CAP_SHORT_ANIMATION_PROMPT_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-flash"))


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


def generate_prompt(storyboard_path: Path, scene_id: str) -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    rows = read_storyboard(storyboard_path)
    target = next((row for row in rows if scene_matches(row, scene_id)), None)
    if not target:
        raise ValueError(f"scene not found: {scene_id}")

    client = genai.Client(api_key=api_key)
    print(f"[INFO] text_model={TEXT_MODEL}")
    response = client.models.generate_content(model=TEXT_MODEL, contents=build_request(target))
    prompt = clean_prompt(getattr(response, "text", "") or "")
    if not prompt:
        prompt = fallback_prompt(target)
        print("[WARN] empty model response; used fallback prompt")
    target["animation_prompt"] = prompt
    write_storyboard(storyboard_path, rows)
    print(f"[INFO] saved_storyboard={storyboard_path}")
    print(f"[INFO] animation_prompt={prompt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator: suggest animation prompt for one scene.")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--scene-id", required=True)
    args = parser.parse_args()
    generate_prompt(Path(args.storyboard).resolve(), args.scene_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
