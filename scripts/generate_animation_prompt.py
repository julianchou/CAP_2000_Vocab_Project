import argparse
import csv
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()

client = genai.Client()
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(BASE_DIR / "workspace")))
TEXT_MODEL = os.environ.get("CAP_TEXT_MODEL", "gemini-2.5-flash")
MAX_PROMPT_WORDS = 90
RISKY_TERMS = [
    "attack",
    "blood",
    "blocked",
    "chase",
    "collapse",
    "danger",
    "dangerous",
    "dead",
    "death",
    "distress",
    "explode",
    "explosion",
    "fear",
    "fight",
    "fighting",
    "fire",
    "gun",
    "hurt",
    "injured",
    "injury",
    "knife",
    "panic",
    "panicked",
    "poison",
    "poisonous",
    "recoil",
    "recoils",
    "scream",
    "screaming",
    "shiver",
    "shivers",
    "shock",
    "terrified",
    "threat",
    "threatening",
    "toxic",
    "violence",
    "violent",
    "vomit",
    "weapon",
    "wound",
]


def find_episode_folder(ep_num: int) -> Path | None:
    return next(WORKSPACE_DIR.glob(f"Ep{ep_num:02d}_*"), None)


def read_storyboard_rows(csv_path: Path) -> tuple[list[dict], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_storyboard_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    for extra_col in ["animation_prompt", "animation_video_path"]:
        if extra_col not in fieldnames:
            fieldnames.append(extra_col)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.strip("\"'`")
    return text


def trim_words(text: str, limit: int = MAX_PROMPT_WORDS) -> str:
    words = normalize_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(",.;: ") + "."


def contains_risky_terms(text: str) -> list[str]:
    lowered = f" {normalize_text(text).lower()} "
    hits = []
    for term in RISKY_TERMS:
        if f" {term} " in lowered:
            hits.append(term)
    return hits


def safe_scene_hint(scene: dict, max_words: int = 14) -> str:
    candidates = [
        str(scene.get("flashcard_word", "")).strip(),
        str(scene.get("subtitle_reference", "")).strip(),
        str(scene.get("reason", "")).strip(),
        str(scene.get("image_prompt", "")).strip(),
    ]
    for candidate in candidates:
        candidate = normalize_text(candidate)
        if not candidate:
            continue
        candidate = trim_words(candidate, max_words)
        if contains_risky_terms(candidate):
            continue
        return candidate
    return "the original educational scene"


def build_prompt_request(scene: dict) -> str:
    return f"""
You are writing a policy-safe image-to-video motion prompt for a 16:9 educational YouTube video scene.

Output rules:
- Output exactly one English prompt only.
- Do not use bullet points, JSON, markdown, labels, or quotes.
- Keep it under {MAX_PROMPT_WORDS} words.
- Preserve the existing subjects, composition, props, and educational context.
- Describe only subtle, calm motion, camera movement, atmosphere, and continuity.
- Make the scene feel suitable for a silent background animation.

Safety rules:
- Keep all characters calm, friendly, and mild.
- Avoid any wording about danger, fear, panic, pain, sickness, poison, injury, violence, threat, warning symbols, alarms, collisions, or distress.
- Avoid strong reactions or dramatic cause-and-effect.
- If the scene implies something negative, convert it into a neutral educational moment with gentle gestures and infographic movement.

Scene info:
- source_type: {scene.get("source_type", "")}
- flashcard_word: {scene.get("flashcard_word", "")}
- reason: {scene.get("reason", "")}
- image_prompt: {scene.get("image_prompt", "")}
- subtitle_reference: {scene.get("subtitle_reference", "")}
""".strip()


def fallback_prompt(scene: dict) -> str:
    hint = safe_scene_hint(scene)
    prompt = (
        f"A calm educational animation of {hint}, with subtle breathing, gentle head turns, "
        "small hand gestures, soft infographic motion, and slow camera drift. Keep the original "
        "subjects, layout, props, and colors unchanged. Maintain friendly expressions, soft studio "
        "lighting, light background parallax, and smooth Pixar-like continuity throughout."
    )
    return trim_words(prompt)


def sanitize_generated_prompt(text: str) -> str:
    prompt = normalize_text(text)
    prompt = re.sub(r"^[\-\*\d\.\)\s]+", "", prompt)
    prompt = trim_words(prompt)
    return prompt


def is_compliant_prompt(text: str) -> tuple[bool, str]:
    prompt = normalize_text(text)
    if not prompt:
        return False, "empty"
    if len(prompt.split()) > MAX_PROMPT_WORDS:
        return False, "too_long"
    hits = contains_risky_terms(prompt)
    if hits:
        return False, f"risky_terms={','.join(hits)}"
    if any(token in prompt for token in ["\n", "{", "}", "[", "]"]):
        return False, "structured_output"
    return True, "ok"


def generate_animation_prompt(ep_num: int, scene_id: str) -> None:
    episode_folder = find_episode_folder(ep_num)
    if not episode_folder:
        raise FileNotFoundError(f"找不到第 {ep_num:02d} 集資料夾")

    storyboard_csv = episode_folder / "03_storyboards" / "storyboard.csv"
    if not storyboard_csv.exists():
        raise FileNotFoundError(f"找不到 storyboard.csv: {storyboard_csv}")

    rows, fieldnames = read_storyboard_rows(storyboard_csv)
    target_row = next((row for row in rows if str(row.get("scene_id", "")).strip() == str(scene_id)), None)
    if not target_row:
        raise ValueError(f"找不到 scene_id={scene_id}")

    print(f"Generating safe animation prompt for Scene {scene_id}...")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=build_prompt_request(target_row),
    )
    model_prompt = sanitize_generated_prompt((getattr(response, "text", "") or "").strip())
    ok, reason = is_compliant_prompt(model_prompt)

    if ok:
        suggested_prompt = model_prompt
        print("prompt_source=model")
    else:
        suggested_prompt = fallback_prompt(target_row)
        print(f"prompt_source=fallback reason={reason}")

    target_row["animation_prompt"] = suggested_prompt
    write_storyboard_rows(storyboard_csv, rows, fieldnames)
    print(f"saved_to={storyboard_csv}")
    print(suggested_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="為指定 scene 產生合規動畫 Prompt")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()
    generate_animation_prompt(args.ep, str(args.scene_id))


if __name__ == "__main__":
    main()
