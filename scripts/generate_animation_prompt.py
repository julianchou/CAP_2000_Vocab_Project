import argparse
import csv
import os
import re
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
WORKSPACE_DIR = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(BASE_DIR / "workspace")))
VALID_PROVIDERS = {"auto", "gemini", "openai"}
TEXT_MODEL = os.environ.get("CAP_ANIMATION_PROMPT_GEMINI_MODEL", os.environ.get("CAP_TEXT_MODEL", "gemini-2.5-flash"))
OPENAI_TEXT_MODEL = os.environ.get("CAP_ANIMATION_PROMPT_OPENAI_MODEL", os.environ.get("OPENAI_TEXT_MODEL", "gpt-5.2"))
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


def episode_search_roots() -> list[Path]:
    roots = [WORKSPACE_DIR, BASE_DIR / "workspaces" / "story", BASE_DIR / "workspaces" / "vocab", BASE_DIR / "workspace"]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def find_episode_folder(ep_num: int) -> Path | None:
    for root in episode_search_roots():
        target = next(root.glob(f"Ep{ep_num:02d}_*"), None) if root.exists() else None
        if target:
            return target
    return None


def storyboard_csv_for_episode(episode_folder: Path) -> Path:
    story_path = episode_folder / "05_storyboards" / "storyboard.csv"
    if story_path.exists():
        return story_path
    return episode_folder / "03_storyboards" / "storyboard.csv"


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


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_ANIMATION_PROMPT_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "spending cap" in text or "quota" in text or "429" in text


def generate_prompt_with_gemini(prompt: str) -> str:
    from google import genai

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    print(f"animation_prompt_provider=gemini model={TEXT_MODEL}")
    response = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
    )
    return str(getattr(response, "text", "") or "")


def generate_prompt_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"animation_prompt_provider=openai model={OPENAI_TEXT_MODEL}")
    response = OpenAI().responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {
                "role": "system",
                "content": "Return exactly one policy-safe English image-to-video motion prompt. No markdown, labels, bullets, or quotes.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def generate_prompt_text(prompt: str, provider: str) -> tuple[str, str]:
    provider = normalize_provider(provider)
    errors: list[str] = []
    if provider in {"auto", "gemini"}:
        try:
            return generate_prompt_with_gemini(prompt), f"gemini:{TEXT_MODEL}"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print(f"Gemini quota/spending cap reached; falling back to OpenAI: {exc}")
            else:
                print(f"Gemini animation prompt failed; falling back to OpenAI: {exc}")

    if provider in {"auto", "openai"}:
        try:
            return generate_prompt_with_openai(prompt), f"openai:{OPENAI_TEXT_MODEL}"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            raise RuntimeError("animation prompt generation failed: " + " | ".join(errors)) from exc

    raise RuntimeError("animation prompt generation failed: " + " | ".join(errors))


def generate_animation_prompt(ep_num: int, scene_id: str, provider: str = "auto") -> None:
    episode_folder = find_episode_folder(ep_num)
    if not episode_folder:
        raise FileNotFoundError(f"找不到第 {ep_num:02d} 集資料夾")

    storyboard_csv = storyboard_csv_for_episode(episode_folder)
    if not storyboard_csv.exists():
        raise FileNotFoundError(f"找不到 storyboard.csv: {storyboard_csv}")

    rows, fieldnames = read_storyboard_rows(storyboard_csv)
    target_row = next((row for row in rows if str(row.get("scene_id", "")).strip() == str(scene_id)), None)
    if not target_row:
        raise ValueError(f"找不到 scene_id={scene_id}")

    print(f"Generating safe animation prompt for Scene {scene_id}...")
    try:
        raw_text, provider_used = generate_prompt_text(build_prompt_request(target_row), provider)
        model_prompt = sanitize_generated_prompt(raw_text.strip())
        ok, reason = is_compliant_prompt(model_prompt)
    except Exception as exc:
        provider_used = "fallback"
        model_prompt = ""
        ok = False
        reason = f"provider_error={exc}"

    if ok:
        suggested_prompt = model_prompt
        print(f"prompt_source=model provider_used={provider_used}")
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
    parser.add_argument("--provider", choices=sorted(VALID_PROVIDERS), default=os.getenv("CAP_ANIMATION_PROMPT_PROVIDER", "auto"))
    args = parser.parse_args()
    generate_animation_prompt(args.ep, str(args.scene_id), provider=args.provider)


if __name__ == "__main__":
    main()
