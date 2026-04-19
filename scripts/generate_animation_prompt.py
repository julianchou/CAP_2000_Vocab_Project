import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()

client = genai.Client()
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(BASE_DIR / "workspace")))
TEXT_MODEL = os.environ.get("CAP_TEXT_MODEL", "gemini-2.5-flash")


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


def build_prompt_request(scene: dict) -> str:
    return f"""
You are writing an image-to-video motion prompt for a 16:9 educational YouTube video scene.

Requirements:
- Output exactly one English prompt only.
- Do not use bullet points, JSON, markdown, labels, or quotes.
- Keep it under 90 words.
- Describe subtle motion, camera movement, atmosphere, and continuity.
- Preserve the original scene identity instead of changing the subject.
- Suitable for silent background animation.

Scene info:
- source_type: {scene.get("source_type", "")}
- flashcard_word: {scene.get("flashcard_word", "")}
- reason: {scene.get("reason", "")}
- image_prompt: {scene.get("image_prompt", "")}
- subtitle_reference: {scene.get("subtitle_reference", "")}
""".strip()


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

    print(f"🧠 正在產生 Scene {scene_id} 的動畫 Prompt...")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=build_prompt_request(target_row),
    )
    suggested_prompt = (getattr(response, "text", "") or "").strip()
    if not suggested_prompt:
        raise RuntimeError("模型沒有回傳可用的動畫 Prompt")

    target_row["animation_prompt"] = suggested_prompt
    write_storyboard_rows(storyboard_csv, rows, fieldnames)
    print("✅ 動畫 Prompt 已寫回 storyboard.csv")
    print(suggested_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="為指定 scene 產生動畫 Prompt")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()
    generate_animation_prompt(args.ep, str(args.scene_id))


if __name__ == "__main__":
    main()
