import argparse
import csv
import mimetypes
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

client = genai.Client()
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(BASE_DIR / "workspace")))
VIDEO_MODEL = os.environ.get("CAP_ANIMATION_MODEL", "veo-2.0-generate-001")
POLL_SECONDS = int(os.environ.get("CAP_ANIMATION_POLL_SECONDS", "10"))
DEFAULT_ANIMATION_SECONDS = int(os.environ.get("CAP_ANIMATION_SECONDS", "5"))


def find_episode_folder(ep_num: int) -> Path | None:
    return next(WORKSPACE_DIR.glob(f"Ep{ep_num:02d}_*"), None)


def read_storyboard_rows(csv_path: Path) -> tuple[list[dict], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def next_animation_variant_path(animation_dir: Path, scene_id: str) -> Path:
    variants = sorted(animation_dir.glob(f"scene_{scene_id}__v*.mp4"))
    if not variants:
        return animation_dir / f"scene_{scene_id}__v01.mp4"
    last = variants[-1]
    marker = last.stem.split("__v")[-1]
    try:
        next_no = int(marker) + 1
    except Exception:
        next_no = len(variants) + 1
    return animation_dir / f"scene_{scene_id}__v{next_no:02d}.mp4"


def guess_mime_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/png"


def scene_image_path(episode_folder: Path, row: dict) -> Path | None:
    custom_image_path = str(row.get("custom_image_path", "")).strip()
    if custom_image_path and Path(custom_image_path).exists():
        return Path(custom_image_path)
    start_token = str(row.get("start_time", "")).strip()
    ai_image_path = episode_folder / "04_images" / "ai_generated" / f"img_{start_token}.png"
    return ai_image_path if ai_image_path.exists() else None


def strip_audio_track(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "copy",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )


def generate_scene_animation(ep_num: int, scene_id: str) -> None:
    episode_folder = find_episode_folder(ep_num)
    if not episode_folder:
        raise FileNotFoundError(f"找不到第 {ep_num:02d} 集資料夾")

    storyboard_csv = episode_folder / "03_storyboards" / "storyboard.csv"
    if not storyboard_csv.exists():
        raise FileNotFoundError(f"找不到 storyboard.csv: {storyboard_csv}")

    rows, _fieldnames = read_storyboard_rows(storyboard_csv)
    target_row = next((row for row in rows if str(row.get("scene_id", "")).strip() == str(scene_id)), None)
    if not target_row:
        raise ValueError(f"找不到 scene_id={scene_id}")

    animation_prompt = str(target_row.get("animation_prompt", "")).strip()
    if not animation_prompt:
        raise ValueError("animation_prompt 為空，請先產生或編輯動畫 Prompt")

    image_path = scene_image_path(episode_folder, target_row)
    if not image_path or not image_path.exists():
        raise FileNotFoundError("找不到可用的 scene 圖片，請先確認圖卡或 AI 圖已就緒")

    animation_dir = episode_folder / "04_images" / "animations"
    animation_dir.mkdir(parents=True, exist_ok=True)
    output_path = next_animation_variant_path(animation_dir, scene_id)

    with image_path.open("rb") as f:
        image_bytes = f.read()

    scene_duration = 0
    try:
        start_time = float(target_row.get("start_time", 0) or 0)
        end_time = float(target_row.get("end_time", 0) or 0)
        scene_duration = int(max(end_time - start_time, 0))
    except Exception:
        scene_duration = 0
    request_duration = max(min(scene_duration, 8), 4) if scene_duration else DEFAULT_ANIMATION_SECONDS

    print("STEP 1/4 validate inputs")
    print(f"scene_id={scene_id}")
    print(f"image_path={image_path}")
    print(f"output_path={output_path}")
    print(f"duration_seconds={request_duration}")
    print("STEP 2/4 submit image-to-video request")

    operation = client.models.generate_videos(
        model=VIDEO_MODEL,
        source=types.GenerateVideosSource(
            prompt=animation_prompt,
            image=types.Image(
                image_bytes=image_bytes,
                mime_type=guess_mime_type(image_path),
            ),
        ),
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=request_duration,
            aspect_ratio="16:9",
        ),
    )
    print(f"operation_submitted done={operation.done}")

    poll_count = 0
    while not operation.done:
        poll_count += 1
        print(f"STEP 3/4 polling operation poll_count={poll_count} wait_seconds={POLL_SECONDS}")
        time.sleep(POLL_SECONDS)
        operation = client.operations.get(operation)
        print(f"operation_status done={operation.done}")

    result = operation.result or operation.response
    if not result or not getattr(result, "generated_videos", None):
        raise RuntimeError("動畫生成完成，但沒有回傳可下載的影片")

    print("STEP 4/4 download generated video")
    generated_video = result.generated_videos[0]
    video_bytes = client.files.download(file=generated_video)
    raw_output_path = output_path.with_name(output_path.stem + "__raw" + output_path.suffix)
    raw_output_path.write_bytes(video_bytes)

    print("STEP 4.5/4 strip audio track from generated video")
    strip_audio_track(raw_output_path, output_path)
    try:
        raw_output_path.unlink()
    except FileNotFoundError:
        pass

    print("DONE animation generated")
    print(f"saved_to={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="為指定 scene 產生無聲動畫")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()
    generate_scene_animation(args.ep, str(args.scene_id))


if __name__ == "__main__":
    main()
