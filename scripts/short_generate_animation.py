import argparse
import json
import mimetypes
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


VIDEO_MODEL = os.getenv("CAP_SHORT_VIDEO_MODEL", "veo-2.0-generate-001")
POLL_SECONDS = int(os.getenv("CAP_SHORT_VIDEO_POLL_SECONDS", "10"))
DEFAULT_DURATION_SECONDS = int(os.getenv("CAP_SHORT_VIDEO_SECONDS", "5"))


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


def mime_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/png"


def resolve_path(ep_dir: Path, value: str) -> Path:
    raw = Path(str(value or "").strip())
    return raw if raw.is_absolute() else ep_dir / raw


def next_video_path(ep_dir: Path, scene_id: str) -> Path:
    out_dir = ep_dir / "04_images" / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"scene_{int(scene_id):03d}.mp4" if str(scene_id).isdigit() else out_dir / f"scene_{scene_id}.mp4"
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = base.with_name(f"{base.stem}__v{idx}{base.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def build_config(duration_seconds: int):
    kwargs = {
        "number_of_videos": 1,
        "duration_seconds": max(min(int(duration_seconds), 8), 5),
        "aspect_ratio": "9:16",
    }
    try:
        return types.GenerateVideosConfig(**kwargs)
    except TypeError:
        return types.GenerateVideosConfig(number_of_videos=1)


def download_video(client, target) -> bytes:
    try:
        return client.files.download(file=target)
    except TypeError:
        return client.files.download(target)


def generate_animation(ep_dir: Path, storyboard_path: Path, scene_id: str, image_path_override: str = "") -> Path:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    rows = read_storyboard(storyboard_path)
    target = next((row for row in rows if scene_matches(row, scene_id)), None)
    if not target:
        raise ValueError(f"scene not found: {scene_id}")

    prompt = str(target.get("animation_prompt", "")).strip()
    if not prompt:
        raise ValueError("animation_prompt is empty")
    image_value = image_path_override or str(target.get("asset", "")).strip()
    image_path = resolve_path(ep_dir, image_value)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    duration = DEFAULT_DURATION_SECONDS
    try:
        duration = int(max(float(target.get("end_seconds", 0)) - float(target.get("start_seconds", 0)), 0))
    except Exception:
        pass
    duration = max(min(duration, 8), 5)

    client = genai.Client(api_key=api_key)
    output_path = next_video_path(ep_dir, scene_id)
    print(f"[INFO] video_model={VIDEO_MODEL}")
    print(f"[INFO] scene={scene_id}")
    print(f"[INFO] image_path={image_path}")
    print(f"[INFO] output_path={output_path}")
    print(f"[INFO] duration_seconds={duration}")
    print(f"[INFO] prompt={prompt[:500]}")

    image_bytes = image_path.read_bytes()
    operation = client.models.generate_videos(
        model=VIDEO_MODEL,
        source=types.GenerateVideosSource(
            prompt=prompt,
            image=types.Image(image_bytes=image_bytes, mime_type=mime_type(image_path)),
        ),
        config=build_config(duration),
    )
    while not getattr(operation, "done", False):
        print(f"[INFO] polling wait_seconds={POLL_SECONDS}", flush=True)
        time.sleep(POLL_SECONDS)
        operation = client.operations.get(operation)

    result = getattr(operation, "result", None) or getattr(operation, "response", None)
    if getattr(operation, "error", None):
        raise RuntimeError(f"video operation failed: {operation.error}")
    generated_videos = getattr(result, "generated_videos", None) if result else None
    if not generated_videos:
        raise RuntimeError("video operation returned no generated_videos")
    generated_video = generated_videos[0]
    download_target = getattr(generated_video, "video", None) or generated_video
    video_bytes = download_video(client, download_target)
    if not video_bytes:
        raise RuntimeError("downloaded video is empty")
    output_path.write_bytes(video_bytes)
    target["animation_asset"] = output_path.relative_to(ep_dir).as_posix()
    target["animation_video_path"] = target["animation_asset"]
    write_storyboard(storyboard_path, rows)
    print(f"[INFO] animation_asset={target['animation_asset']}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator: generate animation for one scene.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--image", default="")
    args = parser.parse_args()
    generate_animation(
        ep_dir=Path(args.ep_dir).resolve(),
        storyboard_path=Path(args.storyboard).resolve(),
        scene_id=args.scene_id,
        image_path_override=args.image,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
