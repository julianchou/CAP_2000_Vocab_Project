import argparse
import base64
import json
import mimetypes
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

VALID_ANIMATION_PROVIDERS = {"gemini", "openai", "nvidia"}
VIDEO_MODEL = os.getenv("CAP_SHORT_VIDEO_MODEL", "veo-2.0-generate-001")
OPENAI_VIDEO_MODEL = os.getenv("CAP_SHORT_OPENAI_VIDEO_MODEL", os.getenv("CAP_OPENAI_VIDEO_MODEL", "sora-2"))
OPENAI_VIDEO_SIZE = os.getenv("CAP_SHORT_OPENAI_VIDEO_SIZE", os.getenv("CAP_OPENAI_VIDEO_SIZE", "720x1280"))
OPENAI_VIDEO_POLL_SECONDS = int(os.getenv("CAP_SHORT_OPENAI_VIDEO_POLL_SECONDS", os.getenv("CAP_OPENAI_VIDEO_POLL_SECONDS", "10")))
OPENAI_VIDEO_HTTP_RETRIES = int(os.getenv("CAP_SHORT_OPENAI_VIDEO_HTTP_RETRIES", os.getenv("CAP_OPENAI_VIDEO_HTTP_RETRIES", "5")))
OPENAI_VIDEO_HTTP_RETRY_SECONDS = int(os.getenv("CAP_SHORT_OPENAI_VIDEO_HTTP_RETRY_SECONDS", os.getenv("CAP_OPENAI_VIDEO_HTTP_RETRY_SECONDS", "10")))
NVIDIA_VIDEO_BASE_URL = os.getenv(
    "CAP_SHORT_NVIDIA_VIDEO_BASE_URL",
    os.getenv("NVIDIA_VIDEO_BASE_URL", "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion"),
)
NVIDIA_VIDEO_CFG_SCALE = float(os.getenv("CAP_SHORT_NVIDIA_VIDEO_CFG_SCALE", os.getenv("NVIDIA_VIDEO_CFG_SCALE", "1.8")))
NVIDIA_VIDEO_MOTION_BUCKET_ID = int(os.getenv("CAP_SHORT_NVIDIA_VIDEO_MOTION_BUCKET_ID", os.getenv("NVIDIA_VIDEO_MOTION_BUCKET_ID", "127")))
NVIDIA_VIDEO_SEED = int(os.getenv("CAP_SHORT_NVIDIA_VIDEO_SEED", os.getenv("NVIDIA_VIDEO_SEED", "0")))
POLL_SECONDS = int(os.getenv("CAP_SHORT_VIDEO_POLL_SECONDS", "10"))
DEFAULT_DURATION_SECONDS = int(os.getenv("CAP_SHORT_VIDEO_SECONDS", "5"))


def normalize_provider(value: str | None) -> str:
    provider = str(value or "gemini").strip().lower()
    return provider if provider in VALID_ANIMATION_PROVIDERS else "gemini"


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


def parse_video_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = str(size_text or "").lower().split("x", 1)
        width = int(width_text.strip())
        height = int(height_text.strip())
    except Exception as exc:
        raise RuntimeError(f"invalid OpenAI video size: {size_text}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid OpenAI video size: {size_text}")
    return width, height


def prepare_openai_reference_image(image_path: Path, size_text: str) -> Path:
    target_size = parse_video_size(size_text)
    tmp = tempfile.NamedTemporaryFile(prefix="short_sora_ref_", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size != target_size:
                print(f"[INFO] resize_openai_reference_image={img.size}->{target_size}", flush=True)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(tmp_path, "PNG")
        return tmp_path
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def request_with_retries(method: str, url: str, **kwargs):
    import requests

    attempts = max(1, OPENAI_VIDEO_HTTP_RETRIES)
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"[WARN] openai_video_http_retry method={method} attempt={attempt}/{attempts} "
                f"wait_seconds={OPENAI_VIDEO_HTTP_RETRY_SECONDS} error={exc}",
                flush=True,
            )
            time.sleep(OPENAI_VIDEO_HTTP_RETRY_SECONDS)
    raise RuntimeError(f"OpenAI Sora HTTP request failed after {attempts} attempts: {last_exc}") from last_exc


def openai_sora_seconds(duration_seconds: int) -> str:
    return "4" if duration_seconds <= 4 else "8"


def download_openai_video_bytes(video_id: str, headers: dict) -> bytes:
    video_id = str(video_id or "").strip()
    if not video_id:
        raise RuntimeError("OpenAI Sora video id is empty")
    video_job = {"status": "in_progress", "id": video_id}
    while str(video_job.get("status", "")).lower() in {"queued", "in_progress"}:
        print(f"[INFO] openai_video_status={video_job.get('status')}", flush=True)
        print(f"[INFO] openai_video_progress={video_job.get('progress')}", flush=True)
        time.sleep(OPENAI_VIDEO_POLL_SECONDS)
        status_response = request_with_retries(
            "GET",
            f"https://api.openai.com/v1/videos/{video_id}",
            headers=headers,
            timeout=120,
        )
        if status_response.status_code >= 400:
            raise RuntimeError(f"OpenAI Sora status failed: {status_response.status_code} {status_response.text[:1000]}")
        video_job = status_response.json()
    if str(video_job.get("status", "")).lower() != "completed":
        raise RuntimeError(f"OpenAI Sora generation failed or stopped: {video_job}")
    content_response = request_with_retries(
        "GET",
        f"https://api.openai.com/v1/videos/{video_id}/content",
        headers=headers,
        timeout=600,
    )
    if content_response.status_code >= 400:
        raise RuntimeError(f"OpenAI Sora download failed: {content_response.status_code} {content_response.text[:1000]}")
    if not content_response.content:
        raise RuntimeError("OpenAI Sora returned empty video content")
    return content_response.content


def generate_openai_video_bytes(prompt: str, image_path: Path, duration_seconds: int) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    seconds = os.getenv("CAP_SHORT_OPENAI_VIDEO_SECONDS") or os.getenv("CAP_OPENAI_VIDEO_SECONDS") or openai_sora_seconds(duration_seconds)
    headers = {"Authorization": f"Bearer {api_key}"}
    reference_path = prepare_openai_reference_image(image_path, OPENAI_VIDEO_SIZE)
    try:
        with reference_path.open("rb") as image_file:
            response = request_with_retries(
                "POST",
                "https://api.openai.com/v1/videos",
                headers=headers,
                data={
                    "model": OPENAI_VIDEO_MODEL,
                    "prompt": prompt,
                    "size": OPENAI_VIDEO_SIZE,
                    "seconds": seconds,
                },
                files={"input_reference": (reference_path.name, image_file, "image/png")},
                timeout=120,
            )
    finally:
        try:
            reference_path.unlink()
        except FileNotFoundError:
            pass
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI Sora submit failed: {response.status_code} {response.text[:1000]}")
    video_job = response.json()
    video_id = str(video_job.get("id") or "").strip()
    if not video_id:
        raise RuntimeError(f"OpenAI Sora response missing id: {video_job}")
    print(f"[INFO] openai_video_id={video_id}", flush=True)
    print(f"[INFO] openai_video_model={OPENAI_VIDEO_MODEL}", flush=True)
    print(f"[INFO] openai_video_size={OPENAI_VIDEO_SIZE}", flush=True)
    print(f"[INFO] openai_video_seconds={seconds}", flush=True)
    return download_openai_video_bytes(video_id, headers)


def prepare_nvidia_svd_image_data_uri(image_path: Path) -> str:
    target_size = (1024, 576)
    max_bytes = 200_000
    tmp = tempfile.NamedTemporaryFile(prefix="short_nvidia_svd_ref_", suffix=".jpg", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size != target_size:
                print(f"[INFO] resize_nvidia_svd_reference_image={img.size}->{target_size}", flush=True)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            for quality in [85, 75, 65, 55, 45, 35]:
                img.save(tmp_path, "JPEG", quality=quality, optimize=True)
                if tmp_path.stat().st_size <= max_bytes:
                    break
            if tmp_path.stat().st_size > max_bytes:
                raise RuntimeError(f"NVIDIA SVD reference image is still too large: {tmp_path.stat().st_size} bytes")
        return "data:image/jpeg;base64," + base64.b64encode(tmp_path.read_bytes()).decode("ascii")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def extract_nvidia_video_bytes(payload: dict) -> bytes:
    candidates = []
    if isinstance(payload, dict):
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if isinstance(artifact, dict):
                    candidates.extend([
                        artifact.get("base64"),
                        artifact.get("b64"),
                        artifact.get("video"),
                    ])
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    candidates.extend([
                        item.get("b64_json"),
                        item.get("base64"),
                        item.get("video"),
                    ])
        candidates.extend([
            payload.get("base64"),
            payload.get("b64"),
            payload.get("video"),
        ])

    for value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        if "," in text and text.lower().startswith("data:"):
            text = text.split(",", 1)[1]
        try:
            video_bytes = base64.b64decode(text)
        except Exception:
            continue
        if video_bytes:
            return video_bytes
    raise RuntimeError(f"NVIDIA SVD response has no base64 video payload: {str(payload)[:1000]}")


def generate_nvidia_video_bytes(image_path: Path) -> bytes:
    import requests

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")
    image_data_uri = prepare_nvidia_svd_image_data_uri(image_path)
    print(f"[INFO] nvidia_video_base_url={NVIDIA_VIDEO_BASE_URL}", flush=True)
    print(f"[INFO] nvidia_video_cfg_scale={NVIDIA_VIDEO_CFG_SCALE}", flush=True)
    print(f"[INFO] nvidia_video_motion_bucket_id={NVIDIA_VIDEO_MOTION_BUCKET_ID}", flush=True)
    response = requests.post(
        NVIDIA_VIDEO_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "image": image_data_uri,
            "seed": NVIDIA_VIDEO_SEED,
            "cfg_scale": NVIDIA_VIDEO_CFG_SCALE,
            "motion_bucket_id": NVIDIA_VIDEO_MOTION_BUCKET_ID,
        },
        timeout=600,
    )
    if response.status_code >= 400:
        if response.status_code == 404 and "Not found for account" in response.text:
            raise RuntimeError(
                "NVIDIA Stable Video Diffusion is not available for the configured NVIDIA_API_KEY account. "
                "Request/enable access to stabilityai/stable-video-diffusion in NVIDIA API Catalog, "
                "or set CAP_SHORT_NVIDIA_VIDEO_BASE_URL to a deployed/entitled SVD endpoint."
            )
        raise RuntimeError(f"NVIDIA SVD submit failed: {response.status_code} {response.text[:1000]}")
    return extract_nvidia_video_bytes(response.json())


def download_video(client, target) -> bytes:
    try:
        return client.files.download(file=target)
    except TypeError:
        return client.files.download(target)


def generate_animation(ep_dir: Path, storyboard_path: Path, scene_id: str, image_path_override: str = "", provider: str = "gemini") -> Path:
    load_dotenv()
    provider = normalize_provider(provider)
    if provider == "nvidia":
        if not NVIDIA_VIDEO_BASE_URL:
            raise RuntimeError("CAP_SHORT_NVIDIA_VIDEO_BASE_URL or NVIDIA_VIDEO_BASE_URL is not configured")
    api_key = os.getenv("GEMINI_API_KEY")
    if provider == "gemini" and not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    rows = read_storyboard(storyboard_path)
    target = next((row for row in rows if scene_matches(row, scene_id)), None)
    if not target:
        raise ValueError(f"scene not found: {scene_id}")

    prompt = str(target.get("animation_prompt", "")).strip()
    if not prompt and provider != "nvidia":
        raise ValueError("animation_prompt is empty")
    if not prompt:
        prompt = "NVIDIA Stable Video Diffusion image-to-video generation"
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

    client = genai.Client(api_key=api_key) if provider == "gemini" else None
    output_path = next_video_path(ep_dir, scene_id)
    print(f"[INFO] provider={provider}")
    print(f"[INFO] video_model={VIDEO_MODEL}")
    print(f"[INFO] openai_video_model={OPENAI_VIDEO_MODEL}")
    print(f"[INFO] openai_video_size={OPENAI_VIDEO_SIZE}")
    print(f"[INFO] scene={scene_id}")
    print(f"[INFO] image_path={image_path}")
    print(f"[INFO] output_path={output_path}")
    print(f"[INFO] duration_seconds={duration}")
    print(f"[INFO] prompt={prompt[:500]}")

    if provider == "openai":
        video_bytes = generate_openai_video_bytes(prompt, image_path, duration)
    elif provider == "nvidia":
        video_bytes = generate_nvidia_video_bytes(image_path)
    else:
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
    parser.add_argument("--provider", default=os.getenv("CAP_SHORT_ANIMATION_PROVIDER", "gemini"), choices=sorted(VALID_ANIMATION_PROVIDERS))
    args = parser.parse_args()
    generate_animation(
        ep_dir=Path(args.ep_dir).resolve(),
        storyboard_path=Path(args.storyboard).resolve(),
        scene_id=args.scene_id,
        image_path_override=args.image,
        provider=args.provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
