import argparse
import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw


IMAGEN_MODEL = os.getenv("CAP_SHORT_IMAGE_MODEL", "imagen-4.0-generate-001")
OPENAI_IMAGE_MODEL = os.getenv("CAP_SHORT_OPENAI_IMAGE_MODEL", os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"))
OPENAI_IMAGE_SIZE = os.getenv("CAP_SHORT_OPENAI_IMAGE_SIZE", "1024x1536")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}


class ImageQuotaExhausted(RuntimeError):
    pass


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "spending cap" in text or "quota" in text or "429" in text


def is_gemini_service_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in [
            "500 internal",
            "503 unavailable",
            "internal error encountered",
            "failed to retrieve rai response",
            "server disconnected",
        ]
    )


def read_storyboard(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty storyboard: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        raise ValueError("storyboard must be a JSON array or an object with scenes array")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def scene_number(scene: dict, fallback: int) -> int:
    value = scene.get("scene", scene.get("scene_id", fallback))
    try:
        return int(value)
    except Exception:
        return fallback


def scene_prompt(scene: dict) -> str:
    prompt = str(scene.get("image_prompt") or scene.get("prompt") or "").strip()
    if not prompt:
        summary = str(scene.get("summary") or scene.get("subtitle") or "Educational short video scene").strip()
        prompt = f"{summary}. Vertical 9:16 short video frame, cinematic lighting, clear subject."
    if "9:16" not in prompt and "vertical" not in prompt.lower():
        prompt = prompt.rstrip(" .") + ". Vertical 9:16 short video frame."
    return prompt


def save_placeholder(path: Path, scene: dict, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1080, 1920), color=(35, 42, 54))
    draw = ImageDraw.Draw(img)
    scene_no = str(scene.get("scene", scene.get("scene_id", ""))).strip()
    title = f"Scene {scene_no}" if scene_no else "Scene"
    draw.rectangle((80, 120, 1000, 1800), outline=(120, 150, 190), width=6)
    draw.text((120, 160), title, fill=(235, 240, 245))
    draw.text((120, 220), message[:120], fill=(210, 220, 230))
    img.save(path)


def scene_has_failed_image(scene: dict) -> bool:
    error_text = str(scene.get("image_error") or "").strip()
    if error_text:
        return True
    return str(scene.get("asset_status") or "").strip().lower() in {
        "placeholder",
        "image_generation_failed",
        "failed",
        "error",
    }


def resolve_scene_asset(ep_dir: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ep_dir / path
    return path if path.exists() and path.is_file() and path.stat().st_size > 0 else None


def existing_assigned_visual(ep_dir: Path, scene: dict) -> tuple[Path | None, str]:
    for key in ["animation_asset", "animation_video_path"]:
        path = resolve_scene_asset(ep_dir, str(scene.get(key, "")))
        if path and path.suffix.lower() in VIDEO_SUFFIXES:
            return path, key
    for key in ["asset", "image_asset", "custom_image_path"]:
        path = resolve_scene_asset(ep_dir, str(scene.get(key, "")))
        if path and path.suffix.lower() in IMAGE_SUFFIXES:
            return path, key
    return None, ""


def generate_gemini_image(client, prompt: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = client.models.generate_images(
        model=IMAGEN_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="9:16"),
    )
    generated_image = result.generated_images[0].image
    generated_image.save(output_path)


def generate_openai_image(prompt: str, output_path: Path) -> None:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)
    print(f"[IMAGE_FALLBACK] provider=openai model={OPENAI_IMAGE_MODEL} size={OPENAI_IMAGE_SIZE}", flush=True)
    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=OPENAI_IMAGE_SIZE,
        n=1,
    )
    item = response.data[0]
    b64_text = getattr(item, "b64_json", None)
    if b64_text:
        output_path.write_bytes(base64.b64decode(b64_text))
        return

    image_url = getattr(item, "url", None)
    if image_url:
        import requests

        result = requests.get(image_url, timeout=120)
        result.raise_for_status()
        output_path.write_bytes(result.content)
        return

    raise RuntimeError("OpenAI image response has neither b64_json nor url")


def generate_storyboard_images(ep_dir: Path, storyboard_path: Path, manifest_path: Path, force: bool, scene_id: str = "") -> Path:
    load_dotenv()
    storyboard = read_storyboard(storyboard_path)
    if not storyboard:
        raise ValueError("storyboard has no scenes")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not gemini_api_key and not openai_api_key:
        raise RuntimeError("Neither GEMINI_API_KEY nor OPENAI_API_KEY is configured")
    client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
    images_dir = ep_dir / "04_images" / "storyboard"
    rows = []
    gemini_quota_exhausted = False

    for idx, scene in enumerate(storyboard, 1):
        no = scene_number(scene, idx)
        if scene_id and str(no) != str(scene_id).strip():
            continue
        output_path = images_dir / f"scene_{no:03d}.png"
        rel_path = output_path.relative_to(ep_dir).as_posix()
        prompt = scene_prompt(scene)
        should_regenerate_failed_image = scene_has_failed_image(scene)
        assigned_visual, assigned_key = existing_assigned_visual(ep_dir, scene)
        if assigned_visual and not force and not should_regenerate_failed_image:
            assigned_rel = assigned_visual.relative_to(ep_dir).as_posix() if assigned_visual.is_relative_to(ep_dir) else str(assigned_visual)
            print(f"[SKIP] scene={no} assigned_{assigned_key}={assigned_visual}", flush=True)
            rows.append({
                "scene": no,
                "status": "skipped_assigned_visual",
                "path": assigned_rel,
                "source": assigned_key,
            })
            continue
        if output_path.exists() and output_path.stat().st_size > 0 and not force and not should_regenerate_failed_image:
            print(f"[SKIP] scene={no} existing={output_path}", flush=True)
            scene["asset"] = rel_path
            rows.append({"scene": no, "status": "skipped", "path": rel_path})
            continue
        if should_regenerate_failed_image and output_path.exists() and not force:
            print(f"[RETRY] scene={no} existing file is marked as failed/placeholder; regenerating.", flush=True)
        try:
            provider_used = "gemini"
            print(f"[IMAGE] scene={no} provider=gemini model={IMAGEN_MODEL}", flush=True)
            print(f"[PROMPT] {prompt[:500]}", flush=True)
            if gemini_quota_exhausted or client is None:
                raise ImageQuotaExhausted("Gemini/Imagen quota is unavailable; using OpenAI fallback.")
            generate_gemini_image(client, prompt, output_path)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("empty generated image")
            scene["asset"] = rel_path
            scene.pop("image_error", None)
            scene.pop("asset_status", None)
            rows.append({"scene": no, "status": "done", "provider": provider_used, "path": rel_path})
            print(f"[DONE] scene={no} path={output_path}", flush=True)
        except Exception as exc:
            gemini_error_kind = "quota_exhausted" if is_quota_error(exc) else (
                "service_error" if is_gemini_service_error(exc) else ""
            )
            if gemini_error_kind:
                if gemini_error_kind == "quota_exhausted":
                    gemini_quota_exhausted = True
                print(f"[WARN] Gemini/Imagen {gemini_error_kind} for scene={no}; switching to OpenAI. raw_error={exc}", flush=True)
                try:
                    provider_used = "openai"
                    generate_openai_image(prompt, output_path)
                    if not output_path.exists() or output_path.stat().st_size == 0:
                        raise RuntimeError("empty OpenAI generated image")
                    scene["asset"] = rel_path
                    scene.pop("image_error", None)
                    scene.pop("asset_status", None)
                    rows.append({
                        "scene": no,
                        "status": "done",
                        "provider": provider_used,
                        "path": rel_path,
                        "fallback_reason": f"gemini_{gemini_error_kind}",
                    })
                    print(f"[DONE] scene={no} provider=openai path={output_path}", flush=True)
                    continue
                except Exception as openai_exc:
                    message = f"Gemini {gemini_error_kind} and OpenAI image fallback failed."
                    print(f"[ERROR] scene={no} {message} openai_error={openai_exc}", flush=True)
                    rows.append({
                        "scene": no,
                        "status": "image_generation_failed",
                        "path": "",
                        "error": str(openai_exc),
                        "gemini_error": str(exc),
                    })
                    manifest = {
                        "model": IMAGEN_MODEL,
                        "openai_model": OPENAI_IMAGE_MODEL,
                        "aspect_ratio": "9:16",
                        "status": "image_generation_failed",
                        "error": str(openai_exc),
                        "gemini_error": str(exc),
                        "storyboard": str(storyboard_path),
                        "images_dir": str(images_dir),
                        "items": rows,
                    }
                    write_json(manifest_path, manifest)
                    raise ImageQuotaExhausted(message) from openai_exc
            print(f"[ERROR] scene={no} image generation failed: {exc}", flush=True)
            save_placeholder(output_path, scene, f"Image generation failed: {exc}")
            scene["asset"] = rel_path
            scene["image_error"] = str(exc)
            scene["asset_status"] = "placeholder"
            rows.append({"scene": no, "status": "placeholder", "provider": "placeholder", "path": rel_path, "error": str(exc)})

    write_json(storyboard_path, storyboard)
    manifest = {
        "model": IMAGEN_MODEL,
        "openai_model": OPENAI_IMAGE_MODEL,
        "aspect_ratio": "9:16",
        "status": "done",
        "fallback": "openai" if any(item.get("provider") == "openai" for item in rows) else "",
        "storyboard": str(storyboard_path),
        "images_dir": str(images_dir),
        "items": rows,
    }
    write_json(manifest_path, manifest)
    print(f"[INFO] manifest={manifest_path}", flush=True)
    print(f"[INFO] image_count={len(rows)}", flush=True)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 7: generate 9:16 storyboard images.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene-id", default="")
    args = parser.parse_args()
    try:
        generate_storyboard_images(
            ep_dir=Path(args.ep_dir).resolve(),
            storyboard_path=Path(args.storyboard).resolve(),
            manifest_path=Path(args.manifest).resolve(),
            force=bool(args.force),
            scene_id=str(args.scene_id or "").strip(),
        )
        return 0
    except ImageQuotaExhausted as exc:
        print(f"[FATAL] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
