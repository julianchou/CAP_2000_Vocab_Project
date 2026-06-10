import os
import csv
import json
import argparse
import traceback
import tempfile
import sys
import re
import base64
import time
from io import BytesIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from PIL import Image
from app_utils.asset_tags import tag_asset_from_row

load_dotenv(BASE_DIR / ".env")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}
VALID_IMAGE_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
GEMINI_IMAGE_MODEL = os.getenv("CAP_STORYBOARD_GEMINI_IMAGE_MODEL", os.getenv("CAP_IMAGE_MODEL", "imagen-4.0-generate-001"))
OPENAI_IMAGE_MODEL = os.getenv("CAP_STORYBOARD_OPENAI_IMAGE_MODEL", os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5"))
OPENAI_IMAGE_SIZE = os.getenv("CAP_STORYBOARD_OPENAI_IMAGE_SIZE", "1536x1024")
NVIDIA_IMAGE_BASE_URL = os.getenv("CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL", os.getenv("NVIDIA_IMAGE_BASE_URL", ""))
NVIDIA_IMAGE_MODEL = os.getenv("CAP_STORYBOARD_NVIDIA_IMAGE_MODEL", os.getenv("NVIDIA_IMAGE_MODEL", "black-forest-labs/flux.2-klein-4b"))
NVIDIA_IMAGE_SIZE = os.getenv("CAP_STORYBOARD_NVIDIA_IMAGE_SIZE", os.getenv("NVIDIA_IMAGE_SIZE", "1344x768"))
NVIDIA_PROMPT_MAX_CHARS = int(os.getenv("CAP_STORYBOARD_NVIDIA_PROMPT_MAX_CHARS", "800"))

# --- 目錄設定 ---
base_dir = BASE_DIR
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_STORYBOARD_IMAGE_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_IMAGE_PROVIDERS else "auto"

def validate_image_provider_config(provider: str) -> None:
    provider = normalize_provider(provider)
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if provider == "nvidia":
        if not os.getenv("NVIDIA_API_KEY"):
            raise RuntimeError("NVIDIA_API_KEY is not configured")
        if not NVIDIA_IMAGE_BASE_URL:
            raise RuntimeError("CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL or NVIDIA_IMAGE_BASE_URL is not configured")

def is_quota_or_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ["resource_exhausted", "quota", "spending cap", "rate limit", "429"])

def save_response_image_data(image_data, target_path: Path) -> None:
    if not image_data:
        raise RuntimeError("image provider returned empty image data")
    raw = None
    b64_value = getattr(image_data, "b64_json", None)
    if b64_value:
        raw = base64.b64decode(b64_value)
    elif isinstance(image_data, dict) and image_data.get("b64_json"):
        raw = base64.b64decode(image_data["b64_json"])
    elif getattr(image_data, "url", None) or (isinstance(image_data, dict) and image_data.get("url")):
        raise RuntimeError("image provider returned a URL instead of base64 image data")
    if not raw:
        raise RuntimeError("image provider response has no b64_json image data")
    with Image.open(BytesIO(raw)) as img:
        save_png_atomic(img.convert("RGB"), target_path)

def parse_image_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = str(size_text or "").lower().split("x", 1)
        width = int(width_text.strip())
        height = int(height_text.strip())
    except Exception as exc:
        raise RuntimeError(f"invalid image size: {size_text}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid image size: {size_text}")
    return width, height

def save_nvidia_infer_response(payload: dict, output_path: Path) -> None:
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if not artifacts:
        raise RuntimeError(f"NVIDIA response has no artifacts: {str(payload)[:500]}")
    image_b64 = artifacts[0].get("base64") if isinstance(artifacts[0], dict) else ""
    if not image_b64:
        raise RuntimeError(f"NVIDIA response artifact has no base64 image: {str(payload)[:500]}")
    raw = base64.b64decode(image_b64)
    with Image.open(BytesIO(raw)) as img:
        save_png_atomic(img.convert("RGB"), output_path)

def generate_gemini_image(prompt_text: str, output_path: Path) -> None:
    from google import genai
    from google.genai import types

    print(f"🎨 provider=gemini model={GEMINI_IMAGE_MODEL}", flush=True)
    last_error = None
    for attempt in range(1, 3):
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or None)
            result = client.models.generate_images(
                model=GEMINI_IMAGE_MODEL,
                prompt=prompt_text,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="16:9"
                )
            )
            generated_image = result.generated_images[0].image
            generated_image.save(output_path)
            return
        except RuntimeError as exc:
            last_error = exc
            if "client has been closed" not in str(exc).lower() or attempt >= 2:
                raise
            print(
                f"WARN gemini_client_closed_retry attempt={attempt}/2 wait_seconds=2 error={exc}",
                flush=True,
            )
            time.sleep(2)
    if last_error:
        raise last_error

def generate_openai_image(prompt_text: str, output_path: Path) -> None:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"🎨 provider=openai model={OPENAI_IMAGE_MODEL} size={OPENAI_IMAGE_SIZE}", flush=True)
    response = OpenAI().images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt_text,
        size=OPENAI_IMAGE_SIZE,
        n=1,
    )
    save_response_image_data(response.data[0], output_path)

def generate_nvidia_image(prompt_text: str, output_path: Path) -> None:
    from openai import OpenAI
    import requests

    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is not configured")
    if not NVIDIA_IMAGE_BASE_URL:
        raise RuntimeError("CAP_STORYBOARD_NVIDIA_IMAGE_BASE_URL or NVIDIA_IMAGE_BASE_URL is not configured")
    prompt_text = compact_prompt_for_nvidia(prompt_text)
    print(f"🎨 provider=nvidia model={NVIDIA_IMAGE_MODEL} size={NVIDIA_IMAGE_SIZE}", flush=True)
    if "/genai/" in NVIDIA_IMAGE_BASE_URL:
        width, height = parse_image_size(NVIDIA_IMAGE_SIZE)
        response = requests.post(
            NVIDIA_IMAGE_BASE_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "samples": 1,
                "seed": 0,
                "steps": 4,
            },
            timeout=180,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"NVIDIA hosted image endpoint failed: {response.status_code} {response.text[:1000]}")
        save_nvidia_infer_response(response.json(), output_path)
        return

    response = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"), base_url=NVIDIA_IMAGE_BASE_URL).images.generate(
        model=NVIDIA_IMAGE_MODEL,
        prompt=prompt_text,
        size=NVIDIA_IMAGE_SIZE,
        n=1,
        response_format="b64_json",
    )
    save_response_image_data(response.data[0], output_path)

def compact_prompt_for_nvidia(prompt_text: str) -> str:
    text = re.sub(r"\s+", " ", str(prompt_text or "")).strip()
    if len(text) <= NVIDIA_PROMPT_MAX_CHARS:
        return text

    scene_marker = "Scene content:"
    scene_text = text
    if scene_marker in text:
        scene_text = text.split(scene_marker, 1)[1].strip()

    prefix = (
        "Cute 3D cartoon educational vocabulary video, 16:9 cinematic wide shot. "
        "Keep recurring hosts consistent: John is a cheerful boy with messy brown hair, green hoodie, gaming headset; "
        "Mary is a smart girl with long blonde pigtails, round glasses, yellow teacher outfit. "
        "Avoid close-up portraits; keep teaching object/action as focus. Scene: "
    )
    compacted = prefix + scene_text
    if len(compacted) > NVIDIA_PROMPT_MAX_CHARS:
        compacted = compacted[: NVIDIA_PROMPT_MAX_CHARS - 1].rstrip(" ,.;") + "."
    print(
        f"⚠️ NVIDIA prompt compacted chars={len(text)}->{len(compacted)} max={NVIDIA_PROMPT_MAX_CHARS}",
        flush=True,
    )
    return compacted

def generate_image_with_provider(prompt_text: str, output_path: Path, provider: str) -> str:
    provider = normalize_provider(provider)
    errors: list[str] = []
    if provider in {"auto", "gemini"}:
        try:
            generate_gemini_image(prompt_text, output_path)
            return "gemini"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            if is_quota_or_limit_error(exc):
                print(f"⚠️ Gemini/Imagen quota or limit hit; falling back to OpenAI, then NVIDIA if needed. raw_error={exc}", flush=True)
            else:
                print(f"⚠️ Gemini/Imagen failed; falling back to OpenAI, then NVIDIA if needed. raw_error={exc}", flush=True)

    if provider in {"auto", "openai"}:
        try:
            generate_openai_image(prompt_text, output_path)
            return "openai"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            if provider == "openai":
                raise RuntimeError("image generation failed: " + " | ".join(errors)) from exc
            print(f"⚠️ OpenAI image generation failed; falling back to NVIDIA. raw_error={exc}", flush=True)

    if provider in {"auto", "nvidia"}:
        try:
            generate_nvidia_image(prompt_text, output_path)
            return "nvidia"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            raise RuntimeError("image generation failed: " + " | ".join(errors)) from exc

    raise RuntimeError("image generation failed: " + " | ".join(errors))

def episode_search_roots() -> list[Path]:
    roots = [workspace_dir, base_dir / "workspaces" / "story", base_dir / "workspaces" / "vocab", base_dir / "workspace"]
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

def storyboard_csv_for_episode(target_folder: Path) -> Path:
    story_path = target_folder / "05_storyboards" / "storyboard.csv"
    if story_path.exists():
        return story_path
    return target_folder / "03_storyboards" / "storyboard.csv"

def image_dirs_for_episode(target_folder: Path, storyboard_csv: Path) -> tuple[Path, Path, Path]:
    if "05_storyboards" in storyboard_csv.parts:
        root = target_folder / "05_storyboards" / "images"
        return root / "ai_generated", root / "flashcards", target_folder / "05_storyboards" / "ffmpeg"
    return (
        target_folder / "04_images" / "ai_generated",
        target_folder / "04_images" / "flashcards",
        target_folder / "05_output",
    )


def load_outro_settings(target_folder: Path) -> dict | None:
    config_path = target_folder / "05_output" / "outro" / "outro_config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠️ 片尾設定讀取失敗，略過片尾：{exc}", flush=True)
        return None
    media_rel = str(config.get("media_path") or "").strip()
    if not media_rel:
        return None
    media_path = (target_folder / media_rel).resolve()
    if not media_path.exists():
        print(f"⚠️ 片尾媒體不存在，略過片尾：{media_path}", flush=True)
        return None
    try:
        duration = float(config.get("duration_seconds") or 0)
    except Exception:
        duration = 0
    if duration <= 0:
        print("⚠️ 片尾長度未設定或小於等於 0，略過片尾。", flush=True)
        return None
    return {"media_path": media_path, "duration": duration}


def split_names(value: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r"[,，、/|]+", str(value or "")):
        name = part.strip()
        if name and name not in names:
            names.append(name)
    return names

def load_character_profiles(target_folder: Path) -> dict[str, dict]:
    path = target_folder / "03_characters_scenes" / "characters.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    profiles: dict[str, dict] = {}
    for item in data.get("characters") or []:
        name = str(item.get("name", "")).strip()
        if name:
            profiles[name] = item
    return profiles

def character_reference_prompt(target_folder: Path, row: dict) -> str:
    profiles = load_character_profiles(target_folder)
    if not profiles:
        return ""
    row_text = " ".join(
        str(row.get(key, "") or "")
        for key in ["characters", "image_prompt", "summary", "reason", "subtitle_reference"]
    )
    names = split_names(str(row.get("characters", "")))
    for name in profiles:
        if name in row_text and name not in names:
            names.append(name)
    refs: list[str] = []
    for name in names:
        profile = profiles.get(name)
        if not profile:
            continue
        visual = profile.get("visual_design") or {}
        image_generation = profile.get("image_generation") or {}
        palette = ", ".join(visual.get("color_palette") or [])
        tags = ", ".join(visual.get("consistency_tags") or [])
        parts = [
            f"{name} must keep the same design in every scene",
            str(image_generation.get("prompt_en", "")).strip(),
            f"fixed color palette: {palette}" if palette else "",
            f"fixed consistency tags: {tags}" if tags else "",
            f"fixed default expression: {visual.get('default_expression', '')}" if visual.get("default_expression") else "",
            f"fixed costume/features: {visual.get('costume_or_features', '')}" if visual.get("costume_or_features") else "",
        ]
        ref = ". ".join(part for part in parts if part)
        if ref:
            refs.append(ref)
    if not refs:
        return ""
    return (
        "Character consistency reference from No.3.1 characters.json. This reference has higher priority than the scene action: "
        + " ".join(refs)
        + " Do not redesign these characters between scenes; keep face shape, body shape, colors, costume/features, and overall style identical. Scene action prompt: "
    )

def next_variant_path(base_path: Path) -> Path:
    stem = base_path.stem
    suffix = base_path.suffix
    variants = sorted(base_path.parent.glob(f"{stem}__v*{suffix}"))
    if not variants:
        return base_path.parent / f"{stem}__v02{suffix}"
    last = variants[-1]
    marker = last.stem.split("__v")[-1]
    try:
        next_no = int(marker) + 1
    except Exception:
        next_no = len(variants) + 2
    return base_path.parent / f"{stem}__v{next_no:02d}{suffix}"

def save_png_atomic(image: Image.Image, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_str = tempfile.mkstemp(
        prefix=f"{target_path.stem}__",
        suffix=target_path.suffix,
        dir=str(target_path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_path_str)
    try:
        image.save(temp_path, "PNG")
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

def process_image_standard(img_path):
    """
    ⚡ 關鍵新增：強制圖片標準化
    確保所有圖片都是 1920x1080, RGB 模式，避免 FFmpeg 渲染中斷。
    """
    try:
        with Image.open(img_path) as img:
            # 強制轉換為 RGB (防止 RGBA 或 P 模式導致 FFmpeg 報錯)
            img = img.convert('RGB')
            
            # 如果尺寸不對，強制縮放
            if img.size != (1920, 1080):
                print(f"🔧 修正尺寸：{img_path.name} ({img.size} -> 1920x1080)")
                img = img.resize((1920, 1080), Image.Resampling.LANCZOS)
            
            save_png_atomic(img, img_path)
    except Exception as e:
        print(f"🚨 無法標準化圖片 {img_path.name}: {e}")

def resolve_scene_asset(target_folder: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = target_folder / path
    return path if path.exists() and path.is_file() and path.stat().st_size > 0 else None

def existing_assigned_visual(target_folder: Path, row: dict) -> tuple[Path | None, str]:
    for key in ["animation_video_path", "animation_asset"]:
        path = resolve_scene_asset(target_folder, row.get(key, ""))
        if path and path.suffix.lower() in VIDEO_SUFFIXES:
            return path, key
    for key in ["custom_image_path", "asset", "image_asset"]:
        path = resolve_scene_asset(target_folder, row.get(key, ""))
        if path and path.suffix.lower() in IMAGE_SUFFIXES:
            return path, key
    return None, ""

def generate_scene_image(
    target_folder: Path,
    row: dict,
    force_ai_regenerate: bool = False,
    storyboard_csv: Path | None = None,
    provider: str = "auto",
):
    storyboard_csv = storyboard_csv or storyboard_csv_for_episode(target_folder)
    ai_images_folder, flashcards_folder, _output_folder = image_dirs_for_episode(target_folder, storyboard_csv)
    ai_images_folder.mkdir(parents=True, exist_ok=True)

    source_type = row.get("source_type", "AI").strip().lower()

    if not force_ai_regenerate:
        assigned_visual, assigned_key = existing_assigned_visual(target_folder, row)
        if assigned_visual:
            print(f"⏭️ Scene {row.get('scene_id', '')} 已指定素材 ({assigned_key})，略過 AI 產圖：{assigned_visual}")
            if assigned_visual.suffix.lower() in IMAGE_SUFFIXES:
                process_image_standard(assigned_visual)
                tag_asset_from_row(assigned_visual, row, asset_kind="image", overwrite=False)
            elif assigned_visual.suffix.lower() in VIDEO_SUFFIXES:
                tag_asset_from_row(assigned_visual, row, asset_kind="animation", overwrite=False)
            return assigned_visual

    if source_type == "flashcard":
        custom_image_path = row.get("custom_image_path", "").strip()
        if custom_image_path:
            custom_img = Path(custom_image_path)
            if custom_img.exists():
                print(f"📁 [Flashcard] 使用自訂圖卡: {custom_img.name}")
                process_image_standard(custom_img)
                tag_asset_from_row(custom_img, row, asset_kind="image", overwrite=False)
                return custom_img
        word = row.get("flashcard_word", "").strip()
        img_path = flashcards_folder / f"{word}.png"
        if not img_path.exists():
            print(f"⚠️ 警告：找不到單字圖卡 {word}.png")
            fallback_img = Image.new('RGB', (1920, 1080), color=(44, 62, 80))
            img_path.parent.mkdir(parents=True, exist_ok=True)
            save_png_atomic(fallback_img, img_path)
        else:
            print(f"📁 [Flashcard] 使用並標準化圖卡: {word}.png")
            process_image_standard(img_path)
        tag_asset_from_row(img_path, row, asset_kind="image", overwrite=False)
        return img_path

    start_t = str(row["start_time"]).strip()
    img_name = f"img_{start_t}.png"
    base_img_path = ai_images_folder / img_name
    img_path = next_variant_path(base_img_path) if force_ai_regenerate else base_img_path
    raw_prompt = row.get("image_prompt", "").strip()
    prompt_text = raw_prompt if raw_prompt else "A clean educational background, soft blue and white gradient, 16:9"
    character_ref = character_reference_prompt(target_folder, row)
    if character_ref:
        prompt_text = character_ref + prompt_text.strip()

    if base_img_path.exists() and not force_ai_regenerate:
        print(f"⏭️ {img_name} 已存在，重新標準化以確保安全。")
        process_image_standard(base_img_path)
        tag_asset_from_row(base_img_path, row, asset_kind="image", overwrite=False)
        return base_img_path

    try:
        print(f"🎨 正在為 AI 場景生成圖片 ({img_path.name})...")
        fd, temp_path_str = tempfile.mkstemp(
            prefix=f"{img_path.stem}__",
            suffix=img_path.suffix,
            dir=str(img_path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_path_str)
        try:
            provider_used = generate_image_with_provider(prompt_text, temp_path, provider)
            process_image_standard(temp_path)
            temp_path.replace(img_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
        print(f"✅ {img_path.name} 產圖成功且標準化！provider={provider_used}")
    except Exception as e:
        print(f"🚨 {img_path.name} 生成失敗: {e}")
        if force_ai_regenerate:
            raise
        fallback_img = Image.new('RGB', (1920, 1080), color=(44, 62, 80))
        save_png_atomic(fallback_img, img_path)

    tag_asset_from_row(img_path, row, asset_kind="image", overwrite=True)
    return img_path

def set_scene_image_default(storyboard_csv: Path, scene_id: int | str, image_path: Path) -> None:
    with storyboard_csv.open(mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for col in ["custom_image_path"]:
        if col not in fieldnames:
            fieldnames.append(col)
    target_path = str(image_path).replace("\\", "/")
    changed = False
    for row in rows:
        if str(row.get("scene_id", "")).strip() == str(scene_id):
            row["custom_image_path"] = target_path
            changed = True
    if changed:
        with storyboard_csv.open(mode="w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def generate_preview(ep_num, scene_id: int | None = None, provider: str = "auto"):
    provider = normalize_provider(provider)
    validate_image_provider_config(provider)
    print(f"\n🎞️ [Stage 5] 正在處理第 {ep_num:02d} 集的分鏡配圖與 FFmpeg 列表...")
    print(
        f"image_provider={provider} gemini_model={GEMINI_IMAGE_MODEL} "
        f"openai_model={OPENAI_IMAGE_MODEL} nvidia_model={NVIDIA_IMAGE_MODEL}"
    )
    
    target_folder = find_episode_folder(ep_num)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num:02d} 集的資料夾。")
        return

    storyboard_csv = storyboard_csv_for_episode(target_folder)
    ai_images_folder, _flashcards_folder, output_folder = image_dirs_for_episode(target_folder, storyboard_csv)

    if not storyboard_csv.exists():
        print(f"❌ 找不到分鏡表：{storyboard_csv}")
        return

    ai_images_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    inputs_list = []

    with open(storyboard_csv, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if scene_id is not None:
        target_row = next((row for row in rows if str(row.get("scene_id", "")).strip() == str(scene_id)), None)
        if not target_row:
            print(f"❌ 找不到 scene_id={scene_id}")
            return None
        img_path = generate_scene_image(
            target_folder,
            target_row,
            force_ai_regenerate=True,
            storyboard_csv=storyboard_csv,
            provider=provider,
        )
        set_scene_image_default(storyboard_csv, scene_id, img_path)
        print(f"✅ Scene {scene_id} 已完成單獨產圖：{img_path}")
        return img_path

    cumulative_time = 0.0

    for i, row in enumerate(rows):
        # 1. 計算停留時間 (Duration)
        if i < len(rows) - 1:
            next_start_time = float(rows[i+1]["start_time"])
            duration = next_start_time - cumulative_time
        else:
            end_time = float(row["end_time"])
            duration = end_time - cumulative_time
            if duration <= 0: duration = 2.0

        img_path = generate_scene_image(
            target_folder,
            row,
            force_ai_regenerate=False,
            storyboard_csv=storyboard_csv,
            provider=provider,
        )

        # 3. 寫入 FFmpeg inputs 列表
        rel_img_path = os.path.relpath(img_path, output_folder)
        safe_img_path = str(rel_img_path).replace('\\', '/')
        inputs_list.append(f"file '{safe_img_path}'\nduration {duration:.3f}")
        cumulative_time += duration

    outro_settings = load_outro_settings(target_folder)
    if outro_settings:
        outro_media_path = Path(outro_settings["media_path"])
        outro_duration = float(outro_settings["duration"])
        rel_outro_path = os.path.relpath(outro_media_path, output_folder)
        safe_outro_path = str(rel_outro_path).replace("\\", "/")
        inputs_list.append(f"file '{safe_outro_path}'\nduration {outro_duration:.3f}")
        print(f"✅ 已將片尾加入 inputs.txt：{safe_outro_path} ({outro_duration:.3f}s)")

    # 產出 inputs.txt
    inputs_txt_path = output_folder / "inputs.txt"
    if inputs_list:
        with open(inputs_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(inputs_list))
            f.write(f"\n{inputs_list[-1].splitlines()[0]}")
        print(f"✅ 第 {ep_num:02d} 集 inputs.txt 更新完成！")
    return inputs_txt_path if inputs_list else None

def main():
    parser = argparse.ArgumentParser(description="產生匹配的圖片與 FFmpeg inputs.txt")
    parser.add_argument("--ep", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--scene_id", type=int, help="只重產指定 scene 的圖片")
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_IMAGE_PROVIDERS),
        default=os.getenv("CAP_STORYBOARD_IMAGE_PROVIDER", "auto"),
        help="Image provider: auto tries Gemini first, then OpenAI, then NVIDIA.",
    )
    args = parser.parse_args()

    if args.ep is not None:
        start_ep, end_ep = args.ep, args.ep
    elif args.start is not None:
        start_ep = args.start
        end_ep = args.end if args.end else args.start
    else:
        parser.error("請提供 --ep 或 --start 參數")

    for ep in range(start_ep, end_ep + 1):
        generate_preview(ep, scene_id=args.scene_id, provider=args.provider)

if __name__ == "__main__":
    main()
