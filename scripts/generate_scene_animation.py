import argparse
import csv
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from generate_animation_prompt import fallback_prompt
from app_utils.asset_tags import tag_asset_from_row


load_dotenv()

client = genai.Client()
WORKSPACE_DIR = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(BASE_DIR / "workspace")))
VALID_ANIMATION_PROVIDERS = {"gemini", "openai"}
ANIMATION_PROVIDER = os.environ.get("CAP_ANIMATION_PROVIDER", "gemini")
VIDEO_MODEL = os.environ.get("CAP_ANIMATION_MODEL", "veo-2.0-generate-001")
OPENAI_VIDEO_MODEL = os.environ.get("CAP_OPENAI_VIDEO_MODEL", "sora-2")
OPENAI_VIDEO_SIZE = os.environ.get("CAP_OPENAI_VIDEO_SIZE", "1280x720")
OPENAI_VIDEO_POLL_SECONDS = int(os.environ.get("CAP_OPENAI_VIDEO_POLL_SECONDS", "10"))
OPENAI_VIDEO_HTTP_RETRIES = int(os.environ.get("CAP_OPENAI_VIDEO_HTTP_RETRIES", "5"))
OPENAI_VIDEO_HTTP_RETRY_SECONDS = int(os.environ.get("CAP_OPENAI_VIDEO_HTTP_RETRY_SECONDS", "10"))
POLL_SECONDS = int(os.environ.get("CAP_ANIMATION_POLL_SECONDS", "10"))
DEFAULT_ANIMATION_SECONDS = int(os.environ.get("CAP_ANIMATION_SECONDS", "5"))
PERSON_GENERATION = os.environ.get("CAP_ANIMATION_PERSON_GENERATION", "allow_adult")
NETWORK_RETRY_SECONDS = int(os.environ.get("CAP_ANIMATION_NETWORK_RETRY_SECONDS", "5"))


class SceneAnimationError(RuntimeError):
    """Known scene-animation failure with a user-facing message."""


def safe_repr(obj, max_len: int = 3000) -> str:
    try:
        text = repr(obj)
    except Exception as e:
        text = f"<repr failed: {e}>"
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def safe_json(obj, max_len: int = 3000) -> str:
    try:
        if obj is None:
            return "null"

        if hasattr(obj, "model_dump"):
            data = obj.model_dump()
        elif hasattr(obj, "to_dict"):
            data = obj.to_dict()
        elif hasattr(obj, "__dict__"):
            data = obj.__dict__
        else:
            return safe_repr(obj, max_len=max_len)

        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        text = f"<json dump failed: {e}> | fallback={safe_repr(obj, max_len=max_len)}"

    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def log_kv(title: str, value) -> None:
    print(f"{title}={value}")


def log_section(title: str) -> None:
    print(f"\n===== {title} =====")


def debug_operation(operation, stage: str) -> None:
    log_section(f"DEBUG OPERATION [{stage}]")
    log_kv("operation_type", type(operation))
    log_kv("operation_done", getattr(operation, "done", None))
    log_kv("operation_name", getattr(operation, "name", None))
    log_kv("operation_id", getattr(operation, "id", None))
    log_kv("operation_error", safe_repr(getattr(operation, "error", None)))
    log_kv("operation_metadata", safe_json(getattr(operation, "metadata", None)))
    log_kv("operation_result_type", type(getattr(operation, "result", None)))
    log_kv("operation_response_type", type(getattr(operation, "response", None)))
    log_kv("operation_repr", safe_repr(operation))


def debug_result(result, stage: str) -> None:
    log_section(f"DEBUG RESULT [{stage}]")
    log_kv("result_is_none", result is None)
    log_kv("result_type", type(result))

    if result is None:
        return

    log_kv("result_repr", safe_repr(result))
    log_kv("result_json", safe_json(result))

    try:
        attrs = [x for x in dir(result) if not x.startswith("_")]
        log_kv("result_attrs", attrs)
    except Exception as e:
        log_kv("result_attrs_error", e)

    generated_videos = getattr(result, "generated_videos", None)
    log_kv("generated_videos_exists", generated_videos is not None)

    if generated_videos is not None:
        try:
            log_kv("generated_videos_len", len(generated_videos))
        except Exception as e:
            log_kv("generated_videos_len_error", e)

        try:
            if len(generated_videos) > 0:
                gv0 = generated_videos[0]
                log_kv("generated_video_0_type", type(gv0))
                log_kv("generated_video_0_repr", safe_repr(gv0))
                log_kv("generated_video_0_json", safe_json(gv0))
                log_kv("generated_video_0_video", safe_repr(getattr(gv0, "video", None)))
                log_kv("generated_video_0_uri", getattr(gv0, "uri", None))
                log_kv("generated_video_0_name", getattr(gv0, "name", None))
        except Exception as e:
            log_kv("generated_video_0_debug_error", e)


def operation_error_message(operation_error) -> str:
    if operation_error is None:
        return ""
    message = getattr(operation_error, "message", None)
    if message:
        return str(message)
    if isinstance(operation_error, dict):
        return str(operation_error.get("message", ""))
    return str(operation_error)


def operation_error_code(operation_error):
    if operation_error is None:
        return None
    code = getattr(operation_error, "code", None)
    if code is not None:
        return code
    if isinstance(operation_error, dict):
        return operation_error.get("code")
    return None


def write_storyboard_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    final_fieldnames = list(fieldnames)
    for extra_col in ["animation_prompt", "animation_video_path"]:
        if extra_col not in final_fieldnames:
            final_fieldnames.append(extra_col)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_face_safety_block(operation_error) -> bool:
    message = operation_error_message(operation_error).lower()
    return (
        "blocked by your current safety settings" in message
        and ("person/face" in message or "face generation" in message)
    )


def is_prompt_policy_block(operation_error) -> bool:
    message = operation_error_message(operation_error).lower()
    return (
        "sensitive words" in message
        or "responsible ai" in message
        or "usage guidelines" in message
    )


def is_retryable_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = [
        "server disconnected without sending a response",
        "remoteprotocolerror",
        "read timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    ]
    return any(marker in message for marker in retry_markers)


def persist_animation_prompt(
    storyboard_csv: Path,
    rows: list[dict],
    fieldnames: list[str],
    scene_id: str,
    new_prompt: str,
) -> None:
    for row in rows:
        if str(row.get("scene_id", "")).strip() == str(scene_id).strip():
            row["animation_prompt"] = new_prompt
            break
    write_storyboard_rows(storyboard_csv, rows, fieldnames)


def format_operation_failure(operation, operation_error) -> str:
    if is_face_safety_block(operation_error):
        return (
            "動畫生成被服務端安全設定擋下：輸入圖片包含人物或臉部內容，"
            "目前這個 image-to-video 設定不允許產生。\n"
            "建議處理方式：\n"
            "1. 改用不含清楚人臉的圖片\n"
            "2. 裁切或更換這張 scene 圖\n"
            "3. 這個 scene 直接保留靜態圖片，不使用動畫\n"
            f"error_code={operation_error_code(operation_error)}\n"
            f"error_message={operation_error_message(operation_error)}"
        )

    return (
        "動畫生成作業已結束，但服務端回報錯誤。\n"
        f"operation_error={safe_repr(operation_error)}\n"
        f"metadata={safe_json(getattr(operation, 'metadata', None))}"
    )


def result_filtered_reasons(result) -> list[str]:
    reasons = getattr(result, "rai_media_filtered_reasons", None)
    if not reasons:
        return []
    return [str(reason) for reason in reasons if str(reason).strip()]


def result_filtered_count(result) -> int:
    count = getattr(result, "rai_media_filtered_count", None)
    try:
        return int(count or 0)
    except Exception:
        return 0


def is_result_policy_filtered(result) -> bool:
    return result_filtered_count(result) > 0 or bool(result_filtered_reasons(result))


def format_result_policy_filtered(result) -> str:
    reasons = result_filtered_reasons(result)
    lines = [
        "動畫生成作業已完成，但輸出影片被服務端內容審核/使用政策過濾，因此沒有返回可下載影片。",
        "建議處理方式：",
        "1. 重新改寫 animation prompt，避免危險、受傷、衝突或其他可能觸發政策的描述",
        "2. 降低人物反應強度，改成更中性、教學式、溫和的動作",
        "3. 若這個 scene 一直被擋，直接保留靜態圖片，不使用動畫",
        f"filtered_count={result_filtered_count(result)}",
    ]
    if reasons:
        lines.append("filtered_reasons=")
        lines.extend(reasons)
    return "\n".join(lines)


def safer_retry_prompt(target_row: dict, current_prompt: str) -> str | None:
    retry_prompt = fallback_prompt(target_row).strip()
    if not retry_prompt:
        return None
    if retry_prompt == str(current_prompt or "").strip():
        return None
    return retry_prompt


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


def ai_image_dir_for_episode(episode_folder: Path, storyboard_csv: Path) -> Path:
    if "05_storyboards" in storyboard_csv.parts:
        return episode_folder / "05_storyboards" / "images" / "ai_generated"
    return episode_folder / "04_images" / "ai_generated"


def animation_dir_for_episode(episode_folder: Path, storyboard_csv: Path) -> Path:
    if "05_storyboards" in storyboard_csv.parts:
        return episode_folder / "05_storyboards" / "animations"
    return episode_folder / "04_images" / "animations"


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


def scene_image_path(episode_folder: Path, row: dict, storyboard_csv: Path | None = None) -> Path | None:
    custom_image_path = str(row.get("custom_image_path", "")).strip()
    if custom_image_path and Path(custom_image_path).exists():
        return Path(custom_image_path)

    start_token = str(row.get("start_time", "")).strip()
    storyboard_csv = storyboard_csv or storyboard_csv_for_episode(episode_folder)
    ai_image_path = ai_image_dir_for_episode(episode_folder, storyboard_csv) / f"img_{start_token}.png"
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


def resolve_generated_video_download_target(generated_video):
    video_obj = getattr(generated_video, "video", None)
    if video_obj is not None:
        return video_obj
    return generated_video


def build_generate_videos_config(duration_seconds: int) -> types.GenerateVideosConfig:
    base_kwargs = {
        "number_of_videos": 1,
        "duration_seconds": duration_seconds,
        "aspect_ratio": "16:9",
    }
    config_kwargs = {**base_kwargs, "person_generation": PERSON_GENERATION}
    try:
        return types.GenerateVideosConfig(**config_kwargs)
    except TypeError as e:
        print(
            "WARN person_generation_not_supported_by_sdk "
            f"person_generation={PERSON_GENERATION} error={e}"
        )
        return types.GenerateVideosConfig(**base_kwargs)


def normalize_animation_provider(value: str | None) -> str:
    provider = str(value or ANIMATION_PROVIDER or "gemini").strip().lower()
    return provider if provider in VALID_ANIMATION_PROVIDERS else "gemini"


def openai_sora_seconds(duration_seconds: int) -> str:
    if duration_seconds <= 4:
        return "4"
    return "8"


def parse_video_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = str(size_text or "").lower().split("x", 1)
        width = int(width_text.strip())
        height = int(height_text.strip())
    except Exception as exc:
        raise SceneAnimationError(f"Invalid CAP_OPENAI_VIDEO_SIZE: {size_text}") from exc
    if width <= 0 or height <= 0:
        raise SceneAnimationError(f"Invalid CAP_OPENAI_VIDEO_SIZE: {size_text}")
    return width, height


def prepare_openai_reference_image(image_path: Path, size_text: str) -> Path:
    target_size = parse_video_size(size_text)
    tmp = tempfile.NamedTemporaryFile(prefix="sora_ref_", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size != target_size:
                print(f"resize_openai_reference_image={img.size}->{target_size}")
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
                f"WARN openai_video_http_retry method={method} attempt={attempt}/{attempts} "
                f"wait_seconds={OPENAI_VIDEO_HTTP_RETRY_SECONDS} error={exc}"
            )
            time.sleep(OPENAI_VIDEO_HTTP_RETRY_SECONDS)
    raise SceneAnimationError(f"OpenAI Sora HTTP request failed after {attempts} attempts: {last_exc}") from last_exc


def generate_openai_video_bytes(animation_prompt: str, image_path: Path, duration_seconds: int) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SceneAnimationError("OPENAI_API_KEY is not configured")

    seconds = os.environ.get("CAP_OPENAI_VIDEO_SECONDS") or openai_sora_seconds(duration_seconds)
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
                    "prompt": animation_prompt,
                    "size": OPENAI_VIDEO_SIZE,
                    "seconds": seconds,
                },
                files={
                    "input_reference": (
                        reference_path.name,
                        image_file,
                        "image/png",
                    )
                },
                timeout=120,
            )
    finally:
        try:
            reference_path.unlink()
        except FileNotFoundError:
            pass
    if response.status_code >= 400:
        raise SceneAnimationError(f"OpenAI Sora submit failed: {response.status_code} {response.text[:1000]}")

    video_job = response.json()
    video_id = str(video_job.get("id") or "").strip()
    if not video_id:
        raise SceneAnimationError(f"OpenAI Sora response missing id: {video_job}")

    log_kv("openai_video_id", video_id)
    log_kv("openai_video_model", OPENAI_VIDEO_MODEL)
    log_kv("openai_video_size", OPENAI_VIDEO_SIZE)
    log_kv("openai_video_seconds", seconds)
    return download_openai_video_bytes(video_id, headers)


def download_openai_video_bytes(video_id: str, headers: dict) -> bytes:
    video_id = str(video_id or "").strip()
    if not video_id:
        raise SceneAnimationError("OpenAI Sora video id is empty")

    video_job = {"status": "in_progress", "id": video_id}
    log_kv("openai_video_id", video_id)

    while str(video_job.get("status", "")).lower() in {"queued", "in_progress"}:
        log_kv("openai_video_status", video_job.get("status"))
        log_kv("openai_video_progress", video_job.get("progress"))
        time.sleep(OPENAI_VIDEO_POLL_SECONDS)
        status_response = request_with_retries(
            "GET",
            f"https://api.openai.com/v1/videos/{video_id}",
            headers=headers,
            timeout=120,
        )
        if status_response.status_code >= 400:
            raise SceneAnimationError(f"OpenAI Sora status failed: {status_response.status_code} {status_response.text[:1000]}")
        video_job = status_response.json()

    if str(video_job.get("status", "")).lower() != "completed":
        raise SceneAnimationError(f"OpenAI Sora generation failed or stopped: {video_job}")

    content_response = request_with_retries(
        "GET",
        f"https://api.openai.com/v1/videos/{video_id}/content",
        headers=headers,
        timeout=600,
    )
    if content_response.status_code >= 400:
        raise SceneAnimationError(f"OpenAI Sora download failed: {content_response.status_code} {content_response.text[:1000]}")
    if not content_response.content:
        raise SceneAnimationError("OpenAI Sora returned empty video content")
    return content_response.content


def generate_scene_animation(
    ep_num: int,
    scene_id: str,
    image_path_override: str = "",
    provider: str = "gemini",
    openai_video_id: str = "",
) -> Path:
    provider = normalize_animation_provider(provider)
    episode_folder = find_episode_folder(ep_num)
    if not episode_folder:
        raise FileNotFoundError(f"找不到第 {ep_num:02d} 集資料夾")

    storyboard_csv = storyboard_csv_for_episode(episode_folder)
    if not storyboard_csv.exists():
        raise FileNotFoundError(f"找不到 storyboard.csv: {storyboard_csv}")

    rows, fieldnames = read_storyboard_rows(storyboard_csv)
    target_row = next((row for row in rows if str(row.get("scene_id", "")).strip() == str(scene_id)), None)
    if not target_row and provider == "openai" and openai_video_id:
        animation_dir = animation_dir_for_episode(episode_folder, storyboard_csv)
        animation_dir.mkdir(parents=True, exist_ok=True)
        output_path = next_animation_variant_path(animation_dir, scene_id)
        print("STEP 1/4 resume without storyboard row")
        log_kv("scene_id", scene_id)
        log_kv("output_path", output_path)
        log_kv("animation_provider", provider)
        log_kv("openai_video_id", openai_video_id)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        video_bytes = download_openai_video_bytes(openai_video_id, headers)

        raw_output_path = output_path.with_name(output_path.stem + "__raw" + output_path.suffix)
        raw_output_path.write_bytes(video_bytes)
        log_kv("raw_output_path", raw_output_path)
        log_kv("raw_video_size_bytes", len(video_bytes))
        print("STEP 4.5/4 strip audio track from generated video")
        strip_audio_track(raw_output_path, output_path)
        try:
            raw_output_path.unlink()
        except FileNotFoundError:
            pass
        print("DONE animation generated")
        log_kv("saved_to", output_path)
        print("WARN storyboard row was missing; video was saved but storyboard.csv was not updated.")
        return output_path
    if not target_row:
        raise ValueError(f"找不到 scene_id={scene_id}")

    animation_prompt = str(target_row.get("animation_prompt", "")).strip()
    if not animation_prompt:
        raise ValueError("animation_prompt 為空，請先產生或編輯動畫 Prompt")

    explicit_image_path = Path(str(image_path_override or "").strip()) if str(image_path_override or "").strip() else None
    image_path = explicit_image_path if explicit_image_path else scene_image_path(episode_folder, target_row, storyboard_csv=storyboard_csv)
    if not image_path or not image_path.exists():
        raise FileNotFoundError("找不到這個 scene 對應圖片，請先確認 AI 圖片或自訂圖片來源")

    animation_dir = animation_dir_for_episode(episode_folder, storyboard_csv)
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

    request_duration = max(min(scene_duration, 8), 5) if scene_duration else DEFAULT_ANIMATION_SECONDS

    print("STEP 1/4 validate inputs")
    log_kv("scene_id", scene_id)
    log_kv("image_path", image_path)
    log_kv("output_path", output_path)
    log_kv("duration_seconds", request_duration)
    log_kv("animation_provider", provider)
    log_kv("video_model", VIDEO_MODEL)
    log_kv("openai_video_model", OPENAI_VIDEO_MODEL)
    log_kv("poll_seconds", POLL_SECONDS)
    log_kv("person_generation", PERSON_GENERATION)
    log_kv("image_mime_type", guess_mime_type(image_path))
    log_kv("image_size_bytes", len(image_bytes))
    log_kv("prompt_length", len(animation_prompt))
    log_kv("prompt_preview", animation_prompt[:500])

    video_config = build_generate_videos_config(request_duration)
    log_kv("video_config", safe_json(video_config))

    safe_prompt_retried = False
    network_retried = False
    video_bytes = b""

    if provider == "openai":
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        if openai_video_id:
            print("STEP 2/4 resume OpenAI Sora video request")
            video_bytes = download_openai_video_bytes(openai_video_id, headers)
        else:
            print("STEP 2/4 submit OpenAI Sora image-to-video request")
            video_bytes = generate_openai_video_bytes(animation_prompt, image_path, request_duration)

    while not video_bytes:
        video_bytes = b""
        target_row["animation_prompt"] = animation_prompt
        print("STEP 2/4 submit image-to-video request")
        try:
            operation = client.models.generate_videos(
                model=VIDEO_MODEL,
                source=types.GenerateVideosSource(
                    prompt=animation_prompt,
                    image=types.Image(
                        image_bytes=image_bytes,
                        mime_type=guess_mime_type(image_path),
                    ),
                ),
                config=video_config,
            )
        except Exception as e:
            if not network_retried and is_retryable_network_error(e):
                network_retried = True
                print(
                    f"WARN network_retry submit scene_id={scene_id} wait_seconds={NETWORK_RETRY_SECONDS} error={e}"
                )
                time.sleep(NETWORK_RETRY_SECONDS)
                continue
            raise SceneAnimationError(f"提交動畫生成請求失敗：{e}") from e

        log_kv("operation_submitted_done", getattr(operation, "done", None))
        debug_operation(operation, "after_submit")

        poll_count = 0
        while not getattr(operation, "done", False):
            poll_count += 1
            print(f"STEP 3/4 polling operation poll_count={poll_count} wait_seconds={POLL_SECONDS}")
            time.sleep(POLL_SECONDS)

            try:
                operation = client.operations.get(operation)
            except Exception as e:
                if not network_retried and is_retryable_network_error(e):
                    network_retried = True
                    print(
                        f"WARN network_retry poll scene_id={scene_id} wait_seconds={NETWORK_RETRY_SECONDS} error={e}"
                    )
                    time.sleep(NETWORK_RETRY_SECONDS)
                    break
                raise SceneAnimationError(f"輪詢動畫生成狀態失敗：{e}") from e

            log_kv("operation_status_done", getattr(operation, "done", None))
            log_kv("operation_status_error", safe_repr(getattr(operation, "error", None)))
        else:
            debug_operation(operation, "after_polling_done")

            result = getattr(operation, "result", None) or getattr(operation, "response", None)
            debug_result(result, "after_polling_done")

            operation_error = getattr(operation, "error", None)
            if operation_error:
                retry_prompt = None
                if (not safe_prompt_retried) and is_prompt_policy_block(operation_error):
                    retry_prompt = safer_retry_prompt(target_row, animation_prompt)
                if retry_prompt:
                    safe_prompt_retried = True
                    animation_prompt = retry_prompt
                    persist_animation_prompt(storyboard_csv, rows, fieldnames, str(scene_id), animation_prompt)
                    print(f"WARN safe_prompt_retry scene_id={scene_id} reason=operation_error")
                    log_kv("retry_animation_prompt", animation_prompt)
                    continue
                raise SceneAnimationError(format_operation_failure(operation, operation_error))

            if result is None:
                raise SceneAnimationError(
                    "動畫生成作業已完成，但 operation.result / operation.response 都是空值。\n"
                    f"operation={safe_repr(operation)}"
                )

            generated_videos = getattr(result, "generated_videos", None)
            if not generated_videos:
                retry_prompt = None
                if (not safe_prompt_retried) and is_result_policy_filtered(result):
                    retry_prompt = safer_retry_prompt(target_row, animation_prompt)
                if retry_prompt:
                    safe_prompt_retried = True
                    animation_prompt = retry_prompt
                    persist_animation_prompt(storyboard_csv, rows, fieldnames, str(scene_id), animation_prompt)
                    print(f"WARN safe_prompt_retry scene_id={scene_id} reason=policy_filtered")
                    log_kv("retry_animation_prompt", animation_prompt)
                    continue
                if is_result_policy_filtered(result):
                    raise SceneAnimationError(format_result_policy_filtered(result))
                raise SceneAnimationError(
                    "動畫生成作業已完成，但結果內沒有 generated_videos。\n"
                    f"result_type={type(result)}\n"
                    f"result_json={safe_json(result)}\n"
                    f"operation_metadata={safe_json(getattr(operation, 'metadata', None))}\n"
                    f"operation_error={safe_repr(operation_error)}"
                )

            print("STEP 4/4 download generated video")
            generated_video = generated_videos[0]
            download_target = resolve_generated_video_download_target(generated_video)

            log_section("DEBUG DOWNLOAD TARGET")
            log_kv("generated_video_type", type(generated_video))
            log_kv("generated_video_repr", safe_repr(generated_video))
            log_kv("generated_video_json", safe_json(generated_video))
            log_kv("download_target_type", type(download_target))
            log_kv("download_target_repr", safe_repr(download_target))
            log_kv("download_target_json", safe_json(download_target))

            try:
                video_bytes = client.files.download(file=download_target)
            except Exception as e:
                if not network_retried and is_retryable_network_error(e):
                    network_retried = True
                    print(
                        f"WARN network_retry download scene_id={scene_id} wait_seconds={NETWORK_RETRY_SECONDS} error={e}"
                    )
                    time.sleep(NETWORK_RETRY_SECONDS)
                    continue
                raise SceneAnimationError(
                    "動畫生成完成，但下載結果影片失敗。\n"
                    f"download_target={safe_repr(download_target)}\n"
                    f"generated_video={safe_repr(generated_video)}\n"
                    f"error={e}"
                ) from e

            if not video_bytes:
                raise SceneAnimationError("影片下載回傳空內容，找不到可寫入的動畫資料。")

            break

        if video_bytes:
            break
        continue

    raw_output_path = output_path.with_name(output_path.stem + "__raw" + output_path.suffix)
    raw_output_path.write_bytes(video_bytes)
    log_kv("raw_output_path", raw_output_path)
    log_kv("raw_video_size_bytes", len(video_bytes))

    print("STEP 4.5/4 strip audio track from generated video")
    try:
        strip_audio_track(raw_output_path, output_path)
    except Exception as e:
        raise SceneAnimationError(
            f"影片已下載，但移除音軌失敗：{e}\n"
            f"raw_output_path={raw_output_path}"
        ) from e

    try:
        raw_output_path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARN failed_to_delete_raw_output error={e}")

    print("DONE animation generated")
    log_kv("saved_to", output_path)
    tag_asset_from_row(output_path, target_row, asset_kind="animation", overwrite=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="為指定 scene 產生動畫")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--image_path", default="")
    parser.add_argument("--provider", default=os.environ.get("CAP_ANIMATION_PROVIDER", "gemini"), choices=sorted(VALID_ANIMATION_PROVIDERS))
    parser.add_argument("--openai_video_id", default="", help="Resume an existing OpenAI Sora video job and download it.")
    args = parser.parse_args()

    try:
        generate_scene_animation(
            args.ep,
            str(args.scene_id),
            image_path_override=str(args.image_path or ""),
            provider=args.provider,
            openai_video_id=str(args.openai_video_id or ""),
        )
    except (SceneAnimationError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
