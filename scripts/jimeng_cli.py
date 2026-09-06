import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_POLL_SECONDS = 10
TERMINAL_SUCCESS_STATUSES = {"succeeded", "completed", "success"}
TERMINAL_FAILURE_STATUSES = {"failed", "cancelled", "canceled", "expired"}


class JimengCliError(RuntimeError):
    pass


def api_key() -> str:
    value = os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_ARK_API_KEY")
    if not value:
        raise JimengCliError(
            "ARK_API_KEY is not configured. Add it to .env or set it in the current shell."
        )
    return value


def base_url() -> str:
    return os.getenv("ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=120, **kwargs)
    except requests.RequestException as exc:
        raise JimengCliError(f"Ark API request failed: {exc}") from exc

    if response.status_code >= 400:
        raise JimengCliError(
            f"Ark API returned HTTP {response.status_code}: {response.text[:2000]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise JimengCliError(
            f"Ark API returned invalid JSON: {response.text[:1000]}"
        ) from exc
    if not isinstance(payload, dict):
        raise JimengCliError(f"Ark API returned an unexpected payload: {payload!r}")
    return payload


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise JimengCliError(f"Image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime.startswith("image/"):
        raise JimengCliError(f"Unsupported image type: {mime}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_reference(image: str) -> str:
    value = str(image or "").strip()
    if value.startswith(("https://", "http://", "data:image/")):
        return value
    return image_data_url(Path(value).expanduser().resolve())


def build_prompt(
    prompt: str,
    resolution: str,
    duration: int,
    aspect_ratio: str,
    camera_fixed: bool,
    watermark: bool,
) -> str:
    options = [
        f"--resolution {resolution}",
        f"--duration {duration}",
        f"--ratio {aspect_ratio}",
        f"--camerafixed {str(camera_fixed).lower()}",
        f"--watermark {str(watermark).lower()}",
    ]
    return f"{prompt.strip()} {' '.join(options)}".strip()


def create_task(
    model: str,
    prompt: str,
    image: str = "",
    resolution: str = "720p",
    duration: int = 5,
    aspect_ratio: str = "16:9",
    camera_fixed: bool = False,
    watermark: bool = False,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_prompt(
                prompt,
                resolution=resolution,
                duration=duration,
                aspect_ratio=aspect_ratio,
                camera_fixed=camera_fixed,
                watermark=watermark,
            ),
        }
    ]
    if image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_reference(image)},
            }
        )

    return request_json(
        "POST",
        f"{base_url()}/contents/generations/tasks",
        headers=headers(),
        json={"model": model, "content": content},
    )


def get_task(task_id: str) -> dict[str, Any]:
    return request_json(
        "GET",
        f"{base_url()}/contents/generations/tasks/{task_id}",
        headers=headers(),
    )


def task_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or "").strip().lower()


def task_id(payload: dict[str, Any]) -> str:
    return str(payload.get("id") or payload.get("task_id") or "").strip()


def video_url(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, dict):
        value = content.get("video_url") or content.get("url")
        if value:
            return str(value)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            value = item.get("video_url") or item.get("url")
            if value:
                return str(value)
    output = payload.get("output")
    if isinstance(output, dict):
        value = output.get("video_url") or output.get("url")
        if value:
            return str(value)
    return str(payload.get("video_url") or "").strip()


def wait_for_task(task_id_value: str, poll_seconds: int) -> dict[str, Any]:
    while True:
        payload = get_task(task_id_value)
        status = task_status(payload)
        print(f"[INFO] task_id={task_id_value} status={status or 'unknown'}", flush=True)
        if status in TERMINAL_SUCCESS_STATUSES:
            return payload
        if status in TERMINAL_FAILURE_STATUSES:
            raise JimengCliError(
                f"Video generation ended with status={status}: "
                f"{json.dumps(payload, ensure_ascii=False)[:2000]}"
            )
        time.sleep(max(1, poll_seconds))


def download_file(url: str, output: Path) -> Path:
    if not url:
        raise JimengCliError("Completed task does not contain a video URL.")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=600) as response:
            if response.status_code >= 400:
                raise JimengCliError(
                    f"Video download returned HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )
            with output.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
    except requests.RequestException as exc:
        raise JimengCliError(f"Video download failed: {exc}") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise JimengCliError(f"Downloaded video is empty: {output}")
    return output


def configured_model(value: str) -> str:
    model = str(value or os.getenv("ARK_VIDEO_MODEL") or "").strip()
    if not model:
        raise JimengCliError(
            "No Ark video model configured. Pass --model or set ARK_VIDEO_MODEL."
        )
    return model


def output_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def add_task_id_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task_id", help="Ark video generation task ID")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="即夢風格影片 CLI：透過火山方舟影片生成 API 建立與下載影片。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="建立文字或圖片生成影片任務")
    generate.add_argument("--prompt", required=True, help="影片動態提示詞")
    generate.add_argument("--image", default="", help="本機圖片路徑或公開圖片 URL")
    generate.add_argument("--model", default="", help="方舟模型 ID 或推理接入點 ID")
    generate.add_argument("--output", default="", help="等待完成並下載至此 MP4 路徑")
    generate.add_argument("--resolution", default="720p", choices=["480p", "720p", "1080p"])
    generate.add_argument("--duration", type=int, default=5)
    generate.add_argument("--ratio", default="16:9")
    generate.add_argument("--camera-fixed", action="store_true")
    generate.add_argument("--watermark", action="store_true")
    generate.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.getenv("ARK_VIDEO_POLL_SECONDS", DEFAULT_POLL_SECONDS)),
    )
    generate.add_argument(
        "--no-wait",
        action="store_true",
        help="只提交任務並輸出 task ID，不等待結果",
    )

    status = subparsers.add_parser("status", help="查詢任務狀態")
    add_task_id_argument(status)

    wait = subparsers.add_parser("wait", help="等待既有任務完成")
    add_task_id_argument(wait)
    wait.add_argument("--output", required=True, help="下載 MP4 路徑")
    wait.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.getenv("ARK_VIDEO_POLL_SECONDS", DEFAULT_POLL_SECONDS)),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "status":
        output_payload(get_task(args.task_id))
        return 0

    if args.command == "wait":
        payload = wait_for_task(args.task_id, args.poll_seconds)
        output_payload(payload)
        saved = download_file(video_url(payload), Path(args.output))
        print(f"[DONE] saved_to={saved}")
        return 0

    payload = create_task(
        model=configured_model(args.model),
        prompt=args.prompt,
        image=args.image,
        resolution=args.resolution,
        duration=args.duration,
        aspect_ratio=args.ratio,
        camera_fixed=args.camera_fixed,
        watermark=args.watermark,
    )
    output_payload(payload)
    created_task_id = task_id(payload)
    if not created_task_id:
        raise JimengCliError(f"Create response does not contain a task ID: {payload}")
    print(f"[INFO] task_id={created_task_id}")
    if args.no_wait:
        return 0

    completed = wait_for_task(created_task_id, args.poll_seconds)
    output_payload(completed)
    if args.output:
        saved = download_file(video_url(completed), Path(args.output))
        print(f"[DONE] saved_to={saved}")
    else:
        print(f"[DONE] video_url={video_url(completed)}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except JimengCliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
