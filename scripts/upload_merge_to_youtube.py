import argparse
import json
import mimetypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_utils.power import keep_system_awake
from upload_to_youtube import (
    DEFAULT_AUDIO_LANGUAGE,
    DEFAULT_LANGUAGE,
    VIDEO_UPLOAD_CHUNK_SIZE,
    VIDEO_UPLOAD_MAX_RETRIES,
    add_video_to_playlist,
    get_authenticated_service,
    prepare_thumbnail_for_upload,
    resumable_upload_with_retries,
    update_video_languages_and_localization,
    upload_subtitle,
    upload_thumbnail,
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_merge_upload_record(target_path: Path, record_data: dict) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(record_data, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"[merge-upload] upload record saved: {target_path.name}", flush=True)


def format_publish_at(value: str) -> tuple[str | None, str]:
    text = str(value or "").strip()
    if not text:
        return None, ""
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        return None, f"時間格式錯誤：{text}，請使用 YYYY-MM-DD HH:MM"
    return dt.strftime("%Y-%m-%dT%H:%M:%S+08:00"), ""


def metadata_description(meta: dict) -> str:
    description = str(meta.get("description") or "").strip()
    if description:
        return description
    summary = str(meta.get("summary") or "").strip()
    chapters = meta.get("chapters") or []
    chapter_text = "\n".join(
        f"{item.get('time', '')} {item.get('title', '')}".strip()
        for item in chapters
        if isinstance(item, dict)
    )
    return "\n\n".join(part for part in [summary, "章節時間軸：\n" + chapter_text if chapter_text else ""] if part).strip()


def upload_merge_job(
    job_path: Path,
    privacy_status: str,
    publish_at: str = "",
    playlist_id: str = "",
    cover_path: Path | None = None,
    proc_path: Path | None = None,
) -> int:
    job = read_json(job_path)
    if not job:
        raise FileNotFoundError(f"merge job not found or invalid: {job_path}")

    output_dir = Path(str(job.get("output_dir") or "")).resolve()
    video_path = Path(str(job.get("output_video") or output_dir / "final_video.mp4")).resolve()
    subtitle_path = Path(str(job.get("output_subtitle") or output_dir / "merged_subtitles.srt")).resolve()
    meta_path = output_dir / "youtube_meta.json"
    record_path = output_dir / "upload_record.json"
    if cover_path is None:
        cover_candidates = [
            output_dir / "cover.png",
            output_dir / "cover.jpg",
            output_dir / "cover.jpeg",
            output_dir / "thumbnail.png",
            output_dir / "thumbnail.jpg",
            output_dir / "thumbnail.jpeg",
        ]
        cover_path = next((path for path in cover_candidates if path.exists()), None)

    if not video_path.exists():
        raise FileNotFoundError(f"缺少合併影片：{video_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 youtube_meta.json：{meta_path}")

    meta_data = read_json(meta_path)
    title = str(meta_data.get("title") or job.get("title") or "Merged Episode").strip()
    tags = [str(tag).strip() for tag in (meta_data.get("tags") or []) if str(tag).strip()]
    description = metadata_description(meta_data)
    formatted_publish_at, publish_error = format_publish_at(publish_at)
    if publish_error:
        raise ValueError(publish_error)
    if formatted_publish_at:
        privacy_status = "private"

    proc_path = proc_path or job_path.with_suffix(job_path.suffix + ".upload.proc.json")
    proc_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": int(time.time())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    record = {
        "merge_job_id": job.get("id", ""),
        "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "starting",
        "video_id": None,
        "title": title,
        "privacy": privacy_status,
        "publish_at": publish_at,
        "playlist_id": playlist_id,
        "subtitle_uploaded": False,
        "thumbnail_uploaded": False,
        "playlist_added": False,
        "default_language": DEFAULT_LANGUAGE,
        "default_audio_language": DEFAULT_AUDIO_LANGUAGE,
        "video_upload_chunk_size": VIDEO_UPLOAD_CHUNK_SIZE,
        "video_upload_max_retries": VIDEO_UPLOAD_MAX_RETRIES,
        "video_path": str(video_path),
        "subtitle_path": str(subtitle_path),
        "metadata_path": str(meta_path),
        "cover_path": str(cover_path or ""),
    }
    save_merge_upload_record(record_path, record)

    try:
        with keep_system_awake(f"Merge YouTube upload {job.get('id') or job_path.stem}"):
            youtube = get_authenticated_service()
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "27",
                    "defaultLanguage": DEFAULT_LANGUAGE,
                    "defaultAudioLanguage": DEFAULT_AUDIO_LANGUAGE,
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False,
                },
            }
            if formatted_publish_at:
                body["status"]["publishAt"] = formatted_publish_at

            print("[merge-upload] uploading video", flush=True)
            print(f"[merge-upload] video={video_path}", flush=True)
            print(f"[merge-upload] size={video_path.stat().st_size:,} bytes", flush=True)
            media_body = MediaFileUpload(
                str(video_path),
                mimetype=mimetypes.guess_type(str(video_path))[0] or "video/mp4",
                chunksize=VIDEO_UPLOAD_CHUNK_SIZE,
                resumable=True,
            )
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media_body)
            response = resumable_upload_with_retries(request)
            video_id = response["id"]
            record["video_id"] = video_id
            print(f"[merge-upload] uploaded video_id={video_id}", flush=True)

            update_video_languages_and_localization(youtube, video_id, title, description, tags)

            if subtitle_path.exists():
                record["subtitle_uploaded"] = upload_subtitle(youtube, video_id, subtitle_path)
            else:
                print(f"[merge-upload] subtitle not found, skipped: {subtitle_path}", flush=True)

            if cover_path and cover_path.exists():
                prepared_cover, thumbnail_info = prepare_thumbnail_for_upload(cover_path, output_dir)
                record["thumbnail_info"] = thumbnail_info
                record["thumbnail_uploaded"] = upload_thumbnail(youtube, video_id, prepared_cover)
            else:
                print("[merge-upload] cover not found, thumbnail skipped", flush=True)

            if playlist_id:
                record["playlist_added"] = add_video_to_playlist(youtube, video_id, playlist_id)

            record["status"] = "success"
            record["ended_at"] = datetime.now().isoformat(timespec="seconds")
            save_merge_upload_record(record_path, record)
            (output_dir / "upload.ok").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            print("[merge-upload] success", flush=True)
            return 0
    except Exception as exc:
        record["status"] = f"failed: {exc}"
        record["ended_at"] = datetime.now().isoformat(timespec="seconds")
        save_merge_upload_record(record_path, record)
        print(f"[merge-upload] failed: {exc}", flush=True)
        return 1
    finally:
        try:
            proc_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload merged episode video to YouTube.")
    parser.add_argument("--job", required=True, help="Path to merge job JSON.")
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private")
    parser.add_argument("--publish_at", default="", help="YYYY-MM-DD HH:MM in Asia/Taipei.")
    parser.add_argument("--playlist_id", default="")
    parser.add_argument("--cover_path", default="")
    parser.add_argument("--proc_path", default="")
    args = parser.parse_args()
    return upload_merge_job(
        Path(args.job).resolve(),
        privacy_status=args.privacy,
        publish_at=args.publish_at,
        playlist_id=args.playlist_id,
        cover_path=Path(args.cover_path).resolve() if args.cover_path else None,
        proc_path=Path(args.proc_path).resolve() if args.proc_path else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
