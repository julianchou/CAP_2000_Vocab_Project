import argparse
import json
import mimetypes
import sys
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
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_episode_status(ep_dir: Path, status: str, video_id: str = "") -> None:
    episode_path = ep_dir / "episode.json"
    data = read_json(episode_path)
    data["status"] = status
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if video_id:
        data["youtube_video_id"] = video_id
        data["youtube_url"] = f"https://www.youtube.com/watch?v={video_id}"
        data["uploaded_at"] = data["updated_at"]
    write_json(episode_path, data)


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
    hashtags = meta.get("hashtags") if isinstance(meta.get("hashtags"), list) else []
    hashtag_line = " ".join(str(item).strip() for item in hashtags if str(item).strip())
    pinned = str(meta.get("pinned_comment") or "").strip()
    parts = [description, hashtag_line, f"置頂留言建議：{pinned}" if pinned else ""]
    return "\n\n".join(part for part in parts if part).strip()


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def upload_short(
    ep_dir: Path,
    privacy_status: str,
    publish_at: str = "",
    playlist_id: str = "",
    cover_path: Path | None = None,
) -> int:
    output_dir = ep_dir / "05_output"
    video_path = output_dir / "final_short.mp4"
    meta_path = output_dir / "short_metadata.json"
    record_path = output_dir / "upload_record.json"
    subtitle_path = first_existing([
        ep_dir / "02_subtitles" / "final.srt",
        ep_dir / "02_subtitles" / "reviewed.srt",
        ep_dir / "02_subtitles" / "whisper.srt",
    ])

    if cover_path is None:
        cover_path = first_existing([
            output_dir / "cover.png",
            output_dir / "cover.jpg",
            output_dir / "cover.jpeg",
            ep_dir / "04_images" / "storyboard" / "scene_001.png",
        ])

    if not video_path.exists():
        raise FileNotFoundError(f"缺少 final_short.mp4：{video_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 short_metadata.json：{meta_path}")

    meta = read_json(meta_path)
    title = str(meta.get("title") or ep_dir.name).strip()
    description = metadata_description(meta)
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    tags = [str(tag).strip().lstrip("#") for tag in tags if str(tag).strip()]
    if not tags:
        tags = [str(tag).strip().lstrip("#") for tag in meta.get("hashtags", []) if str(tag).strip()]

    formatted_publish_at, publish_error = format_publish_at(publish_at)
    if publish_error:
        raise ValueError(publish_error)
    if formatted_publish_at:
        privacy_status = "private"

    record = {
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
        "subtitle_path": str(subtitle_path or ""),
        "metadata_path": str(meta_path),
        "cover_path": str(cover_path or ""),
    }
    write_json(record_path, record)

    try:
        with keep_system_awake(f"Short YouTube upload {ep_dir.name}"):
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

            print(f"[short-upload] video={video_path}", flush=True)
            print(f"[short-upload] title={title}", flush=True)
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
            print(f"[short-upload] uploaded video_id={video_id}", flush=True)

            update_video_languages_and_localization(youtube, video_id, title, description, tags)
            if subtitle_path and subtitle_path.exists():
                record["subtitle_uploaded"] = upload_subtitle(youtube, video_id, subtitle_path)
            if cover_path and cover_path.exists():
                prepared_cover, thumbnail_info = prepare_thumbnail_for_upload(cover_path, output_dir)
                record["thumbnail_info"] = thumbnail_info
                record["thumbnail_uploaded"] = upload_thumbnail(youtube, video_id, prepared_cover)
            if playlist_id:
                record["playlist_added"] = add_video_to_playlist(youtube, video_id, playlist_id)

            record["status"] = "success"
            record["ended_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(record_path, record)
            (output_dir / "upload.ok").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            (output_dir / "uploaded.txt").write_text(record["ended_at"], encoding="utf-8")
            update_episode_status(ep_dir, "已上傳", video_id)
            print("[short-upload] success", flush=True)
            return 0
    except Exception as exc:
        record["status"] = f"failed: {exc}"
        record["ended_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(record_path, record)
        print(f"[short-upload] failed: {exc}", flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Short generator final video to YouTube.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private")
    parser.add_argument("--publish_at", default="")
    parser.add_argument("--playlist_id", default="")
    parser.add_argument("--cover_path", default="")
    args = parser.parse_args()
    return upload_short(
        ep_dir=Path(args.ep_dir).resolve(),
        privacy_status=args.privacy,
        publish_at=args.publish_at,
        playlist_id=args.playlist_id,
        cover_path=Path(args.cover_path).resolve() if args.cover_path else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
