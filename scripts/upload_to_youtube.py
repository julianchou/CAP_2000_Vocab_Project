import os
import csv
import json
import pickle
import argparse
import mimetypes
from datetime import datetime
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# 權限範圍：上傳影片 / 設定字幕 / 設定封面 / 更新影片資訊 / 播放清單
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
CLIENT_SECRETS_FILE = base_dir / "client_secret.json"
TOKEN_FILE = base_dir / "token_youtube.pickle"
SCHEDULE_FILE = base_dir / "upload_schedule.csv"

DEFAULT_LANGUAGE = "zh-TW"
DEFAULT_AUDIO_LANGUAGE = "zh-TW"
YOUTUBE_THUMBNAIL_MAX_BYTES = 2 * 1024 * 1024

try:
    from PIL import Image
except Exception:
    Image = None


def load_upload_schedule():
    """讀取 CSV 排程表並回傳字典"""
    schedule = {}
    if SCHEDULE_FILE.exists():
        with open(SCHEDULE_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(row['episode'])
                    schedule[ep] = {
                        "publish_time": row['publish_time'].strip(),
                        "privacy": row['privacy'].strip()
                    }
                except (ValueError, KeyError):
                    continue
    return schedule


def find_episode_folder(ep_num):
    return next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)


def load_publish_settings_for_episode(ep_num):
    target_folder = find_episode_folder(ep_num)
    if not target_folder:
        return {}
    path = target_folder / "05_output" / "publish_settings.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception as e:
        print(f"⚠️ 讀取發布設定失敗（Ep{ep_num:02d}）：{e}")
        return {}


def resolve_episode_upload_options(ep_num, args, schedule_data):
    saved = load_publish_settings_for_episode(ep_num)
    schedule_row = schedule_data.get(ep_num, {})

    saved_privacy = str(saved.get("privacy", "")).strip()
    saved_schedule_on = bool(saved.get("schedule_on"))
    saved_publish_at = str(saved.get("publish_at", "")).strip() if saved_schedule_on else ""
    saved_playlist_id = str(saved.get("playlist_id", "")).strip()
    saved_cover_name = str(saved.get("cover_name", "")).strip()

    privacy = args.privacy or saved_privacy or schedule_row.get("privacy", "private")
    publish_at = args.publish_at if args.publish_at is not None else (saved_publish_at or schedule_row.get("publish_time"))
    playlist_id = args.playlist_id if args.playlist_id is not None else saved_playlist_id
    cover_name = args.cover_name if args.cover_name is not None else (saved_cover_name or "cover.png")

    return {
        "privacy": privacy,
        "publish_at": publish_at,
        "playlist_id": playlist_id,
        "cover_name": cover_name,
    }


from google.auth.exceptions import RefreshError

def get_authenticated_service():
    """獲取 YouTube API 服務（支援 token 失效時自動重授權）"""
    creds = None

    # 1. 先讀既有 token
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            print(f"⚠️ 讀取 token 檔失敗，將重新授權：{e}")
            creds = None

    # 2. 若沒有憑證或憑證無效，嘗試 refresh / 重新登入
    if not creds or not creds.valid:
        need_new_login = False

        if creds and creds.expired and creds.refresh_token:
            try:
                print("🔄 YouTube token 已過期，嘗試自動刷新...")
                creds.refresh(Request())
                print("✅ Token 刷新成功。")
            except RefreshError as e:
                print(f"⚠️ Token 刷新失敗，可能已失效或被撤銷：{e}")
                need_new_login = True
            except Exception as e:
                print(f"⚠️ Token 刷新時發生未知錯誤：{e}")
                need_new_login = True
        else:
            need_new_login = True

        # 3. refresh 失敗或無法 refresh，就重新登入
        if need_new_login:
            try:
                if TOKEN_FILE.exists():
                    TOKEN_FILE.unlink()
                    print(f"🗑️ 已刪除失效 token：{TOKEN_FILE.name}")
            except Exception as e:
                print(f"⚠️ 刪除舊 token 失敗：{e}")

            if not CLIENT_SECRETS_FILE.exists():
                raise FileNotFoundError(f"找不到 client_secret.json：{CLIENT_SECRETS_FILE}")

            print("🌐 開始進行 YouTube OAuth 重新授權...")
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
            print("✅ 重新授權成功。")

        # 4. 儲存最新 token
        try:
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
            print(f"💾 已儲存最新 token：{TOKEN_FILE.name}")
        except Exception as e:
            print(f"⚠️ 儲存 token 失敗：{e}")

    return build('youtube', 'v3', credentials=creds)
    
    """獲取 YouTube API 服務"""
    creds = None
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return build('youtube', 'v3', credentials=creds)


def add_video_to_playlist(youtube, video_id, playlist_id):
    """將影片加入指定播放清單"""
    print(f"🔗 正在將影片 {video_id} 加入播放清單 {playlist_id}...")
    try:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        ).execute()
        print("✅ 成功加入播放清單！")
        return True
    except Exception as e:
        print(f"❌ 無法加入播放清單: {e}")
        return False


def upload_subtitle(youtube, video_id, srt_path):
    """上傳字幕檔"""
    if not srt_path.exists():
        print(f"⚠️ 找不到字幕檔: {srt_path.name}，跳過。")
        return False

    print(f"🗨️ 正在上傳繁體中文字幕：{srt_path.name}")
    try:
        youtube.captions().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "language": DEFAULT_LANGUAGE,
                    "name": "繁體中文",
                    "isDraft": False
                }
            },
            media_body=MediaFileUpload(
                str(srt_path),
                mimetype="application/octet-stream",
                resumable=True
            )
        ).execute()
        print("✅ 字幕上傳成功！")
        return True
    except Exception as e:
        print(f"❌ 字幕上傳失敗: {e}")
        return False


def upload_thumbnail(youtube, video_id, image_path: Path):
    """上傳封面圖"""
    if not image_path.exists():
        print(f"⚠️ 找不到封面圖: {image_path.name}，跳過。")
        return False

    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    print(f"🖼️ 正在上傳封面圖：{image_path.name}")
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(
                str(image_path),
                mimetype=mime_type,
                resumable=True
            )
        ).execute()
        print("✅ 封面圖設定完成！")
        return True
    except Exception as e:
        print(f"❌ 封面圖上傳失敗: {e}")
        return False


def prepare_thumbnail_for_upload(image_path: Path, output_dir: Path) -> tuple[Path, dict]:
    info = {
        "source_path": str(image_path),
        "upload_path": str(image_path),
        "source_bytes": image_path.stat().st_size if image_path.exists() else 0,
        "upload_bytes": image_path.stat().st_size if image_path.exists() else 0,
        "compressed": False,
        "reason": "",
    }
    if not image_path.exists():
        info["reason"] = "missing"
        return image_path, info
    if info["source_bytes"] <= YOUTUBE_THUMBNAIL_MAX_BYTES:
        info["reason"] = "within_limit"
        return image_path, info
    if Image is None:
        info["reason"] = "pillow_unavailable"
        print("[WARN] 封面圖超過 2MB，但 Pillow 無法使用，將嘗試上傳原圖。")
        return image_path, info

    output_dir.mkdir(parents=True, exist_ok=True)
    compressed_path = output_dir / f"{image_path.stem}_youtube_thumbnail.jpg"
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")

            working = img
            for scale in (1.0, 0.92, 0.85, 0.78):
                if scale != 1.0:
                    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                    working = img.resize(new_size)
                for quality in (92, 88, 84, 80, 76, 72, 68):
                    working.save(compressed_path, format="JPEG", quality=quality, optimize=True)
                    size = compressed_path.stat().st_size
                    if size <= YOUTUBE_THUMBNAIL_MAX_BYTES:
                        info.update(
                            {
                                "upload_path": str(compressed_path),
                                "upload_bytes": size,
                                "compressed": True,
                                "reason": f"compressed_jpeg_quality_{quality}_scale_{scale}",
                            }
                        )
                        print(
                            "[INFO] 正式封面超過 2MB，已建立 YouTube 上傳用壓縮檔："
                            f"{compressed_path.name} ({size} bytes)"
                        )
                        return compressed_path, info
        info["reason"] = "compression_still_too_large"
        print("[WARN] 封面圖壓縮後仍超過 2MB，將嘗試上傳最後產出的壓縮檔。")
        if compressed_path.exists():
            info.update(
                {
                    "upload_path": str(compressed_path),
                    "upload_bytes": compressed_path.stat().st_size,
                    "compressed": True,
                }
            )
            return compressed_path, info
    except Exception as e:
        info["reason"] = f"compression_failed: {e}"
        print(f"[WARN] 封面圖壓縮失敗，將嘗試上傳原圖：{e}")
    return image_path, info


def update_video_languages_and_localization(youtube, video_id, title, description, tags):
    """
    補強設定：
    1. Video Language -> snippet.defaultAudioLanguage
    2. Title and Description Language -> snippet.defaultLanguage
    3. localizations['zh-TW']
    """
    try:
        current = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        items = current.get("items", [])
        if not items:
            print(f"⚠️ 找不到影片，無法補強語言設定：{video_id}")
            return False

        snippet = items[0].get("snippet", {})
        category_id = snippet.get("categoryId", "27")

        body = {
            "id": video_id,
            "snippet": {
                "categoryId": category_id,
                "title": title,
                "description": description,
                "tags": tags or [],
                "defaultLanguage": DEFAULT_LANGUAGE,
                "defaultAudioLanguage": DEFAULT_AUDIO_LANGUAGE,
            },
            "localizations": {
                DEFAULT_LANGUAGE: {
                    "title": title,
                    "description": description
                }
            }
        }

        youtube.videos().update(
            part="snippet,localizations",
            body=body
        ).execute()

        print("✅ 已更新影片語言設定（zh-TW）與本地化資訊。")
        return True
    except Exception as e:
        print(f"❌ 更新影片語言設定失敗: {e}")
        return False


def save_upload_record(target_path, record_data):
    """儲存執行結果 JSON 檔"""
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=4)
        print(f"💾 上傳紀錄已儲存至: {target_path.name}")
    except Exception as e:
        print(f"❌ 儲存紀錄檔失敗: {e}")


def write_success_marker(output_dir: Path):
    try:
        (output_dir / "upload.ok").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ 寫入 upload.ok 失敗: {e}")


def upload_episode(
    youtube,
    ep_num,
    privacy_status="private",
    publish_at=None,
    playlist_id=None,
    cover_name="cover.png"
):
    print(f"\n🚀 [YouTube Upload] 正在處理第 {ep_num:02d} 集...")

    target_folder = find_episode_folder(ep_num)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num:02d} 集的資料夾。")
        return

    output_dir = target_folder / "05_output"
    video_path = output_dir / "final_video.mp4"

    meta_candidates = [
        output_dir / "youtube_meta.json",
        target_folder / "youtube_meta.json",
    ]

    srt_candidates = [
        output_dir / "notebooklm_audio_fixed.srt",
        target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt",
    ]

    cover_candidates = [
        target_folder / "04_images" / cover_name,
        target_folder / "05_output" / cover_name,
    ]

    ab_cover_candidates = [
        target_folder / "04_images" / "cover.png",
        target_folder / "04_images" / "cover_cute.png",
        target_folder / "05_output" / "cover.png",
        target_folder / "05_output" / "cover_cute.png",
    ]

    meta_path = next((p for p in meta_candidates if p.exists()), meta_candidates[0])
    srt_path = next((p for p in srt_candidates if p.exists()), srt_candidates[0])
    selected_cover_path = next((p for p in cover_candidates if p.exists()), cover_candidates[0])
    cover_path, thumbnail_info = prepare_thumbnail_for_upload(selected_cover_path, output_dir)

    if not video_path.exists():
        print("❌ 缺少 final_video.mp4，跳過。")
        return

    if not meta_path.exists():
        print("❌ 缺少 youtube_meta.json，跳過。")
        print("ℹ️ 上傳流程不會自動產生 Metadata。請先在 Studio & Publisher 執行「產生 / 重產 Metadata」。")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    record = {
        "episode": ep_num,
        "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "starting",
        "video_id": None,
        "title": meta_data.get("title"),
        "privacy": privacy_status,
        "publish_at": publish_at,
        "playlist_id": playlist_id,
        "subtitle_uploaded": False,
        "playlist_added": False,
        "thumbnail_uploaded": False,
        "cover_name": cover_name,
        "thumbnail_info": thumbnail_info,
        "default_language": DEFAULT_LANGUAGE,
        "default_audio_language": DEFAULT_AUDIO_LANGUAGE,
        "ab_test_candidates_found": [str(p) for p in ab_cover_candidates if p.exists()],
    }

    formatted_publish_at = None
    if publish_at:
        try:
            dt = datetime.strptime(publish_at, "%Y-%m-%d %H:%M")
            formatted_publish_at = dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
            privacy_status = "private"
        except ValueError:
            print(f"❌ 時間格式錯誤: {publish_at}")
            return

    body = {
        "snippet": {
            "title": meta_data.get("title", f"會考英文 Ep.{ep_num:02d}"),
            "description": meta_data.get("description", ""),
            "tags": meta_data.get("tags", []),
            "categoryId": "27",
            "defaultLanguage": DEFAULT_LANGUAGE,
            "defaultAudioLanguage": DEFAULT_AUDIO_LANGUAGE,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False
        }
    }

    if formatted_publish_at:
        body["status"]["publishAt"] = formatted_publish_at

    try:
        print("⏳ 正在上傳影片...")
        media_body = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media_body
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"   進度: {int(status.progress() * 100)}%")

        video_id = response["id"]
        record["video_id"] = video_id
        print(f"✅ 上傳成功！ID: {video_id}")

        # 補強語言與本地化設定
        update_video_languages_and_localization(
            youtube=youtube,
            video_id=video_id,
            title=meta_data.get("title", f"會考英文 Ep.{ep_num:02d}"),
            description=meta_data.get("description", ""),
            tags=meta_data.get("tags", []),
        )

        # 正式封面
        record["thumbnail_uploaded"] = upload_thumbnail(youtube, video_id, cover_path)

        # 字幕
        record["subtitle_uploaded"] = upload_subtitle(youtube, video_id, srt_path)

        # 播放清單
        if playlist_id:
            record["playlist_added"] = add_video_to_playlist(youtube, video_id, playlist_id)

        # A/B Test 提醒
        if record["ab_test_candidates_found"]:
            print("ℹ️ 已找到封面候選：")
            for item in record["ab_test_candidates_found"]:
                print(f"   - {item}")
            print("ℹ️ 提醒：YouTube Data API 目前只能設定 1 張正式封面。")
            print("ℹ️ 若要做 cover.png / cover_cute.png 的 A/B Test，請到 YouTube Studio 手動使用 Test & Compare。")

        record["status"] = "success"
        save_upload_record(output_dir / "upload_record.json", record)
        write_success_marker(output_dir)

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        record["status"] = f"failed: {str(e)}"
        save_upload_record(output_dir / "upload_record.json", record)


def main():
    parser = argparse.ArgumentParser(description="YouTube 自動上傳（含字幕、封面、語言設定、播放清單）")
    parser.add_argument("--ep", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--privacy", type=str, choices=["public", "private", "unlisted"])
    parser.add_argument("--publish_at", type=str)
    parser.add_argument("--playlist_id", type=str, help="指定要加入的播放清單 ID")
    parser.add_argument("--cover_name", type=str, default=None, help="正式上傳的封面檔名；若未指定，優先讀取已儲存的發布設定")

    args = parser.parse_args()
    schedule_data = load_upload_schedule()

    if args.ep is not None:
        start_ep, end_ep = args.ep, args.ep
    elif args.start is not None:
        start_ep = args.start
        end_ep = args.end if args.end else args.start
    else:
        print("請指定 --ep 或 --start")
        return

    youtube = get_authenticated_service()

    for ep in range(start_ep, end_ep + 1):
        resolved = resolve_episode_upload_options(ep, args, schedule_data)

        upload_episode(
            youtube=youtube,
            ep_num=ep,
            privacy_status=resolved["privacy"],
            publish_at=resolved["publish_at"],
            playlist_id=resolved["playlist_id"],
            cover_name=resolved["cover_name"],
        )


if __name__ == "__main__":
    main()
