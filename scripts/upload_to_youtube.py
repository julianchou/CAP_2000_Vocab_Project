import os
import csv
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# 1. ⚡ 權限範圍：包含 YouTube 上傳、播放清單與字幕權限
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl'] 

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
CLIENT_SECRETS_FILE = base_dir / "client_secret.json"
TOKEN_FILE = base_dir / "token_youtube.pickle"
SCHEDULE_FILE = base_dir / "upload_schedule.csv"

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

def get_authenticated_service():
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
                    "resourceId": {"kind": "youtube#video", "videoId": video_id}
                }
            }
        ).execute()
        print(f"✅ 成功加入播放清單！")
        return True
    except Exception as e:
        print(f"❌ 無法加入播放清單: {e}")
        return False

def upload_subtitle(youtube, video_id, srt_path):
    """上傳字幕檔"""
    if not srt_path.exists():
        print(f"⚠️ 找不到字幕檔: {srt_path.name}，跳過。")
        return False
    print(f"🗨️ 正在上傳繁體中文字幕...")
    try:
        youtube.captions().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "language": "zh-TW",
                    "name": "繁體中文",
                    "isDraft": False
                }
            },
            media_body=MediaFileUpload(str(srt_path))
        ).execute()
        print(f"✅ 字幕上傳成功！")
        return True
    except Exception as e:
        print(f"❌ 字幕上傳失敗: {e}")
        return False

def save_upload_record(target_path, record_data):
    """儲存執行結果 JSON 檔"""
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=4)
        print(f"💾 上傳紀錄已儲存至: {target_path.name}")
    except Exception as e:
        print(f"❌ 儲存紀錄檔失敗: {e}")

def upload_episode(youtube, ep_num, privacy_status="private", publish_at=None, playlist_id=None):
    print(f"\n🚀 [YouTube Upload] 正在處理第 {ep_num:02d} 集...")
    
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num:02d} 集的資料夾。")
        return

    output_dir = target_folder / "05_output"
    video_path = output_dir / "final_video.mp4"
    meta_path = output_dir / "youtube_meta.json"
    cover_path = target_folder / "04_images" / "cover.png"
    srt_path = output_dir / "notebooklm_audio_fixed.srt"

    if not video_path.exists() or not meta_path.exists():
        print(f"❌ 缺少影片或 Meta 檔，跳過。")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # 建立上傳紀錄字典
    record = {
        "episode": ep_num,
        "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "starting",
        "video_id": None,
        "title": meta_data.get('title'),
        "privacy": privacy_status,
        "publish_at": publish_at,
        "subtitle_uploaded": False,
        "playlist_added": False
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
        'snippet': {
            'title': meta_data.get('title', f'會考英文 Ep.{ep_num:02d}'),
            'description': meta_data.get('description', ''),
            'tags': meta_data.get('tags', []),
            'categoryId': '27'
        },
        'status': {
            'privacyStatus': privacy_status,
            'selfDeclaredMadeForKids': False
        }
    }
    if formatted_publish_at:
        body['status']['publishAt'] = formatted_publish_at

    try:
        # --- 執行影片上傳 ---
        print(f"⏳ 正在上傳影片...")
        media_body = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
        request = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media_body)
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"   進度: {int(status.progress() * 100)}%")

        video_id = response['id']
        record["video_id"] = video_id
        print(f"✅ 上傳成功！ID: {video_id}")

        # --- 設定封面圖 ---
        if cover_path.exists():
            youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(cover_path))).execute()
            print(f"🖼️ 封面圖設定完成！")

        # --- 上傳字幕 ---
        record["subtitle_uploaded"] = upload_subtitle(youtube, video_id, srt_path)

        # --- 加入播放清單 ---
        if playlist_id:
            record["playlist_added"] = add_video_to_playlist(youtube, video_id, playlist_id)

        # --- 完成紀錄 ---
        record["status"] = "success"
        save_upload_record(output_dir / "upload_record.json", record)

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        record["status"] = f"failed: {str(e)}"
        save_upload_record(output_dir / "upload_record.json", record)

def main():
    parser = argparse.ArgumentParser(description="YouTube 自動上傳 (含紀錄檔、字幕與播放清單)")
    parser.add_argument("--ep", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--privacy", type=str, choices=["public", "private", "unlisted"])
    parser.add_argument("--publish_at", type=str)
    parser.add_argument("--playlist_id", type=str, help="指定要加入的播放清單 ID")
    
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
        ep_privacy = args.privacy or schedule_data.get(ep, {}).get("privacy", "private")
        ep_publish = args.publish_at or schedule_data.get(ep, {}).get("publish_time")
        
        upload_episode(
            youtube, ep, 
            privacy_status=ep_privacy, 
            publish_at=ep_publish, 
            playlist_id=args.playlist_id
        )

if __name__ == "__main__":
    main()
