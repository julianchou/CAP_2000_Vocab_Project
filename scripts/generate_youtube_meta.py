import os
import json
import argparse
import pickle
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Google Drive API 相關套件
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# 載入環境變數
load_dotenv()
client = genai.Client()

# --- 目錄與 Drive 設定 ---
base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

# 🔑 包含 YouTube 上傳與 Drive 檔案操作權限
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/youtube.upload'
]

# 你指定的雲端硬碟目錄 ID
TARGET_FOLDER_ID = "1hB1HYVOlOyyBhD0kk_wf9lnav63wF9Kc"

def get_drive_service():
    """獲取 Google Drive API 服務 (對接原本的 client_secret.json)"""
    creds = None
    token_path = base_dir / 'token.pickle'
    creds_path = base_dir / 'client_secret.json'

    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise FileNotFoundError(f"❌ 找不到憑證檔案：{creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def upload_to_drive(file_path, file_name):
    """將檔案上傳至指定目錄並開放『知道連結即可閱讀』權限"""
    try:
        service = get_drive_service()
        file_metadata = {
            'name': file_name,
            'parents': [TARGET_FOLDER_ID]
        }
        media = MediaFileUpload(str(file_path), mimetype='text/csv')
        
        # 1. 執行上傳
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = file.get('id')
        
        # 2. 設定權限：知道連結的人即可檢視 (修正為 'reader')
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'} 
        ).execute()

        # 3. 獲取分享連結
        res = service.files().get(fileId=file_id, fields='webViewLink').execute()
        return res.get('webViewLink')
    except Exception as e:
        print(f"🚨 Drive 上傳失敗: {e}")
        return None  # 👈 失敗時回傳 None

def generate_youtube_meta(ep_num):
    print(f"\n📝 [YouTube Meta] 正在為第 {ep_num:02d} 集生成 Metadata (使用 Gemini 2.5 Pro)...")
    
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num:02d} 集的資料夾。")
        return

    srt_path = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"
    vocab_csv_path = target_folder / "03_storyboards" / "vocab_data.csv"
    output_folder = target_folder / "05_output"
    meta_output_path = output_folder / "youtube_meta.json"

    if not srt_path.exists():
        print(f"❌ 找不到字幕檔 ({srt_path})，中止執行。")
        return

    # --- [Step 1] 上傳單字表到 Google Drive ---
    if not vocab_csv_path.exists():
        print(f"❌ 找不到單字表檔案 {vocab_csv_path}，中止執行。")
        return

    print(f"☁️  正在上傳單字表至指定資料夾...")
    drive_link = upload_to_drive(vocab_csv_path, f"會考英文隨身聽_Ep{ep_num:02d}_單字表.csv")

    # 🛑 檢查點：上傳失敗就不往下走
    if not drive_link:
        print(f"⛔ 因雲端硬碟上傳失敗，已停止後續 Gemini 分析流程。請檢查權限或網路。")
        return

    # --- [Step 2] 讀取字幕內容 ---
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    # --- [Step 3] Gemini 2.5 Pro 萃取分析 ---
    prompt = f"""
    你是一位專業的 YouTube SEO 專家與國中英文老師。
    請根據以下提供的「完整影片字幕檔 (SRT)」，為影片生成 YouTube Metadata。

    【任務要求】：
    1. 這是《會考英文隨身聽》的 2000 單字教學影片。
    2. 主持人會在對話中介紹許多英文單字。請你閱讀完整字幕，精準抓出這集教學了「哪些單字」，以及它們在影片中「出現的精準時間 (MM:SS)」。
    3. 必須將所有單字（通常約 30 個）全部列出，不可使用 '...' 等字眼。
    4. 【重點】請在「summary (摘要)」的結尾，用親切的語氣提醒觀眾：頻道固定在「每週日與每週三晚上 6 點」上傳新影片，邀請大家準時收聽。

    【字幕檔完整內容】：
    {srt_content}

    【輸出要求】：
    請嚴格以 JSON 格式輸出，必須包含：
    1. "summary": 約 100 字摘要（含上片時間提醒）。
    2. "tags": 15 個標籤。
    3. "words": 陣列，包含 "time", "word", "meaning"。
    """

    try:
        print(f"🧠 正在請求 Gemini 2.5 Pro 分析字幕內容...")
        response = client.models.generate_content(
            model='gemini-2.5-pro', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        ai_data = json.loads(response.text)

        # 🛠️ 組裝時間軸文字
        words_list = ai_data.get("words", [])
        timeline_text = "\n".join([f"{item.get('time', '00:00')} {item.get('word', '')}：{item.get('meaning', '')}" for item in words_list])

        # --- [Step 4] 組合 Description ---
        final_description = f"""📖 本集摘要：
{ai_data.get("summary", "")}

📺 更新時間：
每週日、三 晚上 6:00 定時更新，歡迎訂閱開啟小鈴鐺！

📥 本集單字表下載 (Google Drive)：
{drive_link}

🎧 建議聽法：
第一遍：沉浸理解 | 第二遍：影子練習 | 第三遍：直覺反射

⏳ 快速跳轉 (單字時間軸)：
00:00 頻道開場與本集重點
{timeline_text}

🔔 訂閱「會考英文隨身聽」
#國中會考 #英文單字 #2000單 #108課綱 #英文聽力 #隨身聽"""

        # 計算標題
        start_word = (ep_num - 1) * 30 + 1
        end_word = ep_num * 30
        fixed_title = f"考前必聽！國中英文 2000 單字攻略 ({start_word:03d}-{end_word:03d})｜對話式語音摘要，邊聽邊背最輕鬆"

        meta_data = {
            "title": fixed_title,
            "description": final_description,
            "tags": ai_data.get("tags", []),
            "drive_link": drive_link
        }

        output_folder.mkdir(parents=True, exist_ok=True)
        with open(meta_output_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 第 {ep_num:02d} 集 YouTube Metadata 與雲端連結處理完成！")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, help="處理單一集數")
    args = parser.parse_args()

    if args.ep:
        generate_youtube_meta(args.ep)
    else:
        print("請提供 --ep 參數，例如: --ep 3")

if __name__ == "__main__":
    main()
