import argparse
import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client()

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/youtube.upload",
]
TARGET_FOLDER_ID = "1hB1HYVOlOyyBhD0kk_wf9lnav63wF9Kc"

DEFAULT_PROMPT_TEMPLATE = """你是一位專業的 YouTube SEO 專家與國中英文老師。
請根據以下提供的完整影片字幕檔（SRT），為影片生成 YouTube Metadata。

【任務要求】
1. 這是《會考英文隨身聽》的 2000 單字教學影片。
2. 請閱讀完整字幕，精準抓出本集教學單字，以及它們在影片中出現的精準時間（MM:SS）。
3. 必須將本集 {{START_WORD}} 到 {{END_WORD}} 號範圍內的重點單字完整列出，不可使用省略符號。
4. 請在 summary 結尾，用親切語氣提醒觀眾：頻道固定在每週日與每週三晚上 6 點上傳新影片。

【字幕檔完整內容】
{{SRT_CONTENT}}

【輸出要求】
請嚴格輸出 JSON，必須包含：
1. summary：大約 100 字摘要。
2. tags：15 個標籤。
3. words：陣列，每個元素包含 time、word、meaning。
"""


def prompt_template_path_for(target_folder: Path) -> Path:
    return target_folder / "05_output" / "youtube_meta_prompt.txt"


def ensure_prompt_template(target_folder: Path) -> Path:
    prompt_path = prompt_template_path_for(target_folder)
    if not prompt_path.exists():
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def render_prompt(template_text: str, start_word: int, end_word: int, srt_content: str) -> str:
    return (
        template_text
        .replace("{{START_WORD}}", str(start_word))
        .replace("{{END_WORD}}", str(end_word))
        .replace("{{SRT_CONTENT}}", srt_content)
    )


def get_drive_service():
    creds = None
    token_path = base_dir / "token.pickle"
    creds_path = base_dir / "client_secret.json"

    if token_path.exists():
        with token_path.open("rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise FileNotFoundError(f"找不到憑證檔案：{creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with token_path.open("wb") as token:
            pickle.dump(creds, token)

    return build("drive", "v3", credentials=creds)


def upload_to_drive(file_path: Path, file_name: str):
    try:
        service = get_drive_service()
        file_metadata = {"name": file_name, "parents": [TARGET_FOLDER_ID]}
        media = MediaFileUpload(str(file_path), mimetype="text/csv")
        file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        file_id = file.get("id")

        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        res = service.files().get(fileId=file_id, fields="webViewLink").execute()
        return res.get("webViewLink")
    except Exception as e:
        print(f"Drive 上傳失敗：{e}")
        return None


def generate_youtube_meta(ep_num: int):
    print(f"\n[YouTube Meta] 正在為第 {ep_num:02d} 集生成 Metadata...")
    target_folder, start_word, end_word = resolve_episode_range(workspace_dir, ep_num)

    srt_path = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"
    vocab_csv_path = target_folder / "03_storyboards" / "vocab_data.csv"
    output_folder = target_folder / "05_output"
    meta_output_path = output_folder / "youtube_meta.json"

    if not srt_path.exists():
        print(f"找不到字幕檔：{srt_path}")
        return
    if not vocab_csv_path.exists():
        print(f"找不到單字表：{vocab_csv_path}")
        return

    print("正在上傳單字表到 Google Drive...")
    drive_link = upload_to_drive(vocab_csv_path, f"會考英文隨身聽_Ep{ep_num:02d}_單字表.csv")
    if not drive_link:
        print("因 Google Drive 上傳失敗，已停止後續 Metadata 生成。")
        return

    srt_content = srt_path.read_text(encoding="utf-8")
    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = render_prompt(prompt_template, start_word, end_word, srt_content)

    try:
        print("正在請求 Gemini 分析字幕內容...")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        ai_data = json.loads(response.text)
        words_list = ai_data.get("words", [])
        timeline_text = "\n".join(
            [f"{item.get('time', '00:00')} {item.get('word', '')}：{item.get('meaning', '')}" for item in words_list]
        )

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

        fixed_title = f"考前必聽！國中英文 2000 單字攻略 ({start_word:03d}-{end_word:03d})｜對話式語音摘要，邊聽邊背最輕鬆"
        meta_data = {
            "title": fixed_title,
            "description": final_description,
            "tags": ai_data.get("tags", []),
            "drive_link": drive_link,
        }

        output_folder.mkdir(parents=True, exist_ok=True)
        meta_output_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"YouTube Metadata 生成完成：{meta_output_path}")
    except Exception as e:
        print(f"發生錯誤：{e}")


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
