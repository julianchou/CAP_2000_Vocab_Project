import os
import srt
import json
import csv
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

# --- 目錄設定 ---
base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
MODEL_NAME = "gemini-2.5-pro"

def generate_storyboard(ep_num):
    # 尋找對應的集數資料夾
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集的資料夾。")
        return

     # 路徑定義
    srt_folder = target_folder / "02_subtitles"
    storyboard_folder = target_folder / "03_storyboards"
    images_folder = target_folder / "04_images"
     # 尋找字幕檔 (假設為資料夾內第一個 srt)
    srt_file = next(srt_folder.glob("*.srt"), None)
    vocab_csv = storyboard_folder / "vocab_data.csv"
    output_csv = storyboard_folder / "storyboard.csv"

    if not srt_file:
        print(f"❌ 找不到字幕檔於 {srt_folder}")
        return
    if not vocab_csv.exists():
        print(f"❌ 找不到單字表：{vocab_csv}，請先準備好 vocab_data.csv")
        return

    print("📖 正在讀取單字表與字幕檔...")
    # 讀取單字清單，讓 LLM 知道有哪些圖卡可用
    df_vocab = pd.read_csv(vocab_csv)
    word_list = df_vocab['Word'].str.strip().tolist()

    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_text = f.read()
        subs = list(srt.parse(srt_text))

    prompt = f"""
    你是一位專業的英語教學影片分鏡導演。以下是一份英語教學 Podcast 的 SRT 字幕檔。
    你的首要任務是：**精準捕捉每一個「本集重點單字」出現的片段，並將畫面切換為該單字的閃卡 (FLASHCARD)**。

    【本集重點單字清單】（請反覆核對字幕，絕不能漏掉任何一個字！）：
    {word_list}

    【分鏡規劃嚴格規則】：
    1. 🎯 【最高優先級：閃卡切換 (FLASHCARD)】
        - 請逐行檢查字幕。只要主持人在對話中提到、拼寫、解釋或舉例「本集重點單字清單」中的任何一個字，**必須**將該段落獨立切分出一個分鏡！
        - 將 `source_type` 設為 "FLASHCARD"，並在 `flashcard_word` 精準填寫該單字（必須與清單拼法完全一致）。
        - 閃卡分鏡的時間長度由「該單字講解的起訖時間」決定，**不受 10~30 秒的限制**。此時 `image_prompt` 留空。

    2. 🎨 【串場與故事畫面 (AI)】
        - 當主持人是在開場、閒聊、講述串場故事、結語，且**沒有**在講解重點單字時，才將 `source_type` 設為 "AI"。
        - **AI 分鏡**的長度請盡量控制在 10 秒到 30 秒之間。如果故事過長請拆分多個分鏡，過短請與上下文合併。
        - 必須撰寫豐富的純英文 `image_prompt` (描述畫面情境，結尾需固定加上 ", aspect ratio 16:9, cinematic wide shot")。

    3. ⏱️ 【時間連續性】
        - `start_time` 和 `end_time` 必須連貫（例如上一個 end_time 是 15，下一個 start_time 必須也是 15）。
        - 中間不能有空隙，也不能重疊，並覆蓋從 0 秒到影片結束的全部時間。

    輸出格式】：只輸出純 JSON 陣列，不要包含 Markdown 標記 (如 ```json)。請確保 JSON 格式合法。
    格式範例：
    [ 
        {{"start_time": 0, "end_time": 15, "source_type": "AI", "image_prompt": "two podcasters talking in a studio, aspect ratio 16:9, cinematic wide shot", "flashcard_word": "", "reason": "開場閒聊"}},
        {{"start_time": 15, "end_time": 45, "source_type": "FLASHCARD", "image_prompt": "", "flashcard_word": "abandon", "reason": "主持人正在拼寫並解釋單字 abandon"}} 
    ]

    字幕：
    {srt_text}
    """

    print(f"🧠 正在呼叫 {MODEL_NAME} 進行混合分鏡規劃...")
    
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        cleaned_json_str = response.text.replace("```json", "").replace("```", "").strip()
        storyboard_data = json.loads(cleaned_json_str)

        for i, scene in enumerate(storyboard_data):
            start_t = scene['start_time']
            end_t = scene['end_time']

            # 整理對應的字幕內容
            scene_details = []
            for s in subs:
                s_start = s.start.total_seconds()
                if s_start >= start_t and s_start < end_t:
                    t_start = str(s.start).split(',')[0]
                    t_end = str(s.end).split(',')[0]
                    clean_content = s.content.replace('\n', ' ')
                    scene_details.append(f"[{t_start}-->{t_end}] {clean_content}")

            scene['scene_id'] = i + 1
            scene['subtitle_reference'] = " | ".join(scene_details)

            # 處理圖卡路徑
            if scene.get('source_type') == "FLASHCARD" and scene.get('flashcard_word'):
                safe_word = scene['flashcard_word'].replace(' ', '_').replace('?', '')
                # 建立絕對路徑，確保後續 FFmpeg 讀取不出錯
                flashcard_path = images_folder / "flashcards" / f"{safe_word}.png"
                scene['custom_image_path'] = str(flashcard_path.resolve()).replace('\\', '/')
            else:
                scene['custom_image_path'] = ""
                scene['source_type'] = "AI" # 防呆機制，確保非圖卡就是 AI

        # 寫入 CSV
        fieldnames = ['scene_id', 'start_time', 'end_time', 'source_type', 'flashcard_word', 'custom_image_path', 'image_prompt', 'reason', 'subtitle_reference']
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for scene in storyboard_data:
                writer.writerow(scene)

        print(f"✅ 混合分鏡表已生成：{output_csv}")

    except Exception as e:
        print(f"❌ 錯誤：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="要處理的集數")
    args = parser.parse_args()
    generate_storyboard(args.ep)
