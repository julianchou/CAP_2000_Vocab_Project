import os
import json
import argparse
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version': 'v1'})
MODEL_ID = "gemini-2.5-flash"

def expand_batch_with_ai(word_list):
    words_str = ", ".join(word_list)
    prompt = f"""
    任務：你是資深英文老師，請為以下單字清單產生教學內容。
    清單：{words_str}

    請輸出 JSON 陣列，每個物件包含 Word, POS, Meaning, English_Sentence, Chinese_Translation。

    ⚠️ 欄位嚴格要求：
    1. POS (詞性)：必須包含「縮寫」與「中文全名」，例如 "n. 名詞"、"v. 動詞"、"adj. 形容詞"、"adv. 副詞"。
    2. Meaning (中文翻譯)：該單字最常用的繁體中文解釋。
    3. English_Sentence (英文例句)：請產生一個生活化、自然的句子，長度控制在 12 個單字以內，不要生硬。
    4. Chinese_Translation (中文例句翻譯)：繁體中文翻譯。

    請直接輸出 JSON 陣列，不要包含 Markdown 標籤。
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ 批次生成失敗: {e}")
        return []

def process_episode(ep_num):
    base_dir = Path(__file__).parent.parent
    index_path = base_dir / "core/assets/vocab_index.csv"
    if not index_path.exists():
        print("❌ 找不到索引檔"); return

    idx_df = pd.read_csv(index_path)
    start_idx, end_idx = (ep_num - 1) * 30 + 1, ep_num * 30
    target_words = idx_df[(idx_df['Index'] >= start_idx) & (idx_df['Index'] <= end_idx)]['Word'].tolist()

    print(f"🎬 [Episode {ep_num:02d}] 正在產生完整內容...")
    
    full_data = []
    batch_size = 10
    for i in range(0, len(target_words), batch_size):
        batch = target_words[i : i + batch_size]
        print(f"   ⏳ 處理中 {i+1} ~ {i+len(batch)}...")
        batch_result = expand_batch_with_ai(batch)
        full_data.extend(batch_result)
        time.sleep(1)

    if full_data:
        target_folder = next((base_dir / "workspace").glob(f"Ep{ep_num:02d}_*"), None)
        if target_folder:
            out_dir = target_folder / "03_storyboards"
            out_dir.mkdir(parents=True, exist_ok=True)
            # 確保儲存為 UTF-8 with BOM 以防 Excel 開啟亂碼
            pd.DataFrame(full_data).to_csv(out_dir / "vocab_data.csv", index=False, encoding='utf-8-sig')
            print(f"✅ 資料更新成功！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    args = parser.parse_args()
    process_episode(args.start)