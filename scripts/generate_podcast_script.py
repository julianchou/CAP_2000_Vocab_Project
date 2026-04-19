import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client()

base_dir = Path(__file__).resolve().parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
MODEL_NAME = "gemini-2.5-pro"


def generate_json_script(ep_num: int):
    target_folder, start_num, end_num = resolve_episode_range(workspace_dir, ep_num)
    vocab_csv = target_folder / "03_storyboards" / "vocab_data.csv"
    if not vocab_csv.exists():
        print(f"❌ 找不到單字表：{vocab_csv}")
        return

    df = pd.read_csv(vocab_csv)
    vocab_count = len(df)
    vocab_data = df.to_json(orient="records", force_ascii=False)

    prompt = f"""
你是一位金鐘獎等級的 Podcast 編劇，為《會考英文隨身聽：2000 單攻略站》撰寫深度劇本。

【角色設定】
- John：陽光活力男聲，負責提問、分享生活故事、帶動氣氛。
- Mary：溫柔專業女老師，負責解析、分享記憶技巧、朗讀例句。

【核心互動與教學準則】
1. 熱情開場：兩人具名互打招呼，並報出「第 {ep_num} 集」。
2. 內化專業知識：嚴禁說出「根據資料」、「清單顯示」或「編號幾到幾」。
3. 完整覆蓋指令：你必須確保下方【本集單字數據】中的所有 {vocab_count} 個單字，都依照順序自然出現在對話中。
4. 教學深度：提到單字時，Mary 必須解析詞型與含義，並朗讀實用例句。John 則用生活故事來接話。
5. 節奏控制：每段對話約包含 2-3 個單字，隨後進行 3-5 句的深度聊天與記憶法分享，再進入下一組單字。
6. 溫馨結尾：確認所有單字講完後，兩人一起為考生加油並預告下一集。
7. 本集單字範圍是 {start_num}-{end_num}，但不要把數字範圍直接講出來。

【輸出格式要求】
- 只輸出純 JSON 陣列，欄位為 speaker 與 text。
- 嚴禁括號舞台指示。
- 確保 JSON 結構完整，以 [ 開頭、以 ] 結尾。

【本集單字數據】
{vocab_data}

【輸出範例】
[
  {{"speaker": "John", "text": "哈囉大家，我是 John！今天我們繼續用聊天方式背單字。"}},
  {{"speaker": "Mary", "text": "大家好，我是 Mary！今天也會把這一集的核心單字一次帶你記熟。"}}
]
"""

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    script_content = response.text.replace("```json", "").replace("```", "").strip()

    output_path = target_folder / "01_audio" / "podcast_script.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_content, encoding="utf-8")
    print(f"✅ 劇本已生成：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()
    generate_json_script(args.ep)
