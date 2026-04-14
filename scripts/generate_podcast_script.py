import os
import json
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

# --- 設定區 ---
base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
MODEL_NAME = "gemini-2.5-pro"

def generate_json_script(ep_num):
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    vocab_csv = target_folder / "03_storyboards" / "vocab_data.csv"
    df = pd.read_csv(vocab_csv)
    
    # 組合單字參考資料
    vocab_data = df.to_json(orient='records', force_ascii=False)

    prompt = f"""
    你是一位金鐘獎等級的 Podcast 編劇，為『會考英文隨身聽：2000 單攻略站』撰寫深度劇本。
    
    【角色設定】
    - John：陽光活力男聲，負責提問、分享生活故事、帶動氣氛。
    - Mary：溫柔專業女老師，負責解析、分享記憶技巧、朗讀例句。

    【核心互動與教學準則】
    1. **熱情開場**：兩人具名互打招呼，並報出「第 {ep_num} 集」。
    2. **內化專業知識**：嚴禁說出「根據資料」、「清單顯示」或「編號幾到幾」。
    3. **🌟 完整覆蓋指令 (極度重要)**：你必須確保下方【本集單字數據】中的 **所有 30 個單字** 都依照順序出現在對話中。
    4. **教學深度**：提到單字時，Mary 必須解析詞型與含義，並朗讀實用例句。John 則用生活故事來接話。
    5. **節奏控制**：每段對話約包含 2-3 個單字，隨後進行 3-5 句的深度聊天與記憶法分享（如字根字首或相似字辨析），再進入下一組單字。
    6. **溫馨結尾**：確認所有單字講完後，兩人一起為考生加油並預告下一集。

    【輸出格式要求】
    - 只輸出純 JSON 陣列，欄位為 'speaker' 與 'text'。
    - 嚴禁括號舞台指示。確保 JSON 結構完整（以 [ 開頭，以 ] 結尾）。
    
    【本集單字數據】：
    {vocab_data}

    【輸出範例】：
    [
      {{"speaker": "John", "text": "哈囉大家，我是 John！今天我們要來聊聊 A 開頭的單字續集。"}},
      {{"speaker": "Mary", "text": "大家好，我是 Mary！準備好跟我們一起攻克這 30 個核心單字了嗎？"}}
    ]
    """

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    script_content = response.text.replace("```json", "").replace("```", "").strip()
    
    output_path = target_folder / "01_audio" / "podcast_script.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"✅ 劇本已生成：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()
    generate_json_script(args.ep)
