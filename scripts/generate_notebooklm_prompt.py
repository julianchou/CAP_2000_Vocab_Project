import os
import argparse
import pandas as pd
from pathlib import Path

# --- 目錄設定 ---
base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

def generate_notebooklm_instruction(ep_num):
    # 1. 自動尋找對應的集數資料夾
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集的資料夾。")
        return

    vocab_csv = target_folder / "03_storyboards" / "vocab_data.csv"
    if not vocab_csv.exists():
        print(f"❌ 找不到單字表：{vocab_csv}")
        return

    # 2. 讀取單字資料
    df = pd.read_csv(vocab_csv)
    
    # 3. 計算單字編號區間 (每集 30 單)
    start_num = (ep_num - 1) * 30 + 1
    end_num = ep_num * 30

    # 4. 格式化單字與例句清單
    vocab_list_str = ""
    for _, row in df.iterrows():
        vocab_list_str += f"- 單字：{row['Word']} ({row['POS']}) {row['Meaning']}\n"
        vocab_list_str += f"  > 例句：{row['English_Sentence']}\n"
        vocab_list_str += f"  > 翻譯：{row['Chinese_Translation']}\n\n"

    # 5. 組合最終 Prompt
    full_prompt = f"""【節目設定】

節目名稱： 『會考英文隨身聽：2000 單攻略站』

集數： 第 {ep_num} 集

主持人：
Mary (女)： 聲音溫柔、專業，負責深入淺出的解析、詞型補充與朗讀例句。
John (男)： 聲音陽光、有活力，負責帶動氣氛與鼓勵聽眾。

【互動準則】

具名開場： 節目一開始，John 和 Mary 必須熱情地互打招呼（例如：「哈囉大家，我是 John！」、「大家好，我是 Mary！」），並報出頻道名稱與本集集數。

第一人稱對話： 必須以「我、我們」進行自然、口語化的聊天，兩人之間要有頻繁的互動（例如：John 提問、Mary 回答，或 Mary 分享記憶技巧、John 補充例句）。

嚴禁 AI 標籤： 絕對不能說出「根據提供的資料」、「從文件來看」、「這份清單顯示」或「編號 {start_num} 到 {end_num}」等機器化字眼。請將這些單字視為你們內化的專業知識。

內容核心： 輕鬆分享 {start_num} 到 {end_num} 號的國中會考核心單字。內容須包含：單字、詞型、中文含義及實用例句。

氛圍營造： 語氣要像是在陪伴考生通勤、運動或零碎時間學習，讓聽眾感到扎實、權威但沒有壓力。

暖心結尾： 兩位主持人最後要一起為考生加油，並提醒聽眾持續鎖定後續集數。
- 多分享單字聯想或字根字首的記憶法。
- 幫助聽眾更輕鬆地記住相似拼寫的單字。

【本集重點單字與例句參考（請自然地融入對話，作為你們的教學素材）】
{vocab_list_str}"""

    # 6. 輸出至檔案
    output_path = target_folder / f"notebooklm_prompt_ep{ep_num:02d}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_prompt)

    print(f"✅ 第 {ep_num} 集的 NotebookLM 指令已生成！")
    print(f"📂 檔案路徑：{output_path}")
    print(f"👉 請開啟檔案並「全選、複製」內容到 NotebookLM。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="處理集數")
    args = parser.parse_args()
    generate_notebooklm_instruction(args.ep)
