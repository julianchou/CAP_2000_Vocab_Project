import os
import pandas as pd
import argparse
from pathlib import Path

# 設定路徑
base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

def generate_video_description(ep_num):
    # 尋找資料夾
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集資料夾")
        return

    # 1. 讀取分鏡表以產生時間軸 (Timestamps)
    storyboard_path = target_folder / "03_storyboards" / "storyboard.csv"
    timestamps = []
    if storyboard_path.exists():
        df_sb = pd.read_csv(storyboard_path)
        # 篩選出單字切換的時間點
        for _, row in df_sb.iterrows():
            if "flashcards" in str(row['image_path']):
                word_name = Path(row['image_path']).stem
                # 轉換格式如 00:05
                time_str = f"{int(row['start']//60):02d}:{int(row['start']%60):02d}"
                timestamps.append(f"{time_str} 單字記憶：{word_name}")
    
    # 2. 組合說明欄內容
    ep_title = f"國中會考 2000 單字重點記憶 Ep{ep_num:02d}"
    description = [
        f"🔥 {ep_title}",
        "本系列影片幫助你利用碎片時間，高效記憶國中會考核心 2000 單字。",
        "",
        "📌 【本集重點章節】",
        "00:00 課程開始"
    ]
    description.extend(timestamps)
    description.extend([
        "",
        "📌 【標籤】",
        "#國中會考 #2000單字 #英文學習 #會考英文 #Vocabulary"  # 已移除 #NotebookLM
    ])

    # 3. 儲存至 05_output
    output_path = target_folder / "05_output" / "video_description.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(description))
    
    print(f"✅ 第 {ep_num:02d} 集 SEO 說明欄已產出於: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()
    generate_video_description(args.ep)
