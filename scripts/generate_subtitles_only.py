import whisper
import srt
import datetime
import os
import argparse
from pathlib import Path

# --- 設定區 ---
MODEL_SIZE = "medium"
SOFT_LIMIT = 18 
PUNCTUATIONS = ['，', '。', '！', '？', '、', ',', '.', '!', '?']

# 專案路徑設定
base_dir = Path(__file__).parent.parent
workspace_dir = base_dir / "workspace"

def generate_optimized_subtitles(ep_num):
    # 尋找對應的集數資料夾
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集的資料夾。")
        return

    audio_path = target_folder / "01_audio" / "notebooklm_audio.m4a"
    output_srt = target_folder / "02_subtitles" / "notebooklm_audio.srt"

    if not audio_path.exists():
        print(f"❌ 錯誤：找不到音訊檔 {audio_path}")
        return

    print(f"🚀 正在載入 Whisper 模型...")
    model = whisper.load_model(MODEL_SIZE)
    
    print(f"📝 正在辨識第 {ep_num} 集語音並獲取字級時間戳記...")
    result = model.transcribe(
        str(audio_path), 
        verbose=False, 
        initial_prompt="以下是繁體中文的對話內容，請包含正確的標點符號。",
        word_timestamps=True 
    )
    
    subs = []
    sub_index = 1
    
    for segment in result['segments']:
        words = segment.get('words', [])
        if not words: continue
        current_text = ""; start_time = None
        
        for i, word_info in enumerate(words):
            if start_time is None: start_time = word_info['start']
            clean_word = word_info['word'].strip()
            current_text += clean_word
            
            has_punctuation = any(p in clean_word for p in PUNCTUATIONS)
            if has_punctuation or len(current_text) >= SOFT_LIMIT or i == len(words) - 1:
                end_time = word_info['end']
                subs.append(srt.Subtitle(
                    index=sub_index,
                    start=datetime.timedelta(seconds=start_time),
                    end=datetime.timedelta(seconds=end_time),
                    content=current_text
                ))
                sub_index += 1
                current_text = ""; start_time = None 

    # 確保輸出目錄存在並寫入
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    print(f"✅ 字幕產生完成：{output_srt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="處理集數")
    args = parser.parse_args()
    generate_optimized_subtitles(args.ep)