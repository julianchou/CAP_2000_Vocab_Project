import asyncio
import json
import argparse
import os
import time
import re
import random
from pathlib import Path
from edge_tts import Communicate
from pydub import AudioSegment

# --- 參數優化：移除容易報錯的 pitch，僅調整語速 ---
SPEAKER_CONFIG = {
    "John": {
        "voice": "zh-TW-YunJheNeural", 
        "rate": "+12%"  # John 講話較快
    },
    "Mary": {
        "voice": "zh-TW-HsiaoChenNeural", 
        "rate": "+3%"   # Mary 講話較穩
    }
}

base_dir = Path(__file__).parent.parent
workspace_dir = base_dir / "workspace"

async def synthesize_audio(ep_num):
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集資料夾。")
        return

    script_path = target_folder / "01_audio" / "podcast_script.json"
    temp_dir = target_folder / "01_audio" / "temp_segments"
    temp_dir.mkdir(exist_ok=True)

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    combined = AudioSegment.empty()
    print("🎙️ 正在以『安全節奏』模式生成語音...")

    for i, line in enumerate(script):
        raw_text = line['text'].strip()
        # 移除所有括號內容
        text = re.sub(r'[\(\[].*?[\)\]]', '', raw_text).strip()
        if not text: continue

        speaker = line['speaker']
        # 若 speaker 不在清單中，預設使用 Mary 的聲音
        config = SPEAKER_CONFIG.get(speaker, SPEAKER_CONFIG["Mary"])
        file_path = temp_dir / f"{i:03d}_{speaker}.mp3"
        
        try:
            # 簡化參數調用，避免 NoAudioReceived 錯誤
            communicate = Communicate(text, config["voice"], rate=config["rate"])
            await communicate.save(str(file_path))
        except Exception as e:
            print(f"⚠️ 片段 {i} 合成失敗: {e}，嘗試以預設參數重新合成...")
            communicate = Communicate(text, config["voice"])
            await communicate.save(str(file_path))

        # 確保檔案寫入完整
        for _ in range(15):
            if file_path.exists() and file_path.stat().st_size > 0: break
            time.sleep(0.1)

        if not file_path.exists() or file_path.stat().st_size == 0:
            print(f"❌ 片段 {i} 無法產生音訊檔案，跳過。")
            continue

        seg = AudioSegment.from_mp3(str(file_path))
        
        # 執行拼接
        if len(combined) > 100 and len(seg) > 100:
            combined = combined.append(seg, crossfade=100)
        else:
            combined = combined + seg

        # 動態停頓
        last_char = text[-1] if text else ""
        if last_char in ["？", "!", "！"]:
            pause_ms = 700 
        elif last_char in ["。", ".", "…"]:
            pause_ms = 500
        else:
            pause_ms = 300 + random.randint(0, 100)

        combined = combined + AudioSegment.silent(duration=pause_ms)

    output_path = target_folder / "01_audio" / "notebooklm_audio.m4a"
    combined.export(output_path, format="ipod")
    print(f"✅ 語音產出成功：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(synthesize_audio(args.ep))