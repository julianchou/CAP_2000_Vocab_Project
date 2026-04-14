import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="階段 4：AI 字幕轉錄與校對主控")
    parser.add_argument("--ep", type=int, help="處理單一集數")
    parser.add_argument("--start", type=int, help="起始集數 (用於連續處理)")
    parser.add_argument("--end", type=int, help="結束集數 (用於連續處理)")
    args = parser.parse_args()

    # 判斷是單集還是連續集數
    if args.ep is not None:
        start_ep = args.ep
        end_ep = args.ep
    elif args.start is not None:
        start_ep = args.start
        end_ep = args.end if args.end else args.start
    else:
        parser.error("請提供 --ep (單集處理) 或 --start 與 --end (連續處理)")

    scripts_dir = Path(__file__).parent / "scripts"

    for ep in range(start_ep, end_ep + 1):
        print(f"\n🧠 --- 開始第 {ep:02d} 集 AI 處理流程 ---")
        
        try:
            # 1. 執行轉錄 (產生 notebooklm_audio.srt)
            print(f"🎙️ 步驟 1: 執行語音轉錄 (Whisper)...")
            subprocess.run([sys.executable, str(scripts_dir / "generate_subtitles_only.py"), "--ep", str(ep)], check=True)
            
            # 2. 執行 LLM 字幕校對
            print(f"🎬 步驟 2: 執行 AI 字幕校對 (Gemini)...")
            subprocess.run([sys.executable, str(scripts_dir / "llm_subtitle_reviewer.py"), "--ep", str(ep)], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 第 {ep:02d} 集處理失敗，停止後續動作。錯誤訊息: {e}")
            continue 

    print(f"\n🎉 階段 4 完成：第 {start_ep} 至 {end_ep} 集批次處理結束。")

if __name__ == "__main__":
    main()