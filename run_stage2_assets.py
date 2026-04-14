import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="階段 2：素材生成自動化主控")
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
        print(f"\n🚀 --- 開始處理第 {ep:02d} 集素材生成 ---")
        
        # 1. 產生單字內容 (JSON/CSV)
        subprocess.run([sys.executable, str(scripts_dir / "generate_vocab_content.py"), "--start", str(ep)])
        
        # 2. 產生單字圖卡 (flashcards)
        subprocess.run([sys.executable, str(scripts_dir / "generate_flashcards.py"), "--start", str(ep)])
        
        # 3. 產生封面圖
        subprocess.run([sys.executable, str(scripts_dir / "generate_cover.py"), "--ep", str(ep)])
        
        # 4. 產生 NotebookLM 語音摘要 Prompt
        subprocess.run([sys.executable, str(scripts_dir / "generate_notebooklm_prompt.py"), "--ep", str(ep)])

    print(f"\n✅ 階段 2 完成：第 {start_ep} 至 {end_ep} 集素材已備妥。")

if __name__ == "__main__":
    main()