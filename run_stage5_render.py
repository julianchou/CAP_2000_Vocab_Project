import argparse
import subprocess
import sys
from pathlib import Path

def generate_preview(ep_num):
    print(f"\n👀 [Stage 5] 正在執行第 {ep_num:02d} 集的 AI 分鏡與畫面預覽...")
    
    scripts_dir = Path(__file__).parent / "scripts"
    
    try:
        # 1. 執行 AI 分鏡規劃 (Director)
        print(f"🧠 步驟 1: 執行 AI 分鏡規劃 (llm_director.py)...")
        subprocess.run([sys.executable, str(scripts_dir / "llm_director.py"), "--ep", str(ep_num)], check=True)

        # 2. 呼叫產生預覽檔
        print(f"🎞️ 步驟 2: 產生畫面與字幕預覽 (gen_matched_preview_v4.py)...")
        subprocess.run([sys.executable, str(scripts_dir / "gen_matched_preview_v4.py"), "--ep", str(ep_num)], check=True)
        
        print(f"✅ [Stage 5] 第 {ep_num:02d} 集預覽處理完成！")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ [Stage 5] 第 {ep_num:02d} 集預覽處理失敗。錯誤訊息: {e}")

def main():
    parser = argparse.ArgumentParser(description="Stage 5: 分鏡與預覽主控")
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

    for ep in range(start_ep, end_ep + 1):
        generate_preview(ep)

    print(f"\n🎉 階段 5 完成：第 {start_ep} 至 {end_ep} 集預覽產生完畢。")

if __name__ == "__main__":
    main()