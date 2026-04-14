import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command):
    """執行命令並即時輸出終端機日誌"""
    print(f"\n▶ 執行指令: {' '.join(command)}")
    try:
        # check=True 確保如果子程序回傳非 0 代碼會拋出例外
        subprocess.run(command, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 執行失敗，錯誤碼: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 使用者強制中斷執行")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="[Stage 7] 影片發布總管 (Meta 生成 + Drive & YouTube 上傳)")
    parser.add_argument("--ep", type=int, help="處理單一集數 (例如: --ep 3)")
    parser.add_argument("--start", type=int, help="起始集數 (例如: --start 1)")
    parser.add_argument("--end", type=int, help="結束集數 (例如: --end 5)")
    parser.add_argument("--privacy", type=str, choices=["public", "private", "unlisted"], help="影片隱私設定")
    parser.add_argument("--publish_at", type=str, help="排程發布時間 (格式: YYYY-MM-DD HH:MM)")
    parser.add_argument("--playlist_id", type=str, help="指定要加入的 YouTube 播放清單 ID")

    args = parser.parse_args()

    # 1. 確定要處理的集數範圍
    if args.ep is not None:
        episodes = [args.ep]
    elif args.start is not None:
        end_ep = args.end if args.end else args.start
        episodes = list(range(args.start, end_ep + 1))
    else:
        print("❌ 請指定 --ep 或 --start 參數來決定要處理的集數")
        sys.exit(1)

    # 2. 定位腳本路徑
    base_dir = Path(__file__).parent
    # 加上 "scripts" 子目錄
    meta_script = base_dir / "scripts" / "generate_youtube_meta.py"
    upload_script = base_dir / "scripts" / "upload_to_youtube.py"

    # 檢查子腳本是否存在
    if not meta_script.exists() or not upload_script.exists():
        print(f"❌ 找不到必要的腳本檔案。請確保 {meta_script.name} 與 {upload_script.name} 位在 scripts 目錄下。")
        sys.exit(1)

    # 檢查子腳本是否存在
    if not meta_script.exists() or not upload_script.exists():
        print(f"❌ 找不到必要的腳本檔案。請確保 {meta_script.name} 與 {upload_script.name} 與本腳本在同一目錄下。")
        sys.exit(1)

    print(f"🚀 開始執行 YouTube 完整工作流，共計 {len(episodes)} 集準備處理...")

    # 3. 開始依序處理每一集
    for ep in episodes:
        print(f"\n{'='*50}")
        print(f"🎬 正在處理第 {ep:02d} 集")
        print(f"{'='*50}")

        # --- [階段 1] 產生 YouTube Metadata 與上傳單字表 ---
        print("\n[階段 1] 產生 YouTube Metadata 與上傳單字表")
        meta_cmd = [sys.executable, str(meta_script), "--ep", str(ep)]
        
        if not run_command(meta_cmd):
            print(f"⚠️ 第 {ep:02d} 集 Metadata 處理失敗，跳過上傳階段。")
            continue

        # --- [階段 2] 上傳影片至 YouTube ---
        print("\n[階段 2] 上傳影片、封面並設定清單")
        upload_cmd = [sys.executable, str(upload_script), "--ep", str(ep)]
        
        # 將總管收到的上傳相關參數，傳遞給 upload_to_youtube.py
        if args.privacy:
            upload_cmd.extend(["--privacy", args.privacy])
        if args.publish_at:
            upload_cmd.extend(["--publish_at", args.publish_at])
        if args.playlist_id:
            upload_cmd.extend(["--playlist_id", args.playlist_id])

        if not run_command(upload_cmd):
            print(f"⚠️ 第 {ep:02d} 集上傳失敗。")
            continue
        
    print("\n🎉 所有指定集數的 YouTube 工作流處理完畢！")

if __name__ == "__main__":
    main()