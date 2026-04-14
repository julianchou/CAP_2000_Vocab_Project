import os
import argparse
import subprocess
from pathlib import Path

def merge_audio_for_episode(ep_num, base_dir):
    print(f"\n🎬 正在處理第 {ep_num:02d} 集的音檔合併...")
    
    workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
    core_dir = base_dir / "core"
    
    # 開頭音檔固定路徑
    intro_path = core_dir / "assets" / "intro.mp3"
    
    # 利用 glob 模糊比對找出對應集數的資料夾 (例如 Ep05_0121_0150)
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    
    if not target_folder:
        print(f"❌ 找不到第 {ep_num:02d} 集的專案資料夾。")
        return

    # 組合主音檔路徑
    main_audio_path = target_folder / "01_audio" / "notebooklm_audio.m4a"
    temp_output_path = target_folder / "01_audio" / "temp_notebooklm_audio.m4a"

    # 檢查檔案是否存在
    if not intro_path.exists():
        print(f"❌ 找不到開頭音檔: {intro_path}")
        return
    if not main_audio_path.exists():
        print(f"❌ 找不到主音檔: {main_audio_path}")
        return

    print("🎧 開始將 intro.mp3 與 notebooklm_audio.m4a 合併...")

    # 使用 FFmpeg 的 filter_complex 處理不同格式的音檔合併
    command = [
        "ffmpeg",
        "-y",
        "-i", str(intro_path),
        "-i", str(main_audio_path),
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[outa]",
        "-map", "[outa]",
        "-c:a", "aac",
        "-b:a", "192k",
        str(temp_output_path)
    ]

    try:
        # 執行 FFmpeg 並在終端機顯示進度
        print(f"▶ 執行指令: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # 如果成功，用暫存檔覆蓋原始的 m4a
        temp_output_path.replace(main_audio_path)
        print(f"✅ 第 {ep_num:02d} 集音檔合併成功！已覆寫: {main_audio_path.name}")

    except subprocess.CalledProcessError:
        print(f"❌ 第 {ep_num:02d} 集 FFmpeg 執行失敗。")
        if temp_output_path.exists():
            temp_output_path.unlink()  # 清理暫存檔
    except Exception as e:
        print(f"❌ 發生未知錯誤: {e}")

def main():
    parser = argparse.ArgumentParser(description="合併 Intro 音檔與 NotebookLM 主音檔")
    parser.add_argument("--ep", type=int, help="處理單一集數 (例如: --ep 5)")
    parser.add_argument("--start", type=int, help="起始集數")
    parser.add_argument("--end", type=int, help="結束集數")
    args = parser.parse_args()

    # 決定要處理的集數清單
    if args.ep is not None:
        episodes = [args.ep]
    elif args.start is not None:
        end_ep = args.end if args.end else args.start
        episodes = list(range(args.start, end_ep + 1))
    else:
        print("❌ 請指定 --ep 或 --start 參數")
        return

    # 定義 base_dir (假設此腳本放在 scripts/ 下，所以回推一層就是專案根目錄)
    base_dir = Path(__file__).parent.parent

    for ep in episodes:
        merge_audio_for_episode(ep, base_dir)

if __name__ == "__main__":
    main()
