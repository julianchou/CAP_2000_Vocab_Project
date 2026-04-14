import argparse
import subprocess
import sys
from pathlib import Path

def get_ep_folder(workspace_dir, ep_num):
    """根據集數尋找對應的 EpXX 資料夾"""
    folders = list(workspace_dir.glob(f"Ep{ep_num:02d}_*"))
    return folders[0] if folders else None

def render_final_video(ep_num, workspace_dir):
    print(f"\n🎥 [Stage 6] 準備執行第 {ep_num:02d} 集的最終高畫質渲染...")
    
    target_folder = get_ep_folder(workspace_dir, ep_num)
    if not target_folder:
        print(f"⚠️ 找不到第 {ep_num:02d} 集的資料夾，跳過。")
        return

    # 定義檔案路徑 (皆使用相對於 target_folder 的相對路徑)
    inputs_file = "05_output/inputs.txt"
    audio_file = "01_audio/notebooklm_audio.m4a"
    output_file = "05_output/final_video.mp4"
    
    # 檢查字幕檔是否存在 (使用絕對路徑檢查)
    srt_absolute_path = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"
    if not srt_absolute_path.exists():
        print(f"❌ 找不到字幕檔：{srt_absolute_path}，請確認 Stage 4 是否有成功產生。")
        return
        
    # ⚡ 關鍵修正：FFmpeg 濾鏡中使用「相對路徑」，完美避開 Windows C:\ 冒號造成的解析錯誤
    srt_relative_path = "02_subtitles/notebooklm_audio_fixed.srt"

    # 組合 FFmpeg 完整指令
    # 組合 FFmpeg 完整指令
    ffmpeg_cmd = [
        "ffmpeg", "-y", 
        "-f", "concat", "-safe", "0", 
        "-i", inputs_file,
        "-i", audio_file,
        "-map", "0:v", "-map", "1:a",
        # 使用 srt_relative_path
        "-vf", f"fps=30,subtitles={srt_relative_path}:force_style='FontName=Microsoft JhengHei,FontSize=16,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,MarginL=40,MarginR=40,WrapStyle=1'",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",  # ⚡ 關鍵修復：強制將色彩格式轉為 Windows/Mac 相容的標準格式 (解決黑屏問題)
        "-preset", "medium",  # 速度與畫質的平衡
        "-crf", "23",         # 高畫質標準值
        "-c:a", "aac",
        "-b:a", "192k",
        output_file
    ]

    try:
        print(f"🚀 正在呼叫 FFmpeg 進行渲染 (這會花費較長的時間，請耐心等候)...")
        # 設定 cwd=target_folder，讓上面的相對路徑能正確指向檔案
        subprocess.run(ffmpeg_cmd, cwd=target_folder, check=True)
        print(f"✅ [Stage 6] 第 {ep_num:02d} 集最終影片渲染完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ [Stage 6] 第 {ep_num:02d} 集影片渲染失敗。錯誤訊息: {e}")

def main():
    parser = argparse.ArgumentParser(description="Stage 6: 最終影片渲染主控")
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

    # 設定專案的 workspace 目錄
    base_dir = Path(__file__).parent
    workspace_dir = base_dir / "workspace"

    for ep in range(start_ep, end_ep + 1):
        render_final_video(ep, workspace_dir)

    print(f"\n🎉 階段 6 完成：第 {start_ep} 至 {end_ep} 集全數渲染完畢！")

if __name__ == "__main__":
    main()