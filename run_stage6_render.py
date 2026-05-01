import argparse
import csv
import os
import subprocess
from pathlib import Path

from app_utils.power import keep_system_awake


VIDEO_FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


def get_workspace_dir() -> Path:
    base_dir = Path(__file__).resolve().parent
    return Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))


def get_ep_folder(workspace_dir: Path, ep_num: int) -> Path | None:
    folders = list(workspace_dir.glob(f"Ep{ep_num:02d}_*"))
    return folders[0] if folders else None


def read_storyboard_rows(storyboard_csv: Path) -> list[dict]:
    with storyboard_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scene_duration(row: dict) -> float:
    try:
        start_time = float(row.get("start_time", 0) or 0)
        end_time = float(row.get("end_time", 0) or 0)
        duration = end_time - start_time
        return duration if duration > 0 else 0.5
    except Exception:
        return 0.5


def scene_image_path(target_folder: Path, row: dict) -> Path | None:
    custom_image_path = str(row.get("custom_image_path", "")).strip()
    if custom_image_path and Path(custom_image_path).exists():
        return Path(custom_image_path)

    source_type = str(row.get("source_type", "")).strip().upper()
    if source_type == "FLASHCARD":
        flashcard_word = str(row.get("flashcard_word", "")).strip()
        flashcard_path = target_folder / "04_images" / "flashcards" / f"{flashcard_word}.png"
        return flashcard_path if flashcard_path.exists() else None

    start_token = str(row.get("start_time", "")).strip()
    ai_image_path = target_folder / "04_images" / "ai_generated" / f"img_{start_token}.png"
    return ai_image_path if ai_image_path.exists() else None


def scene_animation_path(target_folder: Path, row: dict) -> Path | None:
    custom_animation_path = str(row.get("animation_video_path", "")).strip()
    if custom_animation_path and Path(custom_animation_path).exists():
        return Path(custom_animation_path)

    scene_id = str(row.get("scene_id", "")).strip()
    auto_path = target_folder / "04_images" / "animations" / f"scene_{scene_id}.mp4"
    return auto_path if auto_path.exists() else None


def rel_posix(path: Path, start: Path) -> str:
    return os.path.relpath(path, start).replace("\\", "/")


def ffmpeg_visual_filter() -> str:
    return (
        f"fps={VIDEO_FPS},"
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
        "format=yuv420p"
    )


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_segment_from_image(target_folder: Path, image_path: Path, output_path: Path, duration: float) -> None:
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            rel_posix(image_path, target_folder),
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            ffmpeg_visual_filter(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            rel_posix(output_path, target_folder),
        ],
        cwd=target_folder,
    )


def build_segment_from_animation(target_folder: Path, animation_path: Path, output_path: Path, duration: float) -> None:
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            rel_posix(animation_path, target_folder),
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            ffmpeg_visual_filter(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            rel_posix(output_path, target_folder),
        ],
        cwd=target_folder,
    )


def render_final_video(ep_num: int, workspace_dir: Path) -> None:
    print(f"\n[Stage 6] 開始處理第 {ep_num:02d} 集影片合成...")

    target_folder = get_ep_folder(workspace_dir, ep_num)
    if not target_folder:
        print(f"找不到第 {ep_num:02d} 集資料夾。")
        return

    storyboard_csv = target_folder / "03_storyboards" / "storyboard.csv"
    audio_file = target_folder / "01_audio" / "notebooklm_audio.m4a"
    subtitle_file = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"
    output_dir = target_folder / "05_output"
    output_file = output_dir / "final_video.mp4"
    segments_dir = output_dir / "render_segments"
    concat_list = output_dir / "segments.txt"

    if not storyboard_csv.exists():
        print(f"找不到分鏡檔：{storyboard_csv}")
        return
    if not audio_file.exists():
        print(f"找不到音訊檔：{audio_file}")
        return
    if not subtitle_file.exists():
        print(f"找不到字幕檔：{subtitle_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    rows = read_storyboard_rows(storyboard_csv)
    if not rows:
        print("storyboard.csv 沒有任何 scene。")
        return

    with keep_system_awake(f"No.{ep_num:02d} FFmpeg 影片合成"):
        segment_lines: list[str] = []

        for idx, row in enumerate(rows, start=1):
            duration = scene_duration(row)
            segment_path = segments_dir / f"seg_{idx:03d}.mp4"
            animation_path = scene_animation_path(target_folder, row)
            image_path = scene_image_path(target_folder, row)

            print(
                f"[scene] {row.get('scene_id', idx)} | duration={duration:.3f}s | "
                f"animation={'Y' if animation_path else 'N'} | image={'Y' if image_path else 'N'}"
            )

            if animation_path and animation_path.exists():
                build_segment_from_animation(target_folder, animation_path, segment_path, duration)
            elif image_path and image_path.exists():
                build_segment_from_image(target_folder, image_path, segment_path, duration)
            else:
                raise FileNotFoundError(
                    f"Scene {row.get('scene_id', idx)} 找不到可用的動畫或圖片來源。"
                )

            segment_lines.append(f"file '{rel_posix(segment_path, output_dir)}'")

        concat_list.write_text("\n".join(segment_lines) + "\n", encoding="utf-8")
        print(f"已建立 concat 清單：{concat_list}")

        subtitle_rel = rel_posix(subtitle_file, target_folder)

        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                rel_posix(concat_list, target_folder),
                "-i",
                rel_posix(audio_file, target_folder),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-vf",
                (
                    "subtitles="
                    f"{subtitle_rel}:"
                    "force_style='FontName=Microsoft JhengHei,FontSize=16,"
                    "PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,"
                    "Outline=2,MarginL=40,MarginR=40,WrapStyle=1'"
                ),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                rel_posix(output_file, target_folder),
            ],
            cwd=target_folder,
        )

    print(f"第 {ep_num:02d} 集 final_video.mp4 已輸出：{output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: 產生最終影片並燒錄字幕。")
    parser.add_argument("--ep", type=int, help="單集集數")
    parser.add_argument("--start", type=int, help="起始集數")
    parser.add_argument("--end", type=int, help="結束集數")
    args = parser.parse_args()

    if args.ep is not None:
        start_ep = args.ep
        end_ep = args.ep
    elif args.start is not None:
        start_ep = args.start
        end_ep = args.end if args.end else args.start
    else:
        parser.error("請提供 --ep 或 --start / --end")

    workspace_dir = get_workspace_dir()
    for ep in range(start_ep, end_ep + 1):
        render_final_video(ep, workspace_dir)

    print(f"\nStage 6 完成，處理範圍：{start_ep:02d} - {end_ep:02d}")


if __name__ == "__main__":
    main()
