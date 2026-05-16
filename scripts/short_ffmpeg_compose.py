import argparse
import json
import shutil
import subprocess
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def resolve_path(ep_dir: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ep_dir / path
    return path


def first_existing(ep_dir: Path, values: list[str]) -> Path | None:
    for value in values:
        path = resolve_path(ep_dir, value)
        if path and path.exists() and path.stat().st_size > 0:
            return path
    return None


def run(cmd: list[str]) -> None:
    print("[RUN] " + " ".join(str(part) for part in cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed return_code={result.returncode}")


def concat_escape(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def ffmpeg_subtitle_path(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/")
    text = text.replace(":", "\\:").replace("'", "\\'")
    return text


def build_image_clip(ffmpeg: str, visual: Path, output: Path, duration: float) -> None:
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,format=yuv420p"
    run([
        ffmpeg, "-y",
        "-loop", "1",
        "-t", f"{max(duration, 0.3):.3f}",
        "-i", str(visual),
        "-vf", vf,
        "-r", "30",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def build_video_clip(ffmpeg: str, visual: Path, output: Path, duration: float) -> None:
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30,format=yuv420p"
    run([
        ffmpeg, "-y",
        "-stream_loop", "-1",
        "-t", f"{max(duration, 0.3):.3f}",
        "-i", str(visual),
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def concat_clips(ffmpeg: str, clips: list[Path], concat_list: Path, output: Path) -> None:
    concat_list.write_text("\n".join(f"file '{concat_escape(path)}'" for path in clips) + "\n", encoding="utf-8")
    run([
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output),
    ])


def add_audio_and_subtitles(ffmpeg: str, video: Path, audio: Path | None, subtitle: Path | None, output: Path) -> None:
    vf = []
    if subtitle:
        vf.append(f"subtitles='{ffmpeg_subtitle_path(subtitle)}'")
    cmd = [ffmpeg, "-y", "-i", str(video)]
    if audio:
        cmd.extend(["-i", str(audio)])
    if vf:
        cmd.extend(["-vf", ",".join(vf)])
    if audio:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    else:
        cmd.extend(["-map", "0:v:0"])
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ])
    if audio:
        cmd.extend(["-c:a", "aac"])
    cmd.append(str(output))
    run(cmd)


def compose_short(ep_dir: Path, compose_list_path: Path, output_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    payload = read_json(compose_list_path)
    items = payload.get("items") or []
    if not items:
        raise ValueError("compose list has no items")

    work_dir = ep_dir / "05_output" / "_ffmpeg_short_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for idx, item in enumerate(items, 1):
        visual = first_existing(ep_dir, [item.get("selected_visual", ""), item.get("animation", ""), item.get("image", "")])
        if not visual:
            raise FileNotFoundError(f"scene {item.get('scene', idx)} has no existing visual asset")
        duration = float(item.get("duration_seconds", 0) or 0)
        if duration <= 0:
            duration = 3.0
        clip_path = work_dir / f"clip_{idx:03d}.mp4"
        suffix = visual.suffix.lower()
        print(f"[CLIP] scene={item.get('scene', idx)} visual={visual} duration={duration:.3f}", flush=True)
        if suffix in {".mp4", ".mov", ".webm", ".mkv"}:
            build_video_clip(ffmpeg, visual, clip_path, duration)
        else:
            build_image_clip(ffmpeg, visual, clip_path, duration)
        clips.append(clip_path)

    silent_video = work_dir / "video_no_audio.mp4"
    concat_clips(ffmpeg, clips, work_dir / "concat_list.txt", silent_video)

    audio = first_existing(ep_dir, [
        "01_audio/notebooklm_audio.m4a",
        "01_audio/audio.m4a",
        "01_audio/audio.mp3",
    ])
    if not audio:
        audio_dir = ep_dir / "01_audio"
        audio_candidates = []
        if audio_dir.exists():
            for pattern in ("*.m4a", "*.mp3", "*.wav", "*.aac", "*.ogg"):
                audio_candidates.extend(audio_dir.glob(pattern))
        audio = sorted(audio_candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0] if audio_candidates else None

    subtitle = first_existing(ep_dir, payload.get("subtitle_candidates") or [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    add_audio_and_subtitles(ffmpeg, silent_video, audio, subtitle, output_path)
    print(f"[INFO] output_video={output_path}", flush=True)
    print(f"[INFO] audio={audio or ''}", flush=True)
    print(f"[INFO] subtitle={subtitle or ''}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 9: compose final video with FFmpeg.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--compose-list", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compose_short(
        ep_dir=Path(args.ep_dir).resolve(),
        compose_list_path=Path(args.compose_list).resolve(),
        output_path=Path(args.output).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
