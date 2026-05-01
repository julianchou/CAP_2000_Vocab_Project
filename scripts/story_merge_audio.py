import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def pause_for_line(line: dict, next_line: dict | None, pause_cfg: dict) -> int:
    text = str(line.get("text", "")).strip()
    if next_line and line.get("paragraph_id") != next_line.get("paragraph_id"):
        return int(pause_cfg.get("paragraph", 900))
    if text.endswith(("？", "?")):
        return int(pause_cfg.get("question", 700))
    if text.endswith(("。", "！", "!", ".", "…")):
        return int(pause_cfg.get("sentence", 500))
    if text.endswith(("，", ",")):
        return int(pause_cfg.get("comma", 250))
    return int(pause_cfg.get("default", 350))


def ffmpeg_escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def make_silence_file(path: Path, duration_ms: int) -> None:
    duration = max(duration_ms, 1) / 1000
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{duration:.3f}",
        "-q:a",
        "9",
        "-acodec",
        "libmp3lame",
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def merge_audio(ep_path: Path) -> Path:
    tts_path = ep_path / "04_audio_subtitles" / "tts_lines.json"
    cast_path = ep_path / "04_audio_subtitles" / "voice_cast.json"
    tts_data = read_json(tts_path)
    voice_cast = read_json(cast_path) if cast_path.exists() else {}
    pause_cfg = voice_cast.get("pause_ms") or {}

    lines = [line for line in (tts_data.get("lines") or []) if line.get("audio_path")]
    if not lines:
        raise RuntimeError("no audio segments found in tts_lines.json")
    missing = [line.get("line_id") for line in lines if not (ep_path / str(line.get("audio_path"))).exists()]
    if missing:
        raise FileNotFoundError(f"missing audio segments: {', '.join(str(x) for x in missing[:10])}")

    work_dir = ep_path / "04_audio_subtitles" / "_merge_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    concat_list = work_dir / "concat_list.txt"
    output_path = ep_path / "04_audio_subtitles" / "story_audio.m4a"

    entries: list[Path] = []
    for idx, line in enumerate(lines):
        entries.append(ep_path / str(line.get("audio_path")))
        pause_ms = pause_for_line(line, lines[idx + 1] if idx + 1 < len(lines) else None, pause_cfg)
        if pause_ms > 0:
            silence_path = work_dir / f"silence_{idx + 1:04d}_{pause_ms}ms.mp3"
            if not silence_path.exists() or silence_path.stat().st_size == 0:
                make_silence_file(silence_path, pause_ms)
            entries.append(silence_path)

    concat_list.write_text(
        "\n".join(f"file '{ffmpeg_escape_concat_path(path)}'" for path in entries) + "\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)

    tts_data["audio_merged_at"] = datetime.now().isoformat(timespec="seconds")
    tts_data["merged_audio_path"] = output_path.relative_to(ep_path).as_posix()
    tts_path.write_text(json.dumps(tts_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode No.4.3: merge TTS audio segments.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    output = merge_audio(ep_path)
    print(f"story_audio={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
