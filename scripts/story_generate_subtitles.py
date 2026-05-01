import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


PREFERRED_SPLIT_CHARS = "，。！？、；：,.!?"


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def media_duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".mp3":
        # OpenAI TTS MP3 output is constant bitrate in this workflow.
        # Estimating from file size avoids spawning ffprobe once per line on Windows.
        return max(path.stat().st_size * 8 / 128_000, 0.01)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return max(float((result.stdout or "0").strip() or "0"), 0.01)


def srt_time(seconds: float) -> str:
    millis = int(round(max(seconds, 0) * 1000))
    hours, rem = divmod(millis, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def split_text(text: str, soft_limit: int) -> list[str]:
    text = " ".join(str(text or "").strip().split())
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) >= soft_limit and (char in PREFERRED_SPLIT_CHARS or len(current) >= soft_limit + 8):
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def allocate_chunks(line: dict, start_sec: float, duration_sec: float, max_chars: int) -> list[dict]:
    chunks = split_text(str(line.get("text", "")), max_chars)
    if not chunks:
        return []
    total_weight = sum(max(len(chunk), 1) for chunk in chunks)
    cursor = start_sec
    out = []
    for idx, chunk in enumerate(chunks):
        if idx == len(chunks) - 1:
            end_sec = start_sec + duration_sec
        else:
            ratio = max(len(chunk), 1) / total_weight
            end_sec = cursor + duration_sec * ratio
        if end_sec <= cursor:
            end_sec = cursor + 0.25
        out.append(
            {
                "line_id": line.get("line_id", ""),
                "sequence": line.get("sequence", ""),
                "paragraph_id": line.get("paragraph_id", ""),
                "speaker": line.get("speaker", ""),
                "start": round(cursor, 3),
                "end": round(end_sec, 3),
                "text": chunk,
            }
        )
        cursor = end_sec
    return out


def build_subtitles(ep_path: Path, max_chars: int) -> dict:
    tts_path = ep_path / "04_audio_subtitles" / "tts_lines.json"
    cast_path = ep_path / "04_audio_subtitles" / "voice_cast.json"
    tts_data = read_json(tts_path)
    voice_cast = read_json(cast_path) if cast_path.exists() else {}
    pause_cfg = voice_cast.get("pause_ms") or {}

    lines = [line for line in (tts_data.get("lines") or []) if line.get("audio_path")]
    if not lines:
        raise RuntimeError("no TTS audio segments found. Run No.4.2 first.")

    subtitles = []
    cursor = 0.0
    for idx, line in enumerate(lines):
        audio_path = ep_path / str(line.get("audio_path"))
        if not audio_path.exists():
            raise FileNotFoundError(f"missing audio segment: {audio_path}")
        duration = media_duration_seconds(audio_path)
        subtitles.extend(allocate_chunks(line, cursor, duration, max_chars))
        pause_ms = pause_for_line(line, lines[idx + 1] if idx + 1 < len(lines) else None, pause_cfg)
        cursor += duration + max(pause_ms, 0) / 1000

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "tts_segments",
        "story_title": tts_data.get("story_title", ""),
        "subtitle_count": len(subtitles),
        "duration_seconds": round(cursor, 3),
        "max_chars": max_chars,
        "subtitles": subtitles,
    }


def write_srt(path: Path, subtitles: list[dict]) -> None:
    blocks = []
    for idx, item in enumerate(subtitles, 1):
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"{srt_time(float(item['start']))} --> {srt_time(float(item['end']))}",
                    str(item.get("text", "")).strip(),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode No.4.4: generate subtitles from TTS segment timings.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    parser.add_argument("--max-chars", type=int, default=22)
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    payload = build_subtitles(ep_path, max_chars=max(args.max_chars, 8))
    out_dir = ep_path / "04_audio_subtitles"
    srt_path = out_dir / "story_subtitles.srt"
    json_path = out_dir / "story_subtitles.json"
    write_srt(srt_path, payload["subtitles"])
    write_json(json_path, payload)
    print(f"story_subtitles_srt={srt_path}")
    print(f"story_subtitles_json={json_path}")
    print(f"subtitle_count={payload['subtitle_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
