import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_utils.power import keep_system_awake


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str]) -> None:
    print("[merge] command=" + " ".join(str(part) for part in cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}")


def ffprobe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
            timeout=30,
        )
        value = float((result.stdout or "").strip())
        return value if value >= 0 else None
    except Exception:
        return None


def parse_srt_timestamp(text: str) -> float:
    hms, ms = text.strip().split(",", 1)
    hours, minutes, seconds = [int(part) for part in hms.split(":")]
    return hours * 3600 + minutes * 60 + seconds + int(ms[:3]) / 1000.0


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    entries: list[dict] = []
    for block in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
        try:
            start = parse_srt_timestamp(start_text)
            end = parse_srt_timestamp(end_text)
        except Exception:
            continue
        entries.append({"start": start, "end": end, "content": "\n".join(lines[2:]).strip()})
    return entries


def compose_srt(entries: list[dict]) -> str:
    blocks = []
    for idx, entry in enumerate(entries, start=1):
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"{format_srt_timestamp(entry['start'])} --> {format_srt_timestamp(entry['end'])}",
                    str(entry.get("content", "")).strip(),
                ]
            )
        )
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def selected_merge_video_path(item: dict) -> Path:
    explicit = str(item.get("merge_video_path") or "").strip()
    if explicit:
        return Path(explicit).resolve()
    if bool(item.get("include_outro")) and str(item.get("video_path_with_outro") or "").strip():
        return Path(str(item.get("video_path_with_outro"))).resolve()
    if not bool(item.get("include_outro")) and str(item.get("video_path_no_outro") or "").strip():
        return Path(str(item.get("video_path_no_outro"))).resolve()
    return Path(item["video_path"]).resolve()


def concat_videos(items: list[dict], output_dir: Path) -> Path:
    concat_path = output_dir / "concat_list.txt"
    lines = []
    for item in items:
        video_path = selected_merge_video_path(item)
        escaped = str(video_path).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_path = output_dir / "final_video.mp4"
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def merge_subtitles(items: list[dict], output_dir: Path) -> Path:
    merged: list[dict] = []
    offset = 0.0
    for item in items:
        video_path = selected_merge_video_path(item)
        subtitle_path = Path(str(item.get("subtitle_path") or ""))
        duration = ffprobe_duration(video_path)
        entries = parse_srt(subtitle_path) if subtitle_path.exists() else []
        print(
            f"[merge] subtitles source={subtitle_path if subtitle_path.exists() else '-'} "
            f"entries={len(entries)} offset={offset:.3f}s duration={duration if duration is not None else '-'}",
            flush=True,
        )
        for entry in entries:
            merged.append(
                {
                    "start": entry["start"] + offset,
                    "end": entry["end"] + offset,
                    "content": entry["content"],
                }
            )
        if duration is None:
            duration = max((entry["end"] for entry in entries), default=0.0)
        offset += max(float(duration or 0.0), 0.0)

    output_path = output_dir / "merged_subtitles.srt"
    output_path.write_text(compose_srt(merged), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge selected episode videos and shift subtitles.")
    parser.add_argument("--job", required=True, help="Path to merge job JSON.")
    args = parser.parse_args()

    job_path = Path(args.job).resolve()
    job = read_json(job_path)
    output_dir = Path(job["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(job.get("items") or [])
    if len(items) < 2:
        raise ValueError("at least two episode videos are required")

    proc_path = Path(job.get("proc_path") or (str(job_path) + ".proc.json"))
    proc_last_path = Path(job.get("proc_last_path") or (str(job_path) + ".proc.last.json"))
    proc_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": int(time.time())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        proc_last_path.unlink()
    except FileNotFoundError:
        pass

    try:
        print(
            f"\n==== RUN merge:{job.get('id') or job_path.stem} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====",
            flush=True,
        )
        job.update(
            {
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "ended_at": None,
                "last_message": "merging video",
            }
        )
        write_json(job_path, job)

        with keep_system_awake(f"Episode merge {job.get('id') or job_path.stem}"):
            print(f"[merge] title={job.get('title', '')}", flush=True)
            print(f"[merge] output_dir={output_dir}", flush=True)
            for idx, item in enumerate(items, start=1):
                video_path = selected_merge_video_path(item)
                print(
                    f"[merge] item {idx}: {item.get('profile_name')} Ep{int(item.get('ep', 0)):02d} "
                    f"outro={'yes' if item.get('include_outro') else 'no'} video={video_path}",
                    flush=True,
                )

            video_path = concat_videos(items, output_dir)
            job["last_message"] = "merging subtitles"
            write_json(job_path, job)
            subtitle_path = merge_subtitles(items, output_dir)

        manifest_path = output_dir / "merge_manifest.json"
        write_json(manifest_path, {**job, "output_video": str(video_path), "output_subtitle": str(subtitle_path)})
        job.update(
            {
                "status": "done",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "last_message": "done",
                "output_video": str(video_path),
                "output_subtitle": str(subtitle_path),
                "manifest_path": str(manifest_path),
            }
        )
        write_json(job_path, job)
        print(f"[merge] output_video={video_path}", flush=True)
        print(f"[merge] output_subtitle={subtitle_path}", flush=True)
        print("[merge] done", flush=True)
        print(
            f"==== END merge:{job.get('id') or job_path.stem} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | status=done ====",
            flush=True,
        )
        return 0
    except Exception as exc:
        job.update(
            {
                "status": "error",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "last_message": str(exc),
            }
        )
        write_json(job_path, job)
        print(f"[merge] error: {exc}", flush=True)
        print(
            f"==== END merge:{job.get('id') or job_path.stem} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | status=error ====",
            flush=True,
        )
        return 1
    finally:
        try:
            proc_last_path.write_text(proc_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
        try:
            proc_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
