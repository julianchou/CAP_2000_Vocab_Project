import argparse
import json
import shutil
import subprocess
from pathlib import Path


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def read_storyboard(path: Path) -> list[dict]:
    data = read_json(path, [])
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        raise ValueError(f"storyboard must be a JSON array: {path}")
    return [item for item in data if isinstance(item, dict)]


def resolve_asset(ep_dir: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ep_dir / path
    return path


def scene_asset_candidates(ep_dir: Path, scene_no: int, kind: str) -> list[Path]:
    base_dir = ep_dir / "04_images" / ("storyboard" if kind == "image" else "animations")
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"] if kind == "image" else ["*.mp4", "*.mov", "*.webm", "*.m4v"]
    tokens = [f"{scene_no:03d}", str(scene_no)]
    candidates = []
    for token in dict.fromkeys(tokens):
        for pattern in patterns:
            suffix = Path(pattern).suffix
            candidates.extend(sorted(base_dir.glob(f"scene_{token}{suffix}")))
            candidates.extend(sorted(base_dir.glob(f"scene_{token}_*{suffix}")))
    valid = [path for path in candidates if path.exists() and path.is_file() and path.stat().st_size > 0]
    return sorted(
        valid,
        key=lambda path: (
            0 if "_upload_" in path.name.lower() else 1,
            -path.stat().st_mtime,
            path.name.lower(),
        ),
    )


def fallback_scene_asset(ep_dir: Path, scene_no: int, kind: str) -> Path | None:
    candidates = scene_asset_candidates(ep_dir, scene_no, kind)
    return candidates[0] if candidates else None


def rel_path(ep_dir: Path, path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(ep_dir.resolve()).as_posix()
    except Exception:
        return str(path)


def parse_seconds(row: dict, key: str, fallback: float) -> float:
    try:
        return float(row.get(key, fallback) or fallback)
    except Exception:
        return fallback


def first_audio_file(ep_dir: Path) -> Path | None:
    audio_dir = ep_dir / "01_audio"
    preferred = [
        audio_dir / "notebooklm_audio.m4a",
        audio_dir / "audio.m4a",
        audio_dir / "audio.mp3",
    ]
    for path in preferred:
        if path.exists() and path.stat().st_size > 0:
            return path
    if not audio_dir.exists():
        return None
    candidates = []
    for pattern in ("*.m4a", "*.mp3", "*.wav", "*.aac", "*.ogg"):
        candidates.extend(audio_dir.glob(pattern))
    candidates = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0] if candidates else None


def media_duration_seconds(path: Path | None) -> float | None:
    if not path:
        return None
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
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
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except Exception:
        return None


def build_compose_list(ep_dir: Path, storyboard_path: Path, output_path: Path) -> Path:
    storyboard = read_storyboard(storyboard_path)
    if not storyboard:
        raise ValueError("storyboard has no scenes")

    items = []
    warnings = []
    audio_path = first_audio_file(ep_dir)
    audio_duration = media_duration_seconds(audio_path)
    for idx, scene in enumerate(storyboard, 1):
        scene_no = int(scene.get("scene", scene.get("scene_id", idx)) or idx)
        image_path = resolve_asset(ep_dir, str(scene.get("asset", "") or scene.get("image_asset", "")))
        animation_path = resolve_asset(ep_dir, str(scene.get("animation_asset", "") or scene.get("animation_video_path", "")))
        if not image_path or not image_path.exists():
            image_path = fallback_scene_asset(ep_dir, scene_no, "image")
        if not animation_path or not animation_path.exists():
            animation_path = fallback_scene_asset(ep_dir, scene_no, "animation")
        chosen_visual = animation_path if animation_path and animation_path.exists() else image_path
        visual_type = "animation" if animation_path and animation_path.exists() else "image"
        start_seconds = parse_seconds(scene, "start_seconds", 0)
        end_seconds = parse_seconds(scene, "end_seconds", start_seconds + 3)
        subtitle_duration = max(end_seconds - start_seconds, 0.3)
        next_scene = storyboard[idx] if idx < len(storyboard) else None
        next_start = parse_seconds(next_scene, "start_seconds", end_seconds) if next_scene else None
        compose_end_seconds = end_seconds
        if next_start and next_start > start_seconds:
            compose_end_seconds = next_start
        elif audio_duration and idx == len(storyboard) and audio_duration > start_seconds:
            compose_end_seconds = audio_duration
        duration = max(compose_end_seconds - start_seconds, subtitle_duration, 0.3)

        missing = []
        if image_path and not image_path.exists():
            missing.append(f"image missing: {image_path}")
        if animation_path and not animation_path.exists():
            missing.append(f"animation missing: {animation_path}")
        if not chosen_visual:
            missing.append("no image or animation asset")
        for msg in missing:
            warnings.append({"scene": scene_no, "message": msg})

        items.append(
            {
                "scene": scene_no,
                "start": scene.get("start", ""),
                "end": scene.get("end", ""),
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "compose_end_seconds": round(start_seconds + duration, 3),
                "duration_seconds": round(duration, 3),
                "subtitle": str(scene.get("subtitle", "") or ""),
                "summary": str(scene.get("summary", "") or ""),
                "visual_type": visual_type if chosen_visual else "",
                "image": rel_path(ep_dir, image_path),
                "animation": rel_path(ep_dir, animation_path),
                "selected_visual": rel_path(ep_dir, chosen_visual),
                "image_prompt": str(scene.get("image_prompt", "") or scene.get("prompt", "") or ""),
                "animation_prompt": str(scene.get("animation_prompt", "") or ""),
                "warnings": missing,
            }
        )

    payload = {
        "version": 1,
        "ratio": "9:16",
        "episode_dir": str(ep_dir),
        "storyboard": str(storyboard_path),
        "audio": rel_path(ep_dir, audio_path) if audio_path else "01_audio",
        "audio_duration_seconds": round(audio_duration, 3) if audio_duration else None,
        "subtitle_candidates": [
            "02_subtitles/final.srt",
            "02_subtitles/reviewed.srt",
            "02_subtitles/whisper.srt",
        ],
        "output_video": "05_output/final_short.mp4",
        "scene_count": len(items),
        "warnings": warnings,
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] compose_list={output_path}")
    print(f"[INFO] scene_count={len(items)}")
    print(f"[INFO] warning_count={len(warnings)}")
    for warning in warnings[:30]:
        print(f"[WARN] scene={warning['scene']} {warning['message']}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 8: build compose list.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-output", default="")
    args = parser.parse_args()
    output_path = build_compose_list(
        ep_dir=Path(args.ep_dir).resolve(),
        storyboard_path=Path(args.storyboard).resolve(),
        output_path=Path(args.output).resolve(),
    )
    if args.preview_output:
        from short_ffmpeg_compose import compose_short

        preview_output = Path(args.preview_output).resolve()
        print(f"[INFO] build_preview_video={preview_output}")
        try:
            compose_short(
                ep_dir=Path(args.ep_dir).resolve(),
                compose_list_path=output_path,
                output_path=preview_output,
            )
        except Exception as exc:
            print(f"[WARN] preview video was not generated: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
