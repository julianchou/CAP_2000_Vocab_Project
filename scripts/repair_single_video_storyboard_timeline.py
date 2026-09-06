import argparse
import csv
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "workspaces" / "single_video"


def parse_srt_timestamp(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
    if not match:
        return 0.0
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    cues = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[1].split("-->", 1)]
        cues.append({
            "start": parse_srt_timestamp(start_raw),
            "end": parse_srt_timestamp(end_raw),
            "content": " ".join(lines[2:]).strip(),
        })
    return cues


def subtitle_groups(cues: list[dict], group_count: int) -> list[dict]:
    groups = []
    total = len(cues)
    for idx in range(group_count):
        start_idx = int(idx * total / group_count)
        end_idx = int((idx + 1) * total / group_count) - 1
        end_idx = max(start_idx, min(end_idx, total - 1))
        chunk = cues[start_idx: end_idx + 1]
        groups.append({
            "start": round(float(chunk[0]["start"]), 3),
            "end": round(float(chunk[-1]["end"]), 3),
            "content": " | ".join(cue["content"] for cue in chunk if cue.get("content")),
        })
    return groups


def find_episode_folder(ep_num: int) -> Path:
    candidates = sorted(WORKSPACE_DIR.glob(f"Ep{ep_num:02d}_*_*"))
    if not candidates:
        raise FileNotFoundError(f"No single_video folder found for Ep{ep_num:02d}")
    return candidates[0]


def repair_episode(ep_num: int) -> Path:
    ep_path = find_episode_folder(ep_num)
    storyboard_csv = ep_path / "03_storyboards" / "storyboard.csv"
    srt_path = ep_path / "02_subtitles" / "notebooklm_audio_fixed.srt"
    if not storyboard_csv.exists():
        raise FileNotFoundError(storyboard_csv)
    if not srt_path.exists():
        raise FileNotFoundError(srt_path)

    with storyboard_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No storyboard rows in {storyboard_csv}")

    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError(f"No subtitle cues in {srt_path}")

    groups = subtitle_groups(cues, len(rows))
    for idx, row in enumerate(rows, start=1):
        group = groups[idx - 1]
        row["scene_id"] = str(idx)
        row["start_time"] = f"{group['start']:.3f}".rstrip("0").rstrip(".")
        row["end_time"] = f"{max(group['end'], group['start'] + 0.2):.3f}".rstrip("0").rstrip(".")
        row["source_type"] = "AI"
        row["flashcard_word"] = ""
        row["subtitle_reference"] = group["content"]
        if not str(row.get("custom_image_path") or "").strip():
            for candidate in [
                ep_path / "04_images" / "ai" / f"scene_{idx:02d}.png",
                ep_path / "04_images" / "ai" / f"scene_{idx:03d}.png",
            ]:
                if candidate.exists():
                    row["custom_image_path"] = str(candidate)
                    break

    with storyboard_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    storyboard_json = storyboard_csv.with_suffix(".json")
    storyboard_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return storyboard_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()
    repaired = repair_episode(args.ep)
    print(f"Repaired storyboard timeline: {repaired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
