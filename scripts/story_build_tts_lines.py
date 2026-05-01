import argparse
import csv
import json
import os
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


def character_voice_map(characters_data: dict) -> dict:
    result = {}
    for item in characters_data.get("characters") or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        personality = item.get("personality") or {}
        result[name] = {
            "character_id": item.get("character_id", ""),
            "voice_style": personality.get("voice_style", "清楚、溫柔、適合兒童故事。"),
            "role": item.get("role", ""),
        }
    return result


def add_line(lines: list[dict], paragraph: dict, line_type: str, speaker: str, text: str, voice_hint: str, voice_map: dict) -> None:
    text = str(text or "").strip()
    if not text:
        return
    sequence = len(lines) + 1
    voice_info = voice_map.get(speaker, {}) if line_type == "dialogue" else {}
    lines.append(
        {
            "line_id": f"L{sequence:04d}",
            "sequence": sequence,
            "paragraph_id": paragraph.get("paragraph_id", ""),
            "paragraph_title": paragraph.get("title", ""),
            "line_type": line_type,
            "speaker": speaker,
            "character_id": voice_info.get("character_id", "narrator" if line_type == "narration" else ""),
            "voice_style": voice_info.get("voice_style", "溫暖、清楚、像兒童故事旁白。"),
            "voice_hint": voice_hint,
            "text": text,
            "text_length": len(text),
            "tts_status": "pending",
            "audio_path": "",
        }
    )


def build_tts_lines(script_data: dict, characters_data: dict) -> dict:
    voice_map = character_voice_map(characters_data)
    lines: list[dict] = []
    for paragraph in script_data.get("paragraphs") or []:
        for narration in paragraph.get("narration") or []:
            add_line(
                lines,
                paragraph,
                "narration",
                "旁白",
                narration,
                "溫暖、慢速、清楚，像睡前故事旁白",
                voice_map,
            )
        for dialogue in paragraph.get("dialogues") or []:
            speaker = str(dialogue.get("speaker", "")).strip() or "角色"
            add_line(
                lines,
                paragraph,
                "dialogue",
                speaker,
                dialogue.get("text", ""),
                dialogue.get("voice_hint", ""),
                voice_map,
            )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_story_id": script_data.get("source_story_id", ""),
        "story_title": script_data.get("title", ""),
        "target_age": script_data.get("target_age", "10歲以下"),
        "line_count": len(lines),
        "narration_count": sum(1 for item in lines if item["line_type"] == "narration"),
        "dialogue_count": sum(1 for item in lines if item["line_type"] == "dialogue"),
        "lines": lines,
    }


def write_csv(path: Path, lines: list[dict]) -> None:
    fieldnames = [
        "line_id",
        "sequence",
        "paragraph_id",
        "paragraph_title",
        "line_type",
        "speaker",
        "character_id",
        "voice_style",
        "voice_hint",
        "text",
        "text_length",
        "tts_status",
        "audio_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lines:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 4.1: build TTS line list.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    script_data = read_json(ep_path / "02_story" / "story_script.json")
    characters_path = ep_path / "03_characters_scenes" / "characters.json"
    characters_data = read_json(characters_path) if characters_path.exists() else {}

    payload = build_tts_lines(script_data, characters_data)
    out_dir = ep_path / "04_audio_subtitles"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tts_lines.json"
    csv_path = out_dir / "tts_lines.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, payload["lines"])
    print(f"tts_lines_json={json_path}")
    print(f"tts_lines_csv={csv_path}")
    print(f"line_count={payload['line_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
