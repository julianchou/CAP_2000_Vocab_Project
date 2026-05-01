import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


OPENAI_TTS_VOICES = [
    "marin",
    "cedar",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]


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


def default_voice_cast(tts_data: dict) -> dict:
    speakers = []
    for line in tts_data.get("lines") or []:
        speaker = str(line.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    voice_cycle = ["marin", "coral", "fable", "nova", "sage", "shimmer", "verse", "cedar"]
    voices = {}
    for idx, speaker in enumerate(speakers):
        if speaker == "旁白":
            voice = "marin"
            instructions = "用溫暖、清楚、慢速的繁體中文說故事，像10歲以下小朋友的睡前童話旁白。"
        else:
            voice = voice_cycle[idx % len(voice_cycle)]
            instructions = f"用適合兒童童話角色「{speaker}」的繁體中文聲音說話，語氣自然、清楚、情緒明確，避免誇張尖銳。"
        voices[speaker] = {
            "provider": "openai",
            "voice": voice,
            "instructions": instructions,
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "openai",
        "model": "gpt-4o-mini-tts",
        "response_format": "mp3",
        "voice_segments_dir": "04_audio_subtitles/voice_segments",
        "pause_ms": {
            "default": 350,
            "comma": 250,
            "sentence": 500,
            "question": 700,
            "paragraph": 900,
        },
        "voices": voices,
    }


def safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "line"


def line_audio_path(segments_dir: Path, line: dict, response_format: str) -> Path:
    sequence = int(line.get("sequence", 0) or 0)
    line_id = safe_filename(str(line.get("line_id", f"L{sequence:04d}")))
    speaker = safe_filename(str(line.get("speaker", "speaker")))
    return segments_dir / f"{sequence:04d}_{line_id}_{speaker}.{response_format}"


def line_instructions(line: dict, voice_cfg: dict) -> str:
    parts = [str(voice_cfg.get("instructions", "")).strip()]
    voice_style = str(line.get("voice_style", "")).strip()
    voice_hint = str(line.get("voice_hint", "")).strip()
    if voice_style:
        parts.append(f"角色聲音設定：{voice_style}")
    if voice_hint:
        parts.append(f"本句語氣：{voice_hint}")
    parts.append("請保持繁體中文發音自然，句尾停頓交給後製處理，不要自行加入音效。")
    return "\n".join(part for part in parts if part)


def write_csv(path: Path, lines: list[dict]) -> None:
    if not lines:
        return
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


def synthesize_lines(ep_path: Path, force: bool) -> int:
    tts_path = ep_path / "04_audio_subtitles" / "tts_lines.json"
    cast_path = ep_path / "04_audio_subtitles" / "voice_cast.json"
    csv_path = ep_path / "04_audio_subtitles" / "tts_lines.csv"
    tts_data = read_json(tts_path)

    if cast_path.exists():
        voice_cast = read_json(cast_path)
    else:
        voice_cast = default_voice_cast(tts_data)
        write_json(cast_path, voice_cast)
        print(f"[INFO] voice_cast created: {cast_path}")

    provider = str(voice_cast.get("provider", "openai")).strip().lower()
    if provider != "openai":
        raise ValueError(f"unsupported TTS provider: {provider}")

    model = str(voice_cast.get("model", "gpt-4o-mini-tts")).strip()
    response_format = str(voice_cast.get("response_format", "mp3")).strip().lower()
    if response_format != "mp3":
        raise ValueError("story mode audio merge currently requires response_format=mp3")

    segments_dir = ep_path / str(voice_cast.get("voice_segments_dir", "04_audio_subtitles/voice_segments"))
    segments_dir.mkdir(parents=True, exist_ok=True)
    voices = voice_cast.get("voices") or {}
    client = OpenAI()

    generated = 0
    failed = 0
    for line in tts_data.get("lines") or []:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        speaker = str(line.get("speaker", "旁白")).strip() or "旁白"
        voice_cfg = voices.get(speaker) or voices.get("旁白") or {}
        voice = str(voice_cfg.get("voice", "marin")).strip()
        output_path = line_audio_path(segments_dir, line, response_format)
        rel_output = output_path.relative_to(ep_path).as_posix()

        if output_path.exists() and output_path.stat().st_size > 0 and not force:
            line["tts_status"] = "done"
            line["audio_path"] = rel_output
            print(f"[SKIP] {line.get('line_id')} existing: {rel_output}")
            continue

        try:
            print(f"[TTS] {line.get('line_id')} {speaker}: {text[:40]}")
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                instructions=line_instructions(line, voice_cfg),
                response_format=response_format,
            ) as response:
                response.stream_to_file(output_path)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("empty audio output")
            line["tts_status"] = "done"
            line["audio_path"] = rel_output
            generated += 1
        except Exception as exc:
            line["tts_status"] = "error"
            line["tts_error"] = str(exc)
            failed += 1
            print(f"[ERROR] {line.get('line_id')} failed: {exc}")

    tts_data["tts_generated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(tts_path, tts_data)
    write_csv(csv_path, tts_data.get("lines") or [])
    print(f"[INFO] generated={generated} failed={failed}")
    if failed:
        return 1
    return 0


def synthesize_preview(ep_path: Path, request_path: Path) -> int:
    tts_path = ep_path / "04_audio_subtitles" / "tts_lines.json"
    cast_path = ep_path / "04_audio_subtitles" / "voice_cast.json"
    tts_data = read_json(tts_path)
    if cast_path.exists():
        voice_cast = read_json(cast_path)
    else:
        voice_cast = default_voice_cast(tts_data)
        write_json(cast_path, voice_cast)

    request = read_json(request_path)
    speaker = str(request.get("speaker", "旁白")).strip() or "旁白"
    text = str(request.get("text", "")).strip()
    if not text:
        raise ValueError("preview text is empty")

    response_format = str(voice_cast.get("response_format", "mp3")).strip().lower()
    if response_format != "mp3":
        raise ValueError("story mode voice preview currently requires response_format=mp3")

    voices = voice_cast.get("voices") or {}
    voice_cfg = voices.get(speaker) or voices.get("旁白") or {}
    voice = str(voice_cfg.get("voice", "marin")).strip()
    model = str(voice_cast.get("model", "gpt-4o-mini-tts")).strip()
    preview_dir = ep_path / "04_audio_subtitles" / "voice_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / f"{safe_filename(speaker)}_preview.mp3"
    line = {
        "line_id": "PREVIEW",
        "speaker": speaker,
        "text": text,
        "voice_style": request.get("voice_style", ""),
        "voice_hint": request.get("voice_hint", "預聽聲音"),
    }

    print(f"[PREVIEW] {speaker}: {text[:80]}")
    client = OpenAI()
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=line_instructions(line, voice_cfg),
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("empty preview audio output")
    request["generated_at"] = datetime.now().isoformat(timespec="seconds")
    request["audio_path"] = output_path.relative_to(ep_path).as_posix()
    write_json(request_path, request)
    print(f"preview_audio={output_path}")
    return 0


def run_merge(ep_num: int, workspace_root: Path) -> int:
    script = Path(__file__).resolve().parent / "story_merge_audio.py"
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--ep",
        str(ep_num),
        "--workspace-root",
        str(workspace_root),
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode No.4.2: generate TTS audio segments.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    parser.add_argument("--prepare-cast-only", action="store_true", help="create voice_cast.json without calling the TTS API")
    parser.add_argument("--preview-request", default="", help="generate a single preview audio from this JSON request")
    parser.add_argument("--force", action="store_true", help="regenerate existing segment files")
    parser.add_argument("--merge-after", action="store_true", help="run No.4.3 merge after TTS succeeds")
    args = parser.parse_args()

    load_dotenv()
    workspace_root = Path(args.workspace_root).resolve()
    ep_path = find_episode_path(workspace_root, args.ep)
    if args.prepare_cast_only:
        tts_data = read_json(ep_path / "04_audio_subtitles" / "tts_lines.json")
        cast_path = ep_path / "04_audio_subtitles" / "voice_cast.json"
        if cast_path.exists():
            print(f"[INFO] voice_cast already exists: {cast_path}")
        else:
            write_json(cast_path, default_voice_cast(tts_data))
            print(f"[INFO] voice_cast created: {cast_path}")
        return 0
    if args.preview_request:
        request_path = Path(args.preview_request)
        if not request_path.is_absolute():
            request_path = ep_path / request_path
        return synthesize_preview(ep_path, request_path)
    code = synthesize_lines(ep_path, force=args.force)
    if code != 0:
        return code
    if args.merge_after:
        return run_merge(args.ep, workspace_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
