import argparse
import datetime
import os
from pathlib import Path

import whisper
from dotenv import load_dotenv


SUPPORTED_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".webm", ".mp4"}
PUNCTUATIONS = ["\uFF0C", "\u3002", "\uFF01", "\uFF1F", "\u3001", ",", ".", "!", "?"]
DEFAULT_INITIAL_PROMPT = (
    "\u4ee5\u4e0b\u662f\u7e41\u9ad4\u4e2d\u6587\u7684\u77ed\u5f71\u97f3"
    "\u8a9e\u97f3\u5167\u5bb9\uff0c\u8acb\u5305\u542b\u6b63\u78ba"
    "\u7684\u6a19\u9ede\u7b26\u865f\u3002"
)


def newest_audio_file(audio_dir: Path) -> Path:
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio directory not found: {audio_dir}")
    candidates = [
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"no supported non-empty audio file found in: {audio_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def srt_timestamp(seconds: float) -> str:
    total_ms = int(datetime.timedelta(seconds=max(float(seconds), 0.0)).total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def compose_srt(subtitles: list[dict]) -> str:
    blocks = []
    for idx, item in enumerate(subtitles, 1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"{srt_timestamp(item['start'])} --> {srt_timestamp(item['end'])}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks).strip() + "\n"


def subtitles_from_words(result: dict, soft_limit: int) -> list[dict]:
    subtitles = []
    for segment in result.get("segments") or []:
        words = segment.get("words") or []
        if not words:
            text = str(segment.get("text", "")).strip()
            if text:
                subtitles.append(
                    {
                        "start": float(segment.get("start", 0.0) or 0.0),
                        "end": float(segment.get("end", 0.0) or 0.0),
                        "text": text,
                    }
                )
            continue

        current_text = ""
        start_time = None
        for idx, word_info in enumerate(words):
            if start_time is None:
                start_time = float(word_info.get("start", segment.get("start", 0.0)) or 0.0)
            clean_word = str(word_info.get("word", "")).strip()
            current_text += clean_word
            has_punctuation = any(p in clean_word for p in PUNCTUATIONS)
            is_last = idx == len(words) - 1
            if has_punctuation or len(current_text) >= soft_limit or is_last:
                end_time = float(word_info.get("end", segment.get("end", start_time)) or start_time)
                if current_text.strip():
                    subtitles.append({"start": start_time, "end": end_time, "text": current_text.strip()})
                current_text = ""
                start_time = None
    return subtitles


def transcribe_to_srt(
    ep_dir: Path,
    audio_path: Path | None,
    output_srt: Path,
    model_size: str,
    language: str,
    initial_prompt: str,
    soft_limit: int,
) -> Path:
    load_dotenv()
    selected_audio = audio_path or newest_audio_file(ep_dir / "01_audio")
    if not selected_audio.exists():
        raise FileNotFoundError(f"audio file not found: {selected_audio}")
    if selected_audio.stat().st_size == 0:
        raise RuntimeError(f"audio file is empty: {selected_audio}")

    print(f"[INFO] local_whisper_model={model_size}")
    print(f"[INFO] language={language or 'auto'}")
    print(f"[INFO] input_audio={selected_audio}")
    print(f"[INFO] output_srt={output_srt}")

    model = whisper.load_model(model_size)
    transcribe_kwargs = {
        "verbose": False,
        "initial_prompt": initial_prompt,
        "word_timestamps": True,
    }
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(str(selected_audio), **transcribe_kwargs)
    subtitles = subtitles_from_words(result, soft_limit=max(int(soft_limit), 6))
    if not subtitles:
        raise RuntimeError("local Whisper returned no subtitle segments")

    srt_text = compose_srt(subtitles)
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text(srt_text, encoding="utf-8")
    print(f"[INFO] subtitle_count={len(subtitles)}")
    print(f"[INFO] written_bytes={output_srt.stat().st_size}")
    return output_srt


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator step 3: transcribe uploaded audio to SRT with local Whisper.")
    parser.add_argument("--ep-dir", required=True, help="Short episode directory, e.g. workspaces/short_generator/show/Ep01")
    parser.add_argument("--audio", default="", help="Optional explicit audio file path. Defaults to newest file in 01_audio.")
    parser.add_argument("--output", default="", help="Optional SRT output path. Defaults to 02_subtitles/whisper.srt.")
    parser.add_argument("--model", default=os.getenv("CAP_SHORT_WHISPER_MODEL", "medium"), help="Local Whisper model size.")
    parser.add_argument("--language", default=os.getenv("CAP_SHORT_WHISPER_LANGUAGE", "zh"), help="Whisper language code. Empty means auto.")
    parser.add_argument("--initial-prompt", default=os.getenv("CAP_SHORT_WHISPER_PROMPT", DEFAULT_INITIAL_PROMPT))
    parser.add_argument("--soft-limit", type=int, default=int(os.getenv("CAP_SHORT_SUBTITLE_SOFT_LIMIT", "18")))
    args = parser.parse_args()

    ep_dir = Path(args.ep_dir).resolve()
    audio_path = Path(args.audio).resolve() if args.audio else None
    output_srt = Path(args.output).resolve() if args.output else ep_dir / "02_subtitles" / "whisper.srt"
    transcribe_to_srt(
        ep_dir=ep_dir,
        audio_path=audio_path,
        output_srt=output_srt,
        model_size=args.model.strip(),
        language=args.language.strip(),
        initial_prompt=args.initial_prompt.strip(),
        soft_limit=args.soft_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
