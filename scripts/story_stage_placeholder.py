import argparse
import os
from pathlib import Path


OUTPUTS = {
    ("2", "outline"): "02_story/story_outline.md",
    ("2", "script"): "02_story/story_script.md",
    ("3", "characters"): "03_characters_scenes/characters.json",
    ("3", "locations"): "03_characters_scenes/locations.json",
    ("4", "tts_lines"): "04_audio_subtitles/tts_lines.csv",
    ("4", "tts_audio"): "04_audio_subtitles/voice_segments/line_001.m4a",
    ("4", "merged_audio"): "04_audio_subtitles/story_audio.m4a",
    ("4", "subtitles"): "04_audio_subtitles/story_subtitles.srt",
    ("5", "storyboard"): "05_storyboards/storyboard.csv",
    ("5", "images"): "05_storyboards/images/scene_001.png",
    ("5", "animations"): "05_storyboards/animations/scene_001.mp4",
    ("6", "final_video"): "06_video/final_video.mp4",
    ("7", "cover"): "07_publish/cover.png",
    ("7", "youtube_metadata"): "07_publish/youtube_meta.json",
}


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    rel = OUTPUTS.get((str(args.stage), args.item))
    if rel:
        out = ep_path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in {".json"}:
            out.write_text("{}\n", encoding="utf-8")
        elif out.suffix.lower() == ".csv":
            out.write_text("id,type,content\n", encoding="utf-8")
        else:
            out.write_text(f"Placeholder for story stage {args.stage}: {args.item}\n", encoding="utf-8")
    print(f"story placeholder stage={args.stage} item={args.item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
