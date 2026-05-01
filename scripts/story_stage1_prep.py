import argparse
import os
from pathlib import Path


STORY_DIRS = [
    "00_logs",
    "01_preproduction",
    "02_story",
    "03_characters_scenes",
    "04_audio_subtitles",
    "04_audio_subtitles/voice_segments",
    "05_storyboards",
    "05_storyboards/images",
    "05_storyboards/animations",
    "06_video",
    "07_publish",
]


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    existing = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if existing:
        return existing
    return workspace_root / f"Ep{ep_num:02d}_0000_0000"


def ensure_story_dirs(ep_path: Path) -> None:
    ep_path.mkdir(parents=True, exist_ok=True)
    for rel in STORY_DIRS:
        (ep_path / rel).mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 1.1: create episode folders.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    ep_path = find_episode_path(workspace_root, args.ep)
    ensure_story_dirs(ep_path)
    print(f"story_stage1_prep ep={args.ep} path={ep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
