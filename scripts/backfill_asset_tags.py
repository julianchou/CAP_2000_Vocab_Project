import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import yaml

from app_utils.asset_tags import sync_episode_asset_tags_from_storyboard
from app_utils.filesystem import list_episode_dirs


PROFILES_CFG = BASE_DIR / "config" / "profiles.yaml"


def load_profiles() -> list[dict]:
    if not PROFILES_CFG.exists():
        return []
    data = yaml.safe_load(PROFILES_CFG.read_text(encoding="utf-8")) or {}
    return list(data.get("profiles") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill image/animation asset tags from storyboard prompts.")
    parser.add_argument("--profile", action="append", default=[], help="Profile id to process. Repeatable.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tags instead of only filling missing tags.")
    args = parser.parse_args()

    profiles = load_profiles()
    if args.profile:
        wanted = set(args.profile)
        profiles = [profile for profile in profiles if str(profile.get("id")) in wanted]

    if not profiles:
        raise SystemExit("No profiles found to process.")

    total = {
        "episodes": 0,
        "scene_rows": 0,
        "image_assets_seen": 0,
        "image_assets_tagged": 0,
        "animation_assets_seen": 0,
        "animation_assets_tagged": 0,
    }

    print("===== ASSET TAG BACKFILL START =====")
    print(f"overwrite={bool(args.overwrite)}")

    for profile in profiles:
        profile_id = str(profile.get("id", "")).strip()
        workspace_rel = str(profile.get("workspace", "")).strip()
        if not profile_id or not workspace_rel:
            continue
        episodes = list_episode_dirs(BASE_DIR, workspace_rel)
        print(f"\n[profile] {profile_id} episodes={len(episodes)} workspace={workspace_rel}")
        for episode_path in episodes:
            stats = sync_episode_asset_tags_from_storyboard(episode_path, overwrite=bool(args.overwrite))
            total["episodes"] += 1
            for key in ("scene_rows", "image_assets_seen", "image_assets_tagged", "animation_assets_seen", "animation_assets_tagged"):
                total[key] += int(stats.get(key, 0))
            print(
                f"ep={episode_path.name} "
                f"scenes={stats['scene_rows']} "
                f"images={stats['image_assets_tagged']}/{stats['image_assets_seen']} "
                f"animations={stats['animation_assets_tagged']}/{stats['animation_assets_seen']}"
            )

    print("\n===== ASSET TAG BACKFILL SUMMARY =====")
    for key, value in total.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
