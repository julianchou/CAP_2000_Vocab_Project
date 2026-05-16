import argparse
import csv
import sys
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from generate_animation_prompt import generate_animation_prompt
from generate_scene_animation import (
    SceneAnimationError,
    animation_dir_for_episode,
    find_episode_folder,
    generate_scene_animation,
    read_storyboard_rows,
    storyboard_csv_for_episode,
)
from app_utils.asset_tags import tag_asset_from_row


def write_storyboard_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    extra_cols = ["animation_prompt", "animation_video_path"]
    final_fieldnames = list(fieldnames)
    for col in extra_cols:
        if col not in final_fieldnames:
            final_fieldnames.append(col)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scene_animation_candidate_paths(episode_folder: Path, row: dict) -> list[Path]:
    scene_id = str(row.get("scene_id", "")).strip()
    start_token = str(row.get("start_time", "")).strip()
    animation_dir = animation_dir_for_episode(episode_folder, storyboard_csv_for_episode(episode_folder))

    custom_animation_path = str(row.get("animation_video_path", "")).strip()
    candidates: list[Path] = []
    if custom_animation_path:
        candidates.append(Path(custom_animation_path))

    if scene_id:
        base_path = animation_dir / f"scene_{scene_id}.mp4"
    elif start_token:
        base_path = animation_dir / f"scene_{start_token}.mp4"
    else:
        return []

    candidates.append(base_path)
    candidates.extend(sorted(base_path.parent.glob(f"{base_path.stem}__v*{base_path.suffix}")))

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            out.append(path)
    return out


def scene_uses_animation(episode_folder: Path, row: dict) -> tuple[bool, Path | None]:
    candidates = scene_animation_candidate_paths(episode_folder, row)
    if not candidates:
        return False, None

    custom_animation_path = str(row.get("animation_video_path", "")).strip()
    if custom_animation_path:
        custom_path = Path(custom_animation_path)
        if custom_path.exists():
            return True, custom_path

    newest_path = max(candidates, key=lambda p: p.stat().st_mtime)
    return True, newest_path


def set_scene_animation_default(storyboard_csv: Path, scene_id: str, animation_path: Path) -> bool:
    rows, fieldnames = read_storyboard_rows(storyboard_csv)
    target_path = str(animation_path).replace("\\", "/")
    changed = False

    for row in rows:
        if str(row.get("scene_id", "")).strip() != str(scene_id).strip():
            continue
        if str(row.get("animation_video_path", "")).strip() == target_path:
            continue
        row["animation_video_path"] = target_path
        changed = True

    if changed:
        write_storyboard_rows(storyboard_csv, rows, fieldnames)
    return changed


def collect_ai_scene_rows(ep_num: int) -> tuple[Path, Path, list[dict]]:
    episode_folder = find_episode_folder(ep_num)
    if not episode_folder:
        raise FileNotFoundError(f"episode folder not found for ep {ep_num:02d}")

    storyboard_csv = storyboard_csv_for_episode(episode_folder)
    if not storyboard_csv.exists():
        raise FileNotFoundError(f"missing storyboard.csv: {storyboard_csv}")

    rows, _fieldnames = read_storyboard_rows(storyboard_csv)
    scene_rows: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("source_type", "")).strip().upper() != "AI":
            continue
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id or scene_id in seen:
            continue
        scene_rows.append(dict(row))
        seen.add(scene_id)

    return episode_folder, storyboard_csv, scene_rows


def summarize_batch_status(
    *,
    failed_count: int,
    animation_success_count: int,
    default_updated_count: int,
    skipped_count: int,
) -> str:
    if failed_count == 0:
        return "done"
    usable_count = animation_success_count + default_updated_count + skipped_count
    if usable_count > 0:
        return "partial"
    return "error"


def run_batch_animation_for_episode(ep_num: int) -> str:
    episode_folder, storyboard_csv, scene_rows = collect_ai_scene_rows(ep_num)

    print("===== BATCH ANIMATION START =====")
    print(f"episode={ep_num}")
    print(f"episode_folder={episode_folder}")
    print(f"storyboard_csv={storyboard_csv}")
    print(f"ai_scene_count={len(scene_rows)}")

    if not scene_rows:
        print("No AI scenes found. Nothing to do.")
        print("batch_status=done")
        print("===== BATCH ANIMATION END =====")
        return "done"

    prompt_success_ids: list[str] = []
    animation_success_ids: list[str] = []
    default_updated_ids: list[str] = []
    skipped_ids: list[str] = []
    failed: list[tuple[str, str]] = []

    for idx, row in enumerate(scene_rows, start=1):
        scene_id = str(row.get("scene_id", "")).strip()
        print(f"\n===== BATCH SCENE {idx}/{len(scene_rows)} | scene_id={scene_id} =====")
        try:
            uses_animation, animation_path = scene_uses_animation(episode_folder, row)
            if uses_animation and animation_path:
                tag_asset_from_row(animation_path, row, asset_kind="animation", overwrite=False)
                updated = set_scene_animation_default(storyboard_csv, scene_id, animation_path)
                row["animation_video_path"] = str(animation_path).replace("\\", "/")
                if updated:
                    default_updated_ids.append(scene_id)
                    print(
                        f"DEFAULT RESULT scene_id={scene_id} status=updated "
                        f"path={animation_path}"
                    )
                skipped_ids.append(scene_id)
                print(
                    f"SCENE RESULT scene_id={scene_id} status=skip "
                    f"reason=already_using_animation path={animation_path}"
                )
                continue

            print(f"STEP A generate animation prompt scene_id={scene_id}")
            generate_animation_prompt(ep_num, scene_id)
            prompt_success_ids.append(scene_id)
            print(f"PROMPT RESULT scene_id={scene_id} status=ok")

            print(f"STEP B generate animation video scene_id={scene_id}")
            output_path = generate_scene_animation(ep_num, scene_id)
            animation_success_ids.append(scene_id)
            row["animation_video_path"] = str(output_path).replace("\\", "/")
            if set_scene_animation_default(storyboard_csv, scene_id, output_path):
                default_updated_ids.append(scene_id)
                print(f"DEFAULT RESULT scene_id={scene_id} status=updated path={output_path}")
            print(f"SCENE RESULT scene_id={scene_id} status=ok path={output_path}")
        except (SceneAnimationError, FileNotFoundError, ValueError) as e:
            failed.append((scene_id, str(e)))
            print(f"SCENE RESULT scene_id={scene_id} status=error error={e}")
            traceback.print_exc()
        except Exception as e:
            failed.append((scene_id, f"unexpected error: {e}"))
            print(f"SCENE RESULT scene_id={scene_id} status=error error={e}")
            traceback.print_exc()

    print("\n===== BATCH SUMMARY =====")
    print(f"total={len(scene_rows)}")
    print(f"prompt_success={len(prompt_success_ids)}")
    print(f"animation_success={len(animation_success_ids)}")
    print(f"default_updated={len(default_updated_ids)}")
    print(f"skipped={len(skipped_ids)}")
    print(f"failed={len(failed)}")
    if prompt_success_ids:
        print(f"prompt_success_scene_ids={','.join(prompt_success_ids)}")
    if animation_success_ids:
        print(f"animation_success_scene_ids={','.join(animation_success_ids)}")
    if default_updated_ids:
        print(f"default_updated_scene_ids={','.join(default_updated_ids)}")
    if skipped_ids:
        print(f"skipped_scene_ids={','.join(skipped_ids)}")
    batch_status = summarize_batch_status(
        failed_count=len(failed),
        animation_success_count=len(animation_success_ids),
        default_updated_count=len(default_updated_ids),
        skipped_count=len(skipped_ids),
    )
    if failed and batch_status == "partial":
        for scene_id, message in failed:
            print(f"failed_scene scene_id={scene_id} message={message}")
        print("Batch completed partially. Successful scenes were kept as default animations; some scenes still need manual review.")
    elif failed:
        for scene_id, message in failed:
            print(f"failed_scene scene_id={scene_id} message={message}")
        print("Batch completed with failures. Prompt generation and animation generation continued for later scenes.")
    else:
        print("Batch completed successfully. Existing animation scenes were skipped; remaining prompts were saved, animations were generated, and successful scenes were switched to default animation.")
    print(f"batch_status={batch_status}")
    print("===== BATCH ANIMATION END =====")
    return batch_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate animations for all AI scenes in one episode")
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()

    status = run_batch_animation_for_episode(args.ep)
    if status == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
