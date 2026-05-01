import argparse
import sys
from pathlib import Path


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stages-path", default="")
    parser.add_argument("--profile-id", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app_utils.pipeline import StageRunner

    ep_path = find_episode_path(workspace_root, args.ep)
    ep_info = {
        "ep": args.ep,
        "path": ep_path,
    }
    name_parts = ep_path.name.split("_")
    if len(name_parts) >= 3:
        try:
            ep_info["start"] = int(name_parts[1])
            ep_info["end"] = int(name_parts[2])
        except ValueError:
            pass

    stages_path = Path(args.stages_path) if args.stages_path else None
    runner = StageRunner(root, stages_path=stages_path, workspace_root=workspace_root, profile_id=args.profile_id)
    status = runner.run_stage(ep_info, args.stage)
    return 0 if status.get("stages", {}).get(str(args.stage)) == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
