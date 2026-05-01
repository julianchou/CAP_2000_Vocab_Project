import argparse
import json
import os
from pathlib import Path


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def load_options(ep_path: Path) -> list[dict]:
    path = ep_path / "01_preproduction" / "story_outline_options.json"
    if not path.exists():
        path = ep_path / "01_preproduction" / "story_brainstorm_options.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("options") or [])
    return list(data or [])


def select_option(ep_path: Path, option_id: str) -> dict:
    options = load_options(ep_path)
    selected = next((item for item in options if str(item.get("id")) == str(option_id)), None)
    if selected is None:
        raise ValueError(f"story option not found: {option_id}")

    out_json = ep_path / "01_preproduction" / "selected_story.json"
    out_md = ep_path / "01_preproduction" / "selected_story.md"
    out_json.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                f"# {selected.get('title', '')}",
                "",
                f"- 方案：{selected.get('id', '')}",
                f"- 類型：{selected.get('genre', '')}",
                f"- 目標年齡：{selected.get('target_age', '')}",
                f"- 故事調性：{selected.get('tone', '')}",
                f"- 故事一句話：{selected.get('logline', '')}",
                f"- 場景：{selected.get('setting', '')}",
                f"- 核心衝突：{selected.get('conflict', '')}",
                f"- 正向寓意：{selected.get('moral', '')}",
                f"- 視覺方向：{selected.get('visual_direction', '')}",
                "",
                "## 三幕大綱",
                *[f"{idx}. {text}" for idx, text in enumerate(selected.get("three_act_outline") or [], start=1)],
                "",
                "## 兒童適齡規則",
                *[f"- {text}" for text in selected.get("safety_rules") or []],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"selected story option {option_id}: {out_json}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode: select one generated outline option.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--option-id", required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    select_option(ep_path, args.option_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
