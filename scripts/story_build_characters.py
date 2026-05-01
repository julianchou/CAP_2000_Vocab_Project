import argparse
import json
import os
import re
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


def slugify_name(name: str, index: int) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "")).strip("_").lower()
    return ascii_part or f"character_{index:02d}"


def infer_role(index: int) -> str:
    if index == 1:
        return "主角"
    if index == 2:
        return "主要朋友"
    return "協助角色"


def infer_character_type(name: str, story_title: str) -> str:
    text = f"{name} {story_title}"
    if any(token in text for token in ["兔", "熊", "松鼠", "鳥", "海龜"]):
        return "可愛擬人動物"
    if any(token in text for token in ["星", "雲", "月", "花", "種子"]):
        return "溫柔奇幻角色"
    if any(token in text for token in ["船", "玩具"]):
        return "可愛擬人玩具"
    return "童話角色"


def build_visual_prompt(name: str, role: str, character_type: str, story: dict) -> str:
    title = re.sub(r"^Ep\s*\d+\s*", "", str(story.get("title", "")), flags=re.IGNORECASE).strip()
    setting = story.get("setting", "bright fairy-tale world")
    moral = story.get("moral", "")
    return (
        f"Child-safe fairy-tale character design for {name}, {role}, {character_type}. "
        f"Appears in the story '{title}', set in {setting}. "
        "Cute picture-book style, soft rounded shapes, warm friendly expression, "
        "bright pastel colors, simple readable silhouette, gentle 3D storybook animation look. "
        f"Personality should suggest the story moral: {moral}. "
        "No scary elements, no weapons, no dark realism, suitable for children under 10."
    )


def build_character(name: str, index: int, story: dict, script: dict) -> dict:
    role = infer_role(index)
    character_type = infer_character_type(name, story.get("title", ""))
    base_colors = [
        ["warm cream", "soft brown", "moonlight yellow"],
        ["moss green", "warm orange", "soft beige"],
        ["lavender", "silver", "gentle blue"],
        ["sky blue", "peach", "white"],
    ]
    color_palette = base_colors[(index - 1) % len(base_colors)]
    expressions = ["curious smile", "gentle encouragement", "thoughtful kindness", "happy relief"]
    return {
        "character_id": slugify_name(name, index),
        "name": name,
        "role": role,
        "target_age_suitability": "10歲以下",
        "character_type": character_type,
        "personality": {
            "core_traits": ["友善", "溫暖", "有好奇心"] if index == 1 else ["可靠", "會鼓勵朋友", "願意合作"],
            "emotional_arc": "從遇到小困難，到學會表達、合作與分享。",
            "voice_style": "清楚、溫柔、適合兒童故事。",
        },
        "visual_design": {
            "style": "兒童繪本 / 可愛 3D 童話動畫",
            "shape_language": "圓潤、柔和、容易辨識",
            "color_palette": color_palette,
            "default_expression": expressions[(index - 1) % len(expressions)],
            "costume_or_features": "簡單可愛，避免複雜小細節，適合多場景一致生成。",
            "consistency_tags": [
                f"same character {name}",
                "child-safe fairy tale",
                "soft rounded cute design",
                "bright pastel storybook style",
            ],
        },
        "image_generation": {
            "prompt_en": build_visual_prompt(name, role, character_type, story),
            "negative_prompt_en": (
                "scary, horror, violence, weapon, angry face, dark realism, sharp teeth, "
                "complex adult design, photorealistic, low quality, inconsistent character"
            ),
            "recommended_aspect_ratio": "1:1 for character sheet, 16:9 when placed in scenes",
            "reference_usage": "Use this JSON as the character identity source for later scene image prompts.",
        },
        "story_usage": {
            "appears_in_story": script.get("title") or story.get("title", ""),
            "relationship_to_story_moral": story.get("moral", ""),
        },
    }


def build_characters(ep_path: Path) -> dict:
    story = read_json(ep_path / "01_preproduction" / "selected_story.json")
    script_path = ep_path / "02_story" / "story_script.json"
    script = read_json(script_path) if script_path.exists() else {}
    names = [str(x).strip() for x in (story.get("main_characters") or []) if str(x).strip()]
    if not names:
        names = ["小主角", "好朋友", "溫柔的幫手"]
    characters = [build_character(name, idx, story, script) for idx, name in enumerate(names, start=1)]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_story_id": story.get("id", ""),
        "story_title": story.get("title", ""),
        "target_age": story.get("target_age", "10歲以下"),
        "visual_system": {
            "global_style": "溫暖明亮的兒童童話繪本，可愛 3D 動畫質感",
            "safety": "所有角色必須友善、不可恐怖、不可暴力、適合 10 歲以下小朋友。",
            "continuity_rule": "後續分鏡產圖時，必須沿用 character_id、name、consistency_tags、color_palette。",
        },
        "characters": characters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 3.1: build character JSON for AI image generation.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    output = build_characters(ep_path)
    out_dir = ep_path / "03_characters_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "characters.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"characters_json={out_path}")
    print(f"character_count={len(output.get('characters') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
