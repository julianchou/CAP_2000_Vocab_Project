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


def clean_title(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^Ep\s*\d+\s*", "", text, flags=re.IGNORECASE)
    return text or "童話故事"


def location_id(index: int, name: str) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "")).strip("_").lower()
    return ascii_part or f"location_{index:02d}"


def character_names(characters_data: dict) -> list[str]:
    names = []
    for item in characters_data.get("characters") or []:
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def prompt_for_location(name: str, purpose: str, visual_keywords: list[str], story_title: str, characters: list[str]) -> str:
    character_text = ", ".join(characters[:4]) if characters else "cute fairy-tale characters"
    keyword_text = ", ".join(visual_keywords)
    return (
        f"Child-safe fairy-tale background scene for the story '{story_title}'. "
        f"Location: {name}. Purpose: {purpose}. "
        f"Visual elements: {keyword_text}. "
        f"Designed for characters: {character_text}. "
        "Warm bright picture-book style, cute 3D storybook animation look, soft rounded shapes, "
        "pastel colors, gentle lighting, simple readable composition, no scary elements, "
        "suitable for children under 10, cinematic 16:9 wide shot."
    )


def build_location(index: int, name: str, paragraph: dict, story: dict, characters: list[str], visual_keywords: list[str]) -> dict:
    story_title = clean_title(story.get("title", ""))
    purpose = paragraph.get("purpose") or paragraph.get("title") or "童話故事場景"
    return {
        "location_id": location_id(index, name),
        "name": name,
        "story_role": purpose,
        "target_age_suitability": "10歲以下",
        "linked_paragraph_ids": [str(paragraph.get("paragraph_id") or index)],
        "linked_characters": characters,
        "visual_design": {
            "style": "兒童繪本 / 可愛 3D 童話動畫背景",
            "mood": "溫暖、明亮、安心、可愛",
            "time_of_day": "soft storybook daylight or gentle sunset",
            "color_palette": ["pastel yellow", "soft green", "warm cream", "gentle blue"],
            "shape_language": "圓潤、柔和、沒有尖銳壓迫感",
            "key_props": visual_keywords,
            "composition_notes": "保持畫面乾淨，中央保留角色活動空間，背景細節可愛但不要過度擁擠。",
            "continuity_tags": [
                "same fairy-tale world",
                "child-safe background",
                "soft pastel storybook style",
                "warm gentle lighting",
            ],
        },
        "image_generation": {
            "prompt_en": prompt_for_location(name, purpose, visual_keywords, story_title, characters),
            "negative_prompt_en": (
                "scary, horror, violence, danger, dark realism, gloomy, weapon, monster, "
                "sharp threatening shapes, cluttered composition, photorealistic adult style, low quality"
            ),
            "recommended_aspect_ratio": "16:9",
            "reference_usage": "Use this JSON as the scene/location identity source for storyboard image prompts.",
        },
    }


def build_locations(ep_path: Path) -> dict:
    story = read_json(ep_path / "01_preproduction" / "selected_story.json")
    outline = read_json(ep_path / "02_story" / "story_outline.json")
    characters_path = ep_path / "03_characters_scenes" / "characters.json"
    characters_data = read_json(characters_path) if characters_path.exists() else {}
    names = character_names(characters_data)

    base_setting = str(story.get("setting") or "明亮的童話世界")
    paragraphs = outline.get("paragraphs") or []
    defaults = [
        ("童話世界入口", ["soft morning light", "welcoming path", "cute flowers", "gentle magical glow"]),
        ("小困難發生的地方", ["round tree", "small clues", "drooping flowers", "safe gentle mystery"]),
        ("朋友合作的小路", ["friendly path", "teamwork space", "sparkling pebbles", "hopeful light"]),
        ("溫暖圓滿的回家路", ["honey sunset", "glowing corners", "peaceful path home", "tiny star-like lights"]),
    ]

    locations = []
    for idx, paragraph in enumerate(paragraphs[:4], start=1):
        default_name, keywords = defaults[idx - 1]
        if idx == 1:
            name = base_setting
        else:
            name = default_name
        locations.append(build_location(idx, name, paragraph, story, names, keywords))

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_story_id": story.get("id", ""),
        "story_title": story.get("title", ""),
        "target_age": story.get("target_age", "10歲以下"),
        "scene_world": {
            "global_style": "溫暖明亮的兒童童話繪本，可愛 3D 動畫背景",
            "safety": "場景不可恐怖、不可陰暗壓迫、不可有危險或暴力元素，適合 10 歲以下小朋友。",
            "continuity_rule": "後續分鏡產圖時，必須沿用 location_id、visual_design、continuity_tags，並搭配 characters.json 的角色設定。",
        },
        "locations": locations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 3.2: build location JSON for AI scene generation.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    output = build_locations(ep_path)
    out_dir = ep_path / "03_characters_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "locations.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"locations_json={out_path}")
    print(f"location_count={len(output.get('locations') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
