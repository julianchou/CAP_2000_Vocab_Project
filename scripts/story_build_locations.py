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
    return text or "children story"


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


def paragraph_text(paragraph: dict) -> str:
    parts = [
        paragraph.get("title", ""),
        paragraph.get("source_plot", ""),
        paragraph.get("storybook_text", ""),
        " ".join(paragraph.get("narration") or []),
    ]
    return " ".join(str(part or "") for part in parts)


def default_location_blueprint(index: int, story_setting: str, paragraph: dict) -> dict:
    text = paragraph_text(paragraph)
    if index == 1:
        return {
            "name": story_setting or "會發光的餅乾森林、柔軟草地、銀色月光小路",
            "visual_anchor": "glowing cookie forest, soft grass, silver moonlit path, cozy rabbit home nearby",
            "layout": "wide readable forest establishing shot with a clear path leading from the rabbit home into the glowing cookie trees",
            "key_props": ["glowing cookie trees", "soft green grass", "silver moonlit path", "cozy round rabbit home", "gentle morning light"],
            "palette": ["warm cream", "soft green", "moonlight silver", "pastel yellow"],
            "mood": "warm, magical, safe, inviting",
        }
    if index == 2:
        return {
            "name": "圓圓大樹與低垂小花的線索小路",
            "visual_anchor": "round tree, drooping flowers, dim glowing pebbles, gentle mystery path",
            "layout": "characters walk along a curved path toward a large round tree; clues are visible but not scary",
            "key_props": ["round tree", "drooping flowers", "dim glowing pebbles", "small safe clues", "soft bushes"],
            "palette": ["moss green", "warm brown", "soft beige", "gentle yellow"],
            "mood": "curious, gentle mystery, child-safe",
        }
    if index == 3:
        return {
            "name": "分享月亮餅乾的朋友空地",
            "visual_anchor": "friendly clearing, teamwork space, cookie table or basket, hopeful light",
            "layout": "open clearing with enough space for characters to gather, share, and solve the problem together",
            "key_props": ["friendly clearing", "small cookie basket", "sparkling pebbles", "teamwork space", "hopeful light"],
            "palette": ["pastel yellow", "soft orange", "warm cream", "gentle blue"],
            "mood": "cooperative, hopeful, warm",
        }
    return {
        "name": "溫柔回家路與重新發光的森林",
        "visual_anchor": "glowing corners, honey sunset, peaceful path home, tiny star-like lights",
        "layout": "wide peaceful ending shot with the path returning home and the forest glow restored",
        "key_props": ["honey sunset", "glowing cookie trees", "peaceful path home", "tiny star-like lights", "soft flowers"],
        "palette": ["honey gold", "moonlight yellow", "soft green", "warm cream"],
        "mood": "relieved, grateful, peaceful",
    }


def prompt_for_location(location: dict, story_title: str, characters: list[str]) -> str:
    character_text = ", ".join(characters[:5]) if characters else "cute fairy-tale characters"
    props = ", ".join(location["key_props"])
    palette = ", ".join(location["palette"])
    return (
        f"Fixed background reference for the story '{story_title}'. "
        f"Location name: {location['name']}. "
        f"Visual anchor that must stay consistent: {location['visual_anchor']}. "
        f"Stable layout: {location['layout']}. "
        f"Key props that should recur when this location appears: {props}. "
        f"Fixed color palette: {palette}. "
        f"Mood: {location['mood']}. "
        f"Designed to host these characters without redesigning them: {character_text}. "
        "Warm bright children picture-book style, cute 3D storybook animation look, soft rounded shapes, "
        "gentle lighting, simple readable composition, cinematic 16:9 wide shot. "
        "No scary elements, no violence, no danger, no clutter, no dark realism, suitable for children under 10."
    )


def build_location(index: int, paragraph: dict, story: dict, characters: list[str]) -> dict:
    story_title = clean_title(story.get("title", ""))
    story_setting = str(story.get("setting") or "").strip()
    blueprint = default_location_blueprint(index, story_setting, paragraph)
    purpose = paragraph.get("purpose") or paragraph.get("title") or f"story paragraph {index}"
    return {
        "location_id": location_id(index, blueprint["name"]),
        "name": blueprint["name"],
        "story_role": purpose,
        "target_age_suitability": "under 10",
        "linked_paragraph_ids": [str(paragraph.get("paragraph_id") or index)],
        "linked_characters": characters,
        "visual_design": {
            "style": "children picture-book / cute 3D fairy-tale animation background",
            "visual_anchor": blueprint["visual_anchor"],
            "stable_layout": blueprint["layout"],
            "mood": blueprint["mood"],
            "time_of_day": "soft storybook daylight, moonlight glow, or honey sunset depending on scene timing",
            "color_palette": blueprint["palette"],
            "shape_language": "rounded, soft, readable, welcoming; no sharp threatening silhouettes",
            "key_props": blueprint["key_props"],
            "composition_notes": (
                "Keep a clear foreground/midground/background. Leave open space for characters. "
                "Do not overcrowd the frame. Maintain the same landmark shapes and color palette when reused."
            ),
            "continuity_tags": [
                f"same location {blueprint['name']}",
                blueprint["visual_anchor"],
                "child-safe background",
                "soft pastel storybook style",
                "warm gentle lighting",
                "cinematic 16:9 wide composition",
            ],
        },
        "image_generation": {
            "prompt_en": prompt_for_location(blueprint, story_title, characters),
            "negative_prompt_en": (
                "different location design, changed landmark layout, scary, horror, violence, danger, dark realism, "
                "gloomy, weapon, monster, sharp threatening shapes, cluttered composition, photorealistic adult style, low quality"
            ),
            "recommended_aspect_ratio": "16:9",
            "reference_usage": (
                "Use prompt_en and visual_design as the location identity source for storyboard image prompts. "
                "Scene prompts may change camera angle, character action, and lighting intensity, but should preserve landmarks, palette, and layout."
            ),
        },
    }


def build_locations(ep_path: Path) -> dict:
    story = read_json(ep_path / "01_preproduction" / "selected_story.json")
    outline = read_json(ep_path / "02_story" / "story_outline.json")
    characters_path = ep_path / "03_characters_scenes" / "characters.json"
    characters_data = read_json(characters_path) if characters_path.exists() else {}
    names = character_names(characters_data)
    paragraphs = outline.get("paragraphs") or []
    if not paragraphs:
        paragraphs = [{"paragraph_id": str(i), "title": f"story segment {i}"} for i in range(1, 5)]

    locations = [
        build_location(idx, paragraph, story, names)
        for idx, paragraph in enumerate(paragraphs[:4], start=1)
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_story_id": story.get("id", ""),
        "story_title": story.get("title", ""),
        "target_age": story.get("target_age", "under 10"),
        "scene_world": {
            "global_style": "children picture-book, cute 3D fairy-tale animation, warm, bright, rounded, child-safe",
            "safety": "no horror, no violence, no weapons, no danger, no dark realism, suitable for children under 10",
            "continuity_rule": (
                "Every scene image prompt should preserve location_id, name, visual_anchor, stable_layout, key_props, "
                "color_palette, and continuity_tags. Only camera framing, character action, and moment-specific lighting may change."
            ),
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
