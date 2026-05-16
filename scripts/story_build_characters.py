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


def clean_title(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^Ep\s*\d+\s*", "", text, flags=re.IGNORECASE)
    return text or "children story"


def story_setting(story: dict) -> str:
    return str(story.get("setting") or "a warm bright fairy-tale world").strip()


def infer_role(index: int, name: str) -> str:
    if index == 1:
        return "主角"
    if "奶奶" in name or "爺爺" in name:
        return "智慧協助角色"
    if index == 2:
        return "主要朋友"
    return "重要配角"


def infer_character_type(name: str) -> str:
    if "兔" in name:
        return "可愛擬人小兔"
    if "松鼠" in name:
        return "可愛擬人松鼠"
    if "月亮" in name:
        return "溫柔月光長者"
    return "可愛童話角色"


def fixed_design_for(name: str, index: int) -> dict:
    if "兔" in name:
        return {
            "species_or_form": "small anthropomorphic cream rabbit",
            "fixed_body": "round head, soft rounded body, short arms and legs, small round white tail",
            "fixed_face": "black bead eyes, tiny pink triangular nose, small gentle smile, warm pink inner ears",
            "fixed_ears_hair": "two long upright rabbit ears, cream outer fur, warm pink inner ears",
            "fixed_outfit": "signature red pinafore dress over a white short-sleeve shirt whenever the body is visible",
            "color_palette": [
                "warm cream fur",
                "soft brown shadow accents",
                "warm pink inner ears",
                "tiny pink nose",
                "signature red pinafore dress",
                "white short-sleeve shirt",
            ],
            "default_expression": "curious gentle smile",
        }
    if "松鼠" in name:
        return {
            "species_or_form": "small anthropomorphic squirrel",
            "fixed_body": "round head, soft rounded body, small paws, large fluffy curled tail",
            "fixed_face": "black bead eyes, tiny brown nose, gentle encouraging smile, soft cheek blush",
            "fixed_ears_hair": "small rounded ears, tidy warm brown fur",
            "fixed_outfit": "simple moss-green vest with a tiny acorn button whenever the body is visible",
            "color_palette": [
                "warm orange-brown fur",
                "cream muzzle and belly",
                "moss green vest",
                "tiny acorn button",
                "soft beige highlights",
            ],
            "default_expression": "gentle encouragement",
        }
    if "月亮" in name:
        return {
            "species_or_form": "gentle elderly moon spirit character",
            "fixed_body": "small rounded elderly figure with moon-shaped silhouette and soft robe",
            "fixed_face": "kind crescent-moon smile, black gentle eyes, soft cheek glow",
            "fixed_ears_hair": "silver-white hair or moonlit head glow, no sharp shapes",
            "fixed_outfit": "lavender robe with silver moon trim and tiny star accents",
            "color_palette": [
                "lavender robe",
                "silver moon glow",
                "gentle blue shadows",
                "warm cream highlights",
            ],
            "default_expression": "thoughtful kindness",
        }
    palettes = [
        ["sky blue", "peach", "white"],
        ["honey yellow", "warm brown", "cream"],
        ["mint green", "soft coral", "white"],
    ]
    return {
        "species_or_form": "cute fairy-tale character",
        "fixed_body": "round head, soft rounded body, simple readable silhouette",
        "fixed_face": "black bead eyes, tiny nose, warm friendly smile",
        "fixed_ears_hair": "simple rounded shapes, no sharp details",
        "fixed_outfit": "simple child-safe storybook outfit with one signature color accent",
        "color_palette": palettes[(index - 1) % len(palettes)],
        "default_expression": "warm friendly smile",
    }


def build_visual_prompt(name: str, role: str, character_type: str, design: dict, story: dict) -> str:
    title = clean_title(story.get("title", ""))
    setting = story_setting(story)
    moral = str(story.get("moral", "")).strip()
    palette = ", ".join(design["color_palette"])
    return (
        f"Fixed character reference for {name}, {role}, {character_type}. "
        f"Appears in the story '{title}', set in {setting}. "
        f"Species/form: {design['species_or_form']}. "
        f"Body: {design['fixed_body']}. "
        f"Face: {design['fixed_face']}. "
        f"Ears/hair/head details: {design['fixed_ears_hair']}. "
        f"Outfit/features: {design['fixed_outfit']}. "
        f"Fixed color palette: {palette}. "
        "Cute 3D children storybook style, soft rounded shapes, bright pastel colors, "
        "simple readable silhouette, warm friendly expression. "
        "Keep exactly the same face shape, body proportions, colors, outfit/features, and overall style across all scenes. "
        f"Personality should support the story moral: {moral}. "
        "No scary elements, no weapons, no dark realism, suitable for children under 10."
    )


def build_character(name: str, index: int, story: dict, script: dict) -> dict:
    role = infer_role(index, name)
    character_type = infer_character_type(name)
    design = fixed_design_for(name, index)
    consistency_tags = [
        f"same character {name}",
        design["species_or_form"],
        design["fixed_face"],
        design["fixed_ears_hair"],
        design["fixed_outfit"],
        "child-safe fairy tale",
        "soft rounded cute 3D storybook style",
        "bright pastel storybook style",
    ]
    return {
        "character_id": slugify_name(name, index),
        "name": name,
        "role": role,
        "target_age_suitability": "under 10",
        "character_type": character_type,
        "personality": {
            "core_traits": ["curious", "gentle", "brave"] if index == 1 else ["kind", "supportive", "warm"],
            "emotional_arc": "starts curious or worried, learns through cooperation, ends warm and reassured",
            "voice_style": "clear, warm, child-friendly story voice",
        },
        "visual_design": {
            "style": "children picture-book / cute 3D fairy-tale animation",
            "shape_language": "rounded, soft, easy to recognize, no sharp threatening details",
            "species_or_form": design["species_or_form"],
            "fixed_body": design["fixed_body"],
            "fixed_face": design["fixed_face"],
            "fixed_ears_hair": design["fixed_ears_hair"],
            "fixed_outfit": design["fixed_outfit"],
            "color_palette": design["color_palette"],
            "default_expression": design["default_expression"],
            "costume_or_features": (
                f"Fixed design: {design['species_or_form']}; {design['fixed_body']}; "
                f"{design['fixed_face']}; {design['fixed_ears_hair']}; {design['fixed_outfit']}. "
                "Do not change these features between scenes."
            ),
            "consistency_tags": consistency_tags,
        },
        "image_generation": {
            "prompt_en": build_visual_prompt(name, role, character_type, design, story),
            "negative_prompt_en": (
                "different character design, changed outfit, changed fur color, changed face, "
                "extra limbs, scary, horror, violence, weapon, angry face, dark realism, sharp teeth, "
                "complex adult design, photorealistic, low quality, inconsistent character"
            ),
            "recommended_aspect_ratio": "1:1 for character sheet, 16:9 when placed in scenes",
            "reference_usage": (
                "Use prompt_en and visual_design as the highest-priority identity source for every later scene image prompt. "
                "Scene prompts may change pose/action/location only, not identity, palette, outfit, or proportions."
            ),
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
        names = ["小兔米米", "松鼠可可", "月亮奶奶"]
    characters = [build_character(name, idx, story, script) for idx, name in enumerate(names, start=1)]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_story_id": story.get("id", ""),
        "story_title": story.get("title", ""),
        "target_age": story.get("target_age", "under 10"),
        "visual_system": {
            "global_style": "children picture-book, cute 3D fairy-tale animation, warm, bright, rounded, child-safe",
            "safety": "no horror, no violence, no weapons, no dark realism, suitable for children under 10",
            "continuity_rule": (
                "Every image prompt must preserve character_id, name, fixed_body, fixed_face, fixed_ears_hair, "
                "fixed_outfit, color_palette, and consistency_tags. Only pose, expression intensity, action, and camera may change."
            ),
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
