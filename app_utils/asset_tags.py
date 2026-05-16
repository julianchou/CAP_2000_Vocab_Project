import csv
import json
import re
from datetime import datetime
from pathlib import Path

from .filesystem import parse_episode_info, storyboard_csv_path


ASSET_TAG_FILTER_ALL = "__all__"
ASSET_TAG_FILTER_UNTAGGED = "__untagged__"

GENERIC_TAG_FALLBACK = "Scene"
PROMPT_SPLIT_MARKERS = [
    "aspect ratio",
    "cinematic wide shot",
    "pixar-like",
    "soft studio lighting",
]
TAG_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "showing",
    "featuring",
    "depicting",
    "scene",
    "background",
    "animation",
    "animated",
    "educational",
    "cinematic",
    "wide",
    "shot",
    "aspect",
    "ratio",
    "soft",
    "gentle",
    "calm",
    "friendly",
    "light",
    "studio",
    "lighting",
    "small",
    "subtle",
    "slow",
    "smooth",
    "original",
    "subjects",
    "layout",
    "props",
    "colors",
    "maintain",
    "keep",
    "unchanged",
}


def episode_path_for_asset(asset_path: Path) -> Path | None:
    for parent in Path(asset_path).parents:
        info = parse_episode_info(parent)
        if info.get("ep") is not None:
            return parent
        if re.match(r"^Ep\d+$", parent.name):
            return parent
    return None


def asset_tags_json_path(ep_path: Path) -> Path:
    return ep_path / "04_images" / "asset_tags.json"


def asset_tag_storage_key(asset_path: Path, ep_path: Path | None = None) -> str:
    path = Path(asset_path)
    episode_path = ep_path or episode_path_for_asset(path)
    if episode_path:
        try:
            return path.resolve().relative_to(episode_path.resolve()).as_posix()
        except Exception:
            pass
    return str(path).replace("\\", "/")


def read_episode_asset_tags(ep_path: Path) -> dict[str, str]:
    path = asset_tags_json_path(ep_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("items"), dict):
        raw = raw.get("items") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        tag_text = ""
        if isinstance(value, str):
            tag_text = value.strip()
        elif isinstance(value, dict):
            tag_text = str(value.get("tag", "")).strip()
        if tag_text:
            out[str(key)] = tag_text
    return out


def write_episode_asset_tags(ep_path: Path, tags: dict[str, str]) -> Path:
    path = asset_tags_json_path(ep_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "items": dict(sorted(tags.items())),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def asset_tag_for_path(asset_path: Path | str | None) -> str:
    if not asset_path:
        return ""
    path = Path(asset_path)
    episode_path = episode_path_for_asset(path)
    if not episode_path:
        return ""
    tags = read_episode_asset_tags(episode_path)
    return str(tags.get(asset_tag_storage_key(path, episode_path), "")).strip()


def set_asset_tag_for_path(asset_path: Path | str, tag_text: str) -> Path | None:
    path = Path(asset_path)
    episode_path = episode_path_for_asset(path)
    if not episode_path:
        return None
    tags = read_episode_asset_tags(episode_path)
    storage_key = asset_tag_storage_key(path, episode_path)
    normalized_tag = str(tag_text or "").strip()
    if normalized_tag:
        tags[storage_key] = normalized_tag
    else:
        tags.pop(storage_key, None)
    return write_episode_asset_tags(episode_path, tags)


def collect_asset_tag_options(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    has_untagged = False
    for path in paths:
        tag_text = asset_tag_for_path(path)
        if tag_text:
            seen.add(tag_text)
        else:
            has_untagged = True
    options = [ASSET_TAG_FILTER_ALL]
    if has_untagged:
        options.append(ASSET_TAG_FILTER_UNTAGGED)
    options.extend(sorted(seen))
    return options


def filter_asset_paths_by_tag(paths: list[Path], selected_tag: str) -> list[Path]:
    if not selected_tag or selected_tag == ASSET_TAG_FILTER_ALL:
        return list(paths)
    if selected_tag == ASSET_TAG_FILTER_UNTAGGED:
        return [path for path in paths if not asset_tag_for_path(path)]
    return [path for path in paths if asset_tag_for_path(path) == selected_tag]


def asset_tag_badge(path: Path | None) -> str:
    tag_text = asset_tag_for_path(path) if path else ""
    return tag_text or "未設定"


def normalize_prompt_text(text: str) -> str:
    cleaned = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.;:-")


def _titlecase_ascii(words: list[str]) -> str:
    output = []
    for word in words:
        if re.search(r"[A-Za-z]", word):
            output.append(word.capitalize())
        else:
            output.append(word)
    return " ".join(output).strip()


def derive_tag_from_prompt(prompt_text: str, fallback: str = GENERIC_TAG_FALLBACK) -> str:
    prompt = normalize_prompt_text(prompt_text)
    if not prompt:
        return str(fallback or GENERIC_TAG_FALLBACK).strip()

    lowered = prompt.lower()
    cut_at = len(prompt)
    for marker in PROMPT_SPLIT_MARKERS:
        idx = lowered.find(marker)
        if idx >= 0:
            cut_at = min(cut_at, idx)
    prompt = normalize_prompt_text(prompt[:cut_at])

    segments = [normalize_prompt_text(part) for part in re.split(r"[,;|/]", prompt) if normalize_prompt_text(part)]
    if not segments:
        segments = [prompt]

    for segment in segments:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", segment)
        filtered = [token for token in tokens if token.lower() not in TAG_STOPWORDS]
        if filtered:
            return _titlecase_ascii(filtered[:4])

    for segment in segments:
        trimmed = normalize_prompt_text(segment)
        if trimmed:
            words = trimmed.split()
            return _titlecase_ascii(words[:4]) or str(fallback or GENERIC_TAG_FALLBACK).strip()

    return str(fallback or GENERIC_TAG_FALLBACK).strip()


def derive_asset_tag_from_row(row: dict, asset_kind: str) -> str:
    source_type = str(row.get("source_type", "")).strip().upper()
    flashcard_word = str(row.get("flashcard_word", "")).strip()
    if source_type == "FLASHCARD" and flashcard_word:
        return flashcard_word

    prompt_candidates: list[str] = []
    if asset_kind == "animation":
        prompt_candidates.append(str(row.get("animation_prompt", "")).strip())
    prompt_candidates.append(str(row.get("image_prompt", "")).strip())
    prompt_candidates.append(str(row.get("subtitle_reference", "")).strip())
    prompt_candidates.append(str(row.get("reason", "")).strip())
    if flashcard_word:
        prompt_candidates.append(flashcard_word)

    for prompt_text in prompt_candidates:
        if prompt_text:
            return derive_tag_from_prompt(prompt_text, fallback=GENERIC_TAG_FALLBACK)

    scene_id = str(row.get("scene_id", "")).strip()
    if scene_id:
        return f"Scene {scene_id}"
    return GENERIC_TAG_FALLBACK


def scene_image_asset_paths(ep_path: Path, row: dict) -> list[Path]:
    candidates: list[Path] = []
    source_type = str(row.get("source_type", "")).strip().upper()
    custom_image_path = str(row.get("custom_image_path", "")).strip()
    if custom_image_path:
        candidates.append(Path(custom_image_path))

    if source_type == "FLASHCARD":
        word = str(row.get("flashcard_word", "")).strip()
        if word:
            candidates.append(ep_path / "04_images" / "flashcards" / f"{word}.png")
        return _existing_unique_paths(candidates)

    start_token = str(row.get("start_time", "")).strip()
    if start_token:
        base_path = ep_path / "04_images" / "ai_generated" / f"img_{start_token}.png"
        candidates.append(base_path)
        candidates.extend(sorted(base_path.parent.glob(f"{base_path.stem}__v*{base_path.suffix}")))
    return _existing_unique_paths(candidates)


def scene_animation_asset_paths(ep_path: Path, row: dict) -> list[Path]:
    candidates: list[Path] = []
    custom_animation_path = str(row.get("animation_video_path", "")).strip()
    if custom_animation_path:
        candidates.append(Path(custom_animation_path))

    scene_id = str(row.get("scene_id", "")).strip()
    start_token = str(row.get("start_time", "")).strip()
    if scene_id:
        base_path = ep_path / "04_images" / "animations" / f"scene_{scene_id}.mp4"
    elif start_token:
        base_path = ep_path / "04_images" / "animations" / f"scene_{start_token}.mp4"
    else:
        return _existing_unique_paths(candidates)

    candidates.append(base_path)
    candidates.extend(sorted(base_path.parent.glob(f"{base_path.stem}__v*{base_path.suffix}")))
    return _existing_unique_paths(candidates)


def _existing_unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path).replace("\\", "/")
        if key in seen or not path.exists():
            continue
        seen.add(key)
        out.append(path)
    return out


def tag_asset_from_row(asset_path: Path | str, row: dict, asset_kind: str, overwrite: bool = True) -> bool:
    path = Path(asset_path)
    if not path.exists():
        return False
    tag_text = derive_asset_tag_from_row(row, asset_kind=asset_kind)
    if not tag_text:
        return False
    current_tag = asset_tag_for_path(path)
    if current_tag and not overwrite:
        return False
    set_asset_tag_for_path(path, tag_text)
    return current_tag != tag_text


def sync_episode_asset_tags_from_storyboard(ep_path: Path, overwrite: bool = False) -> dict[str, int]:
    storyboard_path = storyboard_csv_path(ep_path)
    stats = {
        "scene_rows": 0,
        "image_assets_seen": 0,
        "image_assets_tagged": 0,
        "animation_assets_seen": 0,
        "animation_assets_tagged": 0,
    }
    if not storyboard_path.exists():
        return stats

    with storyboard_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    stats["scene_rows"] = len(rows)
    for row in rows:
        for image_path in scene_image_asset_paths(ep_path, row):
            stats["image_assets_seen"] += 1
            if tag_asset_from_row(image_path, row, asset_kind="image", overwrite=overwrite):
                stats["image_assets_tagged"] += 1
        for animation_path in scene_animation_asset_paths(ep_path, row):
            stats["animation_assets_seen"] += 1
            if tag_asset_from_row(animation_path, row, asset_kind="animation", overwrite=overwrite):
                stats["animation_assets_tagged"] += 1
    return stats
