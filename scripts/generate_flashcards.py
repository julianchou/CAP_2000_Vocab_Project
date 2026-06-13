import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from episode_range_utils import find_episode_folder


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv()
base_dir = Path(__file__).resolve().parent.parent
core_assets_dir = base_dir / "core" / "assets"
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
TEMPLATE_FILENAME = "flashcard_template.png"


def normalize_word_key(word: str) -> str:
    return str(word or "").strip().lower()


def safe_filename_token(value: str) -> str:
    token = str(value or "").strip()
    token = token.replace(" ", "_").replace("?", "")
    token = token.replace("/", "_").replace("\\", "_")
    token = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-", "."})
    return token or "card"


def safe_pos_token(value: str) -> str:
    token = safe_filename_token(str(value or "").replace(".", ""))
    return token or "pos"


def flashcard_stems_for_df(df: pd.DataFrame) -> dict[int, str]:
    word_counts = df["Word"].astype(str).str.strip().str.lower().value_counts().to_dict()
    stems: dict[int, str] = {}
    for row_number, (_idx, row) in enumerate(df.iterrows(), start=1):
        word = str(row.get("Word", "")).strip()
        if not word:
            continue
        base_stem = safe_filename_token(word)
        if word_counts.get(normalize_word_key(word), 0) > 1:
            pos_stem = safe_pos_token(row.get("POS", ""))
            stems[row_number] = f"{base_stem}__{row_number:02d}_{pos_stem}"
        else:
            stems[row_number] = base_stem
    return stems


def find_font():
    candidates = [
        core_assets_dir / "msjhbd.ttc",
        core_assets_dir / "arialbd.ttf",
        Path("C:\\Windows\\Fonts\\msjhbd.ttc"),
        Path("C:\\Windows\\Fonts\\arialbd.ttf"),
    ]
    for path in candidates:
        if Path(path).exists():
            return str(path)
    return None


def find_phonetic_font():
    candidates = [
        core_assets_dir / "arial.ttf",
        Path("C:\\Windows\\Fonts\\arial.ttf"),
        Path("C:\\Windows\\Fonts\\seguisym.ttf"),
    ]
    for path in candidates:
        if Path(path).exists():
            return str(path)
    return find_font()


def get_font(font_path: str | None, size: int):
    if font_path:
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def get_font_to_fit(text: str, font_path: str | None, max_width: int, initial_size: int, min_size: int = 20):
    size = initial_size
    while size >= min_size:
        font = get_font(font_path, size)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
        size -= 2
    return get_font(font_path, min_size)


def draw_text_centered(draw: ImageDraw.ImageDraw, text: str, font, color: str, y_pos: int, center_x: int):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((center_x - width // 2, y_pos), text, fill=color, font=font)


def render_flashcard(
    base_img: Image.Image,
    row: pd.Series,
    font_path: str | None,
    phonetic_font_path: str | None = None,
) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    box_y1 = int(img_h * 0.28)
    box_h = int(img_h * 0.45)
    box_bottom = box_y1 + box_h
    center_x = img_w // 2
    max_text_width = int(img_w * 0.9)

    word = str(row.get("Word", "")).strip()
    phonetic = str(row.get("Phonetic", "")).strip()
    f_word = get_font_to_fit(word, font_path, max_text_width, 190, 90)
    draw_text_centered(draw, word, f_word, "#FFFFFF", box_y1 + 45, center_x)

    if phonetic:
        f_phonetic = get_font_to_fit(
            phonetic, phonetic_font_path or find_phonetic_font(), max_text_width, 64, 34
        )
        draw_text_centered(draw, phonetic, f_phonetic, "#9FE7FF", box_y1 + 285, center_x)

    meaning_text = f"({row.get('POS', '')}) {row.get('Meaning', '')}"
    f_mean = get_font_to_fit(meaning_text, font_path, max_text_width, 78, 38)
    draw_text_centered(draw, meaning_text, f_mean, "#FFD700", box_y1 + 370, center_x)

    sent_text = str(row.get("English_Sentence", ""))
    f_sent = get_font_to_fit(sent_text, font_path, max_text_width, 70)
    draw_text_centered(draw, sent_text, f_sent, "#FFFFFF", box_bottom + 60, center_x)

    trans_text = str(row.get("Chinese_Translation", ""))
    f_trans = get_font_to_fit(trans_text, font_path, max_text_width, 55)
    draw_text_centered(draw, trans_text, f_trans, "#BCBCBC", box_bottom + 145, center_x)
    return img


def load_vocab_rows(target_folder: Path) -> pd.DataFrame:
    csv_path = target_folder / "03_storyboards" / "vocab_data.csv"
    json_path = target_folder / "03_storyboards" / "vocab_data.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        raise FileNotFoundError(f"找不到單字資料：{csv_path}")
    return df.fillna("")


def process_episode(ep_num: int, filter_words=None):
    target_folder = find_episode_folder(workspace_dir, ep_num)
    if not target_folder:
        raise FileNotFoundError(f"找不到第 {ep_num:02d} 集資料夾")

    df = load_vocab_rows(target_folder)
    font_path = find_font()
    phonetic_font_path = find_phonetic_font()
    base_img = Image.open(core_assets_dir / TEMPLATE_FILENAME).convert("RGB")
    filter_set = {w.lower() for w in filter_words} if filter_words else None
    output_dir = target_folder / "04_images" / "flashcards"
    output_dir.mkdir(parents=True, exist_ok=True)

    flashcard_stems = flashcard_stems_for_df(df)
    for row_number, (_idx, row) in enumerate(df.iterrows(), start=1):
        word = str(row["Word"]).strip()
        if not word:
            continue
        if filter_set and word.lower() not in filter_set:
            continue

        print(f"👉 產出完整圖卡: {word}")
        img = render_flashcard(base_img, row, font_path, phonetic_font_path)

        safe_name = flashcard_stems[row_number]
        img.save(output_dir / f"{safe_name}.png")
        img.close()

    base_img.close()
    print("✅ 完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--words", type=str)
    args = parser.parse_args()

    f_words = [w.strip() for w in args.words.split(",") if w.strip()] if args.words else None
    target_ep = args.ep if args.ep is not None else args.start
    if target_ep is None:
        parser.error("請提供 --ep 或 --start")
    process_episode(target_ep, f_words)
