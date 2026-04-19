import argparse
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from episode_range_utils import find_episode_folder


load_dotenv()
base_dir = Path(__file__).resolve().parent.parent
core_assets_dir = base_dir / "core" / "assets"
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
TEMPLATE_FILENAME = "flashcard_template.png"


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
    base_img = Image.open(core_assets_dir / TEMPLATE_FILENAME).convert("RGB")
    img_w, img_h = base_img.size

    box_y1 = int(img_h * 0.28)
    box_h = int(img_h * 0.45)
    box_bottom = box_y1 + box_h
    center_x = img_w // 2
    max_text_width = int(img_w * 0.9)

    filter_set = {w.lower() for w in filter_words} if filter_words else None
    output_dir = target_folder / "04_images" / "flashcards"
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        word = str(row["Word"]).strip()
        if not word:
            continue
        if filter_set and word.lower() not in filter_set:
            continue

        print(f"👉 產出完整圖卡: {word}")
        img = base_img.copy()
        draw = ImageDraw.Draw(img)

        f_word = get_font(font_path, 210)
        draw_text_centered(draw, word, f_word, "#FFFFFF", box_y1 + 70, center_x)

        meaning_text = f"({row['POS']}) {row['Meaning']}"
        f_mean = get_font(font_path, 95)
        draw_text_centered(draw, meaning_text, f_mean, "#FFD700", box_y1 + 360, center_x)

        sent_text = str(row["English_Sentence"])
        f_sent = get_font_to_fit(sent_text, font_path, max_text_width, 70)
        draw_text_centered(draw, sent_text, f_sent, "#FFFFFF", box_bottom + 60, center_x)

        trans_text = str(row["Chinese_Translation"])
        f_trans = get_font_to_fit(trans_text, font_path, max_text_width, 55)
        draw_text_centered(draw, trans_text, f_trans, "#BCBCBC", box_bottom + 145, center_x)

        safe_name = word.replace(" ", "_").replace("?", "")
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
