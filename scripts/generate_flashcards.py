import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()
base_dir = Path(__file__).parent.parent
core_assets_dir = base_dir / "core" / "assets"
workspace_dir = base_dir / "workspace"

TEMPLATE_FILENAME = "flashcard_template.png"

def find_font():
    paths = [core_assets_dir / "arialbd.ttf", "C:\\Windows\\Fonts\\msjhbd.ttc"]
    for p in paths:
        if Path(p).exists(): return str(p)
    return None

def get_font_to_fit(text, font_path, max_width, initial_size):
    """自動縮放字體大小以符合指定寬度"""
    size = initial_size
    while size > 20:
        font = ImageFont.truetype(font_path, size)
        bbox = ImageDraw.Draw(Image.new('RGB', (1,1))).textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, 20)

def draw_text_centered(draw, text, font, color, y_pos, center_x):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((center_x - w//2, y_pos), text, fill=color, font=font)

def process_episode(ep_num, filter_words=None):
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    csv_path = target_folder / "03_storyboards" / "vocab_data.csv"
    df = pd.read_csv(csv_path)
    
    font_path = find_font()
    base_img = Image.open(core_assets_dir / TEMPLATE_FILENAME).convert("RGB")
    img_w, img_h = base_img.size
    
    # 設定繪圖區域
    box_y1, box_h = int(img_h * 0.28), int(img_h * 0.45)
    box_bottom = box_y1 + box_h
    center_x = img_w // 2
    max_text_width = int(img_w * 0.9) # 左右各留 5% 邊距

    filter_set = set(w.lower() for w in filter_words) if filter_words else None

    # 🔍 新增：確保 flashcards 子資料夾存在
    output_dir = target_folder / "04_images" / "flashcards"
    output_dir.mkdir(parents=True, exist_ok=True) # 如果資料夾不存在就建立它

    for _, row in df.iterrows():
        word = str(row['Word']).strip()
        if filter_set and word.lower() not in filter_set: continue
            
        print(f"👉 產出完整圖卡: {word}")
        img = base_img.copy()
        draw = ImageDraw.Draw(img)

        # 1. 單字 (框內上方, 固定大字)
        f_word = ImageFont.truetype(font_path, 210)
        draw_text_centered(draw, word, f_word, "#FFFFFF", box_y1 + 70, center_x)

        # 2. 詞性+中文 (框內下方)
        # 這裡會顯示如: (n. 名詞) 角度
        meaning_text = f"({row['POS']}) {row['Meaning']}"
        f_mean = ImageFont.truetype(font_path, 95)
        draw_text_centered(draw, meaning_text, f_mean, "#FFD700", box_y1 + 360, center_x)

        # --- 框外區域 (不分行，自動縮放) ---
        
        # 3. 英文例句 (框底下方 60px)
        sent_text = str(row['English_Sentence'])
        f_sent = get_font_to_fit(sent_text, font_path, max_text_width, 70)
        draw_text_centered(draw, sent_text, f_sent, "#FFFFFF", box_bottom + 60, center_x)

        # 4. 中文例句 (英文下方 85px)
        trans_text = str(row['Chinese_Translation'])
        f_trans = get_font_to_fit(trans_text, font_path, max_text_width, 55)
        draw_text_centered(draw, trans_text, f_trans, "#BCBCBC", box_bottom + 145, center_x)

        safe_name = word.replace(' ', '_').replace('?', '')
        img.save(target_folder / "04_images" / "flashcards" / f"{safe_name}.png")
        img.close()

    base_img.close()
    print(f"✅ 完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--words", type=str)
    args = parser.parse_args()
    f_words = [w.strip() for w in args.words.split(',')] if args.words else None
    process_episode(args.start, f_words)