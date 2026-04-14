import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 設定路徑
base_dir = Path(__file__).parent.parent
core_assets_dir = base_dir / "core" / "assets"
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
TEMPLATE_NAME = "cover_template.png"

def find_font():
    paths = [str(core_assets_dir / "msjhbd.ttc"), "C:\\Windows\\Fonts\\msjhbd.ttc"]
    for p in paths:
        if Path(p).exists(): return p
    return None

def generate_cover(ep_num, start_idx, end_idx):
    # --- 1. 資料清洗與 Log 監控 ---
    # 強制將傳入參數轉為整數，避免字串比較錯誤
    s_idx = int(start_idx)
    e_idx = int(end_idx)
    
    # 判定是否包含 4 位數
    is_4_digit = s_idx >= 1000 or e_idx >= 1000
    
    # print(f"📊 [Log] 集數: {ep_num} | 區間: {s_idx} ~ {e_idx}")
    # print(f"🔍 [Log] 4位數判定: {'是 (字體將縮小)' if is_4_digit else '否 (維持原大字)'}")

    # 尋找目標資料夾
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集的資料夾")
        return

    # 載入底圖
    template_path = core_assets_dir / TEMPLATE_NAME
    if not template_path.exists():
        print(f"❌ 找不到底圖: {template_path}")
        return
    
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_path = find_font()

    # --- 2. 根據判定結果動態設定字體大小 ---
    # 4位數時各減 10 級
    main_size = 90 if is_4_digit else 100
    sub_size = 25 if is_4_digit else 35
    bottom_size = 40 if is_4_digit else 50
    
    # print(f"🎨 [Log] 套用字體大小: 主數字={main_size}, 小標={sub_size}, 底部={bottom_size}")

    # 1. 填補黃色大數字 (350, 360)
    f_main = ImageFont.truetype(font_path, main_size) if font_path else ImageFont.load_default()
    draw.text((350, 360), f"{s_idx}~{e_idx}", fill="#FFFF00", font=f_main, anchor="mm")

    # 2. 填補紅色橫幅白色小數字 (1230, 570)
    f_sub = ImageFont.truetype(font_path, sub_size) if font_path else ImageFont.load_default()
    draw.text((1230, 570), f"{s_idx}-{e_idx}", fill="#FFFFFF", font=f_sub, anchor="mm")

    # 3. 填補最下方白色區塊深色數字 (630, 710)
    f_bottom = ImageFont.truetype(font_path, bottom_size) if font_path else ImageFont.load_default()
    draw.text((630, 710), f"{s_idx}-{e_idx}", fill="#1D2A4B", font=f_bottom, anchor="mm")

    # 儲存
    output_path = target_folder / "04_images" / f"cover.png"
    img.save(output_path)
    img.close()
    print(f"✅ 第 {ep_num:02d} 集封面產生成功！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args()

    # 自動推算逻辑
    s_val = args.start_idx if args.start_idx is not None else (args.ep - 1) * 30 + 1
    e_val = args.end_idx if args.end_idx is not None else args.ep * 30
    
    generate_cover(args.ep, s_val, e_val)
