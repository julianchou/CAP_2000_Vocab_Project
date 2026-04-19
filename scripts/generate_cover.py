import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from episode_range_utils import find_episode_folder, resolve_episode_range


base_dir = Path(__file__).resolve().parent.parent
core_assets_dir = base_dir / "core" / "assets"
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
TEMPLATE_NAME = "cover_template.png"


def find_font():
    paths = [str(core_assets_dir / "msjhbd.ttc"), "C:\\Windows\\Fonts\\msjhbd.ttc"]
    for path in paths:
        if Path(path).exists():
            return path
    return None


def generate_cover(ep_num, start_idx, end_idx):
    s_idx = int(start_idx)
    e_idx = int(end_idx)
    is_4_digit = s_idx >= 1000 or e_idx >= 1000

    target_folder = find_episode_folder(workspace_dir, ep_num)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集的資料夾")
        return

    template_path = core_assets_dir / TEMPLATE_NAME
    if not template_path.exists():
        print(f"❌ 找不到底圖: {template_path}")
        return

    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_path = find_font()

    main_size = 90 if is_4_digit else 100
    sub_size = 25 if is_4_digit else 35
    bottom_size = 40 if is_4_digit else 50

    f_main = ImageFont.truetype(font_path, main_size) if font_path else ImageFont.load_default()
    draw.text((350, 360), f"{s_idx}~{e_idx}", fill="#FFFF00", font=f_main, anchor="mm")

    f_sub = ImageFont.truetype(font_path, sub_size) if font_path else ImageFont.load_default()
    draw.text((1230, 570), f"{s_idx}-{e_idx}", fill="#FFFFFF", font=f_sub, anchor="mm")

    f_bottom = ImageFont.truetype(font_path, bottom_size) if font_path else ImageFont.load_default()
    draw.text((630, 710), f"{s_idx}-{e_idx}", fill="#1D2A4B", font=f_bottom, anchor="mm")

    output_path = target_folder / "04_images" / "cover.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    img.close()
    print(f"✅ 第 {ep_num:02d} 集封面產生成功！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    cli_start = args.start_idx if args.start_idx is not None else args.start
    cli_end = args.end_idx if args.end_idx is not None else args.end
    if cli_start is not None and cli_end is not None:
        s_val = cli_start
        e_val = cli_end
    else:
        _, s_val, e_val = resolve_episode_range(workspace_dir, args.ep)

    generate_cover(args.ep, s_val, e_val)
