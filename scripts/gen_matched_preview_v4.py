import os
import csv
import argparse
import traceback
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
client = genai.Client()

# --- 目錄設定 ---
base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

def process_image_standard(img_path):
    """
    ⚡ 關鍵新增：強制圖片標準化
    確保所有圖片都是 1920x1080, RGB 模式，避免 FFmpeg 渲染中斷。
    """
    try:
        with Image.open(img_path) as img:
            # 強制轉換為 RGB (防止 RGBA 或 P 模式導致 FFmpeg 報錯)
            img = img.convert('RGB')
            
            # 如果尺寸不對，強制縮放
            if img.size != (1920, 1080):
                print(f"🔧 修正尺寸：{img_path.name} ({img.size} -> 1920x1080)")
                img = img.resize((1920, 1080), Image.Resampling.LANCZOS)
            
            img.save(img_path, "PNG")
    except Exception as e:
        print(f"🚨 無法標準化圖片 {img_path.name}: {e}")

def generate_scene_image(target_folder: Path, row: dict, force_ai_regenerate: bool = False):
    ai_images_folder = target_folder / "04_images" / "ai_generated"
    flashcards_folder = target_folder / "04_images" / "flashcards"
    ai_images_folder.mkdir(parents=True, exist_ok=True)

    source_type = row.get("source_type", "AI").strip().lower()

    if source_type == "flashcard":
        word = row.get("flashcard_word", "").strip()
        img_path = flashcards_folder / f"{word}.png"
        if not img_path.exists():
            print(f"⚠️ 警告：找不到單字圖卡 {word}.png")
            fallback_img = Image.new('RGB', (1920, 1080), color=(44, 62, 80))
            img_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_img.save(img_path)
        else:
            print(f"📁 [Flashcard] 使用並標準化圖卡: {word}.png")
            process_image_standard(img_path)
        return img_path

    start_t = str(row["start_time"]).strip()
    img_name = f"img_{start_t}.png"
    img_path = ai_images_folder / img_name
    raw_prompt = row.get("image_prompt", "").strip()
    prompt_text = raw_prompt if raw_prompt else "A clean educational background, soft blue and white gradient, 16:9"

    if img_path.exists() and not force_ai_regenerate:
        print(f"⏭️ {img_name} 已存在，重新標準化以確保安全。")
        process_image_standard(img_path)
        return img_path

    try:
        print(f"🎨 正在為 AI 場景生成圖片 ({img_name})...")
        result = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=prompt_text,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9"
            )
        )
        result.generated_images[0].image.save(img_path)
        process_image_standard(img_path)
        print(f"✅ {img_name} 產圖成功且標準化！")
    except Exception as e:
        print(f"🚨 {img_name} 生成失敗: {e}")
        fallback_img = Image.new('RGB', (1920, 1080), color=(44, 62, 80))
        fallback_img.save(img_path)

    return img_path

def generate_preview(ep_num, scene_id: int | None = None):
    print(f"\n🎞️ [Stage 5] 正在處理第 {ep_num:02d} 集的分鏡配圖與 FFmpeg 列表...")
    
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num:02d} 集的資料夾。")
        return

    storyboard_csv = target_folder / "03_storyboards" / "storyboard.csv"
    ai_images_folder = target_folder / "04_images" / "ai_generated"
    flashcards_folder = target_folder / "04_images" / "flashcards"
    output_folder = target_folder / "05_output"

    if not storyboard_csv.exists():
        print(f"❌ 找不到分鏡表：{storyboard_csv}")
        return

    ai_images_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    inputs_list = []

    with open(storyboard_csv, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if scene_id is not None:
        target_row = next((row for row in rows if str(row.get("scene_id", "")).strip() == str(scene_id)), None)
        if not target_row:
            print(f"❌ 找不到 scene_id={scene_id}")
            return
        img_path = generate_scene_image(target_folder, target_row, force_ai_regenerate=True)
        print(f"✅ Scene {scene_id} 已完成單獨產圖：{img_path}")
        return

    cumulative_time = 0.0

    for i, row in enumerate(rows):
        # 1. 計算停留時間 (Duration)
        if i < len(rows) - 1:
            next_start_time = float(rows[i+1]["start_time"])
            duration = next_start_time - cumulative_time
        else:
            end_time = float(row["end_time"])
            duration = end_time - cumulative_time
            if duration <= 0: duration = 2.0

        img_path = generate_scene_image(target_folder, row, force_ai_regenerate=False)

        # 3. 寫入 FFmpeg inputs 列表
        rel_img_path = os.path.relpath(img_path, output_folder)
        safe_img_path = str(rel_img_path).replace('\\', '/')
        inputs_list.append(f"file '{safe_img_path}'\nduration {duration:.3f}")
        cumulative_time += duration

    # 產出 inputs.txt
    inputs_txt_path = output_folder / "inputs.txt"
    if inputs_list:
        with open(inputs_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(inputs_list))
            f.write(f"\n{inputs_list[-1].splitlines()[0]}")
        print(f"✅ 第 {ep_num:02d} 集 inputs.txt 更新完成！")

def main():
    parser = argparse.ArgumentParser(description="產生匹配的圖片與 FFmpeg inputs.txt")
    parser.add_argument("--ep", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--scene_id", type=int, help="只重產指定 scene 的圖片")
    args = parser.parse_args()

    if args.ep is not None:
        start_ep, end_ep = args.ep, args.ep
    elif args.start is not None:
        start_ep = args.start
        end_ep = args.end if args.end else args.start
    else:
        parser.error("請提供 --ep 或 --start 參數")

    for ep in range(start_ep, end_ep + 1):
        generate_preview(ep, scene_id=args.scene_id)

if __name__ == "__main__":
    main()
