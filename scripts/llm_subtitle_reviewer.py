import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
MODEL_NAME = "gemini-2.5-pro"

def review_subtitles_with_llm(ep_num):
    target_folder = next(workspace_dir.glob(f"Ep{ep_num:02d}_*"), None)
    if not target_folder:
        print(f"❌ 找不到第 {ep_num} 集的資料夾。")
        return

    input_srt = target_folder / "02_subtitles" / "notebooklm_audio.srt"
    output_srt = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"

    if not input_srt.exists():
        print(f"❌ 找不到原始字幕檔：{input_srt}")
        return

    print(f"📖 正在讀取第 {ep_num} 集字幕進行校對與時間軸修復...")
    with open(input_srt, 'r', encoding='utf-8') as f:
        raw_srt_content = f.read()

    # 6.0 結構嚴格版：修正單字但不准隨意合併條目
    prompt = f"""
    你是一位專業的影片字幕校對專家。請針對以下 SRT 內容進行修正，請務必遵守以下「結構不變」原則：

    1. **單字修復與局部合併**：
       - **僅在單字被切斷時合併**：例如「right」與「angle」被拆分成兩個序號時，才將其合併為一個。
       - **嚴禁語義合併**：即便兩句話在語義上連貫（如：中文翻譯緊隨英文之後），**只要它們在原始 SRT 中是分開的序號，就必須保持分開**。
       - **嚴禁跨行合併**：不要將不同語句合併到同一個時間區間（SRT 序號）內。
    
    2. **全方位校對**：
       - 修正「所有」英文單字的拼寫錯誤（不限 A 開頭）。
       - 根據上下文修正音近錯誤的字詞。
       - 為關鍵詞加上「」引號。

    3. **時間軸與格式**：
       - **保持間隙**：確保每一句結束與下一句開始之間有 0.02 秒的空隙 [cite: 3]。
       - **序號連續**：若有局部合併（如修復切斷單字），請重新編排序號 [cite: 3]。
       - 只輸出純 SRT 文字，不要任何 Markdown 標籤或解釋。

    原始字幕內容：
    {raw_srt_content}
    """

    print(f"🧠 正在呼叫 {MODEL_NAME} 執行雙重優化任務...")
    
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        cleaned_srt = response.text.replace("```srt", "").replace("```", "").strip()

        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write(cleaned_srt)
            
        print(f"✅ 校對與時間軸修復完成！存檔至：{output_srt}")

    except Exception as e:
        print(f"❌ 錯誤：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="處理集數")
    args = parser.parse_args()
    review_subtitles_with_llm(args.ep)
