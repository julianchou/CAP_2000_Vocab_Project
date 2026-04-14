import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PyPDF2 import PdfReader

# 1. 初始化設定
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 建立 Client (使用穩定版 v1 接口)
client = genai.Client(
    api_key=API_KEY, 
    http_options={'api_version': 'v1'}
)

# 使用 Flash 模型，解析表格與處理長文本速度最快
MODEL_ID = "gemini-2.5-flash" 

def extract_page_with_llm(text, page_num):
    """請 Gemini 協助解析 PDF 頁面文字，提取精確的單字對照表"""
    prompt = f"""
    任務：從以下 PDF 提取的文字中，整理出「編號」與「英文單字」的對照表。
    
    待處理文字 (第 {page_num} 頁)：
    ---
    {text}
    ---

    要求：
    1. 輸出格式必須是嚴格的 JSON 陣列，例如：[{{"Index": 1, "Word": "a few"}}, {{"Index": 2, "Word": "a little"}}]
    2. 務必保留完整的英文片語（例如 "a lot" 不要只抓 "a"）。
    3. 忽略中文解釋、詞性或其他雜訊。
    4. 只輸出 JSON，不要包含 Markdown 標籤 (```json) 或任何說明。
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt
        )
        
        raw_text = response.text.strip()
        # 清除 Markdown 標籤
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1]) if len(lines) > 2 else raw_text
            
        return json.loads(raw_text)
    except Exception as e:
        print(f"⚠️ 第 {page_num} 頁解析出錯: {e}")
        return []

def main():
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "core" / "assets" / "cap_vocab_2000_source.pdf"
    output_path = base_dir / "core" / "assets" / "vocab_index.csv"

    if not pdf_path.exists():
        print(f"❌ 找不到 PDF 檔案: {pdf_path}")
        return

    print(f"🚀 開始使用 LLM 解析 PDF：{pdf_path.name}")
    reader = PdfReader(pdf_path)
    full_vocab_list = []

    # 逐頁讀取並請 AI 解析
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        print(f"⏳ 正在處理第 {page_num} / {len(reader.pages)} 頁...")
        
        page_text = page.extract_text()
        if not page_text.strip():
            continue
            
        page_data = extract_page_with_llm(page_text, page_num)
        if page_data:
            full_vocab_list.extend(page_data)
            print(f"   ✅ 已擷取 {len(page_data)} 個單字")

    # 儲存結果
    if full_vocab_list:
        df = pd.DataFrame(full_vocab_list)
        
        # 確保 Index 欄位是數字並排序
        df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
        df = df.dropna(subset=['Index'])
        df['Index'] = df['Index'].astype(int)
        
        # 排序並移除重複編號
        df = df.sort_values("Index").drop_duplicates(subset=['Index'])
        
        # 建立目錄並存檔
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✨ 任務完成！")
        print(f"📊 最終抓取單字量: {len(df)}")
        print(f"📂 索引檔已儲存至: {output_path}")
        print(f"🧐 前三個單字預覽：")
        print(df.head(3))
    else:
        print("💥 失敗：未能從 PDF 中擷取到任何有效單字。")

if __name__ == "__main__":
    main()