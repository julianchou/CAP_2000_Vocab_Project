import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1"})
MODEL_ID = "gemini-2.5-flash"


def expand_batch_with_ai(word_list):
    words_str = ", ".join(word_list)
    prompt = f"""
你是資深英文老師，請為以下單字清單產生教學內容：
{words_str}

請嚴格輸出 JSON 陣列，每個物件都必須包含以下欄位：
- Word
- POS
- Meaning
- English_Sentence
- Chinese_Translation

規則：
1. POS 請使用常見縮寫，例如 n., v., adj., adv.
2. Meaning 請使用最常見、自然的繁體中文解釋。
3. English_Sentence 請提供生活化自然例句，控制在 12 個英文單字以內。
4. Chinese_Translation 請提供對應的繁體中文翻譯。

只輸出純 JSON，不要加 Markdown code fence。
"""
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        return json.loads(text)
    except Exception as exc:
        print(f"❌ AI 產生單字內容失敗：{exc}")
        return []


def process_episode(ep_num: int):
    base_dir = Path(__file__).resolve().parent.parent
    workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
    index_path = base_dir / "core" / "assets" / "vocab_index.csv"
    if not index_path.exists():
        print(f"❌ 找不到單字索引檔：{index_path}")
        return

    target_folder, start_idx, end_idx = resolve_episode_range(workspace_dir, ep_num)
    idx_df = pd.read_csv(index_path)
    target_words = idx_df[(idx_df["Index"] >= start_idx) & (idx_df["Index"] <= end_idx)]["Word"].tolist()
    if not target_words:
        print(f"❌ 第 {ep_num:02d} 集找不到對應單字區間：{start_idx}-{end_idx}")
        return

    print(f"🎬 [Episode {ep_num:02d}] 正在產生 {start_idx:04d}-{end_idx:04d} 的單字內容...")

    full_data = []
    batch_size = 10
    for offset in range(0, len(target_words), batch_size):
        batch = target_words[offset: offset + batch_size]
        print(f"   - 處理第 {offset + 1} ~ {offset + len(batch)} 個單字")
        batch_result = expand_batch_with_ai(batch)
        full_data.extend(batch_result)
        time.sleep(1)

    if not full_data:
        print("❌ 沒有成功產出任何單字資料。")
        return

    out_dir = target_folder / "03_storyboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_df = pd.DataFrame(full_data)
    csv_path = out_dir / "vocab_data.csv"
    json_path = out_dir / "vocab_data.json"
    vocab_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(vocab_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ vocab_data.csv 已輸出：{csv_path}")
    print(f"✅ vocab_data.json 已輸出：{json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, help="處理單一集數")
    parser.add_argument("--start", type=int, help="相容舊流程，等同 --ep")
    args = parser.parse_args()

    target_ep = args.ep if args.ep is not None else args.start
    if target_ep is None:
        parser.error("請提供 --ep 或 --start")
    process_episode(target_ep)
