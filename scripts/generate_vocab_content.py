import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from episode_range_utils import resolve_episode_range
from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response


load_dotenv()

GEMINI_MODEL_ID = os.getenv("CAP_VOCAB_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-flash"))
OPENAI_MODEL_ID = os.getenv("CAP_VOCAB_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL_ID = os.getenv("CAP_VOCAB_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)
VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_VOCAB_LLM_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def build_prompt(word_list: list[str]) -> str:
    words_str = ", ".join(word_list)
    return f"""
請為以下英文單字產生適合國中程度學生使用的單字學習資料：
{words_str}

請只回傳 JSON，不要使用 Markdown code fence。格式可以是 JSON array，或是包含 items array 的 JSON object。
每個單字都需要包含以下欄位：
- Word
- POS
- Meaning
- English_Sentence
- Chinese_Translation

規則：
1. POS 請使用常見詞性縮寫，例如 n., v., adj., adv.
2. Meaning 請使用繁體中文，簡潔自然。
3. English_Sentence 請使用清楚、自然、適合國中程度的英文例句，盡量控制在 12 個英文單字以內。
4. Chinese_Translation 請提供例句的繁體中文翻譯。
5. 請維持輸入單字順序，並且每個輸入單字都要產生一筆資料。
""".strip()


def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_text(text: str) -> str:
    text = strip_json_fence(text)
    if not text:
        return text
    if text[0] in "[{":
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def parse_vocab_json(text: str) -> list[dict[str, Any]]:
    raw = json.loads(extract_json_text(text))
    if isinstance(raw, dict):
        if isinstance(raw.get("items"), list):
            raw = raw["items"]
        elif isinstance(raw.get("words"), list):
            raw = raw["words"]
        else:
            raise ValueError("JSON object must contain an items array")
    if not isinstance(raw, list):
        raise ValueError("model response must be a JSON array")

    required = ["Word", "POS", "Meaning", "English_Sentence", "Chinese_Translation"]
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = {key: str(item.get(key, "")).strip() for key in required}
        if row["Word"]:
            rows.append(row)
    return rows


def expand_batch_with_gemini(word_list: list[str]) -> list[dict[str, Any]]:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
    response = client.models.generate_content(model=GEMINI_MODEL_ID, contents=build_prompt(word_list))
    return parse_vocab_json(getattr(response, "text", "") or "")


def expand_batch_with_openai(word_list: list[str]) -> list[dict[str, Any]]:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL_ID,
        input=[
            {
                "role": "system",
                "content": "You generate strict JSON only. Do not include Markdown fences or explanations.",
            },
            {"role": "user", "content": build_prompt(word_list)},
        ],
    )
    return parse_vocab_json(response.output_text)


def expand_batch_with_nvidia(word_list: list[str]) -> list[dict[str, Any]]:
    raw = nvidia_chat_response(
        build_prompt(word_list),
        model=NVIDIA_MODEL_ID,
        system_prompt="You generate strict JSON only. Do not include Markdown fences or explanations.",
        temperature=0.2,
    )
    return parse_vocab_json(raw)


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "spending cap" in text.lower() or "quota" in text.lower()


def expand_batch_with_ai(word_list: list[str], provider: str) -> list[dict[str, Any]]:
    provider = normalize_provider(provider)
    errors: list[str] = []

    if provider in {"auto", "gemini"}:
        try:
            rows = expand_batch_with_gemini(word_list)
            print(f"   provider=gemini model={GEMINI_MODEL_ID} rows={len(rows)}", flush=True)
            return rows
        except Exception as exc:
            message = str(exc)
            errors.append(f"Gemini: {message}")
            if provider == "gemini":
                print(f"❌ Gemini 產生單字內容失敗：{message}", flush=True)
                return []
            if is_gemini_quota_error(exc):
                print("⚠️ Gemini 額度或 spending cap 已耗盡，改用 OpenAI。", flush=True)
            else:
                print(f"⚠️ Gemini 產生失敗，改用 OpenAI：{message}", flush=True)

    if provider in {"auto", "openai"}:
        try:
            rows = expand_batch_with_openai(word_list)
            print(f"   provider=openai model={OPENAI_MODEL_ID} rows={len(rows)}", flush=True)
            return rows
        except Exception as exc:
            message = str(exc)
            errors.append(f"OpenAI: {message}")
            print(f"❌ OpenAI 產生單字內容失敗：{message}", flush=True)

    if provider == "nvidia":
        try:
            rows = expand_batch_with_nvidia(word_list)
            print(f"   provider=nvidia model={NVIDIA_MODEL_ID} rows={len(rows)}", flush=True)
            return rows
        except Exception as exc:
            message = str(exc)
            errors.append(f"NVIDIA: {message}")
            print(f"❌ NVIDIA 產生單字內容失敗：{message}", flush=True)

    if errors:
        print("❌ AI 產生單字內容失敗：" + " | ".join(errors), flush=True)
    return []


def process_episode(ep_num: int, provider: str) -> bool:
    base_dir = Path(__file__).resolve().parent.parent
    workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
    index_path = base_dir / "core" / "assets" / "vocab_index.csv"
    if not index_path.exists():
        print(f"❌ 找不到單字索引檔：{index_path}", flush=True)
        return False

    target_folder, start_idx, end_idx = resolve_episode_range(workspace_dir, ep_num)
    idx_df = pd.read_csv(index_path)
    target_words = idx_df[(idx_df["Index"] >= start_idx) & (idx_df["Index"] <= end_idx)]["Word"].tolist()
    if not target_words:
        print(f"❌ Episode {ep_num:02d} 找不到單字範圍：{start_idx}-{end_idx}", flush=True)
        return False

    provider = normalize_provider(provider)
    print(f"🎬 [Episode {ep_num:02d}] 正在產生 {start_idx:04d}-{end_idx:04d} 的單字內容...", flush=True)
    print(
        f"   llm_provider={provider} gemini_model={GEMINI_MODEL_ID} "
        f"openai_model={OPENAI_MODEL_ID} nvidia_model={NVIDIA_MODEL_ID}",
        flush=True,
    )

    full_data: list[dict[str, Any]] = []
    batch_size = 10
    for offset in range(0, len(target_words), batch_size):
        batch = target_words[offset: offset + batch_size]
        print(f"   - 處理第 {offset + 1} ~ {offset + len(batch)} 個單字", flush=True)
        batch_result = expand_batch_with_ai(batch, provider)
        full_data.extend(batch_result)
        time.sleep(1)

    if not full_data:
        print("❌ 沒有成功產出任何單字資料。", flush=True)
        return False

    out_dir = target_folder / "03_storyboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_df = pd.DataFrame(full_data)
    csv_path = out_dir / "vocab_data.csv"
    json_path = out_dir / "vocab_data.json"
    vocab_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(vocab_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ vocab_data.csv 已輸出：{csv_path}", flush=True)
    print(f"✅ vocab_data.json 已輸出：{json_path}", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, help="要處理的集數")
    parser.add_argument("--start", type=int, help="相容舊參數；等同 --ep")
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_PROVIDERS),
        default=os.getenv("CAP_VOCAB_LLM_PROVIDER", "auto"),
        help="LLM provider: auto 會先用 Gemini，失敗後改用 OpenAI。",
    )
    args = parser.parse_args()

    target_ep = args.ep if args.ep is not None else args.start
    if target_ep is None:
        parser.error("請提供 --ep 或 --start")
    raise SystemExit(0 if process_episode(target_ep, args.provider) else 1)
