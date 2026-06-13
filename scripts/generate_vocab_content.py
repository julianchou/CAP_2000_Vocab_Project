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
SOURCE_POS_TO_OUTPUT = {
    "\u540d\u8a5e": "n.",
    "\u52d5\u8a5e": "v.",
    "\u5f62\u5bb9\u8a5e": "adj.",
    "\u526f\u8a5e": "adv.",
    "\u4ee3\u540d\u8a5e": "pron.",
    "\u4ecb\u7cfb\u8a5e": "prep.",
    "\u9023\u63a5\u8a5e": "conj.",
    "\u9650\u5b9a\u8a5e": "det.",
    "\u6240\u6709\u683c": "poss.",
    "\u52a9\u52d5\u8a5e": "aux.",
    "\u611f\u5606\u8a5e": "interj.",
    "\u6578\u8a5e": "num.",
    "\u7591\u554f\u8a5e": "interrog.",
}
REQUIRED_INDEX_COLUMNS = {"Index", "Word", "POS", "Meaning"}
VOCAB_OUTPUT_COLUMNS = [
    "Index",
    "Word",
    "Phonetic",
    "POS",
    "Meaning",
    "English_Sentence",
    "Chinese_Translation",
]


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_VOCAB_LLM_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def build_prompt(entries: list[dict[str, Any]]) -> str:
    entries_json = json.dumps(entries, ensure_ascii=False, indent=2)
    return f"""
Create one short English example sentence and Traditional Chinese translation for each vocabulary entry.
The source POS and Meaning are authoritative. Use the exact requested sense, especially when the same
English spelling appears more than once.

Source entries:
{entries_json}

Return JSON only, without Markdown. Return a JSON array with exactly one item for each source entry.
Each item must contain:
- Index
- Word
- Phonetic
- English_Sentence
- Chinese_Translation

Rules:
1. Preserve Index, Word, and input order exactly.
2. Phonetic must be the standard American English IPA pronunciation for Word, wrapped in /slashes/.
3. For a phrase, return the pronunciation of the complete phrase.
4. The sentence must demonstrate the supplied Meaning, not another sense of the word.
5. Use clear junior-high-level English and preferably no more than 12 English words.
6. Translate the sentence naturally into Traditional Chinese.
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

    required = ["Index", "Word", "Phonetic", "English_Sentence", "Chinese_Translation"]
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = {key: str(item.get(key, "")).strip() for key in required}
        if row["Index"] and row["Word"]:
            rows.append(row)
    return rows


def expand_batch_with_gemini(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
    response = client.models.generate_content(model=GEMINI_MODEL_ID, contents=build_prompt(entries))
    return parse_vocab_json(getattr(response, "text", "") or "")


def expand_batch_with_openai(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            {"role": "user", "content": build_prompt(entries)},
        ],
    )
    return parse_vocab_json(response.output_text)


def expand_batch_with_nvidia(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = nvidia_chat_response(
        build_prompt(entries),
        model=NVIDIA_MODEL_ID,
        system_prompt="You generate strict JSON only. Do not include Markdown fences or explanations.",
        temperature=0.2,
    )
    return parse_vocab_json(raw)


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "spending cap" in text.lower() or "quota" in text.lower()


def expand_batch_with_ai(entries: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    provider = normalize_provider(provider)
    errors: list[str] = []

    if provider in {"auto", "gemini"}:
        try:
            rows = expand_batch_with_gemini(entries)
            print(f"   provider=gemini model={GEMINI_MODEL_ID} rows={len(rows)}", flush=True)
            return rows
        except Exception as exc:
            message = str(exc)
            errors.append(f"Gemini: {message}")
            if provider == "gemini":
                print(f"❌ Gemini 產生單字內容失敗：{message}", flush=True)
                return []
            if is_gemini_quota_error(exc):
                print("⚠️ Gemini 額度或 spending cap 已耗盡，改用 OpenAI，若 OpenAI 也失敗則改用 NVIDIA。", flush=True)
            else:
                print(f"⚠️ Gemini 產生失敗，改用 OpenAI，若 OpenAI 也失敗則改用 NVIDIA：{message}", flush=True)

    if provider in {"auto", "openai"}:
        try:
            rows = expand_batch_with_openai(entries)
            print(f"   provider=openai model={OPENAI_MODEL_ID} rows={len(rows)}", flush=True)
            return rows
        except Exception as exc:
            message = str(exc)
            errors.append(f"OpenAI: {message}")
            print(f"❌ OpenAI 產生單字內容失敗：{message}", flush=True)

    if provider in {"auto", "nvidia"}:
        try:
            rows = expand_batch_with_nvidia(entries)
            print(f"   provider=nvidia model={NVIDIA_MODEL_ID} rows={len(rows)}", flush=True)
            return rows
        except Exception as exc:
            message = str(exc)
            errors.append(f"NVIDIA: {message}")
            print(f"❌ NVIDIA 產生單字內容失敗：{message}", flush=True)

    if errors:
        print("❌ AI 產生單字內容失敗：" + " | ".join(errors), flush=True)
    return []


def normalize_source_pos(value: str) -> str:
    source_pos = str(value or "").strip()
    parts = [part.strip() for part in source_pos.split("/") if part.strip()]
    if not parts or any(part not in SOURCE_POS_TO_OUTPUT for part in parts):
        raise ValueError(f"Unsupported source POS: {source_pos!r}")
    return "/".join(SOURCE_POS_TO_OUTPUT[part] for part in parts)


def normalize_phonetic(value: str) -> str:
    phonetic = str(value or "").strip()
    if phonetic.startswith("[") and phonetic.endswith("]"):
        phonetic = phonetic[1:-1].strip()
    phonetic = phonetic.strip("/")
    if not phonetic or re.search(r"[\r\n\u3400-\u9fff]", phonetic):
        raise ValueError(f"Invalid IPA phonetic value: {value!r}")
    return f"/{phonetic}/"


def reconcile_batch(
    source_entries: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(generated_rows) != len(source_entries):
        raise ValueError(
            f"LLM returned {len(generated_rows)} rows for {len(source_entries)} source entries."
        )

    generated_by_index: dict[int, dict[str, Any]] = {}
    for row in generated_rows:
        try:
            index = int(str(row.get("Index", "")).strip())
        except ValueError as exc:
            raise ValueError(f"LLM returned an invalid Index: {row.get('Index')!r}") from exc
        if index in generated_by_index:
            raise ValueError(f"LLM returned duplicate Index {index}.")
        generated_by_index[index] = row

    reconciled: list[dict[str, Any]] = []
    for source in source_entries:
        index = int(source["Index"])
        generated = generated_by_index.get(index)
        if generated is None:
            raise ValueError(f"LLM omitted source Index {index}.")
        if str(generated.get("Word", "")).strip() != str(source["Word"]).strip():
            raise ValueError(
                f"LLM changed Word for Index {index}: "
                f"{generated.get('Word')!r} != {source['Word']!r}"
            )

        sentence = str(generated.get("English_Sentence", "")).strip()
        translation = str(generated.get("Chinese_Translation", "")).strip()
        phonetic = normalize_phonetic(str(generated.get("Phonetic", "")))
        if not sentence or not translation:
            raise ValueError(f"LLM returned blank sentence content for Index {index}.")

        reconciled.append(
            {
                "Index": index,
                "Word": str(source["Word"]).strip(),
                "Phonetic": phonetic,
                "POS": normalize_source_pos(str(source["POS"])),
                "Meaning": str(source["Meaning"]).strip(),
                "English_Sentence": sentence,
                "Chinese_Translation": translation,
            }
        )
    return reconciled


def write_vocab_outputs(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    vocab_df = pd.DataFrame(rows)
    missing_columns = set(VOCAB_OUTPUT_COLUMNS).difference(vocab_df.columns)
    if missing_columns:
        raise ValueError(
            "Vocabulary output is missing columns: " + ", ".join(sorted(missing_columns))
        )

    vocab_df = vocab_df[VOCAB_OUTPUT_COLUMNS]
    if vocab_df["Phonetic"].astype(str).str.strip().eq("").any():
        raise ValueError("Vocabulary output contains blank Phonetic values.")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vocab_data.csv"
    json_path = out_dir / "vocab_data.json"
    vocab_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        vocab_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


def process_episode(ep_num: int, provider: str) -> bool:
    base_dir = Path(__file__).resolve().parent.parent
    workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
    index_path = base_dir / "core" / "assets" / "vocab_index.csv"
    if not index_path.exists():
        print(f"❌ 找不到單字索引檔：{index_path}", flush=True)
        return False

    target_folder, start_idx, end_idx = resolve_episode_range(workspace_dir, ep_num)
    idx_df = pd.read_csv(index_path)
    missing_columns = REQUIRED_INDEX_COLUMNS.difference(idx_df.columns)
    if missing_columns:
        print(
            "ERROR vocab_index.csv is missing authoritative columns: "
            + ", ".join(sorted(missing_columns))
            + ". Run scripts/extract_vocab_with_llm.py first.",
            flush=True,
        )
        return False

    target_df = idx_df[(idx_df["Index"] >= start_idx) & (idx_df["Index"] <= end_idx)].copy()
    target_df["Index"] = pd.to_numeric(target_df["Index"], errors="raise").astype(int)
    target_df = target_df.sort_values("Index")
    expected_indexes = list(range(start_idx, end_idx + 1))
    actual_indexes = target_df["Index"].tolist()
    if actual_indexes != expected_indexes:
        print(
            f"ERROR vocab_index.csv does not contain the complete range "
            f"{start_idx}-{end_idx}: found {actual_indexes}",
            flush=True,
        )
        return False

    target_entries = target_df[["Index", "Word", "POS", "Meaning"]].to_dict(orient="records")
    if not target_entries:
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
    for offset in range(0, len(target_entries), batch_size):
        batch = target_entries[offset: offset + batch_size]
        print(f"   - 處理第 {offset + 1} ~ {offset + len(batch)} 個單字", flush=True)
        batch_result = expand_batch_with_ai(batch, provider)
        if not batch_result:
            return False
        try:
            full_data.extend(reconcile_batch(batch, batch_result))
        except ValueError as exc:
            print(f"ERROR invalid LLM vocabulary response: {exc}", flush=True)
            return False
        time.sleep(1)

    if not full_data:
        print("❌ 沒有成功產出任何單字資料。", flush=True)
        return False

    out_dir = target_folder / "03_storyboards"
    try:
        csv_path, json_path = write_vocab_outputs(full_data, out_dir)
    except ValueError as exc:
        print(f"ERROR invalid vocabulary output: {exc}", flush=True)
        return False
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
        help="LLM provider: auto 會先用 Gemini，失敗後改用 OpenAI，再失敗改用 NVIDIA。",
    )
    args = parser.parse_args()

    target_ep = args.ep if args.ep is not None else args.start
    if target_ep is None:
        parser.error("請提供 --ep 或 --start")
    raise SystemExit(0 if process_episode(target_ep, args.provider) else 1)
