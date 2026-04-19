import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
MODEL_ID = os.getenv("CAP_CLOZE_MODEL", "gemini-2.5-flash")
base_dir = Path(__file__).resolve().parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
OPTION_LABELS = ["A", "B", "C", "D"]

DEFAULT_PROMPT_TEMPLATE = """你是專業的英文考題編輯，請依據提供的 vocab_data，為本集每個單字各產生 1 題四選一克漏字題。

你必須嚴格遵守以下規則：
1. 請依 vocab_data 原本順序輸出，總題數必須剛好等於 {{VOCAB_COUNT}} 題。
2. 每題都必須對應到同一列 vocab 的 Word。
3. 請優先以該列的 English_Sentence 為基礎，把目標字詞替換成 `____`，形成 `blank_sentence`。
4. 如果句中需要詞形變化，例如第三人稱單數、過去式、現在分詞，正確選項請使用符合句子的詞形。
5. 干擾選項必須自然、合理、可辨識，不要亂造不存在或很奇怪的字。
6. explanation 請用繁體中文，簡短說明為什麼正確，並可順帶提示中文意思。
7. 只輸出 JSON，不要輸出任何額外文字。

請輸出 JSON 陣列。每個元素都必須包含：
- surface_word
- blank_sentence
- choice_A
- choice_B
- choice_C
- choice_D
- correct_option
- explanation

補充資訊：
- 集數：第 {{EPISODE}} 集
- 單字範圍：{{START_WORD}}-{{END_WORD}}
- 單字資料 JSON：
{{VOCAB_JSON}}
"""


def load_vocab_df(storyboard_dir: Path) -> pd.DataFrame:
    csv_path = storyboard_dir / "vocab_data.csv"
    json_path = storyboard_dir / "vocab_data.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif json_path.exists():
        df = pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))
    else:
        raise FileNotFoundError(f"找不到 vocab_data.csv 或 vocab_data.json：{storyboard_dir}")
    df = df.fillna("")
    if not json_path.exists():
        json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    return df


def prompt_path_for(storyboard_dir: Path) -> Path:
    return storyboard_dir / "cloze_quiz_prompt.txt"


def ensure_prompt_file(storyboard_dir: Path) -> Path:
    prompt_path = prompt_path_for(storyboard_dir)
    if not prompt_path.exists():
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def normalize_pos(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return token.split(",")[0].split()[0]


def find_surface_form(sentence: str, word: str) -> str:
    base = str(word or "").strip()
    if not base:
        return ""
    candidates = [base]
    lower = base.lower()
    if lower.endswith("y") and len(base) > 1:
        candidates.append(base[:-1] + "ies")
    if lower.endswith(("s", "x", "z", "ch", "sh", "o")):
        candidates.append(base + "es")
    else:
        candidates.append(base + "s")
    if lower.endswith("e"):
        candidates.append(base + "d")
        candidates.append(base[:-1] + "ing")
    else:
        candidates.append(base + "ed")
        candidates.append(base + "ing")

    for candidate in candidates:
        match = re.search(rf"\b{re.escape(candidate)}\b", sentence, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def build_blank_sentence(sentence: str, word: str) -> str:
    sentence = str(sentence or "").strip()
    word = str(word or "").strip()
    if not sentence or not word:
        return sentence

    surface_form = find_surface_form(sentence, word)
    if surface_form:
        pattern = re.compile(rf"\b{re.escape(surface_form)}\b", flags=re.IGNORECASE)
        blanked, count = pattern.subn("____", sentence, count=1)
        if count:
            return blanked
    return f"Choose the best word: ____ . Hint: {sentence}"


def inflect_like_surface_form(base_word: str, surface_form: str) -> str:
    base = str(base_word or "").strip()
    surface = str(surface_form or "").strip()
    if not base or not surface:
        return base
    if surface.lower() == base.lower():
        return base
    surface_lower = surface.lower()
    base_lower = base.lower()
    if surface_lower.endswith("ies") and base_lower.endswith("y") and len(base) > 1:
        return base[:-1] + "ies"
    if surface_lower.endswith("ing"):
        if base_lower.endswith("e") and not base_lower.endswith("ee"):
            return base[:-1] + "ing"
        return base + "ing"
    if surface_lower.endswith("ed"):
        if base_lower.endswith("e"):
            return base + "d"
        return base + "ed"
    if surface_lower.endswith("es") and base_lower.endswith(("s", "x", "z", "ch", "sh", "o")):
        return base + "es"
    if surface_lower.endswith("s"):
        return base + "s"
    return base


def choice_pool(df: pd.DataFrame, answer_word: str, answer_pos: str) -> list[str]:
    same_pos = []
    other_words = []
    answer_lower = answer_word.lower()
    for _, row in df.iterrows():
        candidate = str(row.get("Word", "")).strip()
        if not candidate or candidate.lower() == answer_lower:
            continue
        if normalize_pos(row.get("POS", "")) == answer_pos:
            same_pos.append(candidate)
        else:
            other_words.append(candidate)
    return same_pos + other_words


def deterministic_fallback_question(ep_num: int, idx: int, row: pd.Series, df: pd.DataFrame) -> dict:
    word = str(row.get("Word", "")).strip()
    pos = str(row.get("POS", "")).strip()
    meaning = str(row.get("Meaning", "")).strip()
    sentence = str(row.get("English_Sentence", "")).strip()
    translation = str(row.get("Chinese_Translation", "")).strip()
    surface_form = find_surface_form(sentence, word) or word
    blank_sentence = build_blank_sentence(sentence, word)
    rng = random.Random(f"{ep_num}:{idx}:{word}")

    pool = choice_pool(df, word, normalize_pos(pos))
    distractors = []
    for candidate in pool:
        if candidate not in distractors:
            distractors.append(candidate)
        if len(distractors) == 3:
            break
    while len(distractors) < 3:
        distractors.append(f"{word}_{len(distractors) + 1}")

    is_verb = normalize_pos(pos).startswith("v")
    if is_verb:
        options = [inflect_like_surface_form(word, surface_form)] + [
            inflect_like_surface_form(opt, surface_form) for opt in distractors[:3]
        ]
        correct_surface = inflect_like_surface_form(word, surface_form)
    else:
        options = [word] + distractors[:3]
        correct_surface = word
    rng.shuffle(options)
    correct_option = OPTION_LABELS[options.index(correct_surface)]
    if is_verb and correct_surface.lower() != word.lower():
        explanation = f"{word} 在本句要用 {correct_surface}，意思是 {meaning}。原句中文：{translation}"
    else:
        explanation = f"{word} = {meaning}。原句中文：{translation}"

    return {
        "surface_word": correct_surface,
        "blank_sentence": blank_sentence,
        "choice_A": options[0],
        "choice_B": options[1],
        "choice_C": options[2],
        "choice_D": options[3],
        "correct_option": correct_option,
        "explanation": explanation,
    }


def render_prompt(template_text: str, ep_num: int, start_word: int, end_word: int, vocab_df: pd.DataFrame) -> str:
    vocab_json = vocab_df.to_json(orient="records", force_ascii=False, indent=2)
    return (
        template_text
        .replace("{{EPISODE}}", str(ep_num))
        .replace("{{START_WORD}}", str(start_word))
        .replace("{{END_WORD}}", str(end_word))
        .replace("{{VOCAB_COUNT}}", str(len(vocab_df)))
        .replace("{{VOCAB_JSON}}", vocab_json)
    )


def parse_ai_questions(text: str) -> list[dict]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        questions = payload.get("questions")
        if isinstance(questions, list):
            return questions
    if isinstance(payload, list):
        return payload
    raise ValueError("AI 回傳格式不是 JSON 陣列或 questions 物件。")


def generate_questions_with_ai(rendered_prompt: str) -> list[dict]:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=rendered_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    return parse_ai_questions(response.text)


def normalize_ai_question(ep_num: int, idx: int, row: pd.Series, df: pd.DataFrame, raw_item: dict | None) -> dict:
    fallback = deterministic_fallback_question(ep_num, idx, row, df)
    raw_item = raw_item or {}

    choice_map = {}
    for label in OPTION_LABELS:
        val = str(raw_item.get(f"choice_{label}", "")).strip()
        choice_map[label] = val or fallback[f"choice_{label}"]

    correct_option = str(raw_item.get("correct_option", "")).strip().upper()
    if correct_option not in OPTION_LABELS:
        correct_surface = str(raw_item.get("surface_word", "")).strip()
        if correct_surface:
            matched = next((label for label in OPTION_LABELS if choice_map[label].strip().lower() == correct_surface.lower()), "")
            correct_option = matched or fallback["correct_option"]
        else:
            correct_option = fallback["correct_option"]

    return {
        "episode": ep_num,
        "question_no": idx,
        "question_id": f"Ep{ep_num:02d}_Q{idx:02d}",
        "word": str(row.get("Word", "")).strip(),
        "surface_word": str(raw_item.get("surface_word", "")).strip() or fallback["surface_word"],
        "pos": str(row.get("POS", "")).strip(),
        "meaning": str(row.get("Meaning", "")).strip(),
        "sentence": str(row.get("English_Sentence", "")).strip(),
        "blank_sentence": str(raw_item.get("blank_sentence", "")).strip() or fallback["blank_sentence"],
        "translation": str(row.get("Chinese_Translation", "")).strip(),
        "correct_option": correct_option,
        "correct_word": str(row.get("Word", "")).strip(),
        "correct_surface_word": choice_map[correct_option].strip(),
        "explanation": str(raw_item.get("explanation", "")).strip() or fallback["explanation"],
        "choice_A": choice_map["A"],
        "choice_B": choice_map["B"],
        "choice_C": choice_map["C"],
        "choice_D": choice_map["D"],
    }


def build_question_rows(ep_num: int, df: pd.DataFrame, ai_questions: list[dict]) -> list[dict]:
    rows = []
    ai_questions = ai_questions or []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        raw_item = ai_questions[idx - 1] if idx - 1 < len(ai_questions) else None
        rows.append(normalize_ai_question(ep_num, idx, row, df, raw_item))
    return rows


def save_questions(storyboard_dir: Path, questions: list[dict]):
    csv_path = storyboard_dir / "cloze_questions.csv"
    json_path = storyboard_dir / "cloze_questions.json"
    df = pd.DataFrame(questions)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"克漏字題目已輸出：{csv_path}")
    print(f"克漏字題目已輸出：{json_path}")


def main():
    parser = argparse.ArgumentParser(description="依 vocab_data 與可編輯 prompt 產生克漏字題目")
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()

    episode_folder, start_word, end_word = resolve_episode_range(workspace_dir, args.ep)
    storyboard_dir = episode_folder / "03_storyboards"
    vocab_df = load_vocab_df(storyboard_dir)
    if vocab_df.empty:
        raise ValueError("沒有可用的 vocab_data，無法產生克漏字題目。")

    prompt_path = ensure_prompt_file(storyboard_dir)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    rendered_prompt = render_prompt(prompt_template, args.ep, start_word, end_word, vocab_df)
    ai_questions = generate_questions_with_ai(rendered_prompt)
    questions = build_question_rows(args.ep, vocab_df, ai_questions)
    save_questions(storyboard_dir, questions)


if __name__ == "__main__":
    main()
