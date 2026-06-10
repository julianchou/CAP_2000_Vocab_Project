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
from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
MODEL_ID = os.getenv("CAP_CLOZE_MODEL", "gemini-2.5-flash")
REVIEW_MODEL_ID = os.getenv("CAP_CLOZE_REVIEW_MODEL", "gemini-2.5-flash")
OPENAI_MODEL_ID = os.getenv("CAP_CLOZE_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
OPENAI_REVIEW_MODEL_ID = os.getenv("CAP_CLOZE_OPENAI_REVIEW_MODEL", os.getenv("OPENAI_REVIEW_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL_ID = os.getenv("CAP_CLOZE_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)
VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}
VALID_REVIEW_PROVIDERS = {"auto", "gemini", "openai"}
base_dir = Path(__file__).resolve().parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
OPTION_LABELS = ["A", "B", "C", "D"]


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_CLOZE_LLM_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def normalize_review_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_CLOZE_REVIEW_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_REVIEW_PROVIDERS else "auto"


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "spending cap" in text.lower() or "quota" in text.lower()


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


def openai_json_response(prompt: str, model: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    response = OpenAI().responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "Return strict JSON only. Do not include Markdown fences or explanations outside JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text


def nvidia_json_response(prompt: str, model: str) -> str:
    return nvidia_chat_response(
        prompt,
        model=model,
        system_prompt="Return strict JSON only. Do not include Markdown fences or explanations outside JSON.",
        temperature=0.2,
    )


def detect_profile_id() -> str:
    profile_id = str(os.environ.get("CAP_PROFILE_ID", "")).strip()
    if profile_id:
        return profile_id
    if workspace_dir.parent.name == "workspaces" and workspace_dir.name:
        return workspace_dir.name
    return "default"


def shared_prompt_path() -> Path:
    return base_dir / "config" / detect_profile_id() / "prompts" / "cloze_quiz_prompt.txt"


DEFAULT_PROMPT_TEMPLATE = """你是專業的英文考題編輯，請依據提供的 vocab_data，為本集每個單字各產生 1 題四選一克漏字題。

你必須嚴格遵守以下規則：
1. 請依 vocab_data 原本順序輸出，總題數必須剛好等於 {{VOCAB_COUNT}} 題。
2. 每題都必須對應到同一列 vocab 的 Word，不可串到別的單字。
3. `blank_sentence` 必須直接使用該列的 `English_Sentence`，只把目標字詞替換成 `____`；不可改寫成別句，不可借用其他單字的句子。
4. `surface_word` 必須是該列目標字在原句中實際出現的詞形；若原句有詞形變化，正確答案也必須用同一詞形。
5. 四個選項中只能有一個正確答案。其餘三個干擾選項必須在這個句子裡明顯不通順、不合語意，不能出現兩個以上都合理的答案。
6. 不可重用其他題目的句子、答案、解釋或選項。
7. explanation 請用繁體中文，簡短說明為什麼正確，並指出其他選項不適合這句。
8. 只輸出 JSON，不要輸出任何額外文字。

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


def prompt_path_for(_storyboard_dir: Path) -> Path:
    return shared_prompt_path()


def ensure_prompt_file(storyboard_dir: Path) -> Path:
    prompt_path = prompt_path_for(storyboard_dir)
    if not prompt_path.exists():
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def normalize_pos(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return token.split(",")[0].split()[0]


def normalize_sentence_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("’", "'").replace("“", '"').replace("”", '"')
    cleaned = cleaned.strip(" \"'")
    return cleaned


def normalize_word_token(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).strip(" \"'").lower()


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


def expected_surface_word(word: str, pos: str, sentence: str) -> str:
    surface_form = find_surface_form(sentence, word) or str(word or "").strip()
    if normalize_pos(pos).startswith(("v", "n")):
        return inflect_like_surface_form(str(word or "").strip(), surface_form)
    return surface_form


def same_word_family(a: str, b: str) -> bool:
    a_norm = normalize_word_token(a)
    b_norm = normalize_word_token(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    if a_norm.startswith(b_norm[:4]) or b_norm.startswith(a_norm[:4]):
        return min(len(a_norm), len(b_norm)) >= 4
    return False


def choice_pool(df: pd.DataFrame, answer_word: str, answer_pos: str) -> list[str]:
    same_pos = []
    other_words = []
    answer_lower = normalize_word_token(answer_word)
    for _, row in df.iterrows():
        candidate = str(row.get("Word", "")).strip()
        if not candidate or normalize_word_token(candidate) == answer_lower:
            continue
        if same_word_family(candidate, answer_word):
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
    expected_answer = expected_surface_word(word, pos, sentence)
    rng = random.Random(f"{ep_num}:{idx}:{word}")

    pool = choice_pool(df, word, normalize_pos(pos))
    distractors = []
    for candidate in pool:
        inflected = inflect_like_surface_form(candidate, surface_form)
        if normalize_word_token(inflected) == normalize_word_token(expected_answer):
            continue
        if normalize_word_token(inflected) in {normalize_word_token(item) for item in distractors}:
            continue
        distractors.append(inflected)
        if len(distractors) == 3:
            break
    while len(distractors) < 3:
        distractors.append(f"{word}_{len(distractors) + 1}")

    options = [expected_answer] + distractors[:3]
    rng.shuffle(options)
    correct_option = OPTION_LABELS[options.index(expected_answer)]
    explanation = f"{expected_answer} 對應 {meaning}，而且最符合原句語意；其他選項放回這個句子都不自然或不合意思。"

    return {
        "surface_word": expected_answer,
        "blank_sentence": blank_sentence,
        "choice_A": options[0],
        "choice_B": options[1],
        "choice_C": options[2],
        "choice_D": options[3],
        "correct_option": correct_option,
        "explanation": explanation if not translation else f"{explanation} 原句中文：{translation}",
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
    json_text = extract_json_text(text)
    if not json_text:
        raise ValueError("AI response was empty; no JSON questions were returned")
    payload = json.loads(json_text)
    if isinstance(payload, dict):
        questions = payload.get("questions")
        if isinstance(questions, list):
            return questions
    if isinstance(payload, list):
        return payload
    raise ValueError("AI 回傳的 JSON 格式不正確，預期為陣列或含 questions 的物件。")


def generate_questions_with_gemini(rendered_prompt: str) -> list[dict]:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=rendered_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return parse_ai_questions(response.text)


def generate_questions_with_openai(rendered_prompt: str) -> list[dict]:
    return parse_ai_questions(openai_json_response(rendered_prompt, OPENAI_MODEL_ID))


def generate_questions_with_nvidia(rendered_prompt: str) -> list[dict]:
    return parse_ai_questions(nvidia_json_response(rendered_prompt, NVIDIA_MODEL_ID))


def generate_questions_with_ai(rendered_prompt: str, provider: str) -> list[dict]:
    provider = normalize_provider(provider)
    errors = []
    if provider in {"auto", "gemini"}:
        try:
            questions = generate_questions_with_gemini(rendered_prompt)
            print(f"cloze_provider=gemini model={MODEL_ID} questions={len(questions)}")
            return questions
        except Exception as exc:
            message = str(exc)
            errors.append(f"Gemini: {message}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print("WARN Gemini quota/spending cap exhausted; switching cloze generation to OpenAI, then NVIDIA if needed.", flush=True)
            else:
                print(f"WARN Gemini cloze generation failed; switching to OpenAI, then NVIDIA if needed: {message}", flush=True)
    if provider in {"auto", "openai"}:
        try:
            questions = generate_questions_with_openai(rendered_prompt)
            print(f"cloze_provider=openai model={OPENAI_MODEL_ID} questions={len(questions)}")
            return questions
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            print(f"ERROR OpenAI cloze generation failed: {exc}", flush=True)
    if provider in {"auto", "nvidia"}:
        try:
            questions = generate_questions_with_nvidia(rendered_prompt)
            print(f"cloze_provider=nvidia model={NVIDIA_MODEL_ID} questions={len(questions)}")
            return questions
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            print(f"ERROR NVIDIA cloze generation failed: {exc}", flush=True)
    raise RuntimeError("AI cloze generation failed: " + " | ".join(errors))


def build_uniqueness_review_request(raw_item: dict, row: pd.Series) -> str:
    sentence = str(row.get("English_Sentence", "")).strip()
    target_word = str(row.get("Word", "")).strip()
    meaning = str(row.get("Meaning", "")).strip()
    correct_option = str(raw_item.get("correct_option", "")).strip().upper()
    choice_map = {
        label: str(raw_item.get(f"choice_{label}", "")).strip()
        for label in OPTION_LABELS
    }
    return f"""
You are reviewing one English multiple-choice cloze question.
Your only job is to judge whether exactly one option is clearly correct in context.

Return JSON only with:
- verdict: "unique" or "ambiguous" or "invalid"
- plausible_options: array of option labels that a typical English teacher could still accept in context
- reason: short snake_case string
- notes: one short English sentence

Strict review rules:
- If two or more options could reasonably fit the sentence, verdict must be "ambiguous".
- If the intended correct answer is not clearly the only best answer, verdict must be "ambiguous".
- If the blank sentence does not match the source sentence context, verdict must be "invalid".
- Be conservative: if a distractor is still semantically possible, grammatically acceptable, or commonly used in the same sentence pattern, include it in plausible_options.
- For example, meal words like breakfast/lunch/dinner/snack in the same eating sentence are usually ambiguous unless the sentence strongly rules the others out.
- Only return verdict="unique" when plausible_options contains exactly one label and it is the intended correct option.

Question info:
- target_word: {target_word}
- target_meaning: {meaning}
- source_sentence: {sentence}
- blank_sentence: {str(raw_item.get("blank_sentence", "")).strip()}
- surface_word: {str(raw_item.get("surface_word", "")).strip()}
- correct_option: {correct_option}
- choice_A: {choice_map["A"]}
- choice_B: {choice_map["B"]}
- choice_C: {choice_map["C"]}
- choice_D: {choice_map["D"]}
""".strip()


def parse_uniqueness_review_payload(text: str, correct_option: str) -> dict | None:
    payload = json.loads(extract_json_text(text))
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"unique", "ambiguous", "invalid"}:
        return None
    plausible = payload.get("plausible_options") or []
    plausible_labels = []
    if isinstance(plausible, list):
        plausible_labels = [
            str(item).strip().upper()
            for item in plausible
            if str(item).strip().upper() in OPTION_LABELS
        ]
    if verdict == "unique" and plausible_labels != [correct_option]:
        verdict = "ambiguous"
    return {
        "verdict": verdict,
        "plausible_options": plausible_labels,
        "reason": str(payload.get("reason", "")).strip() or "review_result",
        "notes": str(payload.get("notes", "")).strip(),
    }


def review_question_uniqueness_with_gemini(raw_item: dict, row: pd.Series, correct_option: str) -> dict | None:
    response = client.models.generate_content(
        model=REVIEW_MODEL_ID,
        contents=build_uniqueness_review_request(raw_item, row),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    return parse_uniqueness_review_payload((getattr(response, "text", "") or "").strip(), correct_option)


def review_question_uniqueness_with_openai(raw_item: dict, row: pd.Series, correct_option: str) -> dict | None:
    return parse_uniqueness_review_payload(
        openai_json_response(build_uniqueness_review_request(raw_item, row), OPENAI_REVIEW_MODEL_ID),
        correct_option,
    )


def review_question_uniqueness(raw_item: dict, row: pd.Series, provider: str) -> dict:
    correct_option = str(raw_item.get("correct_option", "")).strip().upper()
    provider = normalize_review_provider(provider)
    errors = []
    if provider in {"auto", "gemini"}:
        try:
            payload = review_question_uniqueness_with_gemini(raw_item, row, correct_option)
            if payload:
                return payload
            errors.append("Gemini: review_parse_error")
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                return {
                    "verdict": "ambiguous",
                    "plausible_options": [],
                    "reason": "review_error",
                    "notes": str(exc),
                }
            if is_gemini_quota_error(exc):
                print("WARN Gemini review quota exhausted; switching to OpenAI review.", flush=True)
            else:
                print(f"WARN Gemini review failed; switching to OpenAI review: {exc}", flush=True)
    if provider in {"auto", "openai"}:
        try:
            payload = review_question_uniqueness_with_openai(raw_item, row, correct_option)
            if payload:
                return payload
            errors.append("OpenAI: review_parse_error")
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
        return {
            "verdict": "ambiguous",
            "plausible_options": [],
            "reason": "review_error",
            "notes": " | ".join(errors),
        }
    return {
        "verdict": "ambiguous",
        "plausible_options": [],
        "reason": "review_parse_error",
        "notes": " | ".join(errors),
    }


def validate_ai_question(raw_item: dict, fallback: dict, row: pd.Series) -> list[str]:
    reasons = []
    sentence = str(row.get("English_Sentence", "")).strip()
    word = str(row.get("Word", "")).strip()
    pos = str(row.get("POS", "")).strip()
    expected_blank = build_blank_sentence(sentence, word)
    expected_surface = expected_surface_word(word, pos, sentence)

    blank_sentence = str(raw_item.get("blank_sentence", "")).strip()
    if normalize_sentence_text(blank_sentence) != normalize_sentence_text(expected_blank):
        reasons.append("blank_sentence_not_from_source_sentence")
    if blank_sentence.count("____") != 1:
        reasons.append("blank_sentence_requires_exactly_one_blank")

    surface_word = str(raw_item.get("surface_word", "")).strip()
    if surface_word and normalize_word_token(surface_word) != normalize_word_token(expected_surface):
        reasons.append("surface_word_mismatch")

    choices = [str(raw_item.get(f"choice_{label}", "")).strip() for label in OPTION_LABELS]
    if any(not choice for choice in choices):
        reasons.append("empty_choice")
    normalized_choices = [normalize_word_token(choice) for choice in choices]
    if len(set(normalized_choices)) != len(normalized_choices):
        reasons.append("duplicate_choices")

    correct_option = str(raw_item.get("correct_option", "")).strip().upper()
    if correct_option not in OPTION_LABELS:
        reasons.append("invalid_correct_option")
    else:
        selected = str(raw_item.get(f"choice_{correct_option}", "")).strip()
        if normalize_word_token(selected) != normalize_word_token(expected_surface):
            reasons.append("correct_option_not_target_surface")

    if normalized_choices.count(normalize_word_token(expected_surface)) != 1:
        reasons.append("target_surface_not_unique_in_choices")

    explanation = str(raw_item.get("explanation", "")).strip()
    if not explanation:
        reasons.append("missing_explanation")

    if normalize_sentence_text(fallback["blank_sentence"]).startswith("Choose the best word: ____"):
        reasons.append("source_sentence_missing_target_word")

    return reasons


def normalize_ai_question(
    ep_num: int,
    idx: int,
    row: pd.Series,
    df: pd.DataFrame,
    raw_item: dict | None,
    review_provider: str,
) -> dict:
    fallback = deterministic_fallback_question(ep_num, idx, row, df)
    raw_item = raw_item or {}
    reasons = validate_ai_question(raw_item, fallback, row) if raw_item else ["missing_ai_item"]
    if not reasons:
        review = review_question_uniqueness(raw_item, row, review_provider)
        if review.get("verdict") != "unique":
            plausible = review.get("plausible_options") or []
            plausible_suffix = f"_{'-'.join(plausible)}" if plausible else ""
            reasons.append(f"review_{review.get('reason') or 'ambiguous'}{plausible_suffix}")
    if reasons:
        print(f"fallback_question ep={ep_num} q={idx} word={row.get('Word', '')} reasons={','.join(reasons)}")
        payload = fallback
    else:
        payload = {
            "surface_word": str(raw_item.get("surface_word", "")).strip() or fallback["surface_word"],
            "blank_sentence": str(raw_item.get("blank_sentence", "")).strip() or fallback["blank_sentence"],
            "choice_A": str(raw_item.get("choice_A", "")).strip(),
            "choice_B": str(raw_item.get("choice_B", "")).strip(),
            "choice_C": str(raw_item.get("choice_C", "")).strip(),
            "choice_D": str(raw_item.get("choice_D", "")).strip(),
            "correct_option": str(raw_item.get("correct_option", "")).strip().upper(),
            "explanation": str(raw_item.get("explanation", "")).strip() or fallback["explanation"],
        }

    return {
        "episode": ep_num,
        "question_no": idx,
        "question_id": f"Ep{ep_num:02d}_Q{idx:02d}",
        "word": str(row.get("Word", "")).strip(),
        "surface_word": payload["surface_word"],
        "pos": str(row.get("POS", "")).strip(),
        "meaning": str(row.get("Meaning", "")).strip(),
        "sentence": str(row.get("English_Sentence", "")).strip(),
        "blank_sentence": payload["blank_sentence"],
        "translation": str(row.get("Chinese_Translation", "")).strip(),
        "correct_option": payload["correct_option"],
        "correct_word": str(row.get("Word", "")).strip(),
        "correct_surface_word": payload[f"choice_{payload['correct_option']}"].strip(),
        "explanation": payload["explanation"],
        "choice_A": payload["choice_A"],
        "choice_B": payload["choice_B"],
        "choice_C": payload["choice_C"],
        "choice_D": payload["choice_D"],
    }


def build_question_rows(ep_num: int, df: pd.DataFrame, ai_questions: list[dict], review_provider: str) -> list[dict]:
    rows = []
    ai_questions = ai_questions or []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        raw_item = ai_questions[idx - 1] if idx - 1 < len(ai_questions) else None
        rows.append(normalize_ai_question(ep_num, idx, row, df, raw_item, review_provider))
    return rows


def save_questions(storyboard_dir: Path, questions: list[dict]):
    csv_path = storyboard_dir / "cloze_questions.csv"
    json_path = storyboard_dir / "cloze_questions.json"
    df = pd.DataFrame(questions)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"克漏字題已儲存：{csv_path}")
    print(f"克漏字題已儲存：{json_path}")


def main():
    parser = argparse.ArgumentParser(description="根據 vocab_data 與 shared prompt 產生克漏字題")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_PROVIDERS),
        default=os.getenv("CAP_CLOZE_LLM_PROVIDER", "auto"),
        help="LLM provider: auto tries Gemini first, then OpenAI, then NVIDIA.",
    )
    parser.add_argument(
        "--review-provider",
        choices=sorted(VALID_REVIEW_PROVIDERS),
        default=os.getenv("CAP_CLOZE_REVIEW_PROVIDER", "auto"),
        help="Review LLM provider: auto tries Gemini first, then OpenAI.",
    )
    args = parser.parse_args()
    provider = normalize_provider(args.provider)
    review_provider = normalize_review_provider(args.review_provider)

    episode_folder, start_word, end_word = resolve_episode_range(workspace_dir, args.ep)
    storyboard_dir = episode_folder / "03_storyboards"
    vocab_df = load_vocab_df(storyboard_dir)
    if vocab_df.empty:
        raise ValueError("vocab_data 為空，無法產生克漏字題。")

    prompt_path = ensure_prompt_file(storyboard_dir)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    rendered_prompt = render_prompt(prompt_template, args.ep, start_word, end_word, vocab_df)
    print(
        f"cloze_llm_provider={provider} review_provider={review_provider} "
        f"gemini_model={MODEL_ID} "
        f"openai_model={OPENAI_MODEL_ID} nvidia_model={NVIDIA_MODEL_ID} "
        f"review_gemini_model={REVIEW_MODEL_ID} "
        f"review_openai_model={OPENAI_REVIEW_MODEL_ID}",
        flush=True,
    )
    try:
        ai_questions = generate_questions_with_ai(rendered_prompt, provider)
    except Exception as exc:
        print(
            f"WARN AI cloze generation unavailable; using deterministic fallback questions: {exc}",
            flush=True,
        )
        ai_questions = []
    questions = build_question_rows(args.ep, vocab_df, ai_questions, review_provider)
    save_questions(storyboard_dir, questions)


if __name__ == "__main__":
    main()
