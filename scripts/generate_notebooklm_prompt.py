import argparse
import json
import os
from pathlib import Path

import pandas as pd

from episode_range_utils import resolve_episode_range


base_dir = Path(__file__).resolve().parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

def detect_profile_id() -> str:
    profile_id = str(os.environ.get("CAP_PROFILE_ID", "")).strip()
    if profile_id:
        return profile_id
    if workspace_dir.parent.name == "workspaces" and workspace_dir.name:
        return workspace_dir.name
    return "default"

def shared_prompt_path() -> Path:
    return base_dir / "config" / detect_profile_id() / "prompts" / "notebooklm_prompt_template.txt"

DEFAULT_PROMPT_TEMPLATE = """【節目設定】
這是一集《會考英文隨身聽》節目，主題是國中會考英文 2000 單字教學。
集數：第 {{EPISODE}} 集
單字範圍：{{START_WORD}} 到 {{END_WORD}}

【角色設定】
請用雙主持人對話方式撰寫節目摘要與導聽 prompt：
- Mary：溫柔、清楚、善於整理重點
- John：自然、口語、會補充例句與學習提醒

【節目流程要求】
1. 用自然口語的雙人對話介紹本集單字。
2. 依單字原始順序逐字帶出，不可漏字。
3. 每個單字都要包含詞性、中文意思、英文例句重點與中文提示。
4. 避免只是在念表格，要像節目內容，有銜接、有互動。
5. 結尾簡短鼓勵聽眾持續複習。

{{PREVIOUS_EPISODE_CLOZE_SECTION}}

{{NEXT_EPISODE_CLOZE_SECTION}}

【本集單字資料】
{{VOCAB_LIST}}
"""


def prompt_template_path_for(_target_folder: Path) -> Path:
    return shared_prompt_path()


def ensure_prompt_template(target_folder: Path) -> Path:
    prompt_path = prompt_template_path_for(target_folder)
    if not prompt_path.exists():
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def load_first_cloze_question(target_folder: Path) -> dict | None:
    storyboard_dir = target_folder / "03_storyboards"
    json_path = storyboard_dir / "cloze_questions.json"
    csv_path = storyboard_dir / "cloze_questions.csv"
    try:
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload[0]
        if csv_path.exists():
            df = pd.read_csv(csv_path).fillna("")
            if not df.empty:
                return df.iloc[0].to_dict()
    except Exception:
        return None
    return None


def build_previous_cloze_section(ep_num: int) -> str:
    if ep_num <= 1:
        return ""
    try:
        previous_folder, _prev_start, _prev_end = resolve_episode_range(workspace_dir, ep_num - 1)
    except Exception:
        return ""
    question = load_first_cloze_question(previous_folder)
    if not question:
        return ""

    return f"""【上一集克漏字解答安排】
在正式介紹本集單字前，先安排一段「上一集節目結束前的克漏字題解答」。
- 一開場就清楚說明：這一題是上一集節目結束前出給同學的，現在要公布答案。
- 請先重述題目，再逐一念出選項。
- 接著公布正確答案，並做詳細解說：說明為什麼這個答案最適合句意，也可以簡短排除其他選項。
- 解析完再自然銜接回本集單字教學。

上一集題目資料：
- 來自第 {ep_num - 1} 集的第 1 題
- 題目：{question.get("blank_sentence", "")}
- A：{question.get("choice_A", "")}
- B：{question.get("choice_B", "")}
- C：{question.get("choice_C", "")}
- D：{question.get("choice_D", "")}
- 正確答案：{question.get("correct_option", "")}. {question.get("correct_surface_word", question.get("surface_word", ""))}
- 解說重點：{question.get("explanation", "")}"""


def build_next_cloze_section(target_folder: Path, ep_num: int) -> str:
    question = load_first_cloze_question(target_folder)
    if not question:
        return ""

    return f"""【本集結尾克漏字預告安排】
在節目結束前，請從本集克漏字題庫取出第 1 題當作節目尾聲的小測驗。
- 請明確說明：這題是今天節目結束前要留給同學作答的題目，答案會在下一集開頭公布。
- 請把題目和四個選項完整念出來。
- 不要直接公布正確答案。
- 結尾要提醒同學先自己想想看，下一集會正式揭曉並詳細解析。

本集要出的題目資料：
- 來自第 {ep_num} 集的第 1 題
- 題目：{question.get("blank_sentence", "")}
- A：{question.get("choice_A", "")}
- B：{question.get("choice_B", "")}
- C：{question.get("choice_C", "")}
- D：{question.get("choice_D", "")}"""


def build_vocab_list_str(df: pd.DataFrame) -> str:
    vocab_lines = []
    for _, row in df.iterrows():
        vocab_lines.append(f"- 單字：{row['Word']} ({row['POS']}) {row['Meaning']}")
        vocab_lines.append(f"  > 英文例句：{row['English_Sentence']}")
        vocab_lines.append(f"  > 中文翻譯：{row['Chinese_Translation']}")
        vocab_lines.append("")
    return "\n".join(vocab_lines).strip()


def render_prompt(
    template_text: str,
    ep_num: int,
    start_num: int,
    end_num: int,
    vocab_list_str: str,
    previous_cloze_section: str,
    next_cloze_section: str,
) -> str:
    rendered = (
        template_text
        .replace("{{EPISODE}}", str(ep_num))
        .replace("{{START_WORD}}", str(start_num))
        .replace("{{END_WORD}}", str(end_num))
        .replace("{{VOCAB_LIST}}", vocab_list_str.strip())
        .replace("{{PREVIOUS_EPISODE_CLOZE_SECTION}}", previous_cloze_section.strip())
        .replace("{{NEXT_EPISODE_CLOZE_SECTION}}", next_cloze_section.strip())
    )

    fallback_sections = []
    if "{{PREVIOUS_EPISODE_CLOZE_SECTION}}" not in template_text and previous_cloze_section.strip():
        fallback_sections.append(previous_cloze_section.strip())
    if "{{NEXT_EPISODE_CLOZE_SECTION}}" not in template_text and next_cloze_section.strip():
        fallback_sections.append(next_cloze_section.strip())
    if fallback_sections:
        rendered += "\n\n【額外節目安排】\n" + "\n\n".join(fallback_sections)
    return rendered.strip() + "\n"


def generate_notebooklm_instruction(ep_num: int):
    target_folder, start_num, end_num = resolve_episode_range(workspace_dir, ep_num)
    vocab_csv = target_folder / "03_storyboards" / "vocab_data.csv"
    if not vocab_csv.exists():
        print(f"找不到單字表：{vocab_csv}")
        return

    df = pd.read_csv(vocab_csv).fillna("")
    vocab_list_str = build_vocab_list_str(df)
    previous_cloze_section = build_previous_cloze_section(ep_num)
    next_cloze_section = build_next_cloze_section(target_folder, ep_num)

    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    full_prompt = render_prompt(
        prompt_template,
        ep_num,
        start_num,
        end_num,
        vocab_list_str,
        previous_cloze_section,
        next_cloze_section,
    )

    output_path = target_folder / f"notebooklm_prompt_ep{ep_num:02d}.txt"
    output_path.write_text(full_prompt, encoding="utf-8")
    print(f"已產生第 {ep_num:02d} 集 NotebookLM Prompt")
    print(f"輸出檔案：{output_path}")
    if previous_cloze_section:
        print("已加入上一集克漏字第一題解析段落。")
    if next_cloze_section:
        print("已加入本集結尾克漏字第一題預告段落。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True, help="要處理的集數")
    args = parser.parse_args()
    generate_notebooklm_instruction(args.ep)
