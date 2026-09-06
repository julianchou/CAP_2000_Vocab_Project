import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

from episode_range_utils import resolve_episode_range


load_dotenv()
client = genai.Client()

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(BASE_DIR / "workspace")))
HOST_PROFILE_PATH = BASE_DIR / "core" / "assets" / "host_profiles.json"
TEXT_MODEL = os.environ.get("CAP_COVER_PROMPT_MODEL", "gemini-2.5-flash")

DEFAULT_PROMPT_TEMPLATE = """請為《會考英文隨身聽》設計一張全新的 YouTube 封面圖。

【封面目標】
- 用可愛、吸睛、適合國中英文學習頻道的風格，設計 16:9 橫式封面。
- John 與 Mary 必須同時出現在封面中，並保持角色一致性。
- 請根據本集單字挑選 4 到 6 個最適合視覺化的元素融入畫面，不要把 10 個單字全部塞成一堆文字。
- 畫面要簡潔、集中、容易一眼看懂，不要過度擁擠。
- 適度加入「第 {{EPISODE}} 集」的視覺提示，但避免太多小字。
- 不要做成真實照片，請維持可愛 3D 卡通、明亮、討喜、適合兒少教育內容的質感。

【本集單字資料】
{{VOCAB_LIST}}

【John & Mary 角色設定 JSON】
{{HOST_PROFILE_JSON}}

【額外要求】
- 如果有動物、腳踏車、生日、biology 等元素，優先以童趣、可愛、正向的方式呈現。
- 避免暴力、驚悚、陰暗、過度寫實、複雜背景、凌亂排版。
- 最終請直接輸出一段可拿來產圖的完整 prompt 內容，不要解說，不要分點，不要加 Markdown。
"""


def detect_profile_id() -> str:
    profile_id = str(os.environ.get("CAP_PROFILE_ID", "")).strip()
    if profile_id:
        return profile_id
    if WORKSPACE_DIR.parent.name == "workspaces" and WORKSPACE_DIR.name:
        return WORKSPACE_DIR.name
    return "default"


def shared_prompt_dir() -> Path:
    return BASE_DIR / "config" / detect_profile_id() / "prompts"


def shared_prompt_path() -> Path:
    return shared_prompt_dir() / "cover_image_prompt.txt"


def cover_prompt_mode_path() -> Path:
    return shared_prompt_dir() / "cover_prompt_mode.txt"


def ensure_prompt_template() -> Path:
    prompt_path = shared_prompt_path()
    if not prompt_path.exists():
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def ensure_cover_prompt_mode_file() -> Path:
    mode_path = cover_prompt_mode_path()
    if not mode_path.exists():
        mode_path.parent.mkdir(parents=True, exist_ok=True)
        mode_path.write_text("direct\n", encoding="utf-8")
    return mode_path


def read_cover_prompt_mode() -> str:
    mode_path = ensure_cover_prompt_mode_file()
    value = str(mode_path.read_text(encoding="utf-8")).strip().lower()
    return value if value in {"direct", "gemini"} else "direct"


def load_vocab_df(storyboard_dir: Path) -> pd.DataFrame:
    csv_path = storyboard_dir / "vocab_data.csv"
    json_path = storyboard_dir / "vocab_data.json"
    if csv_path.exists():
        return pd.read_csv(csv_path).fillna("")
    if json_path.exists():
        return pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8"))).fillna("")
    raise FileNotFoundError(f"找不到 vocab_data.csv 或 vocab_data.json：{storyboard_dir}")


def build_vocab_list_text(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(f"- 單字：{row.get('Word', '')} ({row.get('POS', '')}) {row.get('Meaning', '')}")
        lines.append(f"  > 英文例句：{row.get('English_Sentence', '')}")
        lines.append(f"  > 中文翻譯：{row.get('Chinese_Translation', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def load_host_profile_json_text() -> str:
    if not HOST_PROFILE_PATH.exists():
        return "{}"
    try:
        payload = json.loads(HOST_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return "{}"
    return json.dumps(payload, ensure_ascii=False, indent=2)

def read_youtube_title(episode_folder: Path) -> str:
    info_path = episode_folder / "video_info.json"
    if not info_path.exists():
        return ""
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("youtube_title") or "").strip() if isinstance(payload, dict) else ""


def render_prompt(
    template_text: str,
    ep_num: int,
    vocab_list_text: str,
    host_profile_json_text: str,
    youtube_title: str = "",
) -> str:
    return (
        template_text
        .replace("{{EPISODE}}", str(ep_num))
        .replace("{{VOCAB_LIST}}", vocab_list_text)
        .replace("{{HOST_PROFILE_JSON}}", host_profile_json_text)
        .replace("{{YOUTUBE_TITLE}}", youtube_title)
    ).strip() + "\n"


def save_prompt_text(path: Path, prompt_text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt_text, encoding="utf-8")
    return path


def clean_model_output(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```text", "").replace("```markdown", "").replace("```", "").strip()
    return cleaned


def generate_ai_cover(ep_num: int) -> None:
    episode_folder, _start_word, _end_word = resolve_episode_range(WORKSPACE_DIR, ep_num)
    storyboard_dir = episode_folder / "03_storyboards"
    images_dir = episode_folder / "04_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    prompt_template_path = ensure_prompt_template()
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    cover_prompt_mode = read_cover_prompt_mode()
    vocab_df = load_vocab_df(storyboard_dir)
    rendered_prompt = render_prompt(
        prompt_template,
        ep_num,
        build_vocab_list_text(vocab_df),
        load_host_profile_json_text(),
        read_youtube_title(episode_folder),
    )

    prompt_output_path = images_dir / f"cover_prompt_ep{ep_num:02d}.txt"
    print(f"prompt_template_path={prompt_template_path}")
    print(f"cover_prompt_mode={cover_prompt_mode}")
    print(f"prompt_output_path={prompt_output_path}")
    print(f"request_prompt_length={len(rendered_prompt)}")

    if cover_prompt_mode == "gemini":
        print(f"text_model={TEXT_MODEL}")
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=rendered_prompt,
        )
        output_prompt = clean_model_output(response.text)
        if not output_prompt:
            raise ValueError("Gemini 沒有回傳可用的 cover prompt")
        print("RESULT=cover_prompt_generated_via_gemini")
    else:
        output_prompt = rendered_prompt
        print("RESULT=cover_prompt_rendered_directly")

    save_prompt_text(prompt_output_path, output_prompt)
    print(f"output_prompt_length={len(output_prompt)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI cover prompt")
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()
    generate_ai_cover(args.ep)


if __name__ == "__main__":
    main()
