import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from episode_range_utils import resolve_episode_range


base_dir = Path(__file__).resolve().parent.parent
core_assets_dir = base_dir / "core" / "assets"
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))
OPTION_LABELS = ["A", "B", "C", "D"]
TEMPLATE_PATH = core_assets_dir / "cloze_flashcard_template.png"


def find_font_candidates():
    return [
        core_assets_dir / "msjhbd.ttc",
        core_assets_dir / "msjh.ttc",
        Path("C:\\Windows\\Fonts\\msjhbd.ttc"),
        Path("C:\\Windows\\Fonts\\msjh.ttc"),
        Path("C:\\Windows\\Fonts\\arialbd.ttf"),
    ]


def get_font(size: int):
    for path in find_font_candidates():
        if Path(path).exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, initial_size: int, min_size: int = 20):
    for size in range(initial_size, min_size - 1, -2):
        font = get_font(size)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=12)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
    return get_font(min_size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if not text:
        return ""
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def is_ascii_word_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "'_-")


def mixed_text_tokens(text: str) -> list[str]:
    tokens = []
    current = []
    for char in text:
        if is_ascii_word_char(char):
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
        tokens.append(" " if char.isspace() else char)
    if current:
        tokens.append("".join(current))
    return tokens


def wrap_mixed_line(text: str, width: int) -> str:
    lines = []
    current = ""
    for token in mixed_text_tokens(text):
        if token == " ":
            if current and not current.endswith(" "):
                current += " "
            continue

        candidate = f"{current}{token}"
        if current and len(candidate.rstrip()) > width:
            lines.append(current.rstrip())
            current = token
        else:
            current = candidate

    if current:
        lines.append(current.rstrip())
    return "\n".join(lines)


def wrap_mixed_text(text: str, width: int) -> str:
    if not text:
        return ""
    parts = text.splitlines() or [text]
    wrapped = [wrap_mixed_line(part, width) for part in parts]
    return "\n".join(wrapped)


def load_questions(storyboard_dir: Path) -> list[dict]:
    json_path = storyboard_dir / "cloze_questions.json"
    csv_path = storyboard_dir / "cloze_questions.csv"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    if csv_path.exists():
        import pandas as pd
        return pd.read_csv(csv_path).fillna("").to_dict(orient="records")
    raise FileNotFoundError(f"找不到克漏字題目檔：{json_path}")


def draw_centered_multiline(draw: ImageDraw.ImageDraw, text: str, font, fill: str, box: tuple[int, int, int, int], spacing: int = 12):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + max((box[2] - box[0] - text_w) // 2, 0)
    y = box[1] + max((box[3] - box[1] - text_h) // 2, 0)
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def option_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    x1 = int(width * 0.785)
    x2 = int(width * 0.978)
    box_h = int(height * 0.128)
    gap = int(height * 0.03)
    start_y = int(height * 0.245)
    boxes = {}
    for idx, label in enumerate(OPTION_LABELS):
        top = start_y + idx * (box_h + gap)
        boxes[label] = (x1, top, x2, top + box_h)
    return boxes


def render_card(question: dict, output_path: Path, show_answer: bool):
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"找不到克漏字模板：{TEMPLATE_PATH}")

    img = Image.open(TEMPLATE_PATH).convert("RGBA")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    white_box = (int(width * 0.065), int(height * 0.12), int(width * 0.72), int(height * 0.88))
    question_box = (white_box[0] + 60, white_box[1] + 160, white_box[2] - 60, white_box[1] + int(height * 0.28))
    hint_box = (white_box[0] + 60, white_box[1] + int(height * 0.31), white_box[2] - 60, white_box[1] + int(height * 0.47))
    answer_box = (white_box[0] + 440, white_box[1] + int(height * 0.40), white_box[2] - 90, white_box[3] - 90)

    gold = "#B8944E"
    navy = "#20385C"
    gray = "#5D6877"
    highlight_fill = (184, 148, 78, 35)

    header_font = get_font(80)
    draw.text((white_box[0] + 28, white_box[1] + 20), question["question_id"], fill=gold, font=header_font)

    q_draw = ImageDraw.Draw(img)
    question_text = wrap_mixed_text(f"{question['blank_sentence']}", 40)
    q_font = fit_font(q_draw, question_text, question_box[2] - question_box[0], 80, 30)
    draw.multiline_text((question_box[0], question_box[1]), question_text, fill=navy, font=q_font, spacing=18)

    # 詞性和中文提示暫不顯示，避免干擾學生思考
    # hint_text = wrap_mixed_text(f"詞性：{question['pos']}\n中文提示：{question['translation']}", 50)
    # hint_font = fit_font(q_draw, hint_text, hint_box[2] - hint_box[0], 34, 22)
    # draw.multiline_text((hint_box[0], hint_box[1]), hint_text, fill=gray, font=hint_font, spacing=12)

    boxes = option_boxes(width, height)
    for label in OPTION_LABELS:
        option_word = str(question.get(f"choice_{label}", "")).strip()
        option_text = wrap_mixed_text(option_word, 14)
        box = boxes[label]
        font = fit_font(q_draw, option_text, (box[2] - box[0]) - 90, 42, 24)
        if show_answer and label == question["correct_option"]:
            draw.rounded_rectangle(box, radius=18, outline=gold, width=8, fill=highlight_fill)
        text_box = (box[0] + 70, box[1] + 16, box[2] - 24, box[3] - 16)
        draw_centered_multiline(draw, option_text, font, "#F5E5B7", text_box, spacing=8)

    if show_answer:
        answer_title = f"正解：{question['correct_option']}. {question['correct_word']}"
        answer_font = fit_font(q_draw, answer_title, answer_box[2] - answer_box[0], 50, 28)
        draw.text((answer_box[0], answer_box[1]), answer_title, fill=gold, font=answer_font)

        explanation_text = wrap_mixed_text(f"解析：{question['explanation']}", 20)
        explain_font = fit_font(q_draw, explanation_text, answer_box[2] - answer_box[0], 52, 32)
        draw.multiline_text((answer_box[0], answer_box[1] + 70), explanation_text, fill=navy, font=explain_font, spacing=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output_path)
    img.close()


def main():
    parser = argparse.ArgumentParser(description="依克漏字題目產出題目卡與答案卡")
    parser.add_argument("--ep", type=int, required=True)
    args = parser.parse_args()

    episode_folder, _start, _end = resolve_episode_range(workspace_dir, args.ep)
    storyboard_dir = episode_folder / "03_storyboards"
    questions = load_questions(storyboard_dir)

    question_dir = episode_folder / "04_images" / "cloze_cards" / "questions"
    answer_dir = episode_folder / "04_images" / "cloze_cards" / "answers"

    for item in questions:
        safe_word = str(item["word"]).replace(" ", "_").replace("/", "_")
        stem = f"{item['question_id']}_{safe_word}"
        render_card(item, question_dir / f"{stem}.png", show_answer=False)
        render_card(item, answer_dir / f"{stem}_answer.png", show_answer=True)

    print(f"題目卡已輸出：{question_dir}")
    print(f"答案卡已輸出：{answer_dir}")


if __name__ == "__main__":
    main()
