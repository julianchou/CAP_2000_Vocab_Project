import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path


SCRIPT_BRIEF = """故事生成模式 2.2 規格：
- 內容要像給 10 歲以下小朋友閱讀的故事書，不是摘要或製作備忘。
- 正文必須可以直接拿去生成語音。
- 旁白與角色對話都要自然放進正文中，不要只放在分開欄位。
- 正文不要出現不必要的英文或製作代碼，例如 Ep、P1、source_plot。
- 每段正文要有完整事件推進，包含環境描寫、角色動作、情緒變化和自然對話。
- 每段至少包含 3 個旁白段落與 3 到 5 句角色對話。
- 語氣要溫暖、明亮、可愛、安心，適合睡前故事或兒童繪本。
- 避免恐怖、暴力、死亡、成人議題、沉重創傷與複雜抽象說教。
"""


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def clean_title(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^Ep\s*\d+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^第\s*\d+\s*集\s*", "", text)
    return text.strip() or "童話故事"


def clean_tts_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\bEp\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bP\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_paragraph_title(text: str, idx: int) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^P\d+\s*", "", text, flags=re.IGNORECASE)
    return text or f"第{idx}段"


def pick_characters(selected_story: dict) -> list[str]:
    names = [str(x).strip() for x in (selected_story.get("main_characters") or []) if str(x).strip()]
    if len(names) >= 2:
        return names[:3]
    return ["小主角", "好朋友", "溫柔的幫手"]


def line(speaker: str, text: str, voice_hint: str) -> dict:
    return {"speaker": speaker, "text": text, "voice_hint": voice_hint}


def merge_storybook_text(narration: list[str], dialogues: list[dict]) -> str:
    parts = []
    for idx, block in enumerate(narration):
        parts.append(clean_tts_text(block))
        if idx < len(dialogues):
            item = dialogues[idx]
            parts.append(clean_tts_text(f"{item.get('speaker', '')}說：「{item.get('text', '')}」"))
    for item in dialogues[len(narration):]:
        parts.append(clean_tts_text(f"{item.get('speaker', '')}說：「{item.get('text', '')}」"))
    return "\n\n".join(part for part in parts if part)


def opening(lead: str, friend: str, setting: str, plot: str, summary: str) -> tuple[list[str], list[dict]]:
    narration = [
        f"清晨的光像一條柔軟的金色緞帶，輕輕鋪在{setting}。{lead}揉揉眼睛，從小小的被窩裡探出頭來，覺得今天的空氣聞起來有一點甜，也有一點亮。遠處傳來沙沙的聲音，像是樹葉正在說悄悄話，又像是有誰把一個小小的祕密藏進風裡。",
        f"{summary} {lead}慢慢走到門口，發現平常熟悉的地方變得不太一樣。那些會閃亮的小角落，今天好像少了一點光；那些總是開心打招呼的朋友，也都站在路邊，小聲討論著什麼。",
        f"{plot} 雖然事情有一點奇怪，{lead}並沒有害怕。因為這是一個溫柔的童話世界，只要願意聽一聽、看一看，再和朋友一起想辦法，很多問題都會慢慢露出可愛的答案。",
    ]
    dialogues = [
        line(lead, "咦？今天的光怎麼變得小小的？", "好奇、輕聲"),
        line(friend, "我也發現了。我們一起去看看，好不好？", "溫柔、陪伴"),
        line(lead, "好呀，有你陪我，我就不緊張了。", "安心、期待"),
        line(friend, "那我們慢慢走，路上的線索一定會告訴我們答案。", "穩定、鼓勵"),
    ]
    return narration, dialogues


def problem(lead: str, friend: str, helper: str, conflict: str, plot: str) -> tuple[list[str], list[dict]]:
    narration = [
        f"{lead}和{friend}沿著小路往前走。路旁的小花把頭低低垂著，小石子也不像平常那樣亮晶晶。走到一棵圓圓的大樹下時，他們終於發現了問題的開頭：{conflict}",
        f"{plot} {lead}皺起小眉毛，心裡有一點著急。可是，著急像一團打結的毛線，越拉越緊，反而更不容易找到線頭。{friend}看見了，便輕輕拍了拍{lead}的肩膀，提醒他先深深吸一口氣。",
        f"就在這時，{helper}從不遠處走了過來。{helper}沒有急著給答案，而是蹲下來，和大家一起看著地上的小線索。原來，問題不是因為誰壞壞，也不是因為誰故意搗蛋，而是大家都忘了把心裡的想法好好說出來。",
    ]
    dialogues = [
        line(lead, "我想快點把事情變好，可是我不知道從哪裡開始。", "困惑、誠實"),
        line(friend, "我們可以先把看到的事情說出來，一件一件慢慢想。", "安撫、清楚"),
        line(helper, "很多小問題，都喜歡躲在沒有說出口的心情裡喔。", "親切、智慧"),
        line(lead, "那我也要把我的心情說出來。", "鼓起勇氣"),
        line(friend, "我會好好聽你說。", "溫暖、支持"),
    ]
    return narration, dialogues


def teamwork(lead: str, friend: str, helper: str, plot: str) -> tuple[list[str], list[dict]]:
    narration = [
        f"聽完大家的想法以後，{lead}忽然覺得心裡亮了一點。原來，一個人想不到辦法的時候，可以把小小的想法放在一起，變成一個大大的好主意。{friend}負責找線索，{helper}負責提醒大家慢慢來，而{lead}負責把每個人的想法仔細記住。",
        f"{plot} 他們沒有跑得很快，也沒有大聲吵鬧，只是一步一步地試。每試一次，就更靠近答案一點；每互相鼓勵一次，心裡的小燈就亮一點。連路旁的小花也悄悄抬起頭，好像在替他們加油。",
        f"過了一會兒，大家終於找到了一個溫柔的辦法。這個辦法不是只讓一個人開心，而是讓每個人都可以安心地笑出來。{lead}看著朋友們，忽然明白：合作不是誰比較厲害，而是大家願意把自己的小力量放在同一個方向。",
    ]
    dialogues = [
        line(helper, "我有一個點子，我們可以把大家的小方法合在一起。", "活潑、有信心"),
        line(friend, "我來幫忙找需要的東西。", "主動、開心"),
        line(lead, "那我來記住順序，這樣我們就不會忘記了。", "認真、期待"),
        line(helper, "你看，大家一起做，事情就變得沒那麼難了。", "鼓勵、溫柔"),
        line(lead, "原來我的小力量，也可以幫上忙！", "驚喜、自信"),
    ]
    return narration, dialogues


def ending(lead: str, friend: str, helper: str, moral: str, plot: str) -> tuple[list[str], list[dict]]:
    narration = [
        f"夕陽把天空染成蜂蜜一樣的顏色，整個童話世界都安靜地笑了。{plot} 那些原本暗暗的小角落，又一點一點亮了起來，好像有許多小星星躲在裡面眨眼睛。",
        f"{lead}站在朋友中間，回想今天發生的事。剛開始，問題看起來像一座小山；可是當大家願意說出心情、願意分享、願意一起幫忙，那座小山就慢慢變成了一條可以牽手走過的小路。",
        f"回家的路上，{friend}和{helper}陪著{lead}慢慢走。風輕輕吹過，像是在替故事翻到最後一頁。{lead}把今天學到的事放進心裡，暖暖地記住：{moral}",
    ]
    dialogues = [
        line(lead, f"我今天學到了，{moral}", "溫暖、肯定"),
        line(friend, "而且你很勇敢，願意把心裡的話說出來。", "欣賞、真誠"),
        line(helper, "每一次溫柔的選擇，都會讓世界亮一點點。", "柔和、智慧"),
        line(lead, "明天，我也想把這份亮亮的心情分享給更多朋友。", "開心、期待"),
        line(friend, "那我們明天再一起出發吧！", "愉快、收尾"),
    ]
    return narration, dialogues


def build_paragraph_script(outline: dict, selected_story: dict) -> list[dict]:
    characters = pick_characters(selected_story)
    lead = characters[0]
    friend = characters[1] if len(characters) > 1 else "好朋友"
    helper = characters[2] if len(characters) > 2 else "溫柔的幫手"
    moral = clean_tts_text(outline.get("moral") or selected_story.get("moral", "分享會讓快樂變得更多。"))
    setting = clean_tts_text(selected_story.get("setting") or "明亮的童話森林")
    conflict = clean_tts_text(selected_story.get("conflict") or "大家遇到了一個需要合作的小問題。")

    scripts = []
    paragraphs = outline.get("paragraphs") or []
    for idx, paragraph in enumerate(paragraphs, start=1):
        title = clean_paragraph_title(paragraph.get("title", ""), idx)
        plot = clean_tts_text(paragraph.get("plot", ""))
        summary = clean_tts_text(paragraph.get("summary", ""))
        if idx == 1:
            narration, dialogues = opening(lead, friend, setting, plot, summary)
        elif idx == 2:
            narration, dialogues = problem(lead, friend, helper, conflict, plot)
        elif idx == 3:
            narration, dialogues = teamwork(lead, friend, helper, plot)
        else:
            narration, dialogues = ending(lead, friend, helper, moral, plot)

        scripts.append(
            {
                "paragraph_id": str(idx),
                "title": title,
                "source_plot": plot,
                "storybook_text": merge_storybook_text(narration, dialogues),
                "narration": [clean_tts_text(item) for item in narration],
                "dialogues": dialogues,
                "estimated_reading_level": "10歲以下兒童故事書",
            }
        )
    return scripts


def render_markdown(payload: dict) -> str:
    lines = [
        f"# {payload.get('title', '')}",
        "",
        f"- 目標年齡：{payload.get('target_age', '')}",
        f"- 故事調性：{payload.get('tone', '')}",
        f"- 正向寓意：{payload.get('moral', '')}",
        "",
        "## 可直接生成語音的故事正文",
    ]
    for paragraph in payload.get("paragraphs") or []:
        lines.extend(["", f"## {paragraph.get('title', '')}", "", paragraph.get("storybook_text", "")])
    lines.append("")
    return "\n".join(lines)


def write_script(ep_path: Path, outline: dict, selected_story: dict) -> tuple[Path, Path]:
    paragraphs = build_paragraph_script(outline, selected_story)
    clean_story_title = clean_title(outline.get("title") or selected_story.get("title", ""))
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "brief": SCRIPT_BRIEF,
        "source_story_id": outline.get("source_story_id") or selected_story.get("id", ""),
        "source_title": outline.get("title") or selected_story.get("title", ""),
        "title": clean_story_title,
        "target_age": outline.get("target_age") or selected_story.get("target_age", "10歲以下"),
        "tone": outline.get("tone") or selected_story.get("tone", "溫暖、明亮、可愛、安心"),
        "moral": outline.get("moral") or selected_story.get("moral", ""),
        "paragraphs": paragraphs,
        "full_story_text": clean_tts_text("\n\n".join([clean_story_title] + [p["storybook_text"] for p in paragraphs])),
    }
    story_dir = ep_path / "02_story"
    story_dir.mkdir(parents=True, exist_ok=True)
    json_path = story_dir / "story_script.json"
    md_path = story_dir / "story_script.md"
    txt_path = story_dir / "story_text_for_tts.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    txt_path.write_text(payload["full_story_text"], encoding="utf-8")
    return md_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 2.2: write TTS-ready storybook text.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    selected_story = read_json(ep_path / "01_preproduction" / "selected_story.json")
    outline = read_json(ep_path / "02_story" / "story_outline.json")
    md_path, json_path = write_script(ep_path, outline, selected_story)
    print(f"story_script_md={md_path}")
    print(f"story_script_json={json_path}")
    print(f"story_tts_text={ep_path / '02_story' / 'story_text_for_tts.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
