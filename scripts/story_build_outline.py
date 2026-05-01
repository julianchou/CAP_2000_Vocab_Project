import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def load_selected_story(ep_path: Path) -> dict:
    selected_path = ep_path / "01_preproduction" / "selected_story.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"selected story not found: {selected_path}")
    data = json.loads(selected_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("title"):
        raise ValueError(f"invalid selected story: {selected_path}")
    return data


def build_paragraphs(story: dict) -> list[dict]:
    title = story.get("title", "童話故事")
    characters = "、".join(story.get("main_characters") or [])
    setting = story.get("setting", "")
    moral = story.get("moral", "")
    acts = list(story.get("three_act_outline") or [])
    while len(acts) < 3:
        acts.append(story.get("logline", "主角和朋友一起完成一場溫暖的小冒險。"))

    return [
        {
            "paragraph_id": "P1",
            "title": "開場：進入童話世界",
            "purpose": "介紹主角、場景與故事的溫暖氣氛。",
            "summary": f"故事《{title}》從{setting}開始。{characters}登場，觀眾能快速知道這是一個安心、可愛、適合小朋友的童話世界。",
            "plot": acts[0],
            "narration_focus": "用簡單句子描述主角的日常、願望和可愛的小問題。",
            "dialogue_focus": "角色用短句互相打招呼，表現友善與好奇。",
        },
        {
            "paragraph_id": "P2",
            "title": "發展：遇到小困難",
            "purpose": "讓主角遇到安全、容易理解的小挑戰。",
            "summary": f"主角遇到的困難是：{story.get('conflict', '')}。這個衝突不驚嚇、不危險，重點是讓孩子理解角色的心情。",
            "plot": acts[1],
            "narration_focus": "描述主角嘗試解決問題，但一開始還沒有找到好方法。",
            "dialogue_focus": "朋友提出溫柔建議，鼓勵主角說出感受或尋求幫忙。",
        },
        {
            "paragraph_id": "P3",
            "title": "轉折：朋友一起想辦法",
            "purpose": "強化合作、同理心與正向選擇。",
            "summary": "主角不再只靠自己，而是聽聽朋友的想法。大家用溫柔、聰明、可愛的方法一起解決問題。",
            "plot": "主角和朋友把各自的小能力合在一起，找到比一個人更好的辦法。",
            "narration_focus": "描述團隊合作的過程，讓每個角色都有一個小小貢獻。",
            "dialogue_focus": "角色輪流說出鼓勵、道歉、感謝或合作的句子。",
        },
        {
            "paragraph_id": "P4",
            "title": "結尾：溫暖圓滿",
            "purpose": "收束故事並點出寓意。",
            "summary": f"問題被溫柔解決，大家都獲得安心和快樂。故事最後帶出寓意：{moral}",
            "plot": acts[2],
            "narration_focus": "用溫暖語氣描述圓滿結局，讓孩子感到安心。",
            "dialogue_focus": "主角說出學到的事，朋友們用簡短句子回應與祝福。",
        },
    ]


def write_outline(ep_path: Path, story: dict) -> tuple[Path, Path]:
    story_dir = ep_path / "02_story"
    story_dir.mkdir(parents=True, exist_ok=True)
    paragraphs = build_paragraphs(story)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_story_id": story.get("id", ""),
        "title": story.get("title", ""),
        "genre": story.get("genre", "童話故事"),
        "target_age": story.get("target_age", "10歲以下"),
        "tone": story.get("tone", "溫暖、明亮、可愛、安心"),
        "logline": story.get("logline", ""),
        "moral": story.get("moral", ""),
        "safety_rules": story.get("safety_rules", []),
        "paragraphs": paragraphs,
    }

    json_path = story_dir / "story_outline.json"
    md_path = story_dir / "story_outline.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return md_path, json_path


def render_markdown(payload: dict) -> str:
    lines = [
        f"# {payload.get('title', '')}",
        "",
        f"- 類型：{payload.get('genre', '')}",
        f"- 目標年齡：{payload.get('target_age', '')}",
        f"- 故事調性：{payload.get('tone', '')}",
        f"- 故事一句話：{payload.get('logline', '')}",
        f"- 正向寓意：{payload.get('moral', '')}",
        "",
        "## 兒童適齡規則",
    ]
    lines.extend([f"- {item}" for item in payload.get("safety_rules") or []])
    lines.extend(["", "## 段落大綱"])
    for paragraph in payload.get("paragraphs") or []:
        lines.extend(
            [
                "",
                f"### {paragraph.get('paragraph_id', '')} {paragraph.get('title', '')}",
                f"- 目的：{paragraph.get('purpose', '')}",
                f"- 摘要：{paragraph.get('summary', '')}",
                f"- 劇情：{paragraph.get('plot', '')}",
                f"- 旁白重點：{paragraph.get('narration_focus', '')}",
                f"- 對話重點：{paragraph.get('dialogue_focus', '')}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 2.1: build story outline and paragraphs.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    story = load_selected_story(ep_path)
    md_path, json_path = write_outline(ep_path, story)
    print(f"story_outline_md={md_path}")
    print(f"story_outline_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
