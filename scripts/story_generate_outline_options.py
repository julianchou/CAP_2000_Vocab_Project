import argparse
import json
import os
from datetime import datetime
from pathlib import Path


STORY_OUTLINE_BRIEF = """故事生成模式 1.2 規格：
- 類型必須是童話故事。
- 目標觀眾是 10 歲以下小朋友。
- 故事要溫暖、清楚、好理解，避免驚嚇、黑暗、暴力、死亡、複雜戀愛、成人壓力或沉重創傷。
- 主角可以是小朋友、動物、玩具、精靈、雲朵、星星、植物或友善的奇幻角色。
- 每個故事都要有明確的正向寓意，例如分享、勇敢、誠實、合作、同理心、守信用、保護自然。
- 衝突要小而安全，例如迷路、忘記約定、不敢嘗試、朋友誤會、需要合作完成任務。
- 畫面方向要明亮、柔和、可愛，適合兒童繪本或 3D 童話動畫。
"""


def find_episode_path(workspace_root: Path, ep_num: int) -> Path:
    target = next(workspace_root.glob(f"Ep{ep_num:02d}_*"), None)
    if target is None:
        raise FileNotFoundError(f"episode folder not found for ep={ep_num}")
    return target


def option(
    option_id: str,
    ep_num: int,
    title: str,
    logline: str,
    characters: list[str],
    setting: str,
    conflict: str,
    outline: list[str],
    moral: str,
    visual_direction: str,
) -> dict:
    return {
        "id": option_id,
        "title": f"Ep{ep_num:02d} {title}",
        "genre": "童話故事",
        "target_age": "10歲以下",
        "tone": "溫暖、明亮、可愛、安心",
        "safety_rules": [
            "不使用恐怖、暴力、死亡、血腥或成人議題",
            "衝突保持輕量，結尾必須正向圓滿",
            "句子與情節要適合兒童理解",
        ],
        "logline": logline,
        "main_characters": characters,
        "setting": setting,
        "conflict": conflict,
        "three_act_outline": outline,
        "moral": moral,
        "visual_direction": visual_direction,
    }


def build_outline_options(ep_num: int) -> list[dict]:
    return [
        option(
            "A",
            ep_num,
            "月亮餅乾森林",
            "小兔米米發現森林裡的月亮餅乾不見了，於是和朋友們一起找回分享的快樂。",
            ["小兔米米", "松鼠可可", "月亮奶奶"],
            "會發光的餅乾森林、柔軟草地、銀色月光小路",
            "大家都想把月亮餅乾留給自己，結果森林的光變暗了。",
            [
                "米米發現月亮餅乾樹不再發光，森林朋友們都很擔心。",
                "大家一路尋找原因，才知道每個人都偷偷藏了一小塊餅乾。",
                "朋友們把餅乾放回樹下分享，森林重新亮起溫柔月光。",
            ],
            "分享會讓快樂變得更多。",
            "soft pastel fairy-tale forest, cute animals, glowing moon cookies, warm bedtime story style",
        ),
        option(
            "B",
            ep_num,
            "彩虹小郵差",
            "小雲朵波波想把彩虹信送到山谷，卻需要學會向朋友開口求助。",
            ["小雲朵波波", "小鳥亮亮", "彩虹郵局長"],
            "天空郵局、棉花糖雲朵、彩虹橋、花朵山谷",
            "波波覺得自己應該一個人完成任務，但彩虹信太重了。",
            [
                "波波第一次當郵差，興奮地背起一袋彩虹信。",
                "途中風變大，波波不敢求助，差點錯過送信時間。",
                "小鳥和風鈴花一起幫忙，大家合力把彩虹送到山谷。",
            ],
            "遇到困難時，請朋友幫忙也是勇敢。",
            "bright sky kingdom, fluffy clouds, rainbow bridge, cheerful picture-book animation",
        ),
        option(
            "C",
            ep_num,
            "不會唱歌的小星星",
            "小星星露露以為自己唱得不好聽，最後發現自己的閃光也是一種美妙音樂。",
            ["小星星露露", "月光老師", "螢火蟲合唱團"],
            "星星學校、夜空舞台、螢火蟲花園",
            "露露害怕在晚會上表演，因為她唱不出和別人一樣的歌。",
            [
                "星星學校準備夜空晚會，每顆星星都要表演一首歌。",
                "露露練習很久仍然害羞，螢火蟲朋友陪她找自己的節奏。",
                "晚會上，露露用閃閃發亮的節拍帶領大家完成最美的合唱。",
            ],
            "每個人都有不一樣的亮點。",
            "gentle night sky, smiling stars, fireflies, magical but cozy children's fairy tale",
        ),
        option(
            "D",
            ep_num,
            "種子王國的慢慢花",
            "小熊豆豆急著讓花開，卻在照顧慢慢花的過程中學會等待和耐心。",
            ["小熊豆豆", "慢慢花種子", "蝴蝶園丁"],
            "種子王國、蜂蜜小屋、彩色花圃",
            "豆豆每天催種子快點長大，反而讓小種子更緊張。",
            [
                "豆豆得到一顆慢慢花種子，希望明天就能看到大花園。",
                "蝴蝶園丁教他澆水、唱歌、等待，豆豆慢慢學會照顧。",
                "慢慢花終於開放，花瓣上藏著豆豆每天的溫柔努力。",
            ],
            "耐心和照顧會讓美好的事慢慢長大。",
            "sunny garden kingdom, cute bear cub, colorful flowers, soft watercolor storybook look",
        ),
        option(
            "E",
            ep_num,
            "玩具船的小小海",
            "一艘玩具船想成為真正的大船，最後明白陪伴小主人也能完成重要冒險。",
            ["玩具船藍藍", "小女孩安安", "浴缸海龜船長"],
            "浴缸小海洋、泡泡島、毛巾港口",
            "藍藍覺得自己太小，不能像大船一樣勇敢航行。",
            [
                "藍藍羨慕故事書裡的大船，想離開浴缸去真正的大海。",
                "泡泡浪把安安的小紙星星沖走，藍藍決定展開救援。",
                "藍藍找回紙星星，明白小小的自己也能帶來大大的安心。",
            ],
            "不論大小，每個人都能做重要的事。",
            "cozy bathroom ocean fantasy, toy boat, bubbles, soft cute 3D children's animation",
        ),
    ]


def write_options(ep_path: Path, ep_num: int) -> Path:
    preproduction = ep_path / "01_preproduction"
    preproduction.mkdir(parents=True, exist_ok=True)
    options = build_outline_options(ep_num)
    payload = {
        "episode": ep_num,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "brief": STORY_OUTLINE_BRIEF,
        "options": options,
    }
    out_path = preproduction / "story_outline_options.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_path = preproduction / "story_brainstorm_options.json"
    legacy_path.write_text(json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote 5 child-safe fairy-tale outline options: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Story mode stage 1.2: generate five child-safe fairy-tale outline options.")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", "workspaces/story"))
    args = parser.parse_args()

    ep_path = find_episode_path(Path(args.workspace_root).resolve(), args.ep)
    write_options(ep_path, args.ep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
