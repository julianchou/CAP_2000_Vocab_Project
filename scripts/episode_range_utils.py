import re
from pathlib import Path


EP_DIR_PATTERN = re.compile(r"^Ep(?P<ep>\d{2,})_(?P<start>\d{4})_(?P<end>\d{4})$")


def find_episode_folder(workspace_dir: Path, ep_num: int) -> Path | None:
    matches = sorted(
        p for p in workspace_dir.glob(f"Ep{int(ep_num):02d}_*")
        if p.is_dir() and EP_DIR_PATTERN.match(p.name)
    )
    return matches[0] if matches else None


def parse_episode_range(episode_folder: Path) -> tuple[int, int]:
    match = EP_DIR_PATTERN.match(episode_folder.name)
    if not match:
        raise ValueError(f"無法從資料夾名稱解析集數區間：{episode_folder.name}")
    return int(match.group("start")), int(match.group("end"))


def resolve_episode_range(workspace_dir: Path, ep_num: int) -> tuple[Path, int, int]:
    episode_folder = find_episode_folder(workspace_dir, ep_num)
    if not episode_folder:
        raise FileNotFoundError(f"找不到第 {int(ep_num):02d} 集的資料夾。")
    start_num, end_num = parse_episode_range(episode_folder)
    return episode_folder, start_num, end_num
