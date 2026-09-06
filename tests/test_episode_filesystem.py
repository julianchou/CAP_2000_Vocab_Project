from pathlib import Path

from app_utils.filesystem import (
    find_episode_dirs_by_ep,
    list_episode_dirs,
    parse_episode_info,
)


def test_episode_scanner_includes_three_digit_episode(tmp_path: Path):
    workspace = tmp_path / "workspaces" / "vocab"
    workspace.mkdir(parents=True)
    previous_episode = workspace / "Ep99_1081_1090"
    previous_episode.mkdir()
    episode = workspace / "Ep100_1091_1100"
    episode.mkdir()

    episodes = list_episode_dirs(tmp_path, "workspaces/vocab")

    assert episodes == [previous_episode, episode]
    assert find_episode_dirs_by_ep(tmp_path, 100, "workspaces/vocab") == [episode]
    assert parse_episode_info(episode) == {
        "ep": 100,
        "start": 1091,
        "end": 1100,
        "path": episode,
    }
