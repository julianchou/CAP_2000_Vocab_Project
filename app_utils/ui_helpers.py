from typing import Dict

STATUS_EMOJI = {
    "pending": "⚪",
    "running": "🟡",
    "error": "🔴",
    "done": "🟢",
}


def with_emoji(status: str) -> str:
    return f"{STATUS_EMOJI.get(status, '⚪')} {status}"
