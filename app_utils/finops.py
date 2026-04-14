from pathlib import Path
from typing import Dict
import yaml


def load_cost_model(root: Path) -> Dict:
    p = root / "config" / "costs.yaml"
    if not p.exists():
        return {
            "unit_costs": {
                "imagen_per_image_usd": 0.05,
                "gemini_per_card_usd": 0.001,
                "whisper_per_min_usd": 0.006,
                "tts_per_min_usd": 0.015,
                "youtube_upload_quota_unit": 1600,
            },
            "assumptions": {
                "words_per_episode": 30,
                "images_per_word": 1,
                "avg_audio_minutes_per_episode": 8,
            },
            "thresholds": {
                "budget_usd_max_per_batch": 50,
                "youtube_daily_quota_max_units": 10000,
            },
        }
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def estimate_episode_cost(model: Dict) -> Dict:
    u = model["unit_costs"]
    a = model["assumptions"]
    images = a["words_per_episode"] * a.get("images_per_word", 1)
    minutes = a["avg_audio_minutes_per_episode"]
    usd = (
        images * u["imagen_per_image_usd"]
        + a["words_per_episode"] * u["gemini_per_card_usd"]
        + minutes * (u["whisper_per_min_usd"] + u["tts_per_min_usd"])
    )
    quota = u["youtube_upload_quota_unit"]  # naive: 1 upload ~ 1600 units
    return {"usd": usd, "quota": quota}


def estimate_batch_cost(model: Dict, episodes: int) -> Dict:
    ep = estimate_episode_cost(model)
    return {
        "usd": round(ep["usd"] * episodes, 2),
        "quota": ep["quota"] * episodes,
    }


def gating(model: Dict, episodes: int) -> Dict:
    totals = estimate_batch_cost(model, episodes)
    th = model.get("thresholds", {})
    return {
        "over_budget": totals["usd"] > th.get("budget_usd_max_per_batch", 1e9),
        "over_quota": totals["quota"] > th.get("youtube_daily_quota_max_units", 1e9),
        "totals": totals,
    }
