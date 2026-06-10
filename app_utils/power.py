import ctypes
import subprocess
from contextlib import contextmanager
from typing import Iterator


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def wake_timer_policy() -> dict:
    """Return the current Windows wake timer policy when powercfg is available."""
    if not hasattr(ctypes, "windll"):
        return {"available": False, "message": "wake timer checks are only available on Windows"}

    try:
        result = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "RTCWAKE"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "message": str(exc)}

    text = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return {"available": False, "message": text or f"powercfg exited with {result.returncode}"}

    def parse_index(label: str) -> int | None:
        for line in text.splitlines():
            if label in line:
                raw = line.rsplit(":", 1)[-1].strip()
                try:
                    return int(raw, 16)
                except Exception:
                    return None
        return None

    ac = parse_index("目前的 AC 電源設定索引") or parse_index("Current AC Power Setting Index")
    dc = parse_index("目前的 DC 電源設定索引") or parse_index("Current DC Power Setting Index")
    labels = {0: "停用", 1: "啟用", 2: "僅重要喚醒計時器"}
    return {
        "available": True,
        "ac_index": ac,
        "dc_index": dc,
        "ac_label": labels.get(ac, str(ac) if ac is not None else "未知"),
        "dc_label": labels.get(dc, str(dc) if dc is not None else "未知"),
        "ac_enabled": ac in {1, 2},
        "dc_enabled": dc in {1, 2},
        "raw": text,
    }


@contextmanager
def keep_system_awake(reason: str | None = None) -> Iterator[None]:
    """Prevent Windows from entering sleep while a long-running task is active."""
    if reason:
        print(f"[power] Preventing system sleep: {reason}")

    windll = getattr(ctypes, "windll", None)
    if windll is None or not hasattr(windll, "kernel32"):
        yield
        return

    kernel32 = windll.kernel32
    previous_state = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if previous_state == 0:
        print("[power] Failed to enable sleep prevention. Continuing anyway.")
        yield
        return

    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        if reason:
            print(f"[power] Restored normal sleep policy: {reason}")
