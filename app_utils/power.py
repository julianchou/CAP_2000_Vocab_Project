import ctypes
from contextlib import contextmanager
from typing import Iterator


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


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
