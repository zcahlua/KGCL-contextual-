from __future__ import annotations

import os
import time
from typing import Any


def enabled(env_var: str) -> bool:
    return os.environ.get(env_var) == "1"


def start(env_var: str) -> float | None:
    return time.perf_counter() if enabled(env_var) else None


def elapsed_ms(start_time: float | None) -> float:
    if start_time is None:
        return 0.0
    return (time.perf_counter() - start_time) * 1000.0


def log(env_var: str, message: str, **metrics: Any) -> None:
    if not enabled(env_var):
        return
    suffix = ""
    if metrics:
        suffix = " " + " ".join(f"{key}={value}" for key, value in metrics.items())
    print(f"[{env_var}] {message}{suffix}", flush=True)
