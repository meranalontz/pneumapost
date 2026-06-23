from __future__ import annotations

import math


def warmup_cosine(step: int, warmup: int, total: int, floor: float = 0.01) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / (total - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
