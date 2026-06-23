from __future__ import annotations

import numpy as np

from ..tallies.diagnostics import operating_threshold


def calibrate_threshold(
    y: np.ndarray | list[int], probability: np.ndarray | list[float], sensitivity: float
) -> float:
    return operating_threshold(y, probability, sensitivity)


def number_needed(precision: float) -> float:
    if precision <= 0:
        return float("inf")
    return 1.0 / precision
