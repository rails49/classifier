"""Scores of a classifier on positions: the threshold, and false clear and false
occupied rates with confidence intervals from resampling whole placements.

A score is the classifier's belief that a position is occupied; below the
threshold it calls the position clear."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .data import REGIONS, Index

CENTER_BINS_G = (0.0, 50.0, 75.0, 100.0, np.inf)
"""Edges of the bins of distance from the image center, in gauges."""
BOOTSTRAP = 2000
"""Resamples behind each confidence interval."""

_Floats = npt.NDArray[np.float64]
_Bools = npt.NDArray[np.bool_]
_Ints = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Rate:
    """``errors`` of ``n`` positions, and the 95% interval of the rate."""

    errors: int
    n: int
    rate: float | None
    low: float | None
    high: float | None


def threshold(score: _Floats, occupied: _Bools, false_clear: float) -> float:
    """The highest threshold that calls at most ``false_clear`` of the occupied
    positions clear."""
    ranked = np.sort(score[occupied])
    return float(ranked[int(false_clear * len(ranked))])


def false_clear(
    score: _Floats, occupied: _Bools, placement: _Ints, t: float, rng: np.random.Generator
) -> Rate:
    """The occupied positions called clear at threshold ``t``."""
    return _rate((score < t)[occupied], placement[occupied], rng)


def false_occupied(
    score: _Floats, occupied: _Bools, placement: _Ints, t: float, rng: np.random.Generator
) -> Rate:
    """The clear positions called occupied at threshold ``t``."""
    return _rate((score >= t)[~occupied], placement[~occupied], rng)


def report(score: _Floats, index: Index, t: float, rng: np.random.Generator) -> dict[str, Any]:
    """False clear and false occupied at threshold ``t``: overall, by region and
    by distance from the image center."""

    def rates(rows: _Bools) -> dict[str, Any]:
        args = (score[rows], index.occupied[rows], index.placement[rows], t, rng)
        return {
            "false_clear": asdict(false_clear(*args)),
            "false_occupied": asdict(false_occupied(*args)),
        }

    everything = np.ones(len(index), dtype=bool)
    bins = list(zip(CENTER_BINS_G[:-1], CENTER_BINS_G[1:], strict=True))
    return {
        "overall": rates(everything),
        "region": {r: rates(index.region == r) for r in REGIONS},
        "center_g": {
            f"{lo:g}-{hi:g}": rates((index.center_g >= lo) & (index.center_g < hi))
            for lo, hi in bins
        },
    }


def _rate(error: _Bools, placement: _Ints, rng: np.random.Generator) -> Rate:
    if len(error) == 0:
        return Rate(0, 0, None, None, None)
    groups, inverse = np.unique(placement, return_inverse=True)
    errors = np.bincount(inverse, weights=error).astype(float)
    n = np.bincount(inverse).astype(float)
    draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), BOOTSTRAP)
    resampled = (draws @ errors) / (draws @ n)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return Rate(int(error.sum()), len(error), float(error.mean()), float(low), float(high))
