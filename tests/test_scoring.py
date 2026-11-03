import numpy as np
import pytest

from classifier.scoring import false_clear, threshold


def test_confidence_intervals_resample_whole_placements():
    # Placement 0 misses all 10 of its occupied positions; placement 1 none of 990.
    score = np.r_[np.zeros(10), np.ones(990)]
    occupied = np.ones(1000, dtype=bool)
    placement = np.r_[np.zeros(10, int), np.ones(990, int)]

    rate = false_clear(score, occupied, placement, 0.5, np.random.default_rng(0))

    # Resampling positions would give about 1% ± 0.6%.
    assert (rate.errors, rate.n) == (10, 1000)
    assert rate.rate == pytest.approx(0.01)
    assert (rate.low, rate.high) == (0.0, 1.0)


def test_the_threshold_leaves_at_most_the_allowed_share_of_occupied_positions_below_it():
    occupied_scores = np.linspace(0.01, 1, 2000)
    score = np.r_[occupied_scores, np.zeros(5)]
    occupied = np.r_[np.ones(2000, bool), np.zeros(5, bool)]

    t = threshold(score, occupied, 0.001)

    assert np.sum(occupied_scores < t) == 2
