import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from classifier.data import Index
from classifier.patches import Cutter, Jitter, Moves, Shape, in_band
from classifier.train import Setup, batch

G = 16.0


def index_at(x: float, y: float, heading: float) -> Index:
    def one(value: object) -> np.ndarray:
        return np.array([value])

    return Index(
        G, *map(one, ["a", "i.jpg", x, y, heading, False, "elsewhere", 0.0, 0, np.nan, np.nan])
    )


def test_the_patch_is_turned_so_the_track_runs_along_its_long_axis_in_the_heading_direction():
    # A stripe 1/2 G wide leaves (200, 200) at 30°: the track ahead of the position.
    heading = math.radians(30)
    image = Image.new("RGB", (400, 400))
    end = (200 + 100 * math.cos(heading), 200 + 100 * math.sin(heading))
    ImageDraw.Draw(image).line([(200, 200), end], fill="white", width=8)
    cut = Cutter({("a", "i.jpg"): np.asarray(image)}, G, Shape(), Jitter())
    index = index_at(200, 200, heading)

    patch = cut(index, Moves.none(1))[0].mean(0).numpy()
    turned = cut(index, replace(Moves.none(1), turn=np.array([True])))[0].mean(0).numpy()

    assert patch.shape == (48, 96)
    assert patch[22:26, 52:].min() > 0.9  # ahead, on the track
    assert patch[22:26, :44].max() < 0.1  # behind
    assert patch[:16].max() < 0.1 and patch[-16:].max() < 0.1  # across
    assert turned == pytest.approx(patch[::-1, ::-1], abs=1e-3)


def test_a_jittered_position_is_dropped_when_its_point_falls_2_to_2_5_g_from_a_car_span():
    ahead = np.array([2.7, 2.7, np.nan, 0.0, np.nan])
    behind = np.array([np.nan, np.nan, 1.8, 0.0, np.nan])
    along = np.array([0.4, -0.4, 0.4, 0.5, 0.5])

    # Clear 2.7 G behind a car, moved 0.4 G toward it: 2.3 G, in the band.
    # Occupied 1.8 G ahead of a car, moved 0.4 G away: 2.2 G.
    assert in_band(ahead, behind, along).tolist() == [True, False, True, False, False]


def test_a_batch_drops_the_positions_its_moves_put_in_the_band():
    # One clear position 2.7 G behind a car: moved more than 0.2 G toward it, it is in the band.
    def one(value: object) -> np.ndarray:
        return np.array([value])

    index = Index(G, *map(one, ["a", "i.jpg", 0.0, 0.0, 0.0, False, "end", 0.0, 0, 2.7, np.nan]))

    positions, moves = batch(index, np.random.default_rng(0), Setup(per_region=1000))

    assert 400 < len(positions) < 1000
    assert moves.along_g.max() <= 0.2
