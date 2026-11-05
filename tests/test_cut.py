import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageDraw

from classifier import Classifier, Sensor
from classifier.cut import Moves, Shape, cut, sample_points
from classifier.data import Index
from classifier.patches import Cutter, Jitter

SHARED = json.loads((Path(__file__).parent / "cases" / "patches.json").read_text())
SHAPE = Shape(**SHARED["shape"])
CUT_CASES = [c for c in SHARED["cases"] if c["samples"] is not None]
GRID_SAMPLE_LEVELS = 0.01
"""How far the numpy sampler and ``grid_sample`` may differ, in 8-bit levels."""


def ids(case: dict[str, Any]) -> str:
    return case["name"]


def shared_frame() -> npt.NDArray[np.uint8]:
    """The frame the shared cases describe."""
    height, width = SHARED["frame"]["height"], SHARED["frame"]["width"]
    rows, cols = np.mgrid[:height, :width]

    def wave(n: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return 5 * np.minimum(n % 102, 102 - n % 102)

    return np.stack([wave(cols), wave(rows), np.full_like(rows, 128)], axis=-1).astype(np.uint8)


def cut_one(
    frame: npt.NDArray[np.uint8], x: float, y: float, heading: float, dpt: float
) -> npt.NDArray[np.float32]:
    """The patch at one position, height × width × RGB in [0, 255]."""
    return cut(frame, np.array([x]), np.array([y]), np.array([heading]), dpt, SHAPE)[0]


def grid_sampled(
    frame: npt.NDArray[np.uint8], x: float, y: float, heading: float, dpt: float
) -> npt.NDArray[np.float32]:
    """The same patch as training cuts it."""

    def one(value: object) -> np.ndarray:
        return np.array([value])

    index = Index(
        dpt,
        *map(one, ["a", "i.jpg", x, y, heading, False, "elsewhere", 0.0, 0, np.nan, np.nan]),
    )
    patch = Cutter({("a", "i.jpg"): frame}, dpt, SHAPE, Jitter())(index, Moves.none(1))[0]
    return patch.permute(1, 2, 0).numpy() * 255


@pytest.mark.parametrize("case", CUT_CASES, ids=ids)
def test_each_patch_pixel_samples_the_frame_point_the_shared_case_states(case: dict[str, Any]):
    sensor = case["sensor"]

    x, y = sample_points(
        np.array([sensor["x"]]),
        np.array([sensor["y"]]),
        np.array([sensor["heading"]]),
        case["dpt"],
        SHAPE,
        Moves.none(1),
    )

    assert x.shape == y.shape == (1, 48, 96)
    tolerance = SHARED["tolerance"]["point_px"]
    for s in case["samples"]:
        assert x[0, s["row"], s["col"]] == pytest.approx(s["x"], abs=tolerance)
        assert y[0, s["row"], s["col"]] == pytest.approx(s["y"], abs=tolerance)


@pytest.mark.parametrize("case", CUT_CASES, ids=ids)
def test_each_patch_pixel_has_the_value_the_shared_case_states(case: dict[str, Any]):
    sensor = case["sensor"]

    patch = cut_one(shared_frame(), sensor["x"], sensor["y"], sensor["heading"], case["dpt"])

    assert patch.shape == (48, 96, 3)
    for s in case["samples"]:
        assert patch[s["row"], s["col"]] == pytest.approx(
            s["rgb"], abs=SHARED["tolerance"]["rgb_levels"]
        )


@pytest.mark.parametrize("case", SHARED["cases"], ids=ids)
def test_a_sensor_gets_the_patch_the_shared_case_states_or_none(case: dict[str, Any]):
    sensor = Sensor("s", **case["sensor"])

    result = Classifier().score(shared_frame(), [sensor], case["dpt"])["s"]

    if case["samples"] is None:
        assert result.patch is None and result.p_occupied is None
        return
    assert result.patch is not None and result.p_occupied is not None
    for s in case["samples"]:
        assert result.patch[s["row"], s["col"]] == pytest.approx(
            s["rgb"], abs=SHARED["tolerance"]["rgb_levels"]
        )


@pytest.mark.parametrize("case", CUT_CASES, ids=ids)
@pytest.mark.parametrize(("dx", "dy"), [(0.5, 0.0), (0.0, 0.5)])
def test_the_shared_pixels_catch_a_cut_half_a_pixel_off(case: dict[str, Any], dx: float, dy: float):
    s = case["sensor"]

    patch = cut_one(shared_frame(), s["x"] + dx, s["y"] + dy, s["heading"], case["dpt"])

    assert any(
        np.abs(patch[t["row"], t["col"]] - t["rgb"]).max() > SHARED["tolerance"]["rgb_levels"]
        for t in case["samples"]
    )


@pytest.mark.parametrize("case", CUT_CASES, ids=ids)
def test_the_numpy_sampler_agrees_with_grid_sample_on_the_shared_cases(case: dict[str, Any]):
    s, frame = case["sensor"], shared_frame()

    patch = cut_one(frame, s["x"], s["y"], s["heading"], case["dpt"])
    in_training = grid_sampled(frame, s["x"], s["y"], s["heading"], case["dpt"])

    assert np.abs(patch - in_training).max() < GRID_SAMPLE_LEVELS


@pytest.mark.parametrize("dpt", [14.0, 16.0, 19.5])
def test_the_numpy_sampler_agrees_with_grid_sample_on_noise_and_at_the_frames_edge(dpt: float):
    frame = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)

    for x, y, heading in [(150.3, 100.7, 0.7), (3.2, 196.9, -2.1), (299.5, 0.5, 2.9)]:
        patch = cut_one(frame, x, y, heading, dpt)
        in_training = grid_sampled(frame, x, y, heading, dpt)
        assert np.abs(patch - in_training).max() < GRID_SAMPLE_LEVELS


@pytest.mark.parametrize("dpt", [14.0, 19.5])
def test_a_frame_is_cut_at_its_own_dpt_so_a_gauge_covers_the_same_patch_pixels(dpt: float):
    # A stripe 1/2 G wide and 2 G long leaves (200, 200) at 30°.
    heading = math.radians(30)
    length = 2 * dpt
    image = Image.new("RGB", (400, 400))
    end = (200 + length * math.cos(heading), 200 + length * math.sin(heading))
    ImageDraw.Draw(image).line([(200, 200), end], fill="white", width=round(dpt / 2))

    patch = cut_one(np.asarray(image), 200, 200, heading, dpt).mean(-1)

    # In the patch a gauge is 16 pixels, whatever the frame's DPT.
    assert patch[22:26, 50:78].min() > 200  # on the stripe
    assert patch[22:26, 84:].max() < 30  # ahead of its end
    assert patch[22:26, :44].max() < 30  # behind
    assert patch[:16].max() < 30 and patch[-16:].max() < 30  # across


def test_report_the_time_to_cut_50_patches(capsys: pytest.CaptureFixture[str]):
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    x, y, heading = rng.uniform(0, 1920, 50), rng.uniform(0, 1080, 50), rng.uniform(-3, 3, 50)

    start = time.perf_counter()
    for _ in range(10):
        cut(frame, x, y, heading, 16.0, SHAPE)
    ms = (time.perf_counter() - start) / 10 * 1000

    with capsys.disabled():
        print(f"\ncut 50 patches from 1920 × 1080: {ms:.1f} ms")
