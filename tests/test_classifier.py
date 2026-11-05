import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import pytest

from classifier import MODEL, Classifier, Sensor

ENGINES_AGREE = 1e-4
"""How far onnxruntime and OpenVINO's CPU device may differ in P(occupied)."""

ONLY_THE_ENGINE = """
import sys
import numpy as np
import classifier

model = classifier.Classifier(engine=sys.argv[1])
frame = np.full((200, 300, 3), 90, dtype=np.uint8)
result = model.score(frame, [classifier.Sensor("a", 100.0, 100.0, 0.0)], 16.0)
assert result["a"].p_occupied is not None
print(",".join(m for m in ["torch", "r49", "onnxruntime", "openvino"] if m in sys.modules))
"""


def a_frame() -> npt.NDArray[np.uint8]:
    """Dark and light regions with noise, so that scores are not all alike."""
    rng = np.random.default_rng(0)
    frame = rng.normal(100, 40, (200, 300, 3))
    frame[60:90, 60:200] += 80
    return np.clip(frame, 0, 255).astype(np.uint8)


def spy_on_the_model(monkeypatch: pytest.MonkeyPatch) -> list[npt.NDArray[np.float32]]:
    calls: list[npt.NDArray[np.float32]] = []
    run = ort.InferenceSession.run

    def spy(self: ort.InferenceSession, names: Any, feed: dict[str, Any], *args: Any) -> Any:
        calls.append(feed["patches"])
        return run(self, names, feed, *args)

    monkeypatch.setattr(ort.InferenceSession, "run", spy)
    return calls


def test_each_sensor_gets_its_p_occupied_and_its_patch():
    sensors = [Sensor("a", 120.0, 75.0, 0.0), Sensor("b", 200.0, 150.0, 1.0)]

    result = Classifier().score(a_frame(), sensors, 16.0)

    assert list(result) == ["a", "b"]
    for r in result.values():
        assert r.p_occupied is not None and 0.0 <= r.p_occupied <= 1.0
        assert r.patch is not None and r.patch.shape == (48, 96, 3)
        assert r.patch.dtype == np.uint8


def test_a_sensors_score_does_not_depend_on_the_others_in_its_batch():
    a, b = Sensor("a", 120.0, 75.0, 0.0), Sensor("b", 200.0, 150.0, 1.0)
    model, frame = Classifier(), a_frame()

    together = model.score(frame, [a, b], 16.0)

    for sensor in (a, b):
        alone = model.score(frame, [sensor], 16.0)[sensor.id]
        assert together[sensor.id].p_occupied == pytest.approx(alone.p_occupied, abs=1e-5)


def test_a_sensor_whose_center_is_outside_the_frame_has_no_score_and_no_patch():
    sensors = [Sensor("left", -0.5, 100.0, 0.0), Sensor("inside", 100.0, 100.0, 0.0)]

    result = Classifier().score(a_frame(), sensors, 16.0)

    assert result["left"].p_occupied is None and result["left"].patch is None
    assert result["inside"].p_occupied is not None


def test_a_frame_with_no_sensor_inside_makes_no_model_call(monkeypatch: pytest.MonkeyPatch):
    calls = spy_on_the_model(monkeypatch)

    result = Classifier().score(a_frame(), [Sensor("left", -1.0, 100.0, 0.0)], 16.0)

    assert result["left"].p_occupied is None
    assert calls == []


def test_a_sensor_partly_outside_is_scored_with_a_patch_padded_black():
    result = Classifier().score(a_frame(), [Sensor("edge", 10.0, 100.0, 0.0)], 16.0)["edge"]

    assert result.p_occupied is not None
    assert result.patch is not None
    assert result.patch[:, :30].max() == 0  # The patch's first 30 pixels lie left of the frame.
    assert result.patch[:, 60:].mean() > 50


def test_a_frame_makes_one_model_call_with_one_batch_as_training_feeds_it(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = spy_on_the_model(monkeypatch)
    sensors = [Sensor(str(i), 60.0 + 40 * i, 100.0, 0.0) for i in range(4)]

    Classifier().score(a_frame(), [*sensors, Sensor("out", 400.0, 100.0, 0.0)], 16.0)

    assert len(calls) == 1
    assert calls[0].shape == (4, 3, 48, 96)
    assert calls[0].dtype == np.float32
    assert 0.0 <= calls[0].min() and 0.5 < calls[0].max() <= 1.0


def test_the_threshold_comes_from_the_json_file_beside_the_model(tmp_path: Path):
    shutil.copy(MODEL, tmp_path / "m.onnx")
    settings = json.loads(MODEL.with_suffix(".json").read_text())
    (tmp_path / "m.json").write_text(json.dumps({**settings, "threshold": 0.25}))

    assert Classifier(tmp_path / "m.onnx").threshold == 0.25


def test_the_packaged_model_is_the_chosen_baseline_with_its_threshold():
    assert MODEL.name == "2x-ceiling-2026-10-29.onnx"
    assert Classifier().threshold == pytest.approx(0.0028813397511839867)


def test_scoring_on_onnxruntime_and_openvino_cpu_agrees():
    sensors = [Sensor(str(i), 30.0 + 40.0 * i, 75.0, 0.3 * i) for i in range(6)]
    frame = a_frame()

    ours = Classifier(engine="onnxruntime").score(frame, sensors, 16.0)
    theirs = Classifier(engine="openvino", device="CPU").score(frame, sensors, 16.0)

    p_ours = [r.p_occupied for r in ours.values()]
    p_theirs = [r.p_occupied for r in theirs.values()]
    assert p_ours == pytest.approx(p_theirs, abs=ENGINES_AGREE)


def test_openvino_scores_a_batch_of_one():
    result = Classifier(engine="openvino").score(a_frame(), [Sensor("a", 100.0, 100.0, 0.0)], 16.0)

    assert result["a"].p_occupied is not None


def test_onnxruntime_refuses_an_openvino_device_name():
    with pytest.raises(ValueError, match="GPU"):
        Classifier(device="GPU")


@pytest.mark.parametrize(
    ("engine", "imported"), [("onnxruntime", "onnxruntime"), ("openvino", "openvino")]
)
def test_only_the_chosen_engine_is_imported_and_never_torch_or_r49(engine: str, imported: str):
    # Run from a fresh interpreter: this test process has imported Torch.
    out = subprocess.run(
        [sys.executable, "-c", ONLY_THE_ENGINE, engine], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == imported
