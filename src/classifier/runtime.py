"""Score a frame's sensors with an ONNX classifier, with no Torch.

A model is an ONNX file and the JSON file beside it, as ``classifier.train``
describes them."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .cut import Shape, cut

MODEL = Path(__file__).parent / "model" / "2x-ceiling-2026-10-29.onnx"
"""The packaged model."""

Engine = Literal["onnxruntime", "openvino"]
_Run = Callable[[npt.NDArray[np.floating]], Any]


@dataclass(frozen=True)
class Sensor:
    """A point on the track to score: ``x`` and ``y`` in the frame's pixels, and
    ``heading`` the direction the track runs there, in radians, as ``atan2(dy, dx)``."""

    id: str
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class Result:
    """A sensor's P(occupied), and its patch as height × width × RGB, 8 bits. Both
    are ``None`` when the sensor's center is outside the frame."""

    p_occupied: float | None
    patch: npt.NDArray[np.uint8] | None


class Classifier:
    """Scores frames with the ONNX file ``model``. ``engine`` is imported when
    chosen. ``device`` is an OpenVINO device name; onnxruntime runs on the CPU.
    OpenVINO runs in 32-bit floats, as onnxruntime does."""

    def __init__(
        self, model: Path = MODEL, engine: Engine = "onnxruntime", device: str = "CPU"
    ) -> None:
        settings = json.loads(model.with_suffix(".json").read_text())
        self._threshold: float = settings["threshold"]
        self._shape = Shape(settings["across_g"], settings["along_g"], settings["dpt"])
        self._run = _scorer(model, engine, device)

    @property
    def threshold(self) -> float:
        """The model's threshold on P(occupied). Scoring does not apply it."""
        return self._threshold

    def score(
        self, frame: npt.NDArray[np.uint8], sensors: Sequence[Sensor], dpt: float
    ) -> dict[str, Result]:
        """Each sensor's result, by id, from one model call. ``frame`` is height ×
        width × RGB, 8 bits, at DPT ``dpt``. A sensor partly outside the frame is
        padded with black."""
        results = {s.id: Result(None, None) for s in sensors}
        height, width = frame.shape[:2]
        inside = [s for s in sensors if 0 <= s.x < width and 0 <= s.y < height]
        if not inside:
            return results
        patches = cut(
            frame,
            np.array([s.x for s in inside]),
            np.array([s.y for s in inside]),
            np.array([s.heading for s in inside]),
            dpt,
            self._shape,
        )
        scores = self._run((patches / 255).transpose(0, 3, 1, 2))
        pixels = np.rint(patches).astype(np.uint8)
        for sensor, score, patch in zip(inside, scores, pixels, strict=True):
            results[sensor.id] = Result(float(score), patch)
        return results


def _scorer(model: Path, engine: Engine, device: str) -> _Run:
    if engine == "onnxruntime":
        if device != "CPU":
            raise ValueError(f"onnxruntime runs on the CPU, not on {device}")
        import onnxruntime as ort

        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        return lambda patches: session.run(None, {"patches": patches})[0]
    if engine == "openvino":
        import openvino as ov

        compiled = ov.Core().compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f32"})
        return lambda patches: compiled({"patches": patches})[0]
    raise ValueError(f"unknown engine {engine}")
