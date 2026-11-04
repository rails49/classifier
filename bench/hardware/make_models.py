"""Write the models for grid.py: float and int8 at DPT 12, 16 and 20, and at 2×.

ImageNet weights with an untrained head. The int8 models are calibrated on random data, so they
are for timing only: their outputs do not track the float models'.

Usage: uv run python bench/hardware/make_models.py <output dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

from classifier.models import MobileNetV3Small
from classifier.patches import Shape
from classifier.train import export_onnx

SIZES = {"dpt12": (12.0, 1), "dpt16": (16.0, 1), "dpt20": (20.0, 1), "2x": (16.0, 2)}
CALIBRATION_BATCHES = 8


class RandomPatches(CalibrationDataReader):
    def __init__(self, size: tuple[int, int]) -> None:
        rng = np.random.default_rng(0)
        self.batches = iter(
            rng.random((4, 3, *size), dtype=np.float32) for _ in range(CALIBRATION_BATCHES)
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self.batches, None)
        return None if batch is None else {"patches": batch}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, (dpt, upsample) in SIZES.items():
        shape = Shape(dpt=dpt)
        path = out / f"{name}-float.onnx"
        export_onnx(MobileNetV3Small(upsample=upsample), shape, path)
        prepared = out / f"{name}-prepared.onnx"
        quant_pre_process(path, prepared)
        quantize_static(
            prepared,
            out / f"{name}-int8.onnx",
            RandomPatches(shape.size),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        prepared.unlink()
        print(name, shape.size, flush=True)


if __name__ == "__main__":
    main()
