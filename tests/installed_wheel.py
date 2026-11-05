"""Run by bin/check-wheel.sh against the wheel installed alone, with no extras:
importing the package and scoring a frame needs neither Torch nor r49, and the
wheel holds the model."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import classifier

assert classifier.__file__ is not None and "site-packages" in Path(classifier.__file__).parts
for extra in ("torch", "r49", "openvino"):
    assert importlib.util.find_spec(extra) is None, f"{extra} is installed with no extras"

model = classifier.Classifier()
frame = np.full((200, 300, 3), 90, dtype=np.uint8)
sensors = [classifier.Sensor("in", 100.0, 100.0, 0.0), classifier.Sensor("out", -1.0, 0.0, 0.0)]
result = model.score(frame, sensors, 16.0)

assert result["in"].p_occupied is not None and result["in"].patch is not None
assert result["out"].p_occupied is None and result["out"].patch is None
assert 0.0 < model.threshold < 1.0
assert "torch" not in sys.modules and "r49" not in sys.modules
print(f"{classifier.MODEL.name} from the wheel scores a frame with no extras")
