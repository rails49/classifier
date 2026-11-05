# classifier
Per-sensor track occupancy classifier for rails49: patch export, training, the rails49-classifier Python package, and the @rails49/classifier browser package

The classifier sometimes reports a car that is not there, and sometimes misses one. It is not a safety interlock.

## Python package

`rails49-classifier` cuts patches from a frame and scores them with an ONNX model. It runs without Torch.

```python
from classifier import Classifier, Sensor

model = Classifier()  # The packaged model, on onnxruntime.
results = model.score(frame, [Sensor("s1", x=812.0, y=440.5, heading=0.3)], dpt=16.0)
occupied = results["s1"].p_occupied >= model.threshold
```

`src/classifier/runtime.py` describes the arguments and results.

- `Classifier(path)` loads another model.
- `Classifier(engine="openvino", device="GPU")` runs on OpenVINO. Install the `openvino` extra.

The training code needs the `train` extra.

## Development

- `bin/test.sh` runs the checks and tests a change must pass.
- `bin/check-wheel.sh <out>` builds the wheel into `<out>` and scores a frame with it, installed alone with no extras.
- `docs/releasing.md` describes a release.
