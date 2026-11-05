# classifier
Per-sensor track occupancy classifier for rails49: patch export, training, and the @rails49/classifier browser package

The classifier sometimes reports a car that is not there, and sometimes misses one. It is not a safety interlock.

## Python package

`rails49-classifier` cuts patches from a frame and scores them with an ONNX model. It runs without Torch.

```python
from classifier import Classifier, Sensor

model = Classifier()  # The packaged model, on onnxruntime.
results = model.score(frame, [Sensor("s1", x=812.0, y=440.5, heading=0.3)], dpt=16.0)
occupied = results["s1"].p_occupied >= model.threshold
```

- `frame` is a numpy array, height × width × RGB, 8 bits. `x` and `y` are in its pixels, and `heading` is the direction the track runs, in radians.
- `score` returns a `Result` per sensor id, with `p_occupied` and `patch`. Both are `None` when the sensor's center is outside the frame. A sensor partly outside is padded with black.
- `threshold` is the model's. `score` does not apply it.
- `Classifier(path)` loads another model, an ONNX file with a JSON file of the same stem beside it.
- `Classifier(engine="openvino", device="GPU")` runs on OpenVINO. Install the `openvino` extra.

The training code needs the `train` extra.
