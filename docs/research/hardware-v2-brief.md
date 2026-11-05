2026-11-04. The brief `hardware-v2.md` answers, moved here unchanged from `~/rails49/HW.md`. `hardware-v2.md` corrects its machine: 32 GB, host `gleis49`, Ubuntu 26.04.1.

# Classifier figures for hardware selection

The classifier is MobileNetV3-Small, the baseline of rails49/classifier#4.
Compute and model size do not depend on the data. Timings are medians of 50
runs on an idle M2 Max (12 cores), 2026-11-03.

| | 1× (48 × 96 input) | 2× (96 × 192, upsampled inside the model) |
|---|---|---|
| MACs per patch | 6.6 M | 21.0 M |
| MACs per batch of 50 | 0.33 G | 1.05 G |
| Parameters | 1.52 M | 1.52 M |
| Weights (float32 ONNX) | 6.4 MB | 6.4 MB |
| Native onnxruntime CPU, batch of 50, 1 thread | 62 ms | 114 ms |
| Native onnxruntime CPU, batch of 50, automatic threads | 12 ms | 23 ms |
| WASM under Node, batch of 50, 1 thread | 97 ms | 179 ms |
| WASM under Node, batch of 50, onnxruntime-web's default threads | 27 ms | 50 ms |

- The MAC counts are a lower bound. The counter includes only convolutions
  and matrix multiplies, not activations, pooling or the upsampling.
- 2× has the same weights but about 3.2× the compute. Choosing 2× costs time,
  not memory for weights.
- The ONNX input is always B × RGB × 48 × 96 in [0, 1], whichever size you
  pick. On the device, patches of 3 G × 6 G are cut at 16 pixels per gauge
  and sent in batches.
- Not measured:
  - Peak memory for intermediate results.
  - int8 quantization, which would make the weights about 1.5 MB and usually
    runs 2–4× faster on CPU.
  - Any GPU or NPU on the target hardware.
- Accuracy (rails49/classifier
  `docs/research/2026-11-03-baseline-mobilenetv3-small.md`): 2× is the
  better size. At 0.1% false clear it has 0.41% false occupied against 1×'s
  1.29%, with one threshold set over all held-out positions. With the
  per-fold threshold #5 will use, neither size meets both targets. With 2×,
  the hardware must run about 1 G MACs per batch of 50 at the frame rate.

## Hardware

- Server: Mac mini 2018, Core i7-8700B, 6 cores and 12 threads, 3.2 GHz
  base and 4.6 GHz turbo, AVX2, no AVX-512 or VNNI, Intel UHD 630. RAM size
  not recorded; read it on the machine. Ubuntu 24.04 from t2linux,
  installed alongside macOS 15.6 without wiping the disk. Host name: to be
  filled in.
- Camera: OBSBOT Meet 2. 1/2" 48 MP sensor, fixed lens, USB 2.0. Over USB
  it sends 4K MJPEG at 30, 25, 24, 20 or 15 fps and 1080p MJPEG at up to
  60 fps. Diagonal field 79.4° (OBSBOT); horizontal 67.2° (a review, not
  OBSBOT).
- Field: 1.77 m × 1.0 m, the full 16:9 frame, N scale (gauge 8.97 mm).
  With a 67.2° horizontal field the camera hangs about 1.33 m above the
  railhead; check this on the camera. DPT is 19.5 at 4K and 9.7 at 1080p.
  Patches are cut at the classifier's DPT; one affine warp per patch does
  the rotation and the scaling.

## Goals

- 50 sensors at 10 Hz, and 50 sensors at 20 Hz.
- The rate against the number of sensors, up to more than 100.
- Report rates, not pass or fail. DPT, number of sensors and rate trade
  against each other; the control algorithms are chosen to fit what the
  hardware delivers.

## What to measure

1. Throughput in patches per second. Compute does not depend on the
   weights, so untrained models serve for the DPTs not yet trained.
   - Runtimes: onnxruntime 1.31.1 on the CPU, OpenVINO on the CPU,
     OpenVINO on the UHD 630.
   - Threads: 1, 2, 4, 6, 12.
   - Patch DPT: 12 (36 × 72), 16 (48 × 96, the current 1×), 20 (60 × 120),
     and the current 2× (96 × 192 inside the model).
   - Batch size: 1, 25, 50, 100, 150, 200. Batch 1 is one call per patch.
   - float, and int8 from the random-calibration models, marked as timing
     only.
2. Capture cost: CPU use for 4K MJPEG at 10, 15, 20 and 30 fps and 1080p
   MJPEG at 30 and 60 fps, decoded in software and through VA-API.
3. End to end: capture, decode, cutting the patches and classifying, for
   50 sensors at 10 Hz and at 20 Hz, and the largest number of sensors
   reached at each rate.
4. One 30-minute run of the full chain at the highest rate found, with
   rate, CPU clock and temperature over time.

## Out of scope

- Sharing the machine with the railroad stack.
- Accuracy against DPT. That is a training question for rails49/classifier.
- Core ML.
- Other hardware: see rails49/classifier
  `docs/research/hardware-v1.md`.

The answer goes in rails49/classifier `docs/research/hardware-v2.md`.
