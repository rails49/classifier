# Hardware for the classifier, version 3

Written 2026-11-12. The classifier's speed on the two machines of [`hardware-v1.md`](hardware-v1.md), measured
again inside the occupancy-service image, with the packaged model, while each machine runs every stack.
No camera was attached; the input is random. [`hardware-v2.md`](hardware-v2.md) has the 2018 Mac mini.

Evidence tags: M measured here, E arithmetic from measurements.

## Answer

For 50 sensors with the packaged 2× model, as the service runs it (OpenVINO at f32, on the device the
service picks):

| Machine | Device the service picks | Rate | Cores busy |
|---|---|---:|---:|
| Kamrui JK06, Celeron N5100 | GPU, 24 execution units | 26 Hz | 1.0 |
| Raspberry Pi 4, 8 GB | CPU, 4 threads | 4.2 Hz | 3.4 |

1. On the N5100 the GPU is 1.5× the CPU at f32 (26 Hz against 17 Hz on 4 threads) and leaves 3 cores
   free. At OpenVINO's GPU default of FP16 it reaches 38 Hz; the service asks for f32.
2. The Pi 4 reaches 4.2 Hz on all 4 cores. It has no GPU that OpenVINO uses.
3. The Pi 4's figures match version 1 within 1–7%. The N5100's single-thread figure matches; its
   4-thread CPU figures are 1.4× faster and its GPU 4.2× faster than in version 1, for reasons not known.
4. The classifier's peak process memory is 156 MB on the N5100's CPU and 544–787 MB on its GPU at f32
   (25 to 200 sensors), 170 MB on the Pi 4's CPU.

## Machines

- Kamrui JK06: Celeron N5100, 4 Tremont cores at up to 2.8 GHz with SSE4.2 and no AVX, AVX2 or FMA,
  UHD Graphics with 24 execution units at up to 800 MHz, 8 GB. Ubuntu 26.04.1, kernel 7.0.0-43-generic,
  governor `powersave` with `balance_performance`.
- Raspberry Pi 4 Model B Rev 1.4, 8 GB, 1.8 GHz. Raspberry Pi OS 13 (trixie) Lite 64-bit, kernel
  6.18.50+rpt-rpi-v8, governor `ondemand`. 46–47 °C after the runs, not throttled.

Both ran installation release 2026.11.5 on Docker Engine 29.9.3, with 17 containers: control with the
simulator, dccex, JMRI, occupancy and the installation. The occupancy service was paused with no camera
and used 10–16% of one core retrying it. The service's log shows it picks OpenVINO GPU on the N5100 and
OpenVINO CPU on the Pi 4.

## Method

`bench/hardware/service_image.sh` runs `bench/hardware/time_one.py` in the image of the running
occupancy-service container, `occupancy-service:907a814`: Python 3.13, OpenVINO 2026.4.1, onnxruntime
1.31.1, and on amd64 Ubuntu 24.04's `intel-opencl-icd` 23.43. The model is the one the image packages,
`2x-ceiling-2026-10-29.onnx`, input 48 × 96. Each setting is its own container, at nice 10, timed for 3 s
or more after warm-up. `time_one.py` gained `--precision`, which sets OpenVINO's
`INFERENCE_PRECISION_HINT`; the service sets `f32` (`src/classifier/runtime.py`). On the CPU f32 is the
default on both machines.

## Results

Rate for 50 sensors in Hz, 1000 / median ms of a batch of 50 [M]:

| Machine, runtime | 1 thread | 2 | 4 |
|---|---:|---:|---:|
| N5100, OpenVINO CPU | 5.3 | 9.9 | 16.9 |
| N5100, onnxruntime | 5.1 | 9.5 | 15.8 |
| Pi 4, OpenVINO CPU | 2.3 | 3.5 | 4.2 |
| Pi 4, onnxruntime | 1.2 | 2.0 | 2.7 |

Rate in Hz by number of sensors, one batch per frame [M]:

| Machine, device | 25 | 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|---:|
| N5100, GPU, f32 | 50 | 26 | 13.7 | 8.9 | 6.7 |
| N5100, GPU, FP16 | 61 | 38 | 19.9 | 13.3 | 9.8 |
| N5100, OpenVINO CPU, 4 threads | 24 | 16.9 | 5.9 | | |
| Pi 4, OpenVINO CPU, 4 threads | 8.2 | 4.2 | 2.0 | | |

- On 4 threads p95 is 1.2–1.9× the median on both machines; on 1 or 2 threads and on the GPU it is within
  1.15×.
- A GPU run flat out keeps one core busy, as on the Mac mini.
- On the Pi 4 OpenVINO is 1.5× onnxruntime on 4 threads. On the N5100 they are within 7%.
- Against the peaks in version 1, the N5100 reaches about 40% of its CPU's 45 G MAC/s (17.8 G MAC/s), 18%
  of its GPU's 154 G MAC/s at FP32 (27.6) and 13% of 307 at FP16 (39.6) [E, from 21.05 M MACs per patch].
  The Pi 4 does 4.4 G MAC/s [E].
- The N5100's GPU at FP16 is 1.9× slower than the Mac mini's UHD 630 (26.6 ms against 14 ms), where
  version 2 found 8×.

Raw results: `hardware-v3.jsonl`.

## Against version 1

Median ms for a batch of 50 with the 2× model. Version 1 timed an untrained model at default precision,
median of 50 runs [M2, M3 there]; version 2 found trained and untrained models within 0.3% at 2×.

| Machine, runtime | Version 1 | Version 3 |
|---|---:|---:|
| N5100, onnxruntime, 1 thread | 195.5 | 195.4 |
| N5100, onnxruntime, 2 threads | 127.7 | 105.8 |
| N5100, onnxruntime, 4 threads | 90.2 | 63.1 |
| N5100, OpenVINO CPU, 4 threads | 83.2 | 59.1 |
| N5100, OpenVINO GPU, FP16 | 112.5 | 26.6 |
| Pi 4, onnxruntime, 1 thread | 799.4 | 809.9 |
| Pi 4, onnxruntime, 4 threads | 349.9 | 363.7 |
| Pi 4, OpenVINO CPU, 4 threads | 222.8 | 237.4 |

On the N5100 version 1 ran on Ubuntu 24.04 with 15 idle containers, in its own Ubuntu 24.04 container
with Ubuntu's `intel-opencl-icd`. The runtime versions are the same. Which of the host OS, the kernel or
the container explains the faster multi-thread and GPU figures is not tested. The Pi 4 is 1–7% slower
now; during these runs the paused service used part of one core.

## Not measured

- Decoding the camera's frames and cutting the patches, and so the whole chain.
- The occupancy service scoring for minutes with a camera, and its drift check.
- Paced runs and heat over minutes; each setting ran for about 3 s.
- The GPU clock during the runs.
