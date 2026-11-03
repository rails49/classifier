# Hardware for the version-1 classifier

Written 2026-11-03. Research for the hardware question in `~/rails49/HW.md` [R1]: which of four
options can run the MobileNetV3-Small classifier of [#4](https://github.com/rails49/classifier/issues/4)
at the rate [rails49/occupancy#272](https://github.com/rails49/occupancy/issues/272) asks for, and which
to choose. It builds on the detector's hardware and camera notes [R6–R9] and does not repeat them.

Evidence tags: M measured for this note, P published figure, E estimate or arithmetic, with the method
shown. Sources are listed at the end as [R…] (rails49 repos and devices) and [S…] (outside). Sources
marked "secondary" are not the vendor's own. Nothing ran on any of the four candidate devices. The box's
hardware was read over ssh, without running anything else on it [M1].

## Answer

1. 50 patches at 10 Hz is 3.3 G MAC/s at 1× and 10.5 G MAC/s at 2×. Each batch of 50 must finish
   within the 100 ms frame period; this note plans for 50 ms.
2. The Galaxy A13 5G fails on its camera: it records video at 1080p, which is DPT 8 over the layout.
3. The Raspberry Pi 4 fits 1× on its CPU (about 47–63 ms per batch, E) but not 2× (150–200 ms, E).
   With a Camera Module 3 it reaches DPT 20 at 14 fps; with a Camera Module 2, DPT 14.
4. The box (Celeron N5100) has no AVX or FMA. Measured: its CPU runs 1× in 18 ms on 4 threads and 2×
   in 83–90 ms (M2). 2× is inside the 100 ms limit but over the 50 ms target, with all 4 cores busy.
   Its integrated GPU is slower than its CPU (32 ms at 1×, 113 ms at 2×), and int8 is slower than float.
5. The iPhone XS Max has compute to spare: on its Neural Engine a batch should take about 4 ms at 1× and
   7 ms at 2× (E, scaled from 1.2 and 2.3 ms measured on the M2 Max's Neural Engine). Its 4K camera
   gives DPT 16–17. What stops it is not compute. It needs a custom iOS app that stays in the
   foreground on the ceiling around the clock, and it is on iOS 18, which Apple still patches but no
   longer upgrades.
6. Recommendation: a Raspberry Pi 5 with a Camera Module 3, about $200. It is the cheapest option that
   fits 2× at 10 Hz on published-figure estimates (50–105 ms float, E, with int8 as margin), and the
   camera gives DPT 20 over the whole layout from one module hung at 1.62 m. Before buying, measure
   the Pi 4. If 1× is chosen, the box's CPU or the Pi 4 with a Camera Module 3 ($25) is enough.
7. On the M2 Max, idle, the batch-of-50 times match HW.md's busy-machine figures within 3%. int8 cuts
   native CPU time 1.4–7.4×, far more than published phone figures for MobileNetV3-Small (1.0–1.35×).
   Accuracy of the int8 model is not measured.

## The requirement

The goal is 50 sensors at 10 Hz on the computer [rails49/detector#60](https://github.com/rails49/detector/issues/60)
chooses [R2, R5]. It replaced "50 sensors at 10 Hz in Safari on an iPhone XS Max, sustained 30 minutes"
[R3]. The camera is one camera over the layout's center at about 1.5 m, at about DPT 16 [R3]. The
layout is 2.0 × 1.0 m, N scale (gauge 9 mm); the field one camera must cover, with margin, is
2.1 × 1.1 m [R7].

| | 1× (48 × 96 input) | 2× (96 × 192 inside the model) | Source |
|---|---|---|---|
| MACs per patch | 6.56 M | 21.05 M | [R11] `metrics.json` |
| MACs per batch of 50 (one frame) | 0.33 G | 1.05 G | E |
| MACs per second at 10 Hz | 3.3 G | 10.5 G | E |

- Time budget. Latency is at most one frame period, so each frame must finish within it [R9]. At 10 Hz
  that is 100 ms per batch. This note plans for 50 ms, leaving half the period for capture, cutting the
  50 patches and the rest of the machine's work (assumption).
- The MAC counts cover convolutions and matrix multiplies only, so they are a lower bound [R1].
- Which size is needed is open. In the run before the data cleanup, 2× had about half the false
  occupied rate of 1× [R1].
- Camera: DPT 16 over 2.1 m needs 2.1 m × 16 / 9 mm = 3,733 px across the layout, at ≥ 10 fps (E).
- The graph is ONNX opset 20 with 52 Conv, 19 HardSwish, 9 HardSigmoid, 10 ReduceMean (squeeze and
  excite), 2 Gemm and, at 2×, one Resize (M, read from the exported models [R11]). HardSwish matters
  for accelerator support below.

## Comparison

Times are for one batch of 50 patches. Target ≤ 50 ms, limit 100 ms.

| Option | Compute and runtime | 1× (ms) | 2× (ms) | Camera, DPT over 2.1 m, rate | Cost | Main risks |
|---|---|---|---|---|---|---|
| 1. Box, CPU | Celeron N5100, 4 Tremont cores, SSE4.2 only; onnxruntime | 18 (M2) | 83–90 (M2) | 4K UVC: 16.5 with a lens of about 70°, 14 with a Brio-type lens; 30 fps (E, [R7]) | box owned; camera to buy | 2× uses all 4 cores; camera decode not measured |
| 1. Box, iGPU | UHD Graphics, 24 EU, 800 MHz; OpenVINO GPU | 32 (M2) | 113 (M2) | as above | as above | slower than the CPU |
| 2. Raspberry Pi 4 | 4 × Cortex-A72, 1.5 or 1.8 GHz; onnxruntime or LiteRT on CPU | 47–63 (E) | 150–200 (E) | Camera Module 3: 19.7 at 1.62 m, 14 fps; Module 2: 14.1 at 1.74 m, 21 fps (E) | owned; Module 3 $25 | 2× misses 100 ms; ISP at 12 MP and 10–14 fps not measured |
| 3. iPhone XS Max, native app | A12 Neural Engine (8 cores, 5 TOPS); Core ML | ~4 (E) | ~7 (E) | 4K60; about 16.5 at about 1.58 m (E) | owned; $99/year or a rebuild every 7 days | app must stay in the foreground; iOS 18 only |
| 3. Galaxy A13 5G, native app | Dimensity 700: 2 × A76 2.2 GHz + 6 × A55, no NPU named; CPU | 30–40 (E) | 95–125 (E) | 1080p30: DPT 8 (P, E) | owned | camera fails DPT 16; no security updates |
| 4. Raspberry Pi 5 | 4 × Cortex-A76, 2.4 GHz; onnxruntime on CPU | 16–33 (E) | 50–105 (E) | Camera Module 3, as for the Pi 4 | $175 (8 GB) + $25 | estimates only |
| Reference: M2 Max | onnxruntime CPU, 8 threads / Core ML on the Neural Engine | 11.6 / 1.2 (M) | 22.4 / 2.3 (M) | – | owned | – |

## Measurements on the M2 Max

Machine: MacBook Pro Mac14,6, Apple M2 Max, 12-core CPU (8 performance, 4 efficiency), 64 GB, macOS
26.6.2. Measured 2026-11-03, 18:03–18:08, after `uv run train` had finished. During the runs no other
process used more than 7% CPU; macOS reported load averages of 2.3–4.5, which include the benchmarks
themselves. So the machine was idle apart from the measurement.

Models: `runs/baseline/1x-ceiling-2026-10-28.onnx` and `2x-ceiling-2026-10-28.onnx` from the run that
finished at 18:03 [R11]. Batch of 50 random patches, 5 warm-up runs, median of 50.

- Native: onnxruntime 1.31.1, `CPUExecutionProvider`, the same loop as `onnx_cpu_ms` in
  `src/classifier/train.py` [R11], run from a script in the scratchpad.
- WASM: `pnpm -C bench bench <threads>`, onnxruntime-web 1.31.1, Node 24.11.1 [R11].
- Core ML: the classifier's `MobileNetV3Small` wrapper, traced and converted with coremltools 9.0 to an
  FP16 ML Program, timed with `MLModel.predict`. Its weights are ImageNet with an untrained head, not
  the trained model; the compute is the same. coremltools warns that torch 2.15 is untested.
- int8: `quantize_static`, QDQ, per-channel int8 weights, uint8 activations, calibrated on random data.
  The resulting models run the whole network (52 `QLinearConv` after optimization) but their outputs do
  not track the float model's, so they are good for timing only.

| Runtime | 1× (ms) | 2× (ms) |
|---|---|---|
| Native CPU, 1 thread | 62.1 | 115.7–116.3 |
| Native CPU, 2 / 4 / 8 threads | 31.7 / 17.4 / 11.6 | 59.6 / 33.0 / 22.4 |
| Native CPU, threads chosen by onnxruntime | 12.2–12.3 | 23.2–23.6 |
| WASM under Node, 1 thread | 96.2–96.4 | 179.4–179.6 |
| WASM under Node, 0 (onnxruntime-web chooses) or 4 threads | 26.1–27.8 | 47.8–50.0 |
| WASM under Node, 8 threads | 15.8 | 29.7 |
| onnxruntime CoreML EP, ML Program, all units | 21.5 | 49.4 |
| Core ML, CPU only | 12.2 | 20.9 |
| Core ML, CPU and GPU | 2.1 | 3.5 |
| Core ML, CPU and Neural Engine | 1.23 | 2.26 |
| int8, native CPU, 1 thread / onnxruntime's choice | 8.4 / 5.9 | 31.5 / 16.0 |
| int8, WASM, 1 / 4 threads | 59.6 / 50.9 | 205.5 / 105.6 |

- The idle figures match HW.md's busy-machine figures (63, 113, 12, 23, 98, 184, 27 ms) within 3%.
  HW.md's 2× WASM with automatic threads, not measured there, is 48 ms.
- The first WASM pass gave 42–43 ms (1×) and 78 ms (2×) at 0 and 4 threads; three later passes gave
  26–28 and 48–50 ms. The int8 WASM row comes from that first pass and may be high in the same way.
- Per core, native 1× runs at 5.3 G MAC/s and 2× at 9.1. The 2× model uses the core better because its
  feature maps are 4× larger. WASM on one core reaches 3.4 and 5.9, about 0.6× native.
- The onnxruntime CoreML EP is slower than the plain CPU. It splits the graph into 21 partitions,
  because HardSwish is not among the operators the EP builds; HardSigmoid is [S32, S31]. Core ML run
  directly (coremltools converts HardSwish) takes the whole network onto the Neural Engine and is
  10× faster than the CPU.
- int8 helps native code on this CPU a lot, especially at 1×, and does not help WASM.

## Option 1: USB camera on the box

The box is `gleis49.org`, a Kamrui JK06 on Ubuntu 24.04 [R10]. Read over ssh [M1]: Intel Celeron N5100,
4 cores, 4 threads, 7.5 GiB RAM, kernel 7.0, one USB 3 root hub at 10 Gb/s and one USB 2 hub, 15
containers (control, dccex, JMRI, installation), load average 0.00. The CPU flags include SSE4.2 and no
AVX, AVX2 or FMA.

CPU. Intel lists the N5100 at 1.1 GHz base, 2.8 GHz burst, 6 W TDP, Jasper Lake, 10 nm, instruction set
extensions SSE4.2 only [S1]. The Tremont core has one 128-bit floating-point multiplier and one 128-bit
adder [S2, secondary]. That is at most 4 FP32 multiply-adds per cycle, 11 G MAC/s per core at 2.8 GHz,
45 G MAC/s for 4 cores (E). onnxruntime's MLAS selects its SSE kernels on such a CPU; the AVX and FMA3
kernels need AVX and AVX2/FMA [S4, lines 422–432 and 507–562]. The SSE4.1 int8 kernel is selected only
in Windows builds; Linux falls back to the SSE2 int8 kernel [S4, lines 481–491]. onnxruntime ships
x86-64 Linux wheels [S36]; whether they run on a CPU without AVX is not stated, so check it first.

Estimate: 10–20% of peak, the range the M2 Max reaches in the table above if each performance core does
16 FP32 multiply-adds per cycle at 3.5 GHz (Apple publishes neither number, so this is an assumption).
That gives 4–8 G MAC/s on 4 cores, 41–82 ms at 1× and 130–260 ms at 2× (E). 1× may fit, 2× does not.

Integrated GPU. Intel UHD Graphics, 24 execution units, 800 MHz burst [S1]. Gen11 does 16 FP32 FLOPs per
EU per clock, twice that in FP16 [S3, secondary, from Intel's figures]. Peak: 154 G MAC/s FP32, 307 G
MAC/s FP16 (E). At an assumed 5–15% of the FP16 peak, 15–46 G MAC/s: 7–22 ms at 1×, 23–70 ms at 2× (E).
OpenVINO lists Intel UHD Graphics and Ubuntu 24.04 for its GPU plugin, and Atom processors with SSE4.2
for its CPU plugin [S5]. The GPU needs Intel's OpenCL driver. Gen11 is a legacy platform there: no new
features, critical fixes only, from the 24.35 branch; Jasper Lake is not named, Elkhart Lake (the same
graphics) is [S6]. This path is plausible and unmeasured.

Camera. The box has USB 3 [M1], so a UVC camera plugs in directly. A 4K camera whose lens covers 2.1 m
at 1.5 m (about 70° horizontal) gives 3,840 / 2,100 mm × 9 = DPT 16.5 (E). A fixed-lens 4K webcam such as
the Logitech Brio covers about 2.65 m and gives DPT 14 [R7]. 4K webcams usually send MJPEG or H.264 at 4K
(not verified per model). The N5100's media engine decodes JPEG in hardware through Intel's media
driver, which supports Jasper Lake [S7]. Uncompressed 3840 × 2160 YUYV at 10 fps is 166 MB/s (E), within
USB 3. Where the box stands relative to the layout is not recorded; a passive USB 3 cable is limited to a
few metres.

Cost and upkeep. Nothing to buy but the camera. The box is already maintained (Ubuntu, docker), and the
MQTT broker that the classifier's output would go to runs on it [M1]. The classifier would add one
container.

Measured [M2]. Batch of 50, median of 50 runs, random input, box idle before the run. onnxruntime
1.31.1 and OpenVINO 2026.4.1 in an Ubuntu 24.04 container with Ubuntu's `intel-opencl-icd`. The
onnxruntime wheel runs without AVX. OpenVINO's GPU plugin loads the models on the UHD Graphics; its
default GPU precision is FP16.

| Model | ort CPU 1 thread | 2 threads | 4 threads | OpenVINO CPU | OpenVINO GPU |
|---|---|---|---|---|---|
| 1× float | 56.2 | 30.5 | 17.9 | 18.7 | 32.2 |
| 2× float | 195.5 | 127.7 | 90.2 | 83.2 | 112.5 |
| 1× int8 | 84.1 | 57.3 | 55.1 | 40.0 | 73.4 |
| 2× int8 | 280.1 | 173.0 | 159.2 | 138.8 | 148.8 |

Times in ms. The int8 models are the random-calibration ones from the M2 Max section; they are slower
than float on this CPU, which has no VNNI or AVX.

Verdict. 1× fits on the CPU with room to spare. 2× fits the 100 ms limit on the CPU, not the 50 ms
target, and leaves no cores for anything else. The iGPU and int8 do not help.

## Option 2: Raspberry Pi 4 with a Pi camera

Hardware. BCM2711, 4 × Cortex-A72 at 1.5 GHz [S8]; the current product page lists 1.8 GHz, which applies
to later boards [S8]. No accelerator that onnxruntime or LiteRT uses for this model; inference runs on
the CPU (onnxruntime ships aarch64 Linux wheels [S36]).

Throughput. PyTorch's tutorial on a Pi 4 Model B 4 GB runs MobileNetV3-Large, int8 (qnnpack, all 4
cores), at 224 × 224 in 30.7 ms model time [S16]. MobileNetV3-Large is 217 M MACs [S17], so 7.1 G MAC/s
(P, E). Google's TFLite figures for MobileNetV3-Small show int8 only 1.0–1.35× faster than float on Pixel
1–3 [S17], so float on the Pi 4 is about 5–7 G MAC/s (E). That gives 47–63 ms at 1× and 150–200 ms at
2× (E). Both sources measure 224 × 224 inputs at batch 1; the M2 Max table above shows the larger 2×
feature maps run more efficiently, which favours 2× somewhat. A 1.8 GHz board is up to 20% faster (E).

Camera. "pi-cam" does not say which module. The Pi 4 has one camera connector [S10].

| Module | Pixels | Field, horizontal | Height for a 2.1 m field | DPT | Full-resolution rate |
|---|---|---|---|---|---|
| Camera Module 2 (IMX219) | 3280 × 2464 | 62.2° | 1.74 m | 14.1 | 21.19 fps [S11] |
| Camera Module 3 (IMX708) | 4608 × 2592 | 66° | 1.62 m | 19.7 | 14.35 fps [R7] |
| Camera Module 3 Wide | 4608 × 2592 | 102° | 1.5 m covers 3.7 m | 11.2 | as above |
| HQ (IMX477), 6 mm lens | 4056 × 3040 | covers 1.57 m at 1.5 m [R7] | needs two cameras | 23 | 10.00 fps [S11] |

Fields and pixels from [S10]; heights and DPT are E (DPT = pixels / 2,100 mm × 9). The Camera Module 3
costs $25 and stays in production until at least January 2030 [S12]. Whether the Pi 4's ISP keeps up
with 12 MP at 10–14 fps while the CPU runs the classifier is not published for this use; Raspberry Pi
documents 12 MP HQ raw frames at 10 fps to fast storage [S37] (not measured here).

Cost and upkeep. Owned. A Camera Module 3 is $25 if the owned camera is a Module 2. One more Linux
machine to update.

Verdict. Fits 1× at 10 Hz with a Camera Module 3. Does not fit 2×.

## Option 3: phone with a custom app

### iPhone XS Max

Hardware. A12 Bionic: 6-core CPU (2 performance, 4 efficiency), 4-core GPU, 8-core Neural Engine at
"up to 5 trillion operations per second" [S25]. Rear wide camera 12 MP, 4K video at 24, 30 or 60 fps,
optical stabilization for video; the cropping "cinematic" stabilization is listed for 1080p and 720p
only [S24]. It shipped with iOS 12 [S24].

Runtime. Core ML on the Neural Engine. On the M2 Max the converted model runs on the Neural Engine in
1.23 ms (1×) and 2.26 ms (2×) (M). The M2 Max's Neural Engine is rated 15.8 trillion operations per
second [S33]. Scaled by 15.8 / 5, assuming time scales with the rated operations per second: about 4 ms and
7 ms on the A12 (E). Even 10× worse would fit 2× in
under 25 ms. The onnxruntime CoreML EP is not the way: it leaves HardSwish on the CPU (above). The
browser path (WASM in Safari) is not measured on this phone; [rails49/detector#42](https://github.com/rails49/detector/issues/42)
did not reach the XS Max either [R4].

Camera. Apple does not list the field of view [S24]. Assuming the 26 mm-equivalent wide camera of the
iPhone 16 measured in [R7] (2.00 m field at 1.5 m), the phone hangs at about 1.58 m for a 2.1 m field
and gives 3,840 / 2,100 mm × 9 = DPT 16.5 (E).

Operating a phone. iOS does not let an app use the camera in the background [S30], so the app must stay
in the foreground with the screen on. A free Apple account installs it for 7 days, after which it must
be rebuilt and reinstalled; the Developer Program costs $99 a year [S28, S29]. The XS Max does not run
iOS 26 [S26]. It still gets iOS 18 security updates: iOS 18.7.11 on 2026-10-26 [S27]. Apple has
announced no end date. detector#42 measured the browser path for the YOLO detector: on the ceiling
the screen and touch are out of reach, and screen mirroring stops while the camera runs [R4]. A native
app that publishes sensor states over the network needs neither. Continuous charging and heat over
months are not published for any iPhone.

Verdict. Compute and camera fit with margin. Choose it only if the custom app is acceptable work.

### Samsung Galaxy A13 5G (SM-A136U1)

The user's phone is the US unlocked SM-A136U1. Samsung's US spec sheet (for SM-A136U on AT&T) lists:
MediaTek Dimensity 700, CPU 2 × 2.2 GHz + 6 × 2.0 GHz, 4 GB RAM, 64 GB storage, 50 MP main camera
(f/1.8, 0.64 µm pixels), USB 2.0, Android 11 at launch, "OS Update 2 Years" and "OS Security Update 4
Years" [S18]. MediaTek lists the Dimensity 700 as 7 nm, 2 × Cortex-A76 up to 2.2 GHz and 6 × Cortex-A55
up to 2 GHz, Mali-G57 MC2, video encode up to 2K at 30 fps; it names no AI processor [S20]. The SM-A136U1
is taken to have the same hardware as the SM-A136U (not checked against a Samsung page for the U1).

Software. The A13 5G received Android 14 (One UI 6) [S22, secondary]. It is no longer on Samsung's
security-update list [S21]. Android deprecated NNAPI in Android 15 [S23]; on this phone NNAPI is still
present but has no NPU to reach, so inference would run on the CPU or on the Mali GPU through LiteRT's GPU
delegate.

Camera. Samsung lists rear video at FHD 1920 × 1080, 30 fps for the A13 5G [S19]; the US sheet leaves the
field blank [S18]. 1,920 px over 2.1 m is DPT 8.2 (E), half of what the classifier is trained on. Full
stills from the 50 MP sensor would give more pixels, but no rate for them is published and Camera2's
largest YUV stream size on this phone is unknown.

Compute (for completeness). Google's TFLite figure for MobileNetV3-Small float on one Pixel 3 large core
is 14.4 ms for 66 M MACs, 4.6 G MAC/s [S17]. Taking one A76 at 2.2 GHz as equal, the two large cores give
about 9 G MAC/s: 30–40 ms at 1×, 95–125 ms at 2× (E).

Verdict. Fails on the camera. Also out of security support.

## Option 4: something to buy

Only candidates that clearly fit are listed.

- Raspberry Pi 5 with a Camera Module 3. BCM2712, 4 × Cortex-A76 at 2.4 GHz [S9]; Raspberry Pi states
  two to three times the CPU performance of the Pi 4 [S13]. On the Pi 4 estimate above that is
  10–21 G MAC/s: 16–33 ms at 1×, 50–105 ms at 2× (E). The A76 has dot-product instructions, so int8 may
  add margin (on the M2 Max int8 cut 2× time 1.4–3.7×, M; not measured on a Pi). Two camera connectors
  [S13], so a second camera is possible later. Price: $110 (4 GB) or $175 (8 GB) at PiShop on
  2026-11-01, after Raspberry Pi's April 2026 memory-driven increases [R8, S14]. Camera Module 3: $25
  [S12]. Power supply, cooler and card are extra. The Pi 5 needs a different camera cable from the Pi 4
  (22-pin connector) [S10].
- Pi 5 with the AI HAT+ (Hailo-8L, $76.95, [R8]). Only if the Pi 5 CPU falls short. Hailo's zoo runs its
  "mobilenet_v3" at 224 × 224 (2 GOPs) at 1,936 fps on a Hailo-8L, on an x86 host with PCIe Gen 3 x4
  [S34], which is 0.5 ms for the 1 G MAC of a 2× batch (E). Two unknowns: whether HardSwish and
  squeeze-and-excite compile as they are in our model, and the Pi 5's single PCIe lane, which cost
  multi-context models 40–50% in [R8].
- An Intel N150 mini PC, to replace or sit beside the box. 4 Gracemont cores up to 3.6 GHz, AVX2,
  6 W base power [S35], so onnxruntime gets its FMA3 kernels [S4]. Fanless N150 boxes started at
  $218–247 barebones in 2024 [R8, secondary]. Not estimated in detail: no published MobileNet figure was
  found for it.

Not listed:

- Raspberry Pi AI Camera (IMX500). The sensor's ISP turns each frame into one input tensor for the
  on-sensor network [S15]. 50 rotated patches per frame do not map onto that without a different model.
- Jetson Orin Nano Super ($399), a Mac mini (M6 from $899) or a recent iPhone: each fits the classifier
  many times over [R6], but they are sized for running the detector too. That is detector#60's question.

## Recommendation

1. Measure first, at no cost:
   - On the box: done [M2]. 1× takes 18 ms, 2× 83–90 ms on 4 cores; the iGPU is slower.
   - On the Pi 4: onnxruntime CPU, 4 threads, float and int8, 1× and 2×, with the camera running.
2. If 1× is the model: the box's CPU with a 4K UVC camera, or the Pi 4 with a Camera Module 3 ($25).
3. If 2× is the model: buy a Raspberry Pi 5 (8 GB, $175) and a Camera
   Module 3 ($25). It sends sensor states to the broker on the box over the network.
4. Do not use the Galaxy A13 5G. Use the iPhone XS Max only if a custom iOS app that runs in the
   foreground on the ceiling is acceptable; it is the only owned device with clear compute margin for
   2×.

Whatever runs the classifier, a native onnxruntime is about 1.6× faster than WASM on one core (M2 Max,
M). If `@rails49/classifier` runs in a browser on the device, use the WASM column.

Both models run in the live path [R12]. If the same computer must also run the detector at 0.3–1.5 Hz (0.04–0.21 TFLOP/s for n at DPT 16
[R9]), none of the four options does that on published figures, except possibly the XS Max's Neural
Engine; that belongs to detector#60.

## Open questions to measure on the device

1. Box: answered [M2].
2. Box: answered [M2].
3. Box: CPU load of capturing 4K at 10 fps from a UVC camera, MJPEG decoded in software and through
   VA-API.
4. Pi 4: which camera module is owned? Batch time at 1× and 2× with the camera streaming 12 MP at
   10–14 fps. Board revision (1.5 or 1.8 GHz).
5. int8: accuracy at the false-clear threshold with a calibration set of real patches, and its speed on
   each device. The int8 timings here use random calibration.
6. Sustained rate over 30 minutes or more, on whichever device is chosen, with the camera running.
7. iPhone XS Max: Core ML time on the Neural Engine with the 4K camera running, and the 4K field at the
   mount height.
8. Galaxy A13 5G, only if stills are considered: the largest YUV stream size and its frame rate from
   `StreamConfigurationMap` on the main camera.

## Sources

rails49 and devices

- [R1] `~/rails49/HW.md`, classifier figures and the four options (2026-11-03).
- [R2] rails49/occupancy#272, comment "Goal change", 2026-11-03:
  <https://github.com/rails49/occupancy/issues/272#issuecomment-6605156548>
- [R3] rails49/occupancy#272, issue body (goals, camera, DPT 16):
  <https://github.com/rails49/occupancy/issues/272>
- [R4] rails49/detector#42, "Resolution: the iPhone is for demonstrations only", 2026-11-02:
  <https://github.com/rails49/detector/issues/42#issuecomment-6575244808>
- [R5] rails49/detector#60, "Choose the computer that runs the detector in operation":
  <https://github.com/rails49/detector/issues/60>
- [R6] rails49/detector `docs/research/compute-hardware.md`:
  <https://github.com/rails49/detector/blob/main/docs/research/compute-hardware.md>
- [R7] rails49/detector `docs/research/cameras.md` (layout, field, Camera Module 3 rate, iPhone 16 field,
  webcam DPT): <https://github.com/rails49/detector/blob/main/docs/research/cameras.md>
- [R8] rails49/detector `docs/research/hailo-8.md` (Pi 5 and AI HAT+ prices, N150 boxes, Pi 5 PCIe):
  <https://github.com/rails49/detector/blob/main/docs/research/hailo-8.md>
- [R9] rails49/detector `docs/research/model-dpt-hardware.md` (latency at most one frame period; detector
  compute): <https://github.com/rails49/detector/blob/main/docs/research/model-dpt-hardware.md>
- [R10] rails49/control `docs/DEPLOY.md` § The layout server:
  <https://github.com/rails49/control/blob/main/docs/DEPLOY.md>
- [R11] rails49/classifier working tree, 2026-11-03: `src/classifier/train.py` (`onnx_cpu_ms`),
  `src/classifier/models.py`, `bench/bench.ts`, `runs/baseline/metrics.json` and the ONNX models.
- [R12] rails49/occupancy ADR 0022, the classifier joins the live path:
  <https://github.com/rails49/occupancy/blob/main/docs/adr/0022-the-classifier-joins-the-live-path.md>
- [M1] The box, read over ssh on 2026-11-03: `lscpu`, `free -h`, `lsusb -t`, `lspci`, `uname -a`,
  `/etc/os-release`, `docker ps`, `uptime`.
- [M2] The box, 2026-11-03, load average 0.13 before the run: `bench_box.py` (median ms for a batch of
  50; onnxruntime CPU at 1, 2 and 4 threads, OpenVINO CPU and GPU with the LATENCY hint) in a
  throwaway `ubuntu:24.04` container with `/dev/dri` passed in. The container, image and files were
  removed afterwards.

Intel and x86 software

- [S1] Intel, Celeron N5100 specifications:
  <https://www.intel.com/content/www/us/en/products/sku/212329/intel-celeron-processor-n5100-4m-cache-up-to-2-80-ghz/specifications.html>
- [S2] WikiChip Fuse, "Intel Unveils the Tremont Microarchitecture" (secondary):
  <https://fuse.wikichip.org/news/2851/intel-unveils-the-tremont-microarchitecture-going-after-st-performance/3>;
  AnandTech, "Intel's new Atom Microarchitecture: The Tremont Core" (secondary):
  <https://www.anandtech.com/show/15009/intels-new-atom-microarchitecture-the-tremont-core/3>
- [S3] Tom's Hardware, "Intel Unveils New Gen11 Graphics Architecture Details", FLOPs per clock table from
  Intel's Gen11 paper (secondary): <https://www.tomshardware.com/news/intel-gen11-graphics-architecture-xe,38878.html>
- [S4] onnxruntime v1.31.1, `onnxruntime/core/mlas/lib/platform.cpp`:
  <https://github.com/microsoft/onnxruntime/blob/v1.31.1/onnxruntime/core/mlas/lib/platform.cpp>
- [S5] OpenVINO, system requirements (`docs/articles_en/about-openvino/release-notes-openvino/system-requirements.rst`):
  <https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/about-openvino/release-notes-openvino/system-requirements.rst>
- [S6] Intel compute-runtime, `documentation/LEGACY_PLATFORMS.md`:
  <https://github.com/intel/compute-runtime/blob/master/documentation/LEGACY_PLATFORMS.md>
- [S7] Intel media-driver, `README.md` (JSL/EHL platforms; JPEG decode and encode):
  <https://github.com/intel/media-driver/blob/master/README.md>
- [S35] Intel, Processor N150 specifications:
  <https://www.intel.com/content/www/us/en/products/sku/241636/intel-processor-n150-6m-cache-up-to-3-60-ghz/specifications.html>
- [S36] PyPI, onnxruntime 1.31.1 files (manylinux x86_64 and aarch64 wheels):
  <https://pypi.org/project/onnxruntime/1.31.1/#files>

Raspberry Pi

- [S8] Raspberry Pi documentation, BCM2711 (`documentation/asciidoc/computers/processors/bcm2711.adoc`):
  <https://www.raspberrypi.com/documentation/computers/processors.html#bcm2711>; Raspberry Pi 4 product
  page (1.8 GHz): <https://www.raspberrypi.com/products/raspberry-pi-4-model-b/>
- [S9] Raspberry Pi documentation, BCM2712:
  <https://www.raspberrypi.com/documentation/computers/processors.html#bcm2712>
- [S10] Raspberry Pi documentation, camera hardware specification and installation (connectors):
  <https://www.raspberrypi.com/documentation/accessories/camera.html>
- [S11] Raspberry Pi documentation, `rpicam-options-common.adoc` (`--list-cameras` modes for IMX219 and
  IMX477): <https://github.com/raspberrypi/documentation/blob/develop/documentation/asciidoc/accessories/camera/rpicam-options-common.adoc>
- [S37] Raspberry Pi documentation, `rpicam-raw.adoc`:
  <https://github.com/raspberrypi/documentation/blob/develop/documentation/asciidoc/accessories/camera/rpicam-raw.adoc>
- [S12] Raspberry Pi, Camera Module 3 product page: <https://www.raspberrypi.com/products/camera-module-3/>
- [S13] Raspberry Pi news, "Introducing Raspberry Pi 5": <https://www.raspberrypi.com/news/introducing-raspberry-pi-5/>
- [S14] Raspberry Pi news, "A new 3GB Raspberry Pi 4 for $83.75, and more memory-driven price increases",
  2026-04-01: <https://www.raspberrypi.com/news/a-new-3gb-raspberry-pi-4-for-83-75-and-more-memory-driven-price-increases/>
- [S15] Raspberry Pi documentation, AI Camera (`ai-camera.adoc`):
  <https://www.raspberrypi.com/documentation/accessories/ai-camera.html>

Models and runtimes

- [S16] PyTorch tutorial, "Real Time Inference on Raspberry Pi 4 and 5":
  <https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html>
- [S17] TensorFlow models, `research/slim/nets/mobilenet/README.md` (MobileNetV3 MACs and Pixel 1–3
  latencies, large core):
  <https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md>
- [S23] Android NDK, Neural Networks API (deprecated in Android 15):
  <https://developer.android.com/ndk/guides/neuralnetworks>
- [S31] onnxruntime, CoreML execution provider (supported operators, compute units):
  <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>
- [S32] onnxruntime v1.31.1, `onnxruntime/core/providers/coreml/builders/op_builder_factory.cc`:
  <https://github.com/microsoft/onnxruntime/blob/v1.31.1/onnxruntime/core/providers/coreml/builders/op_builder_factory.cc>
- [S34] Hailo Model Zoo, Hailo-8L classification table (`docs/public_models/HAILO8L/HAILO8L_classification.rst`):
  <https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_classification.rst>

Samsung and MediaTek

- [S18] Samsung, Galaxy A13 5G spec sheet for AT&T Business (2022):
  <https://image-us.samsung.com/SamsungUS/samsungbusiness/pdfs/spec-sheet/Galaxy-A13-5G-Series-Spec-Sheet-AT-T.pdf>
- [S19] Samsung Philippines, Galaxy A13 5G (SM-A136B) specifications:
  <https://www.samsung.com/ph/smartphones/galaxy-a/galaxy-a13-5g-awesome-black-128gb-sm-a136bzkhphl/>
- [S20] MediaTek, Dimensity 700: <https://i.mediatek.com/dimensity-700>
- [S21] Samsung Mobile Security, security updates scope (read 2026-11-03; no A13 entry):
  <https://security.samsungmobile.com/workScope.smsb>
- [S22] SamMobile, "Galaxy A13 5G Android 14 One UI 6 update" (secondary):
  <https://www.sammobile.com/news/galaxy-a13-5g-android-14-one-ui-6-update/>

Apple

- [S24] Apple Support, iPhone XS Max technical specifications: <https://support.apple.com/en-us/111880>
- [S25] Apple Newsroom, "iPhone Xs and iPhone Xs Max bring the best and biggest displays to iPhone",
  2018-09: <https://www.apple.com/newsroom/2018/09/iphone-xs-and-iphone-xs-max-bring-the-best-and-biggest-displays-to-iphone/>
- [S26] Apple Support, iPhone models compatible with iOS 26:
  <https://support.apple.com/guide/iphone/iphone-models-compatible-with-ios-26-iphe3fa5df43/26/ios/26>
- [S27] Apple Support, Apple security releases (iOS 18.7.11, 2026-10-26): <https://support.apple.com/en-us/100100>
- [S28] Apple Developer, compare memberships (free accounts: profiles expire after 7 days):
  <https://developer.apple.com/support/compare-memberships/>
- [S29] Apple Developer Program ($99 annual membership): <https://developer.apple.com/programs/>
- [S30] Apple Developer, `AVCaptureSession.InterruptionReason.videoDeviceNotAvailableInBackground`:
  <https://developer.apple.com/documentation/avfoundation/avcapturesession/interruptionreason/videodevicenotavailableinbackground>
- [S33] Apple Newsroom, "Apple unveils M2 Pro and M2 Max", 2023-01 (Neural Engine 15.8 trillion
  operations per second):
  <https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/>
