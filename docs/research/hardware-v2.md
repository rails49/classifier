# Hardware for the classifier, version 2

Written 2026-11-04. Measurements for the brief in [`hardware-v2-brief.md`](hardware-v2-brief.md). The Mac mini is in use until it
no longer keeps up, then gets replaced. No camera was attached, so USB capture is not measured.
Item 3 is covered only in part and item 4 not at all.

Evidence tags: M measured here, E arithmetic from measurements.

## Answer

For 50 sensors at 2× (compute only, with no capture, decode or patch cutting):

| Runtime | 10 Hz, 5 min | 20 Hz, 5 min | Flat out |
|---|---|---|---|
| OpenVINO on the UHD 630 | 0 frames missed, 17.9 ms median, 20.7 ms max, 0.2 core, 53–70 °C | 0 missed, 17.8 ms median, 19.7 ms max, 0.4 core, 57–69 °C | 70 Hz, short runs only |
| onnxruntime CPU, 6 threads | 0 missed, 14.0 ms median, 38.1 ms max, 4 cores, 80–100 °C | 6 missed, 14.6 ms median, 60.4 ms max, 5.5 cores, 65–100 °C | 68 Hz, short runs only |
| onnxruntime CPU, 12 threads | not run | not run | 57 Hz for 11 min, then 45 Hz to the end at 20 min |

1. Use the GPU with float models. Both rates ran with no missed frames and no stalls, about 30 °C
   below the CPU's limit, leaving at least 5 of the 6 cores free.
2. The CPU also reaches both rates, but at temperatures that make it throttle. Under sustained load it
   stalls.
3. 4K decode is within reach: about 54 fps on one core in software at the largest likely frame size.
   VA-API with frames copied back to system memory is slower, at 26–28 fps. If patches are cut on the
   CPU, decode in software.
4. Per frame at 20 Hz, sequential work has 50 ms: decode 18.5 ms and classify 17.8 ms leave about 14 ms
   for capture and cutting (E). Patch cutting is not measured. Work done in parallel has more room.
5. int8 does not help on this machine. It is slower than float on the CPU, and about 8× slower on the
   GPU at 2×.

## Machine

Mac mini 2018 (`Macmini8,1`), Core i7-8700B, 6 cores and 12 threads, 32 GB, UHD 630. Host `gleis49`,
Ubuntu 26.04.1 (not 24.04 as the brief says), kernel 7.3.1-2-t2-resolute. Governor `powersave` with
`balance_performance`, left as installed. `t2fanrd` is not running; the SMC manages the fan.

Runtimes in an Ubuntu 24.04 container (`bench/hardware/Dockerfile`): onnxruntime 1.31.1, OpenVINO
2026.4.1, Intel compute runtime and iHD media driver 24.1.0 from Ubuntu. All runs at nice 10. The 15
layout containers kept running with no railroad loaded and used 3% of the CPU or less.

## Throughput (item 1)

Rate for 50 sensors in Hz, from batches of 50: patches per second / 50. Float, 3 s or more per
setting, random input, in shuffled order on a hot machine. `*` marks a run caught in a slow episode
(see Heat).

| Runtime, size | 1 thread | 2 | 4 | 6 | 12 |
|---|---:|---:|---:|---:|---:|
| onnxruntime, DPT 12 | 15* | 121 | 191 | 239 | 194 |
| onnxruntime, DPT 16 (1×) | 71 | 106 | 162 | 208 | 180 |
| onnxruntime, DPT 20 | 51 | 79 | 140 | 46* | 152 |
| onnxruntime, 2× | 20 | 33 | 59 | 68 | 60 |
| OpenVINO CPU, DPT 12 | 92 | 147 | 192 | 218 | 179 |
| OpenVINO CPU, DPT 16 (1×) | 77 | 128 | 170 | 190 | 160 |
| OpenVINO CPU, DPT 20 | 54 | 92 | 130 | 147 | 127 |
| OpenVINO CPU, 2× | 23 | 43 | 66 | 80 | 73 |

OpenVINO on the UHD 630, by number of sensors (one batch per frame):

| Size | 25 | 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|---:|
| DPT 12 | 258 | 180 | 116 | 84 | 66 |
| DPT 16 (1×) | 229 | 153 | 97 | 67 | 52 |
| DPT 20 | 187 | 122 | 71 | 49 | 38 |
| 2× | 124 | 70 | 39 | 26 | 20 |

- One call per patch is much slower on the GPU: 717–849 patches/s against 3,091–13,197 in batches.
  On the CPU it gives about 970–2,700 patches/s, apart from one run in a slow episode. Send batches.
- 12 threads is slower than 6 at every size in batches of 50, in all but the runs marked `*`.
- GPU runs at full speed keep about one core busy. Paced at 20 Hz, that falls to 0.37 core.
- Peak process memory at 2×, batch 50: onnxruntime 134 MB, OpenVINO CPU 161 MB, GPU 696 MB.
- Trained and untrained models time within 1.5% (1×) and 0.3% (2×).
- int8 models come from random calibration and are good for timing only. 2× at batch 50, 6 threads:
  onnxruntime 2,024 against 3,397 patches/s, OpenVINO CPU 2,477 against 3,995. On the GPU: 450 against
  3,475.
- The UHD 630 is about 8× faster than the 24-unit GPU in `hardware-v1.md` (14 ms against 112 ms for
  2×). The clocks alone do not explain that, and the cause is not known. The device name, outputs
  within 0.0015 of the CPU's and uncore power confirm the work runs on the GPU.

Raw results: `hardware-v2-grid.jsonl`. Scripts: `bench/hardware/`.

## Heat

The die reaches about 100 °C within 5 s of full CPU load at about 60 W, while the sensor near the CPU
stays at 46–56 °C and the case stays warm. The reported fan speed goes from 1,700 RPM to its 4,400 RPM
maximum in about 90 s, and the fan was heard running. Power stays under the 100 W limit, so
temperature limits the clock: about 3.1 GHz on all cores.

Under sustained CPU load, two kinds of slowdown appear. Their cause and trigger are not known.

- Long episodes. During the 38-minute grid, 26 of 400 CPU runs at batch 25 or above fell by 3.4–5.4×,
  with all cores near 800 MHz and power at 5–8 W. They recur about every 60–100 s and last long enough
  to cover several runs of 3–5 s. The die cools to about 50 °C, then returns to about 100 °C. They did
  not appear in the 20-minute flat-out run.
- Short stalls. Calls of 3–4× the median started at about 300 s in the grid and 660 s in the flat-out
  run, then stayed. In the flat-out run p95 went from 21 ms to 57 ms and the rate from 57 Hz to 45 Hz.
  The kernel logged throttle events of up to 212 ms.

Some GPU runs in the grid show p95 of about 9× the median, on a die heated by the CPU runs around them.
No paced GPU run shows this.

At 10 Hz the CPU run has 14 ms of work per 100 ms but keeps 4 cores busy and the die at a median of
96 °C. onnxruntime's worker threads spinning between calls would explain that (not tested).

## Decode (item 2, in part)

Clips are made from the 48 MP ceiling photos, scaled to DPT 19.5 (4K) and 9.7 (1080p) and cropped, as
4:2:2 JPEG at three sizes. USB 2.0's isochronous limit of about 24 MB/s allows about 800 kB per 4K
frame at 30 fps (E). The camera's own frame size is not known. ffmpeg 6.1, one process, no USB.

Highest rate reached, in frames per second:

| Clip, kB per frame | Software, 1 core | VA-API, frames on GPU | VA-API, copied to memory |
|---|---:|---:|---:|
| 4K, 812 | 54 | 254 | 26 |
| 4K, 529 | 55 | 254 | 28 |
| 4K, 304 | 75 | 261 | 27 |
| 1080p, 311 | 154 | 924 | 120 |
| 1080p, 195 | 191 | 985 | 126 |
| 1080p, 100 | 265 | 992 | 132 |

Cores busy when paced, for the 4K 812 kB clip:

| Mode | 10 fps | 15 | 20 | 30 |
|---|---:|---:|---:|---:|
| Software | 0.41 | 0.50 | 0.51 | 0.64 |
| VA-API, frames on GPU | 0.15 | 0.25 | 0.38 | 0.50 |
| VA-API, copied to memory | 0.48 | 0.54 | 0.69 | 0.91 (reached 27 fps) |

Software and copied modes show a floor of about 0.4 core at low rates. It is above what the highest
rate implies (0.19 core at 10 fps, E), and the cause is not known.

Raw results: `hardware-v2-decode.jsonl`.

## Not measured

- USB capture and real camera frames, including their size and whether they are 4:2:0 or 4:2:2.
- Patch cutting, and so the whole chain together (item 3). Decode and classification were not run at
  the same time.
- The 30-minute run of the full chain (item 4). The paced GPU runs lasted 5 minutes, and the CPU's
  stalls began after 5–11 minutes in other runs.
- The GPU flat out for more than a few seconds.
- OpenVINO on the CPU at a paced rate, and onnxruntime with thread spinning turned off.
- Running trains while classifying.
