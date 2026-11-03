# Baseline: MobileNetV3-Small

2026-11-03, rails49/classifier#4. The numbers come from `uv run train` on the cache that
`uv run export ~/rails49/r49/archives/iot49/*.r49 --dpt 16` wrote from the three
`ceiling-2026-10-*` archives as of 18:05 that day. The metrics file is
[`2026-11-03-baseline-mobilenetv3-small.json`](2026-11-03-baseline-mobilenetv3-small.json).

## Result

- Neither input size meets both targets when the threshold is set as #5 will set it.
  Over all three held-out archives, 1× has 0.17% false clear and 2.89% false occupied;
  2× has 0.42% false clear and 0.70% false occupied. The targets are 0.1% and 1%.
- 2× is the better size. At the same false clear on the pooled held-out positions,
  it has a third to a half of 1×'s false occupied (table below). Its cost is 3.2× the
  compute.
- The threshold is the weak part. It is set by the one or two lowest occupied scores
  in 4 to 6 placements, and ranges from 0.000 to 0.91 across folds. One fold's
  threshold of 0.000 gives 1× its 7.65% false occupied.
- Most of the lowest-scored occupied positions are 1.6 to 1.9 G beyond a train end,
  near the 2 G limit of occupied.

## Corpus

- 65,957 positions from 34 labeled-complete images in 28 placements, in 3 archives of
  one layout, photographed from the ceiling. 10-29b IMG_2918 is not labeled complete and
  is left out.
- Occupied: 8,674 inside a train, 1,550 near a train end. Clear: 55,470 elsewhere,
  153 near a train end, 110 beside a car on another track.
- Positions where another track comes within 1.5 G within 3 G along the track are left
  out, so turnouts and joins are not scored.
- Track the camera cannot see (under boards, and the lower helix track hidden by the
  raised one) is not in the layout. No sensors go there.

## Setup

- Patch: 3 G across × 6 G along the track, RGB, cut from the DPT 16 cache (48 × 96 px)
  and turned so the track runs along the width. The 2× model upsamples the patch to
  96 × 192 inside the model, so both ONNX models take 48 × 96.
- Model: torchvision `mobilenet_v3_small` with ImageNet weights and one output. The
  ImageNet normalization is inside the model; the ONNX input is RGB in [0, 1] and the
  output is the occupied score in [0, 1].
- Moves: random 180° turn and mirror; jitter across ±0.3 G, along ±0.5 G, angle ±4°,
  scale ±10%. A position jittered to 2–2.5 G from the nearest car span on its track is
  dropped.
- Augmentation: brightness 0.6–1.4, contrast 0.7–1.3, gamma 0.7–1.4, white balance
  ±12% per channel, hue ±15°, grayscale 10%; Gaussian blur (30%, σ 0.3–1.2 px), motion
  blur along the track (30%, up to 1 G), noise (σ up to 0.04), JPEG (50%, quality 25–90).
- Batches: 64 positions from each of the four regions (inside, end, beside, elsewhere),
  before the band rule drops some.
- Training: 4,000 steps, AdamW, learning rate 1e-3 with one-cycle schedule, weight decay
  1e-4, binary cross-entropy. About 6 minutes per fold at 1× and 9 at 2× on the M2 Max GPU.
- Folds: one per archive. A placement goes with the archive of its first image, so
  placement 23 (10-29 IMG_2902 and 10-29b IMG_2903) is in the 10-29 fold.
- Threshold: a quarter of the training archives' placements (seed 0) are held out of
  training. The threshold is the highest that calls at most 0.1% of their occupied
  positions clear.
- Confidence intervals: 95%, from 2,000 resamples of whole placements.

## Rates at each fold's threshold

Rate (errors/positions) [95% interval]. "All" scores every held-out position at its own
fold's threshold.

### 1×

| Held out | Threshold | False clear | False occupied |
|---|---|---|---|
| ceiling-2026-10-28 | 0.4194 | 0.21% (6/2898) [0.07%, 0.34%] | 0.08% (18/21298) [0.04%, 0.13%] |
| ceiling-2026-10-29 | 0.0000 | 0.05% (2/4127) [0.00%, 0.12%] | 7.65% (1577/20620) [6.54%, 8.62%] |
| ceiling-2026-10-29b | 0.1398 | 0.28% (9/3199) [0.00%, 0.62%] | 0.11% (15/13815) [0.03%, 0.49%] |
| All | | 0.17% (17/10224) [0.07%, 0.25%] | 2.89% (1610/55733) [1.60%, 4.58%] |

| Region | False clear | False occupied |
|---|---|---|
| inside | 0.01% (1/8674) [0.00%, 0.04%] | – |
| end | 1.03% (16/1550) [0.40%, 1.61%] | 27.45% (42/153) [14.97%, 41.74%] |
| beside | – | 15.45% (17/110) [7.29%, 26.67%] |
| elsewhere | – | 2.80% (1551/55470) [1.51%, 4.53%] |

| From image center (G) | False clear | False occupied |
|---|---|---|
| 0–50 | 0.09% (2/2302) [0.00%, 0.20%] | 4.32% (544/12601) [2.05%, 7.02%] |
| 50–75 | 0.27% (9/3331) [0.04%, 0.42%] | 4.10% (565/13767) [2.24%, 6.26%] |
| 75–100 | 0.20% (6/3066) [0.05%, 0.41%] | 1.68% (319/19008) [0.88%, 2.70%] |
| 100 and more | 0.00% (0/1525) [0.00%, 0.00%] | 1.76% (182/10357) [0.84%, 3.08%] |

### 2×

| Held out | Threshold | False clear | False occupied |
|---|---|---|---|
| ceiling-2026-10-28 | 0.9142 | 1.24% (36/2898) [0.48%, 2.10%] | 0.09% (20/21298) [0.04%, 0.15%] |
| ceiling-2026-10-29 | 0.0029 | 0.00% (0/4127) [0.00%, 0.00%] | 1.75% (361/20620) [1.38%, 2.21%] |
| ceiling-2026-10-29b | 0.6545 | 0.22% (7/3199) [0.12%, 0.43%] | 0.07% (9/13815) [0.00%, 0.39%] |
| All | | 0.42% (43/10224) [0.18%, 0.81%] | 0.70% (390/55733) [0.39%, 1.11%] |

| Region | False clear | False occupied |
|---|---|---|
| inside | 0.35% (30/8674) [0.10%, 0.75%] | – |
| end | 0.84% (13/1550) [0.34%, 1.34%] | 15.69% (24/153) [9.46%, 24.36%] |
| beside | – | 12.73% (14/110) [5.35%, 22.81%] |
| elsewhere | – | 0.63% (352/55470) [0.34%, 1.05%] |

| From image center (G) | False clear | False occupied |
|---|---|---|
| 0–50 | 0.22% (5/2302) [0.05%, 0.39%] | 0.79% (100/12601) [0.33%, 1.41%] |
| 50–75 | 0.51% (17/3331) [0.16%, 1.24%] | 0.96% (132/13767) [0.51%, 1.52%] |
| 75–100 | 0.16% (5/3066) [0.00%, 0.43%] | 0.42% (80/19008) [0.17%, 0.78%] |
| 100 and more | 1.05% (16/1525) [0.00%, 2.50%] | 0.75% (78/10357) [0.28%, 1.52%] |

## Sizes at the same false clear

One threshold over all held-out scores of a size, set to give the false clear in the
first column. A deployed classifier cannot set its threshold this way; the table shows
how well each size separates occupied from clear, apart from the threshold rule.

| False clear | 1× false occupied | 2× false occupied |
|---|---|---|
| 0.10% | 1.29% | 0.41% |
| 0.17% | 0.65% | 0.32% |
| 0.42% | 0.24% | 0.20% |
| 1.00% | 0.10% | 0.13% |

## Cost

Per patch, and for one batch of 50 patches on the M2 Max (12 cores), median of 50 runs
after 5 warm-up runs.

| | 1× | 2× |
|---|---|---|
| Multiply-accumulates per patch | 6.6 M | 21.0 M |
| Parameters | 1.52 M | 1.52 M |
| ONNX file (float32) | 6.4 MB | 6.4 MB |
| Native onnxruntime CPU, 1 thread | 62 ms | 114 ms |
| Native onnxruntime CPU, threads chosen by onnxruntime | 12 ms | 23 ms |
| onnxruntime-web 1.31 WASM under Node 24, 1 thread | 97 ms | 179 ms |
| onnxruntime-web 1.31 WASM under Node 24, default threads | 27 ms | 50 ms |

Native times are in the metrics file. WASM times come from
`pnpm -C bench bench <threads> runs/baseline/*.onnx`.

## Caveats

- The corpus is one layout and one set of stock. The results hold for this layout only.
- It has 274 train ends in 28 placements. Images of one placement differ only in light,
  so these are the independent cases near train ends, where most false clears are.
  That is too few to confirm a false clear of 0.1%.
- The threshold rule needs more held-out placements, or a different rule, before the
  per-fold rates mean much. #5 ships with this rule.
- The multiply-accumulate counts come from PyTorch's `FlopCounterMode`, which counts
  convolutions and matrix multiplies only. They are a lower bound.
- Codec artefacts are JPEG only, not video codecs.
- Beside has 110 positions and clear near a train end 153, so their rates have wide
  intervals.
- The score on the 4K video archive (#2) is not here yet.
