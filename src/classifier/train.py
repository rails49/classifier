"""Train and score a classifier in 3 folds, one per archive, and measure its cost.

For each input size and fold, the classifier trains on the other archives
minus a quarter of their placements. Those placements set the threshold: the
highest that calls at most ``FALSE_CLEAR`` of their occupied positions clear.
The held-out archive is scored at that threshold.

Writes, under the output directory:

- ``<size>-<archive>.onnx``: the classifier of each input size and fold. It
  takes ``patches``, B × RGB × 48 × 96 in [0, 1] with the track along the
  width, and returns ``occupied``, B scores in [0, 1].
- ``metrics.json``: per input size, compute per patch, CPU time for a batch of
  50 in native onnxruntime, and per fold the threshold and the rates on the
  held-out archive; the rates of all held-out positions together, each at its
  own fold's threshold. Rates are those of ``scoring.report``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from .augment import Augment
from .data import REGIONS, Index, folds, load_images, read_index
from .models import MobileNetV3Small
from .patches import Cutter, Jitter, Moves, Shape, in_band
from .scoring import report, threshold

FALSE_CLEAR = 0.001
"""The share of occupied positions the threshold may call clear."""
THRESHOLD_SHARE = 0.25
"""Share of the training archives' placements held out to set the threshold."""


@dataclass(frozen=True)
class Setup:
    """How a classifier is trained. ``per_region`` positions of each region
    make a batch, before the band rule drops some."""

    shape: Shape = Shape()
    jitter: Jitter = Jitter()
    steps: int = 4000
    per_region: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


def batch(index: Index, rng: np.random.Generator, setup: Setup) -> tuple[Index, Moves]:
    """Positions drawn evenly across the regions, and their moves. A position
    the moves put in the band is dropped."""
    rows = np.concatenate(
        [
            rng.choice(np.flatnonzero(index.region == region), setup.per_region)
            for region in REGIONS
            if np.any(index.region == region)
        ]
    )
    moves = setup.jitter.draw(rng, len(rows))
    keep = ~in_band(index.ahead_g[rows], index.behind_g[rows], moves.along_g)
    return index.take(rows[keep]), moves.take(keep)


def train(model: nn.Module, cut: Cutter, index: Index, setup: Setup, device: torch.device) -> None:
    """Train ``model`` on ``index``'s positions."""
    rng = np.random.default_rng(setup.seed)
    augment = Augment(g_px=setup.shape.dpt)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), setup.lr, weight_decay=setup.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, setup.lr, total_steps=setup.steps)
    for step in range(setup.steps):
        positions, moves = batch(index, rng, setup)
        x = augment(cut(positions, moves), rng)
        y = torch.from_numpy(positions.occupied).float().to(device)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == setup.steps - 1:
            print(f"  step {step}: loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def predict(model: nn.Module, cut: Cutter, index: Index, device: torch.device) -> np.ndarray:
    """The score of each of ``index``'s positions, unmoved."""
    model.to(device).eval()
    scores = []
    for start in range(0, len(index), 1024):
        rows = np.arange(start, min(start + 1024, len(index)))
        scores.append(torch.sigmoid(model(cut(index.take(rows), Moves.none(len(rows))))).cpu())
    return torch.cat(scores).double().numpy()


def held_out_for_threshold(index: Index, rng: np.random.Generator) -> npt.NDArray[np.bool_]:
    """The positions of ``THRESHOLD_SHARE`` of ``index``'s placements, drawn at random."""
    placements = np.unique(index.placement)
    n = max(1, round(THRESHOLD_SHARE * len(placements)))
    return np.isin(index.placement, rng.choice(placements, n, replace=False))


def compute(model: nn.Module, shape: Shape) -> dict[str, float]:
    """Multiply-accumulates and parameters per patch."""
    model.cpu().eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, *shape.size))
    return {
        "macs": counter.get_total_flops() / 2,
        "params": sum(p.numel() for p in model.parameters()),
    }


def export_onnx(model: nn.Module, shape: Shape, path: Path) -> None:
    model.cpu().eval()
    scored = nn.Sequential(model, nn.Sigmoid()).eval()
    torch.onnx.export(
        scored,
        (torch.zeros(2, 3, *shape.size),),
        path,
        input_names=["patches"],
        output_names=["occupied"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        external_data=False,
    )


def onnx_cpu_ms(path: Path, shape: Shape, threads: int) -> float:
    """Median time, in ms, for one batch of 50 patches in native onnxruntime on CPU.
    ``threads`` 0 lets onnxruntime choose."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((50, 3, *shape.size), dtype=np.float32)
    for _ in range(5):
        session.run(None, {"patches": x})
    times = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, {"patches": x})
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def run(
    cache: Path,
    out: Path,
    sizes: dict[str, Callable[[], nn.Module]],
    setup: Setup,
    device: torch.device,
) -> dict[str, Any]:
    """Train and score each of ``sizes``' classifiers in 3 folds; see the module."""
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(cache / "index.csv", setup.shape.dpt)
    fold = folds(index.archive, index.image, index.placement)
    cut = Cutter(load_images(cache, index), index.dpt, setup.shape, setup.jitter, device)
    archives = sorted(set(fold.tolist()))
    metrics: dict[str, Any] = {"setup": asdict(setup), "false_clear_target": FALSE_CLEAR}
    for size, make in sizes.items():
        print(f"{size}", flush=True)
        result: dict[str, Any] = {"folds": {}}
        over_threshold = np.zeros(len(index))
        for archive in archives:
            print(f" {archive}", flush=True)
            rng = np.random.default_rng(setup.seed)
            training = index.take(fold != archive)
            for_threshold = held_out_for_threshold(training, rng)
            model = make()
            train(model, cut, training.take(~for_threshold), setup, device)
            setting = training.take(for_threshold)
            t = threshold(predict(model, cut, setting, device), setting.occupied, FALSE_CLEAR)
            held_out = index.take(fold == archive)
            score = predict(model, cut, held_out, device)
            over_threshold[fold == archive] = score - t
            result["folds"][archive] = {
                "threshold": t,
                "threshold_placements": np.unique(training.placement[for_threshold]).tolist(),
                "held_out": report(score, held_out, t, rng),
            }
            export_onnx(model, setup.shape, out / f"{size}-{archive}.onnx")
        result["all"] = report(over_threshold, index, 0.0, np.random.default_rng(setup.seed))
        result["compute"] = compute(make(), setup.shape)
        result["onnx_cpu_ms"] = {
            f"threads_{n or 'auto'}": onnx_cpu_ms(
                out / f"{size}-{archives[0]}.onnx", setup.shape, n
            )
            for n in (1, 0)
        }
        metrics[size] = result
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score MobileNetV3-Small in 3 folds.")
    parser.add_argument("--cache", type=Path, default=Path("cache"), help="default: cache")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/baseline"), help="default: runs/baseline"
    )
    parser.add_argument("--steps", type=int, default=Setup.steps, help=f"default: {Setup.steps}")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    sizes: dict[str, Callable[[], nn.Module]] = {
        "1x": lambda: MobileNetV3Small(upsample=1),
        "2x": lambda: MobileNetV3Small(upsample=2),
    }
    run(args.cache, args.out, sizes, Setup(steps=args.steps), device)
