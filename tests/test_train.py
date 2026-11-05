import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from classifier import Classifier, Sensor
from classifier.cut import Shape
from classifier.train import Setup, run

SHAPE = Shape(across_g=2.0, along_g=4.0, dpt=8.0)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(1, 2, 3))


def write_cache(cache: Path) -> None:
    """Two archives of four placements, one image each, with an occupied and a clear position."""
    rng = np.random.default_rng(0)
    rows = []
    for archive, first_placement in (("a", 0), ("b", 4)):
        (cache / "dpt8" / archive).mkdir(parents=True)
        for i in range(4):
            pixels = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(cache / "dpt8" / archive / f"{i}.png")
            for label, region, x, ahead in (
                ("occupied", "inside", 50, 0),
                ("clear", "end", 110, 3),
            ):
                rows.append(
                    {
                        "archive": archive,
                        "image": f"{i}.jpg",
                        "dpt": 8,
                        "x": x,
                        "y": 60,
                        "heading": 0,
                        "label": label,
                        "region": region,
                        "center_g": 1,
                        "placement": first_placement + i,
                        "ahead_g": ahead,
                        "behind_g": ahead,
                    }
                )
    with (cache / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_training_writes_beside_each_model_the_json_file_that_run_time_reads(tmp_path: Path):
    write_cache(tmp_path / "cache")
    out = tmp_path / "out"

    metrics = run(
        tmp_path / "cache",
        out,
        {"tiny": Tiny},
        Setup(shape=SHAPE, steps=2, per_region=2),
        torch.device("cpu"),
    )

    for archive in ("a", "b"):
        assert (out / f"tiny-{archive}.onnx").exists()
        assert json.loads((out / f"tiny-{archive}.json").read_text()) == {
            "across_g": 2.0,
            "along_g": 4.0,
            "dpt": 8.0,
            "threshold": metrics["tiny"]["folds"][archive]["threshold"],
        }
    assert "held_out" in json.loads((out / "metrics.json").read_text())["tiny"]["folds"]["a"]

    model = Classifier(out / "tiny-a.onnx")
    frame = np.full((120, 160, 3), 90, dtype=np.uint8)
    result = model.score(frame, [Sensor("s", 80.0, 60.0, 0.0)], 8.0)["s"]

    assert model.threshold == metrics["tiny"]["folds"]["a"]["threshold"]
    assert result.p_occupied is not None
    assert result.patch is not None and result.patch.shape == (16, 32, 3)
