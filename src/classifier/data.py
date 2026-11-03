"""The export's index and image cache, read for training and scoring."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

REGIONS = ("inside", "end", "beside", "elsewhere")


@dataclass(frozen=True)
class Index:
    """The positions of ``index.csv``, one array entry each. ``x`` and ``y`` are in
    the pixels of the image cache at DPT ``dpt``; a blank ``ahead_g`` or
    ``behind_g`` is NaN."""

    dpt: float
    archive: npt.NDArray[np.str_]
    image: npt.NDArray[np.str_]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    heading: npt.NDArray[np.float64]
    occupied: npt.NDArray[np.bool_]
    region: npt.NDArray[np.str_]
    center_g: npt.NDArray[np.float64]
    placement: npt.NDArray[np.int64]
    ahead_g: npt.NDArray[np.float64]
    behind_g: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.x)

    def take(self, rows: npt.NDArray[np.bool_] | npt.NDArray[np.intp]) -> Index:
        """The positions ``rows`` selects, as a mask or as indices."""
        return Index(
            self.dpt, *(getattr(self, f.name)[rows] for f in fields(self) if f.name != "dpt")
        )


def read_index(path: Path, dpt: float) -> Index:
    """Read ``index.csv`` with positions scaled to the image cache at ``dpt``."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def column(name: str) -> npt.NDArray[np.float64]:
        return np.array([float(r[name] or "nan") for r in rows])

    scale = dpt / column("dpt")
    return Index(
        dpt=dpt,
        archive=np.array([r["archive"] for r in rows]),
        image=np.array([r["image"] for r in rows]),
        x=column("x") * scale,
        y=column("y") * scale,
        heading=column("heading"),
        occupied=np.array([r["label"] == "occupied" for r in rows]),
        region=np.array([r["region"] for r in rows]),
        center_g=column("center_g"),
        placement=np.array([int(r["placement"]) for r in rows]),
        ahead_g=column("ahead_g"),
        behind_g=column("behind_g"),
    )


def folds(
    archive: npt.NDArray[np.str_], image: npt.NDArray[np.str_], placement: npt.NDArray[np.int64]
) -> npt.NDArray[np.str_]:
    """The fold of each position, named by archive. A placement belongs to the
    archive of its first image, by archive name and then image filename, so it
    never spans two folds."""
    first: dict[int, tuple[str, str]] = {}
    for a, i, p in zip(archive.tolist(), image.tolist(), placement.tolist(), strict=True):
        first[p] = min(first.get(p, (a, i)), (a, i))
    return np.array([first[p][0] for p in placement.tolist()])


def load_images(cache: Path, index: Index) -> dict[tuple[str, str], npt.NDArray[np.uint8]]:
    """The cached image of each archive and image filename in ``index``, as
    height × width × RGB."""
    directory = cache / f"dpt{index.dpt:g}"
    images: dict[tuple[str, str], npt.NDArray[np.uint8]] = {}
    for archive, image in sorted(
        set(zip(index.archive.tolist(), index.image.tolist(), strict=True))
    ):
        with Image.open(directory / archive / f"{Path(image).stem}.png") as f:
            images[(archive, image)] = np.asarray(f.convert("RGB"))
    return images
