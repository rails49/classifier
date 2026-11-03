"""Export a position index and an image cache from ``.r49`` archives.

Writes, under the output directory:

- ``index.csv``: one row per labeled position of each labeled-complete image.
  ``x`` and ``y`` are in the archive's image pixels, ``heading`` in radians as
  ``TrackNode.heading``, and ``dpt`` is the archive's. A position at DPT ``d``
  is at ``x · d / dpt``.
- ``counts.json``: per archive, the positions written and those left out, by reason.
- ``dpt<d>/<archive>/<image stem>.png``: each labeled-complete image resampled
  to DPT ``d``. A cached image of the size that DPT gives is not resampled again.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path

import r49
from PIL import Image

from .placement import placements
from .positions import LeftOut, label_positions

COLUMNS = [
    "archive",
    "image",
    "x",
    "y",
    "heading",
    "label",
    "region",
    "center_g",
    "placement",
    "dpt",
]


def export(archives: Sequence[Path], out: Path, dpts: Sequence[float]) -> None:
    """Export ``archives`` into ``out``, caching their images at each of ``dpts``."""
    loaded = [(path.stem, r49.load(path)) for path in archives]
    fitted = [(name, archive.manifest, _dpt(name, archive)) for name, archive in loaded]
    groups = placements(fitted)

    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    with (out / "index.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for (name, archive), (_, manifest, dpt) in zip(loaded, fitted, strict=True):
            size = (manifest.camera.resolution.width, manifest.camera.resolution.height)
            done = [image for image in manifest.images if image.labeled_complete]
            left_out = LeftOut()
            occupied = clear = 0
            for image in done:
                positions, out_of_image = label_positions(
                    manifest.layout.tracks, image.labels, dpt, size
                )
                left_out.outside += out_of_image.outside
                left_out.band += out_of_image.band
                left_out.near_track += out_of_image.near_track
                for p in positions:
                    writer.writerow(
                        [
                            name,
                            image.filename,
                            f"{p.x:.2f}",
                            f"{p.y:.2f}",
                            f"{p.heading:.4f}",
                            p.label,
                            p.region,
                            f"{p.center_g:.2f}",
                            groups[(name, image.filename)],
                            dpt,
                        ]
                    )
                    occupied += p.label == "occupied"
                    clear += p.label == "clear"
                for target in dpts:
                    _cache(archive, image.filename, dpt, target, out / f"dpt{target:g}" / name)
            counts[name] = {
                "images": len(manifest.images),
                "not_labeled_complete": len(manifest.images) - len(done),
                "positions": occupied + clear,
                "occupied": occupied,
                "clear": clear,
                "band": left_out.band,
                "outside": left_out.outside,
                "near_track": left_out.near_track,
            }
    (out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")


def _dpt(name: str, archive: r49.R49Archive) -> float:
    dpt = r49.calibration_fit(archive.manifest).dpt
    if dpt is None:
        raise SystemExit(f"{name}: the calibration fit gives no DPT")
    return dpt


def _cache(archive: r49.R49Archive, filename: str, dpt: float, target: float, into: Path) -> None:
    path = into / f"{Path(filename).stem}.png"
    resolution = archive.manifest.camera.resolution
    size = (round(resolution.width * target / dpt), round(resolution.height * target / dpt))
    if path.exists():
        with Image.open(path) as cached:
            if cached.size == size:
                return
    data = archive.image(filename)
    if data is None:
        raise SystemExit(f"{filename} is missing from the archive")
    with Image.open(io.BytesIO(data)) as image:
        resampled = image.convert("RGB").resize(size, Image.Resampling.LANCZOS, reducing_gap=3)
    into.mkdir(parents=True, exist_ok=True)
    resampled.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a position index and an image cache.")
    parser.add_argument("archives", nargs="+", type=Path, help=".r49 archives")
    parser.add_argument("--out", type=Path, default=Path("cache"), help="default: cache")
    parser.add_argument(
        "--dpt", type=float, action="append", help="cache images at this DPT (repeatable; 16)"
    )
    args = parser.parse_args()
    export(args.archives, args.out, args.dpt or [16.0])
