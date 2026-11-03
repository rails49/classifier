"""Which labeled-complete images share a placement, within and across archives."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import numpy.typing as npt
from r49 import Image, Manifest

PLACEMENT_G = 3.0
"""Two images share a placement when every car in each has a car in the other
whose span midpoint is this close."""

_Points = npt.NDArray[np.float64]


def placements(archives: Sequence[tuple[str, Manifest, float]]) -> dict[tuple[str, str], int]:
    """A placement id for each labeled-complete image, by archive name and image
    filename. ``archives`` holds each archive's name, manifest and DPT.

    Pixel coordinates of two archives are compared through the homography their
    shared calibration points give; archives with fewer than four shared points
    share no placement."""
    images = [
        (name, manifest, dpt, image)
        for name, manifest, dpt in archives
        for image in manifest.images
        if image.labeled_complete
    ]
    parent = list(range(len(images)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for (i, (_, ma, dpt, a)), (j, (_, mb, _, b)) in combinations(enumerate(images), 2):
        h = _homography(ma, mb)
        if h is not None and _same(_apply(h, _midpoints(a)), _midpoints(b), PLACEMENT_G * dpt):
            parent[root(j)] = root(i)

    ids: dict[int, int] = {}
    return {
        (name, image.filename): ids.setdefault(root(i), len(ids))
        for i, (name, _, _, image) in enumerate(images)
    }


def _midpoints(image: Image) -> _Points:
    return np.array([((c.p0.x + c.p1.x) / 2, (c.p0.y + c.p1.y) / 2) for c in image.labels]).reshape(
        -1, 2
    )


def _same(a: _Points, b: _Points, tolerance: float) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return bool(d.min(axis=1).max() <= tolerance and d.min(axis=0).max() <= tolerance)


def _homography(a: Manifest, b: Manifest) -> _Points | None:
    """The homography from ``a``'s pixel coordinates to ``b``'s, fitted to the
    calibration points with the same world coordinates in both."""
    if a is b:
        return np.eye(3)

    def by_world(m: Manifest) -> dict[tuple[float, float, float], tuple[float, float]]:
        return {
            (p.world.x, p.world.y, p.world.z): (p.px.x, p.px.y) for p in m.layout.calibration.points
        }

    pa, pb = by_world(a), by_world(b)
    shared = [w for w in pa if w in pb]
    if len(shared) < 4:
        return None
    rows: list[list[float]] = []
    for w in shared:
        (x, y), (u, v) = pa[w], pb[w]
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    return h / h[2, 2]


def _apply(h: _Points, points: _Points) -> _Points:
    q = np.c_[points, np.ones(len(points))] @ h.T
    return q[:, :2] / q[:, 2:]
