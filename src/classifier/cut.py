"""Where a patch's pixels sample an image, used by training and at run time, and
the cut of a live frame at run time. Pixel ``i`` of an image or frame covers
[i, i + 1), so the pixel's center is at i + 0.5."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

_Floats = npt.NDArray[np.float64]
_Bools = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Shape:
    """A patch ``across_g`` × ``along_g`` gauges at ``dpt``, the track along its width."""

    across_g: float = 3.0
    along_g: float = 6.0
    dpt: float = 16.0

    @property
    def size(self) -> tuple[int, int]:
        """Height and width in pixels."""
        return round(self.across_g * self.dpt), round(self.along_g * self.dpt)


@dataclass(frozen=True)
class Moves:
    """How each patch departs from its position: shifted ``along_g`` with the
    track's heading and ``across_g`` to its right in image pixels, turned by
    ``angle`` radians, scaled by ``scale``, turned 180° and mirrored across the
    track."""

    along_g: _Floats
    across_g: _Floats
    angle: _Floats
    scale: _Floats
    turn: _Bools
    mirror: _Bools

    @staticmethod
    def none(n: int) -> Moves:
        zeros, falses = np.zeros(n), np.zeros(n, dtype=bool)
        return Moves(zeros, zeros, zeros, np.ones(n), falses, falses)

    def take(self, rows: _Bools) -> Moves:
        """The moves ``rows`` selects."""
        return Moves(*(getattr(self, f.name)[rows] for f in fields(self)))


def sample_points(
    x: _Floats, y: _Floats, heading: _Floats, dpt: float, shape: Shape, moves: Moves
) -> tuple[_Floats, _Floats]:
    """The x and y that each pixel of each patch samples in an image at ``dpt``,
    each B × height × width. ``x``, ``y`` and ``heading`` are the patches'
    centers in the image's pixels and their headings in radians."""
    sx = x + (moves.along_g * np.cos(heading) - moves.across_g * np.sin(heading)) * dpt
    sy = y + (moves.along_g * np.sin(heading) + moves.across_g * np.cos(heading)) * dpt
    phi = heading + moves.angle + np.pi * moves.turn
    k = dpt / shape.dpt * moves.scale
    h, w = shape.size
    u = np.arange(w) + 0.5 - w / 2
    v = np.arange(h) + 0.5 - h / 2
    v_b = np.where(moves.mirror[:, None], -v, v) * k[:, None]  # B × h
    u_b = u * k[:, None]  # B × w
    cos, sin = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]
    return (
        sx[:, None, None] + u_b[:, None, :] * cos - v_b[:, :, None] * sin,
        sy[:, None, None] + u_b[:, None, :] * sin + v_b[:, :, None] * cos,
    )


def cut(
    frame: npt.NDArray[np.uint8],
    x: _Floats,
    y: _Floats,
    heading: _Floats,
    dpt: float,
    shape: Shape,
) -> npt.NDArray[np.float32]:
    """The patches at ``x``, ``y`` and ``heading`` of ``frame``, height × width ×
    RGB, 8 bits, at DPT ``dpt``, as B × height × width × RGB in [0, 255]. Sampling
    is bilinear with black outside the frame, as in training."""
    sx, sy = sample_points(x, y, heading, dpt, shape, Moves.none(len(x)))
    height, width = frame.shape[:2]
    px, py = sx - 0.5, sy - 0.5
    x0, y0 = np.floor(px).astype(np.intp), np.floor(py).astype(np.intp)
    fx, fy = (px - x0)[..., None], (py - y0)[..., None]

    def at(i: npt.NDArray[np.intp], j: npt.NDArray[np.intp]) -> _Floats:
        inside = (i >= 0) & (i < width) & (j >= 0) & (j < height)
        return frame[np.clip(j, 0, height - 1), np.clip(i, 0, width - 1)] * inside[..., None]

    top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
    bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)
