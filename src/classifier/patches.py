"""Patches cut from the image cache, turned to the track, with the geometric
jitter of training and the band rule that drops a jittered position."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from .data import Index
from .positions import CLEAR_G, OCCUPIED_G

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


@dataclass(frozen=True)
class Jitter:
    """The largest moves training draws, uniformly within ±each."""

    along_g: float = 0.5
    across_g: float = 0.3
    angle_deg: float = 4.0
    scale: float = 0.1

    def draw(self, rng: np.random.Generator, n: int) -> Moves:
        def uniform(limit: float) -> _Floats:
            return rng.uniform(-limit, limit, n)

        return Moves(
            along_g=uniform(self.along_g),
            across_g=uniform(self.across_g),
            angle=np.radians(uniform(self.angle_deg)),
            scale=1 + uniform(self.scale),
            turn=rng.random(n) < 0.5,
            mirror=rng.random(n) < 0.5,
        )


def in_band(ahead_g: _Floats, behind_g: _Floats, along_g: _Floats) -> _Bools:
    """Whether a position moved ``along_g`` lands between ``OCCUPIED_G`` and
    ``CLEAR_G`` from the nearest car span on its track. ``ahead_g`` and
    ``behind_g`` are the index's, NaN where no span lies that way."""
    ahead = np.nan_to_num(ahead_g, nan=np.inf) - along_g
    behind = np.nan_to_num(behind_g, nan=np.inf) + along_g
    gap = np.minimum(np.maximum(ahead, 0), np.maximum(behind, 0))
    return (gap > OCCUPIED_G) & (gap <= CLEAR_G)


class Cutter:
    """Cuts patches of ``shape`` from ``images``, each height × width × RGB at
    DPT ``dpt`` and keyed by archive and image filename. Moves may reach as far
    as ``jitter``'s."""

    def __init__(
        self,
        images: Mapping[tuple[str, str], npt.NDArray[np.uint8]],
        dpt: float,
        shape: Shape,
        jitter: Jitter,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dpt, self.shape, self.device = dpt, shape, torch.device(device)
        reach_g = math.hypot(shape.across_g, shape.along_g) / 2 * (1 + jitter.scale)
        reach_g += math.hypot(jitter.across_g, jitter.along_g)
        self._r = math.ceil(reach_g * dpt) + 2
        pad = ((self._r, self._r), (self._r, self._r), (0, 0))
        self._images = {key: np.pad(image, pad) for key, image in images.items()}

    def __call__(self, index: Index, moves: Moves) -> torch.Tensor:
        """The patches at ``index``'s positions, moved, as B × RGB × height ×
        width in [0, 1] on the cutter's device."""
        r, px_per_g = self._r, self.dpt
        cx, cy = np.round(index.x).astype(int), np.round(index.y).astype(int)
        windows = np.stack(
            [
                self._images[(a, i)][y : y + 2 * r, x : x + 2 * r]
                for a, i, x, y in zip(index.archive, index.image, cx, cy, strict=True)
            ]
        )
        t = index.heading
        x = index.x + (moves.along_g * np.cos(t) - moves.across_g * np.sin(t)) * px_per_g - cx + r
        y = index.y + (moves.along_g * np.sin(t) + moves.across_g * np.cos(t)) * px_per_g - cy + r
        phi = t + moves.angle + np.pi * moves.turn
        k = px_per_g / self.shape.dpt * moves.scale
        h, w = self.shape.size
        u = np.arange(w) + 0.5 - w / 2
        v = np.arange(h) + 0.5 - h / 2
        v_b = np.where(moves.mirror[:, None], -v, v) * k[:, None]  # B × h
        u_b = u * k[:, None]  # B × w
        cos, sin = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]
        sx = x[:, None, None] + u_b[:, None, :] * cos - v_b[:, :, None] * sin
        sy = y[:, None, None] + u_b[:, None, :] * sin + v_b[:, :, None] * cos
        grid = np.stack([sx / r - 1, sy / r - 1], axis=-1).astype(np.float32)
        source = torch.from_numpy(windows).to(self.device).permute(0, 3, 1, 2).float() / 255
        return F.grid_sample(
            source, torch.from_numpy(grid).to(self.device), mode="bilinear", align_corners=False
        )
