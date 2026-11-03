"""Positions along an archive's tracks, and the label of each in one image.

Lengths are in image pixels; ``dpt`` converts gauges to pixels.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import numpy.typing as npt
from r49 import CarLabel, Track, point_at, segment

Label = Literal["occupied", "clear"]
Region = Literal["inside", "end", "beside", "elsewhere"]
"""Inside a train, within 3 G of a train end, clear beside an occupied track, clear elsewhere."""

OCCUPIED_G = 2.0
"""A position is occupied when a car on its track comes within this many gauges."""
CLEAR_G = 2.5
"""A position is clear when no car on its track comes within this many gauges."""
NEAR_TRACK_G = 1.5
"""A position with another track this close, as at a turnout or a join, is left out."""
END_G = 3.0
"""A position this close to a train end, along its track, is in the region ``end``."""
BESIDE_G = 3.0
"""A clear position this close to a car on another track is in the region ``beside``."""

_STEP_PX = 1.0
"""Spacing of the points a track's curve is sampled at."""


@dataclass(frozen=True)
class Position:
    """A position on a track, labeled for one image."""

    x: float
    y: float
    heading: float
    """The direction the track runs, in radians, as ``TrackNode.heading``."""
    label: Label
    region: Region
    center_g: float
    """Distance from the image center, in gauges."""


@dataclass
class LeftOut:
    """Positions left out of one image, by reason."""

    margin: int = 0
    """Between 2 and 2.5 G from a car on their track."""
    near_track: int = 0
    """Within 1.5 G of another track."""


def label_positions(
    tracks: Sequence[Track], cars: Sequence[CarLabel], dpt: float, size: tuple[int, int]
) -> tuple[list[Position], LeftOut]:
    """Every position along ``tracks``, one per gauge, labeled from ``cars``.
    ``size`` is the image's width and height."""
    if not tracks:
        return [], LeftOut()
    lines = [_Line(track) for track in tracks]
    spans = [_span(lines, car) for car in cars]
    # Cars whose ends coincide form a train; an end no other car shares is a train end.
    ends = Counter((p.x, p.y) for car in cars for p in (car.p0, car.p1))
    cx, cy = size[0] / 2, size[1] / 2
    positions: list[Position] = []
    left_out = LeftOut()
    for index, line in enumerate(lines):
        on_track = [(s0, s1) for track, s0, s1 in spans if track == index]
        train_ends = [
            s
            for car, (track, s0, s1) in zip(cars, spans, strict=True)
            if track == index
            for p, s in ((car.p0, s0), (car.p1, s1))
            if ends[(p.x, p.y)] == 1
        ]
        beside = [car for car, (track, _, _) in zip(cars, spans, strict=True) if track != index]
        for s in np.arange(math.floor(line.length / dpt + 1e-9) + 1) * dpt:
            x, y, heading = line.at(float(s))
            others = (other.project((x, y))[1] for other in lines if other is not line)
            if min(others, default=math.inf) <= NEAR_TRACK_G * dpt:
                left_out.near_track += 1
                continue
            gap = min((_outside(s, s0, s1) for s0, s1 in on_track), default=math.inf) / dpt
            if OCCUPIED_G < gap <= CLEAR_G:
                left_out.margin += 1
                continue
            label: Label = "occupied" if gap <= OCCUPIED_G else "clear"
            region: Region = "elsewhere"
            if any(abs(s - end) <= END_G * dpt for end in train_ends):
                region = "end"
            elif label == "occupied":
                region = "inside"
            elif any(_to_span((x, y), car) <= BESIDE_G * dpt for car in beside):
                region = "beside"
            positions.append(
                Position(
                    x=x,
                    y=y,
                    heading=heading,
                    label=label,
                    region=region,
                    center_g=math.hypot(x - cx, y - cy) / dpt,
                )
            )
    return positions, left_out


def _span(lines: Sequence[_Line], car: CarLabel) -> tuple[int, float, float]:
    """The track a car is on, nearest its span midpoint, and the arc lengths of
    ``p0`` and ``p1`` projected onto that track."""
    mid = ((car.p0.x + car.p1.x) / 2, (car.p0.y + car.p1.y) / 2)
    track = min(range(len(lines)), key=lambda i: lines[i].project(mid)[1])
    s0 = lines[track].project((car.p0.x, car.p0.y))[0]
    s1 = lines[track].project((car.p1.x, car.p1.y))[0]
    return track, s0, s1


def _outside(s: float, s0: float, s1: float) -> float:
    """How far ``s`` lies outside the interval between ``s0`` and ``s1``."""
    return max(min(s0, s1) - s, s - max(s0, s1), 0.0)


def _to_span(p: tuple[float, float], car: CarLabel) -> float:
    """The distance from ``p`` to the straight line between a car's span ends."""
    ax, ay, bx, by = car.p0.x, car.p0.y, car.p1.x, car.p1.y
    dx, dy = bx - ax, by - ay
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy or 1.0)
    t = min(max(t, 0.0), 1.0)
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


class _Line:
    """A track's curve as a dense polyline, with arc length along it."""

    def __init__(self, track: Track) -> None:
        points: list[tuple[float, float]] = [(track.nodes[0].x, track.nodes[0].y)]
        for a, b in pairwise(track.nodes):
            seg = segment(a, b)
            n = max(4, math.ceil(math.hypot(b.x - a.x, b.y - a.y) / _STEP_PX))
            for k in range(1, n + 1):
                p = point_at(seg, k / n)
                points.append((p.x, p.y))
        self.points: npt.NDArray[np.float64] = np.array(points)
        steps = np.hypot(*np.diff(self.points, axis=0).T)
        self.s: npt.NDArray[np.float64] = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self.s[-1])

    def at(self, s: float) -> tuple[float, float, float]:
        """The point at arc length ``s``, and the direction the line runs there."""
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2))
        a, b = self.points[i], self.points[i + 1]
        t = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        x, y = a + t * (b - a)
        return float(x), float(y), math.atan2(b[1] - a[1], b[0] - a[0])

    def project(self, p: tuple[float, float]) -> tuple[float, float]:
        """The arc length of the line's point nearest ``p``, and the distance to it."""
        d = np.hypot(self.points[:, 0] - p[0], self.points[:, 1] - p[1])
        i = int(np.argmin(d))
        return float(self.s[i]), float(d[i])
