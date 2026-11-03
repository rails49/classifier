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
"""Inside a train, within ``END_G`` of a train end, clear within ``BESIDE_G`` of a car on
another track, or clear elsewhere."""

OCCUPIED_G = 2.0
"""A position is occupied when a car on its track comes within this many gauges."""
CLEAR_G = 2.5
"""A position is clear when no car on its track comes within this many gauges."""
NEAR_TRACK_G = 1.5
"""A position with another track this close, as at a turnout or a join, is left out."""
NEAR_TRACK_ALONG_G = 3.0
"""So is a position where another track comes within ``NEAR_TRACK_G`` of its track this
many gauges along it, half a patch's length."""
UNDER_CAR_G = 1.5
"""A car across two tracks covers each only where it runs within this many gauges of
the straight line between the car's ends."""
OFF_TRACK_G = 0.3
"""A car end farther than this from every track is a labeling error."""
END_G = 3.0
"""A position this close to a train end, along its track, is in the region ``end``."""
BESIDE_G = 3.0
"""A clear position this close to a car on another track is in the region ``beside``."""

_STEP_PX = 1.0
"""Spacing of the polyline points along a track's curve."""


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
    ahead_g: float | None
    """Distance along the track, in gauges, to the nearest car span on it in the direction
    ``heading`` points; 0 inside a span, ``None`` when no span lies that way."""
    behind_g: float | None
    """As ``ahead_g``, against ``heading``."""


@dataclass
class LeftOut:
    """Positions left out, by reason."""

    outside: int = 0
    """Outside the image."""
    band: int = 0
    """Between ``OCCUPIED_G`` and ``CLEAR_G`` from a car on their track."""
    near_track: int = 0
    """Within ``NEAR_TRACK_ALONG_G``, along their track, of where another track comes within
    ``NEAR_TRACK_G``."""


def label_positions(
    tracks: Sequence[Track], cars: Sequence[CarLabel], dpt: float, size: tuple[int, int]
) -> tuple[list[Position], LeftOut]:
    """Every position along ``tracks``, one per gauge, labeled from ``cars``.
    ``size`` is the image's width and height."""
    if not tracks:
        return [], LeftOut()
    lines = [_Line(track) for track in tracks]
    near = _near_other_tracks(lines, dpt)
    spans = [(car, *span) for car in cars for span in _spans(lines, car, dpt)]
    # Cars whose ends coincide form a train; an end no other car shares is a train end.
    ends = Counter((p.x, p.y) for car in cars for p in (car.p0, car.p1))
    cx, cy = size[0] / 2, size[1] / 2
    positions: list[Position] = []
    left_out = LeftOut()
    for index, line in enumerate(lines):
        on_track = [(s0, s1) for _, track, s0, s1 in spans if track == index]
        train_ends = [
            s
            for car, track, s0, s1 in spans
            if track == index
            for p, s in ((car.p0, s0), (car.p1, s1))
            if ends[(p.x, p.y)] == 1
        ]
        here = {id(car) for car, track, _, _ in spans if track == index}
        beside = [car for car in cars if id(car) not in here]
        for s in np.arange(math.floor(line.length / dpt + 1e-9) + 1) * dpt:
            x, y, heading = line.at(float(s))
            if not (0 <= x < size[0] and 0 <= y < size[1]):
                left_out.outside += 1
                continue
            if np.any(np.abs(near[index] - s) <= NEAR_TRACK_ALONG_G * dpt):
                left_out.near_track += 1
                continue
            gap = min((_outside(s, s0, s1) for s0, s1 in on_track), default=math.inf) / dpt
            if OCCUPIED_G < gap <= CLEAR_G:
                left_out.band += 1
                continue
            label: Label = "occupied" if gap <= OCCUPIED_G else "clear"
            region: Region = "elsewhere"
            if any(abs(s - end) <= END_G * dpt for end in train_ends):
                region = "end"
            elif label == "occupied":
                region = "inside"
            elif any(_to_span((x, y), car) <= BESIDE_G * dpt for car in beside):
                region = "beside"
            spans_ahead = [max(min(s0, s1) - s, 0.0) for s0, s1 in on_track if max(s0, s1) >= s]
            spans_behind = [max(s - max(s0, s1), 0.0) for s0, s1 in on_track if min(s0, s1) <= s]
            positions.append(
                Position(
                    x=x,
                    y=y,
                    heading=heading,
                    label=label,
                    region=region,
                    center_g=math.hypot(x - cx, y - cy) / dpt,
                    ahead_g=min(spans_ahead) / dpt if spans_ahead else None,
                    behind_g=min(spans_behind) / dpt if spans_behind else None,
                )
            )
    return positions, left_out


def off_track_ends(
    tracks: Sequence[Track], cars: Sequence[CarLabel], dpt: float
) -> list[tuple[str, str, float]]:
    """The car id, end (``p0`` or ``p1``) and distance in gauges of each car end
    farther than ``OFF_TRACK_G`` from every track."""
    lines = [_Line(track) for track in tracks]
    off = []
    for car in cars:
        for end, p in (("p0", car.p0), ("p1", car.p1)):
            d = min((line.project((p.x, p.y))[1] for line in lines), default=math.inf) / dpt
            if d > OFF_TRACK_G:
                off.append((car.id, end, d))
    return off


def _near_other_tracks(lines: Sequence[_Line], dpt: float) -> list[npt.NDArray[np.float64]]:
    """For each line, the arc lengths, every quarter gauge, where another line comes
    within ``NEAR_TRACK_G``."""
    sampled = [line.sample(dpt / 4) for line in lines]
    near = []
    for i, (s, points) in enumerate(sampled):
        close = np.zeros(len(s), dtype=bool)
        for j, (_, other) in enumerate(sampled):
            if j != i:
                d = np.hypot(*(points[:, None, :] - other[None, :, :]).transpose(2, 0, 1))
                close |= d.min(axis=1) <= NEAR_TRACK_G * dpt
        near.append(s[close])
    return near


def _spans(lines: Sequence[_Line], car: CarLabel, dpt: float) -> list[tuple[int, float, float]]:
    """The tracks a car is on, those its ends lie on, and on each the arc lengths of
    the points nearest ``p0`` and ``p1``. A car whose ends lie on two tracks, as across
    a turnout or a join, is on both; on each, its span runs from the end there toward
    the other end and stops where the track runs farther than ``UNDER_CAR_G`` from the
    straight line between the ends. The points are sought within 1.5 car lengths of
    the end on that track, so an end nearer another pass of the same track, as on a
    helix, stays on the car's pass."""
    ends = [(car.p0.x, car.p0.y), (car.p1.x, car.p1.y)]
    on = [min(range(len(lines)), key=lambda i: lines[i].project(end)[1]) for end in ends]
    reach = 1.5 * math.hypot(car.p1.x - car.p0.x, car.p1.y - car.p0.y)
    spans = []
    for track in sorted(set(on)):
        line = lines[track]
        mine = on.index(track)
        anchor = line.project(ends[mine])[0]
        within = (anchor - reach, anchor + reach)
        s = [line.project(ends[0], within)[0], line.project(ends[1], within)[0]]
        if len(set(on)) == 2:
            s[1 - mine] = line.leaves(s[mine], s[1 - mine], car, UNDER_CAR_G * dpt)
        spans.append((track, s[0], s[1]))
    return spans


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

    def sample(self, step: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Arc lengths every ``step`` along the line, and the points there."""
        s = np.append(np.arange(0.0, self.length, step), self.length)
        points = np.stack([np.interp(s, self.s, self.points[:, k]) for k in (0, 1)], axis=1)
        return s, points

    def leaves(self, start: float, toward: float, car: CarLabel, d: float) -> float:
        """The arc length, going from ``start`` toward ``toward``, where the line first
        runs farther than ``d`` from the straight line between ``car``'s ends;
        ``toward`` if it never does."""
        step = 1 if toward >= start else -1
        between = np.flatnonzero((self.s - start) * step >= 0)
        between = between[(self.s[between] - toward) * step <= 0][::step]
        a = np.array([car.p0.x, car.p0.y])
        ab = np.array([car.p1.x, car.p1.y]) - a
        t = np.clip((self.points[between] - a) @ ab / (ab @ ab or 1.0), 0.0, 1.0)
        off = np.hypot(*(self.points[between] - a - t[:, None] * ab).T)
        far = np.flatnonzero(off > d)
        return float(self.s[between[far[0]]]) if len(far) else toward

    def project(
        self, p: tuple[float, float], within: tuple[float, float] = (-math.inf, math.inf)
    ) -> tuple[float, float]:
        """The arc length of the line's point nearest ``p`` with arc length
        ``within``, and the distance to it."""
        d = np.hypot(self.points[:, 0] - p[0], self.points[:, 1] - p[1])
        d[(self.s < within[0]) | (self.s > within[1])] = np.inf
        i = int(np.argmin(d))
        return float(self.s[i]), float(d[i])
