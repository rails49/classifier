import math

import pytest
from r49 import CarLabel, Point, Track, TrackNode

from classifier.positions import label_positions

G = 10.0
SIZE = (400, 300)


def straight(x0: float, y0: float, x1: float, y1: float) -> Track:
    heading = math.atan2(y1 - y0, x1 - x0)
    return Track(
        nodes=[TrackNode(x=x0, y=y0, heading=heading), TrackNode(x=x1, y=y1, heading=heading)]
    )


def car(id: str, x0: float, y0: float, x1: float, y1: float) -> CarLabel:
    return CarLabel.model_validate(
        {
            "id": id,
            "class": "stock",
            "p0": Point(x=x0, y=y0),
            "p1": Point(x=x1, y=y1),
            "provenance": "human",
        }
    )


def test_a_position_every_gauge_along_a_track_with_its_direction_and_distance_from_center():
    positions, _ = label_positions([straight(100, 150, 200, 150)], [], G, SIZE)

    assert [p.x for p in positions] == pytest.approx(
        [100, 110, 120, 130, 140, 150] + [160, 170, 180, 190, 200]
    )
    assert all(p.y == pytest.approx(150) for p in positions)
    assert all(p.heading == pytest.approx(0) for p in positions)
    # The image center is (200, 150): the first position is 10 G from it.
    assert positions[0].center_g == pytest.approx(10)
    assert all(p.label == "clear" and p.region == "elsewhere" for p in positions)


def test_occupied_within_2_g_of_a_car_clear_beyond_2_5_g_and_the_band_between_left_out():
    # The car spans x 133..167, projected onto the track at y 150.
    positions, left_out = label_positions(
        [straight(100, 150, 200, 150)], [car("a", 133, 153, 167, 153)], G, SIZE
    )

    labels = {round(p.x): p.label for p in positions}
    assert labels == {
        100: "clear",
        120: "occupied",
        130: "occupied",
        140: "occupied",
        150: "occupied",
        160: "occupied",
        170: "occupied",
        180: "occupied",
        200: "clear",
    }
    assert left_out.margin == 2


def test_positions_within_1_5_g_of_another_track_are_left_out():
    # B runs 1 G beside A from x 150, as at a turnout.
    a = straight(100, 150, 200, 150)
    b = straight(150, 160, 200, 160)

    positions, left_out = label_positions([a, b], [], G, SIZE)

    # A's position at x 140 is 1.41 G from B's first node; x 130 is 2.24 G.
    assert [round(p.x) for p in positions] == [100, 110, 120, 130]
    assert left_out.near_track == 7 + 6


def test_regions_inside_a_train_near_its_ends_beside_an_occupied_track_and_elsewhere():
    a = straight(0, 150, 400, 150)
    b = straight(250, 180, 390, 180)
    # Two cars on A share an end, so they form one train from x 100 to 200.
    # One car stands on B, 3 G beside A, from x 300 to 340.
    cars = [
        car("1", 100, 150, 150, 150),
        car("2", 150, 150, 200, 150),
        car("3", 300, 180, 340, 180),
    ]

    positions, _ = label_positions([a, b], cars, G, SIZE)

    on_a = {round(p.x): p.region for p in positions if p.y == pytest.approx(150)}
    expected = {x: "elsewhere" for x in range(0, 401, 10)}
    expected |= {x: "end" for x in [*range(70, 131, 10), *range(170, 231, 10)]}
    expected |= {x: "inside" for x in range(140, 161, 10)}
    expected |= {x: "beside" for x in range(300, 341, 10)}
    assert on_a == expected
