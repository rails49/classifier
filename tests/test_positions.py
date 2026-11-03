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
    assert left_out.band == 2


def test_positions_within_1_5_g_of_another_track_are_left_out():
    # B runs 1 G beside A from x 150, as at a turnout.
    a = straight(100, 150, 200, 150)
    b = straight(150, 160, 200, 160)

    positions, left_out = label_positions([a, b], [], G, SIZE)

    # B comes within 1.5 G of A from x 139; a patch at x 110 reaches 3 G along to x 140.
    assert [round(p.x) for p in positions] == [100]
    assert left_out.near_track == 10 + 6


def test_regions_inside_a_train_near_its_ends_beside_an_occupied_track_and_elsewhere():
    a = straight(0, 150, 390, 150)
    b = straight(250, 178, 390, 178)
    # Two cars on A share an end, so they form one train from x 107 to 193.
    # One car stands on B, 2.8 G beside A, from x 300 to 340.
    cars = [
        car("1", 107, 150, 150, 150),
        car("2", 150, 150, 193, 150),
        car("3", 300, 178, 340, 178),
    ]

    positions, _ = label_positions([a, b], cars, G, SIZE)

    on_a = {round(p.x): p.region for p in positions if p.y == pytest.approx(150)}
    expected = {x: "elsewhere" for x in range(0, 391, 10)}
    # x 80 and 220 are clear, 2.7 G from the train, and within 3 G of its ends.
    expected |= {x: "end" for x in [*range(80, 131, 10), *range(170, 221, 10)]}
    expected |= {x: "inside" for x in range(140, 161, 10)}
    # x 290 and 350 are 2.97 G from the car on B.
    expected |= {x: "beside" for x in range(290, 351, 10)}
    assert on_a == expected


def test_positions_outside_the_image_are_left_out():
    positions, left_out = label_positions([straight(-30, 150, 50, 150)], [], G, SIZE)

    assert [round(p.x) for p in positions] == [0, 10, 20, 30, 40, 50]
    assert left_out.outside == 3


def test_along_track_distance_to_the_nearest_car_span_ahead_and_behind():
    # The car spans x 133..167; the track runs toward +x, so ahead is +x.
    positions, _ = label_positions(
        [straight(100, 150, 200, 150)], [car("a", 133, 153, 167, 153)], G, SIZE
    )

    by_x = {round(p.x): (p.ahead_g, p.behind_g) for p in positions}
    assert by_x[100] == (pytest.approx(3.3, abs=0.01), None)
    assert by_x[120] == (pytest.approx(1.3, abs=0.01), None)
    assert by_x[150] == (0, 0)
    assert by_x[200] == (None, pytest.approx(3.3, abs=0.01))


def test_no_car_on_the_track_leaves_both_distances_blank():
    a = straight(100, 150, 200, 150)
    b = straight(100, 250, 200, 250)

    positions, _ = label_positions([a, b], [car("b", 133, 250, 167, 250)], G, SIZE)

    assert all(p.ahead_g is None and p.behind_g is None for p in positions if p.y == 150)


def test_a_car_end_nearer_another_pass_of_its_own_track_stays_on_the_car_s_pass():
    # A hairpin: the track runs right along y 100, turns, and runs back along y 110,
    # as a helix does in the image. The car stands on the y 110 pass; its right end
    # is labeled nearer the y 100 pass.
    hairpin = Track(
        nodes=[
            TrackNode(x=50, y=100, heading=0),
            TrackNode(x=350, y=100, heading=0),
            TrackNode(x=350, y=110, heading=math.pi),
            TrackNode(x=50, y=110, heading=math.pi),
        ]
    )

    positions, _ = label_positions([hairpin], [car("a", 100, 110, 180, 104)], G, SIZE)

    occupied = {(round(p.x), round(p.y)) for p in positions if p.label == "occupied"}
    assert all(y > 105 for _, y in occupied)
    assert min(x for x, _ in occupied) <= 100 and max(x for x, _ in occupied) >= 180


def test_positions_whose_patch_reaches_a_turnout_are_left_out():
    # B leaves A at x 150 and diverges; it is within 1.5 G of A from x 135 to 212.
    # A patch reaches 3 G along the track, so positions on A from x 105 to 242 show it.
    a = straight(0, 150, 390, 150)
    b = straight(150, 150, 390, 210)

    positions, _ = label_positions([a, b], [], G, SIZE)

    on_a = {round(p.x) for p in positions if p.y == pytest.approx(150)}
    assert {100, 250} <= on_a
    assert not on_a & {110, 240}


def test_a_car_across_a_turnout_covers_a_track_only_while_the_track_stays_under_it():
    # B leaves A at x 100 and climbs 1 in 4. The car stands from x 60 on A to x 220 on
    # B. A runs more than 1.5 G from the car's straight line beyond x 141.
    a = straight(0, 150, 390, 150)
    b = straight(100, 150, 390, 222.5)

    positions, _ = label_positions([a, b], [car("a", 60, 150, 220, 180)], G, SIZE)

    on_a = {round(p.x): p.label for p in positions if p.y == pytest.approx(150)}
    on_b = {round(p.x): p.label for p in positions if p.y > 150}
    assert {x: on_a[x] for x in (200, 210, 220)} == dict.fromkeys((200, 210, 220), "clear")
    assert any(label == "occupied" for x, label in on_b.items() if 195 < x < 225)


def test_a_car_whose_ends_lie_on_two_tracks_covers_both():
    # A and B meet end to end at x 200; the car stands across the join from x 150 to 250.
    a = straight(0, 150, 200, 150)
    b = straight(200, 150, 390, 150)

    positions, _ = label_positions([a, b], [car("a", 150, 150, 250, 150)], G, SIZE)

    labels = {round(p.x): p.label for p in positions if p.x > 245}
    # x 270 is 2 G beyond the car's end; x 280 is 3 G.
    assert labels[250] == labels[260] == labels[270] == "occupied"
    assert labels[280] == "clear"
