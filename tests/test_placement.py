import json
from collections.abc import Sequence

from r49 import Manifest, parse_manifest

from classifier.placement import placements

DPT = 10.0
CORNERS = [(0, 0), (1000, 0), (1000, 500), (0, 500)]


def manifest(
    images: Sequence[tuple[str, Sequence[tuple[float, float]]]], shift=(0.0, 0.0), complete=True
):
    """An archive whose images hold one car per given midpoint, seen by a camera
    moved so that everything appears shifted by ``shift``."""
    dx, dy = shift

    def point(x: float, y: float) -> dict[str, float]:
        return {"x": x + dx, "y": y + dy}

    return parse_manifest(
        json.dumps(
            {
                "version": 4,
                "layout": {
                    "scale": "N",
                    "calibration": {
                        "points": [
                            {"px": point(x, y), "world": {"x": x, "y": y, "z": 0}}
                            for x, y in CORNERS
                        ]
                    },
                },
                "camera": {"resolution": {"width": 1000, "height": 500}},
                "images": [
                    {
                        "filename": name,
                        "labeled_complete": complete,
                        "labels": [
                            {
                                "id": str(i),
                                "class": "stock",
                                "p0": point(x - 50, y),
                                "p1": point(x + 50, y),
                                "provenance": "human",
                            }
                            for i, (x, y) in enumerate(cars)
                        ],
                    }
                    for name, cars in images
                ],
            }
        ).encode()
    )


def test_images_whose_cars_stand_in_the_same_places_share_a_placement():
    a = manifest(
        [
            ("1.jpg", [(100, 100), (300, 200)]),
            ("2.jpg", [(105, 98), (300, 204)]),  # the same, labeled again
            ("3.jpg", [(100, 100), (600, 200)]),  # one car moved
        ]
    )

    groups = placements([("a", a, DPT)])

    assert groups[("a", "1.jpg")] == groups[("a", "2.jpg")]
    assert groups[("a", "3.jpg")] != groups[("a", "1.jpg")]


def test_a_placement_spans_archives_whose_camera_moved():
    cars = [(100, 100), (300, 200)]
    a = manifest([("1.jpg", cars)])
    b = manifest([("2.jpg", cars), ("3.jpg", [(100, 100)])], shift=(80, -60))

    groups = placements([("a", a, DPT), ("b", b, DPT)])

    assert groups[("a", "1.jpg")] == groups[("b", "2.jpg")]
    assert groups[("b", "3.jpg")] != groups[("a", "1.jpg")]


def test_only_labeled_complete_images_are_placed():
    m: Manifest = manifest([("1.jpg", [(100, 100)])], complete=False)

    assert placements([("a", m, DPT)]) == {}
