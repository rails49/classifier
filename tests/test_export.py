import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from classifier.export import export

# N scale, 2 px per mm: one gauge (1435 mm / 160) is 17.9375 px.
DPT = 17.9375


def write_archive(path: Path, px_per_mm: float = 2, p1_y: float = 100) -> None:
    jpeg = io.BytesIO()
    Image.new("RGB", (400, 200), "gray").save(jpeg, "JPEG")
    car = {
        "id": "c",
        "class": "stock",
        "p0": {"x": 100, "y": 100},
        "p1": {"x": 300, "y": p1_y},
        "provenance": "human",
    }
    manifest = {
        "version": 4,
        "layout": {
            "scale": "N",
            "calibration": {
                "points": [
                    {"px": {"x": 0, "y": 0}, "world": {"x": 0, "y": 0, "z": 0}},
                    {"px": {"x": 100 * px_per_mm, "y": 0}, "world": {"x": 100, "y": 0, "z": 0}},
                ]
            },
            "tracks": [
                {
                    "nodes": [
                        {"x": 0, "y": 100, "heading": 0},
                        {"x": 400, "y": 100, "heading": 0},
                    ]
                }
            ],
        },
        "camera": {"resolution": {"width": 400, "height": 200}},
        "images": [
            {"filename": "done.jpg", "labeled_complete": True, "labels": [car]},
            {"filename": "open.jpg", "labels": [car]},
        ],
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("done.jpg", jpeg.getvalue())
        z.writestr("open.jpg", jpeg.getvalue())


def test_export_writes_the_index_the_counts_and_the_image_cache(tmp_path: Path):
    archive = tmp_path / "layout.r49"
    write_archive(archive)
    out = tmp_path / "out"

    export([archive], out, [DPT / 2])

    with (out / "index.csv").open() as f:
        rows = list(csv.DictReader(f))
    # Positions every gauge from x 0 to 400: 0, 1, ..., 22 G.
    # The car spans 5.6 to 16.7 G; 19 G is in the band 2-2.5 G from it.
    assert len(rows) == 23 - 1
    assert {(r["archive"], r["image"], r["placement"]) for r in rows} == {
        ("layout", "done.jpg", "0")
    }
    by_x = {round(float(r["x"]) / DPT): r for r in rows}
    assert by_x[11]["label"] == "occupied"
    assert by_x[11]["region"] == "inside"
    assert by_x[0]["label"] == "clear"
    assert float(by_x[0]["y"]) == pytest.approx(100)
    assert float(by_x[0]["heading"]) == pytest.approx(0)
    assert float(by_x[0]["dpt"]) == pytest.approx(DPT)
    assert float(by_x[0]["center_g"]) == pytest.approx(200 / DPT, abs=0.01)
    assert float(by_x[0]["ahead_g"]) == pytest.approx(100 / DPT, abs=0.01)
    assert by_x[0]["behind_g"] == ""
    assert (by_x[11]["ahead_g"], by_x[11]["behind_g"]) == ("0.00", "0.00")

    counts = json.loads((out / "counts.json").read_text())
    assert counts == {
        "layout": {
            "images": 2,
            "not_labeled_complete": 1,
            "positions": 22,
            "occupied": 15,
            "clear": 7,
            "band": 1,
            "outside": 0,
            "near_track": 0,
        }
    }

    with Image.open(out / "dpt8.96875" / "layout" / "done.png") as cached:
        assert cached.size == (200, 100)
    assert not (out / "dpt8.96875" / "layout" / "open.png").exists()


def test_a_cached_image_is_resampled_again_when_its_archive_dpt_changes(tmp_path: Path):
    archive = tmp_path / "layout.r49"
    out = tmp_path / "out"
    write_archive(archive)
    export([archive], out, [DPT / 2])

    write_archive(archive, px_per_mm=4)  # DPT doubles
    export([archive], out, [DPT / 2])

    with Image.open(out / "dpt8.96875" / "layout" / "done.png") as cached:
        assert cached.size == (100, 50)


def test_the_export_stops_on_a_car_end_more_than_0_3_g_from_every_track(tmp_path: Path):
    archive = tmp_path / "layout.r49"
    write_archive(archive, p1_y=110)  # 10 px is 0.56 G off the track at y 100

    with pytest.raises(SystemExit, match=r"layout image 1 done.jpg: car c p1 0.56 G"):
        export([archive], tmp_path / "out", [DPT / 2])
