"""Measure the CPU that MJPEG decoding takes, from clips rather than a camera.

Each clip plays at each frame rate, in software and through VA-API, then once as fast as it
goes. VA-API runs both with the frames left on the GPU and copied to system memory, which
is what cutting patches on the CPU needs. Appends JSON lines. USB transfer is not included.

Usage: decode.py <clips dir> <results.jsonl>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SECONDS = 20
RATES = {"4k": [10, 15, 20, 30], "1080p": [30, 60]}
VAAPI = ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"]
MODES = {
    "software": [],
    "vaapi-on-gpu": [*VAAPI, "-hwaccel_output_format", "vaapi"],
    "vaapi-copied": VAAPI,
}


def measure(clip: Path, mode: str, fps: int | None) -> dict[str, object]:
    pace = ["-re"] if fps else []
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-benchmark",
        *MODES[mode],
        *pace,
        "-stream_loop",
        "-1",
        "-framerate",
        str(fps or 30),
        "-f",
        "mjpeg",
        "-i",
        str(clip),
        "-t",
        str(SECONDS),
        "-f",
        "null",
        "-",
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=SECONDS * 20)
    row: dict[str, object] = {"clip": clip.stem, "mode": mode, "fps": fps}
    bench = re.search(r"bench: utime=([\d.]+)s stime=([\d.]+)s rtime=([\d.]+)s", done.stderr)
    if done.returncode != 0 or bench is None:
        return {**row, "error": done.stderr[-500:]}
    user, system, wall = map(float, bench.groups())
    frames = re.findall(r"frame=\s*(\d+)", done.stderr)
    row |= {"cores_busy": (user + system) / wall, "wall_s": wall}
    if frames:
        row["fps_reached"] = int(frames[-1]) / wall
    return row


def main() -> None:
    clips, path = Path(sys.argv[1]), Path(sys.argv[2])
    with path.open("a") as out:
        for clip in sorted(clips.glob("*.mjpeg")):
            for mode in MODES:
                for fps in [*RATES[clip.stem.split("-")[0]], None]:
                    row = measure(clip, mode, fps)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(row, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
