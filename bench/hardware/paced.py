"""Run batches at a fixed rate and log call times, temperature and clock each second.

A call that overruns its frame drops the frames it covered, as a live camera would. A rate of 0
runs flat out. Writes one JSON line per second.

Usage: paced.py --runtime ort --model m.onnx --threads 6 --fps 10 --seconds 300 --out log.jsonl
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
from time_one import openvino_call, ort_call

BATCH = 50
CLOCKS = list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))


def coretemp() -> Path:
    for name in Path("/sys/class/hwmon").glob("*/name"):
        if name.read_text().strip() == "coretemp":
            return name.parent / "temp1_input"
    raise RuntimeError("no coretemp sensor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["ort", "ov-cpu", "ov-gpu"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    x = np.random.default_rng(0).random((BATCH, 3, 48, 96), dtype=np.float32)
    if a.runtime == "ort":
        call = ort_call(a.model, a.threads, x)
    else:
        call = openvino_call(a.model, a.runtime[3:].upper(), a.threads, x)
    for _ in range(5):
        call()

    temp = coretemp()
    period = 1 / a.fps if a.fps else 0.0
    start = next_frame = time.perf_counter()
    second_end = start + 1
    times: list[float] = []
    missed = 0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    busy = usage.ru_utime + usage.ru_stime
    with open(a.out, "a") as out:
        while (now := time.perf_counter()) - start < a.seconds:
            if period:
                if now < next_frame:
                    time.sleep(next_frame - now)
                next_frame += period
            t = time.perf_counter()
            call()
            done = time.perf_counter()
            times.append(done - t)
            if period and done > next_frame:
                skipped = int((done - next_frame) / period) + 1
                missed += skipped
                next_frame += skipped * period
            if done >= second_end:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                row = {
                    "t": round(done - start, 2),
                    "runtime": a.runtime,
                    "fps": a.fps,
                    "calls": len(times),
                    "missed": missed,
                    "median_ms": round(statistics.median(times) * 1000, 2),
                    "ms": [round(v * 1000, 1) for v in times],
                    "cores_busy": round(usage.ru_utime + usage.ru_stime - busy, 2),
                    "temp_c": int(temp.read_text()) // 1000,
                    "clock_mhz": sum(int(c.read_text()) for c in CLOCKS) // len(CLOCKS) // 1000,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                busy = usage.ru_utime + usage.ru_stime
                times, missed = [], 0
                second_end = done + 1


if __name__ == "__main__":
    main()
