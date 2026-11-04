"""Run time_one.py over the throughput grid of HW.md, appending to a JSON-lines file.

The order is shuffled, so heating over the run does not line up with any one setting. A
reference configuration repeats through the run to show drift. Each configuration is its own
process. Rerunning skips what the file already holds.

Usage: grid.py <models dir> <results.jsonl>
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SIZES = {"dpt12": (36, 72), "dpt16": (48, 96), "dpt20": (60, 120), "2x": (48, 96)}
THREADS = [1, 2, 4, 6, 12]
BATCHES = [1, 25, 50, 100, 150, 200]
FORMATS = ["float", "int8"]
REFERENCE = {"runtime": "ort", "size": "dpt16", "format": "float", "threads": 6, "batch": 50}
REFERENCE_EVERY = 25
TIMEOUT_S = 300


def configurations() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for size in SIZES:
        for fmt in FORMATS:
            for batch in BATCHES:
                configs.append(
                    {"runtime": "ov-gpu", "size": size, "format": fmt, "threads": 0, "batch": batch}
                )
                for runtime in ["ort", "ov-cpu"]:
                    for threads in THREADS:
                        configs.append(
                            {
                                "runtime": runtime,
                                "size": size,
                                "format": fmt,
                                "threads": threads,
                                "batch": batch,
                            }
                        )
    random.Random(49).shuffle(configs)
    return configs


def key(config: dict[str, object]) -> tuple[object, ...]:
    return tuple(config[k] for k in ("runtime", "size", "format", "threads", "batch"))


def run(config: dict[str, object], models: Path) -> dict[str, object]:
    height, width = SIZES[str(config["size"])]
    command = [
        sys.executable,
        str(HERE / "time_one.py"),
        f"--runtime={config['runtime']}",
        f"--model={models / f'{config["size"]}-{config["format"]}.onnx'}",
        f"--threads={config['threads']}",
        f"--batch={config['batch']}",
        f"--height={height}",
        f"--width={width}",
    ]
    started = time.time()
    result: dict[str, object] = {**config, "started": started}
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {**result, "error": f"timeout {TIMEOUT_S} s"}
    if done.returncode != 0:
        return {**result, "error": done.stderr[-500:] or f"exit {done.returncode}"}
    return {**result, **json.loads(done.stdout.splitlines()[-1]), "ended": time.time()}


def main() -> None:
    models, path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    seen = {key(row) for row in rows if not row.get("reference")}
    todo = [c for c in configurations() if key(c) not in seen]
    print(f"{len(todo)} configurations to run", flush=True)
    with path.open("a") as out:
        for i, config in enumerate(todo):
            if i % REFERENCE_EVERY == 0:
                row = {**run(REFERENCE, models), "reference": True}
                out.write(json.dumps(row) + "\n")
                out.flush()
            row = run(config, models)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(i + 1, row.get("patches_per_s", row.get("error")), config, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
