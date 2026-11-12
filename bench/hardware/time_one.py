"""Time one runtime, model, thread count and batch size. Prints one JSON line.

Input is random, in [0, 1]. A batch of 1 is one call per patch. The rate is the batch size
over the median call time.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from collections.abc import Callable

import numpy as np

WARMUP_RUNS = 5
WARMUP_SECONDS = 2.0
MIN_RUNS = 10
MIN_SECONDS = 3.0


def ort_call(model: str, threads: int, x: np.ndarray) -> Callable[[], object]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    return lambda: session.run(None, {"patches": x})


def openvino_call(
    model: str, device: str, threads: int, precision: str | None, x: np.ndarray
) -> Callable[[], object]:
    import openvino as ov

    core = ov.Core()
    network = core.read_model(model)
    network.reshape(list(x.shape))
    config: dict[str, object] = {"PERFORMANCE_HINT": "LATENCY"}
    if precision is not None:
        config["INFERENCE_PRECISION_HINT"] = precision
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = threads
        config["ENABLE_HYPER_THREADING"] = threads > 6
    request = core.compile_model(network, device, config).create_infer_request()
    return lambda: request.infer({0: x})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["ort", "ov-cpu", "ov-gpu"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--precision", choices=["f32", "f16"], help="OpenVINO only; default: the device's"
    )
    a = parser.parse_args()

    shape = (a.batch, 3, a.height, a.width)
    x = np.random.default_rng(0).random(shape, dtype=np.float32)

    start = time.perf_counter()
    if a.runtime == "ort":
        call = ort_call(a.model, a.threads, x)
    else:
        call = openvino_call(a.model, a.runtime[3:].upper(), a.threads, a.precision, x)
    load_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(WARMUP_RUNS):
        call()
        if time.perf_counter() - start > WARMUP_SECONDS:
            break

    times: list[float] = []
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    while len(times) < MIN_RUNS or time.perf_counter() - start < MIN_SECONDS:
        t = time.perf_counter()
        call()
        times.append(time.perf_counter() - t)
    wall = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)

    median = statistics.median(times)
    cpu = usage.ru_utime + usage.ru_stime - usage_start.ru_utime - usage_start.ru_stime
    print(
        json.dumps(
            {
                "median_ms": median * 1000,
                "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
                "patches_per_s": a.batch / median,
                "runs": len(times),
                "cores_busy": cpu / wall,
                "load_s": load_s,
                "peak_rss_mb": usage.ru_maxrss / 1024,
            }
        )
    )


if __name__ == "__main__":
    main()
