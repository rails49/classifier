# The runtimes for docs/research/hardware-v2.md, kept off the server's own system.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    clinfo ffmpeg intel-media-va-driver-non-free intel-opencl-icd ocl-icd-libopencl1 \
    python3 python3-venv vainfo \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir \
    numpy onnxruntime==1.31.1 openvino==2026.4.1
ENV PATH=/venv/bin:$PATH HOME=/tmp
