#!/bin/bash
# Run time_one.py on the packaged model, in the image of the running occupancy-service container,
# so the runtimes and drivers are the service's. One JSON line per setting on stdout.
# Run on the box as a user who can run docker. Usage: service_image.sh [gpu]
set -e
here=$(cd "$(dirname "$0")" && pwd)
img=$(docker inspect -f '{{.Config.Image}}' occupancy-service-1)
dev=()
[ "$1" = gpu ] && dev=(--device /dev/dri)
in_image() { docker run --rm "${dev[@]}" -v "$here:/bench:ro" --entrypoint nice "$img" -n 10 "$@"; }
read -r model height width < <(in_image python -c '
import glob, os, classifier, onnxruntime
model = glob.glob(os.path.join(os.path.dirname(classifier.__file__), "model", "*.onnx"))[0]
_, _, h, w = onnxruntime.InferenceSession(model).get_inputs()[0].shape
print(model, h, w)')

run() { # runtime threads batch precision
  out=$(in_image python /bench/time_one.py --model "$model" --height "$height" --width "$width" \
    --runtime "$1" --threads "$2" --batch "$3" ${4:+--precision "$4"})
  echo "{\"model\": \"$(basename "$model")\", \"runtime\": \"$1\", \"threads\": $2, \"batch\": $3, \"precision\": \"${4:-default}\", ${out#\{}"
}
for t in 1 2 4; do run ov-cpu "$t" 50 f32; done
for t in 1 2 4; do run ort "$t" 50; done
for b in 25 100; do run ov-cpu 4 "$b" f32; done
if [ "$1" = gpu ]; then
  for p in f32 ""; do for b in 25 50 100 150 200; do run ov-gpu 1 "$b" "$p"; done; done
fi
