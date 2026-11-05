#!/usr/bin/env bash
# The wheel as it ships: builds it into <out>, installs it alone with no extras
# in a clean environment, and scores a frame with it from outside the repository,
# so `import classifier` can only be the wheel. The gate runs it on a temporary
# directory; the release workflow publishes what it leaves in <out>.
#
#   bin/check-wheel.sh <out>
set -euo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd)
cd "$(dirname "$0")/.."
repo=$(pwd)
python=$(cat .python-version)

rm -f "$out"/*.whl
uv build --quiet --wheel --out-dir "$out"
wheel=$(echo "$out"/*.whl)
cd "$out"
uv run --quiet --isolated --no-project --python "$python" --with "$wheel" -- \
  python -I "$repo/tests/installed_wheel.py"
