#!/usr/bin/env bash
# Download OpenNeuro ds005953 (Hermes et al., CC0) into the parent directory,
# which this repo expects to be the BIDS root.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Fetching ds005953 into $DEST"
if command -v openneuro-py >/dev/null 2>&1; then
  openneuro-py download --dataset=ds005953 --target-dir="$DEST"
else
  echo "openneuro-py not found. Install it with:  uv pip install openneuro-py"
  echo "or download manually from https://openneuro.org/datasets/ds005953"
  exit 1
fi
