#!/usr/bin/env bash
set -eu
TRAINING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SMALLDOG_ASSET_DIR="$TRAINING_ROOT/v67_cts/assets/smalldog_v67"
export PYTHONPATH="$TRAINING_ROOT/runtime/source/rsl_rl:$TRAINING_ROOT/runtime/source/robot_lab${PYTHONPATH:+:$PYTHONPATH}"
cd "$TRAINING_ROOT/runtime"
exec "${SMALLDOG_PYTHON:-python}" "$@"
