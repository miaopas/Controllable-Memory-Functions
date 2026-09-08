#!/bin/bash
# Run a suite as Ray tasks: all GPUs of one machine, or a whole Ray cluster.
# Needs:  pip install -e ".[ray]"
set -euo pipefail
MODEL=${MODEL:-examples.my_model:ResidualGRU}
SUITE=${SUITE:-standard}

# One machine: four small trials share each GPU.
cmf run --model "${MODEL}" --suite "${SUITE}" --backend ray --gpus-per-trial 0.25

# Existing Ray cluster (the repository must be installed on every node, e.g. on a shared filesystem):
#   cmf run --model "${MODEL}" --suite "${SUITE}" --backend ray --ray-address auto --gpus-per-trial 0.25

# CPU-only machine or cluster: one trial per CPU core.
#   cmf run --model "${MODEL}" --suite "${SUITE}" --backend ray --cpus-per-trial 1
