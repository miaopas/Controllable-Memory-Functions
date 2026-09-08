#!/bin/bash
# Split a cmf suite over a Slurm array job; every array task runs 1/N of the trials.
#
#   sbatch examples/cluster/slurm_array.sh
#   MODEL=lstm SUITE=quick sbatch --array=0-3 examples/cluster/slurm_array.sh
#
# All tasks write into the same run directory (runs/<model>_<suite>/results.shard-I-of-N.csv).
# Re-submitting resumes: finished trials are skipped. When all tasks are done:
#
#   cmf report runs/ResidualGRU_standard --baselines all --out reports/ResidualGRU.html
#
#SBATCH --job-name=cmf
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cmf_%A_%a.out

set -euo pipefail
MODEL=${MODEL:-examples.my_model:ResidualGRU}
SUITE=${SUITE:-standard}
N=${SLURM_ARRAY_TASK_COUNT:?run this as an array job}

cd "${SLURM_SUBMIT_DIR}"
# Activate your environment here, e.g.:  source .venv/bin/activate
mkdir -p logs
cmf run --model "${MODEL}" --suite "${SUITE}" --device auto \
    --shard "${SLURM_ARRAY_TASK_ID}/${N}"
