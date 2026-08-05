#!/usr/bin/env bash
# Example: build the Apptainer/Singularity image and run one experiment inside it.
# Set WANDB_API_KEY in your environment first (export WANDB_API_KEY=...), never hard-code it.

# Build the image (pick one)
apptainer build --fakeroot image.sif apptainer1.def
# apptainer build --sandbox /tmp/image_dir/ apptainer1.def && apptainer build image.sif /tmp/image_dir/

# Run a single experiment on GPU
singularity exec \
  --nv \
  --bind $PWD:/workspace \
  --pwd /workspace \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env WANDB_CONSOLE=off \
  --env WANDB_SILENT=true \
  image.sif /app/.venv/bin/python src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/exp/coarse +skip_confirm=true
