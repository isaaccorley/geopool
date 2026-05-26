#!/bin/bash
#SBATCH --job-name=embed_olmoearth_pastis
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1

python scripts/embed_olmoearth_pastis.py \
  --data-dir data/pastis-r/PASTIS-R \
  --meta data/pastis-r/PASTIS-R/metadata.geojson \
  --output data/pastis-olmoearth \
  --model-size nano \
  --n-timesteps 12 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda
