#!/bin/bash
#SBATCH --job-name=area_breakdown
#SBATCH --partition=cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate

export PYTHONUNBUFFERED=1

python -u scripts/area_breakdown.py \
  --datasets aef tessera olmoearth \
  --embeddings-root embeddings \
  --output results/area_breakdown.csv \
  --c-values 0.01 0.1 1.0 10.0 100.0
