#!/bin/bash
#SBATCH --job-name=probe_seed
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate

# Args: $1 = dataset-name (results dir), $2 = embeddings-dir, $3 = seed
DATASET="$1"
EMBDIR="$2"
SEED="$3"

echo "=== Linear probe seed=$SEED dataset=$DATASET ==="
python scripts/linearprobe.py \
  --dataset-name "$DATASET" \
  --embeddings-dir "$EMBDIR" \
  --seed "$SEED" \
  --output-csv "results/${DATASET}/linear_results_seed${SEED}.csv"
