#!/bin/bash
#SBATCH --job-name=probe_pastis_aef
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

echo "=== KNN probe (FAISS GPU) ==="
python scripts/knnprobe.py \
  --dataset-name pastis-aef \
  --embeddings-dir embeddings/pastis-aef \
  --splits pastis \
  --k-values 1 3 5 7 9

echo "=== Linear probe (GPU L-BFGS) ==="
python scripts/linearprobe.py \
  --dataset-name pastis-aef \
  --embeddings-dir embeddings/pastis-aef \
  --splits pastis
