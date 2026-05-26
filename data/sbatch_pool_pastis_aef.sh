#!/bin/bash
#SBATCH --job-name=pool_pastis_aef
#SBATCH --partition=cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate

python scripts/pool_pastis.py \
  --dataset-name aef \
  --emb-dir data/pastis-aef \
  --ann-dir data/pastis-r/PASTIS-R/ANNOTATIONS \
  --meta data/pastis-r/PASTIS-R/metadata.geojson \
  --output-dir embeddings/pastis-aef \
  --test-fold 5 \
  --min-pixels 5 \
  --num-workers 16 \
  --bovw-batch-size 50000 \
  --pca-batch-size 512 \
  --overwrite
