#!/bin/bash
#SBATCH --job-name=pool_pastis_cv
#SBATCH --partition=cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate

DATASETS=(aef olmoearth tessera)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python scripts/pool_pastis_cv.py \
  --dataset-name "$DS" \
  --emb-dir "data/pastis-$DS" \
  --ann-dir data/pastis-r/PASTIS-R/ANNOTATIONS \
  --meta data/pastis-r/PASTIS-R/metadata.geojson \
  --output-dir "embeddings_cv/pastis-$DS" \
  --min-pixels 5 \
  --num-workers 16
