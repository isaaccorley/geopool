#!/bin/bash
#SBATCH --job-name=probe_pastis_cv
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate

DATASETS=(aef olmoearth tessera)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python scripts/probe_pastis_cv.py \
  --dataset-name "$DS" \
  --embeddings-dir "embeddings_cv/pastis-$DS" \
  --output-csv "results/pastis-$DS/cv_results.csv"
