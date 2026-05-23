#!/bin/bash
#SBATCH --job-name=adaptive_nc_gem
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

METHOD=adaptive_nc_gem

echo "====== Linear probe ======"
for DS in aef olmoearth tessera; do
    echo "--- pastis-$DS ---"
    python scripts/linearprobe.py \
        --dataset-name pastis-$DS \
        --embeddings-dir embeddings/pastis-$DS \
        --splits pastis \
        --methods $METHOD
done

echo "====== KNN probe (k=5) ======"
for DS in aef olmoearth tessera; do
    echo "--- pastis-$DS ---"
    python scripts/knnprobe.py \
        --dataset-name pastis-$DS \
        --embeddings-dir embeddings/pastis-$DS \
        --splits pastis \
        --k-values 5 \
        --methods $METHOD
done

echo "====== Area breakdown ======"
# Run area_breakdown only for adaptive_nc_gem by passing all other methods as --skip-methods
ALL_METHODS="mean std mean_std stats max gem signed_non_cancelling_gem mean_max percentiles center_weighted_mean median_iqr pca_64 bovw_128"
python -u scripts/area_breakdown.py \
    --datasets aef olmoearth tessera \
    --embeddings-root embeddings \
    --output results/area_breakdown_adaptive.csv \
    --skip-methods $ALL_METHODS \
    --c-values 0.01 0.1 1.0 10.0 100.0

echo "====== Done ======"
