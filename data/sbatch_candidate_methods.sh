#!/bin/bash
#SBATCH --job-name=candidate_methods
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

METHODS="nd_switch sign_adaptive_nc_gem whitened_mean"

echo "====== Pool PASTIS parcels ======"
for DS in aef olmoearth tessera; do
    echo "--- pastis-$DS ---"
    python scripts/pool_pastis.py \
        --dataset-name pastis-$DS \
        --emb-dir data/pastis-$DS \
        --output-dir embeddings/pastis-$DS \
        --methods $METHODS
done

echo "====== Linear probe ======"
for DS in aef olmoearth tessera; do
    echo "--- pastis-$DS ---"
    python scripts/linearprobe.py \
        --dataset-name pastis-$DS \
        --embeddings-dir embeddings/pastis-$DS \
        --splits pastis \
        --methods $METHODS
done

echo "====== KNN probe (k=5) ======"
for DS in aef olmoearth tessera; do
    echo "--- pastis-$DS ---"
    python scripts/knnprobe.py \
        --dataset-name pastis-$DS \
        --embeddings-dir embeddings/pastis-$DS \
        --splits pastis \
        --k-values 5 \
        --methods $METHODS
done

echo "====== Area breakdown ======"
ALL_METHODS="mean std mean_std stats max gem signed_non_cancelling_gem adaptive_nc_gem mean_max percentiles center_weighted_mean median_iqr pca_64 bovw_128 flattened_cov signed_sqrt_mean trimmed_mean"
python -u scripts/area_breakdown.py \
    --datasets aef olmoearth tessera \
    --embeddings-root embeddings \
    --output results/area_breakdown_candidates.csv \
    --skip-methods $ALL_METHODS \
    --c-values 0.01 0.1 1.0 10.0 100.0

echo "====== Done ======"
