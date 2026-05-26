#!/bin/bash
#SBATCH --job-name=pastis_hf_upload
#SBATCH --partition=cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/hf_staging

source .venv/bin/activate
export PYTHONUNBUFFERED=1

STAGING="data/hf_staging"
NCPU=32  # pigz parallelism

echo "=== PASTIS HuggingFace Upload Job ==="
echo "Started: $(date)"
echo "CPUs: $NCPU"
echo ""

# ---------------------------------------------------------------------------
# 1. Raw pixel embedding tarballs
# ---------------------------------------------------------------------------

pack_raw() {
    local name=$1   # e.g. aef
    local src=$2    # e.g. data/pastis-aef
    local out="$STAGING/pastis-${name}.tar.gz"

    if [[ -f "$out" ]]; then
        echo "  $out already exists ($(du -sh "$out" | cut -f1)), skipping compression."
        return
    fi

    echo "Compressing $src -> $out ..."
    tar -I "pigz -p $NCPU" -cf "$out" -C "$(dirname "$src")" "$(basename "$src")"
    echo "  Done: $(du -sh "$out" | cut -f1)"
}

echo "--- Raw embeddings ---"
pack_raw aef       data/pastis-aef
pack_raw olmoearth data/pastis-olmoearth
pack_raw tessera   data/pastis-tessera

# ---------------------------------------------------------------------------
# 2. Pooled embedding tarballs (one .npz per method, structured dir)
# ---------------------------------------------------------------------------

pack_pooled() {
    local name=$1   # e.g. aef
    local src=$2    # e.g. embeddings/pastis-aef
    local out="$STAGING/embeddings-${name}-pooled.tar.gz"

    if [[ -f "$out" ]]; then
        echo "  $out already exists ($(du -sh "$out" | cut -f1)), skipping compression."
        return
    fi

    echo "Compressing $src -> $out ..."
    tar -I "pigz -p $NCPU" -cf "$out" -C "$(dirname "$src")" "$(basename "$src")"
    echo "  Done: $(du -sh "$out" | cut -f1)"
}

echo ""
echo "--- Pooled embeddings ---"
pack_pooled aef       embeddings/pastis-aef
pack_pooled olmoearth embeddings/pastis-olmoearth
pack_pooled tessera   embeddings/pastis-tessera

# ---------------------------------------------------------------------------
# 3. Upload to HuggingFace
# ---------------------------------------------------------------------------

echo ""
echo "--- Staging sizes ---"
du -sh "$STAGING"/*.tar.gz

echo ""
echo "--- Uploading to HuggingFace ---"
python data/upload_pastis_hf.py

echo ""
echo "Finished: $(date)"
