#!/bin/bash
#SBATCH --job-name=pastis_tessera
#SBATCH --partition=cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate

python data/download_pastis_tessera.py \
  --tiles data/pastis_tiles.parquet \
  --output data/pastis-tessera \
  --year 2019 \
  --num-workers 4
