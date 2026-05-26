#!/bin/bash
#SBATCH --job-name=pastis_aef
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

export AWS_NO_SIGN_REQUEST=YES
export AWS_REGION=us-west-2
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export VSI_CACHE=TRUE
export CPL_VSIL_CURL_CACHE_SIZE=500000000

python data/download_pastis_aef.py \
  --tiles data/pastis_tiles.parquet \
  --output data/pastis-aef \
  --year 2019 \
  --num-workers 32
