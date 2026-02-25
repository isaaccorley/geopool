# geopool

Benchmark for evaluating pixel-to-patch pooling methods on geospatial foundation model embeddings.

## Overview

This repository accompanies the paper: From `From Pixels to Patches: Pooling Strategies for Earth Embeddings`. We evaluate 13 pooling methods across 3 GFMs (AlphaEarth, OlmoEarth-Nano, Tessera) using the EuroSAT land-cover classification task.

## Setup

### Install dependencies

```bash
uv sync --dev
```

## Datasets

The dense pixel embedding variants of EuroSAT and their pooled versions for each pooling strategy can be accessed on HuggingFace [here](https://huggingface.co/datasets/isaaccorley/eurosat-embed).

```bash
# pooled embeddings
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-aef-pooled.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-olmoearth-nano-pooled.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-olmoearth-tiny-pooled.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-olmoearth-base-pooled.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-tessera-pooled.tar.gz

# pixel embeddings
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-aef.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-olmoearth-nano.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-olmoearth-tiny.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-olmoearth-base.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-tessera.tar.gz
```

## Evaluation

Once the pooled embeddings are downloaded or created and stored in the `embeddings/` folder, run the following for KNN and Linear probing evaluation.

### Run kNN and linear probes

```bash
uv run python scripts/knnprobe.py --dataset-name aef
uv run python scripts/linearprobe.py --dataset-name aef
```

### Generate paper tables

```bash
uv run python scripts/knn_table.py --output paper/knn_table.tex
uv run python scripts/linear_table.py --output paper/linear_table.tex
```

### Plot results

```bash
uv run python scripts/plot_results.py
```


## (Optional) Generating Pixel Embeddings

All data is made available on HuggingFace above, however, if you want to regenerate the pixel and pooled embeddings, run the following:

### Download EuroSAT and splits

```bash
uv run python data/download_eurosat.py
```

### Create EuroSAT-AEF from Google Earth Engine (requires GEE authentication)

```bash
uv run python data/download_eurosat_aef.py
uv run python data/convert_aef.py
```

### Generate EuroSAT-OlmoEarth and EuroSAT-Tessera embeddings (optional)

Optional: Install the olmoearth_pretrain package (needed to generate olmoearth pixel embeddings)

```bash
uv pip install 'olmoearth_pretrain @ git+https://github.com/allenai/olmoearth_pretrain.git'
```

```bash
uv run python data/embed_olmoearth.py --model-size nano
uv run python data/embed_tessera.py
```

### Cache pooled embeddings

```bash
uv run python scripts/pool.py --dataset-name aef
uv run python scripts/pool.py --dataset-name tessera
uv run python scripts/pool.py --dataset-name olmoearth-nano
uv run python scripts/pool.py --dataset-name olmoearth-tiny
uv run python scripts/pool.py --dataset-name olmoearth-base
```

If you are running out of memory (OOM) then you can alternatively use the `scripts/pool-stream.py` scripts which will stream in batches albeit much slower.

## Pooling Methods

This section describes the pooling methods implemented in `src/geopool/pool.py`. Each method transforms a spatial embedding tensor of shape $(H, W, D)$ into a fixed-length patch descriptor.

### Simple Pooling Methods

| Method          | Key                    | Output Dim | Description                              |
| --------------- | ---------------------- | ---------- | ---------------------------------------- |
| Mean            | `mean`                 | $D$        | Global average pooling                   |
| Max             | `max`                  | $D$        | Global max pooling                       |
| Std             | `std`                  | $D$        | Global standard deviation                |
| GeM             | `gem`                  | $D$        | Generalized mean pooling ($p=3$)         |
| Center-Weighted | `center_weighted_mean` | $D$        | Gaussian-weighted mean (center focus)    |
| Mean+Std        | `mean_std`             | $2D$       | Concatenation of mean and std            |
| Mean+Max        | `mean_max`             | $2D$       | Concatenation of mean and max            |
| Median+IQR      | `median_iqr`           | $2D$       | Median and interquartile range           |
| Stats           | `stats`                | $4D$       | min, max, mean, std concatenated         |
| Percentiles     | `percentiles`          | $5D$       | 10th, 25th, 50th, 75th, 90th percentiles |
| Covariance      | `flattened_cov`        | $D(D+1)/2$ | Upper triangle of covariance matrix      |

### Fitted Methods (require training data)

| Method | Key        | Output Dim | Description                              |
| ------ | ---------- | ---------- | ---------------------------------------- |
| PCA    | `pca_64`   | 64         | PCA on mean-pooled embeddings            |
| BoVW   | `bovw_128` | 128        | Bag of Visual Words (k-means clustering) |

### Output Dimensions (for $D=64$)

| Method                                    | Output Dim |
| ----------------------------------------- | ---------- |
| mean, std, max, gem, center_weighted_mean | 64         |
| mean_std, mean_max, median_iqr            | 128        |
| stats                                     | 256        |
| percentiles                               | 320        |
| flattened_cov                             | 2080       |
| pca_64                                    | 64         |
| bovw_128                                  | 128        |

## Development

### Lint and format

```bash
uv run ruff format && uv run ruff check --fix --unsafe-fixes && uv run ty check
```

### Run tests

```bash
uv run pytest -v
```
