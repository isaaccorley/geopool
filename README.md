# 🏊 GeoPool: From Pixels to Patches — Pooling Strategies for Earth Embeddings

> **Accepted** at the [ICLR 2026 ML4RS Workshop](https://ml-for-rs.github.io/iclr2026/) in Rio de Janeiro, Brazil 🇧🇷

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/iclr2026_conference.pdf)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-EuroSAT--Embed-blue)](https://huggingface.co/datasets/isaaccorley/eurosat-embed)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-green)](LICENSE)

Benchmark for evaluating pixel-to-patch pooling methods on geospatial foundation model embeddings.

<p align="center">
  <img src="paper/figures/hero.png" width="400"/>
</p>

As geospatial foundation models shift from patch-level to pixel-level embeddings, practitioners must aggregate thousands of pixel vectors into patch representations. We benchmark **13 pooling methods** across **3 GFMs** (AlphaEarth, OlmoEarth, Tessera) on EuroSAT land-cover classification and release **EuroSAT-Embed**: 81,000 embedding GeoTIFFs for reproducible pooling research. Across the benchmark, simple distributional pooling methods improve spatial-split performance over mean pooling and substantially reduce the random-to-spatial generalization gap.

## 📊 Key Results

**Stats pooling** (min/max/mean/std) is the strongest default: for linear probes it improves average spatial accuracy from **87.3%** to **93.0%** and cuts the random-to-spatial gap from **8.8 pp** to **3.8 pp**. **Covariance pooling** reaches the best accuracy on two of three encoders when higher dimensionality is acceptable, while **mean+max** is another strong low-complexity option with a small spatial generalization gap.

<p align="center">
  <img src="paper/figures/random_vs_spatial.png" width="700"/>
</p>

*Random vs. spatial split accuracy across encoders. Points near the diagonal generalize better under geographic shift.*

## 💡 Recommendation: Mean Baseline, Stats Default

Use **mean** as a low-cost baseline. When a modest increase in dimensionality is acceptable, use **stats pooling** as the default. When maximum accuracy matters more than representation size, use **covariance pooling**. If you want a lighter-weight alternative to covariance, **mean+max** is a strong option.

**NumPy implementations:**

```python
import numpy as np

def stats_pool(x: np.ndarray) -> np.ndarray:
    """Stats pooling over spatial dims. x: (H, W, D) -> (4D,)"""
    mins = x.min(axis=(0, 1))
    maxs = x.max(axis=(0, 1))
    means = x.mean(axis=(0, 1))
    stds = x.std(axis=(0, 1))
    return np.concatenate([mins, maxs, means, stds], axis=-1)


def mean_max_pool(x: np.ndarray) -> np.ndarray:
    """Mean+max pooling over spatial dims. x: (H, W, D) -> (2D,)"""
    means = x.mean(axis=(0, 1))
    maxs = x.max(axis=(0, 1))
    return np.concatenate([means, maxs], axis=-1)


def covariance_pool(x: np.ndarray) -> np.ndarray:
    """Upper-triangular covariance pooling. x: (H, W, D) -> (D(D+1)/2,)"""
    pixels = x.reshape(-1, x.shape[-1])
    centered = pixels - pixels.mean(axis=0, keepdims=True)
    denom = max(pixels.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    tri = np.triu_indices(cov.shape[0])
    return cov[tri]
```

## 🗂️ Pooling Methods

### Training-Free

| Method | Key | Dim | Description |
|---|---|---|---|
| Mean | `mean` | D | Global average pooling |
| Max | `max` | D | Global max pooling |
| Std | `std` | D | Global standard deviation |
| GeM | `gem` | D | Generalized mean pooling (p=3) |
| Center-Weighted | `center_weighted_mean` | D | Gaussian-weighted mean (center focus) |
| Mean+Std | `mean_std` | 2D | Concatenation of mean and std |
| Mean+Max | `mean_max` | 2D | Concatenation of mean and max |
| Median+IQR | `median_iqr` | 2D | Median and interquartile range |
| Stats | `stats` | 4D | min, max, mean, std concatenated |
| Percentiles | `percentiles` | 5D | 10th, 25th, 50th, 75th, 90th percentiles |
| Covariance | `flattened_cov` | D(D+1)/2 | Upper triangle of covariance matrix |

### Parametric (require training data)

| Method | Key | Dim | Description |
|---|---|---|---|
| PCA | `pca_64` | 64 | PCA on mean-pooled embeddings |
| BoVW | `bovw_128` | 128 | Bag of Visual Words (k-means clustering) |

## ⚙️ Setup

```bash
make install
```

For the download/embed workflow, install the extra dependency group too:

```bash
uv sync --dev --group download
```

## 📦 Datasets

The dense pixel embedding variants of EuroSAT and pooled versions are on [HuggingFace](https://huggingface.co/datasets/isaaccorley/eurosat-embed).

```bash
# pooled embeddings
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-aef-pooled.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-olmoearth-nano-pooled.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/embeddings-tessera-pooled.tar.gz

# pixel embeddings
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-aef.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-olmoearth-nano.tar.gz
wget https://hf.co/datasets/isaaccorley/eurosat-embed/resolve/main/eurosat-tessera.tar.gz
```

## 🧪 Evaluation

Once pooled embeddings are downloaded/created in `embeddings/`:

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

## 🔧 (Optional) Generating Pixel Embeddings

All data is available on HuggingFace above. To regenerate from scratch:

### Download EuroSAT and splits

```bash
uv run python data/download_eurosat.py
```

### Create EuroSAT-AEF from Google Earth Engine (requires GEE auth)

```bash
uv run python data/download_eurosat_aef.py
uv run python data/convert_aef.py
```

### Generate EuroSAT-OlmoEarth and EuroSAT-Tessera embeddings

Install the download/embed dependency group first:

```bash
uv sync --dev --group download
```

```bash
uv run python scripts/embed_olmoearth.py --model-size nano
uv run python scripts/embed_tessera.py
```

### Cache pooled embeddings

```bash
uv run python scripts/pool.py --dataset-name aef
uv run python scripts/pool.py --dataset-name tessera
uv run python scripts/pool.py --dataset-name olmoearth-nano
uv run python scripts/pool.py --dataset-name olmoearth-tiny
uv run python scripts/pool.py --dataset-name olmoearth-base
```

If running OOM, use `scripts/pool-stream.py` which streams in batches (slower).

## 🛠️ Development

```bash
make install # install deps
make check  # lint + format + typecheck
make test   # run tests
```

## 📝 Citation

```bibtex
@article{corley2026pixels,
  title={From Pixels to Patches: Pooling Strategies for Earth Embeddings},
  author={Corley, Isaac and Robinson, Caleb and Becker-Reshef, Inbal and Ferres, Juan M Lavista},
  journal={arXiv preprint arXiv:2603.02080},
  year={2026}
}
```
