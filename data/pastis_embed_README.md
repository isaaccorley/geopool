---
license: cc-by-4.0
task_categories:
- image-classification
- image-segmentation
language:
- en
tags:
- remote-sensing
- geospatial
- sentinel-2
- earth-observation
- crop-classification
- embeddings
- pastis
pretty_name: PASTIS Embed
size_categories:
- 10G<n<100G
---

# PASTIS Embed

Pixel-level and pooled patch-level embeddings of the [PASTIS-R](https://github.com/VSainteuf/pastis-benchmark) panoptic segmentation dataset, derived from three geospatial foundation models (GFMs).

## Dataset Description

PASTIS-R contains 2,433 Sentinel-2 time-series patches covering France with instance-level annotations of ~85,000 agricultural parcels across 18 crop classes. This dataset provides pre-computed pixel-level embeddings from three GFMs, plus pooled parcel-level representations using 14 pooling strategies.

### Embedding Sources

| File | Model | Dimension | Format |
|------|-------|-----------|--------|
| `pastis-aef.tar.gz` | [AlphaEarth Foundation (AEF)](https://huggingface.co/microsoft/AlphaEarthFoundation) | 64-d int8 | GeoTIFF per patch |
| `pastis-olmoearth.tar.gz` | [OlmoEarth-Nano](https://huggingface.co/allenai/OlmoEarth-Nano) | 128-d float32 | GeoTIFF per patch |
| `pastis-tessera.tar.gz` | [Tessera](https://huggingface.co/microsoft/tessera) | 128-d float32 | GeoTIFF per patch |

### Pooled Embeddings (14 pooling strategies)

| File | Model | Contents |
|------|-------|----------|
| `embeddings-aef-pooled.tar.gz` | AEF | One `.npz` per pooling method |
| `embeddings-olmoearth-pooled.tar.gz` | OlmoEarth-Nano | One `.npz` per pooling method |
| `embeddings-tessera-pooled.tar.gz` | Tessera | One `.npz` per pooling method |

Each pooled `.npz` contains:
- `x_train`, `x_test`: pooled embeddings (N_parcels × D)
- `y_train`, `y_test`: crop class labels (0–17)
- `n_pixels_train`, `n_pixels_test`: number of pixels per parcel

Pooling methods: `mean`, `std`, `gem`, `signed_non_cancelling_gem`, `center_weighted_mean`, `max`, `mean_std`, `mean_max`, `percentiles`, `median_iqr`, `stats`, `flattened_cov`, `pca_64`, `bovw_128`.

## Dataset Structure

```
pastis-aef/
  S2_10000.tif   # H×W×64 int8 pixel embeddings
  S2_10001.tif
  ...

embeddings-aef-pooled/
  mean/pastis.npz
  stats/pastis.npz
  ...
```

## Usage

```python
import numpy as np

# Load pooled embeddings (e.g. mean pooling with AEF)
data = np.load("embeddings-aef-pooled/mean/pastis.npz")
X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]
n_pixels = data["n_pixels_test"]  # parcel size in pixels

# Fit a linear probe
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
print(clf.score(X_test_scaled, y_test))
```

## Evaluation Protocol

We use the standard PASTIS 5-fold CV protocol: train on folds 1–4, test on fold 5.
Metric: F1-macro over 18 crop classes. Parcels with fewer than 5 pixels are excluded.

## Citation

```bibtex
@article{corley2026geopool,
  title={From Pixels to Patches: Pooling Strategies for Earth Embeddings},
  author={Corley, Isaac and Robinson, Caleb and Becker-Reshef, Inbal and Lavista Ferres, Juan M.},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026}
}
```

See also: [EuroSAT-Embed](https://huggingface.co/datasets/isaaccorley/eurosat-embed)
