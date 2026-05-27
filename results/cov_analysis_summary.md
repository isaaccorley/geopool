# PASTIS covariance diagnostics & shrinkage controls

## 1. Covariance conditioning (sample covariance)

Fraction of parcels that are rank-deficient (`rank < D`) and under-sampled (`N < D`):

| Encoder | D | N<D | rank<D | median cond# |
|---|---|---|---|---|
| aef | 64 | 22.6% | 25.4% | 56750.4 |
| olmoearth | 128 | 47.0% | 100.0% | 26.6 |
| tessera | 128 | 47.0% | 76.9% | 73641.3 |

### By parcel-size bin (rank-deficient fraction)

| Encoder | tiny(<50) | small(50-150) | medium(150-500) | large(>500) |
|---|---|---|---|---|
| aef | 100% | 24% | 3% | 3% |
| olmoearth | 100% | 100% | 100% | 100% |
| tessera | 100% | 94% | 55% | 56% |

## 2. Does shrinkage rescue covariance? (linear probe, macro-F1, 5-fold mean)


### aef

| Method | tiny(<50) | small(50-150) | medium(150-500) | large(>500) | overall |
|---|---|---|---|---|---|
| sample_cov | 37.2 | 51.0 | 65.2 | 70.5 | 59.2 |
| ledoit_wolf | 35.8 | 51.0 | 65.2 | 71.5 | 59.2 |
| oas | 36.2 | 51.6 | 65.3 | 71.5 | 59.4 |
| shrink_a0.01 | 36.7 | 51.1 | 65.4 | 70.9 | 59.2 |
| shrink_a0.05 | 36.4 | 50.8 | 65.5 | 71.1 | 59.2 |
| shrink_a0.1 | 36.2 | 51.2 | 65.3 | 70.8 | 59.2 |
| shrink_a0.25 | 36.8 | 51.1 | 65.0 | 70.6 | 59.0 |
| shrink_a0.5 | 37.2 | 51.0 | 65.2 | 70.5 | 59.2 |
| diag_std | 30.5 | 40.7 | 52.5 | 55.6 | 48.0 |

### olmoearth

| Method | tiny(<50) | small(50-150) | medium(150-500) | large(>500) | overall |
|---|---|---|---|---|---|
| sample_cov | 19.1 | 31.9 | 49.1 | 55.1 | 42.3 |
| ledoit_wolf | 18.9 | 31.8 | 49.0 | 55.0 | 42.3 |
| oas | 19.1 | 31.8 | 49.2 | 55.2 | 42.5 |
| shrink_a0.01 | 19.1 | 31.9 | 49.0 | 55.3 | 42.3 |
| shrink_a0.05 | 19.1 | 31.9 | 49.1 | 55.1 | 42.4 |
| shrink_a0.1 | 19.1 | 31.9 | 49.0 | 55.3 | 42.3 |
| shrink_a0.25 | 19.0 | 31.9 | 49.0 | 54.8 | 42.3 |
| shrink_a0.5 | 19.0 | 31.9 | 49.1 | 55.0 | 42.4 |
| diag_std | 9.9 | 16.3 | 26.2 | 29.7 | 22.5 |

### tessera

| Method | tiny(<50) | small(50-150) | medium(150-500) | large(>500) | overall |
|---|---|---|---|---|---|
| sample_cov | 36.3 | 46.6 | 51.1 | 51.3 | 48.5 |
| ledoit_wolf | 36.0 | 46.5 | 51.1 | 51.3 | 48.5 |
| oas | 35.3 | 46.4 | 51.1 | 51.2 | 48.4 |
| shrink_a0.01 | 36.5 | 46.5 | 51.0 | 50.7 | 48.4 |
| shrink_a0.05 | 36.2 | 46.6 | 51.0 | 50.6 | 48.4 |
| shrink_a0.1 | 36.2 | 46.6 | 51.0 | 50.5 | 48.4 |
| shrink_a0.25 | 35.9 | 46.6 | 51.0 | 51.3 | 48.4 |
| shrink_a0.5 | 36.5 | 46.6 | 51.0 | 50.6 | 48.4 |
| diag_std | 29.8 | 40.3 | 44.1 | 43.1 | 41.6 |

## 3. Verdicts

- **aef** tiny parcels: sample_cov=37.2, best shrinkage (shrink_a0.5)=37.2 (Δ=+0.0 pp).
- **olmoearth** tiny parcels: sample_cov=19.1, best shrinkage (shrink_a0.05)=19.1 (Δ=+0.1 pp).
- **tessera** tiny parcels: sample_cov=36.3, best shrinkage (shrink_a0.5)=36.5 (Δ=+0.3 pp).

*Interpretation: if shrinkage closes most of the small-parcel gap, the collapse is an estimation problem (rank-deficiency), consistent with the N/D explanation; if it does not, the collapse reflects information loss beyond conditioning.*
