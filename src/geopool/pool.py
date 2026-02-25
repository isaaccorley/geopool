"""Pooling methods for pixel-level embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable


def _spatial_axes(emb: np.ndarray) -> tuple[int, ...]:
    """Return axes to reduce over for spatial pooling."""
    if emb.ndim == 3:
        return (0, 1)
    if emb.ndim == 4:
        return (1, 2)
    raise ValueError(f"need 3D or 4D array, got {emb.ndim}D")


def _concat_axis(emb: np.ndarray) -> int | None:
    """Return axis for concatenation (None for 3D, 1 for 4D)."""
    if emb.ndim == 3:
        return None
    if emb.ndim == 4:
        return 1
    raise ValueError(f"need 3D or 4D array, got {emb.ndim}D")


def pool_mean(emb: np.ndarray) -> np.ndarray:
    """Average over spatial dims."""
    return emb.mean(axis=_spatial_axes(emb))


def pool_std(emb: np.ndarray) -> np.ndarray:
    """Standard deviation over spatial dims."""
    return emb.std(axis=_spatial_axes(emb))


def pool_max(emb: np.ndarray) -> np.ndarray:
    """Max over spatial dims."""
    return emb.max(axis=_spatial_axes(emb))


def pool_mean_std(emb: np.ndarray) -> np.ndarray:
    """Concatenate mean and std."""
    ax = _spatial_axes(emb)
    mean, std = emb.mean(axis=ax), emb.std(axis=ax)
    return np.concatenate([mean, std], axis=_concat_axis(emb))


def pool_stats(emb: np.ndarray) -> np.ndarray:
    """Concatenate min, max, mean, std."""
    ax = _spatial_axes(emb)
    parts = [emb.min(axis=ax), emb.max(axis=ax), emb.mean(axis=ax), emb.std(axis=ax)]
    return np.concatenate(parts, axis=_concat_axis(emb))


def pool_mean_max(emb: np.ndarray) -> np.ndarray:
    """Concatenate mean and max."""
    ax = _spatial_axes(emb)
    return np.concatenate([emb.mean(axis=ax), emb.max(axis=ax)], axis=_concat_axis(emb))


def pool_gem(emb: np.ndarray, p: float = 3.0) -> np.ndarray:
    """Generalized mean pooling. Interpolates between mean (p=1) and max (p->inf)."""
    if p <= 0:
        raise ValueError("p must be positive")
    powered = np.sign(emb) * np.abs(emb) ** p
    pooled = np.mean(powered, axis=_spatial_axes(emb))
    return np.sign(pooled) * np.abs(pooled) ** (1.0 / p)


def pool_signed_non_cancelling_gem(
    emb: np.ndarray, p: float = 3.0, eps: float = 1e-6
) -> np.ndarray:
    """Sign-preserving GeM via positive/negative split without cancellation."""
    if p <= 0:
        raise ValueError("p must be positive")
    pos = np.maximum(emb, 0.0)
    neg = np.maximum(-emb, 0.0)
    ax = _spatial_axes(emb)
    pos_power = np.maximum(pos, eps) ** p
    neg_power = np.maximum(neg, eps) ** p
    pos_pooled = np.mean(pos_power, axis=ax) ** (1.0 / p)
    neg_pooled = np.mean(neg_power, axis=ax) ** (1.0 / p)
    return np.concatenate([pos_pooled, neg_pooled], axis=_concat_axis(emb))


def pool_percentiles(
    emb: np.ndarray, percentiles: tuple[int, ...] = (10, 25, 50, 75, 90)
) -> np.ndarray:
    """Multiple percentiles per dimension. Captures distribution shape."""
    if emb.ndim == 3:
        H, W, D = emb.shape
        flat = emb.reshape(H * W, D)
        pcts = np.percentile(flat, percentiles, axis=0)
        return pcts.flatten()
    if emb.ndim == 4:
        N, H, W, D = emb.shape
        flat = emb.reshape(N, H * W, D)
        pcts = np.percentile(flat, percentiles, axis=1)
        return pcts.transpose(1, 0, 2).reshape(N, -1)
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


def _gaussian_kernel(H: int, W: int, sigma: float | None = None) -> np.ndarray:
    if sigma is None:
        sigma = min(H, W) / 4.0
    y, x = np.ogrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    kernel = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return kernel / kernel.sum()


def pool_center_weighted_mean(emb: np.ndarray) -> np.ndarray:
    """Gaussian-weighted mean, emphasizing patch center."""
    if emb.ndim == 3:
        H, W, _ = emb.shape
        kernel = _gaussian_kernel(H, W)
        return np.einsum("hw,hwd->d", kernel, emb)
    if emb.ndim == 4:
        _, H, W, _ = emb.shape
        kernel = _gaussian_kernel(H, W)
        return np.einsum("hw,nhwd->nd", kernel, emb)
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


def pool_median_iqr(emb: np.ndarray) -> np.ndarray:
    """Median and interquartile range. Robust alternative to mean+std."""
    if emb.ndim == 3:
        H, W, D = emb.shape
        flat = emb.reshape(H * W, D)
        median = np.median(flat, axis=0)
        q75, q25 = np.percentile(flat, [75, 25], axis=0)
        return np.concatenate([median, q75 - q25])
    if emb.ndim == 4:
        N, H, W, D = emb.shape
        flat = emb.reshape(N, H * W, D)
        median = np.median(flat, axis=1)
        q75, q25 = np.percentile(flat, [75, 25], axis=1)
        return np.concatenate([median, q75 - q25], axis=1)
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


def pool_flattened_covariance(emb: np.ndarray) -> np.ndarray:
    """Upper triangle of covariance matrix. Full second-order statistics."""
    if emb.ndim == 3:
        H, W, D = emb.shape
        flat = emb.reshape(H * W, D)
        cov = np.cov(flat, rowvar=False)
        return cov[np.triu_indices(D)]
    if emb.ndim == 4:
        N, H, W, D = emb.shape
        flat = emb.reshape(N, H * W, D)
        n_triu = D * (D + 1) // 2
        result = np.zeros((N, n_triu))
        triu_idx = np.triu_indices(D)
        for i in range(N):
            cov = np.cov(flat[i], rowvar=False)
            result[i] = cov[triu_idx]
        return result
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


class PCAPooler:
    """Flatten patch to vector, reduce with PCA."""

    def __init__(self, n_components: int = 64) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, X: np.ndarray) -> PCAPooler:
        N = X.shape[0]
        self.pca.fit(X.reshape(N, -1))
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCAPooler must be fit before transform")
        return self.pca.transform(X.reshape(X.shape[0], -1))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class BoVWPooler:
    """Bag of Visual Words via k-means clustering of pixels."""

    def __init__(
        self,
        n_clusters: int = 128,
        random_state: int = 42,
        batch_size: int = 10_000,
    ) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init="auto",
        )
        self._fitted = False

    def _partial_fit_flat(self, flat: np.ndarray) -> None:
        for start in range(0, flat.shape[0], self.batch_size):
            end = min(start + self.batch_size, flat.shape[0])
            self.kmeans.partial_fit(flat[start:end])

    def _histogram_from_flat(self, flat: np.ndarray) -> np.ndarray:
        labels = self.kmeans.predict(flat)
        hist, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))
        return hist / max(hist.sum(), 1)

    def _histograms_for_batch(self, embeddings: list[np.ndarray]) -> np.ndarray:
        counts = [emb.shape[0] * emb.shape[1] for emb in embeddings]
        if not counts:
            return np.empty((0, self.n_clusters), dtype=np.float32)

        flats = [
            emb.reshape(-1, emb.shape[-1]).astype(np.float32, copy=False) for emb in embeddings
        ]
        all_pixels = np.concatenate(flats, axis=0)
        labels = self.kmeans.predict(all_pixels)

        result = np.empty((len(embeddings), self.n_clusters), dtype=np.float32)
        offsets = np.cumsum([0] + counts)
        for i in range(len(embeddings)):
            start, end = offsets[i], offsets[i + 1]
            hist = np.bincount(labels[start:end], minlength=self.n_clusters)
            result[i] = hist / max(hist.sum(), 1)
        return result

    def fit(
        self,
        X: np.ndarray | list[np.ndarray],
        random_state: int = 42,
        shuffle_pixels: bool = True,
        image_batch_size: int = 32,
    ) -> BoVWPooler:
        if isinstance(X, list):
            rng = np.random.default_rng(random_state)
            for start in range(0, len(X), image_batch_size):
                batch = X[start : start + image_batch_size]
                flats = [
                    emb.reshape(-1, emb.shape[-1]).astype(np.float32, copy=False) for emb in batch
                ]
                flat = np.concatenate(flats, axis=0)
                if shuffle_pixels and flat.shape[0] > 1:
                    flat = flat[rng.permutation(flat.shape[0])]
                self._partial_fit_flat(flat)
        else:
            N, _, _, D = X.shape
            for i in range(N):
                flat = X[i].reshape(-1, D).astype(np.float32, copy=False)
                self._partial_fit_flat(flat)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BoVWPooler must be fit before transform")
        N, _, _, D = X.shape
        flat = X.reshape(N, -1, D)
        result = np.zeros((N, self.n_clusters))
        for i in range(N):
            labels = self.kmeans.predict(flat[i])
            hist, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))
            result[i] = hist / hist.sum()
        return result

    def transform_embeddings(
        self,
        embeddings: list[np.ndarray],
        desc: str | None = None,
        image_batch_size: int = 64,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BoVWPooler must be fit before transform")
        result = np.empty((len(embeddings), self.n_clusters), dtype=np.float32)
        iterator = range(0, len(embeddings), image_batch_size)
        if desc:
            iterator = tqdm(iterator, desc=desc)
        for start in iterator:
            end = min(start + image_batch_size, len(embeddings))
            result[start:end] = self._histograms_for_batch(embeddings[start:end])
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


POOL_METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": pool_mean,
    "std": pool_std,
    "mean_std": pool_mean_std,
    "stats": pool_stats,
    "max": pool_max,
    "gem": pool_gem,
    "signed_non_cancelling_gem": pool_signed_non_cancelling_gem,
    "mean_max": pool_mean_max,
    "percentiles": pool_percentiles,
    "center_weighted_mean": pool_center_weighted_mean,
    "median_iqr": pool_median_iqr,
    "flattened_cov": pool_flattened_covariance,
}

PCA_VARIANTS = {"pca_64": 64}
BOVW_VARIANTS = {"bovw_128": 128}
FITTED_METHODS = list(PCA_VARIANTS.keys()) + list(BOVW_VARIANTS.keys())
ALL_METHODS = list(POOL_METHODS.keys()) + FITTED_METHODS


def get_output_dim(method: str, input_dim: int = 64) -> int:
    if method in {"mean", "std", "max", "gem", "center_weighted_mean"}:
        return input_dim
    if method == "signed_non_cancelling_gem":
        return 2 * input_dim
    if method == "mean_std":
        return 2 * input_dim
    if method == "stats":
        return 4 * input_dim
    if method == "mean_max":
        return 2 * input_dim
    if method == "percentiles":
        return 5 * input_dim
    if method == "median_iqr":
        return 2 * input_dim
    if method == "flattened_cov":
        return input_dim * (input_dim + 1) // 2
    if method.startswith(("pca_", "bovw_")):
        return int(method.split("_")[1])
    raise ValueError(f"unknown method: {method}")
