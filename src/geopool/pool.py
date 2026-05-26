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


def pool_adaptive_nc_gem(
    emb: np.ndarray, p_max: float = 3.0, eps: float = 1e-6
) -> np.ndarray:
    """NC-GeM with power p that scales with effective sample size N.

    When N is small relative to D (few pixels per region), p→1 (≈mean pooling),
    avoiding noisy power-mean amplification. When N≫D, p→p_max (≈standard NC-GeM).
    N_ref is set to the embedding dimension D so the transition is at N/D=1.
    """
    if emb.ndim == 3:
        N = emb.shape[0] * emb.shape[1]
        D = emb.shape[2]
    elif emb.ndim == 4:
        N = emb.shape[1] * emb.shape[2]
        D = emb.shape[3]
    else:
        raise ValueError(f"need 3D or 4D array, got {emb.ndim}D")
    n_ref = float(D)
    p = 1.0 + (p_max - 1.0) * min(1.0, N / n_ref)
    pos = np.maximum(emb, 0.0)
    neg = np.maximum(-emb, 0.0)
    ax = _spatial_axes(emb)
    pos_pooled = np.mean(np.maximum(pos, eps) ** p, axis=ax) ** (1.0 / p)
    neg_pooled = np.mean(np.maximum(neg, eps) ** p, axis=ax) ** (1.0 / p)
    return np.concatenate([pos_pooled, neg_pooled], axis=_concat_axis(emb))


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


def pool_signed_sqrt_mean(emb: np.ndarray) -> np.ndarray:
    """Power-normalized mean: sign(x)*sqrt(|x|) per pixel, then mean. D-dim."""
    ax = _spatial_axes(emb)
    normed = np.sign(emb) * np.sqrt(np.abs(emb))
    return normed.mean(axis=ax)


def pool_trimmed_mean(emb: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """Trimmed mean: drop top/bottom trim fraction per channel, then mean. D-dim."""
    ax = _spatial_axes(emb)
    if emb.ndim == 3:
        H, W, D = emb.shape
        flat = emb.reshape(H * W, D)
        lo = np.percentile(flat, trim * 100, axis=0)
        hi = np.percentile(flat, (1 - trim) * 100, axis=0)
        clipped = np.clip(flat, lo, hi)
        return clipped.mean(axis=0)
    if emb.ndim == 4:
        N, H, W, D = emb.shape
        flat = emb.reshape(N, H * W, D)
        lo = np.percentile(flat, trim * 100, axis=1)
        hi = np.percentile(flat, (1 - trim) * 100, axis=1)
        clipped = np.clip(flat, lo[:, None, :], hi[:, None, :])
        return clipped.mean(axis=1)
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


def pool_nd_switch(emb: np.ndarray) -> np.ndarray:
    """Use mean+std when N<D, signed NC-GeM when N>=D. Always 2D output."""
    if emb.ndim == 3:
        H, W, D = emb.shape
        N = H * W
        if N < D:
            return pool_mean_std(emb)
        return pool_signed_non_cancelling_gem(emb)
    if emb.ndim == 4:
        return np.stack([pool_nd_switch(emb[i]) for i in range(emb.shape[0])])
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


def pool_sign_adaptive_nc_gem(emb: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """Signed NC-GeM when N>=D; GeM+std when N<D. Always 2D output."""
    if emb.ndim == 3:
        H, W, D = emb.shape
        N = H * W
        if N >= D:
            return pool_signed_non_cancelling_gem(emb, p=p, eps=eps)
        flat = emb.reshape(N, D).astype(np.float64)
        gem = np.mean(np.abs(flat) ** p, axis=0) ** (1.0 / p)
        std = flat.std(axis=0)
        return np.concatenate([gem, std]).astype(np.float32)
    if emb.ndim == 4:
        return np.stack([pool_sign_adaptive_nc_gem(emb[i], p=p, eps=eps) for i in range(emb.shape[0])])
    raise ValueError(f"need 3D or 4D, got {emb.ndim}D")


class WhitenedMeanPooler:
    """ZCA-whitened mean pooling.

    Fits IncrementalPCA (whiten=True) on training pixels; for each patch/parcel
    transforms pixels to whitened space then takes the mean. Output: n_components-dim
    (defaults to input D, same dimensionality as plain mean).
    """

    def __init__(self, n_components: int | None = None, batch_size: int = 10_000) -> None:
        from sklearn.decomposition import IncrementalPCA
        self.n_components = n_components
        self.batch_size = batch_size
        self._pca: IncrementalPCA | None = None
        self._fitted = False

    def partial_fit(self, flat: np.ndarray) -> None:
        from sklearn.decomposition import IncrementalPCA
        if self._pca is None:
            nc = self.n_components or flat.shape[1]
            self._pca = IncrementalPCA(n_components=nc, whiten=True)
        nc = self._pca.n_components
        for start in range(0, flat.shape[0], self.batch_size):
            chunk = flat[start : start + self.batch_size].astype(np.float64)
            if chunk.shape[0] < nc:
                continue  # too few samples for this chunk; caller should send larger batches
            self._pca.partial_fit(chunk)

    def transform_parcel(self, px: np.ndarray) -> np.ndarray:
        if not self._fitted or self._pca is None:
            raise RuntimeError("WhitenedMeanPooler must be fit before transform")
        whitened = self._pca.transform(px.astype(np.float64))
        return whitened.mean(axis=0).astype(np.float32)


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


class VLADPooler:
    """Vector of Locally Aggregated Descriptors (Jégou et al., 2010).

    Fits k cluster centers on train pixels; encodes each parcel as the
    sum of per-cluster residuals (pixel - center), intra-L2-normalized,
    then globally L2-normalized. Output dim = k * D.
    """

    def __init__(self, n_clusters: int = 16, random_state: int = 42, batch_size: int = 10_000) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init="auto",
        )
        self._fitted = False

    def _encode_flat(self, flat: np.ndarray) -> np.ndarray:
        """Encode N×D pixel array → k*D VLAD descriptor."""
        flat = flat.astype(np.float32)
        labels = self.kmeans.predict(flat)
        centers = self.kmeans.cluster_centers_.astype(np.float32)
        D = flat.shape[1]
        vlad = np.zeros((self.n_clusters, D), dtype=np.float32)
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                vlad[k] = (flat[mask] - centers[k]).sum(axis=0)
        # intra-normalization
        norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad = vlad / np.where(norms > 0, norms, 1.0)
        # global L2
        out = vlad.ravel()
        n = np.linalg.norm(out)
        return out / max(n, 1e-8)

    def partial_fit(self, flat: np.ndarray) -> None:
        for start in range(0, flat.shape[0], self.batch_size):
            self.kmeans.partial_fit(flat[start : start + self.batch_size])

    def transform_parcel(self, px: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("VLADPooler must be fit before transform")
        return self._encode_flat(px.astype(np.float32))


POOL_METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": pool_mean,
    "std": pool_std,
    "mean_std": pool_mean_std,
    "stats": pool_stats,
    "max": pool_max,
    "gem": pool_gem,
    "signed_non_cancelling_gem": pool_signed_non_cancelling_gem,
    "adaptive_nc_gem": pool_adaptive_nc_gem,
    "mean_max": pool_mean_max,
    "percentiles": pool_percentiles,
    "center_weighted_mean": pool_center_weighted_mean,
    "median_iqr": pool_median_iqr,
    "flattened_cov": pool_flattened_covariance,
    "signed_sqrt_mean": pool_signed_sqrt_mean,
    "trimmed_mean": pool_trimmed_mean,
    "nd_switch": pool_nd_switch,
    "sign_adaptive_nc_gem": pool_sign_adaptive_nc_gem,
}

PCA_VARIANTS = {"pca_64": 64}
BOVW_VARIANTS = {"bovw_128": 128}
VLAD_VARIANTS = {"vlad_16": 16, "vlad_32": 32}
WHITENED_MEAN_VARIANTS: dict[str, None] = {"whitened_mean": None}
FITTED_METHODS = list(PCA_VARIANTS.keys()) + list(BOVW_VARIANTS.keys()) + list(VLAD_VARIANTS.keys()) + list(WHITENED_MEAN_VARIANTS.keys())
ALL_METHODS = list(POOL_METHODS.keys()) + FITTED_METHODS


def get_output_dim(method: str, input_dim: int = 64) -> int:
    if method in {"mean", "std", "max", "gem", "center_weighted_mean", "signed_sqrt_mean", "trimmed_mean", "whitened_mean"}:
        return input_dim
    if method in {"signed_non_cancelling_gem", "adaptive_nc_gem", "nd_switch", "sign_adaptive_nc_gem"}:
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
    if method.startswith("pca_"):
        return int(method.split("_")[1])
    if method.startswith("bovw_"):
        return int(method.split("_")[1])
    if method.startswith("vlad_"):
        return int(method.split("_")[1]) * input_dim
    raise ValueError(f"unknown method: {method}")
