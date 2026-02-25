import numpy as np
import pytest
from sklearn.datasets import make_classification

from geopool.pool import (
    BOVW_VARIANTS,
    PCA_VARIANTS,
    POOL_METHODS,
    BoVWPooler,
    PCAPooler,
    get_output_dim,
    pool_center_weighted_mean,
    pool_flattened_covariance,
    pool_gem,
    pool_max,
    pool_mean,
    pool_mean_max,
    pool_mean_std,
    pool_median_iqr,
    pool_percentiles,
    pool_stats,
    pool_std,
)


@pytest.fixture
def emb_3d():
    np.random.seed(42)
    return np.random.randn(64, 64, 64).astype(np.float32)


@pytest.fixture
def emb_4d():
    np.random.seed(42)
    return np.random.randn(10, 64, 64, 64).astype(np.float32)


@pytest.mark.parametrize(
    ("pool_fn", "expected_3d", "expected_4d"),
    [
        (pool_mean, (64,), (10, 64)),
        (pool_std, (64,), (10, 64)),
        (pool_max, (64,), (10, 64)),
        (pool_gem, (64,), (10, 64)),
        (pool_center_weighted_mean, (64,), (10, 64)),
        (pool_mean_std, (128,), (10, 128)),
        (pool_mean_max, (128,), (10, 128)),
        (pool_median_iqr, (128,), (10, 128)),
        (pool_stats, (256,), (10, 256)),
        (pool_percentiles, (320,), (10, 320)),
        (pool_flattened_covariance, (2080,), (10, 2080)),
    ],
)
def test_pool_shapes(emb_3d, emb_4d, pool_fn, expected_3d, expected_4d):
    assert pool_fn(emb_3d).shape == expected_3d
    assert pool_fn(emb_4d).shape == expected_4d


def test_pca_pooler():
    X, _ = make_classification(n_samples=100, n_features=64, n_informative=32, random_state=42)
    data = X.reshape(100, 4, 4, 4).astype(np.float32)
    train, test = data[:70], data[70:]

    pca = PCAPooler(n_components=32)
    assert pca.fit_transform(train).shape == (70, 32)
    assert pca.transform(test).shape == (30, 32)


def test_bovw_pooler():
    X, _ = make_classification(
        n_samples=50 * 64, n_features=16, n_informative=10, n_clusters_per_class=4, random_state=42
    )
    data = X.reshape(50, 8, 8, 16).astype(np.float32)
    train, test = data[:35], data[35:]

    bovw = BoVWPooler(n_clusters=16)
    bovw.fit(train)
    train_out = bovw.transform(train)
    test_out = bovw.transform(test)

    assert train_out.shape == (35, 16)
    assert test_out.shape == (15, 16)
    assert np.allclose(train_out.sum(axis=1), 1.0)
    assert np.allclose(test_out.sum(axis=1), 1.0)


def test_bovw_not_fitted():
    bovw = BoVWPooler(n_clusters=16)
    with pytest.raises(RuntimeError, match="must be fit"):
        bovw.transform(np.zeros((5, 8, 8, 16)))


def test_registry_has_all_methods():
    expected = [
        "mean",
        "std",
        "mean_std",
        "stats",
        "max",
        "gem",
        "mean_max",
        "percentiles",
        "center_weighted_mean",
        "median_iqr",
        "flattened_cov",
    ]
    for m in expected:
        assert m in POOL_METHODS
    assert "pca_64" in PCA_VARIANTS
    assert "bovw_128" in BOVW_VARIANTS


@pytest.mark.parametrize(
    ("method", "dim"),
    [
        ("mean", 64),
        ("std", 64),
        ("max", 64),
        ("gem", 64),
        ("center_weighted_mean", 64),
        ("mean_std", 128),
        ("mean_max", 128),
        ("median_iqr", 128),
        ("stats", 256),
        ("percentiles", 320),
        ("flattened_cov", 2080),
        ("pca_64", 64),
        ("bovw_128", 128),
    ],
)
def test_output_dims(method, dim):
    assert get_output_dim(method) == dim


def test_output_dim_custom_input():
    assert get_output_dim("mean", input_dim=512) == 512
    assert get_output_dim("percentiles", input_dim=512) == 2560
    assert get_output_dim("flattened_cov", input_dim=512) == 512 * 513 // 2


def test_output_dim_unknown():
    with pytest.raises(ValueError, match="unknown"):
        get_output_dim("nonexistent")


def test_zero_input():
    pool_mean(np.zeros((64, 64, 64)))


def test_constant_input():
    const = np.ones((64, 64, 64)) * 5.0
    assert np.allclose(pool_mean(const), 5.0)


def test_gem_rejects_bad_p(emb_3d):
    with pytest.raises(ValueError, match="positive"):
        pool_gem(emb_3d, p=0)


def test_center_weighted_nonzero_at_center():
    emb = np.zeros((64, 64, 64))
    emb[32, 32, :] = 1.0
    result = pool_center_weighted_mean(emb)
    assert np.any(result != 0)
