"""Evaluation utilities for probing experiments."""

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def subsample_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a dataset to a fixed number of samples using stratified sampling.

    Args:
        X: Feature array of shape (N, D).
        y: Label array of shape (N,).
        n_samples: Number of samples to keep. If >= len(y), returns original arrays.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_subset, y_subset).
    """
    if n_samples >= len(y):
        return X, y

    rng = np.random.default_rng(seed)

    # Stratified sampling: sample proportionally from each class
    classes, class_counts = np.unique(y, return_counts=True)
    class_fractions = class_counts / len(y)
    samples_per_class = np.round(class_fractions * n_samples).astype(int)

    # Adjust to hit exactly n_samples (rounding errors)
    diff = n_samples - samples_per_class.sum()
    if diff != 0:
        # Add/remove from largest classes first
        sorted_idx = np.argsort(class_counts)[::-1]
        for i in range(abs(diff)):
            idx = sorted_idx[i % len(sorted_idx)]
            samples_per_class[idx] += 1 if diff > 0 else -1

    # Ensure at least 1 sample per class if possible
    samples_per_class = np.maximum(samples_per_class, 1)
    samples_per_class = np.minimum(samples_per_class, class_counts)

    indices = []
    for cls, n in zip(classes, samples_per_class, strict=False):
        cls_indices = np.where(y == cls)[0]
        selected = rng.choice(cls_indices, size=n, replace=False)
        indices.extend(selected)

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]


def sanitize_features(X: np.ndarray, label: str) -> np.ndarray:
    """Replace non-finite values with 0 and warn if any found."""
    invalid_mask = ~np.isfinite(X)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(f"Warning: {label} has {invalid_count} non-finite values; replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }


def evaluate_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> dict[str, np.ndarray | dict[str, float]]:
    X_train = sanitize_features(X_train, "KNN X_train")
    X_test = sanitize_features(X_test, "KNN X_test")

    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric="cosine", algorithm="brute")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics["k"] = k

    return {"metrics": metrics, "y_pred": y_pred}


def evaluate_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
    C: float | list[float] | None = None,
    cv_folds: int = 3,
) -> dict[str, np.ndarray | dict[str, float]]:
    X_train = sanitize_features(X_train, "Linear X_train")
    X_test = sanitize_features(X_test, "Linear X_test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if C is None:
        C_values = C_GRID
    elif isinstance(C, list):
        C_values = C
    else:
        C_values = [C]

    if len(C_values) > 1:
        clf = LogisticRegressionCV(
            Cs=C_values,
            cv=cv_folds,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
            refit=True,
            use_legacy_attributes=False,
            l1_ratios=(0,),
        )
        clf.fit(X_train_scaled, y_train)
        chosen_c = float(clf.C_)
    else:
        chosen_c = C_values[0]
        clf = LogisticRegression(max_iter=max_iter, C=chosen_c, solver="lbfgs")
        clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    metrics = compute_metrics(y_test, y_pred)
    metrics["C"] = chosen_c
    return {"metrics": metrics, "y_pred": y_pred}


def bootstrap_resample(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample training data with replacement (stratified by class).

    Args:
        X: Feature array of shape (N, D).
        y: Label array of shape (N,).
        rng: NumPy random generator.

    Returns:
        Tuple of (X_resampled, y_resampled) with same size as input.
    """
    classes, class_counts = np.unique(y, return_counts=True)

    indices = []
    for cls, count in zip(classes, class_counts, strict=False):
        cls_indices = np.where(y == cls)[0]
        selected = rng.choice(cls_indices, size=count, replace=True)
        indices.extend(selected)

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]


def aggregate_bootstrap_metrics(
    all_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate bootstrap metrics into mean, std, and 95% CI.

    Args:
        all_metrics: List of metric dictionaries from each bootstrap iteration.

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper for each metric.
    """
    result = {}
    metric_names = [k for k in all_metrics[0] if k != "k"]

    for name in metric_names:
        values = np.array([m[name] for m in all_metrics])
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
        result[f"{name}_ci_lower"] = float(np.percentile(values, 2.5))
        result[f"{name}_ci_upper"] = float(np.percentile(values, 97.5))

    return result


def evaluate_knn_bootstrap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
    n_bootstraps: int = 100,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate KNN with bootstrap resampling of training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        k: Number of neighbors.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with aggregated metrics (mean, std, 95% CI).
    """
    rng = np.random.default_rng(seed)
    all_metrics = []

    X_train = sanitize_features(X_train, "KNN X_train")
    X_test = sanitize_features(X_test, "KNN X_test")

    for _ in range(n_bootstraps):
        X_boot, y_boot = bootstrap_resample(X_train, y_train, rng)
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric="cosine", algorithm="brute")
        clf.fit(X_boot, y_boot)
        y_pred = clf.predict(X_test)
        all_metrics.append(compute_metrics(y_test, y_pred))

    result = aggregate_bootstrap_metrics(all_metrics)
    result["k"] = k
    result["n_bootstraps"] = n_bootstraps
    return result


def evaluate_linear_bootstrap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstraps: int = 100,
    seed: int | None = None,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict[str, float]:
    """Evaluate linear probe with bootstrap resampling of training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        max_iter: Maximum iterations for logistic regression.
        C: Regularization parameter.

    Returns:
        Dictionary with aggregated metrics (mean, std, 95% CI).
    """
    rng = np.random.default_rng(seed)
    all_metrics = []

    X_train = sanitize_features(X_train, "Linear X_train")
    X_test = sanitize_features(X_test, "Linear X_test")

    for _ in range(n_bootstraps):
        X_boot, y_boot = bootstrap_resample(X_train, y_train, rng)

        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
        clf.fit(X_boot_scaled, y_boot)
        y_pred = clf.predict(X_test_scaled)
        all_metrics.append(compute_metrics(y_test, y_pred))

    result = aggregate_bootstrap_metrics(all_metrics)
    result["n_bootstraps"] = n_bootstraps
    return result
