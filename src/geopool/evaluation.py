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


def evaluate_knn_faiss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
) -> dict[str, np.ndarray | dict[str, float]]:
    """KNN via FAISS with cosine similarity (inner-product on L2-normalised vectors).

    Uses GPU index when `use_gpu=True` and a GPU is available, falls back to CPU.
    """
    import faiss

    X_train = sanitize_features(X_train, "KNN X_train").astype(np.float32, copy=False)
    X_test = sanitize_features(X_test, "KNN X_test").astype(np.float32, copy=False)

    # L2-normalise for cosine via inner product
    faiss.normalize_L2(X_train)
    faiss.normalize_L2(X_test)

    D = X_train.shape[1]
    index_cpu = faiss.IndexFlatIP(D)  # inner product == cosine after L2-norm

    n_gpus = faiss.get_num_gpus()
    if use_gpu and n_gpus > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu

    index.add(X_train)
    _, indices = index.search(X_test, k)  # (N_test, k)

    # Majority vote
    neighbor_labels = y_train[indices]  # (N_test, k)
    y_pred = np.array([
        np.bincount(row, minlength=int(y_train.max()) + 1).argmax()
        for row in neighbor_labels
    ])

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


def evaluate_linear_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_grid: list[float] | None = None,
    max_iter: int = 200,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray | dict[str, float]]:
    """GPU logistic regression via PyTorch L-BFGS with C grid search.

    Standardises features, holds out `val_fraction` of train to pick C, then
    refits on the full train set with the chosen C.  Falls back to CPU sklearn
    when CUDA is unavailable.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    if C_grid is None:
        C_grid = C_GRID

    if not torch.cuda.is_available():
        return evaluate_linear(X_train, y_train, X_test, y_test, C=C_grid)

    X_train = sanitize_features(X_train, "Linear X_train").astype(np.float32)
    X_test = sanitize_features(X_test, "Linear X_test").astype(np.float32)

    # Standardise on full train
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    # Remap labels to contiguous 0-indexed ints
    classes = np.unique(y_train)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    y_tr_full = np.vectorize(label_to_idx.get)(y_train).astype(np.int64)
    y_te = np.vectorize(label_to_idx.get)(y_test).astype(np.int64)
    n_classes = len(classes)
    D = X_train_s.shape[1]

    dev = torch.device("cuda")

    # Stratified val split for C selection
    rng = np.random.default_rng(seed)
    n_val = int(len(X_train_s) * val_fraction)
    perm = rng.permutation(len(X_train_s))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr_np, y_tr_np = X_train_s[tr_idx], y_tr_full[tr_idx]
    X_val_np, y_val_np = X_train_s[val_idx], y_tr_full[val_idx]

    def _fit(X_np: np.ndarray, y_np: np.ndarray, C: float) -> nn.Linear:
        X_t = torch.from_numpy(X_np).to(dev)
        y_t = torch.from_numpy(y_np).to(dev)
        model = nn.Linear(D, n_classes, bias=True).to(dev)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        # L2 reg: weight_decay = 1/(C * N) matches sklearn's C convention
        wd = 1.0 / (C * len(X_np))
        opt = torch.optim.LBFGS(
            model.parameters(), max_iter=max_iter,
            tolerance_grad=1e-6, line_search_fn="strong_wolfe",
        )
        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.cross_entropy(model(X_t), y_t)
            l2 = sum(p.pow(2).sum() for p in model.parameters())
            (loss + wd * l2).backward()
            return loss
        opt.step(closure)
        return model

    def _predict(model: nn.Linear, X_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(X_np).to(dev)).argmax(1).cpu().numpy()

    # Grid search C on val set
    best_C, best_val_acc = C_grid[0], -1.0
    for C in C_grid:
        model = _fit(X_tr_np, y_tr_np, C)
        val_acc = (_predict(model, X_val_np) == y_val_np).mean()
        if val_acc > best_val_acc:
            best_val_acc, best_C = val_acc, C

    # Refit on full train with best C
    model = _fit(X_train_s, y_tr_full, best_C)
    y_pred_idx = _predict(model, X_test_s)
    y_pred = np.vectorize(idx_to_label.get)(y_pred_idx)

    metrics = compute_metrics(y_test, y_pred)
    metrics["C"] = best_C
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
