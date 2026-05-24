"""Analyse how pooling method rankings shift with parcel area.

For each PASTIS embedding source + pooling method:
  - Fit a linear probe on the full train set
  - Evaluate on test parcels stratified by n_pixels bins
  - Report F1-macro per bin and method rank per bin

Usage:
    python scripts/area_breakdown.py \
        --datasets aef tessera olmoearth \
        --output results/area_breakdown.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

BINS = [0, 50, 150, 500, 10_000]
BIN_LABELS = ["tiny\n(<50)", "small\n(50-150)", "medium\n(150-500)", "large\n(>500)"]


def fit_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 200,
    device: torch.device | None = None,
) -> nn.Linear:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    X = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y = torch.from_numpy(y_enc).long().to(device)

    clf = nn.Linear(X.shape[1], n_classes).to(device)
    nn.init.zeros_(clf.weight)
    nn.init.zeros_(clf.bias)

    wd = 1.0 / (C * len(X_train))
    opt = torch.optim.LBFGS(clf.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(clf(X), y) + wd * (clf.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return clf, le


@torch.inference_mode()
def predict(clf: nn.Linear, le: LabelEncoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = clf(Xt)
    preds_enc = logits.argmax(dim=1).cpu().numpy()
    return le.inverse_transform(preds_enc)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["aef", "tessera", "olmoearth"])
    parser.add_argument("--embeddings-root", default="embeddings")
    parser.add_argument("--output", default="results/area_breakdown.csv")
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--skip-methods", nargs="+", default=[], help="Methods to skip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []

    for ds in args.datasets:
        emb_root = Path(args.embeddings_root) / f"pastis-{ds}"
        method_dirs = sorted(emb_root.iterdir())

        for method_dir in method_dirs:
            if method_dir.name in args.skip_methods:
                continue
            npz_path = method_dir / "pastis.npz"
            if not npz_path.exists():
                continue

            data = np.load(npz_path)
            if "n_pixels_test" not in data:
                print(f"  skip {ds}/{method_dir.name} — no n_pixels_test (re-run pool_pastis.py)")
                continue

            X_train = data["x_train"].astype(np.float32)
            y_train = data["y_train"]
            X_test = data["x_test"].astype(np.float32)
            y_test = data["y_test"]
            n_pixels_test = data["n_pixels_test"]

            # Standardise
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # C tuning on 15% stratified val split
            rng = np.random.default_rng(42)
            val_mask = rng.random(len(X_train_sc)) < args.val_fraction
            Xv, yv = X_train_sc[val_mask], y_train[val_mask]
            Xtr, ytr = X_train_sc[~val_mask], y_train[~val_mask]

            best_c, best_f1 = args.c_values[0], -1.0
            for c in args.c_values:
                try:
                    clf, le = fit_linear(Xtr, ytr, C=c, device=device)
                    preds = predict(clf, le, Xv, device)
                    score = f1_macro(yv, preds)
                    if score > best_f1:
                        best_f1, best_c = score, c
                except Exception:
                    pass

            # Refit on full train with best C
            clf, le = fit_linear(X_train_sc, y_train, C=best_c, device=device)

            # Overall
            preds_all = predict(clf, le, X_test_sc, device)
            f1_all = f1_macro(y_test, preds_all)
            rows.append(dict(dataset=ds, method=method_dir.name, bin="overall", f1=f1_all, n=len(y_test), best_c=best_c))
            print(f"  {ds:10s} {method_dir.name:30s} overall F1={f1_all:.4f}  C={best_c}")

            # Per-bin
            for lo, hi, label in zip(BINS[:-1], BINS[1:], BIN_LABELS):
                mask = (n_pixels_test >= lo) & (n_pixels_test < hi)
                if mask.sum() < 10:
                    continue
                preds_bin = predict(clf, le, X_test_sc[mask], device)
                f1_bin = f1_macro(y_test[mask], preds_bin)
                rows.append(dict(dataset=ds, method=method_dir.name, bin=label, f1=f1_bin, n=int(mask.sum()), best_c=best_c))

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Print pivot: method rankings per bin for each dataset
    for ds in args.datasets:
        sub = df[df.dataset == ds]
        print(f"\n=== {ds} — method rank by bin (F1 macro) ===")
        pivot = sub.pivot_table(index="method", columns="bin", values="f1")
        # Rank within each bin (1=best)
        ranked = pivot.rank(ascending=False).astype(int)
        print(ranked.to_string())


if __name__ == "__main__":
    main()
