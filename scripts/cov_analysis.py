"""Covariance diagnostics + shrinkage-covariance controls for PASTIS pooling.

Single extraction pass per encoder, then:
  1. Per-parcel covariance diagnostics (rank, eigenvalues, condition number).
  2. Shrinkage feature variants pooled per parcel:
       - sample covariance (upper triangle)
       - Ledoit-Wolf covariance (upper triangle)
       - OAS covariance (upper triangle)
       - diagonal / std-only
       - shrinkage grid: (1-a)*S + a*diag(S), a in {0.01,0.05,0.1,0.25,0.5}
  3. 5-fold CV linear + kNN probes (paper protocol), macro-F1 overall and by
     parcel-size bin.

Writes (per encoder, merged later by merge_cov_analysis.py):
  results/cov/<ds>_cov_diagnostics.csv
  results/cov/<ds>_shrinkage_results.csv

Run on a SLURM compute node (see data/sbatch_cov_analysis.sh) — never the login node.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.covariance import OAS, LedoitWolf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for scripts.*

from geopool.evaluation import evaluate_knn, evaluate_linear
from scripts.pool_pastis import VALID_CLASSES

EPS = 1e-6
SHRINK_ALPHAS = [0.01, 0.05, 0.1, 0.25, 0.5]
BIN_EDGES = [(0, 50, "tiny(<50)"), (50, 150, "small(50-150)"),
             (150, 500, "medium(150-500)"), (500, 10**9, "large(>500)")]


def bin_of(n: int) -> str:
    for lo, hi, name in BIN_EDGES:
        if lo <= n < hi:
            return name
    return "large(>500)"


# ── Extraction (returns parcel identity for diagnostics) ──────────────────────
def extract_patch(args: tuple) -> tuple[int, list]:
    patch_id, emb_dir, ann_dir, min_pixels = args
    try:
        with rasterio.open(Path(emb_dir) / f"S2_{patch_id}.tif") as src:
            emb = src.read(out_dtype="float32")  # (D,H,W)
        emb = np.moveaxis(emb, 0, -1)  # (H,W,D)
        D = emb.shape[-1]
        pid = np.load(Path(ann_dir) / f"ParcelIDs_{patch_id}.npy").ravel()
        sem = np.load(Path(ann_dir) / f"TARGET_{patch_id}.npy")[0].ravel()
        flat = emb.reshape(-1, D)
        out = []
        for parcel_id in np.unique(pid):
            mask = pid == parcel_id
            classes = np.unique(sem[mask])
            cls = int(classes[0]) if len(classes) == 1 else int(np.bincount(sem[mask].astype(np.intp)).argmax())
            if cls not in VALID_CLASSES:
                continue
            px = flat[mask]
            if len(px) < min_pixels:
                continue
            out.append((int(parcel_id), cls, px.astype(np.float32)))
        return patch_id, out
    except Exception as e:  # noqa: BLE001 - worker boundary; surface and continue
        print(f"  patch {patch_id} FAILED: {e}")
        return patch_id, []


def load_all(emb_dir, ann_dir, fold_by_patch, min_pixels, workers):
    ids = sorted(fold_by_patch)
    wargs = [(p, str(emb_dir), str(ann_dir), min_pixels) for p in ids]
    res: dict[int, list] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_patch, a): a[0] for a in wargs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="extract"):
            pid, parcels = f.result()
            res[pid] = parcels
    parcels = []
    for pid in ids:
        fold = fold_by_patch[pid]
        for parcel_id, cls, px in res[pid]:
            parcels.append((pid, parcel_id, cls, px, fold))
    return parcels


# ── Per-parcel covariance computation ─────────────────────────────────────────
def parcel_cov_features(px: np.ndarray, triu, diag_pos, D: int):
    """Return (diag_row, sample_tri, lw_tri, oas_tri) and diagnostics dict."""
    n = len(px)
    T = D * (D + 1) // 2
    diag = px.std(axis=0).astype(np.float32)  # std-only feature

    if n < 2:
        zeros = np.zeros(T, dtype=np.float32)
        diag_info = dict(cov_rank=0, rank_over_D=0.0, eig_min_nonzero=np.nan,
                         eig_max=np.nan, cond_number=np.nan)
        return diag, zeros, zeros.copy(), zeros.copy(), diag_info

    S = np.cov(px, rowvar=False).astype(np.float64)  # (D,D)
    eig = np.linalg.eigvalsh(S)  # ascending, real
    eig = np.clip(eig, 0.0, None)  # kill tiny negative numerical noise
    nonzero = eig[eig > EPS]
    rank = int(nonzero.size)
    eig_max = float(eig[-1])
    eig_min_nz = float(nonzero.min()) if rank else np.nan
    cond = eig_max / max(eig_min_nz, EPS) if rank else np.nan
    diag_info = dict(cov_rank=rank, rank_over_D=rank / D,
                     eig_min_nonzero=eig_min_nz, eig_max=eig_max, cond_number=cond)

    sample_tri = S[triu].astype(np.float32)
    lw_tri = LedoitWolf(assume_centered=False).fit(px).covariance_[triu].astype(np.float32)
    oas_tri = OAS(assume_centered=False).fit(px).covariance_[triu].astype(np.float32)
    return diag, sample_tri, lw_tri, oas_tri, diag_info


def shrink_from_sample(sample_tri: np.ndarray, diag_pos: np.ndarray, alpha: float) -> np.ndarray:
    """(1-a)*S + a*diag(S) on the flattened upper triangle."""
    out = (1.0 - alpha) * sample_tri
    out[:, diag_pos] = sample_tri[:, diag_pos]  # diagonal restored to full variance
    return out


# ── Probe: 5-fold CV, macro-F1 overall + per bin ──────────────────────────────
def cv_probe(X, y, fold, n_pixels, method, ds, use_gpu):
    rows = []
    bins = np.array([bin_of(int(n)) for n in n_pixels])
    linear_fn = None
    if use_gpu:
        try:
            from geopool.evaluation import evaluate_linear_gpu
            linear_fn = evaluate_linear_gpu
        except Exception:  # noqa: BLE001
            linear_fn = evaluate_linear
    else:
        linear_fn = evaluate_linear

    for f in sorted(set(fold.tolist())):
        te = fold == f
        tr = ~te
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        bins_te = bins[te]
        for probe, fn in [("linear", linear_fn), ("knn", lambda a, b, c, d: evaluate_knn(a, b, c, d, k=5))]:
            res = fn(Xtr, ytr, Xte, yte)
            yp = np.asarray(res["y_pred"])
            from sklearn.metrics import f1_score
            rows.append(dict(dataset=ds, method=method, probe=probe, test_fold=int(f),
                             bin="overall", f1_macro=float(f1_score(yte, yp, average="macro", zero_division=0)),
                             n_test=int(te.sum())))
            for _, _, bname in BIN_EDGES:
                m = bins_te == bname
                if m.sum() == 0:
                    continue
                rows.append(dict(dataset=ds, method=method, probe=probe, test_fold=int(f),
                                 bin=bname, f1_macro=float(f1_score(yte[m], yp[m], average="macro", zero_division=0)),
                                 n_test=int(m.sum())))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--emb-dir", default=None)
    p.add_argument("--ann-dir", default="data/pastis-r/PASTIS-R/ANNOTATIONS")
    p.add_argument("--meta", default="data/pastis-r/PASTIS-R/metadata.geojson")
    p.add_argument("--output-dir", default="results/cov")
    p.add_argument("--min-pixels", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: cap #parcels (after extraction).")
    p.add_argument("--max-patches", type=int, default=None, help="Smoke test: only read this many patches (caps extraction cost).")
    p.add_argument("--no-probe", action="store_true", help="Diagnostics only.")
    p.add_argument("--cpu", action="store_true", help="Force CPU linear probe.")
    p.add_argument("--methods", nargs="+", default=None, help="Restrict probed methods (smoke tests).")
    args = p.parse_args()

    ds = args.dataset_name
    emb_dir = Path(args.emb_dir) if args.emb_dir else Path(f"data/pastis-{ds}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = False
    if not args.cpu:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            use_gpu = False
    print(f"[{ds}] linear probe: {'GPU' if use_gpu else 'CPU'}")

    with open(args.meta) as f:
        meta = json.load(f)
    fold_by_patch = {ft["properties"]["ID_PATCH"]: int(ft["properties"]["Fold"]) for ft in meta["features"]}
    if args.max_patches:
        ids = sorted(fold_by_patch)
        stride = max(1, len(ids) // args.max_patches)  # spread across folds
        keep = ids[::stride][: args.max_patches]
        fold_by_patch = {k: fold_by_patch[k] for k in keep}

    print(f"[{ds}] extracting parcels...")
    parcels = load_all(emb_dir, args.ann_dir, fold_by_patch, args.min_pixels, args.num_workers)
    if args.limit:
        parcels = parcels[: args.limit]
    print(f"[{ds}] {len(parcels)} parcels")

    D = parcels[0][3].shape[1]
    triu = np.triu_indices(D)
    # positions in the flattened upper triangle that are diagonal entries (row==col)
    diag_pos = np.where(triu[0] == triu[1])[0]

    diag_rows, sample_list, lw_list, oas_list = [], [], [], []
    diag_records = []
    y, fold, npix = [], [], []
    for pid, parcel_id, cls, px, fld in tqdm(parcels, desc=f"{ds} cov"):
        diag, s_tri, lw_tri, oas_tri, dinfo = parcel_cov_features(px, triu, diag_pos, D)
        diag_rows.append(diag)
        sample_list.append(s_tri)
        lw_list.append(lw_tri)
        oas_list.append(oas_tri)
        y.append(cls)
        fold.append(fld)
        npix.append(len(px))
        diag_records.append(dict(dataset=ds, parcel_id=f"{pid}_{parcel_id}", fold=fld,
                                 **{"class": cls}, n_pixels=len(px), D=D,
                                 N_over_D=len(px) / D, **dinfo))

    y = np.array(y, dtype=np.int32)
    fold = np.array(fold, dtype=np.int8)
    npix = np.array(npix, dtype=np.int32)

    diag_df = pd.DataFrame(diag_records)
    diag_csv = out_dir / f"{ds}_cov_diagnostics.csv"
    diag_df.to_csv(diag_csv, index=False)
    print(f"[{ds}] wrote {diag_csv}  ({len(diag_df)} parcels)")

    # quick console summary
    print(f"[{ds}] N<D: {(diag_df.N_over_D < 1).mean():.1%} | rank<D: {(diag_df.rank_over_D < 1).mean():.1%}")
    for _, _, bname in BIN_EDGES:
        sub = diag_df[[bin_of(int(n)) == bname for n in diag_df.n_pixels]]
        if len(sub):
            print(f"    {bname:16s} N<D={ (sub.N_over_D<1).mean():.0%}  rank<D={ (sub.rank_over_D<1).mean():.0%}  (n={len(sub)})")

    if args.no_probe:
        return

    sample_arr = np.stack(sample_list)
    lw_arr = np.stack(lw_list)
    oas_arr = np.stack(oas_list)
    diag_arr = np.stack(diag_rows)
    del sample_list, lw_list, oas_list, diag_rows

    methods = {
        "sample_cov": sample_arr,
        "ledoit_wolf": lw_arr,
        "oas": oas_arr,
        "diag_std": diag_arr,
    }
    for a in SHRINK_ALPHAS:
        methods[f"shrink_a{a}"] = shrink_from_sample(sample_arr, diag_pos, a)

    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    all_rows = []
    for name, X in methods.items():
        print(f"[{ds}] probe {name}  (dim={X.shape[1]})")
        all_rows.extend(cv_probe(X, y, fold, npix, name, ds, use_gpu))

    res_df = pd.DataFrame(all_rows)
    res_csv = out_dir / f"{ds}_shrinkage_results.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"[{ds}] wrote {res_csv}  ({len(res_df)} rows)")


if __name__ == "__main__":
    main()
