"""Aggregate per-seed linear-probe CSVs into seed_variance_summary.csv."""

from pathlib import Path

import numpy as np
import pandas as pd

ENCODERS = ["eurosat-aef", "eurosat-olmoearth", "eurosat-tessera"]
SEEDS = [42, 0, 1]
RESULTS_ROOT = Path("results")


def main() -> None:
    rows = []
    missing = []
    for enc in ENCODERS:
        per_seed = []
        for s in SEEDS:
            p = RESULTS_ROOT / enc / f"linear_results_seed{s}.csv"
            if not p.exists():
                missing.append(str(p))
                continue
            df = pd.read_csv(p)
            df["seed"] = s
            per_seed.append(df)
        if not per_seed:
            print(f"WARN: no CSVs for {enc}")
            continue
        combined = pd.concat(per_seed, ignore_index=True)
        g = combined.groupby(["pool", "split_type"], as_index=False)
        agg = g.agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            n_seeds=("accuracy", "count"),
        )
        agg.insert(0, "encoder", enc)
        rows.append(agg)

    if missing:
        print("Missing CSVs:")
        for m in missing:
            print(f"  {m}")

    out = pd.concat(rows, ignore_index=True)
    out_path = RESULTS_ROOT / "seed_variance_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")

    # Pretty stdout
    out_sorted = out.sort_values(["encoder", "split_type", "acc_mean"], ascending=[True, True, False])
    print(f"{'encoder':22s} {'pool':28s} {'split':10s} {'acc mean±std':16s} {'f1 mean±std':16s} n")
    for _, r in out_sorted.iterrows():
        acc = f"{r.acc_mean*100:.2f}±{(r.acc_std or 0)*100:.2f}"
        f1 = f"{r.f1_mean*100:.2f}±{(r.f1_std or 0)*100:.2f}"
        print(f"{r.encoder:22s} {r.pool:28s} {r.split_type:10s} {acc:16s} {f1:16s} {int(r.n_seeds)}")

    # Flag std > 0.5pp
    flagged = out[(out["acc_std"].fillna(0) * 100) > 0.5]
    if len(flagged):
        print("\n=== std > 0.5pp on accuracy ===")
        for _, r in flagged.iterrows():
            print(f"  {r.encoder:22s} {r.pool:28s} {r.split_type:10s} acc_std={r.acc_std*100:.2f}pp")

    # Sanity vs existing linear_results.csv (seed-42 expected to match)
    print("\n=== Sanity: seed42 vs existing linear_results.csv (acc diff > 0.001 flagged) ===")
    for enc in ENCODERS:
        existing_p = RESULTS_ROOT / enc / "linear_results.csv"
        seed42_p = RESULTS_ROOT / enc / "linear_results_seed42.csv"
        if not (existing_p.exists() and seed42_p.exists()):
            continue
        e = pd.read_csv(existing_p)
        s = pd.read_csv(seed42_p)
        merged = e.merge(s, on=["pool", "split_type"], suffixes=("_old", "_new"))
        merged["diff"] = (merged["accuracy_new"] - merged["accuracy_old"]).abs()
        bad = merged[merged["diff"] > 0.001]
        if len(bad) == 0:
            print(f"  {enc}: all rows match within 0.001")
        else:
            for _, r in bad.iterrows():
                print(f"  {enc} {r.pool} {r.split_type}: old={r.accuracy_old:.4f} new={r.accuracy_new:.4f} diff={r.diff:.4f}")


if __name__ == "__main__":
    main()
