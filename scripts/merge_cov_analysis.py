"""Merge per-encoder covariance-analysis outputs and write a markdown summary.

Reads results/cov/<ds>_cov_diagnostics.csv and <ds>_shrinkage_results.csv for
aef/olmoearth/tessera, concatenates into:
  results/pastis_cov_diagnostics.csv
  results/pastis_shrinkage_results.csv
and writes results/cov_analysis_summary.md answering:
  - does shrinkage rescue covariance on small parcels?
  - does the N/D explanation still hold?
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COV = REPO / "results/cov"
DATASETS = ["aef", "olmoearth", "tessera"]
BIN_ORDER = ["tiny(<50)", "small(50-150)", "medium(150-500)", "large(>500)", "overall"]


def main() -> None:
    diag = pd.concat([pd.read_csv(COV / f"{d}_cov_diagnostics.csv") for d in DATASETS
                      if (COV / f"{d}_cov_diagnostics.csv").exists()], ignore_index=True)
    res = pd.concat([pd.read_csv(COV / f"{d}_shrinkage_results.csv") for d in DATASETS
                     if (COV / f"{d}_shrinkage_results.csv").exists()], ignore_index=True)

    diag_out = REPO / "results/pastis_cov_diagnostics.csv"
    res_out = REPO / "results/pastis_shrinkage_results.csv"
    diag.to_csv(diag_out, index=False)
    res.to_csv(res_out, index=False)

    # ── Diagnostics summary ──────────────────────────────────────────────────
    def bin_name(n):
        if n < 50:
            return "tiny(<50)"
        if n < 150:
            return "small(50-150)"
        if n < 500:
            return "medium(150-500)"
        return "large(>500)"

    diag["bin"] = diag.n_pixels.map(bin_name)
    lines = ["# PASTIS covariance diagnostics & shrinkage controls\n"]
    lines.append("## 1. Covariance conditioning (sample covariance)\n")
    lines.append("Fraction of parcels that are rank-deficient (`rank < D`) and under-sampled (`N < D`):\n")
    lines.append("| Encoder | D | N<D | rank<D | median cond# |")
    lines.append("|---|---|---|---|---|")
    for d in DATASETS:
        sub = diag[diag.dataset == d]
        if not len(sub):
            continue
        D = int(sub.D.iloc[0])
        lines.append(f"| {d} | {D} | {(sub.N_over_D<1).mean():.1%} | {(sub.rank_over_D<1).mean():.1%} | {sub.cond_number.median():.1f} |")
    lines.append("\n### By parcel-size bin (rank-deficient fraction)\n")
    lines.append("| Encoder | " + " | ".join(b for b in BIN_ORDER if b != "overall") + " |")
    lines.append("|---|" + "---|" * 4)
    for d in DATASETS:
        sub = diag[diag.dataset == d]
        if not len(sub):
            continue
        cells = []
        for b in BIN_ORDER[:-1]:
            s = sub[sub.bin == b]
            cells.append(f"{(s.rank_over_D<1).mean():.0%}" if len(s) else "—")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")

    # ── Shrinkage probe summary (linear, overall + by bin) ───────────────────
    lines.append("\n## 2. Does shrinkage rescue covariance? (linear probe, macro-F1, 5-fold mean)\n")
    lin = res[res.probe == "linear"]
    agg = lin.groupby(["dataset", "method", "bin"]).f1_macro.mean().reset_index()
    for d in DATASETS:
        sub = agg[agg.dataset == d]
        if not len(sub):
            continue
        lines.append(f"\n### {d}\n")
        methods = ["sample_cov", "ledoit_wolf", "oas", "shrink_a0.01", "shrink_a0.05",
                   "shrink_a0.1", "shrink_a0.25", "shrink_a0.5", "diag_std"]
        methods = [m for m in methods if m in set(sub.method)]
        lines.append("| Method | " + " | ".join(BIN_ORDER) + " |")
        lines.append("|---|" + "---|" * len(BIN_ORDER))
        for m in methods:
            row = sub[sub.method == m].set_index("bin").f1_macro
            cells = [f"{100*row[b]:.1f}" if b in row.index else "—" for b in BIN_ORDER]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

    # ── Auto verdicts ────────────────────────────────────────────────────────
    lines.append("\n## 3. Verdicts\n")
    # Shrinkage rescue on tiny parcels: best shrinkage vs sample_cov in tiny bin
    verdicts = []
    for d in DATASETS:
        sub = agg[(agg.dataset == d)]
        if not len(sub):
            continue
        tiny = sub[sub.bin == "tiny(<50)"]
        if not len(tiny):
            continue
        samp = tiny[tiny.method == "sample_cov"].f1_macro
        shrink_methods = tiny[tiny.method.str.startswith(("shrink_", "ledoit", "oas"))]
        if len(samp) and len(shrink_methods):
            best = shrink_methods.loc[shrink_methods.f1_macro.idxmax()]
            verdicts.append(f"- **{d}** tiny parcels: sample_cov={100*samp.iloc[0]:.1f}, "
                            f"best shrinkage ({best.method})={100*best.f1_macro:.1f} "
                            f"(Δ={100*(best.f1_macro-samp.iloc[0]):+.1f} pp).")
    lines.extend(verdicts)
    lines.append("\n*Interpretation: if shrinkage closes most of the small-parcel gap, "
                 "the collapse is an estimation problem (rank-deficiency), consistent with the "
                 "N/D explanation; if it does not, the collapse reflects information loss beyond "
                 "conditioning.*\n")

    out_md = REPO / "results/cov_analysis_summary.md"
    out_md.write_text("\n".join(lines))
    print(f"Wrote {diag_out}\nWrote {res_out}\nWrote {out_md}")


if __name__ == "__main__":
    main()
