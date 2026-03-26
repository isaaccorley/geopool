import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = ["aef", "olmoearth-nano", "tessera"]
EXCLUDED_POOLS: set[str] = {"signed_non_cancelling_gem"}

POOL_LABELS = {
    "mean": "Mean",
    "std": "Std",
    "max": "Max",
    "gem": "GeM",
    "mean_std": "Mean+Std",
    "mean_max": "Mean+Max",
    "stats": "Stats",
    "median_iqr": "Median+IQR",
    "percentiles": "Percentiles",
    "center_weighted_mean": "Center-Weighted",
    "flattened_cov": "Covariance",
    "pca_64": "PCA",
    "bovw_128": "BoVW",
}


def label_pool(pool: str) -> str:
    if pool in POOL_LABELS:
        return POOL_LABELS[pool]
    return pool.replace("_", " ")


def shade_value(value: float, vmin: float, vmax: float, low: int, high: int) -> int:
    if vmax == vmin:
        return round((low + high) / 2)
    return round(low + (value - vmin) / (vmax - vmin) * (high - low))


def infer_shared_pools(results_dir: Path, datasets: list[str], probe: str) -> list[str]:
    pools: set[str] | None = None
    for dataset in datasets:
        csv_path = results_dir / dataset / "linear_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing results file: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[df["probe"] == probe]
        dataset_pools = set(df["pool"].unique().tolist()) - EXCLUDED_POOLS
        pools = dataset_pools if pools is None else pools & dataset_pools
    return sorted(pools or [])


def parse_results(
    results_dir: Path,
    datasets: list[str],
    pools: list[str],
    probe: str,
) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in datasets:
        csv_path = results_dir / dataset / "linear_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing results file: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[(df["probe"] == probe) & (df["pool"].isin(pools))]
        for split in ["standard", "spatial"]:
            subset = df[df["split_type"] == split].set_index("pool")
            if subset.empty:
                raise ValueError(
                    f"No rows for dataset={dataset}, split={split}, probe={probe} in {csv_path}"
                )
            values[(dataset, split)] = subset["accuracy"].to_dict()
    return values


def compute_avg_values(
    values: dict[tuple[str, str], dict[str, float]],
    pools: list[str],
    datasets: list[str],
) -> dict[str, dict[str, float]]:
    avg_values: dict[str, dict[str, float]] = {"standard": {}, "spatial": {}}
    for split in ["standard", "spatial"]:
        for pool in pools:
            avg_values[split][pool] = sum(values[(ds, split)][pool] for ds in datasets) / len(
                datasets
            )
    return avg_values


def compute_shades(
    values: dict[tuple[str, str], dict[str, float]],
    avg_values: dict[str, dict[str, float]],
    pools: list[str],
    datasets: list[str],
    low: int,
    high: int,
) -> dict[tuple[str, str], dict[str, int]]:
    shades: dict[tuple[str, str], dict[str, int]] = {}
    for split in ["standard", "spatial"]:
        for dataset in datasets:
            vals = values[(dataset, split)]
            ordered = [vals[p] for p in pools]
            vmin, vmax = min(ordered), max(ordered)
            shades[(dataset, split)] = {
                p: shade_value(vals[p], vmin, vmax, low, high) for p in pools
            }
        avg_vals = avg_values[split]
        ordered_avg = [avg_vals[p] for p in pools]
        vmin, vmax = min(ordered_avg), max(ordered_avg)
        shades[("avg", split)] = {p: shade_value(avg_vals[p], vmin, vmax, low, high) for p in pools}
    return shades


def compute_rankings(
    values: dict[tuple[str, str], dict[str, float]],
    avg_values: dict[str, dict[str, float]],
    pools: list[str],
    datasets: list[str],
) -> dict[tuple[str, str], tuple[str, str]]:
    rankings: dict[tuple[str, str], tuple[str, str]] = {}
    for split in ["standard", "spatial"]:
        for dataset in datasets:
            items = sorted(
                ((pool, values[(dataset, split)][pool]) for pool in pools),
                key=lambda x: x[1],
                reverse=True,
            )
            rankings[(dataset, split)] = (items[0][0], items[1][0])
        items = sorted(
            ((pool, avg_values[split][pool]) for pool in pools),
            key=lambda x: x[1],
            reverse=True,
        )
        rankings[("avg", split)] = (items[0][0], items[1][0])
    return rankings


def format_value(value: float, best: str, second: str, pool: str) -> str:
    text = f"{value * 100:.1f}"
    if pool == best:
        return f"\\textbf{{{text}}}"
    if pool == second:
        return f"\\textit{{{text}}}"
    return text


def format_table(
    values: dict[tuple[str, str], dict[str, float]],
    avg_values: dict[str, dict[str, float]],
    shades: dict[tuple[str, str], dict[str, int]],
    rankings: dict[tuple[str, str], tuple[str, str]],
    datasets: list[str],
    pools: list[str],
    caption: str,
    label: str,
    shade_low: int = 15,
    shade_high: int = 60,
) -> str:
    # Compute gap values and shading (lower gap = better = darker)
    gap_values = {
        pool: (avg_values["standard"][pool] - avg_values["spatial"][pool]) * 100 for pool in pools
    }
    gap_min = min(gap_values.values())
    gap_max = max(gap_values.values())
    # Invert shading: lower gap -> higher shade (darker)
    gap_shades = {
        pool: shade_value(gap_max - gap, 0, gap_max - gap_min, shade_low, shade_high)
        for pool, gap in gap_values.items()
    }
    # Find best (lowest) and second-best gap
    sorted_gaps = sorted(gap_values.items(), key=lambda x: x[1])
    best_gap = sorted_gaps[0][0] if sorted_gaps else None
    second_gap = sorted_gaps[1][0] if len(sorted_gaps) > 1 else None

    header = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\footnotesize",
        f"  \\caption{{{caption}}}\\label{{{label}}}",
        "  \\begin{tabular}{lcccc|cccc|c}",
        "    \\toprule",
        "    & \\multicolumn{4}{c}{Spatial split} & \\multicolumn{4}{c}{Random split} & \\\\",
        "    \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "    Pooling & AEF & OlmoEarth$^*$ & Tessera & Avg & AEF & OlmoEarth$^*$ & Tessera & Avg & Gap$\\downarrow$ \\\\",
        "    \\midrule",
    ]

    def render_rows(row_pools: list[str]) -> list[str]:
        rows: list[str] = []
        for pool in row_pools:
            row = [label_pool(pool)]
            # Spatial split columns first
            for dataset in datasets + ["avg"]:
                value = (
                    avg_values["spatial"][pool]
                    if dataset == "avg"
                    else values[(dataset, "spatial")][pool]
                )
                shade = shades[(dataset, "spatial")][pool]
                best, second = rankings[(dataset, "spatial")]
                row.append(f"\\cellperf{{{shade}}}{{{format_value(value, best, second, pool)}}}")
            # Random split columns second
            for dataset in datasets + ["avg"]:
                value = (
                    avg_values["standard"][pool]
                    if dataset == "avg"
                    else values[(dataset, "standard")][pool]
                )
                shade = shades[(dataset, "standard")][pool]
                best, second = rankings[(dataset, "standard")]
                row.append(f"\\cellperf{{{shade}}}{{{format_value(value, best, second, pool)}}}")
            # Add Gap column (avg random - avg spatial) with shading
            gap = gap_values[pool]
            gap_shade = gap_shades[pool]
            gap_text = f"{gap:.1f}"
            if pool == best_gap:
                gap_text = f"\\textbf{{{gap_text}}}"
            elif pool == second_gap:
                gap_text = f"\\textit{{{gap_text}}}"
            row.append(f"\\cellperf{{{gap_shade}}}{{{gap_text}}}")
            rows.append("    " + " & ".join(row) + " \\\\")
        return rows

    pca_pools = [pool for pool in pools if pool.startswith("pca_")]
    bovw_pools = [pool for pool in pools if pool.startswith("bovw_")]
    parametric_pools = pca_pools + bovw_pools
    other_pools = [pool for pool in pools if pool not in parametric_pools]

    def sort_by_spatial_avg(items: list[str]) -> list[str]:
        return sorted(items, key=lambda p: avg_values["spatial"][p])

    # Sort by spatial split Avg (ascending)
    other_pools = sort_by_spatial_avg(other_pools)
    parametric_pools = sort_by_spatial_avg(parametric_pools)

    body = []
    body.extend(render_rows(other_pools))
    if parametric_pools:
        body.append("    \\midrule")
        body.append("    \\multicolumn{10}{l}{\\textit{Parametric pools (fit on train set)}} \\\\")
        body.extend(render_rows(parametric_pools))

    footer = [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(header + body + footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX linear probe table from results/*/linear_results.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing dataset result folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset folder names to include (default: aef olmoearth tessera).",
    )
    parser.add_argument(
        "--probe",
        type=str,
        default="linear",
        help="Probe name to select (default: linear).",
    )
    parser.add_argument(
        "--shade-min",
        type=int,
        default=15,
        help="Minimum shading value (default: 15).",
    )
    parser.add_argument(
        "--shade-max",
        type=int,
        default=60,
        help="Maximum shading value (default: 60).",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Linear probe accuracy across embedding sources. Cell shading indicates higher "
            "accuracy within each column (darker = higher). Best per column in \\textbf{bold} "
            "and second-best in \\textit{italics}. Gap = accuracy drop from random to spatial split. "
            "$^*$OlmoEarth-Nano variant."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:linear_all",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file to write the LaTeX table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = infer_shared_pools(args.results_dir, args.datasets, args.probe)
    if not pools:
        raise ValueError("No shared pools found across datasets.")
    values = parse_results(args.results_dir, args.datasets, pools, args.probe)
    avg_values = compute_avg_values(values, pools, args.datasets)
    shades = compute_shades(
        values, avg_values, pools, args.datasets, args.shade_min, args.shade_max
    )
    rankings = compute_rankings(values, avg_values, pools, args.datasets)
    table = format_table(
        values,
        avg_values,
        shades,
        rankings,
        args.datasets,
        pools,
        args.caption,
        args.label,
    )
    if args.output:
        args.output.write_text(table)
    else:
        print(table)


if __name__ == "__main__":
    main()
