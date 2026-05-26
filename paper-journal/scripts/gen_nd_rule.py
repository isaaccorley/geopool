"""Generate the compact N/D decision-rule figure for the journal paper."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
FIG_OUT = REPO / "paper-journal" / "figures" / "nd_rule.pdf"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

left = FancyBboxPatch(
    (0.04, 0.22),
    0.38,
    0.56,
    boxstyle="round,pad=0.02,rounding_size=0.035",
    facecolor="#ECF7EF",
    edgecolor="#2E7D32",
    linewidth=1.2,
)
right = FancyBboxPatch(
    (0.58, 0.22),
    0.38,
    0.56,
    boxstyle="round,pad=0.02,rounding_size=0.035",
    facecolor="#FFF4E6",
    edgecolor="#B26A00",
    linewidth=1.2,
)
ax.add_patch(left)
ax.add_patch(right)

ax.annotate(
    "",
    xy=(0.57, 0.50),
    xytext=(0.43, 0.50),
    arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": "#444444"},
)
ax.text(0.50, 0.57, "$N/D$", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.text(0.23, 0.69, "$N \\gg D$", ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(0.23, 0.56, "fixed chips", ha="center", va="center", fontsize=8.5)
ax.text(0.23, 0.43, "higher moments stable", ha="center", va="center", fontsize=8)
ax.text(0.23, 0.31, "use stats / covariance", ha="center", va="center", fontsize=8.5)

ax.text(0.77, 0.69, "$N \\approx D$", ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(0.77, 0.56, "parcels / objects", ha="center", va="center", fontsize=8.5)
ax.text(0.77, 0.43, "covariance unstable", ha="center", va="center", fontsize=8)
ax.text(0.77, 0.31, "use first-order summaries", ha="center", va="center", fontsize=8.5)

ax.text(
    0.50,
    0.08,
    "same embedding product, different region size $\\rightarrow$ different pooling rule",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#333333",
)

fig.savefig(FIG_OUT, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {FIG_OUT}")
