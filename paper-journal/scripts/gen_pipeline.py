"""
Combined 2-row pipeline figure:
  Row A (EuroSAT):   RGB image  → GFM → Pixel embeddings          → Pool → Vector → Patch label
  Row B (PASTIS):    4 S2 frames → GFM → Pixel embs (masked parcel) → Pool → Vector → Crop label
Column headers appear once above row A; row labels on the left.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import rasterio
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO     = Path("/u/isaaccorley/github/geopool")
EUR_RGB  = Path("/u/isaaccorley/github/torchgeo-bench/data/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif")
PAS_DIR  = REPO / "data/pastis-r/PASTIS-R"
PAS_EMB  = REPO / "data/pastis-aef/S2_10000.tif"
OUT      = REPO / "paper-journal/figures/pipeline.pdf"

# ── EuroSAT ────────────────────────────────────────────────────────────────
# Use 13-band S2 data: RGB for input panel, all 13 bands as features for embedding panel
EUR_CLASS, EUR_ID = "AnnualCrop", "AnnualCrop_1002"

with rasterio.open(EUR_RGB / EUR_CLASS / f"{EUR_ID}.tif") as src:
    all_bands = src.read().astype(np.float32)  # 13,H,W

# RGB: bands 4 (R), 3 (G), 2 (B) — 1-indexed
r, g, b = all_bands[3], all_bands[2], all_bands[1]
eur_rgb = np.clip(np.fliplr(np.stack([r, g, b], axis=-1)) / 2500.0, 0, 1)

# "Embedding" visualization: PCA of all 13 spectral bands (simulates what AEF embeddings look like)
eur_emb = all_bands.transpose(1, 2, 0).astype(np.float32)  # H,W,13
flat_e  = eur_emb.reshape(-1, 13)
pca_e   = PCA(n_components=3).fit(flat_e)
eur_pca = pca_e.transform(flat_e).reshape(64, 64, 3)
for i in range(3):
    p2, p98 = np.percentile(eur_pca[:,:,i], [2, 98])
    eur_pca[:,:,i] = np.clip((eur_pca[:,:,i] - p2) / (p98 - p2 + 1e-8), 0, 1)

eur_pooled = flat_e.mean(axis=0)   # 13-dim mean (representative)
eur_pool_n = (eur_pooled - eur_pooled.min()) / (eur_pooled.max() - eur_pooled.min() + 1e-6)

D_eur = 64  # AEF output dim (for label purposes)

# ── PASTIS ─────────────────────────────────────────────────────────────────
PATCH = "10000"
s2  = np.load(PAS_DIR / f"DATA_S2/S2_{PATCH}.npy").astype(np.float32)   # T,C,H,W
ann = np.load(PAS_DIR / f"ANNOTATIONS/ParcelIDs_{PATCH}.npy", allow_pickle=True)  # H,W

with rasterio.open(PAS_EMB) as src:
    pas_emb_full = src.read().astype(np.float32)  # D,H,W

H, W, D_pas = *ann.shape, pas_emb_full.shape[0]

# Pick a medium parcel near center
uids, counts = np.unique(ann, return_counts=True)
uids, counts = uids[1:], counts[1:]  # drop background
best_uid = None
for uid, cnt in zip(uids[counts.argsort()[::-1]], sorted(counts, reverse=True)):
    rows_u, cols_u = np.where(ann == uid)
    cr, cc = rows_u.mean(), cols_u.mean()
    if 100 < cnt < 400 and 25 < cr < 103 and 25 < cc < 103:
        best_uid = uid
        break
if best_uid is None:
    best_uid = uids[counts.argsort()[::-1][4]]

mask = (ann == best_uid)
rows_m, cols_m = np.where(mask)
r0 = max(0, rows_m.min() - 8);  r1 = min(H, rows_m.max() + 8)
c0 = max(0, cols_m.min() - 8);  c1 = min(W, cols_m.max() + 8)

T_IDXS = [3, 12, 25, 38]
MONTHS  = ["Jan", "Apr", "Jul", "Oct"]

def s2_rgb(t_idx):
    img = s2[t_idx][[2,1,0]].transpose(1,2,0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return np.clip(img, 0, 1)[r0:r1, c0:c1]

flat_p    = pas_emb_full.reshape(D_pas, -1).T
pca_p     = PCA(n_components=3).fit(flat_p)
pas_pca   = pca_p.transform(flat_p).reshape(H, W, 3)
for i in range(3):
    pas_pca[:,:,i] -= pas_pca[:,:,i].min()
    pas_pca[:,:,i] /= (pas_pca[:,:,i].max() + 1e-6)

pas_pca_crop  = pas_pca[r0:r1, c0:c1]
mask_crop     = mask[r0:r1, c0:c1]
shade         = np.zeros((r1-r0, c1-c0, 4), dtype=float)
shade[~mask_crop] = [0, 0, 0, 0.55]

pas_pooled = pas_emb_full[:, mask].mean(axis=1)
pas_pool_n = (pas_pooled - pas_pooled.min()) / (pas_pooled.max() - pas_pooled.min() + 1e-6)

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 4.2), facecolor="white")

ROW_H  = 0.34   # height of image/emb panels
ROW_A  = 0.57   # bottom of row A
ROW_B  = 0.10   # bottom of row B
HDR_Y  = ROW_A + ROW_H + 0.04   # column header y
VEC_H  = 0.055  # height of 1-D colour bar

# x column boundaries
ROW_LBL_X = 0.005
INPUT_L, INPUT_R  = 0.055, 0.210
ARR1_L,  ARR1_R   = 0.210, 0.290
EMB_L,   EMB_R    = 0.290, 0.495
ARR2_L,  ARR2_R   = 0.495, 0.575
VEC_L,   VEC_R    = 0.575, 0.730
ARR3_L,  ARR3_R   = 0.730, 0.790
LBL_L,   LBL_R    = 0.790, 0.960

HDR_KW = dict(fontsize=10, fontweight="bold", ha="center", va="bottom", color="black")
SUB_KW = dict(fontsize=7.5, ha="center", va="top", color="#444")

def col_mid(l, r):  return (l + r) / 2

# ── Column headers (once, above row A) ────────────────────────────────────
fig.text(col_mid(INPUT_L, INPUT_R), HDR_Y, "Input",         **HDR_KW)
fig.text(col_mid(EMB_L,   EMB_R),   HDR_Y, "Pixel Embeddings", **HDR_KW)
fig.text(col_mid(VEC_L,   VEC_R),   HDR_Y, "Embedding",     **HDR_KW)
fig.text(col_mid(LBL_L,   LBL_R),   HDR_Y, "Label",         **HDR_KW)

# ── Helper: horizontal arrow with optional italic label ────────────────────
def h_arrow(l, r, mid_y, label=None):
    ax = fig.add_axes([l, mid_y - 0.04, r - l, 0.08])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0, mutation_scale=18))
    if label:
        fig.text(col_mid(l, r), mid_y + 0.055,
                 label, ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")

# ── Helper: image panel ────────────────────────────────────────────────────
def img_panel(x, y, w, h, img, cmap=None, aspect="equal"):
    ax = fig.add_axes([x, y, w, h])
    kw = dict(cmap=cmap, aspect=aspect) if cmap else dict(aspect=aspect)
    ax.imshow(img, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.5); sp.set_color("#333")
    return ax

def color_box(x, y, w, h, color, text, fontsize=12):
    """Solid colored panel with centred bold text."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.5); sp.set_color("#333")
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white")
    return ax

# ══════════════════════════════════════════════════════════
# ROW A — EuroSAT
# ══════════════════════════════════════════════════════════
mid_A = ROW_A + ROW_H / 2

# Row label (rotated)
fig.text(ROW_LBL_X, mid_A, "EuroSAT", ha="left", va="center",
         fontsize=10, fontweight="bold", color="#222", rotation=90)

# Input: RGB patch
img_panel(INPUT_L, ROW_A, INPUT_R - INPUT_L, ROW_H, eur_rgb)
fig.text(col_mid(INPUT_L, INPUT_R), ROW_A - 0.03,
         f"64×64×12  (int16)", **SUB_KW)

# → GFM →
h_arrow(ARR1_L, ARR1_R, mid_A, "GFM")

# Pixel embeddings
img_panel(EMB_L, ROW_A, EMB_R - EMB_L, ROW_H, eur_pca)
fig.text(col_mid(EMB_L, EMB_R), ROW_A - 0.03,
         f"64×64×{D_eur}  (float32)", **SUB_KW)

# → Pool →
h_arrow(ARR2_L, ARR2_R, mid_A, "Pool")

# Pooled vector bar
img_panel(VEC_L, mid_A - VEC_H/2, VEC_R - VEC_L, VEC_H,
          eur_pool_n[np.newaxis, :], cmap="viridis", aspect="auto")
fig.text(col_mid(VEC_L, VEC_R), mid_A - VEC_H/2 - 0.03,
         f"{D_eur}×1", **SUB_KW)

# →
h_arrow(ARR3_L, ARR3_R, mid_A)

# Label box — solid green with bold text
color_box(LBL_L, ROW_A, LBL_R - LBL_L, ROW_H, "#228B22", "Annual\nCrop")

# ── separator ────────────────────────────────────────────────────────────
sep_y = (ROW_A + ROW_B + ROW_H) / 2
fig.add_artist(Line2D([0.04, 0.97], [sep_y, sep_y],
                      transform=fig.transFigure, color="#CCCCCC", lw=1.0, linestyle="--"))

# ══════════════════════════════════════════════════════════
# ROW B — PASTIS
# ══════════════════════════════════════════════════════════
mid_B = ROW_B + ROW_H / 2

# Row label
fig.text(ROW_LBL_X, mid_B, "PASTIS", ha="left", va="center",
         fontsize=10, fontweight="bold", color="#222", rotation=90)

# Input: 4 S2 frames stacked + overlapping like a fanned deck of photos
fr_w, fr_h = 0.092, 0.29        # single frame size (figure fractions)
dx,   dy   = 0.016, 0.010       # offset per layer (right, up)
stack_w = fr_w + 3 * dx
stack_h = fr_h + 3 * dy
# Centre the whole stack inside the INPUT column
sx0 = (INPUT_L + INPUT_R) / 2 - stack_w / 2
sy0 = ROW_B + ROW_H / 2 - stack_h / 2

for i, (ti, mo) in enumerate(zip(T_IDXS, MONTHS)):
    fx = sx0 + i * dx
    fy = sy0 + i * dy
    ax_f = fig.add_axes([fx, fy, fr_w, fr_h])
    ax_f.imshow(s2_rgb(ti))
    ax_f.set_xticks([]); ax_f.set_yticks([])
    for sp in ax_f.spines.values():
        sp.set_linewidth(2.0)
        sp.set_color("white")   # white photo-edge border
    # Month label on exposed top-left corner of each frame
    ax_f.text(0.06, 0.94, mo, transform=ax_f.transAxes,
              ha="left", va="top", fontsize=7.5, color="white", fontweight="bold",
              bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"))

# → GFM →
h_arrow(ARR1_L, ARR1_R, mid_B, "GFM")

# Pixel embeddings with parcel mask
ax_pas_emb = img_panel(EMB_L, ROW_B, EMB_R - EMB_L, ROW_H, pas_pca_crop)
ax_pas_emb.contour(mask_crop, levels=[0.5], colors=["white"], linewidths=[2.0])
ax_pas_emb.imshow(shade)
fig.text(col_mid(EMB_L, EMB_R), ROW_B - 0.03,
         f"H×W×{D_pas}  (parcel outlined)", **SUB_KW)

# → Pool →
h_arrow(ARR2_L, ARR2_R, mid_B, "Pool")

# Pooled vector — same physical dimensions as EuroSAT bar
img_panel(VEC_L, mid_B - VEC_H/2, VEC_R - VEC_L, VEC_H,
          pas_pool_n[np.newaxis, :], cmap="viridis", aspect="auto")
fig.text(col_mid(VEC_L, VEC_R), mid_B - VEC_H/2 - 0.03,
         f"{D_pas}×1", **SUB_KW)

# →
h_arrow(ARR3_L, ARR3_R, mid_B)

# Parcel polygon: white/gray background, green-filled parcel shape + text label
parcel_vis = np.ones((r1 - r0, c1 - c0, 3), dtype=np.float32)
parcel_vis[mask_crop]  = [0.18, 0.55, 0.18]   # green parcel pixels
parcel_vis[~mask_crop] = [0.93, 0.93, 0.93]   # light gray background
ax_pas_lbl = img_panel(LBL_L, ROW_B, LBL_R - LBL_L, ROW_H, parcel_vis)
ax_pas_lbl.text(0.5, 0.5, "Corn /\nMaize", transform=ax_pas_lbl.transAxes,
                ha="center", va="center", fontsize=11, fontweight="bold", color="white",
                bbox=dict(facecolor="#1a5e1a", edgecolor="none",
                          boxstyle="round,pad=0.3", alpha=0.88))

# ── Save ─────────────────────────────────────────────────────────────────
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("Saved:", OUT)
print(f"  PASTIS parcel uid={best_uid}, count={mask.sum()}, crop region {r1-r0}×{c1-c0}")
