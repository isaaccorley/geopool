"""Generate OLMoEarth embeddings for PASTIS patches using the actual S2 time series.

PASTIS S2 data: (T=43, 10, 128, 128) int16
OLMoEarth input: (B, H, W, T, 12) float32  -- max T=12 for 128x128 input
Strategy:
  - Uniformly subsample T_in -> N_TIMESTEPS acquisitions
  - Remap 10 PASTIS bands -> 12 OLMoEarth bands (zero-fill B01, B09)
  - Normalise with OLMoEarth Normalizer
  - Encode -> pool_spatially -> upsample 16x16 -> 128x128 (nearest)
  - Save as 128-band float32 GeoTIFF matching AEF/Tessera format

Output is compatible with scripts/pool_pastis.py unchanged.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

# ── Constants ─────────────────────────────────────────────────────────────────

STD_MULTIPLIER = 2.0
PATCH_SIZE = 8
INPUT_RES = 10
N_TIMESTEPS = 12  # max OLMoEarth supports at 128x128 resolution

# PASTIS S2 band order (10 bands)
PASTIS_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# OLMoEarth expects 12 bands in this order
OLMOEARTH_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]

# Precompute: for each OLMoEarth band, which PASTIS index to use (-1 = zero-fill)
BAND_MAP: list[int] = [
    PASTIS_S2_BANDS.index(b) if b in PASTIS_S2_BANDS else -1
    for b in OLMOEARTH_BANDS
]  # e.g. [0,1,2,6,3,4,5,7,8,9,-1,-1]


def remap_bands(data: np.ndarray) -> np.ndarray:
    """Remap PASTIS (T, 10, H, W) -> OLMoEarth (T, 12, H, W), zero-fill missing."""
    T, _, H, W = data.shape
    out = np.zeros((T, 12, H, W), dtype=data.dtype)
    for oe_idx, pastis_idx in enumerate(BAND_MAP):
        if pastis_idx >= 0:
            out[:, oe_idx] = data[:, pastis_idx]
    return out


def subsample_timesteps(
    data: np.ndarray,
    dates: list[int],
    n: int = N_TIMESTEPS,
) -> tuple[np.ndarray, list[int]]:
    """Uniformly subsample T time steps from the full series."""
    T = data.shape[0]
    if T <= n:
        return data, dates
    indices = np.round(np.linspace(0, T - 1, n)).astype(int)
    return data[indices], [dates[i] for i in indices]


def parse_date(d: int) -> tuple[int, int, int]:
    """Parse YYYYMMDD int -> (day, month_0indexed, year).

    month_embed table has shape [12] (0-indexed), so January=0 ... December=11.
    """
    year = d // 10000
    month = (d % 10000) // 100
    day = d % 100
    return day, month - 1, year  # month: 1-12 -> 0-11


def write_tif(output_path: str | Path, embedding: np.ndarray) -> None:
    """Write (H, W, D) float32 embedding as a D-band GeoTIFF (no georef needed)."""
    H, W, D = embedding.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": W,
        "height": H,
        "count": D,
        "compress": "zstd",
        "predictor": 3,
        "interleave": "band",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.moveaxis(embedding.astype(np.float32, copy=False), -1, 0))


@torch.inference_mode()
def embed_batch(
    model: Any,
    normalizer: Normalizer,
    images: torch.Tensor,     # (B, T, 12, H, W) float32 in raw reflectance
    timestamps_np: np.ndarray, # (B, T, 3)  [day, month, year]
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of PASTIS patches. Returns (B, H, W, D) float32."""
    B, T, C, H, W = images.shape
    # OLMoEarth normalizer expects (B, H, W, T, C)
    s2 = images.permute(0, 3, 4, 1, 2).cpu().numpy()  # (B, H, W, T, C)
    # S2 L2A atmospheric correction can produce small negative DN values;
    # clip to zero so the model's internal kernels don't hit device-side asserts.
    s2 = np.clip(s2, 0.0, None)
    s2 = normalizer.normalize(Modality.SENTINEL2_L2A, s2)

    ts = torch.from_numpy(timestamps_np).long().to(device)  # (B, T, 3)

    sample = MaskedOlmoEarthSample(
        timestamps=ts,
        sentinel2_l2a=torch.from_numpy(s2).float().to(device),
        sentinel2_l2a_mask=torch.zeros(B, H, W, T, dtype=torch.long, device=device),
    )

    outputs = model.encoder(sample, patch_size=PATCH_SIZE, input_res=INPUT_RES, fast_pass=True)
    pooled = outputs["tokens_and_masks"].pool_spatially(PoolingType.MEAN)
    # pooled: (B, H//PATCH_SIZE, W//PATCH_SIZE, D) — e.g. (B, 16, 16, 128)

    # Upsample back to original (H, W)
    pooled_up = F.interpolate(
        pooled.permute(0, 3, 1, 2).float(),  # (B, D, h, w)
        size=(H, W),
        mode="nearest",
    )  # (B, D, H, W)

    return pooled_up.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, D)


def main() -> None:
    parser = argparse.ArgumentParser(description="OLMoEarth embeddings for PASTIS time series")
    parser.add_argument("--data-dir", default="data/pastis-r/PASTIS-R", help="PASTIS-R root dir")
    parser.add_argument("--meta", default="data/pastis-r/PASTIS-R/metadata.geojson")
    parser.add_argument("--output", default="data/pastis-olmoearth", help="Output dir")
    parser.add_argument("--model-size", default="nano", choices=["nano", "tiny", "base", "large"])
    parser.add_argument("--n-timesteps", type=int, default=N_TIMESTEPS, help="Time steps to sample (max 12)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4, help="Async write workers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assert args.n_timesteps <= N_TIMESTEPS, f"--n-timesteps max is {N_TIMESTEPS} for 128x128 input"

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    s2_dir = Path(args.data_dir) / "DATA_S2"

    # ── Load model ────────────────────────────────────────────────────────────
    model_id = getattr(ModelID, f"OLMOEARTH_V1_{args.model_size.upper()}")
    print(f"Loading OLMoEarth-{args.model_size}...")
    model = load_model_from_id(model_id, load_weights=True)
    model.eval().to(device)
    if args.compile:
        model = torch.compile(model)
    normalizer = Normalizer(std_multiplier=STD_MULTIPLIER)

    # ── Load metadata (patch IDs + dates) ────────────────────────────────────
    with open(args.meta) as f:
        meta = json.load(f)

    patch_ids = []
    patch_dates: dict[int, list[int]] = {}
    for feat in meta["features"]:
        pid = feat["properties"]["ID_PATCH"]
        dates_raw = feat["properties"]["dates-S2"]
        patch_ids.append(pid)
        patch_dates[pid] = [dates_raw[str(i)] for i in range(len(dates_raw))]

    # Filter already-done
    todo = [pid for pid in patch_ids if args.overwrite or not (output_dir / f"S2_{pid}.tif").exists()]
    print(f"Patches: {len(patch_ids)} total, {len(todo)} to embed")

    # ── Process in batches ────────────────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=args.num_workers)
    write_futures = []
    errors: list[int] = []

    for batch_start in tqdm(range(0, len(todo), args.batch_size), desc="batches"):
        batch_ids = todo[batch_start : batch_start + args.batch_size]

        images_list = []
        ts_list = []

        for pid in batch_ids:
            raw = np.load(s2_dir / f"S2_{pid}.npy")  # (T, 10, H, W) int16
            dates = patch_dates[pid]
            raw_sub, dates_sub = subsample_timesteps(raw, dates, args.n_timesteps)
            raw_12 = remap_bands(raw_sub)  # (T, 12, H, W)
            images_list.append(torch.from_numpy(raw_12.astype(np.float32)))  # (T,12,H,W)
            ts_arr = np.array([parse_date(d) for d in dates_sub], dtype=np.int64)  # (T, 3)
            ts_list.append(ts_arr)

        images = torch.stack(images_list).to(device)  # (B, T, 12, H, W)
        timestamps_np = np.stack(ts_list)  # (B, T, 3)

        try:
            embeddings = embed_batch(model, normalizer, images, timestamps_np, device)
        except torch.AcceleratorError as e:
            # CUDA context is poisoned after a device-side assert; cannot recover.
            print(f"  Batch {batch_start//args.batch_size} FATAL CUDA ({batch_ids}): {e}")
            errors.extend(batch_ids)
            raise  # propagate so we get the full traceback in logs
        except Exception as e:
            print(f"  Batch {batch_start//args.batch_size} FAILED ({batch_ids}): {e}")
            errors.extend(batch_ids)
            continue

        for emb, pid in zip(embeddings, batch_ids, strict=True):
            out_path = output_dir / f"S2_{pid}.tif"
            write_futures.append(executor.submit(write_tif, out_path, emb))

    for fut in tqdm(write_futures, desc="writing"):
        fut.result()
    executor.shutdown()

    done = len(list(output_dir.glob("S2_*.tif")))
    print(f"Done. {done} embeddings -> {output_dir}")


if __name__ == "__main__":
    main()
