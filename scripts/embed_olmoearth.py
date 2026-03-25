import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import torch
from einops import rearrange
from olmoearth_pretrain.data.normalize import load_computed_config  # type: ignore[import-not-found]
from olmoearth_pretrain.model_loader import ModelID  # type: ignore[import-not-found]
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
from torchgeo.datasets import EuroSAT
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

H = W = 64
STD_MULTIPLIER = 2.0
OLMOEARTH_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
)


def build_norm_params(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-band normalization tensors from the OlmoEarth computed config.

    Returns (min_vals, ranges) each shaped (1, C, 1, 1) for broadcasting over (B, C, H, W).
    """
    config = load_computed_config()["sentinel2_l2a"]
    mins, ranges = [], []
    for band in OLMOEARTH_BANDS:
        mean = config[band]["mean"]
        std = config[band]["std"]
        min_val = mean - STD_MULTIPLIER * std
        max_val = mean + STD_MULTIPLIER * std
        mins.append(min_val)
        ranges.append(max_val - min_val)
    return (
        torch.tensor(mins, dtype=torch.float32, device=device).reshape(1, -1, 1, 1),
        torch.tensor(ranges, dtype=torch.float32, device=device).reshape(1, -1, 1, 1),
    )


def normalize(images: torch.Tensor, min_vals: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Apply OlmoEarth normalization: (x - min) / range per band."""
    return (images - min_vals) / ranges


def write(input_path: str, output_path: str, embedding: np.ndarray) -> None:
    embed_dim = embedding.shape[0]
    with rasterio.open(input_path) as src:
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": W,
            "height": H,
            "count": embed_dim,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "zstd",
            "predictor": 3,
            "interleave": "band",
        }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(embedding.astype(np.float32))


@torch.inference_mode()
def embed_batch(model: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
    images = rearrange(images, "b c h w -> b c () h w")
    inputs = [{"sentinel2_l2a": RasterImage(image=img)} for img in images]
    context = ModelContext(inputs=inputs, metadatas=[])
    feature_maps = model(context).feature_maps
    return feature_maps[0].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OlmoEarth embeddings for EuroSAT")
    parser.add_argument("--root", type=str, default="data", help="Data root directory")
    parser.add_argument(
        "--output", type=str, default="data/eurosat-olmoearth", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=96, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=12, help="DataLoader workers")
    parser.add_argument("--write-workers", type=int, default=8, help="Parallel write threads")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, mps, cpu)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], help="Splits to process"
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA only)")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["nano", "tiny", "base", "large"],
        help="OlmoEarth model size",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.model_size == "nano":
        model = OlmoEarth(model_id=ModelID.OLMOEARTH_V1_NANO, patch_size=1)
    elif args.model_size == "tiny":
        model = OlmoEarth(model_id=ModelID.OLMOEARTH_V1_TINY, patch_size=1)
    elif args.model_size == "base":
        model = OlmoEarth(model_id=ModelID.OLMOEARTH_V1_BASE, patch_size=1)
    elif args.model_size == "large":
        model = OlmoEarth(model_id=ModelID.OLMOEARTH_V1_LARGE, patch_size=1)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    model.eval()
    model.to(device)

    if args.compile:
        model = torch.compile(model)

    norm_min, norm_range = build_norm_params(device)

    for split in args.splits:
        print(f"Processing {split} split...")
        dataset = EuroSAT(root=args.root, split=split, bands=OLMOEARTH_BANDS)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=2,
        )

        # Async writes in background
        executor = ThreadPoolExecutor(max_workers=args.write_workers)
        futures = []

        sample_idx = 0
        for batch in tqdm(loader, desc=split):
            batch_size = batch["image"].shape[0]
            batch_samples = dataset.samples[sample_idx : sample_idx + batch_size]
            batch_paths = []
            for filepath, _ in batch_samples:
                folder, filename = filepath.split("/")[-2:]
                output_path = os.path.join(args.output + f"-{args.model_size}", folder, filename)
                batch_paths.append((filepath, output_path))

            all_outputs_exist = all(os.path.exists(output_path) for _, output_path in batch_paths)
            if all_outputs_exist:
                sample_idx += batch_size
                continue

            images = batch["image"].to(device, non_blocking=True)
            images = normalize(images, norm_min, norm_range)
            embeddings = embed_batch(model, images)  # type: ignore[arg-type]

            for emb, (filepath, output_path) in zip(embeddings, batch_paths, strict=False):
                if os.path.exists(output_path):
                    sample_idx += 1
                    continue
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                futures.append(executor.submit(write, filepath, output_path, emb))
                sample_idx += 1

        # Wait for writes to finish
        for f in tqdm(futures, desc=f"{split} writes"):
            f.result()
        executor.shutdown()

    print("Done!")
