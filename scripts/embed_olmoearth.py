import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
from torchgeo.datasets import EuroSAT
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

STD_MULTIPLIER = 2.0
PATCH_SIZE = 8
INPUT_RES = 10
TIME_STEPS = 3
TIMESTAMP_YEAR = 2018
TIMESTAMP_MONTH = 5
TIMESTAMP_DAY = 15
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


def write(input_path: str, output_path: str, embedding: np.ndarray) -> None:
    height, width, embed_dim = embedding.shape
    with rasterio.open(input_path) as src:
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": width,
            "height": height,
            "count": embed_dim,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "zstd",
            "predictor": 3,
            "interleave": "band",
        }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.moveaxis(embedding.astype(np.float32, copy=False), -1, 0))


@torch.inference_mode()
def embed_batch(
    model: Any,
    normalizer: Normalizer,
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of EuroSAT images with OlmoEarth."""
    batch_size, _, height, width = images.shape
    sentinel2_l2a = images.permute(0, 2, 3, 1).cpu().numpy()
    sentinel2_l2a = np.repeat(sentinel2_l2a[:, :, :, None, :], TIME_STEPS, axis=3)
    sentinel2_l2a = normalizer.normalize(Modality.SENTINEL2_L2A, sentinel2_l2a)

    timestamps = torch.zeros(batch_size, TIME_STEPS, 3, dtype=torch.long, device=device)
    timestamps[:, :, 0] = TIMESTAMP_DAY
    timestamps[:, :, 1] = TIMESTAMP_MONTH
    timestamps[:, :, 2] = TIMESTAMP_YEAR

    sample = MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=torch.from_numpy(sentinel2_l2a).float().to(device),
        sentinel2_l2a_mask=torch.zeros(
            batch_size, height, width, TIME_STEPS, dtype=torch.long, device=device
        ),
    )

    outputs = model.encoder(sample, patch_size=PATCH_SIZE, input_res=INPUT_RES, fast_pass=True)
    pooled = outputs["tokens_and_masks"].pool_spatially(PoolingType.MEAN)
    pooled = F.interpolate(
        pooled.permute(0, 3, 1, 2),
        size=(height, width),
        mode="nearest",
    )
    return pooled.permute(0, 2, 3, 1).cpu().numpy()


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

    model_id = getattr(ModelID, f"OLMOEARTH_V1_{args.model_size.upper()}")
    model = load_model_from_id(model_id, load_weights=True)

    model.eval()
    model.to(device)

    if args.compile:
        model = torch.compile(model)

    normalizer = Normalizer(std_multiplier=STD_MULTIPLIER)

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

            images = batch["image"]
            embeddings = embed_batch(model, normalizer, images, device)

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
