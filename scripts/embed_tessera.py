import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import torch
from einops import rearrange
from torchgeo.datasets import EuroSAT
from torchgeo.models import Tessera_Weights, tessera
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

H = W = 64
EMBED_DIM = 512
TESSERA_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


def write(input_path: str, output_path: str, embedding: np.ndarray) -> None:
    with rasterio.open(input_path) as src:
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": W,
            "height": H,
            "count": EMBED_DIM,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "zstd",
            "predictor": 3,
            "interleave": "band",
        }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(embedding.astype(np.float32))


@torch.inference_mode()
def embed_batch(
    model: torch.nn.Module,
    transforms: torch.nn.Module,
    images: torch.Tensor,
    doy: torch.Tensor,
) -> np.ndarray:
    b = images.shape[0]
    doy_batch = doy.expand(b, -1, -1, -1)
    x = torch.cat((images, doy_batch), dim=1)
    x = rearrange(x, "b c h w -> (b h w) 1 c")
    emb = model(transforms(x))
    emb = rearrange(emb, "(b h w) d -> b d h w", b=b, h=H, w=W)
    return emb.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tessera embeddings for EuroSAT")
    parser.add_argument("--root", type=str, default="data", help="Data root directory")
    parser.add_argument(
        "--output", type=str, default="data/eurosat-tessera", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=96, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=12, help="DataLoader workers")
    parser.add_argument("--write-workers", type=int, default=8, help="Parallel write threads")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, mps, cpu)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], help="Splits to process"
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA only)")
    args = parser.parse_args()

    device = torch.device(args.device)

    doy = torch.full((1, 1, H, W), 365 // 2, dtype=torch.float32, device=device)

    weights = Tessera_Weights.TESSERA_SENTINEL2_ENCODER
    model = tessera(weights=weights)
    model.eval()
    model.to(device)
    if args.compile:
        model = torch.compile(model)
    transforms = weights.transforms
    transforms.to(device)

    for split in args.splits:
        print(f"Processing {split} split...")
        dataset = EuroSAT(root=args.root, split=split, bands=TESSERA_BANDS)

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
                output_path = os.path.join(args.output, folder, filename)
                batch_paths.append((filepath, output_path))

            all_outputs_exist = all(os.path.exists(output_path) for _, output_path in batch_paths)
            if all_outputs_exist:
                sample_idx += batch_size
                continue

            images = batch["image"].to(device, non_blocking=True)
            embeddings = embed_batch(model, transforms, images, doy)  # type: ignore[arg-type]

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
