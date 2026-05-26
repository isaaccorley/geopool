"""Upload PASTIS embedding tar.gz files to isaaccorley/pastis-embed on HuggingFace.

Run inside the SLURM job after all tar.gz files have been created.
"""
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "isaaccorley/pastis-embed"
REPO_TYPE = "dataset"

FILES = [
    "pastis-aef.tar.gz",
    "pastis-olmoearth.tar.gz",
    "pastis-tessera.tar.gz",
    "embeddings-aef-pooled.tar.gz",
    "embeddings-olmoearth-pooled.tar.gz",
    "embeddings-tessera-pooled.tar.gz",
]


def upload_with_retry(api, **kwargs):
    """Upload with exponential backoff on 429 rate-limit errors."""
    delay = 60
    for attempt in range(6):
        try:
            api.upload_file(**kwargs)
            return
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"  Rate-limited; waiting {delay}s (attempt {attempt+1}/6) ...")
                time.sleep(delay)
                delay = min(delay * 2, 600)
            else:
                raise


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # Repo already created — skip create_repo to avoid rate limits.
    # If it doesn't exist for some reason, uncomment:
    # api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
    print(f"Targeting repo: https://huggingface.co/datasets/{REPO_ID}")

    # Upload README
    readme_path = Path("data/pastis_embed_README.md")
    if readme_path.exists():
        print("Uploading README.md ...")
        upload_with_retry(api,
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print("  README uploaded.")

    # Upload each tar.gz
    staging = Path("data/hf_staging")
    for fname in FILES:
        fpath = staging / fname
        if not fpath.exists():
            print(f"MISSING: {fpath} — skipping", file=sys.stderr)
            continue
        size_gb = fpath.stat().st_size / 1e9
        print(f"Uploading {fname} ({size_gb:.2f} GB) ...")
        t0 = time.time()
        upload_with_retry(api,
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} min.")

    print("\nAll uploads complete.")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
