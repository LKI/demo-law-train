"""
app/download.py

Purpose:
    Download the base model and dataset from HuggingFace.
    This script should be idempotent (checks if already downloaded).

Inputs:
    - Environment variables or Config for model name.
Outputs:
    - Saved model/dataset in local dirs.
"""

import os
from huggingface_hub import snapshot_download

# In a real production environment, this should be stored securely (e.g. env var, secrets manager)
# User provided token for this setup.
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print(
        "[WARN] HF_TOKEN env var not set. Attempting public download (anonymous access)."
    )


def download_model(model_name: str, output_dir: str):
    """
    Download model using huggingface_hub.
    """
    print(f"Downloading model '{model_name}' to '{output_dir}'...")
    snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN,
    )
    print(f"Model downloaded to {output_dir}")


def download_dataset(dataset_name: str, output_dir: str):
    """
    Download dataset using huggingface_hub.
    """
    print(f"Downloading dataset '{dataset_name}' to '{output_dir}'...")
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN,
    )
    print(f"Dataset downloaded to {output_dir}")


# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "ShengbinYue/DISC-Law-SFT"

if __name__ == "__main__":
    # In a real scenario, these would come from config

    # Ensure directories exist
    os.makedirs("app/models/base", exist_ok=True)
    os.makedirs("app/data/train", exist_ok=True)

    download_model(MODEL_NAME, "app/models/base")
    download_dataset(DATASET_NAME, "app/data/train")
