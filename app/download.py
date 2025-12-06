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

def download_model(model_name: str, output_dir: str):
    """
    Mock function to download model.
    TODO: Implement actual huggingface download logic.
    """
    print(f"[MOCK] Downloading model '{model_name}' to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write("{}")
    print(f"[MOCK] Model downloaded.")

def download_dataset(dataset_name: str, output_dir: str):
    """
    Mock function to download dataset.
    TODO: Implement actual dataset download logic.
    """
    print(f"[MOCK] Downloading dataset '{dataset_name}' to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[MOCK] Dataset downloaded.")

if __name__ == "__main__":
    # In a real scenario, these would come from config
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    DATASET_NAME = "ShengbinYue/DISC-Law-SFT"
    download_model(MODEL_NAME, "app/models/base")
    download_dataset(DATASET_NAME, "app/data/train")
